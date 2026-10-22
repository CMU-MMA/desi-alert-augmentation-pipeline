"""Query the BOOM (kaboom) alert broker and return alerts as a NestedFrame.

This module ports the data-grabbing logic from ``old_code/LSST_For_COSMOS.ipynb``
into a reusable, package-ready interface.

The high level flow is:

1. Mint a short-lived access token from a username/password via ``POST /auth``.
   The token is generated fresh on every call and used immediately -- it is
   never written to disk.
2. Run a server-side filter pipeline over a Julian-date range via
   ``POST /filters/test``.
3. Normalize the JSON results into a :class:`nested_pandas.NestedFrame`, packing
   list-valued fields (notably the ``cross_matches.LSPSC`` list of structs) into
   nested columns so they survive a parquet round trip.

Credentials are read from the ``BOOM_USERNAME`` and ``BOOM_PASSWORD``
environment variables by default, or may be passed explicitly. This keeps the
secrets out of source control (see the GitHub Actions workflow, which injects
them from repository secrets).
"""

import json
import os
import re
import warnings
from collections.abc import Mapping
from datetime import datetime, timedelta
from importlib.resources import files
from typing import Any, Union

import nested_pandas as npd
import pandas as pd
import requests
from astropy.time import Time

# ``pack_seq`` turns a sequence of per-row records (lists of dicts, as returned
# by the BOOM API) into a single nested column.
from nested_pandas.series.packer import pack_seq

KABOOM_BASE_URL = "https://api.kaboom.caltech.edu"

# Anything that can be interpreted as a point in time: an astropy ``Time``, a
# python ``datetime``, an ISO-8601 string, or a raw Julian date as a number.
TimeLike = Union[Time, datetime, str, float, int]

# Width of the query window when only one bound (or neither) is given.
DEFAULT_WINDOW = timedelta(hours=1)

# Columns of the alert frame BOOM returns, in the names it gives them. The
# projection in default_pipeline.json decides which columns come back at all,
# so a rename has to move both.
ALERT_ID_COLUMN = "objectId"
ALERT_RA_COLUMN = "candidate.ra"
ALERT_DEC_COLUMN = "candidate.dec"
ALERT_TIME_COLUMN = "candidate.jd"
# Apparent PSF magnitude of the difference-image detection, and the band it was
# measured in. A magnitude without its band is not a brightness anyone can act
# on, so the two are projected and carried together.
ALERT_MAG_COLUMN = "candidate.magpsf"
ALERT_BAND_COLUMN = "candidate.band"

# Every value ALERT_BAND_COLUMN can take: a lowercase single-letter string
# rather than an integer filter id, because BOOM normalizes every survey it
# ingests onto one set of names. Taken from the ``band`` field of the
# ``LsstAlertToFilter`` record that GET /filters/schemas/LSST returns, where it
# is an Avro enum unioned with null -- so a record may carry no band at all, and
# an alert with none is one no absolute magnitude can be computed for.
#
# The first six are the LSST bands; j, h and k are in the enum because the same
# type describes every survey BOOM ingests, and no LSST alert carries them.
ALERT_BANDS = ("u", "g", "r", "i", "z", "y", "j", "h", "k")

# Nested column names to use for known list-valued BOOM fields. Nested-pandas
# uses ``"<nested>.<field>"`` to address sub-columns, so a nested column may not
# itself contain a dot; anything not listed here gets its dots replaced with
# underscores (see :func:`nested_column_name`).
NESTED_COLUMN_NAMES = {"cross_matches.LSPSC": "lspsc"}

# Path to the default server-side filter pipeline (ported from
# old_code/LSST_For_COSMOS.ipynb). Editing this JSON file is the easiest way to
# customize the default filter; callers can also pass their own ``pipeline`` to
# :func:`query_alerts`.
DEFAULT_PIPELINE_PATH = files("desi_aap").joinpath("default_pipeline.json")

_DURATION_UNITS = {"w": 604800, "d": 86400, "h": 3600, "m": 60, "s": 1}
_DURATION_PART = re.compile(rf"(\d+(?:\.\d+)?)([{''.join(_DURATION_UNITS)}])")


def load_default_pipeline() -> list[dict[str, Any]]:
    """Load the default filter pipeline from ``default_pipeline.json``.

    Returns
    -------
    list of dict
        A freshly parsed copy of the default aggregation pipeline.
    """
    return json.loads(DEFAULT_PIPELINE_PATH.read_text())


def _to_jd(value: TimeLike) -> float:
    """Convert a time-like value into a Julian date (float).

    Numbers are assumed to already be Julian dates. Strings are parsed as
    ISO-8601. ``datetime`` and astropy ``Time`` objects are converted directly.

    Parameters
    ----------
    value : Time, datetime, str, or float
        The point in time to convert.

    Returns
    -------
    float
        The corresponding Julian date.
    """
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, Time):
        return float(value.jd)
    if isinstance(value, datetime):
        return float(Time(value).jd)
    if isinstance(value, str):
        return float(Time(value, format="isot" if "T" in value else None).jd)
    raise TypeError(f"Cannot interpret {value!r} ({type(value).__name__}) as a time.")


def parse_timedelta(text: str) -> timedelta:
    """Parse a compact duration string such as ``"90m"`` or ``"1d12h"``.

    Supported units are ``w`` (weeks), ``d`` (days), ``h`` (hours), ``m``
    (minutes) and ``s`` (seconds). Values may be fractional and several parts
    may be concatenated; a unit is always required.

    Parameters
    ----------
    text : str
        The duration to parse, e.g. ``"7d"``, ``"1.5h"``, ``"1d12h30m"``.

    Returns
    -------
    datetime.timedelta
        The parsed duration.

    Examples
    --------
    >>> parse_timedelta("2h")
    datetime.timedelta(seconds=7200)
    >>> parse_timedelta("1d12h")
    datetime.timedelta(days=1, seconds=43200)
    """
    cleaned = text.strip().lower().replace(" ", "")
    # Every character must be consumed by a <number><unit> part, so "7", "7x"
    # and "" are all rejected.
    if not cleaned or _DURATION_PART.sub("", cleaned):
        raise ValueError(
            f"Cannot parse {text!r} as a duration. Use a value and a unit, e.g. '30m', '2h', '7d' or '1d12h'."
        )
    seconds = sum(float(value) * _DURATION_UNITS[unit] for value, unit in _DURATION_PART.findall(cleaned))
    return timedelta(seconds=seconds)


def _resolve_window(
    start: TimeLike | None,
    end: TimeLike | None,
    window: timedelta,
) -> tuple[float, float]:
    """Resolve a (start, end) pair into Julian dates.

    ``window`` fills in whichever bound is missing:

    * neither bound: the trailing ``window`` ending *now*;
    * ``end`` only: ``end - window`` to ``end``;
    * ``start`` only: ``start`` to ``start + window``;
    * both bounds: ``window`` is unused.

    Parameters
    ----------
    start : time-like or None
        Start of the window. Defaults to ``end - window``.
    end : time-like or None
        End of the window. Defaults to ``start + window``, or now if ``start``
        is also omitted.
    window : timedelta
        The width used to fill in a missing bound. Must be positive.

    Returns
    -------
    tuple of float
        ``(start_jd, end_jd)``.
    """
    if not isinstance(window, timedelta):
        raise TypeError(f"window must be a datetime.timedelta, got {type(window).__name__}.")
    if window <= timedelta(0):
        raise ValueError(f"window must be positive, got {window!r}.")

    window_days = window.total_seconds() / 86400.0
    if start is None and end is None:
        end_jd = float(Time.now().jd)
        start_jd = end_jd - window_days
    elif start is None:
        end_jd = _to_jd(end)
        start_jd = end_jd - window_days
    elif end is None:
        start_jd = _to_jd(start)
        end_jd = start_jd + window_days
    else:
        start_jd, end_jd = _to_jd(start), _to_jd(end)

    if start_jd > end_jd:
        raise ValueError(f"start ({start_jd}) must not be after end ({end_jd}).")
    return start_jd, end_jd


def get_access_token(
    username: str,
    password: str,
    *,
    client_token: str | None = None,
    base_url: str = KABOOM_BASE_URL,
    timeout_s: int = 60,
) -> str:
    """Mint a fresh BOOM access token from a username and password.

    The token is returned to the caller and is never persisted by this
    function. A new token should be requested for each session.

    Parameters
    ----------
    username : str
        BOOM account username.
    password : str
        BOOM account password.
    client_token : str, optional
        Optional Bearer token to send in the ``Authorization`` header of the
        auth request. Omitted by default.
    base_url : str
        Base URL of the BOOM API.
    timeout_s : int
        Request timeout in seconds.

    Returns
    -------
    str
        A short-lived access token for use with the data endpoints.
    """
    url = f"{base_url.rstrip('/')}/auth"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if client_token:
        headers["Authorization"] = f"Bearer {client_token}"

    resp = requests.post(
        url,
        headers=headers,
        data={"username": username, "password": password},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    token_info = resp.json()
    try:
        return token_info["access_token"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Auth response did not contain an access_token: {token_info!r}") from exc


def _run_filter_pipeline(
    *,
    token: str,
    pipeline: list[dict[str, Any]],
    start_jd: float,
    end_jd: float,
    survey: str,
    limit: int | None,
    sort_by: str | None,
    sort_order: str,
    permissions: dict[str, Any] | None,
    base_url: str,
    timeout_s: int,
) -> dict[str, Any]:
    """POST a filter pipeline to ``/filters/test`` and return the JSON response."""
    url = f"{base_url.rstrip('/')}/filters/test"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload: dict[str, Any] = {
        "pipeline": pipeline,
        "survey": survey,
        "permissions": permissions if permissions is not None else {},
        "start_jd": start_jd,
        "end_jd": end_jd,
        "limit": limit,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"POST /filters/test failed: HTTP {resp.status_code}\nResponse: {detail}")
    return resp.json()


def _extract_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the list of result records out of a ``/filters/test`` response.

    The API has been observed to nest results under ``data.results`` as well as
    a top-level ``results`` key, so both shapes are handled.
    """
    data = response.get("data")
    if isinstance(data, dict) and "results" in data:
        return data["results"] or []
    if "results" in response:
        return response["results"] or []
    if isinstance(data, list):
        return data
    return []


def nested_column_name(column: str, nested_names: Mapping[str, str] | None = None) -> str:
    """Pick the nested column name to use for a flattened list-valued field.

    Nested-pandas addresses sub-columns as ``"<nested>.<field>"``, so the nested
    column itself must not contain a dot. Known BOOM fields get a short name
    from ``nested_names``; anything else keeps its full path with dots replaced
    by underscores.

    Parameters
    ----------
    column : str
        The flattened (dotted) column name, e.g. ``"cross_matches.LSPSC"``.
    nested_names : mapping of str to str, optional
        Overrides for the default :data:`NESTED_COLUMN_NAMES` mapping.

    Returns
    -------
    str
        The nested column name.

    Examples
    --------
    >>> nested_column_name("cross_matches.LSPSC")
    'lspsc'
    >>> nested_column_name("cross_matches.OTHER")
    'cross_matches_OTHER'
    """
    names = NESTED_COLUMN_NAMES if nested_names is None else nested_names
    return names.get(column, column.replace(".", "_"))


def to_nested_frame(
    records: list[dict[str, Any]],
    *,
    nested_names: Mapping[str, str] | None = None,
) -> npd.NestedFrame:
    """Normalize raw BOOM records into a :class:`nested_pandas.NestedFrame`.

    Scalar sub-documents (``candidate``, ``properties``, ...) are flattened into
    dotted columns exactly as :func:`pandas.json_normalize` does. Any
    list-valued field -- ``cross_matches.LSPSC`` is a list of structs -- is
    packed into a nested column instead of being left as a column of Python
    lists, which keeps it queryable (``nf["lspsc.score"]``) and lets the frame
    be written straight to parquet.

    Parameters
    ----------
    records : list of dict
        Raw records as returned by ``POST /filters/test``.
    nested_names : mapping of str to str, optional
        Overrides for the default :data:`NESTED_COLUMN_NAMES` mapping of
        flattened column name to nested column name.

    Returns
    -------
    nested_pandas.NestedFrame
        The normalized alerts. Empty (no columns) when ``records`` is empty.
    """
    flat = pd.json_normalize(records)
    list_columns = [c for c in flat.columns if flat[c].map(lambda v: isinstance(v, list)).any()]
    nested = npd.NestedFrame(flat.drop(columns=list_columns))

    for column in list_columns:
        name = nested_column_name(column, nested_names)
        try:
            nested[name] = pack_seq(flat[column], name=name)
        except (ValueError, TypeError) as exc:
            # Packing needs at least one non-empty record to infer the struct
            # schema from, so an all-empty column stays a plain list column.
            warnings.warn(f"Could not pack {column!r} into a nested column ({exc}); keeping lists.")
            nested[name] = flat[column]
    return nested


def query_alerts(
    start: TimeLike | None = None,
    end: TimeLike | None = None,
    *,
    window: timedelta = DEFAULT_WINDOW,
    username: str | None = None,
    password: str | None = None,
    survey: str = "LSST",
    pipeline: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    sort_by: str | None = ALERT_TIME_COLUMN,
    sort_order: str = "Descending",
    permissions: dict[str, Any] | None = None,
    nested_names: Mapping[str, str] | None = None,
    client_token: str | None = None,
    base_url: str = KABOOM_BASE_URL,
    timeout_s: int = 300,
) -> npd.NestedFrame:
    """Grab BOOM alerts between two dates and return them as a NestedFrame.

    By default this queries the **last hour** of real time. Provide ``start``
    and/or ``end`` (as astropy ``Time``, ``datetime``, ISO-8601 string, or raw
    Julian date) to query an arbitrary window, and/or ``window`` as a
    :class:`datetime.timedelta` to set how wide the window is::

        query_alerts(window=timedelta(days=7))                  # last 7 days
        query_alerts(end="2026-06-02T00:00:00", window=timedelta(hours=6))
        query_alerts(start=2461187.0, window=timedelta(days=1))

    Credentials default to the ``BOOM_USERNAME`` and ``BOOM_PASSWORD``
    environment variables. A fresh access token is minted for each call and
    used immediately; nothing is written to disk.

    Parameters
    ----------
    start : time-like, optional
        Start of the window. Defaults to ``end - window``.
    end : time-like, optional
        End of the window. Defaults to ``start + window``, or now if ``start``
        is also omitted.
    window : timedelta
        Width of the window, used to fill in whichever bound is missing.
        Defaults to :data:`DEFAULT_WINDOW` (one hour) and is unused when both
        ``start`` and ``end`` are given.
    username, password : str, optional
        BOOM credentials. Fall back to the ``BOOM_USERNAME`` / ``BOOM_PASSWORD``
        environment variables.
    survey : str
        Survey to query (e.g. ``"LSST"``).
    pipeline : list of dict, optional
        Server-side aggregation pipeline. Defaults to the pipeline loaded from
        ``default_pipeline.json`` (see :func:`load_default_pipeline`).
    limit : int, optional
        Maximum number of records to return.
    sort_by : str, optional
        Field to sort by.
    sort_order : str
        ``"Ascending"`` or ``"Descending"``.
    permissions : dict, optional
        Permissions object for the request. Defaults to ``{}``.
    nested_names : mapping of str to str, optional
        Overrides for the default mapping of flattened list-valued column to
        nested column name (see :func:`to_nested_frame`).
    client_token : str, optional
        Optional Bearer token for the auth request.
    base_url : str
        Base URL of the BOOM API.
    timeout_s : int
        Request timeout in seconds for the data query.

    Returns
    -------
    nested_pandas.NestedFrame
        The normalized alert records, with ``cross_matches.LSPSC`` packed into a
        nested ``lspsc`` column. Empty (no columns) when no alerts match.
    """
    username = username if username is not None else os.environ.get("BOOM_USERNAME")
    password = password if password is not None else os.environ.get("BOOM_PASSWORD")
    if not username or not password:
        raise ValueError(
            "BOOM credentials are required. Pass username/password or set the "
            "BOOM_USERNAME and BOOM_PASSWORD environment variables."
        )
    if client_token is None:
        client_token = os.environ.get("BOOM_CLIENT_TOKEN")

    start_jd, end_jd = _resolve_window(start, end, window)

    token = get_access_token(username, password, client_token=client_token, base_url=base_url)
    response = _run_filter_pipeline(
        token=token,
        pipeline=pipeline if pipeline is not None else load_default_pipeline(),
        start_jd=start_jd,
        end_jd=end_jd,
        survey=survey,
        limit=limit,
        sort_by=sort_by,
        sort_order=sort_order,
        permissions=permissions,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    return to_nested_frame(_extract_results(response), nested_names=nested_names)


def _main(argv: list[str] | None = None) -> int:
    """Command-line entry point: query alerts and optionally write a parquet file.

    Examples
    --------
    Fetch the last hour and print a summary::

        python -m desi_aap.boom

    Fetch the last week::

        python -m desi_aap.boom --window 7d

    Write a fixed window to parquet (used to build the test gold standard)::

        python -m desi_aap.boom --start-jd 2461187.1383197717 \
            --end-jd 2461194.1383197717 --out gold.parquet
    """
    import argparse

    parser = argparse.ArgumentParser(description="Query BOOM alerts into a NestedFrame/parquet.")
    parser.add_argument("--start", help="Start of window (ISO-8601). Overridden by --start-jd.")
    parser.add_argument("--end", help="End of window (ISO-8601). Overridden by --end-jd.")
    parser.add_argument("--start-jd", type=float, help="Start of window as a Julian date.")
    parser.add_argument("--end-jd", type=float, help="End of window as a Julian date.")
    parser.add_argument(
        "--window",
        type=parse_timedelta,
        default=DEFAULT_WINDOW,
        help="Width of the window, used for whichever bound is not given, "
        "e.g. '30m', '2h', '7d', '1d12h'. Defaults to 1h.",
    )
    parser.add_argument("--survey", default="LSST")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pipeline",
        help="Path to a JSON file with a custom filter pipeline. "
        "Defaults to the packaged default_pipeline.json.",
    )
    parser.add_argument("--out", help="Path to write the results as parquet.")
    args = parser.parse_args(argv)

    start = args.start_jd if args.start_jd is not None else args.start
    end = args.end_jd if args.end_jd is not None else args.end
    from pathlib import Path

    pipeline = json.loads(Path(args.pipeline).read_text()) if args.pipeline else None

    alerts = query_alerts(
        start=start,
        end=end,
        window=args.window,
        survey=args.survey,
        limit=args.limit,
        pipeline=pipeline,
    )
    print(f"Retrieved {len(alerts)} alerts.")
    if args.out:
        alerts.to_parquet(args.out)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
