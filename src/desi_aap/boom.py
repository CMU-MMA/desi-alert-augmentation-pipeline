"""Query the BOOM (kaboom) alert broker and return alerts as a pandas DataFrame.

This module ports the data-grabbing logic from ``old_code/LSST_For_COSMOS.ipynb``
into a reusable, package-ready interface.

The high level flow is:

1. Mint a short-lived access token from a username/password via ``POST /auth``.
   The token is generated fresh on every call and used immediately -- it is
   never written to disk.
2. Run a server-side filter pipeline over a Julian-date range via
   ``POST /filters/test``.
3. Normalize the JSON results into a :class:`pandas.DataFrame`.

Credentials are read from the ``BOOM_USERNAME`` and ``BOOM_PASSWORD``
environment variables by default, or may be passed explicitly. This keeps the
secrets out of source control (see the GitHub Actions workflow, which injects
them from repository secrets).
"""

import os
from datetime import datetime, timedelta
from typing import Any, Union

import pandas as pd
import requests
from astropy.time import Time

KABOOM_BASE_URL = "https://api.kaboom.caltech.edu"

# Anything that can be interpreted as a point in time: an astropy ``Time``, a
# python ``datetime``, an ISO-8601 string, or a raw Julian date as a number.
TimeLike = Union[Time, datetime, str, float, int]

# Default server-side filter pipeline ported verbatim from
# ``old_code/LSST_For_COSMOS.ipynb``. Callers may pass their own ``pipeline`` to
# :func:`query_alerts` to override this.
DEFAULT_PIPELINE: list[dict[str, Any]] = [
    {
        "$project": {
            "candidate.ra": 1,
            "candidate.dec": 1,
            "candidate.jd": 1,
            "candidate.isDipole": 1,
            "candidate.isdiffpos": 1,
            "candidate.magpsf": 1,
            "candidate.ndethist": 1,
            "candidate.reliability": 1,
            "cross_matches.LSPSC": 1,
            "distance_arcsec": 1,
            "mag_white": 1,
            "properties.near_brightstar": 1,
            "properties.rock": 1,
            "properties.star": 1,
            "properties.stationary": 1,
            "score": 1,
            "objectId": 1,
        }
    },
    {
        "$match": {
            "$and": [
                {
                    "$and": [
                        {"candidate.magpsf": {"$lte": 22}},
                        {"candidate.isdiffpos": {"$in": [True]}},
                        {"candidate.reliability": {"$gt": 0.7}},
                        {"candidate.isDipole": {"$in": [False]}},
                        {"candidate.ndethist": {"$gte": 2}},
                        {"candidate.dec": {"$gt": -30}},
                    ]
                },
                {
                    "$and": [
                        {
                            "$and": [
                                {"properties.rock": {"$in": [False]}},
                                {"properties.star": {"$in": [False]}},
                                {"properties.near_brightstar": {"$in": [False]}},
                                {"properties.stationary": {"$in": [True]}},
                            ]
                        },
                        {
                            "$expr": {
                                "$not": {
                                    "$anyElementTrue": {
                                        "$map": {
                                            "input": {"$ifNull": ["$cross_matches.LSPSC", []]},
                                            "in": {
                                                "$and": [
                                                    {"$gte": ["$$this.score", 0.7]},
                                                    {
                                                        "$or": [
                                                            {
                                                                "$and": [
                                                                    {"$lte": ["$$this.distance_arcsec", 4]},
                                                                    {"$lte": ["$$this.mag_white", 18]},
                                                                ]
                                                            },
                                                            {
                                                                "$and": [
                                                                    {"$lte": ["$$this.distance_arcsec", 10]},
                                                                    {"$lte": ["$$this.mag_white", 16]},
                                                                ]
                                                            },
                                                            {
                                                                "$and": [
                                                                    {"$lte": ["$$this.distance_arcsec", 30]},
                                                                    {"$lte": ["$$this.mag_white", 15]},
                                                                ]
                                                            },
                                                            {
                                                                "$and": [
                                                                    {"$lte": ["$$this.distance_arcsec", 6]},
                                                                    {"$lte": ["$$this.mag_white", 17.5]},
                                                                ]
                                                            },
                                                        ]
                                                    },
                                                ]
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    ]
                },
            ]
        }
    },
    {
        "$project": {
            "objectId": 1,
            "candidate.ra": 1,
            "candidate.dec": 1,
            "candidate.jd": 1,
            "candidate.isDipole": 1,
            "candidate.isdiffpos": 1,
            "candidate.magpsf": 1,
            "candidate.ndethist": 1,
            "candidate.reliability": 1,
            "cross_matches.LSPSC": 1,
            "distance_arcsec": 1,
            "mag_white": 1,
            "properties.near_brightstar": 1,
            "properties.rock": 1,
            "properties.star": 1,
            "properties.stationary": 1,
            "score": 1,
        }
    },
]


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


def _resolve_window(
    start: TimeLike | None,
    end: TimeLike | None,
    default_window: timedelta,
) -> tuple[float, float]:
    """Resolve a (start, end) pair into Julian dates.

    Defaults to the trailing ``default_window`` ending *now* when both bounds
    are omitted. If only one bound is given, the other is derived from it using
    ``default_window``.

    Parameters
    ----------
    start : time-like or None
        Start of the window. Defaults to ``end - default_window``.
    end : time-like or None
        End of the window. Defaults to now.
    default_window : timedelta
        The width used to fill in a missing bound.

    Returns
    -------
    tuple of float
        ``(start_jd, end_jd)``.
    """
    window_days = default_window.total_seconds() / 86400.0
    end_jd = _to_jd(end) if end is not None else float(Time.now().jd)
    start_jd = _to_jd(start) if start is not None else end_jd - window_days
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


def query_alerts(
    start: TimeLike | None = None,
    end: TimeLike | None = None,
    *,
    username: str | None = None,
    password: str | None = None,
    survey: str = "LSST",
    pipeline: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    sort_by: str | None = "candidate.jd",
    sort_order: str = "Descending",
    permissions: dict[str, Any] | None = None,
    client_token: str | None = None,
    base_url: str = KABOOM_BASE_URL,
    timeout_s: int = 300,
) -> pd.DataFrame:
    """Grab BOOM alerts between two dates and return them as a DataFrame.

    By default this queries the **last hour** of real time. Provide ``start``
    and/or ``end`` (as astropy ``Time``, ``datetime``, ISO-8601 string, or raw
    Julian date) to query an arbitrary window.

    Credentials default to the ``BOOM_USERNAME`` and ``BOOM_PASSWORD``
    environment variables. A fresh access token is minted for each call and
    used immediately; nothing is written to disk.

    Parameters
    ----------
    start : time-like, optional
        Start of the window. Defaults to one hour before ``end``.
    end : time-like, optional
        End of the window. Defaults to now.
    username, password : str, optional
        BOOM credentials. Fall back to the ``BOOM_USERNAME`` / ``BOOM_PASSWORD``
        environment variables.
    survey : str
        Survey to query (e.g. ``"LSST"``).
    pipeline : list of dict, optional
        Server-side aggregation pipeline. Defaults to :data:`DEFAULT_PIPELINE`.
    limit : int, optional
        Maximum number of records to return.
    sort_by : str, optional
        Field to sort by.
    sort_order : str
        ``"Ascending"`` or ``"Descending"``.
    permissions : dict, optional
        Permissions object for the request. Defaults to ``{}``.
    client_token : str, optional
        Optional Bearer token for the auth request.
    base_url : str
        Base URL of the BOOM API.
    timeout_s : int
        Request timeout in seconds for the data query.

    Returns
    -------
    pandas.DataFrame
        The normalized alert records. Empty (no columns) when no alerts match.
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

    start_jd, end_jd = _resolve_window(start, end, timedelta(hours=1))

    token = get_access_token(username, password, client_token=client_token, base_url=base_url)
    response = _run_filter_pipeline(
        token=token,
        pipeline=pipeline if pipeline is not None else DEFAULT_PIPELINE,
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
    return pd.json_normalize(_extract_results(response))


def _main(argv: list[str] | None = None) -> int:
    """Command-line entry point: query alerts and optionally write a CSV.

    Examples
    --------
    Fetch the last hour and print a summary::

        python -m desi_aap.boom

    Write a fixed window to a CSV (used to build the test gold standard)::

        python -m desi_aap.boom --start-jd 2461187.1383197717 \
            --end-jd 2461194.1383197717 --out gold.csv
    """
    import argparse

    parser = argparse.ArgumentParser(description="Query BOOM alerts into a DataFrame/CSV.")
    parser.add_argument("--start", help="Start of window (ISO-8601). Overridden by --start-jd.")
    parser.add_argument("--end", help="End of window (ISO-8601). Overridden by --end-jd.")
    parser.add_argument("--start-jd", type=float, help="Start of window as a Julian date.")
    parser.add_argument("--end-jd", type=float, help="End of window as a Julian date.")
    parser.add_argument("--survey", default="LSST")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", help="Path to write the results as CSV.")
    args = parser.parse_args(argv)

    start = args.start_jd if args.start_jd is not None else args.start
    end = args.end_jd if args.end_jd is not None else args.end

    df = query_alerts(start=start, end=end, survey=args.survey, limit=args.limit)
    print(f"Retrieved {len(df)} alerts.")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
