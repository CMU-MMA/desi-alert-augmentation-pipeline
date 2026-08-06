"""Fetch GraceDB BNS/NSBH superevents and crossmatch them against SN catalogs.

Provides the temporal crossmatch (discovery time vs. GW time) and the 3D
spatial crossmatch (RA/Dec/distance vs. the GW skymap's credible volume).
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from ligo.gracedb.rest import GraceDb
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import crossmatch

from desi_aap.cosmology import COSMOLOGIES

# Time-unit conversions, used for the false-alarm rates GraceDB reports in Hz and for
# the SN-to-GW offsets. astropy's yr is the Julian year, 365.25 days.
SECONDS_PER_DAY = (1 * u.day).to_value(u.s)
JULIAN_YEAR_SECONDS = (1 * u.yr).to_value(u.s)

# crossmatch(..., cosmology=False) ranks the 3D posterior by probability
# density per luminosity-distance volume, matching the units in the skymaps.
# The two cosmology runs differ by the redshift-to-luminosity-distance
# conversion used for each SN.
USE_COMOVING_VOLUME_RANKING = True

# Local output directory for downloaded skymaps.
SKYMAP_DIR = Path("gracedb_skymaps")

# Rank given to names that are not skymaps at all. The rest of the skymap file-selection
# priorities are local to skymap_priority; this one is shared because choose_skymap_file
# filters on it to decide which names are usable.
SKYMAP_PRIORITY_IGNORE = 1000


def as_float(value, default=np.nan):
    """Coerce a value to float, returning a default on failure or None.

    Normalizes the loosely typed values in GraceDB JSON payloads, which may be None,
    numeric strings, or already numeric, into plain floats.

    Parameters
    ----------
    value : object
        Value to coerce. None, and anything float() rejects, falls back to default.
    default : float, optional
        Value returned when coercion fails. Defaults to np.nan.

    Returns
    -------
    float
        The value as a float, or default.
    """
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def response_to_bytes(response):
    """Extract raw bytes from a GraceDB client file response.

    Handles both the file-like objects and the requests-style response objects that
    ligo.gracedb may return, encoding text payloads as UTF-8.

    Parameters
    ----------
    response : object
        Object returned by GraceDb.files(). Read from its read() method when it has one,
        otherwise from its content attribute.

    Returns
    -------
    bytes
        The response body.
    """
    data = response.read() if hasattr(response, "read") else response.content
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data


def unversioned_file_names(files):
    """Return file names from a GraceDB file listing that aren't versioned copies.

    GraceDB exposes every revision of a file under a ",N" suffix (e.g.
    "bayestar.multiorder.fits,0"), alongside the unsuffixed name that points at the latest
    revision. Only the unsuffixed names are kept.

    Parameters
    ----------
    files : iterable of str
        File names, such as the keys of client.files(superevent_id).json().

    Returns
    -------
    list of str
        The names without a ",N" version suffix, in input order.
    """
    return [name for name in files if not re.search(r",\d+$", name)]


def choose_pastro_file(superevent, files):
    """Pick the best p_astro classification file name from a GraceDB file listing.

    Parameters
    ----------
    superevent : dict
        Superevent dict from GraceDb.superevents(). When its preferred_event_data.pipeline
        entry is set, a file named for that pipeline (e.g. "gstlal.p_astro.json") is
        preferred over the generic ones.
    files : iterable of str
        File names available for the superevent.

    Returns
    -------
    str or None
        The chosen file name, or None if the listing has no p_astro file.
    """
    names = unversioned_file_names(files)
    by_lower = {name.lower(): name for name in names}
    pipeline = str(superevent.get("preferred_event_data", {}).get("pipeline", "")).strip()
    candidates = []
    if pipeline:
        candidates.extend(
            [
                f"{pipeline}.p_astro.json",
                f"{pipeline.lower()}.p_astro.json",
                f"{pipeline.upper()}.p_astro.json",
            ]
        )
    candidates.extend(["p_astro.json", "pastro.json"])
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]

    p_astro_files = sorted(
        name
        for name in names
        if name.lower().endswith(".p_astro.json") or name.lower().endswith("p_astro.json")
    )
    return p_astro_files[0] if p_astro_files else None


def load_classification(client, superevent_id, pastro_file):
    """Download and parse a superevent's p_astro classification file.

    The payload is handed to json.loads as bytes, which decodes it itself. Network,
    decoding and JSON-parsing errors all propagate to the caller; fetch_gracedb_superevents
    catches them and records the failure in the row's status field.

    Parameters
    ----------
    client : ligo.gracedb.rest.GraceDb
        Connected GraceDB client.
    superevent_id : str
        Superevent identifier, for example "S190425z".
    pastro_file : str or None
        File name to fetch, typically from choose_pastro_file. A falsy value skips the
        download.

    Returns
    -------
    classification : dict
        The parsed JSON, keyed by class name, with "BNS", "NSBH", "BBH" and "Terrestrial"
        probabilities. Empty when pastro_file is falsy.
    classification_file : str or None
        The name that was fetched, or None when pastro_file is falsy.
    """
    if not pastro_file:
        return {}, None
    payload = response_to_bytes(client.files(superevent_id, pastro_file))
    data = json.loads(payload)
    return data, pastro_file


def skymap_priority(name):
    """Rank a skymap file name by preference; lower is preferred.

    Bilby is preferred over BAYESTAR, and multiorder FITS over flat FITS. The unversioned
    name, which GraceDB points at the newest revision, is preferred over the ",N" snapshots,
    which take a fixed penalty. Names that are not skymaps at all rank at exactly
    SKYMAP_PRIORITY_IGNORE, with no version penalty, since choose_skymap_file discards them
    all alike and never orders them against each other.

    Parameters
    ----------
    name : str
        File name from a GraceDB file listing.

    Returns
    -------
    int
        Priority from the ranking table below, plus the penalty if the name is versioned,
        or SKYMAP_PRIORITY_IGNORE. Values below SKYMAP_PRIORITY_IGNORE identify usable
        skymaps.

    Warns
    -----
    UserWarning
        If name is neither a bare file name nor a bare name with a single ",N" revision,
        for example "bayestar,extra.fits" or "bayestar.fits,0,1". Such a name is still
        ranked, on the text before its first comma, so it is normally ignored and
        choose_skymap_file falls through to the next candidate. The warning is not an
        error: the name came from GraceDB, so it signals that the parsing here needs
        updating rather than that the caller did anything wrong.
    """
    lower = name.lower()
    # A GraceDB name is expected to be a bare file name, optionally followed by a single
    # ",N" revision. Anything else means this module's model of the listing format is
    # incomplete, so it is worth surfacing rather than silently mis-ranking: the split
    # below keeps only the text before the first comma, which for such a name is not the
    # file name at all.
    if not re.fullmatch(r"[^,]*(,\d+)?", lower):
        warnings.warn(
            f"GraceDB file name {name!r} is neither a bare name nor a single ',N' revision; "
            "it is ranked on the text before its first comma and will most likely be ignored.",
            UserWarning,
            # Deliberately points here rather than at the caller: the message is for whoever
            # maintains this parsing, and choose_skymap_file ranks each name from two
            # separate lines, which at stacklevel=2 would warn twice for the same name.
            stacklevel=1,
        )
    unversioned = lower.split(",", 1)[0]
    # Reject any non-skymap files before doing any additional work.
    if not unversioned.endswith((".multiorder.fits", ".fits.gz", ".fits")):
        return SKYMAP_PRIORITY_IGNORE

    versioned_file_penalty = 100
    version_penalty = versioned_file_penalty if re.search(r",\d+$", lower) else 0
    # Ranking table, best first; the first rule whose test passes decides the priority.
    # The gaps between the ranks leave room to slot in a new producer without renumbering.
    priority_rules = (
        (lambda stem: "bilby.multiorder.fits" in stem, 0),
        (lambda stem: "bayestar.multiorder.fits" in stem, 10),
        (lambda stem: stem.endswith(".multiorder.fits"), 20),
        (lambda stem: "bayestar" in stem and stem.endswith(".fits.gz"), 30),
        (lambda stem: stem.endswith(".fits.gz"), 40),
        (lambda stem: stem.endswith(".fits"), 50),
    )
    for matches, priority in priority_rules:
        if matches(unversioned):
            return priority + version_penalty
    # Unreachable while the suffix guard above and the last three rules agree on which
    # suffixes count; kept so that loosening one without the other cannot return None.
    return SKYMAP_PRIORITY_IGNORE


def choose_skymap_file(files):
    """Pick the best available skymap file name from a GraceDB file listing.

    Parameters
    ----------
    files : iterable of str
        File names available for the superevent.

    Returns
    -------
    str or None
        The name with the lowest skymap_priority, ties broken alphabetically and
        case-insensitively, or None if the listing contains no usable skymap.
    """
    names = list(files)
    candidates = [name for name in names if skymap_priority(name) < SKYMAP_PRIORITY_IGNORE]
    if not candidates:
        return None
    return sorted(candidates, key=lambda name: (skymap_priority(name), name.lower()))[0]


def safe_file_part(value):
    """Sanitize a value for use as part of a local file name.

    Parameters
    ----------
    value : object
        Value to sanitize; converted with str() first.

    Returns
    -------
    str
        The string with each run of characters outside [A-Za-z0-9_.-] replaced by a single
        underscore.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def download_gracedb_file(client, superevent_id, filename, outdir=SKYMAP_DIR):
    """Download a GraceDB file to a local cache directory, skipping if already present.

    The local name is "<superevent_id>__<filename>" with both parts sanitized, so files
    from different superevents never collide. An existing file is trusted and not
    re-downloaded, which makes repeat runs of fetch_gracedb_superevents cheap.

    Parameters
    ----------
    client : ligo.gracedb.rest.GraceDb
        Connected GraceDB client.
    superevent_id : str
        Superevent identifier, e.g. "S190425z".
    filename : str
        Remote file name to fetch, e.g. "bilby.multiorder.fits".
    outdir : pathlib.Path, optional
        Directory to write into, created if it does not exist. Defaults to SKYMAP_DIR,
        which is relative to the working directory.

    Returns
    -------
    pathlib.Path
        Path to the local copy of the file.
    """
    outdir.mkdir(exist_ok=True)
    local_name = f"{safe_file_part(superevent_id)}__{safe_file_part(filename)}"
    path = outdir / local_name
    if not path.exists():
        payload = response_to_bytes(client.files(superevent_id, filename))
        path.write_bytes(payload)
    return path


def gps_to_utc(gps_time):
    """Convert a GPS time to a UTC timestamp, passing through NaN.

    The .utc conversion is required: an astropy Time built with format="gps" is
    on the TAI scale, so calling .to_datetime() on it directly yields a TAI datetime,
    which labeling as UTC leaves ahead by the accumulated leap seconds: 37 s for
    O3-era events, so S190425z reads 08:18:42 rather than its true 08:18:05.

    Parameters
    ----------
    gps_time : float
        GPS seconds, as reported by a superevent's t_0 or by its preferred event's
        gpstime. May be NaN.

    Returns
    -------
    pandas.Timestamp or pandas.NaT
        A timezone-aware Timestamp in UTC, or pd.NaT if gps_time is missing.
    """
    if pd.isna(gps_time):
        return pd.NaT
    return pd.Timestamp(Time(float(gps_time), format="gps").utc.to_datetime(), tz="UTC")


def fetch_gracedb_superevents(
    se_types,
    *,
    far_threshold_per_year=2.0,
    min_classification_prob_sum=0.9,
    max_results=None,
):
    """Query GraceDB for superevents passing the FAR and classification cuts.

    Per-superevent failures are not fatal. They are recorded in the row's status column and
    the scan continues.

    Parameters
    ----------
    se_types : list of str
        Superevent type strings to include, matched case-insensitively against the p_astro
        class names. Supported values: "bns", "nsbh", "bbh". Only superevents whose
        combined probability for the requested types exceeds min_classification_prob_sum
        are returned.
    far_threshold_per_year : float, optional
        False-alarm-rate cut in events per Julian year. Applied twice: once in the GraceDB
        query itself, converted to the Hz the API expects, and once locally, since a
        superevent whose own FAR is missing is judged on its preferred event's instead.
        Defaults to 2.0.
    min_classification_prob_sum : float, optional
        Minimum combined p_astro probability across se_types, exclusive. Defaults to 0.9.
        # TODO ask about mins for things like just BBH selection - confirmed that 0.9 is good
    max_results : int or None, optional
        Cap on the number of superevents the query returns, passed straight to
        GraceDb.superevents(). None, the default, means no cap.

    Returns
    -------
    pandas.DataFrame
        One row per passing superevent, sorted by gw_time, with columns:

        superevent_id
            GraceDB identifier, e.g. "S190425z".
        gw_time, gps_time
            Merger time as a UTC Timestamp and as raw GPS seconds.
        far_hz, far_per_year
            False-alarm rate in the GraceDB units and per Julian year.
        p_bns, p_nsbh, p_bbh, p_terrestrial
            p_astro probabilities, 0.0 when absent.
        classification_file
            Name of the p_astro file the probabilities came from.
        preferred_event, pipeline, search, instruments
            Metadata from the superevent's preferred event. A superevent groups the
            candidate events that several pipelines reported for the same signal, and
            GraceDB designates one of them preferred; it is the one that represents the
            superevent, and the source of its FAR and time when the superevent's own are
            missing.
        labels
            The superevent's GraceDB labels, comma-joined.
        skymap_file, skymap_path
            Chosen remote skymap name, and the local path it was downloaded to, or None if
            there was no skymap or the download failed.
        status
            "ok", or the reason the superevent was only partly processed: either
            "file_list_failed: ..." or "skymap_download_failed: ...". A superevent whose
            p_astro could not be read is dropped by the probability cut rather than
            returned, so that failure never appears here.  #TODO is this the desired behavior?

        Superevents whose file listing could not be read appear with only superevent_id,
        far_hz, far_per_year and status set; their other columns are NaN. Empty DataFrame
        if nothing passes the cuts.

    Examples
    --------
    A trimmed view of the result, one row per superevent::

        superevent_id  gw_time                           far_per_year  p_bns  p_nsbh  status
        S190425z       2019-04-25 08:18:05.011549+00:00      1.43e-05  0.999   0.000  ok
        S190814bv      2019-08-14 21:11:16.012957+00:00      6.41e-26  0.000   0.998  ok
    """
    service_url = "https://gracedb.ligo.org/api/"
    category = "Production"
    # Significant figures the FAR threshold is rendered with in the query string. Enough
    # that the rounding cannot move the cut across a real superevent's FAR.
    query_sigfigs = 12
    # Probability read for a class the p_astro payload does not carry. The explicit zero
    # matters: as_float defaults to NaN, which would propagate through the sum below and
    # make the cut drop the superevent outright rather than judge it on the classes it does
    # report.
    default_classification_probability = 0.0

    far_threshold_hz = far_threshold_per_year / JULIAN_YEAR_SECONDS
    query = f"category: {category} far < {far_threshold_hz:.{query_sigfigs}g}"

    classification_keys = [t.upper() for t in se_types]
    client = GraceDb(service_url=service_url)
    rows = []

    for superevent in client.superevents(query=query, max_results=max_results):
        sid = superevent.get("superevent_id")
        preferred = superevent.get("preferred_event_data", {}) or {}
        far_hz = as_float(superevent.get("far"), as_float(preferred.get("far")))
        far_per_year = far_hz * JULIAN_YEAR_SECONDS
        if not np.isfinite(far_per_year) or far_per_year >= far_threshold_per_year:
            continue

        try:
            files = client.files(sid).json()
        except Exception as exc:
            # gw_time is set explicitly because it is the sort key below. Without it, a
            # scan whose superevents all fail here builds a frame with no gw_time column
            # at all and the sort raises KeyError; a stub alongside a normal row only
            # works because pandas fills the column in for it.
            rows.append(
                {
                    "superevent_id": sid,
                    "gw_time": pd.NaT,
                    "status": f"file_list_failed: {exc}",
                    "far_hz": far_hz,
                    "far_per_year": far_per_year,
                }
            )
            continue

        pastro_file = choose_pastro_file(superevent, files)
        try:
            classification, classification_file = load_classification(client, sid, pastro_file)
        except Exception as exc:
            classification = {}
            classification_file = pastro_file
            status = f"p_astro_failed: {exc}"
        else:
            status = "ok"

        prob_sum = sum(
            as_float(classification.get(key), default_classification_probability)
            for key in classification_keys
        )
        if not (prob_sum > min_classification_prob_sum):  # TODO - is this how we want to approach
            continue

        skymap_file = choose_skymap_file(files)
        skymap_path = None
        if skymap_file:
            try:
                skymap_path = download_gracedb_file(client, sid, skymap_file)
            except Exception as exc:
                status = f"skymap_download_failed: {exc}"

        gps_time = as_float(superevent.get("t_0"), as_float(preferred.get("gpstime")))
        gw_time = gps_to_utc(gps_time)
        rows.append(
            {
                "superevent_id": sid,
                "gw_time": gw_time,
                "gps_time": gps_time,
                "far_hz": far_hz,
                "far_per_year": far_per_year,
                "p_bns": as_float(classification.get("BNS"), default_classification_probability),
                "p_nsbh": as_float(classification.get("NSBH"), default_classification_probability),
                "p_bbh": as_float(classification.get("BBH"), default_classification_probability),
                "p_terrestrial": as_float(
                    classification.get("Terrestrial"), default_classification_probability
                ),
                "classification_file": classification_file,
                "preferred_event": preferred.get("graceid"),
                "pipeline": preferred.get("pipeline"),
                "search": preferred.get("search"),
                "instruments": preferred.get("instruments"),
                "labels": ",".join(superevent.get("labels", [])),
                "skymap_file": skymap_file,
                "skymap_path": str(skymap_path) if skymap_path else None,
                "status": status,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("gw_time").reset_index(drop=True)


def temporal_crossmatch_sesn_to_gw(
    df_sesn, gw_events, *, window_days=14, verbose=False
):  # TODO may want to generalize beyond just sesn
    """Match SNe to GW events whose discovery date falls within the temporal window.

    The window is window_days either side of the event's gw_time. Note that an SN close in
    time to two events appears twice, once per event.

    Parameters
    ----------
    df_sesn : pandas.DataFrame
        Transient catalog, such as the cleaned TNS table from clean_tns_catalog. Must have
        a 'discoverydate' column comparable to a pandas Timestamp, that is, tz-aware UTC
        datetimes.
    gw_events : pandas.DataFrame
        Superevent table from fetch_gracedb_superevents. Events with a missing gw_time are
        skipped.
    window_days : float, optional
        Half-width of the matching window in days, applied either side of gw_time and
        inclusive at both ends. Defaults to 14.
    verbose : bool, optional
        If True, print each event as it is checked. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        One row per (SN, event) pair, sorted by gw_time then discoverydate; an SN inside the
        window of two events appears once for each. Empty if either input is empty or
        nothing falls inside the window. Columns are every df_sesn column, then:

        superevent_id, gw_time, gps_time
            Identify the matched event.
        days_from_gw
            Signed offset in days from the GW, negative when the SN was discovered first.
        gw_*
            The event's own fields copied across under a gw_ prefix: gw_far_per_year,
            gw_p_bns, gw_p_nsbh, gw_p_bbh, gw_p_terrestrial, gw_preferred_event,
            gw_pipeline, gw_search, gw_instruments, gw_skymap_file, gw_skymap_path and
            gw_status.

    Examples
    --------
    A trimmed view of the result, showing two SNe matched to the same superevent::

        name     discoverydate              superevent_id  days_from_gw  gw_p_bns  gw_status
        2019ebq  2019-04-25 12:11:31+00:00  S190425z              0.162     0.999  ok
        2019eff  2019-04-25 13:59:31+00:00  S190425z              0.237     0.999  ok
    """
    if df_sesn.empty or gw_events.empty:
        return pd.DataFrame()

    chunks = []
    for i, gw in gw_events.iterrows():
        if pd.isna(gw["gw_time"]):
            continue

        if verbose:
            print(f"Checking grav wave {i}: {gw['superevent_id']} ({gw['gw_time']})")

        start = gw["gw_time"] - pd.Timedelta(days=window_days)
        end = gw["gw_time"] + pd.Timedelta(days=window_days)
        sn = df_sesn[(df_sesn["discoverydate"] >= start) & (df_sesn["discoverydate"] <= end)].copy()
        if sn.empty:
            continue
        sn["superevent_id"] = gw["superevent_id"]
        sn["gw_time"] = gw["gw_time"]
        sn["gps_time"] = gw["gps_time"]
        sn["days_from_gw"] = (sn["discoverydate"] - gw["gw_time"]).dt.total_seconds() / SECONDS_PER_DAY
        for col in [
            "far_per_year",
            "p_bns",
            "p_nsbh",
            "p_bbh",
            "p_terrestrial",
            "preferred_event",
            "pipeline",
            "search",
            "instruments",
            "skymap_file",
            "skymap_path",
            "status",
        ]:
            sn[f"gw_{col}"] = gw.get(col)
        chunks.append(sn)

    if not chunks:
        return pd.DataFrame()
    return (
        pd.concat(chunks, ignore_index=True).sort_values(["gw_time", "discoverydate"]).reset_index(drop=True)
    )


def add_crossmatch_columns(sn_rows, result, *, cosmology_label, distance_column, credible_level):
    """Attach ligo.skymap crossmatch result fields to a copy of the matched SN rows.

    Parameters
    ----------
    sn_rows : pandas.DataFrame
        SN rows that were passed to crossmatch, in the same order as the coordinates given
        to it, so the per-SN result arrays line up positionally.
    result : ligo.skymap.postprocess.crossmatch.CrossmatchResult
        The named tuple returned by crossmatch. Its searched_* fields are one value per
        input coordinate; its contour_* fields are one value per requested contour.
    cosmology_label : str
        Key from COSMOLOGIES naming the cosmology used to place the SNe in distance, for
        example "Planck18".
    distance_column : str
        Name of the sn_rows column holding the luminosity distance in Mpc under that
        cosmology, for example "dist_mpc_Planck18". Pairing it with a cosmology_label it
        does not belong to is not detected: the row would name one cosmology and carry the
        other's distance.
    credible_level : float
        Credible level the inside_*_credible_level flags are judged against.

    Returns
    -------
    pandas.DataFrame
        A copy of sn_rows with the index reset and these columns added:

        cosmology, distance_column, credible_level, sn_dist_mpc
            The arguments above recorded per row, with sn_dist_mpc the distance in Mpc
            actually used for this SN. credible_level makes the frame self-describing, so
            a consumer such as skymap_plots draws and labels the contour the numbers were
            actually computed at rather than assuming a default.
        searched_area_deg2, searched_prob_2d, offset_deg
            Sky-only results, marginalized over distance. The area searched before reaching
            the SN, the SN's 2D credible level, and its angular offset from the skymap's
            maximum-probability point.
        searched_prob_dist
            Credible level of the SN's distance in the distance posterior marginalized over
            the sky.
        searched_vol_mpc3, searched_prob_vol, probdensity_vol
            The 3D results. The volume searched before reaching the SN, the SN's credible
            level under probability density per volume ranking, and the posterior density
            at its position.
        searched_prob_3d_density_rank
            A copy of searched_prob_vol, named to make the ranking explicit next to
            searched_prob_2d.
        credible_volume_mpc3, credible_area_deg2
            Size of the credible_level contour for the event as a whole, so identical on
            every row, or NaN if crossmatch returned no contours.
        inside_2d_credible_level, inside_3d_credible_level
            Whether the SN falls inside the credible_level contour under the 2D and 3D
            rankings. These are not nested quantities, so an SN can be inside one and
            outside the other.
    """
    out = sn_rows.copy().reset_index(drop=True)
    out["cosmology"] = cosmology_label
    out["distance_column"] = distance_column
    out["credible_level"] = credible_level
    out["sn_dist_mpc"] = out[distance_column]
    out["searched_area_deg2"] = np.atleast_1d(result.searched_area)
    out["searched_prob_2d"] = np.atleast_1d(result.searched_prob)
    out["offset_deg"] = np.atleast_1d(result.offset)
    out["searched_prob_dist"] = np.atleast_1d(result.searched_prob_dist)
    out["searched_vol_mpc3"] = np.atleast_1d(result.searched_vol)
    out["searched_prob_vol"] = np.atleast_1d(result.searched_prob_vol)
    out["searched_prob_3d_density_rank"] = out["searched_prob_vol"]
    out["probdensity_vol"] = np.atleast_1d(result.probdensity_vol)
    out["credible_volume_mpc3"] = result.contour_vols[0] if result.contour_vols else np.nan
    out["credible_area_deg2"] = result.contour_areas[0] if result.contour_areas else np.nan
    out["inside_2d_credible_level"] = out["searched_prob_2d"] <= credible_level
    out["inside_3d_credible_level"] = out["searched_prob_vol"] <= credible_level
    return out


def failed_spatial_rows(sn_rows, status, cosmology=np.nan, credible_level=np.nan):
    """Mark SN rows as failing the spatial crossmatch with a given status.

    Gives failed rows the flag columns a successful crossmatch would have, so they can be
    concatenated with successes and filtered on spatial_status instead of being dropped.

    Parameters
    ----------
    sn_rows : pandas.DataFrame
        SN rows that could not be crossmatched.
    status : str
        Reason for the failure, recorded in the spatial_status column. One of
        "missing_skymap", "skymap_has_no_distance_columns", "skymap_read_failed: ..." or
        "crossmatch_failed: ...".
    cosmology : str or float, optional
        Cosmology label to record. Set to a COSMOLOGIES key such as "Planck18" only for
        "crossmatch_failed: ...", the one failure raised inside the per-cosmology loop where
        a label is already known. The other three happen before that loop is entered and
        leave it NaN. Defaults to np.nan.
    credible_level : float, optional
        Credible level to record, following cosmology: set only for
        "crossmatch_failed: ...". The column describes the level a row's measurements were
        computed at, and these rows have none, so NaN is the honest value rather than the
        level the run was configured with. Defaults to np.nan.

    Returns
    -------
    pandas.DataFrame
        A copy of sn_rows with spatial_status, cosmology and credible_level set, and both
        inside_2d_credible_level and inside_3d_credible_level set to False. The crossmatch
        measurement columns are absent here, so they read as NaN once these rows are
        concatenated with successful ones.
    """
    out = sn_rows.copy()
    out["spatial_status"] = status
    out["cosmology"] = cosmology
    out["credible_level"] = credible_level
    out["inside_2d_credible_level"] = False
    out["inside_3d_credible_level"] = False
    return out


def run_3d_spatial_crossmatch(temporal_matches, gw_events, *, credible_level=0.5):
    """Run the 3D credible-volume crossmatch for each cosmology on every temporal match.

    Groups the temporally matched SNe by superevent, reads each event's skymap once and
    caches it, then calls ligo.skymap.postprocess.crossmatch once per cosmology in
    COSMOLOGIES, since each cosmology puts the same SN at a different luminosity distance.
    A superevent whose skymap is missing, unreadable, or carries no DISTMU distance columns
    yields failure rows rather than dropping its SNe.

    Parameters
    ----------
    temporal_matches : pandas.DataFrame
        Output of temporal_crossmatch_sesn_to_gw. Must have superevent_id, ra and
        declination columns in degrees, plus a dist_mpc_<label> column for each label in
        COSMOLOGIES. Rows with a non-finite distance are skipped for that cosmology.
    gw_events : pandas.DataFrame
        Superevent table from fetch_gracedb_superevents, used to look up each event's
        skymap_path by superevent_id. Superevents absent from it are skipped.
    credible_level : float, optional
        Credible level the contours are computed at and the inside_*_credible_level flags
        are judged against. This is the one place the pipeline's chosen level is defaulted;
        it is recorded on every successful row, so downstream consumers read it from the
        frame rather than assuming a value. Defaults to 0.5.

    Returns
    -------
    pandas.DataFrame
        One row per (SN, event, cosmology) combination, so an SN that matched one event
        normally appears once per cosmology. Sorted by superevent_id, name and cosmology
        where those columns are present. Successful rows have spatial_status "ok" and the
        columns described in add_crossmatch_columns; failed rows carry the reason in
        spatial_status and have both inside_*_credible_level flags set to False. Empty
        DataFrame if either input is empty or no group produced rows.

    Examples
    --------
    A trimmed view of the result. The same SN appears once per cosmology, at the luminosity
    distance that cosmology gives it::

        name     superevent_id  cosmology  sn_dist_mpc  inside_3d_credible_level  spatial_status
        2019ebq  S190425z       Planck18         168.5  True                      ok
        2019ebq  S190425z       SHOES            156.2  True                      ok
    """
    if temporal_matches.empty or gw_events.empty:
        return pd.DataFrame()

    event_lookup = gw_events.set_index("superevent_id", drop=False)
    chunks = []
    skymap_cache = {}

    for superevent_id, sn_rows in temporal_matches.groupby("superevent_id"):
        if superevent_id not in event_lookup.index:
            continue
        event = event_lookup.loc[superevent_id]

        skymap_path = event.get("skymap_path")
        # pd.isna is checked first because a missing skymap_path reaches here as NaN as often
        # as it does as None, and NaN is truthy, so "not skymap_path" alone lets it through to
        # Path(), which raises TypeError on a float. A gw_events row that failed its file
        # listing has no skymap_path key at all and pandas fills it with NaN; newer pandas also
        # stores an assigned None as NaN in an object column.
        if pd.isna(skymap_path) or not skymap_path or not Path(skymap_path).exists():
            chunks.append(failed_spatial_rows(sn_rows, "missing_skymap"))
            continue
        if skymap_path not in skymap_cache:
            try:
                skymap_cache[skymap_path] = read_sky_map(skymap_path, moc=True)
            except Exception as exc:
                chunks.append(failed_spatial_rows(sn_rows, f"skymap_read_failed: {exc}"))
                continue
        skymap = skymap_cache[skymap_path]

        if "DISTMU" not in skymap.colnames:
            chunks.append(failed_spatial_rows(sn_rows, "skymap_has_no_distance_columns"))
            continue

        # Calculate distance and crossmatch for both SHOES and Planck18.
        for cosmology_label in COSMOLOGIES:
            # Get distance wrt specific cosmology model.
            distance_column = f"dist_mpc_{cosmology_label}"
            valid = sn_rows[np.isfinite(sn_rows[distance_column])].copy()
            if valid.empty:
                continue
            coords = SkyCoord(
                ra=valid["ra"].to_numpy() * u.deg,
                dec=valid["declination"].to_numpy() * u.deg,
                distance=valid[distance_column].to_numpy() * u.Mpc,
                frame="icrs",
            )

            # Run crossmatch for that distance.
            try:
                result = crossmatch(
                    skymap,
                    coords,
                    contours=(credible_level,),
                    cosmology=USE_COMOVING_VOLUME_RANKING,
                )
                out = add_crossmatch_columns(
                    valid,
                    result,
                    cosmology_label=cosmology_label,
                    distance_column=distance_column,
                    credible_level=credible_level,
                )
                out["spatial_status"] = "ok"
            except Exception as exc:
                out = failed_spatial_rows(valid, f"crossmatch_failed: {exc}", cosmology_label, credible_level)
                out["distance_column"] = distance_column
            chunks.append(out)

    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    sort_cols = [c for c in ["superevent_id", "name", "cosmology"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.reset_index(drop=True)
