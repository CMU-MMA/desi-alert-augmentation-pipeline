"""Fetch GraceDB BNS/NSBH superevents and crossmatch them against SN catalogs.

Provides the temporal crossmatch (discovery time vs. GW time) and the 3D
spatial crossmatch (RA/Dec/distance vs. the GW skymap's credible volume).
"""

import json
import re
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

# GraceDB settings.
GRACEDB_SERVICE_URL = "https://gracedb.ligo.org/api/"
GRACEDB_CATEGORY = "Production"
GRACEDB_MAX_RESULTS = None
GRACEDB_QUERY_SIGFIGS = 12

# Event-selection settings.
FAR_THRESHOLD_PER_YEAR = 2.0
JULIAN_YEAR_DAYS = 365.25
SECONDS_PER_DAY = (1 * u.day).to_value(u.s)
JULIAN_YEAR_SECONDS = (JULIAN_YEAR_DAYS * u.day).to_value(u.s)
FAR_THRESHOLD_HZ = FAR_THRESHOLD_PER_YEAR / JULIAN_YEAR_SECONDS
GRACEDB_QUERY = f"category: {GRACEDB_CATEGORY} far < {FAR_THRESHOLD_HZ:.{GRACEDB_QUERY_SIGFIGS}g}"
MIN_CLASSIFICATION_PROB_SUM = 0.9  # TODO ask about mins for things like just BBH selection
DEFAULT_CLASSIFICATION_PROBABILITY = 0.0


# Temporal/spatial crossmatch settings.
TEMPORAL_WINDOW_DAYS = 14
CREDIBLE_LEVEL = 0.50
REQUIRE_2D_CREDIBLE_LEVEL = False

# crossmatch(..., cosmology=False) ranks the 3D posterior by probability
# density per luminosity-distance volume, matching the units in the skymaps.
# The two cosmology runs differ by the redshift-to-luminosity-distance
# conversion used for each SN.
USE_COMOVING_VOLUME_RANKING = True

# Local output directory for downloaded skymaps.
SKYMAP_DIR = Path("gracedb_skymaps")

# GraceDB skymap file-selection priorities. Lower is preferred.
SKYMAP_PRIORITY_BILBY_MULTIORDER = 0
SKYMAP_PRIORITY_BAYESTAR_MULTIORDER = 10
SKYMAP_PRIORITY_ANY_MULTIORDER = 20
SKYMAP_PRIORITY_BAYESTAR_FITS_GZ = 30
SKYMAP_PRIORITY_ANY_FITS_GZ = 40
SKYMAP_PRIORITY_ANY_FITS = 50
SKYMAP_VERSIONED_FILE_PRIORITY_PENALTY = 100
SKYMAP_PRIORITY_IGNORE = 1000


def as_float(value, default=np.nan):
    """Coerce a value to float, returning a default on failure or None.

    Normalizes the loosely typed values in GraceDB JSON payloads, which may be None,
    numeric strings, or already numeric, into plain floats.

    Args:
        value: Value to coerce. None, and anything float() rejects, falls back to default.
        default: Value returned when coercion fails. Defaults to np.nan.

    Returns:
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

    Args:
        response: Object returned by GraceDb.files(). Read from its read() method when it
            has one, otherwise from its content attribute.

    Returns:
        The response body as bytes.
    """
    if hasattr(response, "read"):
        data = response.read()
    else:
        data = response.content
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data


def unversioned_file_names(files):
    """Return file names from a GraceDB file listing that aren't versioned copies.

    GraceDB exposes every revision of a file under a ",N" suffix, (e.g.
    "bayestar.multiorder.fits,0", alongside the unsuffixed name that points at the latest
    revision. Only the unsuffixed names are kept.

    Args:
        files: Iterable of file names, such as the keys of
            client.files(superevent_id).json().

    Returns:
        List of the names without a ",N" version suffix, in input order.
    """
    return [name for name in files if not re.search(r",\d+$", name)]


def choose_pastro_file(superevent, files):
    """Pick the best p_astro classification file name from a GraceDB file listing.

    Args:
        superevent: Superevent dict from GraceDb.superevents(). Its
            preferred_event_data.pipeline entry, when present, drives the pipeline-specific
            candidate names.
        files: Iterable of file names available for the superevent.

    Returns:
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

    Args:
        client: Connected GraceDb client.
        superevent_id: Superevent identifier, for example "S190425z".
        pastro_file: File name to fetch, typically from choose_pastro_file. A falsy value
            skips the download.

    Returns:
        Tuple of (classification, classification_file). classification is the parsed JSON
        dict keyed by class name, with "BNS", "NSBH", "BBH" and "Terrestrial" probabilities;
        classification_file echoes back the name that was fetched. Returns ({}, None) when
        pastro_file is falsy.
    """
    if not pastro_file:
        return {}, None
    payload = response_to_bytes(client.files(superevent_id, pastro_file))
    data = json.loads(payload)
    return data, pastro_file


def skymap_priority(name):
    """Rank a skymap file name by preference; lower is preferred.

    Bilby is preferred over BAYESTAR, multiorder FITS over flat FITS, and unversioned names
    over the ",N" revisions, which take a fixed SKYMAP_VERSIONED_FILE_PRIORITY_PENALTY.
    Names that are not skymaps at all rank at SKYMAP_PRIORITY_IGNORE or above.

    Args:
        name: File name from a GraceDB file listing.

    Returns:
        Integer priority built from the SKYMAP_PRIORITY_* constants. Values below
        SKYMAP_PRIORITY_IGNORE identify usable skymaps.
    """
    lower = name.lower()
    version_penalty = SKYMAP_VERSIONED_FILE_PRIORITY_PENALTY if re.search(r",\d+$", lower) else 0
    if not lower.endswith((".multiorder.fits", ".fits.gz", ".fits")):
        return SKYMAP_PRIORITY_IGNORE + version_penalty
    if "bilby.multiorder.fits" in lower:
        return SKYMAP_PRIORITY_BILBY_MULTIORDER + version_penalty
    if "bayestar.multiorder.fits" in lower:
        return SKYMAP_PRIORITY_BAYESTAR_MULTIORDER + version_penalty
    if lower.endswith(".multiorder.fits"):
        return SKYMAP_PRIORITY_ANY_MULTIORDER + version_penalty
    if "bayestar" in lower and lower.endswith(".fits.gz"):
        return SKYMAP_PRIORITY_BAYESTAR_FITS_GZ + version_penalty
    if lower.endswith(".fits.gz"):
        return SKYMAP_PRIORITY_ANY_FITS_GZ + version_penalty
    if lower.endswith(".fits"):
        return SKYMAP_PRIORITY_ANY_FITS + version_penalty
    return SKYMAP_PRIORITY_IGNORE + version_penalty


def choose_skymap_file(files):
    """Pick the best available skymap file name from a GraceDB file listing.

    Args:
        files: Iterable of file names available for the superevent.

    Returns:
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

    Args:
        value: Value to sanitize; converted with str() first.

    Returns:
        The string with each run of characters outside [A-Za-z0-9_.-] replaced by a single
        underscore.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def download_gracedb_file(client, superevent_id, filename, outdir=SKYMAP_DIR):
    """Download a GraceDB file to a local cache directory, skipping if already present.

    The local name is "<superevent_id>__<filename>" with both parts sanitized, so files
    from different superevents never collide. An existing file is trusted and not
    re-downloaded, which makes repeat runs of fetch_gracedb_superevents cheap.

    Args:
        client: Connected GraceDb client.
        superevent_id: Superevent identifier, for example "S190425z".
        filename: Remote file name to fetch, for example "bilby.multiorder.fits".
        outdir: Directory to write into, created if it does not exist. Defaults to
            SKYMAP_DIR, which is relative to the working directory.

    Returns:
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

    Args:
        gps_time: GPS seconds, as reported by a superevent's t_0 or by its preferred
            event's gpstime. May be NaN.

    Returns:
        A timezone-aware pandas Timestamp in UTC, or pd.NaT if gps_time is missing.
    """
    if pd.isna(gps_time):
        return pd.NaT
    return pd.Timestamp(Time(float(gps_time), format="gps").to_datetime(), tz="UTC")


def fetch_gracedb_superevents(se_types):
    """Query GraceDB for production superevents of the given types passing the FAR and probability cuts.

    Walks the public GraceDB production superevents whose false-alarm rate is below
    FAR_THRESHOLD_PER_YEAR, reads each one's p_astro classification, keeps those whose
    combined probability across the requested types clears MIN_CLASSIFICATION_PROB_SUM, and
    downloads the best available skymap into SKYMAP_DIR.

    Per-superevent failures are not fatal. They are recorded in the row's status column as
    "file_list_failed: ...", "p_astro_failed: ..." or "skymap_download_failed: ..." and the
    scan continues.

    Args:
        se_types: List of superevent type strings to include, matched case-insensitively
            against the p_astro class names. Supported values: "bns", "nsbh", "bbh".
            Only superevents whose combined probability for the requested types exceeds
            MIN_CLASSIFICATION_PROB_SUM are returned.

    Returns:
        DataFrame with one row per passing superevent, sorted by gw_time, with columns:
          superevent_id: GraceDB identifier, for example "S190425z".
          gw_time, gps_time: merger time as a UTC Timestamp and as raw GPS seconds.
          far_hz, far_per_year: false-alarm rate in the GraceDB units and per Julian year.
          p_bns, p_nsbh, p_bbh, p_terrestrial: p_astro probabilities, 0.0 when absent.
          classification_file: name of the p_astro file the probabilities came from.
          preferred_event, pipeline, search, instruments: preferred-event metadata.
          labels: the superevent's GraceDB labels, comma-joined.
          skymap_file, skymap_path: chosen remote skymap name, and the local path it was
            downloaded to, or None if there was no skymap or the download failed.
          status: "ok", or the reason the superevent was only partly processed.
        Superevents whose file listing could not be read appear with only superevent_id,
        far_hz, far_per_year and status set; their other columns are NaN. Returns an empty
        DataFrame if nothing passes the cuts.
    """
    classification_keys = [t.upper() for t in se_types]
    client = GraceDb(service_url=GRACEDB_SERVICE_URL)
    rows = []

    for superevent in client.superevents(query=GRACEDB_QUERY, max_results=GRACEDB_MAX_RESULTS):
        sid = superevent.get("superevent_id")
        preferred = superevent.get("preferred_event_data", {}) or {}
        far_hz = as_float(superevent.get("far"), as_float(preferred.get("far")))
        far_per_year = far_hz * JULIAN_YEAR_SECONDS
        if not np.isfinite(far_per_year) or far_per_year >= FAR_THRESHOLD_PER_YEAR:
            continue

        try:
            files = client.files(sid).json()
        except Exception as exc:
            rows.append(
                {
                    "superevent_id": sid,
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
            as_float(classification.get(key), DEFAULT_CLASSIFICATION_PROBABILITY)
            for key in classification_keys
        )
        if not (prob_sum > MIN_CLASSIFICATION_PROB_SUM):  # TODO - is this how we want to approach
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
                "p_bns": as_float(classification.get("BNS"), DEFAULT_CLASSIFICATION_PROBABILITY),
                "p_nsbh": as_float(classification.get("NSBH"), DEFAULT_CLASSIFICATION_PROBABILITY),
                "p_bbh": as_float(classification.get("BBH"), DEFAULT_CLASSIFICATION_PROBABILITY),
                "p_terrestrial": as_float(
                    classification.get("Terrestrial"), DEFAULT_CLASSIFICATION_PROBABILITY
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
    df_sesn, gw_events, verbose=False
):  # TODO rename to more general than sesn
    """Match SNe to GW events whose discovery date falls within the temporal window.

    Every SN discovered within TEMPORAL_WINDOW_DAYS on either side of an event's gw_time is
    paired with that event, so one SN can appear once per event it matches.

    Args:
        df_sesn: Transient catalog, such as the cleaned TNS table from clean_tns_catalog.
            Must have a 'discoverydate' column comparable to a pandas Timestamp, that is,
            tz-aware UTC datetimes.
        gw_events: Superevent table from fetch_gracedb_superevents. Events with a missing
            gw_time are skipped.
        verbose: If True, print each event as it is checked. Defaults to False.

    Returns:
        DataFrame with one row per (SN, event) pair, sorted by gw_time then discoverydate.
        It carries every df_sesn column plus superevent_id, gw_time, gps_time and
        days_from_gw, the signed offset in days from the GW, negative when the SN was
        discovered first. The event's own fields are copied across with a gw_ prefix:
        gw_far_per_year, gw_p_bns, gw_p_nsbh, gw_p_bbh, gw_p_terrestrial,
        gw_preferred_event, gw_pipeline, gw_search, gw_instruments, gw_skymap_file,
        gw_skymap_path and gw_status. Returns an empty DataFrame if either input is empty
        or nothing falls inside the window.
    """
    if df_sesn.empty or gw_events.empty:
        return pd.DataFrame()

    chunks = []
    for i, gw in gw_events.iterrows():
        if pd.isna(gw["gw_time"]):
            continue

        if verbose:
            print(f"Checking grav wave {i}: {gw['superevent_id']} ({gw['gw_time']})")

        start = gw["gw_time"] - pd.Timedelta(days=TEMPORAL_WINDOW_DAYS)
        end = gw["gw_time"] + pd.Timedelta(days=TEMPORAL_WINDOW_DAYS)
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


def add_crossmatch_columns(sn_rows, result, cosmology_label, distance_column):
    """Attach ligo.skymap crossmatch result fields to a copy of the matched SN rows.

    Args:
        sn_rows: SN rows that were passed to crossmatch, in the same order as the
            coordinates given to it, so the per-SN result arrays line up positionally.
        result: The CrossmatchResult named tuple returned by
            ligo.skymap.postprocess.crossmatch. Its searched_* fields are one value per
            input coordinate; its contour_* fields are one value per requested contour.
        cosmology_label: Key from COSMOLOGIES naming the cosmology used to place the SNe in
            distance, for example "Planck18".
        distance_column: Name of the sn_rows column holding the luminosity distance in Mpc
            under that cosmology, for example "dist_mpc_Planck18".

    Returns:
        A copy of sn_rows with the index reset and these columns added:
          cosmology, distance_column, sn_dist_mpc: the arguments above recorded per row,
            with sn_dist_mpc the distance in Mpc actually used for this SN.
          searched_area_deg2, searched_prob_2d, offset_deg: sky-only results, marginalized
            over distance. The area searched before reaching the SN, the SN's 2D credible
            level, and its angular offset from the skymap's maximum-probability point.
          searched_prob_dist: credible level of the SN's distance in the distance posterior
            marginalized over the sky.
          searched_vol_mpc3, searched_prob_vol, probdensity_vol: the 3D results. The volume
            searched before reaching the SN, the SN's credible level under probability
            density per volume ranking, and the posterior density at its position.
          searched_prob_3d_density_rank: a copy of searched_prob_vol, named to make the
            ranking explicit next to searched_prob_2d.
          credible_volume_mpc3, credible_area_deg2: size of the CREDIBLE_LEVEL contour for
            the event as a whole, so identical on every row, or NaN if crossmatch returned
            no contours.
          inside_2d_credible_level, inside_3d_credible_level: whether the SN falls inside
            the CREDIBLE_LEVEL contour under the 2D and 3D rankings. These are not nested
            quantities, so an SN can be inside one and outside the other.
    """
    out = sn_rows.copy().reset_index(drop=True)
    out["cosmology"] = cosmology_label
    out["distance_column"] = distance_column
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
    out["inside_2d_credible_level"] = out["searched_prob_2d"] <= CREDIBLE_LEVEL
    out["inside_3d_credible_level"] = out["searched_prob_vol"] <= CREDIBLE_LEVEL
    return out


def failed_spatial_rows(sn_rows, status, cosmology=np.nan):
    """Mark SN rows as failing the spatial crossmatch with a given status.

    Gives failed rows the flag columns a successful crossmatch would have, so they can be
    concatenated with successes and filtered on spatial_status instead of being dropped.

    Args:
        sn_rows: SN rows that could not be crossmatched.
        status: Reason for the failure, recorded in the spatial_status column. One of
            "missing_skymap", "skymap_has_no_distance_columns", "skymap_read_failed: ..."
            or "crossmatch_failed: ...".
        cosmology: Cosmology label to record, or NaN when the failure happened before a
            cosmology was chosen. Defaults to np.nan.

    Returns:
        A copy of sn_rows with spatial_status and cosmology set, and both
        inside_2d_credible_level and inside_3d_credible_level set to False. The crossmatch
        measurement columns are absent here, so they read as NaN once these rows are
        concatenated with successful ones.
    """
    out = sn_rows.copy()
    out["spatial_status"] = status
    out["cosmology"] = cosmology
    out["inside_2d_credible_level"] = False
    out["inside_3d_credible_level"] = False
    return out


def run_3d_spatial_crossmatch(temporal_matches, gw_events):
    """Run the 3D credible-volume crossmatch for each cosmology on every temporal match.

    Groups the temporally matched SNe by superevent, reads each event's skymap once and
    caches it, then calls ligo.skymap.postprocess.crossmatch once per cosmology in
    COSMOLOGIES, since each cosmology puts the same SN at a different luminosity distance.
    A superevent whose skymap is missing, unreadable, or carries no DISTMU distance columns
    yields failure rows rather than dropping its SNe.

    Args:
        temporal_matches: Output of temporal_crossmatch_sesn_to_gw. Must have superevent_id,
            ra and declination columns in degrees, plus a dist_mpc_<label> column for each
            label in COSMOLOGIES. Rows with a non-finite distance are skipped for that
            cosmology.
        gw_events: Superevent table from fetch_gracedb_superevents, used to look up each
            event's skymap_path by superevent_id. Superevents absent from it are skipped.

    Returns:
        DataFrame with one row per (SN, event, cosmology) combination, so an SN that
        matched one event normally appears once per cosmology. Sorted by superevent_id,
        name and cosmology where those columns are present. Successful rows have
        spatial_status "ok" and the columns described in add_crossmatch_columns; failed
        rows carry the reason in spatial_status and have both inside_*_credible_level flags
        set to False. Returns an empty DataFrame if either input is empty or no group
        produced rows.
    """
    if temporal_matches.empty or gw_events.empty:
        return pd.DataFrame()

    event_lookup = gw_events.set_index("superevent_id", drop=False)
    chunks = []
    skymap_cache = {}

    for superevent_id, sn_rows in temporal_matches.groupby("superevent_id"):
        # Get the event.
        if superevent_id not in event_lookup.index:
            continue
        event = event_lookup.loc[superevent_id]

        # Get the skymap (from cache, or get and cache it).
        skymap_path = event.get("skymap_path")
        if not skymap_path or not Path(skymap_path).exists():
            chunks.append(failed_spatial_rows(sn_rows, "missing_skymap"))
            continue
        if skymap_path not in skymap_cache:
            try:
                skymap_cache[skymap_path] = read_sky_map(skymap_path, moc=True)
            except Exception as exc:
                chunks.append(failed_spatial_rows(sn_rows, f"skymap_read_failed: {exc}"))
                continue
        skymap = skymap_cache[skymap_path]

        # Get the distance, if available.
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
                # TODO: what format/schema/etc is result? and what does out look like?
                result = crossmatch(
                    skymap,
                    coords,
                    contours=(CREDIBLE_LEVEL,),
                    cosmology=USE_COMOVING_VOLUME_RANKING,
                )
                out = add_crossmatch_columns(valid, result, cosmology_label, distance_column)
                out["spatial_status"] = "ok"
            except Exception as exc:
                out = failed_spatial_rows(valid, f"crossmatch_failed: {exc}", cosmology_label)
                out["distance_column"] = distance_column
            chunks.append(out)

    # Return a sorted dataframe.
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    sort_cols = [c for c in ["superevent_id", "name", "cosmology"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.reset_index(drop=True)
