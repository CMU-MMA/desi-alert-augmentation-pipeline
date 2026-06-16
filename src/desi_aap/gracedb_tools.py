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
MIN_BNS_NSBH_PROB_SUM = 0.9
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
    """Coerce a value to float, returning a default on failure or None."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def response_to_bytes(response):
    """Extract raw bytes from a GraceDB client file response."""
    if hasattr(response, "read"):
        data = response.read()
    else:
        data = response.content
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data


def response_to_text(response):
    """Extract decoded text from a GraceDB client file response."""
    return response_to_bytes(response).decode("utf-8", errors="replace")


def unversioned_file_names(files):
    """Return file names from a GraceDB file listing that aren't versioned copies."""
    return [name for name in files if not re.search(r",\d+$", name)]


def choose_pastro_file(superevent, files):
    """Pick the best p_astro classification file name from a GraceDB file listing."""
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
    """Download and parse a superevent's p_astro classification file."""
    if not pastro_file:
        return {}, None
    payload = response_to_text(client.files(superevent_id, pastro_file))
    data = json.loads(payload)
    return data, pastro_file


def skymap_priority(name):
    """Rank a skymap file name by preference; lower is preferred."""
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
    """Pick the best available skymap file name from a GraceDB file listing."""
    names = list(files)
    candidates = [name for name in names if skymap_priority(name) < SKYMAP_PRIORITY_IGNORE]
    if not candidates:
        return None
    return sorted(candidates, key=lambda name: (skymap_priority(name), name.lower()))[0]


def safe_file_part(value):
    """Sanitize a value for use as part of a local file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def download_gracedb_file(client, superevent_id, filename, outdir=SKYMAP_DIR):
    """Download a GraceDB file to a local cache directory, skipping if already present."""
    outdir.mkdir(exist_ok=True)
    local_name = f"{safe_file_part(superevent_id)}__{safe_file_part(filename)}"
    path = outdir / local_name
    if not path.exists():
        payload = response_to_bytes(client.files(superevent_id, filename))
        path.write_bytes(payload)
    return path


def gps_to_utc(gps_time):
    """Convert a GPS time to a UTC timestamp, passing through NaN."""
    if pd.isna(gps_time):
        return pd.NaT
    return pd.Timestamp(Time(float(gps_time), format="gps").to_datetime(), tz="UTC")


def fetch_gracedb_bns_nsbh_superevents():
    """Query GraceDB for production BNS/NSBH superevents passing the FAR and probability cuts."""
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

        p_bns = as_float(classification.get("BNS"), DEFAULT_CLASSIFICATION_PROBABILITY)
        p_nsbh = as_float(classification.get("NSBH"), DEFAULT_CLASSIFICATION_PROBABILITY)
        if not (p_bns + p_nsbh > MIN_BNS_NSBH_PROB_SUM):
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
                "p_bns": p_bns,
                "p_nsbh": p_nsbh,
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


def temporal_crossmatch_sesn_to_gw(df_sesn, gw_events):
    """Match SNe to GW events whose discovery date falls within the temporal window."""
    if df_sesn.empty or gw_events.empty:
        return pd.DataFrame()

    chunks = []
    for _, gw in gw_events.iterrows():
        if pd.isna(gw["gw_time"]):
            continue
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
    return pd.concat(chunks, ignore_index=True).sort_values(["gw_time", "discoverydate"]).reset_index(
        drop=True
    )


def add_crossmatch_columns(sn_rows, result, cosmology_label, distance_column):
    """Attach ligo.skymap crossmatch result fields to a copy of the matched SN rows."""
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
    """Mark SN rows as failing the spatial crossmatch with a given status."""
    out = sn_rows.copy()
    out["spatial_status"] = status
    out["cosmology"] = cosmology
    out["inside_2d_credible_level"] = False
    out["inside_3d_credible_level"] = False
    return out


def run_3d_spatial_crossmatch(temporal_matches, gw_events):
    """Run the 3D credible-volume crossmatch for each cosmology on every temporal match."""
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
        if "DISTMU" not in skymap.colnames:
            chunks.append(failed_spatial_rows(sn_rows, "skymap_has_no_distance_columns"))
            continue

        for cosmology_label in COSMOLOGIES:
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
            try:
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

    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    sort_cols = [c for c in ["superevent_id", "name", "cosmology"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.reset_index(drop=True)
