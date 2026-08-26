"""Fetch GraceDB BNS/NSBH superevents and crossmatch them against SN catalogs.

Provides the temporal crossmatch (discovery time vs. GW time) and the 3D
spatial crossmatch (RA/Dec/distance vs. the GW skymap's credible volume).

None of the cuts these functions apply carries a default. A caller names each
one, or takes the pipeline's from the ``[localize]`` table in ``config.toml``. A
default here would be a second place for a value to live, with nothing keeping
it in step with the config a run was actually made from.
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
from desi_aap.gracedb_cache import (
    SKYMAP_SUBDIR,
    atomic_write_bytes,
    latest_revision,
    safe_file_part,
    superevent_fingerprint,
)

# Time-unit conversions, used for the false-alarm rates GraceDB reports in Hz and for
# the SN-to-GW offsets. astropy's yr is the Julian year, 365.25 days.
SECONDS_PER_DAY = (1 * u.day).to_value(u.s)
JULIAN_YEAR_SECONDS = (1 * u.yr).to_value(u.s)

# crossmatch's `cosmology` flag chooses how the 3D posterior is ranked, and with it
# what the volume columns mean (ligo.skymap 2.5.4):
#
#   False  ranks by probability density per luminosity-distance volume, matching the
#          units in the skymaps, and reports Euclidean luminosity-distance volumes.
#   True   ranks by probability density per unit comoving volume, scaling each voxel
#          by dVC_dVL_for_DL(r) first, and reports comoving volumes.
#
# The flag moves searched_vol_mpc3, searched_prob_vol, probdensity_vol and
# credible_volume_mpc3, and so inside_3d_credible_level with them; it is applied after
# the 2D and distance-marginal quantities are computed, which are unaffected. Distinct
# from COSMOLOGIES despite the name: the two cosmology runs differ by the
# redshift-to-luminosity-distance conversion used for each SN, not by this flag.
USE_COMOVING_VOLUME_RANKING = True

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


def download_gracedb_file(client, superevent_id, filename, outdir, *, force=False):
    """Download a GraceDB file to a local cache directory, skipping if already present.

    The local name is "<superevent_id>__<filename>" with both parts sanitized, so files from
    different superevents never collide. An existing file is reused, which makes repeat runs of
    fetch_gracedb_superevents cheap.

    Two things existence alone does not establish, both handled by the caller rather than here:

    The file may have been superseded. GraceDB points an unversioned name at the newest revision
    of that file, so the bytes behind "bayestar.multiorder.fits" change when a new one is uploaded,
    and a copy taken beforehand stays stale forever. fetch_gracedb_superevents compares the
    revision recorded in the cache entry against latest_revision of the current listing and passes
    force=True when it has moved.

    The file may be truncated. Writing goes through atomic_write_bytes, so a run killed partway
    through leaves no file rather than a partial one; a copy written before that was true is not
    detectable here and must be deleted by hand.
    # TODO for Xander: skymaps downloaded before this change carry no recorded revision, so the
    # first run adopts them as they are. Delete the skymap directory once if you want them all
    # re-fetched at known revisions.

    Parameters
    ----------
    client : ligo.gracedb.rest.GraceDb
        Connected GraceDB client.
    superevent_id : str
        Superevent identifier, e.g. "S190425z".
    filename : str
        Remote file name to fetch, e.g. "bilby.multiorder.fits".
    outdir : pathlib.Path
        Directory to write into, created if it does not exist. Required rather than defaulted:
        a module-level default resolved against the working directory is what let the same
        skymaps be downloaded into two different notebook directories. Normally
        GraceDbCache.skymap_dir.
    force : bool, optional
        Download even when a local copy exists, replacing it. Defaults to False.

    Returns
    -------
    pathlib.Path
        Path to the local copy of the file.
    """
    local_name = f"{safe_file_part(superevent_id)}__{safe_file_part(filename)}"
    path = outdir / local_name
    if force or not path.exists():
        payload = response_to_bytes(client.files(superevent_id, filename))
        atomic_write_bytes(path, payload)
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
    cache,
    far_threshold_per_year,
    min_classification_prob_sum,
    max_results=None,
    force_refresh=False,
):
    """Query GraceDB for superevents passing the FAR and classification cuts.

    Per-superevent failures are not fatal. They are recorded in the row's status column and
    the scan continues.

    The superevents() listing is always fetched live; the per-superevent work behind it -- the
    file listing, the p_astro download and the skymap download -- is served from cache when it
    can be. That split is deliberate. The listing is a handful of requests against roughly two
    per superevent, and it is the signal every freshness decision is made from, so keeping it
    live means a retracted, backfilled or re-ranked superevent is noticed with no way for the
    cache to drift. See desi_aap.gracedb_cache for what is stored and how staleness is judged.

    A superevent that fails the classification cut is still cached, so the next run does not
    re-download the p_astro that will fail it again. Only the skymap, which is the expensive
    part, waits until a superevent has passed every cut.

    Parameters
    ----------
    se_types : list of str
        Superevent type strings to include, matched case-insensitively against the p_astro
        class names. Supported values: "bns", "nsbh", "bbh". Only superevents whose
        combined probability for the requested types exceeds min_classification_prob_sum
        are returned.
    cache : desi_aap.gracedb_cache.GraceDbCache
        Where the per-superevent metadata and the skymaps are kept. Required, and with no
        default anywhere in the call chain, so the location is always an explicit decision;
        build one from the pipeline config with GraceDbConfig.to_cache().
    far_threshold_per_year : float
        False-alarm-rate cut in events per Julian year. Applied twice: once in the GraceDB
        query itself, converted to the Hz the API expects, and once locally, since a
        superevent whose own FAR is missing is judged on its preferred event's instead.
    min_classification_prob_sum : float
        Minimum combined p_astro probability across se_types, exclusive.
        # TODO ask about mins for things like just BBH selection - confirmed that 0.9 is good
    max_results : int or None, optional
        Cap on the number of superevents the query returns, passed straight to
        GraceDb.superevents(). None, the default, means no cap.
    force_refresh : bool, optional
        Re-fetch every superevent from GraceDB and overwrite its cache entry and its skymap,
        ignoring what is already stored. Defaults to False. This is the way to repair a cache
        believed to hold wrong bytes rather than merely old ones: deleting an entry file
        re-reads that superevent's metadata, but leaves its skymap in place, since nothing in
        a re-read listing says the local copy is damaged.

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
            there was no skymap or the download failed. The path is absolute so that
            run_3d_spatial_crossmatch and skymap_plots can open it directly: neither is given
            the cache, so neither has a root to resolve a relative path against.
        status
            "ok", or the reason the superevent was only partly processed: either
            "file_list_failed: ..." or "skymap_download_failed: ...". A superevent whose
            p_astro could not be read is dropped by the probability cut rather than
            returned, so that failure never appears here.  #TODO is this the desired behavior?
        cache_status
            What the cache did for this superevent: "hit" (no per-superevent request made),
            "miss" (nothing was cached), "stale_fingerprint" (a field the listing reports had
            moved), "stale_age" (younger than the cache's recheck window, so re-checked on
            principle) or "forced" (force_refresh was set). Reported so a scheduled run can be
            confirmed to be using the cache without watching the network.

        Superevents whose file listing could not be read appear with only superevent_id,
        far_hz, far_per_year, status and cache_status set; their other columns are NaN. Empty
        DataFrame if nothing passes the cuts.

    Examples
    --------
    A trimmed view of the result, one row per superevent::

        superevent_id  gw_time                           far_per_year  p_bns  p_nsbh  status  cache_status
        S190425z       2019-04-25 08:18:05.011549+00:00      1.43e-05  0.999   0.000  ok      hit
        S190814bv      2019-08-14 21:11:16.012957+00:00      6.41e-26  0.000   0.998  ok      hit
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
    # Taken once so every superevent in a scan is judged against the same clock, rather than
    # against one that drifts across a run long enough for it to matter.
    now = pd.Timestamp.now(tz="UTC")

    for superevent in client.superevents(query=query, max_results=max_results):
        sid = superevent.get("superevent_id")
        preferred = superevent.get("preferred_event_data", {}) or {}
        far_hz = as_float(superevent.get("far"), as_float(preferred.get("far")))
        far_per_year = far_hz * JULIAN_YEAR_SECONDS
        if not np.isfinite(far_per_year) or far_per_year >= far_threshold_per_year:
            continue

        # Resolved before the file work rather than just before the row is built, because the
        # cache's age backstop is judged on the merger time.
        gps_time = as_float(superevent.get("t_0"), as_float(preferred.get("gpstime")))
        gw_time = gps_to_utc(gps_time)

        cached = None if force_refresh else cache.read_entry(sid)
        cache_status = (
            "forced" if force_refresh else cache.status(cached, superevent, gw_time=gw_time, now=now)
        )

        if cache_status == "hit":
            file_names = list(cached["files"])
        else:
            try:
                file_names = sorted(client.files(sid).json())
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
                        "cache_status": cache_status,
                    }
                )
                continue

        # A re-check that finds the listing unchanged still saves the p_astro download, which is
        # the common case inside the recheck window: the superevent was looked at again because
        # it is young, not because anything about it moved.
        listing_unchanged = cached is not None and list(cached.get("files") or []) == file_names

        if cache_status == "hit" or listing_unchanged:
            classification = cached.get("classification") or {}
            classification_file = cached.get("classification_file")
            classification_error = cached.get("classification_error")
        else:
            pastro_file = choose_pastro_file(superevent, file_names)
            try:
                classification, classification_file = load_classification(client, sid, pastro_file)
            except Exception as exc:
                classification = {}
                classification_file = pastro_file
                classification_error = str(exc)
            else:
                classification_error = None
        # Replayed from the cache rather than recorded as "ok", so a cached failure reads the same
        # as a fresh one instead of quietly becoming a clean row on the next run.
        status = f"p_astro_failed: {classification_error}" if classification_error else "ok"

        entry = {
            "superevent_id": sid,
            # When the data was fetched, not when it was last confirmed current: a hit leaves the
            # entry untouched, so this stays the age of the bytes rather than of the check.
            "fetched_utc": now.isoformat(),
            "fingerprint": superevent_fingerprint(superevent),
            "files": file_names,
            "classification": classification,
            "classification_file": classification_file,
            "classification_error": classification_error,
            "skymap_file": None,
            "skymap_revision": None,
            "skymap_relpath": None,
        }

        prob_sum = sum(
            as_float(classification.get(key), default_classification_probability)
            for key in classification_keys
        )
        if not (prob_sum > min_classification_prob_sum):  # TODO - is this how we want to approach
            # The entry is written even though this superevent is being dropped. Failing the
            # cut cost the same two requests as passing it would have, and this cut rejects
            # most of what the FAR cut admits -- 334 of 349 against the production API in
            # August 2026 -- so not caching the failures would give up most of the saving.
            if cache_status != "hit":
                cache.write_entry(sid, entry)
            continue

        skymap_file = choose_skymap_file(file_names)
        skymap_path = None
        if skymap_file:
            skymap_revision = latest_revision(file_names, skymap_file)
            entry["skymap_file"] = skymap_file
            entry["skymap_revision"] = skymap_revision
            # GraceDB repoints an unversioned name at each new revision, so a local copy taken
            # before one was uploaded is stale even though the name still matches. force_refresh
            # replaces it too: it is the documented way to recover a skymap whose bytes are
            # wrong rather than merely old, which no revision comparison can detect.
            superseded = cached is not None and cached.get("skymap_revision") != skymap_revision
            try:
                skymap_path = download_gracedb_file(
                    client, sid, skymap_file, cache.skymap_dir, force=force_refresh or superseded
                )
            except Exception as exc:
                status = f"skymap_download_failed: {exc}"
            else:
                entry["skymap_relpath"] = (Path(SKYMAP_SUBDIR) / skymap_path.name).as_posix()

        # A hit can still write: an earlier failed skymap download has retried successfully here.
        # A failure writes too, keeping the metadata that worked; only the download is retried.
        if cache_status != "hit" or entry["skymap_relpath"] != (cached or {}).get("skymap_relpath"):
            cache.write_entry(sid, entry)

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
                "skymap_path": str(cache.resolve(entry["skymap_relpath"])) if skymap_path else None,
                "status": status,
                "cache_status": cache_status,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("gw_time").reset_index(drop=True)


def temporal_crossmatch_sesn_to_gw(
    df_sesn, gw_events, *, window_days, verbose=False
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
    window_days : float
        Half-width of the matching window in days, applied either side of gw_time and
        inclusive at both ends.
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
        ``gw_*``
            The event's own fields copied across under a ``gw_`` prefix: gw_far_per_year,
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


def summarize_temporal_matches(temporal_matches, gw_events):
    """Count the temporally matched SNe for each superevent, as a column on the event table.

    Answers "which events had anything nearby in time, and how much?" over the whole scan,
    including the events nothing matched, which is why it reads gw_events rather than
    counting temporal_matches alone: an event with no match contributes no rows there and
    would otherwise vanish from the tally.

    Parameters
    ----------
    temporal_matches : pandas.DataFrame
        Output of temporal_crossmatch_sesn_to_gw, one row per (SN, event) pair. Counted by
        the (superevent_id, gw_time) pair it copies from gw_events, so an SN matched to two
        events counts once against each.
    gw_events : pandas.DataFrame
        Superevent table from fetch_gracedb_superevents. Every row is kept, matched or not.

    Returns
    -------
    pandas.DataFrame
        A copy of gw_events, in its order, with one integer column added:

        n_temporal_sesn
            Number of SNe that matched the event in time; 0 for an event none matched, and
            for the stub row of an event whose file listing could not be read, which
            carries no gw_time to match on.

        Empty DataFrame if gw_events is empty, since there is then nothing to count
        against, regardless of what temporal_matches holds.

    Examples
    --------
    A trimmed view of the result, showing an event that two SNe matched and one that none
    did::

        superevent_id  gw_time                           far_per_year  p_bns  n_temporal_sesn
        S190425z       2019-04-25 08:18:05.017147+00:00      1.43e-05  0.999                2
        S190814bv      2019-08-14 21:11:16.012957+00:00      6.41e-26  0.000                0
    """
    if gw_events.empty:
        return pd.DataFrame()

    if temporal_matches.empty:
        # Short-circuited rather than merged against an empty frame of counts, for two
        # reasons. temporal_crossmatch_sesn_to_gw returns a bare DataFrame() when nothing
        # matched, which has no columns at all, so grouping it would raise KeyError; and an
        # empty frame built with the key names holds them as object dtype, which merges
        # without complaint but leaves a column that fillna then downcasts, which pandas
        # 2.3 deprecates. Assigning the zero directly sidesteps both.
        summary = gw_events.copy()
        summary["n_temporal_sesn"] = 0
        return summary

    group_keys = ["superevent_id", "gw_time"]
    counts = temporal_matches.groupby(group_keys, dropna=False).size().rename("n_temporal_sesn").reset_index()
    summary = gw_events.merge(counts, on=group_keys, how="left")
    # The misses come back as NaN, which makes the column type float; the counts are whole.
    summary["n_temporal_sesn"] = summary["n_temporal_sesn"].fillna(0).astype(int)
    return summary


def display_temporal_summary(summary):
    """Narrow a temporal summary to the columns worth reading, for display.

    A convenience for the end of a scan, kept separate from summarize_temporal_matches so
    that the full frame stays available to anything working with the result rather than
    looking at it.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of summarize_temporal_matches. Names it does not carry are skipped rather
        than raising, since an empty scan returns a frame with no columns at all.

    Returns
    -------
    pandas.DataFrame
        A view of summary holding the columns below that exist, in this order.
    """
    # In reading order: what the event is, how loud and how likely, and how many SNe fell
    # in its window.
    columns = (
        "superevent_id",
        "gw_time",
        "far_per_year",
        "p_bns",
        "p_nsbh",
        "pipeline",
        "search",
        "skymap_file",
        "n_temporal_sesn",
        "status",
    )
    return summary[[c for c in columns if c in summary.columns]]


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
            rankings. At a given level neither region contains the other, so an SN can
            be inside one and outside the other.
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


def run_3d_spatial_crossmatch(temporal_matches, gw_events, *, credible_level):
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
    credible_level : float
        Credible level the contours are computed at and the inside_*_credible_level flags
        are judged against. Recorded on every successful row, so downstream consumers read
        it from the frame rather than assuming a value.

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


def select_coincidences(spatial_matches, *, require_2d_credible_level):
    """Keep the crossmatched rows whose SN landed inside the GW credible volume.

    The final cut of the pipeline: what survives here is the candidate list.

    Parameters
    ----------
    spatial_matches : pandas.DataFrame
        Output of run_3d_spatial_crossmatch, one row per (SN, event, cosmology). Rows whose
        spatial_status is not "ok" are dropped, so an event whose skymap was missing or
        unreadable contributes nothing rather than contributing an unmeasured row.
    require_2d_credible_level : bool
        Whether a row must fall inside the 2D credible level as well as the 3D one. At a
        given level neither region contains the other, so a row can satisfy one and not
        the other: searched_prob_vol can be well inside the contour while searched_prob_2d
        is outside it, for an SN whose sky position is unremarkable but whose distance
        lands on a high-density slice of the distance posterior.

    Returns
    -------
    pandas.DataFrame
        The surviving rows with their index reset and their columns unchanged. Since the
        input holds one row per cosmology, an SN inside the volume under both cosmologies
        appears twice and one inside under only a single cosmology appears once, which is
        the signal that it sits near the edge. Empty DataFrame if spatial_matches is empty
        or does not carry the columns the cut reads, as it will not when every superevent
        failed before its crossmatch.

    Examples
    --------
    A trimmed view of the result, showing an SN inside the volume under one cosmology but
    not the other::

        name     superevent_id  cosmology  sn_dist_mpc  searched_prob_3d_density_rank
        2019ebq  S190425z       Planck18         168.5                          0.312
    """
    required = {"spatial_status", "inside_3d_credible_level"}
    if require_2d_credible_level:
        required.add("inside_2d_credible_level")
    if spatial_matches.empty or not required.issubset(spatial_matches.columns):
        return pd.DataFrame()

    # .eq(True) rather than the column read as a mask directly: run_3d_spatial_crossmatch
    # concatenates groups that may not all carry these columns, and pandas fills a missing
    # one with NaN, leaving object dtype that cannot index a frame. NaN.eq(True) is False,
    # which is the right reading for a row the crossmatch never measured.
    inside = spatial_matches["inside_3d_credible_level"].eq(True)
    if require_2d_credible_level:
        inside &= spatial_matches["inside_2d_credible_level"].eq(True)
    return spatial_matches[spatial_matches["spatial_status"].eq("ok") & inside].reset_index(drop=True)


def display_coincidences(coincidences):
    """Narrow a coincidence list to the columns worth reading, for display.

    A convenience for the end of a scan, kept separate from select_coincidences so that the
    full frame stays available: skymap_plots.plot_3d_coincidence reads a row's skymap_path
    and credible_level, neither of which is shown here.

    Parameters
    ----------
    coincidences : pandas.DataFrame
        Output of select_coincidences. Names it does not carry are skipped rather than
        raising, since these span several stages: a run whose skymaps all failed to read
        has no searched_prob_2d, and an empty result has no columns at all.

    Returns
    -------
    pandas.DataFrame
        A view of coincidences holding the columns below that exist, in this order.
    """
    # In reading order: the event, then the SN, then the measurements the cut was made on,
    # then the coordinates needed to follow the candidate up.
    columns = (
        "superevent_id",
        "gw_time",
        "gw_far_per_year",
        "gw_p_bns",
        "gw_p_nsbh",
        "name",
        "type",
        "discoverydate",
        "days_from_gw",
        "redshift",
        "cosmology",
        "sn_dist_mpc",
        "searched_prob_2d",
        "searched_prob_3d_density_rank",
        "searched_prob_dist",
        "searched_area_deg2",
        "credible_area_deg2",
        "credible_volume_mpc3",
        "inside_2d_credible_level",
        "inside_3d_credible_level",
        "ra",
        "declination",
        "reporting_group",
        "internal_names",
    )
    return coincidences[[c for c in columns if c in coincidences.columns]]
