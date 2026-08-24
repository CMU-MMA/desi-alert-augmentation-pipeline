"""Score the cross-matched alerts against public GW event localizations.

The stage takes the matched-alert frame the crossmatch stage
produced and asks, of every alert in it, whether it could be the optical
counterpart of a LIGO/Virgo superevent: discovered close enough in time, and
sitting inside the credible volume of that event's 3D sky map.

The measurements themselves are from gracedb_tools, unchanged. That
module was written against the TNS supernova catalog, so it reads a flat frame
with ``name``, ``ra``, ``declination``, ``discoverydate`` and one
``dist_mpc_<label>`` column per cosmology.COSMOLOGIES entry.
The work this module does on top of it is the adaptation
(alerts_to_gw_match_input): an alert has no redshift of its own, so the
distance comes from the DESI host the crossmatch stage attached to it
(nearest_hosts), and its time comes from the alert's Julian date.

The result is written the way the crossmatch stage writes its own: one row per
alert, with the per-superevent results in a nested column
(attach_localizations), so a row that entered the stage as an alert
leaves it as an alert. Only the alerts that landed inside a credible volume are
kept and written; the temporal matches and near misses behind them are counted
in the summary and logged, not persisted.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nested_pandas as npd
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time

from desi_aap.config import PipelineConfig
from desi_aap.cosmology import COSMOLOGIES
from desi_aap.gracedb_tools import (
    fetch_gracedb_superevents,
    run_3d_spatial_crossmatch,
    select_coincidences,
    temporal_crossmatch_sesn_to_gw,
)
from desi_aap.stages.base import StageInputs, StageResult, input_result, write_frame
from desi_aap.stages.crossmatch import STAGE as CROSSMATCH_STAGE
from desi_aap.stages.crossmatch import catalog_specs
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)


STAGE = "localize"

# Prefix of the parquet file this stage writes.
OUTPUT_PREFIX = "coincidences"

# Nested column the per-superevent results land in, alongside the nested columns
# the crossmatch stage wrote.
NESTED_COLUMN = "gw_matches"

# Column carrying each row's alert back through gracedb_tools, so the results can
# be re-joined to the alert they came from. An explicit column rather than the
# index because temporal_crossmatch_sesn_to_gw concatenates with
# ignore_index=True, which discards whatever index went in. Underscore-prefixed
# like the _dist_arcsec and _id columns already in these frames, since it is
# bookkeeping rather than a measurement.
ALERT_KEY_COLUMN = "_alert_row"


def julian_dates_to_utc(julian_dates: pd.Series) -> pd.Series:
    """Convert alert Julian dates to UTC timestamps, passing through missing values.

    The scale is given explicitly rather than left to astropy's default so that
    it is stated where it is relied on: these Julian dates are compared against
    superevent times that gracedb_tools.gps_to_utc puts on UTC, and they are
    selected by a window that boom._to_jd builds on UTC as well.

    Parameters
    ----------
    julian_dates : pandas.Series
        Julian dates, such as an alert frame's ``candidate.jd``. Values that are
        not finite numbers, including a column that arrives as a string, become
        ``NaT`` rather than raising: astropy rejects a NaN outright, and one
        unusable alert should not end the run.

    Returns
    -------
    pandas.Series
        Timezone-aware UTC timestamps, on the input's index.
    """
    values = pd.to_numeric(julian_dates, errors="coerce").astype(float)
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    usable = np.isfinite(values.to_numpy())
    if usable.any():
        converted = Time(values.to_numpy()[usable], format="jd", scale="utc").to_datetime()
        out[usable] = pd.to_datetime(converted, utc=True)
    return out


def _host_fields(matches: npd.NestedFrame, name: str, fields: Sequence[str]) -> pd.DataFrame:
    """Flatten one catalog's nested column, checking it carries the fields wanted.

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        The crossmatch stage's output, one row per alert.
    name : str
        Nested column to read hosts from, i.e. the configured catalog name.
    fields : sequence of str
        Fields to read from the nested column: the redshift, the flag saying
        whether its fit is trustworthy, and the angular separation between alert
        and host.

    Returns
    -------
    pandas.DataFrame
        Flattened nested column, indexed by alert row. Empty DataFrame if the
        nested column is empty.
    """
    flat = matches[name].nest.to_flat()
    missing = [field for field in fields if field not in flat.columns]
    if missing:
        raise ValueError(
            f"Nested column {name!r} has no {', '.join(repr(f) for f in missing)} field, so "
            f"stage {STAGE!r} cannot read a host redshift from it. Every catalog in "
            "[crossmatch.catalogs.*] is read for one, so a catalog that carries no redshift "
            "cannot be cross-matched alongside those that do."
        )
    return flat


def nearest_hosts(
    matches: npd.NestedFrame,
    catalog_names: Sequence[str],
    *,
    min_redshift: float,
    redshift_field: str = "Z",
    warning_field: str = "ZWARN",
    separation_field: str = "_dist_arcsec",
    ok_warning: int = 0,
) -> pd.DataFrame:
    """Pick the host each alert takes its redshift from: the nearest across all catalogs.

    An alert can carry several host candidates (one per neighbour, per catalog)
    and the 3D crossmatch needs exactly one distance. The candidates from every
    catalog are pooled and the closest one on the sky wins, so which catalog an
    alert's redshift comes from is decided by the association rather than by the
    order in which the catalogs are configured.

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        The crossmatch stage's output, one row per alert.
    catalog_names : sequence of str
        Which catalogs to read hosts from, named as they are in
        [crossmatch.catalogs.*] -- each is a nested column on ``matches``. A
        catalog with no column on the frame is skipped, since a crossmatch that
        produced no column for it produced no hosts either. A catalog whose
        column is present but lacks one of the three fields below raises
        instead: that is a schema mismatch rather than an absence.
    min_redshift : float
        Smallest redshift treated as a real measurement, inclusive. Hosts below
        it are discarded rather than clamped, for the reasons set out in
        tns_catalog.clean_tns_catalog: a non-positive redshift yields no usable
        luminosity distance, and small negative ones make ``SkyCoord`` raise
        further down.
    redshift_field, warning_field, separation_field : str
        Fields read from each nested column: the redshift, the flag saying
        whether its fit is trustworthy, and the angular separation between alert
        and host. The first two are the DESI redshift catalog's own names; the
        third is what LSDB's crossmatch_nested records for every match it makes.
    ok_warning : int
        The one warning_field value meaning the redshift fit raised nothing.
        Every other value is a bitmask of the problems the fit hit.

    Returns
    -------
    pandas.DataFrame
        One row per alert that has a usable host, indexed by that alert's row in
        ``matches``, with columns:

        host_catalog
            The nested column the host came from.
        host_redshift
            Its redshift.
        host_sep_arcsec
            Its separation from the alert, which is what it won on.

        An alert appears here only if at least one of its hosts passed both
        cuts. One whose hosts all failed is left out entirely rather than kept
        with a missing redshift, which is what lets the caller read a missing
        row as "this alert has no usable host". If no alert qualifies the result
        is empty but still carries the three columns above.

    Raises
    ------
    ValueError
        If a named nested column is missing one of the three fields.
    """
    host_columns = ["host_catalog", "host_redshift", "host_sep_arcsec"]
    candidates = []
    for name in catalog_names:
        if name not in matches.columns or matches.empty:
            continue
        flat = _host_fields(matches, name, (redshift_field, warning_field, separation_field))
        if flat.empty:  # No candidates from this catalog, so it cannot contribute to the nearest host.
            continue
        candidates.append(
            pd.DataFrame(
                {
                    "host_catalog": name,
                    "host_redshift": pd.to_numeric(flat[redshift_field], errors="coerce"),
                    "host_sep_arcsec": pd.to_numeric(flat[separation_field], errors="coerce"),
                    "host_zwarn": pd.to_numeric(flat[warning_field], errors="coerce"),
                },
                index=flat.index,
            )
        )

    if not candidates:
        return pd.DataFrame(columns=host_columns)

    hosts = pd.concat(candidates)
    # NaN fails all three comparisons, so an unparseable redshift, an absent
    # ZWARN and a missing separation all leave by this line.
    hosts = hosts[
        hosts["host_zwarn"].eq(ok_warning)
        & (hosts["host_redshift"] >= min_redshift)
        & hosts["host_sep_arcsec"].notna()
    ]
    # Sorted on the catalog name as well so that two hosts at the same separation
    # -- the same object present in two releases -- resolve the same way on every
    # run rather than on whichever release happened to be concatenated first.
    hosts = hosts.sort_values(["host_sep_arcsec", "host_catalog"], kind="stable")
    hosts = hosts[~hosts.index.duplicated(keep="first")]
    return hosts[host_columns].sort_index()


# TODO (follow-up PR): the four alert column names defaulted below are BOOM's, and
# the package states them in three places under two mechanisms -- default_pipeline.json
# projects all four, crossmatch.py defines ALERT_RA_COLUMN and ALERT_DEC_COLUMN, and
# boom.py (sort_by) and this module spell candidate.jd and objectId as literals. Move
# all four to boom.py, the module that produces the columns and sits beside the
# projection that selects them, and have boom.query_alerts, the crossmatch stage and
# this one import them from there. Left alone here because it edits two modules outside
# the stage this PR adds; the literals are deliberate in the meantime, so that the alert
# schema is spelled one way in this module rather than two.
def alerts_to_gw_match_input(
    matches: npd.NestedFrame,
    hosts: pd.DataFrame,
    *,
    id_column: str = "objectId",
    ra_column: str = "candidate.ra",
    dec_column: str = "candidate.dec",
    time_column: str = "candidate.jd",
) -> pd.DataFrame:
    """Render the matched alerts as the transient table gracedb_tools reads.

    The column names below are that module's, not this pipeline's: they are what
    gracedb_tools.temporal_crossmatch_sesn_to_gw and
    gracedb_tools.run_3d_spatial_crossmatch look up, so the adaptation happens
    here rather than by parameterizing them and their callers.

    ``discoverydate`` is the alert's own timestamp rather than a discovery date
    in the TNS sense -- an alert is one detection of an object, not the first --
    so an object detected repeatedly is offered to the temporal match once per
    alert.

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        The crossmatch stage's output, one row per alert.
    hosts : pandas.DataFrame
        Chosen hosts from nearest_hosts. A subset of ``matches``, joined by
        index; an alert missing from this frame is dropped from the output
        frame, having no redshift and so no distance to be placed at.
    id_column, ra_column, dec_column, time_column : str
        Alert columns read for the transient's name, coordinates, and time.

    Returns
    -------
    pandas.DataFrame
        One row per alert with a usable host and a usable position and time,
        carrying ``name``, ``ra``, ``declination``, ``discoverydate``,
        ``redshift``, one ``dist_mpc_<label>`` per cosmology, the three host
        columns, and ALERT_KEY_COLUMN holding the alert's row in
        ``matches``. Empty DataFrame with those columns if nothing survives.
    """
    frame = pd.DataFrame(index=matches.index)
    frame[ALERT_KEY_COLUMN] = matches.index
    frame["name"] = matches[id_column].astype(str)
    frame["ra"] = pd.to_numeric(matches[ra_column], errors="coerce")
    frame["declination"] = pd.to_numeric(matches[dec_column], errors="coerce")
    frame["discoverydate"] = julian_dates_to_utc(matches[time_column])
    frame = frame.join(hosts, how="inner")
    frame["redshift"] = frame.pop("host_redshift")

    for label, cosmo in COSMOLOGIES.items():
        dist_column = f"dist_mpc_{label}"
        if frame.empty:
            # astropy's luminosity_distance goes through np.vectorize, which
            # rejects a size-0 input rather than returning an empty result.
            frame[dist_column] = np.array([], dtype=float)
            continue
        frame[dist_column] = cosmo.luminosity_distance(frame["redshift"].to_numpy()).to_value(u.Mpc)

    return frame.dropna(subset=["ra", "declination", "discoverydate"])


def coincident_localizations(
    gw_match_input: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window_days: float,
    credible_level: float,
    require_2d_credible_level: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the temporal and 3D spatial matches, and cut down to the coincidences.

    The three steps of gracedb_tools in the order the notebook runs them, with
    the intermediate frames counted on the way past. Only the last one is
    returned: the temporal matches and the near misses live on as the counts
    below rather than as rows on disk.

    Parameters
    ----------
    gw_match_input : pandas.DataFrame
        The alerts in gracedb_tools' column vocabulary, from
        alerts_to_gw_match_input.
    events : pandas.DataFrame
        Superevent table from gracedb_tools.fetch_gracedb_superevents.
    window_days : float
        Half-width of the temporal window, in days either side of each
        superevent's ``gw_time``.
    credible_level : float
        Credible level the contours are computed at and the coincidence cut is
        made against.
    require_2d_credible_level : bool
        Whether a coincidence must also fall inside the 2D credible level. The
        two regions do not contain one another, so requiring both discards real 3D
        coincidences; see gracedb_tools.select_coincidences.

    Returns
    -------
    coincidences : pandas.DataFrame
        One row per (alert, superevent, cosmology) that landed inside the
        credible volume, as gracedb_tools.select_coincidences returns it. Empty
        DataFrame if nothing did.
    counts : dict
        What each step saw: ``n_temporal_pairs``, ``n_alerts_temporal``,
        ``n_spatial_rows``, ``n_spatial_failed``, ``n_coincidences`` and
        ``n_alerts_coincident``.
    """
    temporal = temporal_crossmatch_sesn_to_gw(gw_match_input, events, window_days=window_days)
    counts: dict[str, Any] = {
        "n_temporal_pairs": len(temporal),
        "n_alerts_temporal": int(temporal[ALERT_KEY_COLUMN].nunique()) if not temporal.empty else 0,
    }

    spatial = run_3d_spatial_crossmatch(temporal, events, credible_level=credible_level)
    failures = (
        spatial["spatial_status"].ne("ok") if "spatial_status" in spatial.columns else pd.Series(dtype=bool)
    )
    counts["n_spatial_rows"] = len(spatial)
    counts["n_spatial_failed"] = int(failures.sum())
    if counts["n_spatial_failed"]:
        logger.info(
            "Spatial crossmatch outcomes: %s",
            spatial["spatial_status"].value_counts().to_dict(),
        )

    coincidences = select_coincidences(spatial, require_2d_credible_level=require_2d_credible_level)
    counts["n_coincidences"] = len(coincidences)
    counts["n_alerts_coincident"] = (
        int(coincidences[ALERT_KEY_COLUMN].nunique()) if not coincidences.empty else 0
    )
    return coincidences, counts


def attach_localizations(
    matches: npd.NestedFrame,
    gw_match_input: pd.DataFrame,
    coincidences: pd.DataFrame,
) -> npd.NestedFrame:
    """Fold the coincidences back onto the alerts they were measured for.

    The stage's output keeps the shape the crossmatch stage established: one row
    per alert, with each kind of match in a nested column. So the alert's own
    values, which gracedb_tools carried through from the transient frame, are
    dropped from the nested structure -- they are already on the row -- and only
    what the GW match added is nested. The distance is the exception, appearing
    in both places: on the row as ``dist_mpc_<label>``, once per cosmology, and
    inside each nested entry as ``sn_dist_mpc``, which is the one of them the
    numbers beside it were actually computed at.

    The written frame looks like this, trimmed to the columns that make the
    shape clear::

        objectId  candidate.ra  host_catalog  host_redshift  dist_mpc_SHOES  desi_dr1  gw_matches
        ZTF001         150.009  desi_dr1              0.030           130.4  [{...}]   [{...}, {...}]
        ZTF002         150.011  desi_dr2              0.021            91.2  [{...}]   [{...}, {...}]

    ``desi_dr1`` is the crossmatch stage's nested column, untouched.
    ``gw_matches`` is this stage's, holding one entry per (superevent,
    cosmology) -- so two entries per superevent while COSMOLOGIES has two
    members::

        superevent_id  gw_time                    cosmology  sn_dist_mpc  searched_prob_2d  spatial_status
        S190425z       2019-04-25 08:18:05+00:00  Planck18         135.9             0.312  ok
        S190425z       2019-04-25 08:18:05+00:00  SHOES            130.4             0.312  ok

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        The crossmatch stage's output, one row per alert.
    gw_match_input : pandas.DataFrame
        The frame alerts_to_gw_match_input built from ``matches``. Read twice
        over: for its column names, which are how the alert's own values are
        told apart from the GW results, and for the host and distance columns
        themselves, which go onto the surviving rows.
    coincidences : pandas.DataFrame
        Coincident rows from coincident_localizations.

    Returns
    -------
    nested_pandas.NestedFrame
        The alerts that had at least one coincidence, in their original order,
        with their host and distance columns added and the per-(superevent,
        cosmology) results in the nested NESTED_COLUMN. Empty NestedFrame if
        there were no coincidences.
    """
    if coincidences.empty:
        return npd.NestedFrame(matches.iloc[:0])

    carried = [column for column in gw_match_input.columns if column in coincidences.columns]
    nested = coincidences.drop(columns=carried).set_index(coincidences[ALERT_KEY_COLUMN])

    # The host and the distances it puts the alert at belong to the alert rather
    # than to any one superevent, so they go on the row; everything else is
    # per-match. The redshift is renamed back on the way out: it is called
    # ``redshift`` in the transient frame only because that is the name
    # gracedb_tools reads, and on an alert row that would read as the alert's own
    # rather than its host's.
    on_row = gw_match_input.drop(
        columns=[ALERT_KEY_COLUMN, "name", "ra", "declination", "discoverydate"]
    ).rename(columns={"redshift": "host_redshift"})
    with_hosts = npd.NestedFrame(matches.join(on_row, how="left"))
    return with_hosts.join_nested(nested, NESTED_COLUMN, how="inner")


def run_localize(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    inputs: StageInputs | None = None,
    stamp: str | None = None,
) -> StageResult:
    """Run the stage: take the matched alerts, score them against GW skymaps, write the hits.

    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration, normally from config.load_config.
    dry_run : bool
        Do the work but write no results. GraceDB is still queried and the
        coincidences still computed, so a dry run reports the same summary a
        real one would. The GraceDB cache is still written, being a cache of
        what was fetched rather than a result of the run.
    inputs : dict of str to StageResult, optional
        Results of the stages that already ran. The alerts come from
        ``crossmatch``.
    stamp : str, optional
        This run's timestamp, naming the output file. Defaults to now.

    Returns
    -------
    StageResult
        The coincident alerts and where they were written. ``frame`` is ``None``
        when there were no matched alerts to score.

    Raises
    ------
    KeyError
        If ``crossmatch`` has not run.
    ValueError
        If no catalogs are configured to take host redshifts from, or if one of
        those catalogs carries no redshift.
    """
    settings = cfg.localize
    catalog_names = [spec.name for spec in catalog_specs(cfg)]
    stage_dir = cfg.run.stage_dir(STAGE)

    stamp = stamp or run_stamp()
    matches = input_result(inputs, CROSSMATCH_STAGE).frame

    if matches is None or matches.empty:
        logger.info("No matched alerts to localize; writing nothing.")
        return StageResult(stage=STAGE, frame=None, stamp=stamp, summary={"n_alerts": 0})

    hosts = nearest_hosts(matches, catalog_names, min_redshift=settings.min_redshift)
    gw_match_input = alerts_to_gw_match_input(matches, hosts)
    summary: dict[str, Any] = {"n_alerts": len(matches), "n_alerts_with_host": len(gw_match_input)}
    logger.info("%d of %d matched alerts have a usable host redshift.", len(gw_match_input), len(matches))

    events = fetch_gracedb_superevents(
        settings.se_types,
        cache=cfg.gracedb.to_cache(),
        far_threshold_per_year=settings.far_threshold_per_year,
        min_classification_prob_sum=settings.min_classification_prob_sum,
    )
    summary["n_superevents"] = len(events)
    logger.info(
        "GraceDB returned %d %s superevent(s) with a false-alarm rate under %.4g per year.",
        len(events),
        "/".join(settings.se_types),
        settings.far_threshold_per_year,
    )

    coincidences, counts = coincident_localizations(
        gw_match_input,
        events,
        window_days=settings.window_days,
        credible_level=settings.credible_level,
        require_2d_credible_level=settings.require_2d_credible_level,
    )
    summary.update(counts)
    frame = attach_localizations(matches, gw_match_input, coincidences)
    logger.info("Localization summary: %s", summary)

    output_path: Path | None = None
    if dry_run:
        logger.info("Dry run: not writing the coincidences.")
    elif frame.empty:
        logger.info("No alert landed inside a credible volume; writing nothing.")
    else:
        output_path = write_frame(frame, stage_dir / f"{OUTPUT_PREFIX}_{stamp}.parquet")
        logger.info("Wrote %d coincident alerts to %s.", len(frame), output_path)

    return StageResult(stage=STAGE, frame=frame, output_path=output_path, stamp=stamp, summary=summary)
