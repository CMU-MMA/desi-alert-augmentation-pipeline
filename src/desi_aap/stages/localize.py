"""Filter the alerts down to those consistent with a public GW event localization.

The stage takes the placed-alert frame the distance stage produced and asks, of
every alert in it, whether it could be the optical counterpart of a LIGO/Virgo
superevent: detected close enough in time, and sitting inside the credible
volume of that event's 3D sky map.

The measurements themselves are from gracedb_tools, unchanged. That
module was written against the TNS supernova catalog, so it reads a flat frame
with ``name``, ``ra``, ``declination``, ``discoverydate`` and one
``dist_mpc_<label>`` column per cosmology.COSMOLOGIES entry.
The work this module does on top of it is the adaptation
(alerts_to_gw_match_input): the alert's time comes from its Julian date, and its
distance comes from the DESI host the distance stage already put on the row, so
this filter and its siblings are all cutting on the same number.

The result is written the way the crossmatch stage writes its own: one row per
alert, with the per-superevent results in a nested column
(attach_localizations), so a row that entered the stage as an alert
leaves it as an alert. Only the alerts that landed inside a credible volume are
kept and written; the temporal matches and near misses behind them are counted
in the summary and logged, not persisted.
"""

import logging
from pathlib import Path
from typing import Any

import nested_pandas as npd
import numpy as np
import pandas as pd
from astropy.time import Time

from desi_aap.boom import ALERT_DEC_COLUMN, ALERT_ID_COLUMN, ALERT_RA_COLUMN, ALERT_TIME_COLUMN
from desi_aap.config import PipelineConfig
from desi_aap.gracedb_tools import (
    fetch_gracedb_superevents,
    run_3d_spatial_crossmatch,
    select_coincidences,
    temporal_crossmatch_sesn_to_gw,
)
from desi_aap.stages.base import SlackDisplay, StageInputs, StageResult, input_result, write_frame
from desi_aap.stages.distance import DIST_COLUMNS, HOST_COLUMNS, dist_column
from desi_aap.stages.distance import STAGE as DISTANCE_STAGE
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)


STAGE = "localize"

# Stages whose output this one consumes. See crossmatch.REQUIRES.
REQUIRES: tuple[str, ...] = (DISTANCE_STAGE,)

# Prefix of the parquet file this stage writes.
OUTPUT_PREFIX = "coincidences"

# Nested column the per-superevent results land in, alongside the nested columns
# the crossmatch stage wrote.
NESTED_COLUMN = "gw_matches"

# Flat column naming the superevents an alert matched, comma-separated. The same
# identifiers are in NESTED_COLUMN, once per cosmology; this is the flattened
# copy, so that "which event is this?" can be read off the row -- in the Slack
# message, or from the parquet -- without unnesting.
SUPEREVENT_COLUMN = "superevent_ids"

# How this filter's candidates are announced. The distance is the one the
# coincidence was judged on, so it earns its place beside the event; it is named
# through the distance stage's own helper rather than spelled out, so that
# renaming a cosmology moves the column here too instead of silently dropping it
# from the message.
SLACK_DISPLAY = SlackDisplay(
    title="GW coincidence candidate",
    columns=(SUPEREVENT_COLUMN, "host_redshift", dist_column("SHOES")),
)

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


def alerts_to_gw_match_input(
    alerts: npd.NestedFrame,
    *,
    id_column: str = ALERT_ID_COLUMN,
    ra_column: str = ALERT_RA_COLUMN,
    dec_column: str = ALERT_DEC_COLUMN,
    time_column: str = ALERT_TIME_COLUMN,
) -> pd.DataFrame:
    """Render the placed alerts as the transient table gracedb_tools reads.

    The column names produced below are that module's, not this pipeline's: they
    are what gracedb_tools.temporal_crossmatch_sesn_to_gw and
    gracedb_tools.run_3d_spatial_crossmatch look up, so the adaptation happens
    here rather than by parameterizing them and their callers. The columns read
    are BOOM's, from desi_aap.boom, and the host and distance columns are the
    distance stage's -- this function renames rather than measures.

    ``discoverydate`` is the alert's own timestamp rather than a discovery date
    in the TNS sense -- an alert is one detection of an object, not the first --
    so an object detected repeatedly is offered to the temporal match once per
    alert.

    Parameters
    ----------
    alerts : nested_pandas.NestedFrame
        The distance stage's output: one row per alert, carrying
        :data:`desi_aap.stages.distance.HOST_COLUMNS` and
        :data:`desi_aap.stages.distance.DIST_COLUMNS`.
    id_column, ra_column, dec_column, time_column : str
        Alert columns read for the transient's name, coordinates, and time.

    Returns
    -------
    pandas.DataFrame
        One row per alert with a usable position and time, carrying ``name``,
        ``ra``, ``declination``, ``discoverydate``, ``redshift``, one
        ``dist_mpc_<label>`` per cosmology, the remaining host columns, and
        ALERT_KEY_COLUMN holding the alert's row in ``alerts``. Empty DataFrame
        with those columns if nothing survives.

    Raises
    ------
    KeyError
        If ``alerts`` carries none of the distance stage's columns, which means
        it came from somewhere other than that stage.
    """
    missing = [column for column in (*HOST_COLUMNS, *DIST_COLUMNS) if column not in alerts.columns]
    if missing:
        raise KeyError(
            f"Alert frame is missing {', '.join(repr(c) for c in missing)}, so stage {STAGE!r} "
            f"cannot place it. These columns come from stage {DISTANCE_STAGE!r}, which must run "
            "first; see desi_aap.pipeline.STAGES."
        )

    frame = pd.DataFrame(index=alerts.index)
    frame[ALERT_KEY_COLUMN] = alerts.index
    frame["name"] = alerts[id_column].astype(str)
    frame["ra"] = pd.to_numeric(alerts[ra_column], errors="coerce")
    frame["declination"] = pd.to_numeric(alerts[dec_column], errors="coerce")
    frame["discoverydate"] = julian_dates_to_utc(alerts[time_column])
    # Renamed because ``redshift`` is what gracedb_tools reads. On an alert row
    # that name would read as the alert's own rather than its host's, which is
    # why the distance stage spells it out and this frame does not.
    frame["redshift"] = alerts["host_redshift"]
    for column in (*DIST_COLUMNS, "host_catalog", "host_sep_arcsec"):
        frame[column] = alerts[column]

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
    alerts: npd.NestedFrame,
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

        objectId  host_catalog  host_redshift  dist_mpc_SHOES  superevent_ids  desi_dr1  gw_matches
        LSST001   desi_dr1              0.030           130.4  S190425z        [{...}]   [{...}, {...}]
        LSST002   desi_dr2              0.021            91.2  S190425z        [{...}]   [{...}, {...}]

    ``host_catalog``, ``host_redshift`` and ``dist_mpc_SHOES`` are the distance
    stage's, untouched, as is the ``desi_dr1`` nested column from the crossmatch
    stage. ``gw_matches`` is this stage's, holding one entry per (superevent,
    cosmology) -- so two entries per superevent while COSMOLOGIES has two
    members::

        superevent_id  gw_time                    cosmology  sn_dist_mpc  searched_prob_2d  spatial_status
        S190425z       2019-04-25 08:18:05+00:00  Planck18         135.9             0.312  ok
        S190425z       2019-04-25 08:18:05+00:00  SHOES            130.4             0.312  ok

    Parameters
    ----------
    alerts : nested_pandas.NestedFrame
        The distance stage's output, one row per alert, already carrying its
        host and distance columns.
    gw_match_input : pandas.DataFrame
        The frame alerts_to_gw_match_input built from ``alerts``. Read for its
        column names, which are how the alert's own values are told apart from
        what the GW match added.
    coincidences : pandas.DataFrame
        Coincident rows from coincident_localizations.

    Returns
    -------
    nested_pandas.NestedFrame
        The alerts that had at least one coincidence, in their original order,
        with SUPEREVENT_COLUMN added and the per-(superevent, cosmology) results
        in the nested NESTED_COLUMN. Empty NestedFrame if there were no
        coincidences.
    """
    if coincidences.empty:
        return npd.NestedFrame(alerts.iloc[:0])

    carried = [column for column in gw_match_input.columns if column in coincidences.columns]
    nested = coincidences.drop(columns=carried).set_index(coincidences[ALERT_KEY_COLUMN])

    # One row per (superevent, cosmology), so the same event appears once per
    # cosmology; the set collapses that back to the events themselves. Sorted so
    # that an alert matching two events reads the same way on every run.
    events = (
        coincidences.groupby(ALERT_KEY_COLUMN)["superevent_id"]
        .agg(lambda ids: ", ".join(sorted(set(ids))))
        .rename(SUPEREVENT_COLUMN)
    )
    with_events = npd.NestedFrame(alerts.join(events, how="left"))
    return with_events.join_nested(nested, NESTED_COLUMN, how="inner")


def run_localize(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    inputs: StageInputs | None = None,
    stamp: str | None = None,
) -> StageResult:
    """Run the stage: take the placed alerts, score them against GW skymaps, write the hits.

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
        ``distance``, which chose each one's host and put it at a distance.
    stamp : str, optional
        This run's timestamp, naming the output file. Defaults to now.

    Returns
    -------
    StageResult
        The coincident alerts and where they were written. ``frame`` is ``None``
        when there were no placed alerts to score.

    Raises
    ------
    KeyError
        If ``distance`` has not run, or if its columns are absent from the frame.
    """
    settings = cfg.localize
    stage_dir = cfg.run.stage_dir(STAGE)

    stamp = stamp or run_stamp()
    (upstream,) = REQUIRES
    alerts = input_result(inputs, upstream).frame

    if alerts is None or alerts.empty:
        logger.info("No placed alerts to localize; writing nothing.")
        return StageResult(stage=STAGE, frame=None, stamp=stamp, summary={"n_alerts": 0})

    gw_match_input = alerts_to_gw_match_input(alerts)
    summary: dict[str, Any] = {"n_alerts": len(alerts), "n_alerts_placed": len(gw_match_input)}

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
    frame = attach_localizations(alerts, gw_match_input, coincidences)
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
