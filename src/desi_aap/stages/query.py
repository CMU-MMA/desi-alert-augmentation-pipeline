"""Query BOOM for recent alerts.

This stage resolves the time window, asks BOOM for the alerts in it, and writes
what comes back to parquet in the stage's output directory. The frame is also
handed to the next stage in memory, so a full run does not write and
immediately re-read it.
"""

import logging
from datetime import timedelta
from pathlib import Path

import nested_pandas as npd
import pandas as pd
import pyarrow as pa
from astropy.time import Time

from desi_aap import boom
from desi_aap.config import PipelineConfig
from desi_aap.stages.base import StageInputs, StageResult, write_frame
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)

STAGE = "query"

# Stages whose output this one consumes; the pipeline reads it to decide
# whether there is anything for this stage to do. Empty: the alerts come
# from the broker rather than from another stage.
REQUIRES: tuple[str, ...] = ()

# Prefix of the parquet file this stage writes.
OUTPUT_PREFIX = "alerts"


def resolve_window(
    start: boom.TimeLike | None,
    end: boom.TimeLike | None,
    *,
    lookback: timedelta | None,
) -> tuple[float, float]:
    """Decide which Julian-date window this run should cover.

    Explicit bounds always win. Otherwise, the window ends *now* and starts
    ``lookback`` before it.

    Parameters
    ----------
    start, end : time-like, optional
        Explicit bounds (astropy ``Time``, ``datetime``, ISO-8601 string, or
        raw Julian date).
    lookback : timedelta
        Width of the trailing window. Required rather than defaulted, so
        ``config.toml`` stays the only place the pipeline's value is written.

    Returns
    -------
    tuple of float
        ``(start_jd, end_jd)``.

    Raises
    ------
    ValueError
        If the resolved start is after the resolved end.
    """
    if lookback is None and (start is None or end is None):
        raise ValueError("a lookback is required to fill in a missing start or end.")

    end_jd = boom._to_jd(end) if end is not None else float(Time.now().jd)
    start_jd = boom._to_jd(start) if start is not None else end_jd - lookback.total_seconds() / 86400.0

    if start_jd > end_jd:
        raise ValueError(f"start ({start_jd}) must not be after end ({end_jd}).")
    return start_jd, end_jd


def query_surveys(
    cfg: PipelineConfig, start_jd: float, end_jd: float
) -> tuple[npd.NestedFrame, dict[str, int]]:
    """Query every configured survey over one window and combine the alerts.

    Each survey is asked with its own pipeline and permissions, its columns are
    renamed as configured, and every row is tagged with the survey it came from
    (:data:`desi_aap.boom.ALERT_SURVEY_COLUMN`). The frames are then stacked;
    a column one survey lacks is missing (NaN, or no sub-rows) on its rows.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        Read for ``[query.boom]``.
    start_jd, end_jd : float
        The window, as :func:`resolve_window` returns it.

    Returns
    -------
    nested_pandas.NestedFrame
        Every survey's alerts, indexed from zero. Empty when none returned any.
    dict of str to int
        How many alerts each survey returned, in query order.
    """
    settings = cfg.query.boom
    frames = []
    counts: dict[str, int] = {}
    for survey in settings.survey_list:
        pipeline_path = settings.pipelines.get(survey)
        kwargs = {"start": start_jd, "end": end_jd, "survey": survey, "limit": settings.limit}
        if pipeline_path is not None:
            kwargs["pipeline"] = boom.load_pipeline(pipeline_path)
        if survey in settings.permissions:
            kwargs["permissions"] = {survey: list(settings.permissions[survey])}
        frame = boom.query_alerts(**kwargs)
        counts[survey] = len(frame)
        logger.info(
            "BOOM returned %d %s alerts%s.",
            len(frame),
            survey,
            f" ({pipeline_path})" if pipeline_path else "",
        )
        if frame.empty:
            continue
        renames = settings.renames.get(survey, {})
        clashes = sorted(new for old, new in renames.items() if old in frame.columns and new in frame.columns)
        if clashes:
            raise ValueError(
                f"[query.boom.renames.{survey}] would overwrite column(s) {', '.join(clashes)} that "
                f"BOOM already returned for {survey}; drop them from the pipeline's projection or "
                "rename to something else."
            )
        frame = frame.rename(columns=renames)
        # Arrow-backed, as the other string columns are, so the frame equals
        # its own parquet round trip.
        frame[boom.ALERT_SURVEY_COLUMN] = pd.Series(
            survey, index=frame.index, dtype=pd.ArrowDtype(pa.large_string())
        )
        frames.append(frame)

    if not frames:
        return npd.NestedFrame(), counts
    if len(frames) == 1:
        return frames[0], counts
    return npd.NestedFrame(pd.concat(frames, ignore_index=True)), counts


def run_query(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    inputs: StageInputs | None = None,
    stamp: str | None = None,
) -> StageResult:
    """Run the stage: resolve the window, query BOOM, write an alert catalog.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        The pipeline configuration.
    dry_run : bool
        Do the work but write nothing. The frame is still built and passed on,
        so a dry run exercises the whole chain in memory.
    inputs : dict of str to StageResult, optional
        Unused -- this stage is the head of the pipeline. Accepted so every
        runner has the same signature.
    stamp : str, optional
        This run's timestamp, naming the output file. Defaults to now.

    Returns
    -------
    StageResult
        The alert frame and where it was written. ``frame`` is ``None`` when
        the window held no usable alerts.
    """
    stamp = stamp or run_stamp()
    window = cfg.query.window
    stage_dir = cfg.run.stage_dir(STAGE)

    start_jd, end_jd = resolve_window(window.start, window.end, lookback=window.lookback)
    logger.info(
        "Querying BOOM for %s alerts in JD [%.6f, %.6f] (%s to %s UTC).",
        ", ".join(cfg.query.boom.survey_list),
        start_jd,
        end_jd,
        Time(start_jd, format="jd").isot,
        Time(end_jd, format="jd").isot,
    )

    alerts, per_survey = query_surveys(cfg, start_jd, end_jd)
    logger.info("BOOM returned %d alerts.", len(alerts))

    summary = {"n_alerts": len(alerts), "start_jd": start_jd, "end_jd": end_jd}
    if len(per_survey) > 1:
        summary["n_alerts_by_survey"] = per_survey

    if alerts.empty:
        logger.info("No alerts in this window; writing nothing.")
        return StageResult(stage=STAGE, frame=None, output_path=None, stamp=stamp, summary=summary)

    output_path: Path | None = None
    if dry_run:
        logger.info("Dry run: not writing the alerts.")
    else:
        output_path = write_frame(alerts, stage_dir / f"{OUTPUT_PREFIX}_{stamp}.parquet")
        logger.info("Wrote %d alerts to %s.", len(alerts), output_path)

    return StageResult(stage=STAGE, frame=alerts, output_path=output_path, stamp=stamp, summary=summary)
