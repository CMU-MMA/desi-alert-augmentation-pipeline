import logging
import time
from collections.abc import Callable

from desi_aap.config import PipelineConfig
from desi_aap.stages.base import StageInputs, StageResult
from desi_aap.stages.crossmatch import run_crossmatch
from desi_aap.stages.query import run_query
from desi_aap.stages.slack_publish import run_slack_publish
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)

# Every stage the pipeline knows about, in the order they must run.
# slack_publish announces the run's results, so it stays last; new data
# stages go before it (and desi_aap.stages.slack_publish.INPUT_STAGE moves).
STAGE_ORDER = [
    "query",
    "crossmatch",
    "slack_publish",
]

# Stage name to the function that runs it. Each takes the whole config, the
# per-invocation `dry_run` flag, and the results of the stages that already
# ran; each returns a StageResult.
STAGE_RUNNERS: dict[str, Callable[..., StageResult]] = {
    "query": run_query,
    "crossmatch": run_crossmatch,
    "slack_publish": run_slack_publish,
}


def _elapsed(started: float) -> str:
    """Format the time since ``started`` as ``HH:MM:SS``."""
    hours, remainder = divmod(int(time.perf_counter() - started), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_pipeline(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    stamp: str | None = None,
) -> StageInputs:
    """Run every stage in order, timing and logging each one.

    Parameters
    ----------
    cfg : PipelineConfig
        The configuration for this run.
    dry_run : bool
        Passed to every stage: do the work but write nothing.
    stamp : str, optional
        This run's timestamp, naming every stage's output. The command line
        passes the one it named the log after; defaults to now otherwise.

    Returns
    -------
    dict of str to StageResult
        Each stage's result, keyed by stage name.
    """
    from desi_aap import __version__

    stamp = stamp or run_stamp()

    logger.info("----- DESI Alert Augmentation Pipeline -----")
    logger.info("desi_aap version : %s", __version__)
    logger.info("Output directory : %s", cfg.run.output_dir)
    logger.info("Stages           : %s", ", ".join(STAGE_ORDER))
    logger.info("Run timestamp    : %s", stamp)
    if cfg.crossmatch.catalogs:
        logger.info("Catalogs         : %s", ", ".join(cfg.crossmatch.catalogs))
    if dry_run:
        logger.info("Dry run          : nothing will be written")
    logger.info("")

    results: StageInputs = {}
    total_started = time.perf_counter()

    for stage in STAGE_ORDER:
        stage_started = time.perf_counter()
        logger.info("[%s] starting...", stage)
        result = STAGE_RUNNERS[stage](cfg, dry_run=dry_run, inputs=results, stamp=stamp)
        results[stage] = result
        logger.info("[%s] done in %s\n", stage, _elapsed(stage_started))
        if result.is_empty:
            logger.info("[%s] produced no results. Exiting...", stage)
            break
    logger.info("Pipeline complete. Total time: %s", _elapsed(total_started))
    return results
