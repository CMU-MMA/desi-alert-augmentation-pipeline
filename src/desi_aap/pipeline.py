"""Run the stages in order, and decide which of them have anything to do.

The pipeline is not a straight line. It narrows to one point -- the alerts,
cross-matched against DESI and put at their hosts' distances -- and then fans
out into *filters*, each asking a different question of those same alerts::

    query -> crossmatch -> distance -+-> localize -----+
                                     |                 |
                                     +-> (luminous) ---+-> slack_publish
                                     |                 |
                                     +-> (nearby) -----+

So a stage that produces nothing *skips* what depends on it rather than ending
the run. Most hours have no GW superevent within ``window_days``, and a run
where ``localize`` finds nothing must still let the other filters report. The
rule is in :func:`_has_input`: a stage runs unless every stage it requires
produced an empty frame.

That also subsumes the older stop-on-empty behaviour, which existed to avoid
cross-matching an empty hour. An empty ``query`` still leaves ``crossmatch``
with nothing to do, so it is skipped, so ``distance`` is, and so on down.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from types import ModuleType

from desi_aap.config import PipelineConfig
from desi_aap.stages import crossmatch, distance, query, slack_publish
from desi_aap.stages.base import StageInputs, StageResult
from desi_aap.stages.filters import filter_descriptors
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageSpec:
    """One stage, and what it needs before it can run.

    Attributes
    ----------
    name : str
        The stage's name, as its module's ``STAGE`` spells it. For the stages
        configured under a section of the same name it is also that section's,
        which is how ``enabled`` and the per-stage Dask settings are found.
        ``slack_publish`` is the exception: it reads ``[slack]``, so nothing
        looks its settings up by stage name.
    run : callable
        Its runner, taking the whole config, ``dry_run``, ``inputs`` and
        ``stamp``, and returning a StageResult.
    requires : tuple of str
        The stages whose frames it consumes. Empty for the first stage. The
        stage is skipped when *every* one of these produced no rows -- so for
        the single-input stages that is "skip when my input is empty", and for
        slack_publish, which consumes every filter, it is "skip only when they
        all found nothing".
    """

    name: str
    run: Callable[..., StageResult]
    requires: tuple[str, ...] = field(default=())
    # Whether this run's config leaves the stage switched on. Resolved when the
    # spec is built (see desi_aap.stages.filters.FilterDescriptor.enabled for
    # who decides), so by here it is a plain fact rather than a lookup rule.
    enabled: bool = True


def _spec(module: ModuleType, run: Callable[..., StageResult]) -> StageSpec:
    """Build a data stage's entry from what the stage module itself declares.

    The name and the dependencies are read off the module rather than restated
    here, because the module is where they are already used: ``STAGE`` names its
    output directory and its config section, and ``REQUIRES`` holds the names it
    passes to :func:`~desi_aap.stages.base.input_result`. Restating either in
    this registry would let the two drift -- a stage could read one upstream
    while the pipeline decided whether to skip it based on another.
    """
    return StageSpec(module.STAGE, run, requires=module.REQUIRES)


# The stages every run has, in order, up to the fan-out. The filters come from
# the config -- see stages_for -- so they cannot be listed here.
DATA_STAGES: tuple[StageSpec, ...] = (
    _spec(query, query.run_query),
    _spec(crossmatch, crossmatch.run_crossmatch),
    _spec(distance, distance.run_distance),
)


def stages_for(cfg: PipelineConfig) -> tuple[StageSpec, ...]:
    """Every stage of this run, in the order they must run.

    A function of the config rather than a module constant, because the filters
    are: each JSON file in the configured filters directory is a stage of its
    own, so what the pipeline *is* cannot be known before the config is read.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        The configuration whose pipeline is wanted.

    Returns
    -------
    tuple of StageSpec
        The data stages, then every filter (code and JSON alike, disabled ones
        included so their skip is recorded), then ``slack_publish``, which
        requires them all.

    Raises
    ------
    desi_aap.config.ConfigError
        If a filter file is malformed or misnamed; see
        :func:`desi_aap.stages.cut_filter.load_cut_filters`.
    """
    filters = filter_descriptors(cfg)
    filter_specs = tuple(StageSpec(d.stage, d.run, requires=d.requires, enabled=d.enabled) for d in filters)
    # The descriptors are bound into the announcing stage rather than re-read
    # when it runs, so the filters a run announces are exactly the filters it
    # was built from: a JSON file edited or deleted mid-run changes the next
    # run, never the tail of this one.
    announce = StageSpec(
        slack_publish.STAGE,
        partial(slack_publish.run_slack_publish, descriptors=filters),
        requires=tuple(d.stage for d in filters),
    )
    return (*DATA_STAGES, *filter_specs, announce)


def stage_order(cfg: PipelineConfig) -> list[str]:
    """The stage names of :func:`stages_for`, in order."""
    return [spec.name for spec in stages_for(cfg)]


def _has_input(spec: StageSpec, results: StageInputs) -> bool:
    """Whether any stage this one requires produced rows for it.

    Parameters
    ----------
    spec : StageSpec
        The stage about to run.
    results : dict of str to StageResult
        What has been produced so far, including any stand-ins supplied by the
        caller.

    Returns
    -------
    bool
        True when ``spec`` requires nothing, when none of what it requires has a
        result yet, or when at least one of those results has rows. The middle
        case matters for ``--from-stage``: a required stage with no result at
        all is not an empty one, so the stage is run rather than quietly
        skipped. A stage that cannot work without its input then says so
        itself, through :func:`~desi_aap.stages.base.input_result`;
        ``slack_publish``, which tolerates a filter that did not run, instead
        posts nothing and says why.
    """
    known = [results[name] for name in spec.requires if name in results]
    return not known or any(not result.is_empty for result in known)


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
    start: str | None = None,
    inputs: StageInputs | None = None,
) -> StageInputs:
    """Run every stage in order, timing and logging each one.

    A stage is skipped, not run, when its config switches it off or when every
    stage it requires produced no rows. A skipped stage still gets a
    StageResult, with an empty frame and a ``skipped`` summary, so that whatever
    depends on it sees an empty input and skips in turn.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        The configuration for this run.
    dry_run : bool
        Passed to every stage: do the work but write nothing.
    stamp : str, optional
        This run's timestamp, naming every stage's output. The command line
        passes the one it named the log after; defaults to now otherwise.
    start : str, optional
        First stage to run; the ones before it are skipped rather than run.
        Whatever the running stages consume must then come in via ``inputs``.
    inputs : dict of str to StageResult, optional
        Stand-ins for the results of stages that are not run this invocation,
        keyed like the stages would have been.

    Returns
    -------
    dict of str to StageResult
        Each stage's result, keyed by stage name. The stand-ins from
        ``inputs`` are included, so the shape matches a full run.

    Raises
    ------
    ValueError
        If ``start`` is not a stage this pipeline knows about.
    """
    from desi_aap import __version__

    stamp = stamp or run_stamp()
    stages = stages_for(cfg)
    order = [spec.name for spec in stages]
    if start is not None and start not in order:
        raise ValueError(f"Unknown stage {start!r}. Stages, in order: {', '.join(order)}.")
    specs = stages if start is None else stages[order.index(start) :]
    # The filters are the stages between the shared data stages and the
    # announcement -- read off the list just built, not the disk again.
    data = {spec.name for spec in DATA_STAGES}
    filters = [name for name in order if name not in data and name != slack_publish.STAGE]

    logger.info("----- DESI Alert Augmentation Pipeline -----")
    logger.info("desi_aap version : %s", __version__)
    logger.info("Output directory : %s", cfg.run.output_dir)
    logger.info("Stages           : %s", ", ".join(spec.name for spec in specs))
    logger.info("Filters          : %s", ", ".join(filters) if filters else "(none configured)")
    logger.info("Run timestamp    : %s", stamp)
    if start is not None:
        logger.info("Starting from    : %s, earlier stages fed from supplied inputs", start)
    if cfg.crossmatch.catalogs:
        logger.info("Catalogs         : %s", ", ".join(cfg.crossmatch.catalogs))
    if dry_run:
        logger.info("Dry run          : nothing will be written")
    logger.info("")

    results: StageInputs = dict(inputs) if inputs else {}
    total_started = time.perf_counter()

    for spec in specs:
        if not spec.enabled:
            logger.info("[%s] switched off in the config; skipping.\n", spec.name)
            results[spec.name] = StageResult(stage=spec.name, stamp=stamp, summary={"skipped": "disabled"})
            continue
        if not _has_input(spec, results):
            logger.info(
                "[%s] every stage it needs (%s) produced no rows; skipping.\n",
                spec.name,
                ", ".join(spec.requires),
            )
            results[spec.name] = StageResult(stage=spec.name, stamp=stamp, summary={"skipped": "no input"})
            continue

        stage_started = time.perf_counter()
        logger.info("[%s] starting...", spec.name)
        results[spec.name] = spec.run(cfg, dry_run=dry_run, inputs=results, stamp=stamp)
        logger.info("[%s] done in %s\n", spec.name, _elapsed(stage_started))

    logger.info("Pipeline complete. Total time: %s", _elapsed(total_started))
    return results
