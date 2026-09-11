"""Which stages are *filters*: the ones that select candidates worth announcing.

A filter reads the placed alerts :mod:`desi_aap.stages.distance` produced and
keeps the ones interesting for one reason -- consistent with a gravitational-wave
event, luminous enough to be a superluminous supernova or a tidal disruption
event, near enough to follow up. They are siblings rather than a chain: each
asks its own question of the same alerts, and one finding nothing says nothing
about the others.

Filters come in two kinds, and this module is where they meet:

* **Code filters** measure something -- the GW localization match needs
  GraceDB, skymaps, and a temporal window. Each is a module declaring
  ``STAGE``, ``REQUIRES``, ``SLACK_DISPLAY``, a runner, and a config section
  with an ``enabled`` switch; it goes in :data:`FILTER_MODULES`.
* **Cut filters** only select on columns the placed alerts already carry. Each
  is one JSON file in the configured filters directory -- the API described in
  :mod:`desi_aap.stages.cut_filter` -- and is switched off by naming it in
  ``[filters] disabled``.

:func:`filter_descriptors` renders both kinds down to one shape, so
:mod:`desi_aap.pipeline` (which turns each into a stage) and
:mod:`desi_aap.stages.slack_publish` (which announces each one's candidates)
never learn the difference. A filter module must not import this one -- that is
the cycle this indirection exists to avoid.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from desi_aap.config import ConfigError, PipelineConfig
from desi_aap.stages import cut_filter, localize
from desi_aap.stages.base import SlackDisplay, StageResult

__all__ = ["FILTER_MODULES", "FilterDescriptor", "filter_descriptors"]

# Every code filter, in the order their results are announced. The JSON cut
# filters follow these, sorted by name.
FILTER_MODULES: tuple[ModuleType, ...] = (localize,)


@dataclass(frozen=True)
class FilterDescriptor:
    """One filter, whichever kind, in the shape its consumers read.

    Attributes
    ----------
    stage : str
        The filter's stage name: its key in run results, its output directory,
        and how ``--from-stage`` addresses it.
    requires : tuple of str
        The stages whose frames it consumes, as
        :class:`desi_aap.pipeline.StageSpec` wants them.
    run : callable
        Its runner, with the ``run(cfg, *, dry_run, inputs, stamp)`` signature
        every stage runner has.
    slack_display : desi_aap.stages.base.SlackDisplay
        How its candidates are announced.
    enabled : bool
        Whether this run's config leaves it switched on. Any filter, whichever
        kind, is off when named in ``[filters] disabled``; a code filter is
        also off when its own section sets ``enabled = false``. Resolved here,
        at build time, so downstream reads a fact rather than a lookup rule.
    """

    stage: str
    requires: tuple[str, ...]
    run: Callable[..., StageResult]
    slack_display: SlackDisplay
    enabled: bool = True


def filter_descriptors(cfg: PipelineConfig) -> tuple[FilterDescriptor, ...]:
    """Every filter this run knows about, code and JSON alike.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        Read for the filters directory, the disabled list, and each code
        filter's ``enabled`` switch.

    Returns
    -------
    tuple of FilterDescriptor
        The code filters in :data:`FILTER_MODULES` order, then the JSON filters
        sorted by name. Disabled filters are included, marked, rather than
        dropped: the pipeline records *why* a stage did not run, and it cannot
        record what it never saw.

    Raises
    ------
    desi_aap.config.ConfigError
        If a filter file is malformed, takes a reserved name, or collides with
        another filter -- see :func:`desi_aap.stages.cut_filter.load_cut_filters`.
    """
    loaded_filters = cut_filter.load_cut_filters(cfg)
    disabled = set(cfg.filters.disabled)
    descriptors = [
        FilterDescriptor(
            stage=module.STAGE,
            requires=module.REQUIRES,
            run=getattr(module, f"run_{module.STAGE}"),
            slack_display=module.SLACK_DISPLAY,
            # Off if either switch says so: the module's own section, or the
            # shared disabled list. The list accepting code filters too means
            # one knob turns any filter off, whichever kind it is.
            enabled=bool(getattr(getattr(cfg, module.STAGE, None), "enabled", True))
            and module.STAGE not in disabled,
        )
        for module in FILTER_MODULES
    ]
    known = {d.stage for d in descriptors} | {loaded.name for loaded in loaded_filters}
    unknown = sorted(disabled - known)
    if unknown:
        raise ConfigError(
            f"[filters] disabled names {', '.join(repr(u) for u in unknown)}, which match no "
            f"filter. This run has: {', '.join(sorted(known))}. A typo here would leave the "
            "filter running, so it is an error rather than a warning."
        )
    descriptors += [
        FilterDescriptor(
            stage=loaded.name,
            requires=cut_filter.REQUIRES,
            run=cut_filter.runner(loaded),
            slack_display=loaded.slack_display,
            enabled=loaded.name not in disabled,
        )
        for loaded in loaded_filters
    ]
    return tuple(descriptors)
