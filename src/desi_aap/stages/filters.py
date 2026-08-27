"""Which stages are *filters*: the ones that select candidates worth announcing.

A filter reads the placed alerts :mod:`desi_aap.stages.distance` produced and
keeps the ones interesting for one reason -- consistent with a gravitational-wave
event, luminous enough to be a superluminous supernova or a tidal disruption
event, near enough to follow up. They are siblings rather than a chain: each
asks its own question of the same alerts, and one finding nothing says nothing
about the others.

This module is the single list of them. :mod:`desi_aap.pipeline` reads it to
know which stages may find nothing without ending the run, and
:mod:`desi_aap.stages.slack_publish` reads it to know what to announce, so
adding a filter is one import and one entry here rather than an edit in each.

It holds modules rather than names because a filter declares more than its name:
``STAGE``, the runner, and the :class:`~desi_aap.stages.base.SlackDisplay`
saying how its candidates are announced. Keeping them together is what lets both
readers stay generic. A filter module must not import this one -- that is the
cycle this indirection exists to avoid.
"""

from types import ModuleType

from desi_aap.stages import localize

__all__ = ["FILTER_MODULES", "FILTER_STAGES"]

# Every filter, in the order their results are announced.
# TODO(#16): the luminous (SLSN/TDE) and distance-limited filters land here.
FILTER_MODULES: tuple[ModuleType, ...] = (localize,)

# Their stage names, which is what the pipeline and the config deal in.
FILTER_STAGES: tuple[str, ...] = tuple(module.STAGE for module in FILTER_MODULES)
