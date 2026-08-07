"""Defines the result of a stage and how to get it as input to a later stage."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nested_pandas as npd

logger = logging.getLogger(__name__)

__all__ = [
    "StageInputs",
    "StageResult",
    "input_result",
]


@dataclass
class StageResult:
    """What a stage produced.

    Attributes
    ----------
    stage : str
        The stage that produced this, as it appears in
        :data:`desi_aap.pipeline.STAGE_ORDER`.
    frame : nested_pandas.NestedFrame or None
        The table this stage produced. ``None`` when the stage had nothing to
        produce, such as a window that returned no alerts.
    output_path : Path or None
        The parquet file it was written to, or ``None`` on a dry run or when
        there was nothing to write.
    stamp : str or None
        The run timestamp its output is named after. Passed down so every
        stage's output for one run carries the same one.
    summary : dict
        Free-form per-stage counts, for logging.
    """

    stage: str
    frame: npd.NestedFrame | None = None
    output_path: Path | None = None
    stamp: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Whether this stage produced no rows for the next one to work on.

        True both when the stage had nothing to produce (``frame`` is None) and
        when it produced an empty table -- an hour with no alerts, or alerts
        that matched nothing. The run stops when a stage produces an empty input.
        """
        return self.frame is None or self.frame.empty


# Results of the stages that already ran this invocation, keyed by stage name.
StageInputs = dict[str, StageResult]


def input_result(inputs: StageInputs | None, producer: str) -> StageResult:
    """Get the result a stage should consume.

    Parameters
    ----------
    inputs : dict of str to StageResult, or None
        Results of the stages that already ran this invocation.
    producer : str
        The stage whose output is wanted.

    Returns
    -------
    StageResult
        What ``producer`` returned. Its ``frame`` is ``None`` when that stage
        ran but had nothing to produce.

    Raises
    ------
    KeyError
        If ``producer`` did not run. Every run executes the whole pipeline in
        order, so this means the stage order or the registry is wrong rather
        than anything an operator did.
    """
    if not inputs or producer not in inputs:
        raise KeyError(
            f"Stage {producer!r} has not run, so its output is not available. "
            f"Stages run in desi_aap.pipeline.STAGE_ORDER; check that {producer!r} precedes "
            "the stage consuming it."
        )
    result = inputs[producer]
    if result.is_empty:
        logger.info("Stage %r produced no data this run; nothing to consume.", producer)
    return result
