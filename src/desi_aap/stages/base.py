"""Defines the result of a stage and how to get it as input to a later stage."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nested_pandas as npd

logger = logging.getLogger(__name__)

__all__ = [
    "SlackDisplay",
    "StageInputs",
    "StageResult",
    "input_result",
    "write_frame",
]


@dataclass(frozen=True)
class SlackDisplay:
    """How one filter stage's candidates are announced.

    A filter decides what its own results are called and which of its columns
    are worth reading in a chat client, so each filter module declares one of
    these and :mod:`desi_aap.stages.slack_publish` stays generic.

    Attributes
    ----------
    title : str
        Names the candidates in the message header, as in "3 GW coincidence
        candidates". A noun phrase, lowercase except for proper nouns, and
        written so it reads with a count in front of it.
    columns : tuple of str
        Flat columns to show after the ones every filter shows
        (:data:`desi_aap.stages.slack_publish.DISPLAY_COLUMNS`), in order. A
        column the frame lacks is skipped rather than raising, so a filter may
        name one that only some runs produce.
    """

    title: str
    columns: tuple[str, ...] = ()


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
        that matched nothing. A stage whose every input is empty is skipped
        rather than run; see :func:`desi_aap.pipeline.run_pipeline`. Skipping
        rather than stopping is what lets one filter find nothing without
        silencing its siblings.
        """
        return self.frame is None or self.frame.empty


# Results of the stages that already ran this invocation, keyed by stage name.
StageInputs = dict[str, StageResult]


def write_frame(frame: npd.NestedFrame, path: Path) -> Path:
    """Write a stage's frame to parquet, preserving its Arrow dtypes.

    Read one back with :func:`nested_pandas.read_parquet`.

    Parameters
    ----------
    frame : nested_pandas.NestedFrame
        The table to write.
    path : Path
        Destination file. Parent directories are created.

    Returns
    -------
    Path
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


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
