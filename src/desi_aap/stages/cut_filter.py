"""Filters defined as JSON files: one file in the filters directory, one stage.

This is the filter API. A filter that is nothing but cuts on the placed-alert
columns -- a brightness threshold, a distance limit, a band selection -- should
not cost a Python module, a config class, and a registry edit. It costs one
JSON file::

    filters/
    |-- luminous.json      -> stage "luminous",  output/luminous/candidates_<stamp>.parquet
    `-- nearby.json        -> stage "nearby",    output/nearby/candidates_<stamp>.parquet

Drop a file in, and the pipeline picks it up on the next run: the filename is
the filter's name, the filter becomes its own stage reading the distance
stage's output, and its candidates get their own parquet file and their own
Slack message. Nothing else changes anywhere.

The file's grammar, all of it::

    {
      "title": "luminous transient candidate",
      "description": "SLSN/TDE screen: brighter than -20 under the SHOES cosmology.",
      "columns": ["abs_mag_SHOES", "host_redshift"],
      "cuts": [
        {"column": "abs_mag_SHOES", "max": -20.0}
      ]
    }

``cuts`` is the filter: every cut must pass for an alert to be a candidate.
Each names a ``column`` of the distance stage's output and at least one of

``min``, ``max``
    Inclusive numeric bounds. A value that is missing or not a number fails.
``one_of``
    Values the column must be among -- for the strings, such as a band.

``title`` names the candidates in the Slack message, reading with a count in
front of it ("3 luminous transient candidates found"); it defaults to
"<name> candidate". ``columns`` are shown in the message after the ones every
filter shows; ``description`` is for the person reading the JSON and is
otherwise unused. Unknown keys are rejected, so a typo fails the run loudly
rather than silently weakening a filter.

What this API cannot say is a filter that *measures* something -- the GW
localization match needs GraceDB, skymaps, and a temporal window, so it is a
Python module (:mod:`desi_aap.stages.localize`). Both kinds meet in
:func:`desi_aap.stages.filters.filter_descriptors`, and downstream of that
nothing knows the difference.

Cadence: JSON files do not merge the way ``--config`` overlays do, so which
filters a scheduled run skips is written in TOML -- ``[filters] disabled``
names them. See "Running filters on different cadences" in the README.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import nested_pandas as npd
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from desi_aap.config import ConfigError, PipelineConfig
from desi_aap.stages.base import SlackDisplay, StageInputs, StageResult, input_result, write_frame
from desi_aap.stages.distance import STAGE as DISTANCE_STAGE
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)

# Stages whose output every cut filter consumes.
REQUIRES: tuple[str, ...] = (DISTANCE_STAGE,)

# Prefix of the parquet file a cut filter writes, under its own name's directory.
OUTPUT_PREFIX = "candidates"

# Names a filter file may not take: the stages that already exist, and the
# config sections that are not stages. A filter's name becomes a stage name,
# an output directory, and a key in run summaries, so a collision with any of
# these would make two different things answer to one name.
RESERVED_NAMES = frozenset(
    {"query", "crossmatch", "distance", "localize", "slack_publish"}
    | {"run", "dask", "gracedb", "slack", "filters", "logs"}
)


class CutSpec(BaseModel):
    """One cut: a column and the values of it that pass.

    Unknown keys are rejected so that a misspelled bound reads as an error
    rather than as a cut that silently stopped cutting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    # Inclusive, both of them: "min": -20 keeps -20 itself. Inclusive because
    # thresholds in the science are quoted that way ("brighter than -20"
    # includes -20) and one convention for both bounds is one thing to remember.
    min: float | None = None
    max: float | None = None
    # For columns that are labels rather than measures, such as the band.
    one_of: list[str | float | int] | None = None

    @model_validator(mode="after")
    def _check_has_a_bound(self) -> "CutSpec":
        """A cut with no bound would pass everything, which is never meant."""
        if self.min is None and self.max is None and self.one_of is None:
            raise ValueError(f"cut on {self.column!r} needs at least one of min, max, one_of")
        if self.one_of is not None and not self.one_of:
            raise ValueError(f"cut on {self.column!r} has an empty one_of, which would pass nothing")
        return self

    def describe(self) -> str:
        """The cut as one readable clause, for summaries and logs."""
        clauses = []
        if self.min is not None:
            clauses.append(f"{self.min} <= {self.column}")
        if self.max is not None:
            clauses.append(f"{self.column} <= {self.max}")
        if self.one_of is not None:
            clauses.append(f"{self.column} in {list(self.one_of)}")
        return " and ".join(clauses)

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        """Which rows of ``frame`` pass this cut.

        A value that is missing, NaN, or (for the numeric bounds) not a number
        fails: a cut is a positive statement about a value, and an absent value
        cannot make one.

        Raises
        ------
        ValueError
            If the column is not on the frame, naming the columns that are.
            An error rather than an all-fail mask, because a filter cutting on
            a column that never exists is a broken filter, not a quiet one.
        """
        if self.column not in frame.columns:
            raise ValueError(
                f"cut column {self.column!r} is not in the placed-alert frame. It has: "
                f"{', '.join(sorted(map(str, frame.columns)))}. Columns come from the "
                f"{DISTANCE_STAGE!r} stage's output."
            )
        series = frame[self.column]
        keep = pd.Series(True, index=frame.index)
        if self.min is not None or self.max is not None:
            values = pd.to_numeric(series, errors="coerce")
            if self.min is not None:
                keep &= values >= self.min  # NaN compares False, so it fails.
            if self.max is not None:
                keep &= values <= self.max
        if self.one_of is not None:
            keep &= series.isin(self.one_of)
        return keep


class CutFilterSpec(BaseModel):
    """One filter file's content, as JSON gives it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cuts: list[CutSpec]
    title: str | None = None
    columns: list[str] = []
    description: str | None = None

    @model_validator(mode="after")
    def _check_has_cuts(self) -> "CutFilterSpec":
        """No cuts would keep every alert, which is a mailing list, not a filter."""
        if not self.cuts:
            raise ValueError("a filter needs at least one entry in 'cuts'")
        return self


@dataclass(frozen=True)
class CutFilter:
    """One loaded filter: its name, its cuts, and where it came from.

    Attributes
    ----------
    name : str
        The filename's stem: the stage's name, its output directory, and its
        key in ``[filters] disabled``.
    spec : CutFilterSpec
        The validated file content.
    path : Path
        The file itself, named in errors and carried into the stage summary so
        a result can be traced to the definition that produced it.
    """

    name: str
    spec: CutFilterSpec
    path: Path

    @property
    def slack_display(self) -> SlackDisplay:
        """How this filter's candidates are announced."""
        return SlackDisplay(
            title=self.spec.title or f"{self.name} candidate",
            columns=tuple(self.spec.columns),
        )


def load_cut_filters(cfg: PipelineConfig) -> tuple[CutFilter, ...]:
    """Read every ``*.json`` in the configured filters directory.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        Read for ``[filters] dir``. A directory that does not exist contributes
        no filters when it is the default -- a clone without the shipped
        definitions still runs -- but is an error when the config named it
        explicitly, because a typo there would otherwise read as "no filters"
        and the pipeline would quietly stop announcing anything.

    Returns
    -------
    tuple of CutFilter
        One per file, sorted by name so that runs, logs, and Slack messages
        order the same way on every machine.

    Raises
    ------
    desi_aap.config.ConfigError
        If a file is not valid JSON, does not satisfy the grammar, or takes a
        reserved name. The error names the file, since "one bad filter" should
        read as exactly that.
    """
    directory = cfg.filters.dir
    if not directory.is_dir():
        if "dir" in cfg.filters.model_fields_set:
            raise ConfigError(
                f"[filters] dir = {str(directory)!r} does not exist. It was set explicitly, so "
                "a missing directory is more likely a typo or a wrong working directory than a "
                "decision to run with no filters. A relative path follows the working directory "
                "-- run from the checkout, or give an absolute path; see 'Scheduled runs' in "
                "the README."
            )
        logger.info("No filters directory at %s; no JSON filters this run.", directory)
        return ()

    filters = []
    for path in sorted(directory.glob("*.json")):
        name = path.stem
        if name in RESERVED_NAMES:
            raise ConfigError(
                f"Filter file {path} takes the reserved name {name!r}. The filename becomes "
                "a stage name and an output directory, so it may not shadow an existing "
                f"stage or config section: {', '.join(sorted(RESERVED_NAMES))}."
            )
        if "." in name:
            raise ConfigError(f"Filter file {path} has a dot in its name, which stage names may not.")
        try:
            spec = CutFilterSpec.model_validate(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Filter file {path} is not valid JSON: {exc}") from exc
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: "
                f"{error['msg'].removeprefix('Value error, ')}"
                for error in exc.errors()
            )
            raise ConfigError(f"Filter file {path}: {problems}") from exc
        filters.append(CutFilter(name=name, spec=spec, path=path))
    return tuple(filters)


def apply_cuts(frame: npd.NestedFrame, cuts: list[CutSpec]) -> tuple[pd.Series, dict[str, int]]:
    """Apply every cut, keeping the rows that pass all of them.

    Parameters
    ----------
    frame : nested_pandas.NestedFrame
        The distance stage's output, one row per placed alert.
    cuts : list of CutSpec
        The filter's cuts, ANDed.

    Returns
    -------
    keep : pandas.Series
        Boolean mask over ``frame``: the candidates.
    survivors : dict of str to int
        How many alerts pass each cut *on its own*, keyed by the cut's
        description. Per-cut rather than cumulative, so a filter that finds
        nothing says which cut was the one nothing passed.

    Raises
    ------
    ValueError
        If a cut names a column the frame does not have.
    """
    keep = pd.Series(True, index=frame.index)
    survivors: dict[str, int] = {}
    for cut in cuts:
        mask = cut.mask(frame)
        survivors[cut.describe()] = int(mask.sum())
        keep &= mask
    return keep, survivors


def run_cut_filter(
    cfg: PipelineConfig,
    *,
    cut_filter: CutFilter,
    dry_run: bool = False,
    inputs: StageInputs | None = None,
    stamp: str | None = None,
) -> StageResult:
    """Run one JSON filter: cut the placed alerts down to its candidates, write them.

    Bind ``cut_filter`` with :func:`runner` to get the signature every stage
    runner has.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        The pipeline configuration.
    cut_filter : CutFilter
        The filter to run, from :func:`load_cut_filters`.
    dry_run : bool
        Do the work but write nothing.
    inputs : dict of str to StageResult, optional
        Results of the stages that already ran. The alerts come from
        ``distance``.
    stamp : str, optional
        This run's timestamp, naming the output file. Defaults to now.

    Returns
    -------
    StageResult
        The candidates and where they were written, under the *filter's* name.
        ``frame`` is ``None`` when there were no placed alerts to cut.

    Raises
    ------
    KeyError
        If ``distance`` has not run.
    ValueError
        If a cut names a column the placed alerts do not carry.
    """
    stamp = stamp or run_stamp()
    (upstream,) = REQUIRES
    alerts = input_result(inputs, upstream).frame

    if alerts is None or alerts.empty:
        logger.info("No placed alerts for filter %r; writing nothing.", cut_filter.name)
        return StageResult(stage=cut_filter.name, frame=None, stamp=stamp, summary={"n_alerts": 0})

    keep, survivors = apply_cuts(alerts, cut_filter.spec.cuts)
    frame = npd.NestedFrame(alerts[keep])
    summary: dict[str, Any] = {
        "n_alerts": len(alerts),
        "n_candidates": len(frame),
        "survivors_by_cut": survivors,
        "definition": str(cut_filter.path),
    }
    logger.info("Filter %r summary: %s", cut_filter.name, summary)

    output_path: Path | None = None
    if dry_run:
        logger.info("Dry run: not writing the candidates.")
    elif frame.empty:
        logger.info("No alert passed every cut of %r; writing nothing.", cut_filter.name)
    else:
        stage_dir = cfg.run.stage_dir(cut_filter.name)
        output_path = write_frame(frame, stage_dir / f"{OUTPUT_PREFIX}_{stamp}.parquet")
        logger.info("Wrote %d candidates to %s.", len(frame), output_path)

    return StageResult(
        stage=cut_filter.name, frame=frame, output_path=output_path, stamp=stamp, summary=summary
    )


def runner(cut_filter: CutFilter) -> Callable[..., StageResult]:
    """Bind one filter into the ``run(cfg, *, dry_run, inputs, stamp)`` shape stages have."""
    return partial(run_cut_filter, cut_filter=cut_filter)
