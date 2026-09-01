"""Load and validate the pipeline configuration from TOML files."""

import tomllib
from collections.abc import Iterable
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from desi_aap.boom import parse_timedelta
from desi_aap.gracedb_cache import GraceDbCache


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or has bad keys."""


def _coerce_duration(value: Any) -> Any:
    """Turn a duration string such as ``"30m"`` or ``"1d12h"`` into a timedelta."""
    if isinstance(value, timedelta):
        return value
    if isinstance(value, str):
        return parse_timedelta(value)
    raise ValueError('expected a duration string such as "30m", "2h", or "1d12h"')


# A timedelta written in the project's own duration format rather than
# pydantic's ISO-8601 one, matching `desi_aap.boom`'s parse_timedelta.
Duration = Annotated[timedelta, BeforeValidator(_coerce_duration)]


class _Section(BaseModel):
    """Base for each TOML section: immutable, and typos are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(_Section):
    """The ``[run]`` section: settings that apply to the whole pipeline."""

    output_dir: Path

    def stage_dir(self, stage: str) -> Path:
        """Directory a given stage writes into, ``output_dir / stage``."""
        return self.output_dir / stage


class BoomConfig(_Section):
    """The ``[query.boom]`` section: which survey to query, and how much."""

    survey: str
    limit: int | None = Field(default=None, ge=1)


class WindowConfig(_Section):
    """The ``[query.window]`` section: which time range to query."""

    start: datetime | float | str | None = None
    end: datetime | float | str | None = None
    lookback: Duration | None = None

    @model_validator(mode="after")
    def _check_window(self) -> "WindowConfig":
        """Require a lookback for any bound not given explicitly."""
        if self.lookback is None and (self.start is None or self.end is None):
            raise ValueError(
                "give 'lookback', or both 'start' and 'end'. A lookback fills in "
                "whichever bound is not given."
            )
        return self


class CrossmatchCatalogConfig(_Section):
    """One ``[crossmatch.catalogs.<name>]`` table: a catalog to match against."""

    catalog: Path
    columns: list[str] | Literal["all"] | None = None
    radius_arcsec: float = Field(gt=0)
    n_neighbors: int = Field(ge=1)


class LocalizeConfig(_Section):
    """The ``[localize]`` section: which superevents to score alerts against, and how."""

    # Required, because these three are what a result means: which events were
    # considered, over what stretch of time, and how tightly. Only credible_level is
    # recorded in the stage's output, so an inherited value for the other two could not
    # be recovered from the results afterwards.
    se_types: list[str]
    window_days: float = Field(gt=0)
    credible_level: float = Field(gt=0, le=1)

    # Defaulted, because these gate detection confidence or guard the arithmetic rather
    # than stating what the search was. This is where the pipeline's defaults live:
    # gracedb_tools carries none of its own, so a caller either names a value or takes
    # it from here.
    far_threshold_per_year: float = Field(default=2.0, gt=0)
    min_classification_prob_sum: float = Field(default=0.9, ge=0, le=1)
    # False because requiring both rankings discards real 3D coincidences, which is what
    # this pipeline is looking for. See select_coincidences for why the two can disagree.
    # TODO: check whether False is the right value here (inherited from notebook, but let's
    # explicitly confirm)
    require_2d_credible_level: bool = False
    min_redshift: float = Field(default=0.0002, gt=0)

    @model_validator(mode="after")
    def _check_se_types(self) -> "LocalizeConfig":
        """Reject a class name p_astro does not carry, which would silently match nothing.

        The names are looked up in the p_astro payload, and a miss reads as a
        probability of zero rather than as an error, so an unrecognized one would
        leave every superevent failing the classification cut and the stage
        reporting an honest-looking zero.
        """
        known = {"BNS", "NSBH", "BBH"}
        unknown = sorted({t for t in self.se_types if t.upper() not in known})
        if not self.se_types or unknown:
            raise ValueError(
                f"se_types must name at least one of {', '.join(sorted(known))}"
                + (f"; got {', '.join(unknown)}" if unknown else "")
            )
        return self


class DaskConfig(_Section):
    """A ``[dask]`` table: arguments passed straight to ``dask.distributed.Client``."""

    kwargs: dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _collect_kwargs(cls, data: Any) -> Any:
        """Fold the table's loose keys into ``kwargs``, since client arguments are open-ended."""
        if not isinstance(data, dict):
            return data
        # Already collected -- a model_copy or an explicit construction.
        if set(data) == {"kwargs"} and isinstance(data["kwargs"], dict):
            return data
        return {"kwargs": data}


class GraceDbConfig(_Section):
    """The ``[gracedb]`` section: where the GraceDB cache lives and how stale it may get."""

    # config.toml sets this and wins; the default is only what a config carrying no [gracedb]
    # section falls back to. Relative -- to pin it, pass to_cache a root.
    cache_dir: Path = Path("gracedb_cache")
    # None means "whatever GraceDbCache defaults to", so the window is written in exactly one
    # place. Spelling the number here as well would let the two drift apart silently.
    recheck_window: Duration | None = None

    def to_cache(self, root: Path | None = None) -> GraceDbCache:
        """Build the cache this section describes.

        Parameters
        ----------
        root : Path, optional
            Directory a relative ``cache_dir`` is resolved against. Without it a relative
            path follows the working directory. Pass the repository root from anywhere that
            does not run from it. An absolute ``cache_dir`` ignores this.

        Returns
        -------
        GraceDbCache
            The cache, with the recheck window left at its own default when this section
            does not name one.
        """
        cache_dir = self.cache_dir
        if root is not None and not cache_dir.is_absolute():
            cache_dir = Path(root) / cache_dir
        if self.recheck_window is None:
            return GraceDbCache(cache_dir=cache_dir)
        return GraceDbCache(cache_dir=cache_dir, recheck_window=self.recheck_window)


class SlackConfig(_Section):
    """The ``[slack]`` section: where each run's results are announced."""

    # Path to a TOML file holding `bot_token = "xoxb-..."`. A path rather than the
    # token itself, so the secret can live outside the repo and outside version
    # control while config.toml stays committed.
    credentials: Path
    channel: str
    # How many candidates the message lists before cutting off. At most 99:
    # Slack's table block holds 100 rows, and the header takes one.
    max_rows: int = Field(default=20, ge=1, le=99)
    # Columns the message's table shows, in this order, skipping any the
    # frame lacks. A nested column, or a `nested.field` path into one, shows
    # the row's sub-rows one per line in a single cell.
    columns: list[str] = ["objectId", "candidate.ra", "candidate.dec"]


class QueryConfig(_Section):
    """The ``[query]`` section: what that stage queries, and over what window."""

    boom: BoomConfig
    window: WindowConfig


class CrossmatchConfig(_Section):
    """The ``[crossmatch]`` section: which catalogs that stage matches against."""

    catalogs: dict[str, CrossmatchCatalogConfig] = {}
    dask: DaskConfig = DaskConfig()

    @model_validator(mode="after")
    def _check_catalog_names(self) -> "CrossmatchConfig":
        """Reject catalog names holding a dot, which would make an unreadable nested column."""
        dotted = sorted(name for name in self.catalogs if "." in name)
        if dotted:
            raise ValueError(
                f"catalog name(s) may not contain a dot: {', '.join(dotted)}. The name "
                "becomes a nested column, addressed as '<name>.<field>'."
            )
        return self


class PipelineConfig(_Section):
    """Top-level configuration, laid out as the TOML files are."""

    run: RunConfig
    query: QueryConfig
    crossmatch: CrossmatchConfig = CrossmatchConfig()
    localize: LocalizeConfig
    dask: DaskConfig = DaskConfig()
    gracedb: GraceDbConfig = GraceDbConfig()
    # Optional, unlike the sections above: a fresh clone has no Slack app or
    # credentials, and every stage still runs without them. The slack_publish
    # stage skips itself, with a log line, when this section is absent.
    slack: SlackConfig | None = None

    def dask_for(self, stage: str) -> dict[str, Any]:
        """Client arguments for one stage: the global ``[dask]``, then its own ``[<stage>.dask]``."""
        overrides = getattr(getattr(self, stage, None), "dask", None)
        return {**self.dask.kwargs, **(overrides.kwargs if overrides else {})}


def _describe(error: dict[str, Any]) -> str:
    """Render one pydantic error as a line naming the setting it is about."""
    location = ".".join(str(part) for part in error["loc"])
    message = error["msg"].removeprefix("Value error, ")
    return f"{location}: {message}" if location else message


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge ``override`` into ``base`` table by table; arrays are replaced whole."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    """Parse one TOML file, reporting problems as :class:`ConfigError`."""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc


def load_config(paths: str | PathLike[str] | Iterable[str | PathLike[str]]) -> PipelineConfig:
    """Load and validate the configuration, layering several files left to right."""
    if isinstance(paths, str | PathLike):
        paths = [paths]
    resolved = [Path(p) for p in paths]
    if not resolved:
        raise ConfigError("No configuration file given.")

    merged: dict[str, Any] = {}
    for path in resolved:
        merged = _deep_merge(merged, _read_toml(path))

    try:
        return PipelineConfig(**merged)
    except ValidationError as exc:
        problems = "\n       ".join(_describe(e) for e in exc.errors())
        raise ConfigError(f"{', '.join(str(p) for p in resolved)}: {problems}") from exc
