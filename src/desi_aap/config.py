"""Load and validate the pipeline configuration from TOML files."""

import tomllib
from collections.abc import Iterable
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from desi_aap.boom import parse_timedelta


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
    lookback: Duration


class CrossmatchCatalogConfig(_Section):
    """One ``[crossmatch.catalogs.<name>]`` table: a catalog to match against."""

    catalog: Path
    columns: list[str] | Literal["all"] | None = None
    radius_arcsec: float = Field(gt=0)
    n_neighbors: int = Field(ge=1)


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


class QueryConfig(_Section):
    """The ``[query]`` section: what that stage queries, and over what window."""

    boom: BoomConfig
    window: WindowConfig
    dask: DaskConfig = DaskConfig()


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
    dask: DaskConfig = DaskConfig()

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
