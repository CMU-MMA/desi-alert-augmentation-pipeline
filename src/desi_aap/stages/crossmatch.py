"""Cross-match the alerts against the configured catalogs.

The stage takes the alert frame :mod:`desi_aap.stages.query` produced,
wraps it in an in-memory LSDB catalog (:func:`alerts_to_catalog`), cross-matches
that against every catalog in the ``[crossmatch.catalogs.*]`` tables
(:func:`crossmatch_catalog`), and computes the result back into a
:class:`nested_pandas.NestedFrame`.

The results use LSDB :meth:`~lsdb.Catalog.crossmatch_nested`, so each catalog's
matches land in their own *nested column*. That keeps one row per alert however
many catalogs are configured. Only the alerts that matched at least one catalog
are kept and written (:func:`filter_matched`).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lsdb
import nested_pandas as npd
import pandas as pd

from desi_aap.config import PipelineConfig
from desi_aap.stages.base import StageInputs, StageResult, input_result, write_frame
from desi_aap.stages.query import STAGE as QUERY_STAGE
from desi_aap.utils import run_stamp
from desi_aap.utils.dask_client import dask_client

logger = logging.getLogger(__name__)

STAGE = "crossmatch"

# Prefix of the parquet file this stage writes.
OUTPUT_PREFIX = "matches"

# Coordinate columns produced by desi_aap.boom's default pipeline.
ALERT_RA_COLUMN = "candidate.ra"
ALERT_DEC_COLUMN = "candidate.dec"


@dataclass(frozen=True)
class CatalogSpec:
    """One catalog to cross-match the alerts against.

    Attributes
    ----------
    name : str
        Name of the nested column the matches are written to. Must not contain
        a dot, since nested-pandas addresses sub-columns as ``"<name>.<field>"``.
    catalog : lsdb.Catalog, str, or Path
        An already-open catalog, or the path to a HATS collection.
    radius_arcsec : float
        Match radius. A radius wider than the catalog's margin can miss matches
        at partition boundaries.
    n_neighbors : int
        How many neighbors to keep per alert.
    columns : list of str, str, or None
        Columns to load. None loads every column.
    """

    name: str
    catalog: "lsdb.Catalog | str | Path"
    radius_arcsec: float
    n_neighbors: int = 1
    columns: list[str] | str | None = None


def alerts_to_catalog(
    alerts: pd.DataFrame,
    *,
    ra_column: str = ALERT_RA_COLUMN,
    dec_column: str = ALERT_DEC_COLUMN,
) -> lsdb.Catalog:
    """Wrap a conditioned alert frame in an in-memory LSDB catalog.

    :meth:`lsdb.Catalog.crossmatch_nested` has no module-level equivalent that
    takes a DataFrame, so the alerts have to become a catalog before they can
    be matched.

    Parameters
    ----------
    alerts : pandas.DataFrame or nested_pandas.NestedFrame
        Alerts as BOOM returned them. Every row needs a usable coordinate, and
        no column may still hold raw lists or dicts -- LSDB can partition
        neither.
    ra_column, dec_column : str
        Names of the coordinate columns, in degrees.

    Returns
    -------
    lsdb.Catalog
        A catalog partitioned on the alert positions. No margin cache is
        generated: the alerts are the *left* side of every crossmatch, where a
        margin is not required.
    """
    return lsdb.from_dataframe(
        alerts,
        ra_column=ra_column,
        dec_column=dec_column,
        margin_threshold=None,
        catalog_name="boom_alerts",
    )


def _check_specs(specs: Sequence[CatalogSpec]) -> None:
    """Reject an empty or ambiguous set of catalogs before any work starts."""
    if not specs:
        raise ValueError("No catalogs given to cross-match against.")
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate catalog name(s): {', '.join(duplicates)}.")


def crossmatch_catalog(
    alerts: lsdb.Catalog,
    specs: Sequence[CatalogSpec],
) -> lsdb.Catalog:
    """Cross-match an alert *catalog* against each catalog, staying lazy.

    Nothing is computed here: each :meth:`~lsdb.Catalog.crossmatch_nested` adds
    a nested column to the lazy catalog, so the caller decides when to compute.
    Starting from an in-memory frame, wrap it with :func:`alerts_to_catalog`
    first.

    Every alert is kept, whether or not it matched: the crossmatches are left
    joins, so an alert that matched one catalog but not another still appears
    once, with an empty nested column for the catalog it missed.

    Parameters
    ----------
    alerts : lsdb.Catalog
        The alerts, as a catalog. See :func:`alerts_to_catalog`.
    specs : sequence of CatalogSpec
        The catalogs to match against, in order. Each contributes one nested
        column named after its ``name``.

    Returns
    -------
    lsdb.Catalog
        A lazy catalog with one nested column per spec.

    Raises
    ------
    ValueError
        If ``specs`` is empty or two specs share a name.
    """
    _check_specs(specs)

    matched = alerts
    for spec in specs:
        catalog = (
            spec.catalog
            if isinstance(spec.catalog, lsdb.Catalog)
            else lsdb.open_catalog(spec.catalog, columns=spec.columns)
        )
        logger.info(
            "Cross-matching against %r within %.2f arcsec (up to %d neighbour(s)).",
            spec.name,
            spec.radius_arcsec,
            spec.n_neighbors,
        )
        matched = matched.crossmatch_nested(
            catalog,
            radius_arcsec=spec.radius_arcsec,
            n_neighbors=spec.n_neighbors,
            nested_column_name=spec.name,
            how="left",
        )
    return matched


def match_counts(matches: npd.NestedFrame, name: str) -> pd.Series:
    """Number of matches each alert got from one catalog.

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        A computed crossmatch, as :func:`crossmatch_catalog` produces.
    name : str
        The nested column to count, i.e. a :attr:`CatalogSpec.name`.

    Returns
    -------
    pandas.Series
        Per-row match counts, zero where the alert matched nothing.
    """
    if name not in matches.columns or matches.empty:
        return pd.Series([0] * len(matches), index=matches.index)
    return pd.Series(matches[name].array.list_lengths, index=matches.index)


def filter_matched(matches: npd.NestedFrame, names: Sequence[str]) -> npd.NestedFrame:
    """Keep only the alerts that matched at least one of the named catalogs.

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        A computed crossmatch, as :func:`crossmatch_catalog` produces.
    names : sequence of str
        The nested columns to consider.

    Returns
    -------
    nested_pandas.NestedFrame
        The matched rows, with a reset index.
    """
    if matches.empty:
        return matches
    keep = pd.Series(False, index=matches.index)
    for name in names:
        keep |= match_counts(matches, name) > 0
    return matches[keep].reset_index(drop=True)


def summarize_matches(
    matches: npd.NestedFrame,
    alerts: pd.DataFrame,
    names: Sequence[str],
) -> dict[str, Any]:
    """Build a small summary dict for logging a crossmatch run.

    Parameters
    ----------
    matches : nested_pandas.NestedFrame
        A computed crossmatch, as :func:`crossmatch_catalog` produces.
    alerts : pandas.DataFrame or nested_pandas.NestedFrame
        The alerts that went in.
    names : sequence of str
        The nested columns to report on.

    Returns
    -------
    dict
        Alerts in, how many matched each catalog (``n_matches_<name>``), and
        how many matched at least one. Every value is a count of *alerts*, so
        an alert matching several sources in one catalog still counts once.
    """
    summary: dict[str, Any] = {"n_alerts": len(alerts)}
    matched_any = pd.Series(False, index=matches.index)
    for name in names:
        counts = match_counts(matches, name)
        matched_any |= counts > 0
        summary[f"n_matches_{name}"] = int((counts > 0).sum())
    summary["n_alerts_matched"] = int(matched_any.sum())
    return summary


def catalog_specs(cfg: PipelineConfig) -> list[CatalogSpec]:
    """Turn the ``[crossmatch.catalogs.<name>]`` tables into catalog specs.

    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration.

    Returns
    -------
    list of CatalogSpec
        One per configured catalog, in the order the config lists them. Each
        table's name becomes the nested column its matches land in.

    Raises
    ------
    ValueError
        If no ``[crossmatch.catalogs.<name>]`` table is configured. The requirement
        lives here, in the stage that needs it, rather than in the config
        model -- a run of other stages alone should not have to declare one.
    """
    catalogs = cfg.crossmatch.catalogs
    if not catalogs:
        raise ValueError(
            f"Stage {STAGE!r} needs at least one [crossmatch.catalogs.<name>] table naming "
            "a catalog to cross-match alerts against."
        )
    return [
        CatalogSpec(
            name=name,
            catalog=entry.catalog,
            radius_arcsec=entry.radius_arcsec,
            n_neighbors=entry.n_neighbors,
            columns=entry.columns,
        )
        for name, entry in catalogs.items()
    ]


def run_crossmatch(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    inputs: StageInputs | None = None,
    stamp: str | None = None,
) -> StageResult:
    """Run the stage: take the alerts, cross-match them, write the matches.

    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration, normally from
        :func:`desi_aap.config.load_config`.
    dry_run : bool
        Do the work but write nothing. The matches are still computed, so a
        dry run reports the same summary a real one would.
    inputs : dict of str to StageResult, optional
        Results of the stages that already ran. The alerts come from
        ``query``.
    stamp : str, optional
        This run's timestamp, naming the output file. Defaults to now.

    Returns
    -------
    StageResult
        The matched alerts and where they were written. ``frame`` is ``None``
        when there were no alerts to match.

    Raises
    ------
    KeyError
        If ``query`` has not run.
    ValueError
        If no catalogs are configured to match against.
    """
    specs = catalog_specs(cfg)
    names = [spec.name for spec in specs]
    stage_dir = cfg.run.stage_dir(STAGE)

    stamp = stamp or run_stamp()
    alerts = input_result(inputs, QUERY_STAGE).frame

    if alerts is None or alerts.empty:
        logger.info("No alerts to cross-match; writing nothing.")
        return StageResult(stage=STAGE, frame=None, stamp=stamp, summary={"n_alerts": 0})

    with dask_client(cfg.dask_for(STAGE)):
        matched = crossmatch_catalog(alerts_to_catalog(alerts), specs)
        matches = matched.compute().reset_index(drop=True)

    # Left joins keep every alert, including those that matched nothing.
    matches = filter_matched(matches, names)
    summary = summarize_matches(matches, alerts, names)
    logger.info("Crossmatch summary: %s", summary)

    output_path: Path | None = None
    if dry_run:
        logger.info("Dry run: not writing the matches.")
    elif matches.empty:
        logger.info("No alerts matched any catalog; writing nothing.")
    else:
        output_path = write_frame(matches, stage_dir / f"{OUTPUT_PREFIX}_{stamp}.parquet")
        logger.info("Wrote %d matched alerts to %s.", len(matches), output_path)

    return StageResult(stage=STAGE, frame=matches, output_path=output_path, stamp=stamp, summary=summary)
