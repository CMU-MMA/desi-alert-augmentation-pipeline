"""Command-line entry point for the DESI alert augmentation pipeline."""

# ruff: noqa: B008
import logging
from pathlib import Path

import typer

from desi_aap.config import ConfigError, load_config
from desi_aap.log import run_log_path, setup_logging
from desi_aap.pipeline import STAGE_ORDER, run_pipeline
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)


app = typer.Typer(
    help="DESI Alert Augmentation Pipeline -- cross-match broker alerts with DESI "
    "spectroscopy, score them against known GW event localizations, and publish "
    "the best follow-up candidates",
    no_args_is_help=True,
)


@app.command()
def run(
    config_paths: list[Path] = typer.Option(
        ...,
        "--config",
        "-c",
        help="TOML config file(s). Give several to layer overrides (left to right).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Do the work but write no results.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Log at DEBUG rather than INFO.",
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        help="Write the log here instead of under <output_dir>/logs/.",
    ),
) -> None:
    """Run the pipeline."""
    try:
        cfg = load_config(config_paths)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    timestamp = run_stamp()
    log_path = log_file or run_log_path(cfg.run.output_dir, timestamp)
    setup_logging(log_path, verbose=verbose)

    try:
        run_pipeline(cfg, dry_run=dry_run, stamp=timestamp)
    except Exception:
        logger.exception("Pipeline failed.")
        raise typer.Exit(1) from None


@app.command("stages")
def list_stages() -> None:
    """List the stages this pipeline knows about, in the order they run."""
    for index, stage in enumerate(STAGE_ORDER, start=1):
        typer.echo(f"{index}. {stage}")
