"""Command-line entry point for the DESI alert augmentation pipeline."""

# ruff: noqa: B008
import logging
from pathlib import Path

import nested_pandas as npd
import typer

from desi_aap.config import ConfigError, load_config
from desi_aap.log import run_log_path, setup_logging
from desi_aap.pipeline import STAGE_ORDER, STAGE_SPECS, run_pipeline
from desi_aap.stages.base import StageInputs, StageResult
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
    from_stage: str | None = typer.Option(
        None,
        "--from-stage",
        help="Start at this stage instead of the first, feeding it --input. "
        "See `desi-aap stages` for the names.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help="Parquet file standing in for the previous stage's output. "
        "Requires --from-stage, and vice versa.",
    ),
) -> None:
    """Run the pipeline."""
    if (from_stage is None) != (input_path is None):
        typer.echo("error: --from-stage and --input go together; give both or neither.", err=True)
        raise typer.Exit(2)
    if from_stage is not None:
        if from_stage not in STAGE_ORDER:
            typer.echo(
                f"error: unknown stage {from_stage!r}. Stages, in order: {', '.join(STAGE_ORDER)}.",
                err=True,
            )
            raise typer.Exit(2)
        if from_stage == STAGE_ORDER[0]:
            typer.echo(
                f"error: {from_stage!r} is already the first stage; run without --from-stage.",
                err=True,
            )
            raise typer.Exit(2)

    try:
        cfg = load_config(config_paths)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    timestamp = run_stamp()
    log_path = log_file or run_log_path(cfg.run.output_dir, timestamp)
    setup_logging(log_path, verbose=verbose)

    try:
        inputs: StageInputs | None = None
        if from_stage is not None:
            # The file stands in for what the starting stage consumes, which the
            # stage itself declares -- not simply whatever precedes it in the
            # listing, since the filters are siblings rather than a chain. A
            # stage consuming several (slack_publish, which reads every filter)
            # gets the file as all of them, one file being all --input offers.
            frame = npd.read_parquet(input_path)
            inputs = {
                producer: StageResult(stage=producer, frame=frame, output_path=input_path, stamp=timestamp)
                for producer in STAGE_SPECS[from_stage].requires
            }
        run_pipeline(cfg, dry_run=dry_run, stamp=timestamp, start=from_stage, inputs=inputs)
    except Exception:
        logger.exception("Pipeline failed.")
        raise typer.Exit(1) from None


@app.command("stages")
def list_stages() -> None:
    """List the stages this pipeline knows about, in the order they run."""
    for index, stage in enumerate(STAGE_ORDER, start=1):
        typer.echo(f"{index}. {stage}")
