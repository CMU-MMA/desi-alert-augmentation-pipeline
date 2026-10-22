"""Command-line entry point for the DESI alert augmentation pipeline."""

# ruff: noqa: B008
import logging
from pathlib import Path

import nested_pandas as npd
import typer

from desi_aap.config import ConfigError, load_config
from desi_aap.log import run_log_path, setup_logging
from desi_aap.pipeline import DATA_STAGES, run_pipeline, stages_for
from desi_aap.stages.base import StageInputs, StageResult
from desi_aap.stages.filters import FILTER_MODULES
from desi_aap.stages.slack_publish import STAGE as SLACK_STAGE
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

    try:
        cfg = load_config(config_paths)
        # Building the stage list reads the filters directory, so a malformed
        # filter file is caught here, with the config errors, not mid-run.
        specs = {spec.name: spec for spec in stages_for(cfg)}
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    # Validated against this config's stages, not a fixed list: every JSON
    # filter is a stage, so what --from-stage may name depends on the config.
    if from_stage is not None:
        if from_stage not in specs:
            typer.echo(
                f"error: unknown stage {from_stage!r}. Stages, in order: {', '.join(specs)}.",
                err=True,
            )
            raise typer.Exit(2)
        if from_stage == next(iter(specs)):
            typer.echo(
                f"error: {from_stage!r} is already the first stage; run without --from-stage.",
                err=True,
            )
            raise typer.Exit(2)

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
            # gets the file as the FIRST of them only: one file is one stage's
            # output, and handing the same rows to every filter would post them
            # to Slack once per filter under names whose cuts never ran. The
            # rest are recorded as not run, which the stage reports as such.
            first, *rest = specs[from_stage].requires
            frame = npd.read_parquet(input_path)
            inputs = {first: StageResult(stage=first, frame=frame, output_path=input_path, stamp=timestamp)}
            inputs |= {
                producer: StageResult(
                    stage=producer, stamp=timestamp, summary={"skipped": "no --input stand-in"}
                )
                for producer in rest
            }
            if rest:
                logger.info(
                    "--input stands in for %r; %s have no stand-in and are treated as not run.",
                    first,
                    ", ".join(repr(r) for r in rest),
                )
        run_pipeline(cfg, dry_run=dry_run, stamp=timestamp, start=from_stage, inputs=inputs)
    except Exception:
        logger.exception("Pipeline failed.")
        raise typer.Exit(1) from None


@app.command("stages")
def list_stages(
    config_paths: list[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="TOML config file(s). With one, the configured JSON filters are listed too.",
    ),
) -> None:
    """List the stages this pipeline knows about, in the order they run.

    The filters are stages, and they come from the config's filters directory,
    so the full list needs a config. Without one, only the stages every
    pipeline has are listed, with a note saying what is missing.
    """
    if config_paths:
        try:
            cfg = load_config(config_paths)
            names = [spec.name for spec in stages_for(cfg)]
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from None
    else:
        # The code filters exist in every pipeline; only the JSON filters need
        # a config to be known.
        names = [
            *(spec.name for spec in DATA_STAGES),
            *(module.STAGE for module in FILTER_MODULES),
            SLACK_STAGE,
        ]
        typer.echo("(no --config: any JSON filters, which land before slack_publish, come from it)")
    for index, stage in enumerate(names, start=1):
        typer.echo(f"{index}. {stage}")
