"""Logging setup for the pipeline's command-line entry point."""

import logging
import sys
from pathlib import Path

PACKAGE_LOGGER = "desi_aap"
LOG_DIR_NAME = "logs"


def run_log_path(output_dir: Path | str, stamp: str) -> Path:
    """Where this run's log goes: ``<output_dir>/logs/<stamp>.log``.

    Parameters
    ----------
    output_dir : Path or str
        ``run.output_dir`` from the config.
    stamp : str
        This run's timestamp, from :func:`desi_aap.utils.run_stamp`. The same
        one names every stage's output, so a run's files group by name.

    Returns
    -------
    Path
        The log file for this run.
    """
    return Path(output_dir) / LOG_DIR_NAME / f"{stamp}.log"


def setup_logging(log_file: Path | None = None, *, verbose: bool = False) -> logging.Logger:
    """Configure the package logger: plain stdout plus an optional log file.

    Parameters
    ----------
    log_file : Path, optional
        File to also write timestamped records to. Parent directories are
        created. Omit for stdout only.
    verbose : bool
        Log at DEBUG rather than INFO.

    Returns
    -------
    logging.Logger
        The configured package logger.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level)
    logger.propagate = False

    # Repeated calls (tests, notebooks) should reconfigure rather than stack up
    # duplicate handlers.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        logger.info("Logging to %s", log_file)

    return logger
