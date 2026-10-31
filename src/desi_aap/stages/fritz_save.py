"""Save filter candidates to Fritz: the source, its redshift, and an annotation.

Each candidate of every filter named in ``[fritz] filters`` is, in order:

1. **saved through the BOOM broker** (``POST /api/brokers/{id}/objects/{objectId}/save``),
   which creates the Fritz object from BOOM with its photometry and cutouts, or
   refreshes the photometry of one that exists;
2. **saved to the groups** (``POST /api/sources``). The broker save only adds a
   *new* object to groups, so an object Fritz already had -- saved by another
   group, or ingested as a candidate -- would otherwise never reach ours. If the
   broker save failed, this call creates a bare source from the alert position
   so the candidate still lands in the group;
3. **given a redshift**, when ``[fritz] set_redshift`` is on and the source has
   none. A redshift this pipeline set from a photo-z is upgraded when a
   spectroscopic one becomes available; anyone else's redshift is never
   touched. ``redshift_origin`` records where it came from, as
   ``<annotation_origin>:<catalog>:<spec|photo>``;
4. **annotated** with what the pipeline measured: host redshift and
   separation, distance, absolute magnitude, g - r, and the BOOM pipeline's own
   ``annotations.*`` fields. One annotation per source, origin
   ``[fritz] annotation_origin``, replaced on every run so it stays current.

A failure on one candidate is logged and recorded, and the rest still go. The
stage writes one row per candidate saying what was done to
``fritz_save/saves_<stamp>.parquet``.

The API token is read from the environment variable ``[fritz] token_env``
(``FRITZ_TOKEN`` by default), never from a file in the repository.
"""

import logging
import math
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from desi_aap import boom
from desi_aap.config import ConfigError, FritzConfig, PipelineConfig
from desi_aap.fritz import FritzClient, FritzError
from desi_aap.stages.base import StageInputs, StageResult, write_frame
from desi_aap.stages.filters import FilterDescriptor, filter_descriptors
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)

STAGE = "fritz_save"

# Like slack_publish, this stage's dependencies are the filters it saves, named
# in the config, so desi_aap.pipeline.stages_for builds them per run.

# Prefix of the parquet file this stage writes.
OUTPUT_PREFIX = "saves"

# Flat columns copied into the annotation when present, under these names.
# Anything else worth showing comes from the BOOM pipeline's annotations.*.
ANNOTATION_COLUMNS = {
    boom.ALERT_SURVEY_COLUMN: "survey",
    "host_catalog": "host_catalog",
    "host_redshift": "host_z",
    "host_redshift_err": "host_z_err",
    "host_redshift_kind": "host_z_kind",
    "host_sep_arcsec": "host_sep_arcsec",
    "host_spectype": "host_spectype",
    "dist_mpc_SHOES": "dist_mpc_SHOES",
    "dist_mpc_Planck18": "dist_mpc_Planck18",
    "abs_mag_SHOES": "abs_mag_SHOES",
    "g_minus_r": "g_minus_r",
    "g_minus_r_dt_days": "g_minus_r_dt_days",
    boom.ALERT_MAG_COLUMN: "mag_latest",
    boom.ALERT_BAND_COLUMN: "band_latest",
    boom.ALERT_TIME_COLUMN: "jd_latest",
}

# The prefix of BOOM pipeline columns carried into the annotation as they are.
BOOM_ANNOTATION_PREFIX = "annotations."

# Columns of the record this stage writes per candidate.
RECORD_COLUMNS = (
    "objectId",
    "survey",
    "filters",
    "broker_saved",
    "group_saved",
    "redshift_action",
    "annotation_action",
    "error",
)


def resolve_filters(
    cfg: PipelineConfig, descriptors: Sequence[FilterDescriptor] | None = None
) -> tuple[str, ...]:
    """The filters ``[fritz] filters`` names, checked against the ones this run has.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        The configuration.
    descriptors : sequence of FilterDescriptor, optional
        This run's filters, as :func:`desi_aap.stages.filters.filter_descriptors`
        returns them. Read from ``cfg`` when omitted.

    Returns
    -------
    tuple of str
        The filter stage names, in the order written; empty without ``[fritz]``.

    Raises
    ------
    desi_aap.config.ConfigError
        If a name matches no filter. A typo would otherwise save nothing, quietly.
    """
    if cfg.fritz is None:
        return ()
    known = {d.stage for d in (descriptors if descriptors is not None else filter_descriptors(cfg))}
    unknown = [name for name in cfg.fritz.filters if name not in known]
    if unknown:
        raise ConfigError(
            f"[fritz] filters names {', '.join(repr(u) for u in unknown)}, which match no filter. "
            f"This run has: {', '.join(sorted(known))}."
        )
    return tuple(cfg.fritz.filters)


def _jsonable(value: Any) -> Any:
    """A JSON-safe scalar, or None for anything missing or not a scalar."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 5)
    if isinstance(value, bool | int | str):
        return value
    return None


def annotation_data(row: pd.Series, filters: Sequence[str]) -> dict[str, Any]:
    """What the pipeline knows about one candidate, as annotation data.

    Parameters
    ----------
    row : pandas.Series
        One candidate row.
    filters : sequence of str
        The filters it passed.

    Returns
    -------
    dict
        JSON-safe values only; anything missing is left out.
    """
    data: dict[str, Any] = {}
    for column, key in ANNOTATION_COLUMNS.items():
        if column in row.index:
            value = _jsonable(row[column])
            if value is not None:
                data[key] = value
    for column in row.index:
        if isinstance(column, str) and column.startswith(BOOM_ANNOTATION_PREFIX):
            value = _jsonable(row[column])
            if value is not None:
                data[column.removeprefix(BOOM_ANNOTATION_PREFIX)] = value
    data["filters"] = ", ".join(filters)
    data["updated_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return data


def redshift_origin(settings: FritzConfig, catalog: str, kind: str) -> str:
    """The ``redshift_origin`` this pipeline writes, ``<origin>:<catalog>:<kind>``."""
    return f"{settings.annotation_origin}:{catalog}:{kind}"


def redshift_decision(source: dict[str, Any], row: pd.Series, settings: FritzConfig) -> str | None:
    """Whether to set the source's redshift from this candidate's host, and why.

    Parameters
    ----------
    source : dict
        The source as ``GET /api/sources/{id}`` returned it.
    row : pandas.Series
        The candidate, with the distance stage's host columns.
    settings : FritzConfig
        Read for ``set_redshift`` and the origin prefix.

    Returns
    -------
    str or None
        ``"set"`` when the source has no redshift, ``"upgrade"`` when its
        redshift is a photo-z this pipeline set and the host's is
        spectroscopic, None to leave it alone.
    """
    if not settings.set_redshift:
        return None
    z = _jsonable(row.get("host_redshift"))
    kind = row.get("host_redshift_kind")
    if z is None or kind not in ("spec", "photo"):
        return None
    if source.get("redshift") is None:
        return "set"
    current = str(source.get("redshift_origin") or "")
    ours = current.startswith(f"{settings.annotation_origin}:")
    if ours and current.endswith(":photo") and kind == "spec":
        return "upgrade"
    return None


def save_candidate(
    client: FritzClient, row: pd.Series, filters: Sequence[str], settings: FritzConfig
) -> dict[str, Any]:
    """Save one candidate to Fritz, returning a record of what was done.

    Parameters
    ----------
    client : FritzClient
        An authenticated client.
    row : pandas.Series
        The candidate.
    filters : sequence of str
        The filters it passed, for the annotation.
    settings : FritzConfig
        The ``[fritz]`` section.

    Returns
    -------
    dict
        One :data:`RECORD_COLUMNS` row. A failure stops the steps after it and
        is recorded under ``error`` rather than raised.
    """
    obj_id = str(row[boom.ALERT_ID_COLUMN])
    record: dict[str, Any] = {
        "objectId": obj_id,
        "survey": _jsonable(row.get(boom.ALERT_SURVEY_COLUMN)),
        "filters": ", ".join(filters),
        "broker_saved": False,
        "group_saved": False,
        "redshift_action": None,
        "annotation_action": None,
        "error": None,
    }
    errors = []
    try:
        client.save_from_broker(settings.broker_id, obj_id, settings.group_ids)
        record["broker_saved"] = True
    except FritzError as exc:
        # Not fatal: the source can still be saved bare, from the alert position.
        errors.append(f"broker save: {exc}")
        logger.warning("Fritz broker save of %s failed; saving it without photometry. %s", obj_id, exc)
    try:
        client.save_source(
            obj_id,
            settings.group_ids,
            ra=_jsonable(row.get(boom.ALERT_RA_COLUMN)),
            dec=_jsonable(row.get(boom.ALERT_DEC_COLUMN)),
        )
        record["group_saved"] = True

        if settings.set_redshift:
            source = client.get_source(obj_id) or {}
            decision = redshift_decision(source, row, settings)
            if decision is not None:
                client.set_redshift(
                    obj_id,
                    float(row["host_redshift"]),
                    _jsonable(row.get("host_redshift_err")),
                    redshift_origin(settings, str(row["host_catalog"]), str(row["host_redshift_kind"])),
                )
            record["redshift_action"] = decision or "kept"

        data = annotation_data(row, filters)
        existing = [
            a for a in client.get_annotations(obj_id) if a.get("origin") == settings.annotation_origin
        ]
        if existing:
            client.put_annotation(
                obj_id, existing[0]["id"], settings.annotation_origin, data, settings.group_ids
            )
            record["annotation_action"] = "updated"
        else:
            client.post_annotation(obj_id, settings.annotation_origin, data, settings.group_ids)
            record["annotation_action"] = "created"
    except FritzError as exc:
        errors.append(str(exc))
        logger.error("Fritz save of %s stopped: %s", obj_id, exc)
    record["error"] = "; ".join(errors) or None
    return record


def gather_candidates(
    inputs: StageInputs, filters: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Every candidate of the named filters, one row per object.

    An object several filters passed is saved once; the row kept is the one
    with the latest alert, and every filter it passed is remembered.

    Parameters
    ----------
    inputs : dict of str to StageResult
        Results of the stages that already ran.
    filters : sequence of str
        The filter stage names.

    Returns
    -------
    pandas.DataFrame
        The candidates, one per objectId.
    dict of str to list of str
        For each objectId, the filters it passed.

    Raises
    ------
    KeyError
        If a named filter did not run at all, which means the stage order is
        wrong rather than that the filter found nothing.
    """
    frames = []
    passed: dict[str, list[str]] = {}
    for name in filters:
        if name not in inputs:
            raise KeyError(f"Filter {name!r} has not run, so its candidates cannot be saved.")
        frame = inputs[name].frame
        if frame is None or frame.empty:
            continue
        for obj_id in frame[boom.ALERT_ID_COLUMN].astype(str):
            passed.setdefault(obj_id, [])
            if name not in passed[obj_id]:
                passed[obj_id].append(name)
        frames.append(pd.DataFrame(frame))
    if not frames:
        return pd.DataFrame(), passed
    candidates = pd.concat(frames, ignore_index=True)
    candidates[boom.ALERT_ID_COLUMN] = candidates[boom.ALERT_ID_COLUMN].astype(str)
    if boom.ALERT_TIME_COLUMN in candidates.columns:
        candidates = candidates.sort_values(boom.ALERT_TIME_COLUMN, ascending=False, kind="stable")
    candidates = candidates.drop_duplicates(boom.ALERT_ID_COLUMN, keep="first").reset_index(drop=True)
    return candidates, passed


def run_fritz_save(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    inputs: StageInputs | None = None,
    stamp: str | None = None,
    descriptors: Sequence[FilterDescriptor] | None = None,
    client: FritzClient | None = None,
) -> StageResult:
    """Run the stage: save every named filter's candidates to Fritz.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        The configuration. Without ``[fritz]`` the stage skips itself.
    dry_run : bool
        Log what would be saved, and make no request at all.
    inputs : dict of str to StageResult, optional
        Results of the stages that already ran, including the filters.
    stamp : str, optional
        This run's timestamp, naming the output file. Defaults to now.
    descriptors : sequence of FilterDescriptor, optional
        This run's filters, bound by the pipeline so the names are checked
        against the same list it ran.
    client : FritzClient, optional
        Injected by tests; built from ``[fritz]`` and the token otherwise.

    Returns
    -------
    StageResult
        One row per candidate saying what was done. ``frame`` is ``None`` when
        there was nothing to save.

    Raises
    ------
    desi_aap.config.ConfigError
        If a named filter does not exist, or the token variable is unset.
    """
    stamp = stamp or run_stamp()
    settings = cfg.fritz
    if settings is None:
        logger.info("No [fritz] section configured; saving nothing.")
        return StageResult(stage=STAGE, stamp=stamp, summary={"skipped": "no [fritz] section"})

    filters = resolve_filters(cfg, descriptors)
    candidates, passed = gather_candidates(inputs or {}, filters)
    summary: dict[str, Any] = {"n_candidates": len(candidates)}
    if candidates.empty:
        logger.info("No candidates in %s; saving nothing.", ", ".join(filters))
        return StageResult(stage=STAGE, stamp=stamp, summary=summary)

    if dry_run:
        for obj_id in candidates[boom.ALERT_ID_COLUMN]:
            logger.info(
                "Dry run: would save %s (%s) to Fritz group(s) %s.",
                obj_id,
                ", ".join(passed[obj_id]),
                settings.group_ids,
            )
        return StageResult(stage=STAGE, stamp=stamp, summary=summary)

    if client is None:
        token = os.environ.get(settings.token_env)
        if not token:
            raise ConfigError(
                f"[fritz] needs an API token in the {settings.token_env} environment variable. "
                "A scheduled run must start a login shell (bash -l) for it to be set."
            )
        client = FritzClient(
            settings.base_url,
            token,
            min_request_interval_s=settings.min_request_interval_s,
            timeout_s=settings.timeout_s,
        )

    records = [
        save_candidate(client, row, passed[row[boom.ALERT_ID_COLUMN]], settings)
        for _, row in candidates.iterrows()
    ]
    frame = pd.DataFrame(records, columns=list(RECORD_COLUMNS))
    summary.update(
        n_group_saved=int(frame["group_saved"].sum()),
        n_broker_saved=int(frame["broker_saved"].sum()),
        n_redshift_set=int(frame["redshift_action"].isin(["set", "upgrade"]).sum()),
        n_errors=int(frame["error"].notna().sum()),
    )
    logger.info("Fritz summary: %s", summary)

    output_path: Path | None = write_frame(
        frame, cfg.run.stage_dir(STAGE) / f"{OUTPUT_PREFIX}_{stamp}.parquet"
    )
    logger.info("Wrote the Fritz save record to %s.", output_path)
    return StageResult(stage=STAGE, frame=frame, output_path=output_path, stamp=stamp, summary=summary)
