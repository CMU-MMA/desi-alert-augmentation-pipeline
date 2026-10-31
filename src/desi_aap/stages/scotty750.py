"""The SCOTTY750 screen: nuclear TDE candidates within 750 Mpc.

SCOTTY750 -- the Spectroscopically Complete catalOg of Tidal disruption events
ouT to 750 Mpc -- is a Gemini program to classify every TDE-like nuclear
transient within 750 Mpc. This filter picks its candidates out of the placed
alerts; :mod:`desi_aap.stages.fritz_save` then saves them to the program's Fritz
group.

The selection is split between BOOM and here. What can be decided from the
alert and BOOM's own cross-matches happens server-side, in the per-survey
pipelines under ``pipelines/scotty750/`` (bogus, star, asteroid and bright-star
rejection; the catalogue AGN vetoes; and nuclear-ness, the alert within 0.5" of
a Legacy Survey DR10 galaxy for ZTF and 0.3" of an LSPSC source for LSST and
DECam). What needs the host redshift or the light curve happens here:

* **distance** -- the host's luminosity distance is at most ``max_dist_mpc``;
* **host** -- the host the redshift came from is the galaxy the alert is
  nuclear to, within ``max_host_sep_arcsec``;
* **QSO host** -- a host whose DESI spectrum is classified QSO is an AGN;
* **archival flare** -- an object whose light curve has a quiet gap of at least
  ``max_quiet_gap_days`` between detection episodes flared before;
* **color** -- TDEs are blue, so the most recent g - r, from detections at most
  ``color_max_dt_days`` apart, is below ``max_g_minus_r``. The window can be set
  per survey (``color_max_dt_days_by_survey``): LSST rarely takes g and r on
  the same night, so its pairs are allowed a week apart.

The light curve comes from the ``lc`` nested column the SCOTTY750 BOOM
pipelines project: one row per positive detection, with ``jd``, ``band`` and
``mag``. The archival-flare veto reads BOOM's
``properties.episode_history.longest_quiet_days``.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import nested_pandas as npd
import numpy as np
import pandas as pd

from desi_aap.boom import ALERT_SURVEY_COLUMN
from desi_aap.config import PipelineConfig
from desi_aap.cosmology import COSMOLOGIES
from desi_aap.stages.base import SlackDisplay, StageInputs, StageResult, input_result, write_frame
from desi_aap.stages.distance import STAGE as DISTANCE_STAGE
from desi_aap.stages.distance import dist_column
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)

STAGE = "scotty750"

# Stages whose output this one consumes. See crossmatch.REQUIRES.
REQUIRES: tuple[str, ...] = (DISTANCE_STAGE,)

# Off unless a [scotty750] section is present: its cuts assume alerts its own
# BOOM pipelines selected, and on any other run's alerts would pass non-nuclear
# transients. See desi_aap.stages.filters._code_filter_enabled.
ENABLED_WITHOUT_SECTION = False

# Prefix of the parquet file this stage writes.
OUTPUT_PREFIX = "candidates"

# The nested light-curve column the SCOTTY750 BOOM pipelines project, and its fields.
LIGHTCURVE_COLUMN = "lc"
LC_JD, LC_BAND, LC_MAG = "jd", "band", "mag"

# BOOM's longest quiet gap between detection episodes, flattened.
QUIET_GAP_COLUMN = "properties.episode_history.longest_quiet_days"

# Columns this stage adds.
COLOR_COLUMN = "g_minus_r"
COLOR_DT_COLUMN = "g_minus_r_dt_days"
COLOR_JD_COLUMN = "g_minus_r_jd"

SLACK_DISPLAY = SlackDisplay(
    title="SCOTTY750 nuclear TDE candidate",
    columns=("survey", dist_column("SHOES"), "host_redshift", "host_redshift_kind", COLOR_COLUMN),
)


def latest_color(
    lightcurve: pd.DataFrame,
    *,
    max_dt_days: float,
    blue: str = "g",
    red: str = "r",
) -> tuple[float, float, float]:
    """The most recent blue - red color of one light curve.

    Every pair of a ``blue`` and a ``red`` detection at most ``max_dt_days``
    apart is a color measurement; the one whose mean epoch is latest wins, the
    nearer pair breaking a tie.

    Parameters
    ----------
    lightcurve : pandas.DataFrame
        One alert's detections, with ``jd``, ``band`` and ``mag`` columns.
    max_dt_days : float
        Largest gap between the two detections of a pair.
    blue, red : str
        The two bands.

    Returns
    -------
    tuple of float
        ``(color, dt_days, mean_jd)`` of the chosen pair, or three NaNs when no
        pair qualifies.
    """
    missing = (np.nan, np.nan, np.nan)
    if lightcurve.empty:
        return missing
    mags = pd.to_numeric(lightcurve[LC_MAG], errors="coerce")
    jds = pd.to_numeric(lightcurve[LC_JD], errors="coerce")
    good = mags.notna() & jds.notna()
    bands = lightcurve[LC_BAND].astype("object")
    b = good & bands.eq(blue)
    r = good & bands.eq(red)
    if not b.any() or not r.any():
        return missing
    b_jd, b_mag = jds[b].to_numpy(float), mags[b].to_numpy(float)
    r_jd, r_mag = jds[r].to_numpy(float), mags[r].to_numpy(float)
    dt = np.abs(b_jd[:, None] - r_jd[None, :])
    ok = dt <= max_dt_days
    if not ok.any():
        return missing
    mean_jd = np.where(ok, (b_jd[:, None] + r_jd[None, :]) / 2, -np.inf)
    # Latest mean epoch first, then the tighter pair.
    order = np.lexsort((dt.ravel(), -mean_jd.ravel()))
    i, j = np.unravel_index(order[0], dt.shape)
    return float(b_mag[i] - r_mag[j]), float(dt[i, j]), float(mean_jd[i, j])


def measure_colors(
    alerts: npd.NestedFrame,
    *,
    max_dt_days: float,
    max_dt_days_by_survey: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """The latest g - r of every alert, from its ``lc`` nested column.

    Parameters
    ----------
    alerts : nested_pandas.NestedFrame
        The placed alerts.
    max_dt_days : float
        See :func:`latest_color`.
    max_dt_days_by_survey : mapping of str to float, optional
        ``max_dt_days`` for the alerts of the surveys named, read from the
        alert's ``survey`` column. Alerts of any other survey, or with no
        survey column, use ``max_dt_days``.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``alerts``, with :data:`COLOR_COLUMN`,
        :data:`COLOR_DT_COLUMN` and :data:`COLOR_JD_COLUMN`; NaN where no
        color could be measured.
    """
    result = pd.DataFrame(
        np.nan, index=alerts.index, columns=[COLOR_COLUMN, COLOR_DT_COLUMN, COLOR_JD_COLUMN]
    )
    if LIGHTCURVE_COLUMN not in alerts.columns or alerts.empty:
        return result
    flat = alerts[LIGHTCURVE_COLUMN].nest.to_flat()
    missing = [field for field in (LC_JD, LC_BAND, LC_MAG) if field not in flat.columns]
    if missing:
        raise ValueError(
            f"The {LIGHTCURVE_COLUMN!r} nested column has no {', '.join(missing)} field(s); check the "
            "final $project of the SCOTTY750 BOOM pipelines."
        )
    windows = pd.Series(max_dt_days, index=alerts.index, dtype=float)
    if max_dt_days_by_survey and ALERT_SURVEY_COLUMN in alerts.columns:
        surveys = alerts[ALERT_SURVEY_COLUMN].astype(object)
        for survey, window in max_dt_days_by_survey.items():
            windows[surveys.eq(survey).fillna(False).to_numpy(bool)] = window
    for index, lightcurve in flat.groupby(level=0, sort=False):
        result.loc[index] = latest_color(lightcurve, max_dt_days=windows[index])
    return result


def select_candidates(alerts: npd.NestedFrame, cfg: PipelineConfig) -> tuple[npd.NestedFrame, dict[str, Any]]:
    """Apply the SCOTTY750 cuts to the placed alerts.

    Parameters
    ----------
    alerts : nested_pandas.NestedFrame
        The distance stage's output.
    cfg : desi_aap.config.PipelineConfig
        Read for ``[scotty750]``.

    Returns
    -------
    nested_pandas.NestedFrame
        The passing alerts, with the color columns added, indexed from zero.
    dict
        How many alerts went in, how many each cut removed on its own, and how
        many passed.

    Raises
    ------
    ValueError
        If the configured cosmology is unknown, the distance column is missing,
        or color is required and the alerts carry no light curve.
    """
    settings = cfg.scotty750
    if settings.cosmology not in COSMOLOGIES:
        raise ValueError(
            f"[scotty750] cosmology {settings.cosmology!r} is not one of {', '.join(COSMOLOGIES)}."
        )
    distance = dist_column(settings.cosmology)
    if distance not in alerts.columns:
        raise ValueError(f"The placed alerts have no {distance!r} column; did the distance stage run?")
    if settings.require_color and LIGHTCURVE_COLUMN not in alerts.columns:
        raise ValueError(
            f"[scotty750] require_color is on, but the alerts carry no {LIGHTCURVE_COLUMN!r} light curve "
            "to measure g - r from. Query with the pipelines in pipelines/scotty750/, whose final "
            "$project builds it, or set require_color = false."
        )

    colors = measure_colors(
        alerts,
        max_dt_days=settings.color_max_dt_days,
        max_dt_days_by_survey=settings.color_max_dt_days_by_survey,
    )
    color = colors[COLOR_COLUMN]

    cuts: dict[str, pd.Series] = {
        "distance": pd.to_numeric(alerts[distance], errors="coerce").astype(float) <= settings.max_dist_mpc,
        "host_sep": pd.to_numeric(alerts["host_sep_arcsec"], errors="coerce").astype(float)
        <= settings.max_host_sep_arcsec,
    }
    if settings.veto_qso_hosts and "host_spectype" in alerts.columns:
        spectype = (
            alerts["host_spectype"]
            .astype("object")
            .map(lambda v: str(v).strip().upper() if isinstance(v, str) else "")
        )
        cuts["qso_host"] = ~spectype.eq("QSO")
    if settings.max_quiet_gap_days is not None and QUIET_GAP_COLUMN in alerts.columns:
        gap = pd.to_numeric(alerts[QUIET_GAP_COLUMN], errors="coerce").astype(float)
        # No gap at all (a single episode) is no evidence of an earlier flare.
        # Filled before negating: on an Arrow column a missing gap compares as
        # missing, and negated-then-filled it would fail every single-episode alert.
        flared = (gap >= settings.max_quiet_gap_days).fillna(False).astype(bool)
        cuts["archival_flare"] = ~flared
    blue = color < settings.max_g_minus_r
    cuts["color"] = blue if settings.require_color else (blue | color.isna())

    keep = pd.Series(True, index=alerts.index)
    summary: dict[str, Any] = {"n_alerts": len(alerts), "n_with_color": int(color.notna().sum())}
    for name, passed in cuts.items():
        passed = passed.fillna(False).astype(bool)
        summary[f"n_failed_{name}"] = int((~passed).sum())
        keep &= passed
    summary["n_candidates"] = int(keep.sum())

    selected = alerts[keep].copy()
    for column in colors.columns:
        selected[column] = colors.loc[keep, column].to_numpy()
    return npd.NestedFrame(selected.reset_index(drop=True)), summary


def run_scotty750(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    inputs: StageInputs | None = None,
    stamp: str | None = None,
) -> StageResult:
    """Run the stage: keep the placed alerts that pass the SCOTTY750 cuts.

    Parameters
    ----------
    cfg : desi_aap.config.PipelineConfig
        The pipeline configuration; ``[scotty750]`` must be present.
    dry_run : bool
        Do the work but write nothing.
    inputs : dict of str to StageResult, optional
        Results of the stages that already ran. The alerts come from ``distance``.
    stamp : str, optional
        This run's timestamp, naming the output file. Defaults to now.

    Returns
    -------
    StageResult
        The candidates and where they were written. ``frame`` is ``None`` when
        there were no placed alerts to screen.
    """
    stamp = stamp or run_stamp()
    stage_dir = cfg.run.stage_dir(STAGE)
    (upstream,) = REQUIRES
    alerts = input_result(inputs, upstream).frame

    if alerts is None or alerts.empty:
        logger.info("No placed alerts to screen; writing nothing.")
        return StageResult(stage=STAGE, frame=None, stamp=stamp, summary={"n_alerts": 0})

    frame, summary = select_candidates(alerts, cfg)
    logger.info("SCOTTY750 summary: %s", summary)

    output_path: Path | None = None
    if dry_run:
        logger.info("Dry run: not writing the candidates.")
    elif frame.empty:
        logger.info("No alert passed the SCOTTY750 cuts; writing nothing.")
    else:
        output_path = write_frame(frame, stage_dir / f"{OUTPUT_PREFIX}_{stamp}.parquet")
        logger.info("Wrote %d SCOTTY750 candidates to %s.", len(frame), output_path)

    return StageResult(stage=STAGE, frame=frame, output_path=output_path, stamp=stamp, summary=summary)
