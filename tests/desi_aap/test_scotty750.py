"""The SCOTTY750 screen: color from the light curve, and each of its cuts."""

import math

import nested_pandas as npd
import pandas as pd
import pytest
from conftest import host, make_placed

from desi_aap import pipeline
from desi_aap.config import Scotty750Config
from desi_aap.stages import scotty750
from desi_aap.stages.base import StageResult
from desi_aap.stages.scotty750 import COLOR_COLUMN, latest_color, run_scotty750, select_candidates

# z = 0.03 is ~135 Mpc under SHOES; z = 0.3 is far beyond 750 Mpc.
NEAR_Z, FAR_Z = 0.03, 0.3

BLUE_LC = [(100.0, "g", 19.0), (100.1, "r", 19.3)]  # g - r = -0.3
RED_LC = [(100.0, "g", 19.5), (100.1, "r", 19.0)]  # g - r = +0.5


def lightcurve(points):
    return pd.DataFrame(points, columns=["jd", "band", "mag"])


def with_lightcurves(placed, lightcurves):
    """Attach an `lc` nested column, one list of (jd, band, mag) per alert row."""
    flat = pd.DataFrame(
        [point for points in lightcurves for point in points],
        columns=["jd", "band", "mag"],
        index=[i for i, points in enumerate(lightcurves) for _ in points],
    )
    return placed.join_nested(flat, "lc", how="left")


@pytest.fixture
def scotty_config(pipeline_config):
    return pipeline_config.model_copy(update={"scotty750": Scotty750Config()})


def placed_alerts(hosts, lightcurves, **columns):
    placed = with_lightcurves(make_placed({"desi_dr1": hosts}), lightcurves)
    if not columns:
        return placed
    # Through pandas, since NestedFrame reads a dotted name as a nested column.
    extra = pd.DataFrame(columns, index=placed.index)
    kept = [c for c in placed.columns if c not in extra.columns]
    return npd.NestedFrame(pd.concat([pd.DataFrame(placed[kept]), extra], axis=1))


# --- latest_color -----------------------------------------------------------


def test_the_latest_pair_gives_the_color():
    lc = lightcurve([(100.0, "g", 20.0), (100.2, "r", 19.0), (110.0, "g", 19.0), (110.3, "r", 19.4)])
    color, dt, jd = latest_color(lc, max_dt_days=1.0)
    assert color == pytest.approx(-0.4)
    assert dt == pytest.approx(0.3)
    assert jd == pytest.approx(110.15)


def test_a_pair_too_far_apart_is_no_color():
    lc = lightcurve([(100.0, "g", 20.0), (103.0, "r", 19.0)])
    assert all(math.isnan(v) for v in latest_color(lc, max_dt_days=1.0))


def test_one_band_alone_is_no_color():
    assert all(math.isnan(v) for v in latest_color(lightcurve([(100.0, "g", 20.0)]), max_dt_days=1.0))


# --- the cuts ----------------------------------------------------------------


def test_a_blue_nearby_nuclear_alert_passes(scotty_config):
    alerts = placed_alerts([[host(z=NEAR_Z, sep=0.2)]], [BLUE_LC])
    candidates, summary = select_candidates(alerts, scotty_config)
    assert len(candidates) == 1
    assert candidates.loc[0, COLOR_COLUMN] == pytest.approx(-0.3)
    assert summary["n_candidates"] == 1


@pytest.mark.parametrize(
    ("hosts", "lightcurves", "extra", "failed"),
    [
        ([[host(z=FAR_Z, sep=0.2)]], [BLUE_LC], {}, "distance"),
        ([[host(z=NEAR_Z, sep=3.0)]], [BLUE_LC], {}, "host_sep"),
        ([[host(z=NEAR_Z, sep=0.2)]], [RED_LC], {}, "color"),
        ([[host(z=NEAR_Z, sep=0.2)]], [[(100.0, "g", 19.0)]], {}, "color"),
        (
            [[host(z=NEAR_Z, sep=0.2)]],
            [BLUE_LC],
            {"properties.episode_history.longest_quiet_days": [200.0]},
            "archival_flare",
        ),
        ([[host(z=NEAR_Z, sep=0.2)]], [BLUE_LC], {"host_spectype": ["QSO"]}, "qso_host"),
    ],
    ids=["too-far", "off-host", "red", "no-color", "earlier-flare", "qso-host"],
)
def test_each_cut_removes_what_it_says(scotty_config, hosts, lightcurves, extra, failed):
    alerts = placed_alerts(hosts, lightcurves, **extra)
    candidates, summary = select_candidates(alerts, scotty_config)
    assert candidates.empty
    assert summary[f"n_failed_{failed}"] == 1


def test_a_single_episode_is_no_evidence_of_an_earlier_flare(scotty_config):
    alerts = placed_alerts(
        [[host(z=NEAR_Z, sep=0.2)]], [BLUE_LC], **{"properties.episode_history.longest_quiet_days": [None]}
    )
    assert len(select_candidates(alerts, scotty_config)[0]) == 1


def test_a_missing_gap_in_an_arrow_column_passes(scotty_config):
    """BOOM's columns arrive Arrow-typed, where a null compares as null rather than False."""
    import pyarrow as pa

    gaps = pd.array([None, 200.0], dtype=pd.ArrowDtype(pa.float64()))
    alerts = placed_alerts(
        [[host(z=NEAR_Z, sep=0.2)], [host(z=NEAR_Z, sep=0.2)]],
        [BLUE_LC, BLUE_LC],
        **{"properties.episode_history.longest_quiet_days": gaps},
    )
    candidates, summary = select_candidates(alerts, scotty_config)
    assert len(candidates) == 1
    assert summary["n_failed_archival_flare"] == 1


def test_color_can_be_optional(scotty_config):
    cfg = scotty_config.model_copy(update={"scotty750": Scotty750Config(require_color=False)})
    alerts = placed_alerts([[host(z=NEAR_Z, sep=0.2)]], [[(100.0, "g", 19.0)]])
    assert len(select_candidates(alerts, cfg)[0]) == 1


def test_lsst_colors_may_come_from_detections_days_apart(scotty_config):
    """LSST rarely takes g and r the same night; a week-apart pair still counts for it."""
    cfg = scotty_config.model_copy(
        update={"scotty750": Scotty750Config(color_max_dt_days_by_survey={"LSST": 7.0})}
    )
    days_apart = [(100.0, "g", 22.0), (104.0, "r", 22.3)]  # g - r = -0.3, 4 days apart
    alerts = placed_alerts(
        [[host(z=NEAR_Z, sep=0.2)], [host(z=NEAR_Z, sep=0.2)]],
        [days_apart, days_apart],
        survey=["LSST", "ZTF"],
    )
    candidates, summary = select_candidates(alerts, cfg)
    assert list(candidates["survey"]) == ["LSST"]
    assert candidates.loc[0, "g_minus_r_dt_days"] == pytest.approx(4.0)
    assert summary["n_failed_color"] == 1  # the ZTF alert keeps the one-day window


def test_a_per_survey_window_must_be_positive():
    with pytest.raises(ValueError):
        Scotty750Config(color_max_dt_days_by_survey={"LSST": 0})


def test_requiring_color_without_a_light_curve_fails_loudly(scotty_config):
    alerts = make_placed({"desi_dr1": [[host(z=NEAR_Z, sep=0.2)]]})
    with pytest.raises(ValueError, match="lc"):
        select_candidates(alerts, scotty_config)


def test_an_unknown_cosmology_is_an_error(scotty_config):
    cfg = scotty_config.model_copy(update={"scotty750": Scotty750Config(cosmology="WMAP1")})
    with pytest.raises(ValueError, match="WMAP1"):
        select_candidates(placed_alerts([[host()]], [BLUE_LC]), cfg)


# --- as a stage --------------------------------------------------------------


def test_the_stage_writes_its_candidates(scotty_config):
    alerts = placed_alerts([[host(z=NEAR_Z, sep=0.2)]], [BLUE_LC])
    inputs = {"distance": StageResult(stage="distance", frame=alerts)}
    result = run_scotty750(scotty_config, inputs=inputs, stamp="20261027T000000Z")
    assert len(result.frame) == 1
    assert result.output_path.name == "candidates_20261027T000000Z.parquet"
    assert len(npd.read_parquet(result.output_path)) == 1


def test_it_is_off_without_its_section(pipeline_config):
    specs = {spec.name: spec for spec in pipeline.stages_for(pipeline_config)}
    assert not specs[scotty750.STAGE].enabled


def test_it_is_on_with_its_section(scotty_config):
    specs = {spec.name: spec for spec in pipeline.stages_for(scotty_config)}
    assert specs[scotty750.STAGE].enabled


def test_its_section_can_switch_it_off(pipeline_config):
    cfg = pipeline_config.model_copy(update={"scotty750": Scotty750Config(enabled=False)})
    specs = {spec.name: spec for spec in pipeline.stages_for(cfg)}
    assert not specs[scotty750.STAGE].enabled
