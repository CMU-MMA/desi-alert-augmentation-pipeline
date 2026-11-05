"""Querying several surveys in one run, and the shipped SCOTTY750 config and BOOM pipelines."""

import json
from datetime import timedelta
from pathlib import Path

import nested_pandas as npd
import pytest

from desi_aap import boom
from desi_aap.config import BoomConfig, ConfigError, load_config
from desi_aap.stages.query import query_surveys

REPO = Path(__file__).resolve().parents[2]


def survey_frame(survey, n):
    """What a survey's pipeline might return: its own magnitude column, and a light curve."""
    mag = "candidate.magap" if survey == "DECAM" else "candidate.magpsf"
    records = [
        {
            "objectId": f"{survey}{i}",
            "candidate": {"ra": 10.0 + i, "dec": -5.0, "jd": 2461000.0 + i, mag.split(".")[1]: 20.0},
            "lc": [{"jd": 2461000.0, "band": "g", "mag": 20.0}],
        }
        for i in range(n)
    ]
    return boom.to_nested_frame(records)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_query_alerts(**kwargs):
        seen.append(kwargs)
        return survey_frame(kwargs["survey"], {"ZTF": 2, "LSST": 0, "DECAM": 1}[kwargs["survey"]])

    monkeypatch.setattr(boom, "query_alerts", fake_query_alerts)
    return seen


@pytest.fixture
def multi_config(pipeline_config, tmp_path):
    pipe = tmp_path / "ztf.json"
    pipe.write_text(json.dumps([{"$match": {}}, {"$project": {"objectId": 1}}]))
    boom_cfg = BoomConfig(
        surveys=["ZTF", "LSST", "DECAM"],
        pipelines={"ZTF": pipe},
        permissions={"ZTF": [1, 2]},
        renames={"DECAM": {"candidate.magap": "candidate.magpsf"}},
    )
    query = pipeline_config.query.model_copy(update={"boom": boom_cfg})
    return pipeline_config.model_copy(update={"query": query})


def test_every_survey_is_asked_with_its_own_settings(multi_config, calls):
    query_surveys(multi_config, 1.0, 2.0)
    by_survey = {call["survey"]: call for call in calls}
    assert list(by_survey) == ["ZTF", "LSST", "DECAM"]
    assert by_survey["ZTF"]["pipeline"] == [{"$match": {}}, {"$project": {"objectId": 1}}]
    assert by_survey["ZTF"]["permissions"] == {"ZTF": [1, 2]}
    assert "pipeline" not in by_survey["LSST"]
    assert "permissions" not in by_survey["DECAM"]


def test_the_alerts_are_combined_tagged_and_renamed(multi_config, calls):
    alerts, counts = query_surveys(multi_config, 1.0, 2.0)
    assert counts == {"ZTF": 2, "LSST": 0, "DECAM": 1}
    assert isinstance(alerts, npd.NestedFrame)
    assert list(alerts[boom.ALERT_SURVEY_COLUMN]) == ["ZTF", "ZTF", "DECAM"]
    assert "candidate.magap" not in alerts.columns
    assert alerts[boom.ALERT_MAG_COLUMN].notna().all()
    assert len(alerts["lc"].nest.to_flat()) == 3


def test_a_rename_onto_a_returned_column_is_an_error(multi_config, monkeypatch):
    def both(**kwargs):
        return boom.to_nested_frame([{"objectId": "a", "candidate": {"magap": 1.0, "magpsf": 2.0}}])

    monkeypatch.setattr(boom, "query_alerts", both)
    with pytest.raises(ValueError, match="overwrite"):
        query_surveys(multi_config, 1.0, 2.0)


def test_surveys_wins_over_survey():
    assert BoomConfig(survey="LSST", surveys=["ZTF", "DECAM"]).survey_list == ["ZTF", "DECAM"]
    assert BoomConfig(survey="LSST").survey_list == ["LSST"]


def test_settings_for_a_survey_not_queried_are_an_error():
    with pytest.raises(ValueError, match="DECAM"):
        BoomConfig(surveys=["ZTF"], pipelines={"DECAM": "x.json"})


def test_some_survey_is_required():
    with pytest.raises(ValueError, match="survey"):
        BoomConfig()


# --- the shipped files -----------------------------------------------------


def test_the_shipped_overlay_validates_over_the_main_config():
    cfg = load_config([REPO / "config.toml", REPO / "scotty750.toml"])
    assert cfg.query.boom.survey_list == ["ZTF", "LSST", "DECAM"]
    assert cfg.fritz.group_ids == [2064]
    assert cfg.fritz.filters == ["scotty750"]
    assert not cfg.localize.enabled
    assert cfg.crossmatch.catalogs["dr11_photoz"].redshift.kind_column == "Z_KIND"
    assert cfg.scotty750.color_max_dt_days_by_survey == {"LSST": 7.0}
    # The overlay narrows the DESI columns without losing the catalog paths.
    assert cfg.crossmatch.catalogs["desi_dr2"].catalog.name == "desi_dr2_zcat"
    for survey, path in cfg.query.boom.pipelines.items():
        assert (REPO / path).exists(), f"{survey} pipeline {path} is missing"


def test_the_nightly_sweep_fits_in_one_boom_query():
    """BOOM rejects a window over 7.0 days; the sweep must stay inside it."""
    cfg = load_config([REPO / "config.toml", REPO / "scotty750.toml", REPO / "scotty750_nightly.toml"])
    assert timedelta(days=6) < cfg.query.window.lookback < timedelta(days=7)
    assert cfg.fritz.filters == ["scotty750"]


@pytest.mark.parametrize("survey", ["ztf", "lsst", "decam"])
def test_the_shipped_pipelines_follow_booms_rules(survey):
    """BOOM rejects a pipeline without a $match, or not ending in a $project keeping objectId."""
    pipe = boom.load_pipeline(REPO / "pipelines" / "scotty750" / f"{survey}.json")
    stages = [next(iter(stage)) for stage in pipe]
    assert "$match" in stages
    assert not {"$group", "$unwind", "$lookup", "$graphLookup", "$unionWith"} & set(stages)
    last = pipe[-1]["$project"]
    assert last["objectId"] == 1
    # What the SCOTTY750 screen and the distance stage read.
    for column in (
        "candidate.ra",
        "candidate.dec",
        "candidate.jd",
        "candidate.band",
        "lc",
        "nuclear_sep_arcsec",
    ):
        assert column in last, column
    assert "properties.episode_history.longest_quiet_days" in last


def test_a_pipeline_file_must_be_a_list(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="list"):
        boom.load_pipeline(bad)


def test_a_misnamed_overlay_section_is_a_config_error(tmp_path):
    overlay = tmp_path / "bad.toml"
    overlay.write_text(
        '[fritz]\ngroup_ids = [2064]\nfilters = ["scotty750"]\nannotation_origin = "has space"\n'
    )
    with pytest.raises(ConfigError, match="annotation_origin"):
        load_config([REPO / "config.toml", overlay])
