"""Running the pipeline from a later stage, with supplied stand-in inputs."""

import pytest

from desi_aap import boom, pipeline
from desi_aap.stages.base import StageResult
from desi_aap.stages.query import STAGE as QUERY_STAGE
from desi_aap.stages.slack_publish import INPUT_STAGE as SLACK_INPUT_STAGE
from desi_aap.stages.slack_publish import STAGE as SLACK_STAGE


@pytest.fixture
def broker_must_not_be_called(monkeypatch):
    """Fail the test if anything reaches for the live broker."""

    def _explode(**kwargs):
        raise AssertionError("query stage ran: the broker was called")

    monkeypatch.setattr(boom, "query_alerts", _explode)


def test_run_from_a_stage_skips_the_earlier_ones(
    pipeline_config, gold_standard_alerts, broker_must_not_be_called
):
    supplied = StageResult(stage=SLACK_INPUT_STAGE, frame=gold_standard_alerts)
    results = pipeline.run_pipeline(pipeline_config, start=SLACK_STAGE, inputs={SLACK_INPUT_STAGE: supplied})

    assert QUERY_STAGE not in results
    assert results[SLACK_INPUT_STAGE] is supplied
    # No [slack] section in this config, so the stage skips but still passes the frame through.
    assert results[SLACK_STAGE].summary["posted"] is False
    assert results[SLACK_STAGE].summary["n_rows"] == len(gold_standard_alerts)


def test_an_unknown_start_stage_names_the_real_ones(pipeline_config):
    with pytest.raises(ValueError, match="query, crossmatch, localize, slack_publish"):
        pipeline.run_pipeline(pipeline_config, start="does_not_exist")


def test_starting_without_the_needed_input_fails_loudly(pipeline_config, broker_must_not_be_called):
    with pytest.raises(KeyError, match=SLACK_INPUT_STAGE):
        pipeline.run_pipeline(pipeline_config, start=SLACK_STAGE)
