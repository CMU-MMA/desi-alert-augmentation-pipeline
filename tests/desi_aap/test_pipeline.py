"""How the pipeline decides which stages run: fan-out, skipping, and start-from."""

import pytest

from desi_aap import boom, pipeline
from desi_aap.config import ConfigError
from desi_aap.stages.base import StageResult
from desi_aap.stages.crossmatch import STAGE as CROSSMATCH_STAGE
from desi_aap.stages.distance import STAGE as DISTANCE_STAGE
from desi_aap.stages.localize import STAGE as LOCALIZE_STAGE
from desi_aap.stages.query import STAGE as QUERY_STAGE
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
    supplied = StageResult(stage=LOCALIZE_STAGE, frame=gold_standard_alerts)
    results = pipeline.run_pipeline(pipeline_config, start=SLACK_STAGE, inputs={LOCALIZE_STAGE: supplied})

    assert QUERY_STAGE not in results
    assert results[LOCALIZE_STAGE] is supplied
    # No [slack] section in this config, so the stage skips without posting, but
    # it still reports what each filter would have contributed.
    assert results[SLACK_STAGE].summary["n_posted"] == 0
    assert results[SLACK_STAGE].summary["rows_by_filter"] == {LOCALIZE_STAGE: len(gold_standard_alerts)}


def test_an_unknown_start_stage_names_the_real_ones(pipeline_config):
    with pytest.raises(ValueError, match=", ".join(pipeline.stage_order(pipeline_config))):
        pipeline.run_pipeline(pipeline_config, start="does_not_exist")


def test_starting_without_the_needed_input_fails_loudly(pipeline_config, broker_must_not_be_called):
    # slack_publish tolerates a filter that did not run, so start at the filter
    # itself: that one does need its input, and says which stage produces it.
    with pytest.raises(KeyError, match=DISTANCE_STAGE):
        pipeline.run_pipeline(pipeline_config, start=LOCALIZE_STAGE)


def test_an_empty_stage_skips_what_needs_it_instead_of_ending_the_run(
    pipeline_config, stub_boom_no_alerts, stub_gracedb
):
    """The fan-out's load-bearing property: nothing halts, everything downstream skips.

    The old pipeline broke out of its loop on the first empty stage, which meant
    a filter finding nothing silenced every filter after it. Now an empty stage
    only skips what depends on it, and the run reaches the end.
    """
    results = pipeline.run_pipeline(pipeline_config)

    # Every stage has a result, including the ones that never ran.
    assert set(results) == set(pipeline.stage_order(pipeline_config))
    assert results[QUERY_STAGE].is_empty
    for stage in (CROSSMATCH_STAGE, DISTANCE_STAGE, LOCALIZE_STAGE, SLACK_STAGE):
        assert results[stage].summary["skipped"] == "no input", stage


def test_a_disabled_filter_is_skipped(pipeline_config, gold_standard_alerts, broker_must_not_be_called):
    """Verify `enabled = false` keeps a filter from running, which is how cadences differ."""
    disabled = pipeline_config.model_copy(
        update={"localize": pipeline_config.localize.model_copy(update={"enabled": False})}
    )
    supplied = StageResult(stage=DISTANCE_STAGE, frame=gold_standard_alerts)
    results = pipeline.run_pipeline(disabled, start=LOCALIZE_STAGE, inputs={DISTANCE_STAGE: supplied})

    assert results[LOCALIZE_STAGE].summary["skipped"] == "disabled"
    # It is the only filter configured, so nothing is left to announce.
    assert results[SLACK_STAGE].summary["skipped"] == "no input"


def test_one_empty_input_does_not_skip_a_stage_that_has_another(gold_standard_alerts):
    """Verify the rule that makes filters siblings: any live input is enough.

    slack_publish requires every filter, and one filter finding nothing must not
    stop it announcing the ones that did. With a single filter configured today
    the pipeline cannot show that end to end, so the rule itself is the subject
    here.
    """
    spec = pipeline.StageSpec("consumer", run=lambda **kwargs: None, requires=("empty", "full"))
    results = {
        "empty": StageResult(stage="empty", frame=None),
        "full": StageResult(stage="full", frame=gold_standard_alerts),
    }

    assert pipeline._has_input(spec, results)
    assert not pipeline._has_input(spec, {"empty": results["empty"]})
    # A required stage with no result at all is not an empty one: the stage runs
    # and raises its own error naming what must run first.
    assert pipeline._has_input(spec, {})


def test_every_stage_requires_one_that_runs_before_it(pipeline_config):
    """The registry is a DAG in the order it is written, not just a list."""
    seen: set[str] = set()
    for spec in pipeline.stages_for(pipeline_config):
        assert set(spec.requires) <= seen, f"{spec.name} requires a stage that has not run"
        seen.add(spec.name)


def test_a_json_filter_becomes_a_stage(pipeline_config):
    """The filter API's contract: drop a file in the directory, get a stage.

    Between distance and slack_publish, requiring the former and required by
    the latter, so its candidates are announced like any code filter's.
    """
    (pipeline_config.filters.dir / "shiny.json").write_text(
        '{"cuts": [{"column": "dist_mpc_SHOES", "max": 100}]}'
    )

    order = pipeline.stage_order(pipeline_config)
    specs = {spec.name: spec for spec in pipeline.stages_for(pipeline_config)}

    assert order.index(DISTANCE_STAGE) < order.index("shiny") < order.index(SLACK_STAGE)
    assert specs["shiny"].requires == (DISTANCE_STAGE,)
    assert "shiny" in specs[SLACK_STAGE].requires
    assert specs["shiny"].enabled


def test_a_disabled_json_filter_is_skipped_like_any_other(
    pipeline_config, gold_standard_alerts, broker_must_not_be_called
):
    """[filters] disabled is the cadence knob for JSON filters, as enabled is for code ones."""
    (pipeline_config.filters.dir / "shiny.json").write_text(
        '{"cuts": [{"column": "dist_mpc_SHOES", "max": 100}]}'
    )
    cfg = pipeline_config.model_copy(
        update={"filters": pipeline_config.filters.model_copy(update={"disabled": ["shiny"]})}
    )
    supplied = StageResult(stage=DISTANCE_STAGE, frame=gold_standard_alerts)

    results = pipeline.run_pipeline(cfg, start="shiny", inputs={DISTANCE_STAGE: supplied})

    assert results["shiny"].summary["skipped"] == "disabled"


def test_the_disabled_list_switches_off_a_code_filter_too(pipeline_config):
    """One knob turns any filter off: [filters] disabled works beyond the JSON ones."""
    cfg = pipeline_config.model_copy(
        update={"filters": pipeline_config.filters.model_copy(update={"disabled": [LOCALIZE_STAGE]})}
    )

    specs = {spec.name: spec for spec in pipeline.stages_for(cfg)}

    assert not specs[LOCALIZE_STAGE].enabled


def test_disabling_a_filter_that_does_not_exist_is_an_error(pipeline_config):
    """A typo in [filters] disabled must not leave the real filter running."""
    cfg = pipeline_config.model_copy(
        update={"filters": pipeline_config.filters.model_copy(update={"disabled": ["lumnious"]})}
    )

    with pytest.raises(ConfigError, match="lumnious"):
        pipeline.stages_for(cfg)
