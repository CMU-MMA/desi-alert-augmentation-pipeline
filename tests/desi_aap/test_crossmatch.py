"""The crossmatch stage: opening catalogs, matching alerts, and filtering the misses."""

import nested_pandas as npd
import pytest

from desi_aap import pipeline
from desi_aap.stages.base import StageResult
from desi_aap.stages.crossmatch import (
    STAGE,
    CatalogSpec,
    alerts_to_catalog,
    catalog_specs,
    crossmatch_catalog,
    run_crossmatch,
    summarize_matches,
)
from desi_aap.stages.query import STAGE as QUERY_STAGE


@pytest.fixture
def alert_inputs(gold_standard_alerts):
    """What the query stage would have handed this one."""
    return {QUERY_STAGE: StageResult(stage=QUERY_STAGE, frame=gold_standard_alerts)}


@pytest.fixture
def unmatched_alerts(gold_standard_alerts):
    """Real alerts from the far side of the sky, none of them near the COSMOS field."""
    frame = gold_standard_alerts[gold_standard_alerts["candidate.ra"] > 200].reset_index(drop=True)
    assert not frame.empty, "expected the snapshot to hold alerts away from COSMOS"
    return {QUERY_STAGE: StageResult(stage=QUERY_STAGE, frame=frame)}


def test_run_crossmatch(pipeline_config, alert_inputs, stub_boom):
    """Verify unmatched alerts are filtered out, so the filter does real work"""
    result = run_crossmatch(pipeline_config, inputs=alert_inputs, stamp="20260807T120000Z")

    assert result.summary["n_alerts"] == 327
    assert result.summary["n_alerts_matched"] == 8
    assert result.summary["n_matches_desi_dr1"] == 8

    output_path = pipeline_config.run.stage_dir(STAGE) / "matches_20260807T120000Z.parquet"
    assert result.output_path == output_path

    matches = npd.read_parquet(output_path)
    assert len(matches) == 8
    assert "desi_dr1" in matches.columns
    assert all(matches["desi_dr1"].array.list_lengths == 1)
    assert all(matches["desi_dr1._dist_arcsec"] <= 5.0)

    # The results should be the same whether we call the stage directly or run the whole pipeline
    results = pipeline.run_pipeline(pipeline_config, stamp="20260807T120000Z")
    assert results["crossmatch"].frame.equals(matches)


def test_catalog_specs_come_from_the_config(pipeline_config):
    """Verify each configured table becomes one spec, named after the table"""
    (spec,) = catalog_specs(pipeline_config)

    assert spec.name == "desi_dr1"
    assert spec.radius_arcsec == 5.0
    assert spec.n_neighbors == 1


def test_configuring_no_catalogs_is_an_error(pipeline_config):
    """Verify the stage refuses to run with nothing to match against"""
    empty = pipeline_config.crossmatch.model_copy(update={"catalogs": {}})
    bare = pipeline_config.model_copy(update={"crossmatch": empty})

    with pytest.raises(ValueError, match=r"\[crossmatch\.catalogs"):
        catalog_specs(bare)


def test_duplicate_catalog_names_are_rejected(gold_standard_alerts, desi_dr1_cosmos_dir):
    """Verify two specs sharing a name fail before any matching work starts"""
    spec = CatalogSpec("desi_dr1", desi_dr1_cosmos_dir, radius_arcsec=5.0, n_neighbors=1)

    with pytest.raises(ValueError, match="Duplicate catalog name"):
        crossmatch_catalog(alerts_to_catalog(gold_standard_alerts), [spec, spec])


def test_matching_against_no_catalogs_is_rejected(gold_standard_alerts):
    """Verify an empty spec list fails before any matching work starts"""
    with pytest.raises(ValueError, match="No catalogs given"):
        crossmatch_catalog(alerts_to_catalog(gold_standard_alerts), [])


def test_dry_run_computes_the_summary_without_writing(pipeline_config, alert_inputs):
    """Verify a dry run still reports the counts a real run would"""
    result = run_crossmatch(pipeline_config, dry_run=True, inputs=alert_inputs, stamp="20260807T120000Z")

    assert result.output_path is None
    assert not pipeline_config.run.stage_dir(STAGE).exists()
    assert result.summary["n_alerts_matched"] == 8


def test_no_alert_matching_writes_nothing(pipeline_config, unmatched_alerts):
    """Verify a window whose alerts all miss produces no output file"""
    result = run_crossmatch(pipeline_config, inputs=unmatched_alerts, stamp="20260807T120000Z")

    assert result.output_path is None
    assert result.frame.empty
    assert result.is_empty


def test_an_empty_upstream_is_passed_through(pipeline_config):
    """Verify a window with no alerts is a normal outcome, not a missing input"""
    empty = {QUERY_STAGE: StageResult(stage=QUERY_STAGE, frame=None)}
    result = run_crossmatch(pipeline_config, inputs=empty, stamp="20260807T120000Z")

    assert result.frame is None
    assert result.output_path is None
    assert result.summary["n_alerts"] == 0


def test_a_missing_upstream_names_the_stage_that_must_run_first(pipeline_config):
    """Verify running without the query stage's result fails loudly"""
    with pytest.raises(KeyError, match=QUERY_STAGE):
        run_crossmatch(pipeline_config, inputs=None, stamp="20260807T120000Z")


def test_output_is_named_and_placed_by_stage_and_stamp(pipeline_config, alert_inputs):
    """Verify the file lands under <output_dir>/crossmatch/, named for the run"""
    result = run_crossmatch(pipeline_config, inputs=alert_inputs, stamp="20260101T000000Z")

    assert result.output_path.parent == pipeline_config.run.output_dir / "crossmatch"
    assert result.output_path.name == "matches_20260101T000000Z.parquet"


def test_summarize_reports_zero_when_nothing_matched(gold_standard_alerts):
    """Verify the summary shape holds even with no matches to report"""
    empty = npd.NestedFrame({"objectId": []})
    summary = summarize_matches(empty, gold_standard_alerts, ["desi_dr1"])

    assert summary["n_alerts"] == 327
    assert summary["n_alerts_matched"] == 0
    assert summary["n_matches_desi_dr1"] == 0
