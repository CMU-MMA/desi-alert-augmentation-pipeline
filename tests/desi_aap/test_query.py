"""The query stage: resolving the window, and what it does with what BOOM returns."""

from datetime import timedelta

import nested_pandas as npd
import pytest
from astropy.time import Time

from desi_aap import pipeline
from desi_aap.stages.query import STAGE, resolve_window, run_query


def test_run_query(pipeline_config, stub_boom, stub_gracedb):
    """Verify the stage writes parquet and hands the same frame to the next stage"""
    result = run_query(pipeline_config, stamp="20260807T120000Z")

    assert result.stage == "query"
    assert result.stamp == "20260807T120000Z"
    assert result.summary["end_jd"] - result.summary["start_jd"] == pytest.approx(1 / 24)
    assert result.summary["n_alerts"] == 327

    assert isinstance(result.frame, npd.NestedFrame)
    assert not result.is_empty
    assert len(result.frame) == 327

    output_path = pipeline_config.run.stage_dir(STAGE) / "alerts_20260807T120000Z.parquet"
    assert result.output_path == output_path
    alerts = npd.read_parquet(output_path)
    assert alerts.equals(result.frame)

    # The results should be the same whether we call the stage directly or run the whole pipeline
    results = pipeline.run_pipeline(pipeline_config, stamp="20260807T120000Z")
    assert results["query"].frame.equals(alerts)


def test_window_defaults_to_a_trailing_lookback():
    """Verify the window ends now and starts `lookback` before it"""
    start_jd, end_jd = resolve_window(None, None, lookback=timedelta(hours=2))

    assert end_jd - start_jd == pytest.approx(2 / 24)
    assert end_jd == pytest.approx(float(Time.now().jd), abs=1e-3)


def test_window_prefers_explicit_bounds():
    """Verify explicit bounds win over the lookback, which is then unused"""
    assert resolve_window(2461187.0, 2461194.0, lookback=timedelta(hours=1)) == (2461187.0, 2461194.0)


def test_window_fills_in_a_missing_start():
    """Verify an explicit end still gets a lookback-wide window before it"""
    start_jd, end_jd = resolve_window(None, 2461194.0, lookback=timedelta(days=7))

    assert end_jd == 2461194.0
    assert start_jd == pytest.approx(2461187.0)


def test_window_rejects_an_inverted_range():
    """Verify a start after the end is an error rather than an empty query"""
    with pytest.raises(ValueError, match="must not be after"):
        resolve_window(2461194.0, 2461187.0, lookback=timedelta(hours=1))


def test_dry_run_writes_nothing_but_still_passes_the_frame(pipeline_config, stub_boom):
    """Verify a dry run builds the frame in memory without touching disk"""
    result = run_query(pipeline_config, dry_run=True, stamp="20260807T120000Z")

    assert len(result.frame) == 327
    assert result.output_path is None
    assert not pipeline_config.run.stage_dir(STAGE).exists()


def test_an_empty_window_produces_no_frame(pipeline_config, stub_boom_no_alerts):
    """Verify a window with no alerts yields nothing to pass on, rather than failing"""
    result = run_query(pipeline_config, stamp="20260807T120000Z")

    assert result.frame is None
    assert result.output_path is None
    assert result.is_empty
    assert result.summary["n_alerts"] == 0
