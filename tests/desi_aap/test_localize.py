"""The localize filter: adapting placed alerts, and nesting the GW results."""

import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from conftest import GW190425_JD, GW190425_UTC, NEARBY_HOST_Z, host, make_matches, make_placed
from pydantic import ValidationError

from desi_aap.config import LocalizeConfig
from desi_aap.cosmology import COSMOLOGIES
from desi_aap.stages.base import StageResult
from desi_aap.stages.distance import STAGE as DISTANCE_STAGE
from desi_aap.stages.localize import (
    ALERT_KEY_COLUMN,
    NESTED_COLUMN,
    SLACK_DISPLAY,
    STAGE,
    SUPEREVENT_COLUMN,
    alerts_to_gw_match_input,
    attach_localizations,
    coincident_localizations,
    julian_dates_to_utc,
    run_localize,
)


@pytest.fixture(name="placed")
def placed_fixture():
    """Two alerts at the time of GW190425, each already put at its host's distance."""
    return make_placed({"desi_dr1": [[host(sep=1.0)], [host(sep=2.0)]]})


@pytest.fixture(name="match_inputs")
def match_inputs_fixture(placed):
    """What the distance stage would have handed this one."""
    return {DISTANCE_STAGE: StageResult(stage=DISTANCE_STAGE, frame=placed)}


def test_julian_dates_to_utc_converts_a_known_time() -> None:
    """Verify an alert's Julian date lands on the UTC instant it names"""
    converted = julian_dates_to_utc(pd.Series([GW190425_JD]))

    assert converted.dt.tz is not None
    assert abs(converted[0] - GW190425_UTC) < pd.Timedelta(seconds=1)


def test_julian_dates_to_utc_passes_through_missing_values() -> None:
    """Verify an unusable Julian date becomes NaT rather than ending the run"""
    converted = julian_dates_to_utc(pd.Series([GW190425_JD, np.nan, None]))

    assert pd.notna(converted[0])
    assert converted[1:].isna().all()


def test_alerts_to_gw_match_input_builds_the_columns_gracedb_tools_reads(placed) -> None:
    """Verify the adapter names its columns the way the crossmatch functions look them up"""
    gw_match_input = alerts_to_gw_match_input(placed)

    for column in ["name", "ra", "declination", "discoverydate", "redshift", ALERT_KEY_COLUMN]:
        assert column in gw_match_input.columns
    for label in COSMOLOGIES:
        assert gw_match_input[f"dist_mpc_{label}"].gt(0).all()
    assert gw_match_input["name"].tolist() == ["LSST000", "LSST001"]
    assert gw_match_input["ra"].tolist() == [240.0, 240.0]
    assert gw_match_input[ALERT_KEY_COLUMN].tolist() == [0, 1]


def test_alerts_to_gw_match_input_needs_the_distance_stages_columns() -> None:
    """Verify a frame that never went through the distance stage says so, not KeyError on a column"""
    unplaced = make_matches({"desi_dr1": [[host()]]})

    with pytest.raises(KeyError, match=DISTANCE_STAGE):
        alerts_to_gw_match_input(unplaced)


def test_alerts_to_gw_match_input_drops_alerts_without_a_usable_time(placed) -> None:
    """Verify an alert with no Julian date is dropped rather than crossmatched at NaT"""
    placed.loc[0, "candidate.jd"] = np.nan

    gw_match_input = alerts_to_gw_match_input(placed)

    assert gw_match_input[ALERT_KEY_COLUMN].tolist() == [1]


def test_coincident_localizations_counts_every_step(placed, superevent_table, stub_gracedb) -> None:
    """Verify the intermediate frames are counted even though only the last is returned"""
    gw_match_input = alerts_to_gw_match_input(placed)

    coincidences, counts = coincident_localizations(
        gw_match_input,
        superevent_table,
        window_days=14.0,
        credible_level=0.5,
        require_2d_credible_level=False,
    )

    assert counts["n_temporal_pairs"] == 2
    assert counts["n_alerts_temporal"] == 2
    assert counts["n_spatial_rows"] == 2 * len(COSMOLOGIES)
    assert counts["n_spatial_failed"] == 0
    assert counts["n_coincidences"] == len(coincidences) == 2 * len(COSMOLOGIES)
    assert counts["n_alerts_coincident"] == 2


def test_coincident_localizations_counts_a_missing_skymap(placed, superevent_table, stub_gracedb) -> None:
    """Verify an event whose skymap never downloaded is reported rather than silently absent"""
    superevent_table.loc[0, "skymap_path"] = None
    gw_match_input = alerts_to_gw_match_input(placed)

    coincidences, counts = coincident_localizations(
        gw_match_input,
        superevent_table,
        window_days=14.0,
        credible_level=0.5,
        require_2d_credible_level=False,
    )

    assert counts["n_temporal_pairs"] == 2
    assert counts["n_spatial_failed"] == 2
    assert counts["n_coincidences"] == 0
    assert coincidences.empty


def test_attach_localizations_keeps_one_row_per_alert(placed, superevent_table, stub_gracedb) -> None:
    """Verify the results nest onto the alert rather than multiplying it"""
    gw_match_input = alerts_to_gw_match_input(placed)
    coincidences, _ = coincident_localizations(
        gw_match_input,
        superevent_table,
        window_days=14.0,
        credible_level=0.5,
        require_2d_credible_level=False,
    )

    frame = attach_localizations(placed, gw_match_input, coincidences)

    assert len(frame) == len(placed)
    assert list(frame[NESTED_COLUMN].array.list_lengths) == [len(COSMOLOGIES)] * 2
    # The alert's own values stay on the row; only the GW results are nested.
    nested_fields = frame[NESTED_COLUMN].nest.columns
    assert "superevent_id" in nested_fields
    assert "sn_dist_mpc" in nested_fields
    assert "ra" not in nested_fields
    assert ALERT_KEY_COLUMN not in nested_fields
    assert frame["host_catalog"].tolist() == ["desi_dr1", "desi_dr1"]
    assert frame["host_redshift"].tolist() == [NEARBY_HOST_Z, NEARBY_HOST_Z]
    assert frame[f"{NESTED_COLUMN}.superevent_id"].unique().tolist() == ["S190425z"]
    # The same events, flattened onto the row so they can be read without unnesting.
    # One entry despite two nested rows per alert, since the cosmologies collapse.
    assert frame[SUPEREVENT_COLUMN].tolist() == ["S190425z", "S190425z"]


def test_every_column_the_slack_message_shows_is_one_the_stage_writes(
    placed, superevent_table, stub_gracedb
) -> None:
    """Verify the filter's declared display columns exist, so none is silently dropped"""
    gw_match_input = alerts_to_gw_match_input(placed)
    coincidences, _ = coincident_localizations(
        gw_match_input,
        superevent_table,
        window_days=14.0,
        credible_level=0.5,
        require_2d_credible_level=False,
    )

    frame = attach_localizations(placed, gw_match_input, coincidences)

    for column in SLACK_DISPLAY.columns:
        assert column in frame.columns


def test_attach_localizations_drops_alerts_with_no_coincidence(
    placed, superevent_table, stub_gracedb
) -> None:
    """Verify only the alerts inside a credible volume survive, keeping their nested columns"""
    gw_match_input = alerts_to_gw_match_input(placed)
    coincidences, _ = coincident_localizations(
        gw_match_input,
        superevent_table,
        window_days=14.0,
        credible_level=0.5,
        require_2d_credible_level=False,
    )
    only_second = coincidences[coincidences[ALERT_KEY_COLUMN] == 1]

    frame = attach_localizations(placed, gw_match_input, only_second)

    assert frame["objectId"].tolist() == ["LSST001"]
    assert "desi_dr1" in frame.columns


def test_attach_localizations_returns_an_empty_frame(placed) -> None:
    """Verify no coincidences yields an empty frame of the right shape, not a raise"""
    frame = attach_localizations(placed, pd.DataFrame(), pd.DataFrame())

    assert frame.empty
    assert "objectId" in frame.columns


def test_run_localize_writes_the_coincidences(pipeline_config, match_inputs, stub_gracedb) -> None:
    """Verify the stage writes one row per coincident alert, and it round-trips through parquet"""
    result = run_localize(pipeline_config, inputs=match_inputs, stamp="20260807T120000Z")

    assert result.summary["n_alerts"] == 2
    assert result.summary["n_alerts_placed"] == 2
    assert result.summary["n_superevents"] == 1
    assert result.summary["n_alerts_coincident"] == 2

    output_path = pipeline_config.run.stage_dir(STAGE) / "coincidences_20260807T120000Z.parquet"
    assert result.output_path == output_path

    written = npd.read_parquet(output_path)
    assert len(written) == 2
    assert list(written[NESTED_COLUMN].array.list_lengths) == [len(COSMOLOGIES)] * 2
    assert written[f"{NESTED_COLUMN}.superevent_id"].unique().tolist() == ["S190425z"]


def test_run_localize_passes_the_configured_cuts_to_gracedb(
    pipeline_config, match_inputs, stub_gracedb
) -> None:
    """Verify the [localize] settings reach the GraceDB query rather than its own defaults"""
    run_localize(pipeline_config, inputs=match_inputs, stamp="20260807T120000Z")

    ((se_types, kwargs),) = stub_gracedb.fetches
    assert se_types == ["BNS", "NSBH"]
    assert kwargs["far_threshold_per_year"] == 2.0
    assert kwargs["min_classification_prob_sum"] == 0.9
    assert kwargs["cache"].cache_dir == pipeline_config.gracedb.cache_dir


def test_run_localize_writes_nothing_when_nothing_is_coincident(
    pipeline_config, match_inputs, stub_gracedb
) -> None:
    """Verify a run whose alerts all fall outside the credible volume produces no file"""
    stub_gracedb.searched_prob_vol = 0.99

    result = run_localize(pipeline_config, inputs=match_inputs, stamp="20260807T120000Z")

    assert result.output_path is None
    assert result.frame.empty
    assert result.is_empty
    assert result.summary["n_spatial_rows"] == 2 * len(COSMOLOGIES)
    assert result.summary["n_coincidences"] == 0


def test_run_localize_dry_run_computes_without_writing(pipeline_config, match_inputs, stub_gracedb) -> None:
    """Verify a dry run reports the counts a real one would"""
    result = run_localize(pipeline_config, dry_run=True, inputs=match_inputs, stamp="20260807T120000Z")

    assert result.output_path is None
    assert not pipeline_config.run.stage_dir(STAGE).exists()
    assert result.summary["n_alerts_coincident"] == 2


def test_run_localize_passes_an_empty_upstream_through(pipeline_config) -> None:
    """Verify a run whose alerts all missed the catalogs is a normal outcome"""
    empty = {DISTANCE_STAGE: StageResult(stage=DISTANCE_STAGE, frame=None)}

    result = run_localize(pipeline_config, inputs=empty, stamp="20260807T120000Z")

    assert result.frame is None
    assert result.output_path is None
    assert result.summary["n_alerts"] == 0


def test_only_the_search_defining_settings_must_be_given(pipeline_config, match_inputs, stub_gracedb) -> None:
    """Verify the confidence gates and the redshift guard default when left out"""
    minimal = pipeline_config.model_copy(
        update={"localize": LocalizeConfig(se_types=["BNS"], window_days=14.0, credible_level=0.5)}
    )

    result = run_localize(minimal, inputs=match_inputs, stamp="20260807T120000Z")

    ((se_types, kwargs),) = stub_gracedb.fetches
    assert se_types == ["BNS"]
    assert kwargs["far_threshold_per_year"] == 2.0
    assert kwargs["min_classification_prob_sum"] == 0.9
    assert result.summary["n_alerts_coincident"] == 2


@pytest.mark.parametrize("omitted", ["se_types", "window_days", "credible_level"])
def test_a_search_defining_setting_has_no_silent_default(omitted) -> None:
    """Verify what the search was cannot be inherited: those three must be written down"""
    settings = {"se_types": ["BNS"], "window_days": 14.0, "credible_level": 0.5}
    settings.pop(omitted)

    with pytest.raises(ValidationError, match=omitted):
        LocalizeConfig(**settings)


def test_run_localize_names_the_stage_that_must_run_first(pipeline_config) -> None:
    """Verify running without the distance stage's result fails loudly"""
    with pytest.raises(KeyError, match=DISTANCE_STAGE):
        run_localize(pipeline_config, inputs=None, stamp="20260807T120000Z")


def test_output_is_named_and_placed_by_stage_and_stamp(pipeline_config, match_inputs, stub_gracedb) -> None:
    """Verify the file lands under <output_dir>/localize/, named for the run"""
    result = run_localize(pipeline_config, inputs=match_inputs, stamp="20260101T000000Z")

    assert result.output_path.parent == pipeline_config.run.output_dir / "localize"
    assert result.output_path.name == "coincidences_20260101T000000Z.parquet"
