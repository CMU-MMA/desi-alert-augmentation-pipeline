"""The distance stage: choosing each alert's host, and putting it at a distance."""

import logging

import nested_pandas as npd
import numpy as np
import pytest
from conftest import NEARBY_HOST_Z, TEST_MIN_REDSHIFT, host, make_matches

from desi_aap.cosmology import COSMOLOGIES
from desi_aap.stages.base import StageResult
from desi_aap.stages.crossmatch import STAGE as CROSSMATCH_STAGE
from desi_aap.stages.distance import (
    ABS_MAG_COLUMNS,
    DIST_COLUMNS,
    HOST_COLUMNS,
    OUTPUT_PREFIX,
    STAGE,
    abs_mag_column,
    attach_distances,
    dist_column,
    nearest_hosts,
    run_distance,
)


@pytest.fixture(name="matches")
def matches_fixture():
    """Two alerts, each with one clean nearby host."""
    return make_matches({"desi_dr1": [[host(sep=1.0)], [host(sep=2.0)]]})


@pytest.fixture(name="match_inputs")
def match_inputs_fixture(matches):
    """What the crossmatch stage would have handed this one."""
    return {CROSSMATCH_STAGE: StageResult(stage=CROSSMATCH_STAGE, frame=matches)}


def test_nearest_hosts_takes_the_closest_across_catalogs() -> None:
    """Verify the winning host is the nearest one, not the first catalog's"""
    matches = make_matches(
        {
            "desi_dr1": [[host(z=0.10, sep=3.0)]],
            "desi_dr2": [[host(z=0.20, sep=1.0)]],
        }
    )

    hosts = nearest_hosts(matches, ["desi_dr1", "desi_dr2"], min_redshift=TEST_MIN_REDSHIFT)

    assert hosts["host_catalog"].tolist() == ["desi_dr2"]
    assert hosts["host_redshift"].tolist() == [0.20]
    assert hosts["host_sep_arcsec"].tolist() == [1.0]


def test_nearest_hosts_takes_the_closest_neighbour() -> None:
    """Verify a catalog matching several neighbours contributes only its closest"""
    matches = make_matches({"desi_dr1": [[host(z=0.10, sep=4.0), host(z=0.20, sep=0.5)]]})

    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    assert len(hosts) == 1
    assert hosts["host_redshift"].tolist() == [0.20]


def test_nearest_hosts_drops_flagged_and_unusable_redshifts() -> None:
    """Verify a warned fit loses to a clean one, and an alert with only warned fits is dropped"""
    matches = make_matches(
        {
            "desi_dr1": [
                [host(z=0.10, zwarn=4, sep=0.5), host(z=0.20, zwarn=0, sep=2.0)],
                [host(z=0.10, zwarn=4, sep=0.5)],
                [host(z=0.0, zwarn=0, sep=0.5)],
            ]
        }
    )

    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    assert hosts.index.tolist() == [0]
    assert hosts["host_redshift"].tolist() == [0.20]


def test_nearest_hosts_skips_a_catalog_the_frame_does_not_carry() -> None:
    """Verify a configured catalog missing from the frame is not an error"""
    matches = make_matches({"desi_dr1": [[host()]]})

    hosts = nearest_hosts(matches, ["desi_dr1", "desi_dr2"], min_redshift=TEST_MIN_REDSHIFT)

    assert hosts["host_catalog"].tolist() == ["desi_dr1"]


def test_nearest_hosts_rejects_a_catalog_without_a_redshift() -> None:
    """Verify a catalog carrying no redshift fails loudly rather than being skipped"""
    matches = make_matches({"lspsc": [[{"mag_white": 18.0, "_dist_arcsec": 1.0}]]})

    with pytest.raises(ValueError, match="no 'Z'"):
        nearest_hosts(matches, ["lspsc"], min_redshift=TEST_MIN_REDSHIFT)


def test_nearest_hosts_returns_its_columns_when_nothing_qualifies() -> None:
    """Verify an all-flagged frame yields an empty result rather than a shapeless one"""
    matches = make_matches({"desi_dr1": [[host(zwarn=4)]]})

    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    assert hosts.empty
    assert list(hosts.columns) == list(HOST_COLUMNS)


def test_attach_distances_adds_a_column_per_cosmology(matches) -> None:
    """Verify every configured cosmology gets its own distance, all of them positive"""
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    placed = attach_distances(matches, hosts)

    assert set(DIST_COLUMNS) == {dist_column(label) for label in COSMOLOGIES}
    for column in DIST_COLUMNS:
        assert placed[column].gt(0).all()
    # The same redshift under two cosmologies is two different distances; that
    # is why the choice is left to whoever cuts on the number.
    distances = {placed[column].iloc[0] for column in DIST_COLUMNS}
    assert len(distances) == len(DIST_COLUMNS)


def test_attach_distances_keeps_the_alerts_own_columns(matches) -> None:
    """Verify placing an alert leaves it an alert, nested crossmatch column and all"""
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    placed = attach_distances(matches, hosts)

    assert placed["objectId"].tolist() == ["LSST000", "LSST001"]
    assert isinstance(placed.dtypes["desi_dr1"], npd.NestedDtype)
    assert placed["host_redshift"].tolist() == [NEARBY_HOST_Z, NEARBY_HOST_Z]


def test_attach_distances_computes_the_distance_modulus() -> None:
    """Verify M = m - 5(log10 d_Mpc + 5), per cosmology, from the alert's own magnitude."""
    matches = make_matches({"desi_dr1": [[host(sep=1.0)], [host(sep=2.0)]]}, magpsf=[20.0, 22.5])
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    placed = attach_distances(matches, hosts)

    for label in COSMOLOGIES:
        expected = matches["candidate.magpsf"] - 5 * (np.log10(placed[dist_column(label)]) + 5)
        np.testing.assert_allclose(placed[abs_mag_column(label)], expected)


def test_no_magnitude_column_means_no_abs_mag_columns_at_all(matches, caplog) -> None:
    """Verify a frame without magpsf gets no abs_mag columns rather than all-NaN ones.

    All-NaN columns would let a brightness filter run and find nothing forever,
    indistinguishable from a quiet sky; absent columns make the same filter
    fail loudly, naming the column.
    """
    assert "candidate.magpsf" not in matches.columns  # what make_matches builds
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    with caplog.at_level(logging.WARNING):
        placed = attach_distances(matches, hosts)

    for column in ABS_MAG_COLUMNS:
        assert column not in placed.columns
    assert "candidate.magpsf" in caplog.text


def test_attach_distances_drops_alerts_without_a_host() -> None:
    """Verify an alert whose hosts were all rejected leaves here rather than going on unplaced"""
    matches = make_matches({"desi_dr1": [[host()], [host(zwarn=4)]]})
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    placed = attach_distances(matches, hosts)

    assert placed.index.tolist() == [0]


def test_attach_distances_returns_its_columns_when_no_alert_has_a_host() -> None:
    """Verify an empty result still carries the columns the filters read"""
    matches = make_matches({"desi_dr1": [[host(zwarn=4)]]})
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    placed = attach_distances(matches, hosts)

    assert placed.empty
    for column in (*HOST_COLUMNS, *DIST_COLUMNS):
        assert column in placed.columns


def test_run_distance_writes_the_placed_alerts(pipeline_config, match_inputs) -> None:
    """Verify the stage writes one file and reports what it placed"""
    result = run_distance(pipeline_config, inputs=match_inputs, stamp="20260101T000000Z")

    assert result.output_path == (
        pipeline_config.run.stage_dir(STAGE) / f"{OUTPUT_PREFIX}_20260101T000000Z.parquet"
    )
    assert result.output_path.exists()
    assert result.summary["n_alerts"] == 2
    assert result.summary["n_alerts_with_host"] == 2
    assert result.summary["n_alerts_without_host"] == 0
    assert result.summary["hosts_by_catalog"] == {"desi_dr1": 2}


def test_run_distance_counts_the_alerts_it_dropped(pipeline_config) -> None:
    """Verify a run that loses alerts to missing hosts says so rather than quietly shrinking"""
    matches = make_matches({"desi_dr1": [[host()], [host(zwarn=4)]]})
    inputs = {CROSSMATCH_STAGE: StageResult(stage=CROSSMATCH_STAGE, frame=matches)}

    result = run_distance(pipeline_config, inputs=inputs)

    assert result.summary["n_alerts"] == 2
    assert result.summary["n_alerts_with_host"] == 1
    assert result.summary["n_alerts_without_host"] == 1


def test_run_distance_writes_nothing_on_a_dry_run(pipeline_config, match_inputs) -> None:
    """Verify a dry run reports the same summary but leaves no file"""
    result = run_distance(pipeline_config, dry_run=True, inputs=match_inputs)

    assert result.output_path is None
    assert result.summary["n_alerts_with_host"] == 2
    assert not result.is_empty


def test_run_distance_passes_an_empty_upstream_through(pipeline_config) -> None:
    """Verify an empty crossmatch yields an empty result rather than raising"""
    inputs = {CROSSMATCH_STAGE: StageResult(stage=CROSSMATCH_STAGE, frame=None)}

    result = run_distance(pipeline_config, inputs=inputs)

    assert result.is_empty
    assert result.summary == {"n_alerts": 0}


def test_run_distance_names_the_stage_that_must_run_first(pipeline_config) -> None:
    """Verify a missing upstream says which stage produces it"""
    with pytest.raises(KeyError, match=CROSSMATCH_STAGE):
        run_distance(pipeline_config, inputs={})
