"""The localize stage: choosing a host, adapting alerts, and nesting the GW results."""

import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from desi_aap.config import LocalizeConfig
from desi_aap.cosmology import COSMOLOGIES
from desi_aap.stages.base import StageResult
from desi_aap.stages.crossmatch import STAGE as CROSSMATCH_STAGE
from desi_aap.stages.localize import (
    ALERT_KEY_COLUMN,
    NESTED_COLUMN,
    STAGE,
    alerts_to_gw_match_input,
    attach_localizations,
    coincident_localizations,
    julian_dates_to_utc,
    nearest_hosts,
    run_localize,
)

# GW190425's merger, as a UTC timestamp and as the UTC Julian date an alert
# carries. Both written out rather than converted from each other, since the
# conversion between them is what is under test here; the same timestamp is the
# `superevent_table` fixture's gw_time.
GW190425_UTC = pd.Timestamp("2019-04-25 08:18:05.017147", tz="UTC")
GW190425_JD = 2458598.845891402

# The redshift floor the shared config uses, and a host redshift comfortably
# above it that puts the alert inside a plausible GW horizon.
TEST_MIN_REDSHIFT = 0.0002
NEARBY_HOST_Z = 0.03


def make_matches(hosts_by_catalog, *, jd=GW190425_JD, ra=240.0, dec=-20.0, n_alerts=None):
    """Build a crossmatch-stage frame: alerts with one nested column per catalog.

    Parameters
    ----------
    hosts_by_catalog : dict
        Maps a catalog name to a list, one entry per alert, of the host rows that
        catalog matched to it. Each host row is a dict of nested field to value.
    jd, ra, dec : float or sequence
        The alerts' own columns, broadcast when scalar.
    n_alerts : int, optional
        Number of alerts, inferred from the first catalog's list otherwise.
    """
    if n_alerts is None:
        n_alerts = len(next(iter(hosts_by_catalog.values())))
    frame = npd.NestedFrame(
        {
            "objectId": [f"ZTF{i:03d}" for i in range(n_alerts)],
            "candidate.ra": np.broadcast_to(np.asarray(ra, dtype=float), (n_alerts,)).copy(),
            "candidate.dec": np.broadcast_to(np.asarray(dec, dtype=float), (n_alerts,)).copy(),
            "candidate.jd": np.broadcast_to(np.asarray(jd, dtype=float), (n_alerts,)).copy(),
        },
        index=range(n_alerts),
    )
    for name, per_alert in hosts_by_catalog.items():
        flat = pd.DataFrame(
            [host for hosts in per_alert for host in hosts],
            index=[i for i, hosts in enumerate(per_alert) for _ in hosts],
        )
        frame = frame.join_nested(flat, name, how="left")
    return frame


def host(z=NEARBY_HOST_Z, zwarn=0, sep=1.0):
    """One nested host row, in the fields nearest_hosts reads."""
    return {"Z": z, "ZWARN": zwarn, "_dist_arcsec": sep}


@pytest.fixture(name="matches")
def matches_fixture():
    """Two alerts at the time of GW190425, each with one clean nearby host."""
    return make_matches({"desi_dr1": [[host(sep=1.0)], [host(sep=2.0)]]})


@pytest.fixture(name="match_inputs")
def match_inputs_fixture(matches):
    """What the crossmatch stage would have handed this one."""
    return {CROSSMATCH_STAGE: StageResult(stage=CROSSMATCH_STAGE, frame=matches)}


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
    assert list(hosts.columns) == ["host_catalog", "host_redshift", "host_sep_arcsec"]


def test_alerts_to_gw_match_input_builds_the_columns_gracedb_tools_reads(matches) -> None:
    """Verify the adapter names its columns the way the crossmatch functions look them up"""
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    gw_match_input = alerts_to_gw_match_input(matches, hosts)

    for column in ["name", "ra", "declination", "discoverydate", "redshift", ALERT_KEY_COLUMN]:
        assert column in gw_match_input.columns
    for label in COSMOLOGIES:
        assert gw_match_input[f"dist_mpc_{label}"].gt(0).all()
    assert gw_match_input["name"].tolist() == ["ZTF000", "ZTF001"]
    assert gw_match_input["ra"].tolist() == [240.0, 240.0]
    assert gw_match_input[ALERT_KEY_COLUMN].tolist() == [0, 1]


def test_alerts_to_gw_match_input_drops_alerts_without_a_host() -> None:
    """Verify an alert whose hosts were all rejected does not reach the GW match"""
    matches = make_matches({"desi_dr1": [[host()], [host(zwarn=4)]]})
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    gw_match_input = alerts_to_gw_match_input(matches, hosts)

    assert gw_match_input[ALERT_KEY_COLUMN].tolist() == [0]


def test_alerts_to_gw_match_input_drops_alerts_without_a_usable_time(matches) -> None:
    """Verify an alert with no Julian date is dropped rather than crossmatched at NaT"""
    matches.loc[0, "candidate.jd"] = np.nan
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)

    gw_match_input = alerts_to_gw_match_input(matches, hosts)

    assert gw_match_input[ALERT_KEY_COLUMN].tolist() == [1]


def test_coincident_localizations_counts_every_step(matches, superevent_table, stub_gracedb) -> None:
    """Verify the intermediate frames are counted even though only the last is returned"""
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)
    gw_match_input = alerts_to_gw_match_input(matches, hosts)

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


def test_coincident_localizations_counts_a_missing_skymap(matches, superevent_table, stub_gracedb) -> None:
    """Verify an event whose skymap never downloaded is reported rather than silently absent"""
    superevent_table.loc[0, "skymap_path"] = None
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)
    gw_match_input = alerts_to_gw_match_input(matches, hosts)

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


def test_attach_localizations_keeps_one_row_per_alert(matches, superevent_table, stub_gracedb) -> None:
    """Verify the results nest onto the alert rather than multiplying it"""
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)
    gw_match_input = alerts_to_gw_match_input(matches, hosts)
    coincidences, _ = coincident_localizations(
        gw_match_input,
        superevent_table,
        window_days=14.0,
        credible_level=0.5,
        require_2d_credible_level=False,
    )

    frame = attach_localizations(matches, gw_match_input, coincidences)

    assert len(frame) == len(matches)
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


def test_attach_localizations_drops_alerts_with_no_coincidence(
    matches, superevent_table, stub_gracedb
) -> None:
    """Verify only the alerts inside a credible volume survive, keeping their nested columns"""
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)
    gw_match_input = alerts_to_gw_match_input(matches, hosts)
    coincidences, _ = coincident_localizations(
        gw_match_input,
        superevent_table,
        window_days=14.0,
        credible_level=0.5,
        require_2d_credible_level=False,
    )
    only_second = coincidences[coincidences[ALERT_KEY_COLUMN] == 1]

    frame = attach_localizations(matches, gw_match_input, only_second)

    assert frame["objectId"].tolist() == ["ZTF001"]
    assert "desi_dr1" in frame.columns


def test_attach_localizations_returns_an_empty_frame(matches) -> None:
    """Verify no coincidences yields an empty frame of the right shape, not a raise"""
    frame = attach_localizations(matches, pd.DataFrame(), pd.DataFrame())

    assert frame.empty
    assert "objectId" in frame.columns


def test_run_localize_writes_the_coincidences(pipeline_config, match_inputs, stub_gracedb) -> None:
    """Verify the stage writes one row per coincident alert, and it round-trips through parquet"""
    result = run_localize(pipeline_config, inputs=match_inputs, stamp="20260807T120000Z")

    assert result.summary["n_alerts"] == 2
    assert result.summary["n_alerts_with_host"] == 2
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
    empty = {CROSSMATCH_STAGE: StageResult(stage=CROSSMATCH_STAGE, frame=None)}

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
    """Verify running without the crossmatch stage's result fails loudly"""
    with pytest.raises(KeyError, match=CROSSMATCH_STAGE):
        run_localize(pipeline_config, inputs=None, stamp="20260807T120000Z")


def test_output_is_named_and_placed_by_stage_and_stamp(pipeline_config, match_inputs, stub_gracedb) -> None:
    """Verify the file lands under <output_dir>/localize/, named for the run"""
    result = run_localize(pipeline_config, inputs=match_inputs, stamp="20260101T000000Z")

    assert result.output_path.parent == pipeline_config.run.output_dir / "localize"
    assert result.output_path.name == "coincidences_20260101T000000Z.parquet"
