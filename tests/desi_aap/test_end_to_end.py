"""The whole pipeline, run end to end on the committed test data.

Every other module tests one stage against inputs built by hand. These run
:func:`desi_aap.pipeline.run_pipeline` itself, from the broker's alerts through
to the message that goes out, and assert what came out of each stage on the way.
That is the only place the wiring is under test: the stage order, what each one
hands the next, and whether a run gets all the way to the end.

Only two things are stubbed, both of them the network: the broker query serves
the committed alert snapshot, and GraceDB serves a one-row superevent table with
a stubbed skymap crossmatch. Everything between them -- the LSDB crossmatch, the
host selection, the cosmology, the nesting, the message building -- is the real
code doing real work.
"""

import nested_pandas as npd
import pytest

from desi_aap import pipeline
from desi_aap.boom import ALERT_BAND_COLUMN, ALERT_BANDS, ALERT_MAG_COLUMN
from desi_aap.stages.crossmatch import STAGE as CROSSMATCH_STAGE
from desi_aap.stages.distance import STAGE as DISTANCE_STAGE
from desi_aap.stages.localize import NESTED_COLUMN, SUPEREVENT_COLUMN
from desi_aap.stages.localize import STAGE as LOCALIZE_STAGE
from desi_aap.stages.query import STAGE as QUERY_STAGE
from desi_aap.stages.slack_publish import STAGE as SLACK_STAGE

STAMP = "20260807T120000Z"

# What the committed snapshot yields at each step, with the fixtures' catalog and
# superevent: 327 alerts from the broker, 8 of them near a DESI source, all 8
# with a usable host redshift. Written down so that a change in the test data or
# in any stage's cuts shows up here as a number that moved, rather than as a
# vaguer "something still came out the end".
N_ALERTS = 327
N_MATCHED = 8


@pytest.fixture
def full_run(slack_config, stub_boom, stub_gracedb, superevent_during_alerts, posted):
    """One complete run, with the superevent moved into the alerts' own window.

    Returns
    -------
    types.SimpleNamespace-like tuple
        The stage results and the messages that went to Slack.
    """
    results = pipeline.run_pipeline(slack_config, stamp=STAMP)
    return results, posted


def test_every_stage_runs_and_produces_rows(full_run):
    """Verify a run with candidates reaches every stage rather than stopping short."""
    results, _ = full_run

    assert list(results) == pipeline.STAGE_ORDER
    for stage in (QUERY_STAGE, CROSSMATCH_STAGE, DISTANCE_STAGE, LOCALIZE_STAGE):
        assert not results[stage].is_empty, f"{stage} produced nothing"
        assert "skipped" not in results[stage].summary, f"{stage} was skipped"


def test_the_counts_narrow_the_way_the_stages_say_they_do(full_run):
    """Verify each stage's summary agrees with the next stage's input."""
    results, _ = full_run

    assert results[QUERY_STAGE].summary["n_alerts"] == N_ALERTS
    assert results[CROSSMATCH_STAGE].summary["n_alerts"] == N_ALERTS
    assert results[CROSSMATCH_STAGE].summary["n_alerts_matched"] == N_MATCHED

    distance = results[DISTANCE_STAGE].summary
    assert distance["n_alerts"] == N_MATCHED
    assert distance["n_alerts_with_host"] + distance["n_alerts_without_host"] == N_MATCHED

    localize = results[LOCALIZE_STAGE].summary
    assert localize["n_alerts"] == distance["n_alerts_with_host"]
    assert localize["n_alerts_coincident"] == len(results[LOCALIZE_STAGE].frame)


def test_each_stage_writes_a_parquet_that_reads_back(full_run, slack_config):
    """Verify the run leaves one readable file per data stage, named for the run."""
    results, _ = full_run

    for stage in (QUERY_STAGE, CROSSMATCH_STAGE, DISTANCE_STAGE, LOCALIZE_STAGE):
        path = results[stage].output_path
        assert path is not None, f"{stage} wrote no file"
        assert path.parent == slack_config.run.output_dir / stage
        assert STAMP in path.name
        assert len(npd.read_parquet(path)) == len(results[stage].frame)


def test_the_coincidences_carry_what_the_filter_measured(full_run):
    """Verify the alert leaves the run an alert, with each kind of match nested on it."""
    results, _ = full_run
    frame = results[LOCALIZE_STAGE].frame

    # The crossmatch stage's nested column, the distance stage's flat columns,
    # and the filter's own -- all on one row, as they were all the way down.
    assert isinstance(frame.dtypes["desi_dr1"], npd.NestedDtype)
    assert frame["host_catalog"].eq("desi_dr1").all()
    assert frame["host_redshift"].gt(0).all()
    assert frame["dist_mpc_SHOES"].gt(0).all()
    assert frame[SUPEREVENT_COLUMN].eq("S190425z").all()
    assert frame[f"{NESTED_COLUMN}.superevent_id"].unique().tolist() == ["S190425z"]


def test_the_band_survives_every_stage_and_reaches_the_message(full_run):
    """Verify the photometric band makes it from the broker to the message intact.

    A magnitude without its band is not a brightness anyone can act on, and the
    band is what the absolute-magnitude filters will be computed per. It is
    projected by default_pipeline.json and then merely carried, so what is under
    test is that no stage drops it: the LSDB round-trip through the crossmatch,
    the joins in the distance stage, and the nesting in the filter.
    """
    results, posted = full_run

    bands = set()
    for stage in (QUERY_STAGE, CROSSMATCH_STAGE, DISTANCE_STAGE, LOCALIZE_STAGE):
        frame = results[stage].frame
        assert ALERT_BAND_COLUMN in frame.columns, f"{stage} dropped the band"
        assert frame[ALERT_BAND_COLUMN].notna().all(), f"{stage} lost band values"
        bands |= set(frame[ALERT_BAND_COLUMN])

    # The committed snapshot really does carry several LSST bands, so this is
    # testing that they are carried rather than that one constant survives.
    assert bands <= set(ALERT_BANDS), f"unexpected band(s): {bands - set(ALERT_BANDS)}"
    assert len(set(results[QUERY_STAGE].frame[ALERT_BAND_COLUMN])) > 1

    # And it reaches a human: the band column is in the posted table's header,
    # beside the magnitude it qualifies.
    (call,) = posted
    header = [cell["text"] for cell in call["blocks"][1]["rows"][0]]
    assert ALERT_MAG_COLUMN in header
    assert ALERT_BAND_COLUMN in header
    assert header.index(ALERT_BAND_COLUMN) == header.index(ALERT_MAG_COLUMN) + 1


def test_the_run_posts_one_message_naming_what_it_found(full_run):
    """Verify the candidates reach Slack, which is the point of the whole run."""
    results, posted = full_run
    n_coincident = len(results[LOCALIZE_STAGE].frame)

    (call,) = posted
    assert call["channel"] == "#desi-alerts"
    assert STAMP in call["text"]
    assert f"{n_coincident} GW coincidence candidate" in call["text"]

    assert results[SLACK_STAGE].summary["n_posted"] == 1
    assert results[SLACK_STAGE].summary["rows_by_filter"] == {LOCALIZE_STAGE: n_coincident}
    # The message points at the file the filter actually wrote.
    context = call["blocks"][-1]
    assert str(results[LOCALIZE_STAGE].output_path) in context["elements"][0]["text"]


def test_a_dry_run_does_the_same_work_and_writes_nothing(
    slack_config, stub_boom, stub_gracedb, superevent_during_alerts, posted
):
    """Verify --dry-run reports the counts a real run would, but leaves no trace."""
    results = pipeline.run_pipeline(slack_config, dry_run=True, stamp=STAMP)

    assert posted == []
    assert not slack_config.run.output_dir.exists()
    for stage in (QUERY_STAGE, CROSSMATCH_STAGE, DISTANCE_STAGE, LOCALIZE_STAGE):
        assert results[stage].output_path is None
        assert not results[stage].is_empty
    assert results[LOCALIZE_STAGE].summary["n_alerts_coincident"] > 0


def test_a_quiet_run_reaches_the_end_without_posting(
    slack_config, stub_boom, stub_gracedb, superevent_during_alerts, posted
):
    """Verify the ordinary hour -- alerts, but no coincidence -- is not an error.

    Most hours look like this. The stages upstream of the filter still run and
    still write, the filter finds nothing, and nothing is announced.
    """
    stub_gracedb.searched_prob_vol = 0.99  # outside the credible volume the config cuts at

    results = pipeline.run_pipeline(slack_config, stamp=STAMP)

    assert posted == []
    assert not results[DISTANCE_STAGE].is_empty
    assert results[DISTANCE_STAGE].output_path.exists()
    assert results[LOCALIZE_STAGE].is_empty
    assert results[LOCALIZE_STAGE].output_path is None
    assert list(results) == pipeline.STAGE_ORDER


def test_a_disabled_filter_leaves_the_earlier_stages_alone(
    slack_config, stub_boom, stub_gracedb, superevent_during_alerts, posted
):
    """Verify the nightly-sweep shape: filters off, everything they depend on still runs.

    This is what makes the cadence split in the README work -- a config that
    switches a filter off still pays for one query and one crossmatch, and hands
    the same placed alerts to whichever filters it does want.
    """
    off = slack_config.model_copy(
        update={"localize": slack_config.localize.model_copy(update={"enabled": False})}
    )

    results = pipeline.run_pipeline(off, stamp=STAMP)

    assert posted == []
    assert results[LOCALIZE_STAGE].summary["skipped"] == "disabled"
    for stage in (QUERY_STAGE, CROSSMATCH_STAGE, DISTANCE_STAGE):
        assert not results[stage].is_empty
        assert results[stage].output_path.exists()
