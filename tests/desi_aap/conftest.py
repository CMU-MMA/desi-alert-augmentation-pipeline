from pathlib import Path
from types import SimpleNamespace

import lsdb
import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from desi_aap import boom, gracedb_tools
from desi_aap.config import PipelineConfig, SlackConfig
from desi_aap.gracedb_cache import GraceDbCache
from desi_aap.stages import localize, slack_publish

TEST_DIR = Path(__file__).parent
DATA_DIR_NAME = "data"
DESI_DR1_COSMOS_PATH = "desi_dr1_cosmos"
MOCK_DESI_DR2_COSMOS_PATH = "mock_desi_dr2"
GOLD_STANDARD_ALERTS_PATH = "gold_standard_alerts.parquet"

# GW190425: t_0 in GPS seconds, the UTC merger time it converts to, and the UTC
# Julian date an alert detected at that instant carries. The three are written
# out rather than converted from each other, since those conversions are
# themselves under test.
GW190425_GPS = 1240215503.017147
GW190425_UTC = pd.Timestamp("2019-04-25 08:18:05.017147", tz="UTC")
GW190425_JD = 2458598.845891402

# The redshift floor `pipeline_config` uses, and a host redshift comfortably
# above it that puts an alert inside a plausible GW horizon.
TEST_MIN_REDSHIFT = 0.0002
NEARBY_HOST_Z = 0.03


def host(z=NEARBY_HOST_Z, zwarn=0, sep=1.0):
    """One nested host row, in the fields the distance stage reads."""
    return {"Z": z, "ZWARN": zwarn, "_dist_arcsec": sep}


def make_matches(hosts_by_catalog, *, jd=GW190425_JD, ra=240.0, dec=-20.0, n_alerts=None):
    """Build a crossmatch-stage frame: alerts with one nested column per catalog.

    Here rather than in one test module because it describes the input to the
    distance stage, which both that stage's tests and the filter tests
    downstream of it need to build.

    Parameters
    ----------
    hosts_by_catalog : dict
        Maps a catalog name to a list, one entry per alert, of the host rows that
        catalog matched to it. Each host row is a dict of nested field to value,
        as :func:`host` returns.
    jd, ra, dec : float or sequence
        The alerts' own columns, broadcast when scalar.
    n_alerts : int, optional
        Number of alerts, inferred from the first catalog's list otherwise.

    Returns
    -------
    nested_pandas.NestedFrame
        One row per alert, indexed from zero.
    """
    if n_alerts is None:
        n_alerts = len(next(iter(hosts_by_catalog.values())))
    frame = npd.NestedFrame(
        {
            "objectId": [f"LSST{i:03d}" for i in range(n_alerts)],
            "candidate.ra": np.broadcast_to(np.asarray(ra, dtype=float), (n_alerts,)).copy(),
            "candidate.dec": np.broadcast_to(np.asarray(dec, dtype=float), (n_alerts,)).copy(),
            "candidate.jd": np.broadcast_to(np.asarray(jd, dtype=float), (n_alerts,)).copy(),
        },
        index=range(n_alerts),
    )
    for name, per_alert in hosts_by_catalog.items():
        flat = pd.DataFrame(
            [host_row for hosts in per_alert for host_row in hosts],
            index=[i for i, hosts in enumerate(per_alert) for _ in hosts],
        )
        frame = frame.join_nested(flat, name, how="left")
    return frame


def make_placed(hosts_by_catalog, *, catalog_names=None, **kwargs):
    """Build a distance-stage frame: alerts already put at their hosts' distances.

    Runs the real distance stage over :func:`make_matches` rather than
    hand-writing the placed shape, so that a change to what that stage produces
    reaches the filter tests instead of leaving them asserting against a shape
    nothing produces any more.

    Parameters
    ----------
    hosts_by_catalog : dict
        As :func:`make_matches` takes it.
    catalog_names : sequence of str, optional
        Catalogs to take hosts from. Defaults to every key of
        ``hosts_by_catalog``.
    **kwargs
        Passed to :func:`make_matches`.

    Returns
    -------
    nested_pandas.NestedFrame
        One row per alert with a usable host, carrying the host and distance
        columns every filter reads.
    """
    from desi_aap.stages.distance import attach_distances, nearest_hosts

    matches = make_matches(hosts_by_catalog, **kwargs)
    names = list(hosts_by_catalog) if catalog_names is None else list(catalog_names)
    return attach_distances(matches, nearest_hosts(matches, names, min_redshift=TEST_MIN_REDSHIFT))


@pytest.fixture
def test_data_dir():
    return Path(TEST_DIR) / DATA_DIR_NAME


@pytest.fixture
def superevent_cache(tmp_path):
    """A GraceDB cache rooted under tmp_path, so no test depends on the working directory.

    Named for what it holds rather than for its module, since a fixture in conftest is
    visible to the whole suite and `gracedb_cache` is the module several of these tests
    reach through for `superevent_fingerprint` and friends.
    """
    return GraceDbCache(cache_dir=tmp_path / "gracedb_cache")


@pytest.fixture
def desi_dr1_cosmos_dir(test_data_dir):
    return test_data_dir / DESI_DR1_COSMOS_PATH


@pytest.fixture
def mock_desi_dr2_cosmos_dir(test_data_dir):
    return test_data_dir / MOCK_DESI_DR2_COSMOS_PATH


@pytest.fixture
def desi_dr1_cosmos_catalog(desi_dr1_cosmos_dir):
    """Slice of DESI DR1 catalog in the COSMOS field, for testing"""
    return lsdb.open_catalog(desi_dr1_cosmos_dir)


@pytest.fixture
def mock_desi_dr2_cosmos_catalog(mock_desi_dr2_cosmos_dir):
    """Mock of DESI DR2 catalog in the COSMOS field, for testing"""
    return lsdb.open_catalog(mock_desi_dr2_cosmos_dir)


@pytest.fixture
def gold_standard_alerts(test_data_dir):
    """Snapshot of real BOOM alerts in the COSMOS field, for testing"""
    return npd.read_parquet(test_data_dir / GOLD_STANDARD_ALERTS_PATH)


@pytest.fixture
def pipeline_config(tmp_path, desi_dr1_cosmos_dir):
    """A config wired to the COSMOS test catalog and a temp output directory.

    Spells out every [distance] and [localize] setting rather than leaning on
    the defaults, so that a change to one of those defaults shows up as a test
    failure here rather than silently moving what the suite exercises.
    """
    return PipelineConfig.model_validate(
        {
            "run": {"output_dir": str(tmp_path / "out")},
            "query": {"boom": {"survey": "LSST"}, "window": {"lookback": "1h"}},
            "crossmatch": {
                "catalogs": {
                    "desi_dr1": {
                        "catalog": str(desi_dr1_cosmos_dir),
                        "radius_arcsec": 5.0,
                        "n_neighbors": 1,
                    }
                }
            },
            "distance": {"min_redshift": 0.0002},
            "localize": {
                "enabled": True,
                "se_types": ["BNS", "NSBH"],
                "far_threshold_per_year": 2.0,
                "min_classification_prob_sum": 0.9,
                "window_days": 14.0,
                "credible_level": 0.5,
                "require_2d_credible_level": False,
            },
        }
    )


class FakeSkymap:
    """Stand-in for the table returned by ligo.skymap.io.read_sky_map."""

    def __init__(self, colnames=("UNIQ", "PROBDENSITY", "DISTMU", "DISTSIGMA", "DISTNORM")):
        self.colnames = list(colnames)


@pytest.fixture
def superevent_table(tmp_path):
    """A single-row superevent table, shaped like fetch_gracedb_superevents output.

    Its skymap file exists on disk but holds nothing readable, since every test
    using it reads through the fake in `stub_gracedb`. What matters is that the
    existence check in run_3d_spatial_crossmatch passes.
    """
    skymap_path = tmp_path / "S190425z__bayestar.multiorder.fits"
    skymap_path.write_bytes(b"FITS")
    return pd.DataFrame(
        [
            {
                "superevent_id": "S190425z",
                "gw_time": GW190425_UTC,
                "gps_time": GW190425_GPS,
                "far_per_year": 0.03,
                "p_bns": 0.95,
                "p_nsbh": 0.01,
                "p_bbh": 0.01,
                "p_terrestrial": 0.03,
                "preferred_event": "G330561",
                "pipeline": "gstlal",
                "search": "AllSky",
                "instruments": "H1,L1",
                "skymap_file": "bayestar.multiorder.fits",
                "skymap_path": str(skymap_path),
                "status": "ok",
            }
        ]
    )


@pytest.fixture
def stub_gracedb(monkeypatch, superevent_table):
    """Answer the localize stage from memory instead of GraceDB and ligo.skymap.

    Replaces the three calls that leave the process: the superevent query, the
    skymap read, and the crossmatch itself. Any test that runs the whole
    pipeline needs this, or the localize stage at the end of it will query the
    live service.

    Returns
    -------
    types.SimpleNamespace
        events
            The superevent table served. Edit it to change what the stage sees.
        searched_prob, searched_prob_vol
            The 2D and 3D credible levels every crossmatched SN comes back with.
            Both default to inside the 0.5 contour the config uses; raise one
            above it to make the coincidence cut reject everything.
        fetches
            One (se_types, kwargs) tuple per superevent query the stage made.
    """
    stub = SimpleNamespace(
        events=superevent_table,
        searched_prob=0.1,
        searched_prob_vol=0.2,
        fetches=[],
    )

    def fake_fetch(se_types, **kwargs):
        stub.fetches.append((list(se_types), kwargs))
        return stub.events

    def fake_crossmatch(skymap, coords, contours=None, cosmology=None):
        n = len(coords)
        return SimpleNamespace(
            searched_area=np.full(n, 12.5),
            searched_prob=np.full(n, stub.searched_prob),
            offset=np.full(n, 1.5),
            searched_prob_dist=np.full(n, 0.3),
            searched_vol=np.full(n, 1.0e6),
            searched_prob_vol=np.full(n, stub.searched_prob_vol),
            probdensity_vol=np.full(n, 4.0e-9),
            contour_vols=[2.0e6],
            contour_areas=[300.0],
        )

    monkeypatch.setattr(localize, "fetch_gracedb_superevents", fake_fetch)
    monkeypatch.setattr(gracedb_tools, "read_sky_map", lambda path, moc=False: FakeSkymap())
    monkeypatch.setattr(gracedb_tools, "crossmatch", fake_crossmatch)
    return stub


@pytest.fixture
def stub_boom(monkeypatch, gold_standard_alerts):
    """Serve the committed alert snapshot instead of calling the live broker."""
    monkeypatch.setattr(boom, "query_alerts", lambda **kwargs: gold_standard_alerts)
    return gold_standard_alerts


@pytest.fixture
def stub_boom_no_alerts(monkeypatch):
    """A window the broker returns nothing for."""
    frame = npd.NestedFrame({"objectId": [], "candidate.ra": [], "candidate.dec": []})
    monkeypatch.setattr(boom, "query_alerts", lambda **kwargs: frame)
    return frame


@pytest.fixture
def superevent_during_alerts(stub_gracedb, gold_standard_alerts):
    """Move the stubbed superevent into the alert snapshot's own time window.

    The committed alerts and the GW190425 superevent the other fixtures use are
    seven years apart, so the temporal cut rejects every pair and no run built
    on them can reach a coincidence. That is the right default -- most hours
    really do have no superevent -- but it means the happy path cannot be
    exercised without moving one of the two together.

    The superevent moves rather than the alerts, since the alerts are a real
    snapshot that `test_gold_standard` diffs against the live broker.

    Returns
    -------
    pandas.Timestamp
        The superevent's new time: the midpoint of the alerts' own range, so
        every alert in the snapshot falls inside a `window_days` of it.
    """
    from desi_aap.stages.localize import julian_dates_to_utc

    times = julian_dates_to_utc(gold_standard_alerts[boom.ALERT_TIME_COLUMN])
    midpoint = times.min() + (times.max() - times.min()) / 2
    stub_gracedb.events.loc[0, "gw_time"] = midpoint
    # Kept consistent with gw_time even though the stubbed spatial crossmatch
    # never reads it: it is carried through to the results, where a GPS time
    # from a different decade than the UTC beside it would be a puzzle.
    stub_gracedb.events.loc[0, "gps_time"] = Time(midpoint).gps
    return midpoint


@pytest.fixture
def slack_credentials(tmp_path):
    """A credentials file holding a fake bot token."""
    path = tmp_path / "slack.toml"
    path.write_text('bot_token = "xoxb-test-token"\n')
    return path


@pytest.fixture
def slack_config(pipeline_config, slack_credentials):
    """The shared config, with a [slack] section pointing at the fake credentials."""
    section = SlackConfig(credentials=slack_credentials, channel="#desi-alerts", max_rows=5)
    return pipeline_config.model_copy(update={"slack": section})


@pytest.fixture
def posted(monkeypatch):
    """Capture what would have gone to Slack instead of calling the Web API."""
    calls = []

    class FakeWebClient:
        def __init__(self, token):
            self.token = token

        def chat_postMessage(self, **kwargs):  # noqa: N802 -- the slack_sdk method name
            calls.append({"token": self.token, **kwargs})

    monkeypatch.setattr(slack_publish, "WebClient", FakeWebClient)
    return calls
