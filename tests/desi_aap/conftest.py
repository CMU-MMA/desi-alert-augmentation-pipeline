from pathlib import Path
from types import SimpleNamespace

import lsdb
import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest

from desi_aap import boom, gracedb_tools
from desi_aap.config import PipelineConfig
from desi_aap.gracedb_cache import GraceDbCache
from desi_aap.stages import localize

TEST_DIR = Path(__file__).parent
DATA_DIR_NAME = "data"
DESI_DR1_COSMOS_PATH = "desi_dr1_cosmos"
MOCK_DESI_DR2_COSMOS_PATH = "mock_desi_dr2"
GOLD_STANDARD_ALERTS_PATH = "gold_standard_alerts.parquet"

# GW190425: t_0 in GPS seconds, and the UTC merger time it converts to.
GW190425_GPS = 1240215503.017147
GW190425_UTC = pd.Timestamp("2019-04-25 08:18:05.017147", tz="UTC")


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

    Spells out every [localize] setting rather than leaning on the defaults, so
    that a change to one of those defaults shows up as a test failure here
    rather than silently moving what the suite exercises.
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
            "localize": {
                "se_types": ["BNS", "NSBH"],
                "far_threshold_per_year": 2.0,
                "min_classification_prob_sum": 0.9,
                "window_days": 14.0,
                "credible_level": 0.5,
                "require_2d_credible_level": False,
                "min_redshift": 0.0002,
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
