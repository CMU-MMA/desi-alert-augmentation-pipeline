from pathlib import Path

import lsdb
import nested_pandas as npd
import pytest

from desi_aap import boom
from desi_aap.config import PipelineConfig
from desi_aap.gracedb_cache import GraceDbCache

TEST_DIR = Path(__file__).parent
DATA_DIR_NAME = "data"
DESI_DR1_COSMOS_PATH = "desi_dr1_cosmos"
MOCK_DESI_DR2_COSMOS_PATH = "mock_desi_dr2"
GOLD_STANDARD_ALERTS_PATH = "gold_standard_alerts.parquet"


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
    """A config wired to the COSMOS test catalog and a temp output directory."""
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
        }
    )


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
