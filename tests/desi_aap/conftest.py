from pathlib import Path

import lsdb
import nested_pandas as npd
import pytest

TEST_DIR = Path(__file__).parent
DATA_DIR_NAME = "data"
DESI_DR1_COSMOS_PATH = "desi_dr1_cosmos"
MOCK_DESI_DR2_COSMOS_PATH = "mock_desi_dr2"
GOLD_STANDARD_ALERTS_PATH = "gold_standard_alerts.parquet"


@pytest.fixture
def test_data_dir():
    return Path(TEST_DIR) / DATA_DIR_NAME


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
