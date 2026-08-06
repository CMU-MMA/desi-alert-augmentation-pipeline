from pathlib import Path

import lsdb
import pytest

TEST_DIR = Path(__file__).parent
DATA_DIR_NAME = "data"
DESI_DR1_COSMOS_PATH = "desi_dr1_cosmos"


@pytest.fixture
def test_data_dir():
    return Path(TEST_DIR) / DATA_DIR_NAME


@pytest.fixture
def desi_dr1_cosmos_dir(test_data_dir):
    return test_data_dir / DESI_DR1_COSMOS_PATH


@pytest.fixture
def desi_dr1_cosmos_catalog(desi_dr1_cosmos_dir):
    return lsdb.open_catalog(desi_dr1_cosmos_dir)
