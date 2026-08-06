def test_read_desi_dr1_cosmos_catalog(desi_dr1_cosmos_catalog):
    assert desi_dr1_cosmos_catalog is not None
    assert len(desi_dr1_cosmos_catalog) > 0
