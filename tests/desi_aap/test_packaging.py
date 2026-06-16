import desi_aap


def test_version():
    """Check to see that we can get the package version"""
    assert desi_aap.__version__ is not None
