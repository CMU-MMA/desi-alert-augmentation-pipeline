"""The distance stage reading catalogs laid out differently, and preferring spectra to photo-z."""

import pandas as pd
import pytest
from conftest import TEST_MIN_REDSHIFT, host, make_matches

from desi_aap.config import RedshiftFieldsConfig
from desi_aap.stages.distance import HOST_COLUMNS, nearest_hosts

PHOTO_FIELDS = RedshiftFieldsConfig(
    column="Z",
    error_column="Z_ERR",
    ok_column="Z_OK",
    ok_value=True,
    kind_column="Z_KIND",
    spectype_column="SPECTYPE",
)


def photo_host(z=0.05, sep=0.1, kind="photo", ok=True, err=0.01, spectype=""):
    """One nested row of the DR11 photo-z catalog, as the import script writes it."""
    return {"Z": z, "Z_ERR": err, "Z_KIND": kind, "Z_OK": ok, "SPECTYPE": spectype, "_dist_arcsec": sep}


def hosts_for(matches, preference=0.5):
    return nearest_hosts(
        matches,
        ["desi_dr1", "photoz"],
        min_redshift=TEST_MIN_REDSHIFT,
        fields={"desi_dr1": RedshiftFieldsConfig(), "photoz": PHOTO_FIELDS},
        spec_preference_arcsec=preference,
    )


def test_a_spectrum_of_the_same_galaxy_beats_a_nearer_photo_z():
    """Astrometry alone can put the photo-z row nearer; the spectrum still wins."""
    matches = make_matches(
        {"desi_dr1": [[host(z=0.031, sep=0.35)]], "photoz": [[photo_host(z=0.05, sep=0.10)]]}
    )
    hosts = hosts_for(matches)
    assert hosts.loc[0, "host_catalog"] == "desi_dr1"
    assert hosts.loc[0, "host_redshift"] == pytest.approx(0.031)
    assert hosts.loc[0, "host_redshift_kind"] == "spec"


def test_a_spectrum_of_a_neighbour_does_not_beat_the_host_photo_z():
    """A spectroscopic galaxy arcseconds away is a different galaxy."""
    matches = make_matches(
        {"desi_dr1": [[host(z=0.031, sep=3.0)]], "photoz": [[photo_host(z=0.05, sep=0.10)]]}
    )
    hosts = hosts_for(matches)
    assert hosts.loc[0, "host_catalog"] == "photoz"
    assert hosts.loc[0, "host_redshift_kind"] == "photo"
    assert hosts.loc[0, "host_redshift_err"] == pytest.approx(0.01)


def test_without_a_preference_the_nearest_host_wins():
    matches = make_matches(
        {"desi_dr1": [[host(z=0.031, sep=0.35)]], "photoz": [[photo_host(z=0.05, sep=0.10)]]}
    )
    assert hosts_for(matches, preference=0.0).loc[0, "host_catalog"] == "photoz"


def test_the_kind_can_come_per_row():
    """The photo-z catalog carries DESI spectra too, marked per row."""
    matches = make_matches({"photoz": [[photo_host(z=0.02, kind="spec", spectype="QSO")]]})
    hosts = hosts_for(matches)
    assert hosts.loc[0, "host_redshift_kind"] == "spec"
    assert hosts.loc[0, "host_spectype"] == "QSO"


def test_an_untrustworthy_row_is_not_a_host():
    matches = make_matches({"photoz": [[photo_host(ok=False)]]})
    assert hosts_for(matches).empty


def test_a_kind_other_than_spec_or_photo_is_not_a_host():
    matches = make_matches({"photoz": [[photo_host(kind="")]]})
    assert hosts_for(matches).empty


def test_optional_fields_a_catalog_lacks_are_missing_not_errors():
    """The DESI test rows have no error or spectral class column."""
    matches = make_matches({"desi_dr1": [[host()]]})
    hosts = nearest_hosts(matches, ["desi_dr1"], min_redshift=TEST_MIN_REDSHIFT)
    assert list(hosts.columns) == list(HOST_COLUMNS)
    assert pd.isna(hosts.loc[0, "host_redshift_err"])
    assert pd.isna(hosts.loc[0, "host_spectype"])
    assert hosts.loc[0, "host_redshift_kind"] == "spec"


def test_a_configured_kind_column_the_catalog_lacks_is_an_error():
    matches = make_matches({"photoz": [[{"Z": 0.05, "Z_OK": True, "_dist_arcsec": 0.1}]]})
    with pytest.raises(ValueError, match="Z_KIND"):
        nearest_hosts(matches, ["photoz"], min_redshift=TEST_MIN_REDSHIFT, fields={"photoz": PHOTO_FIELDS})
