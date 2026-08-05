import json

import numpy as np
import pandas as pd
import pytest
import requests
from astropy import units as u
from desi_aap import tns_catalog
from desi_aap.cosmology import COSMOLOGIES

# Redshifts chosen against MAX_STRIPPED_ENVELOPE_DISTANCE_MPC = 500. The middle one is
# inside the cut under SHOES and outside it under Planck18, which is the band that proves
# the cut is applied per cosmology rather than to a single distance.
NEARBY_Z = 0.1
SPLIT_Z = 0.105
DISTANT_Z = 0.15

KEPT_COLUMNS = [
    "name",
    "ra",
    "declination",
    "redshift",
    "type",
    "discoverydate",
    "reporting_group",
    "internal_names",
]


def make_tns_frame(rows):
    """Build a raw TNS-shaped catalog from partial row dicts, filling in valid defaults.

    Parameters
    ----------
    rows : list of dict
        Overrides applied over a nearby SN Ib row. Any key may be overridden, including
        with the unparseable strings that TNS leaves in its CSV.

    Returns
    -------
    pandas.DataFrame
        A frame with every column clean_tns_catalog reads, plus objid and sndec, which it
        is expected to drop.
    """
    defaults = {
        "objid": 12345,
        "name": "2019ebq",
        "ra": "255.326411",
        "declination": "-7.002923",
        "redshift": NEARBY_Z,
        "type": "SN Ib",
        "discoverydate": "2019-04-25 12:11:31.000",
        "reporting_group": "ZTF",
        "internal_names": "ZTF19aatlmbo",
        "sndec": -7.002923,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def luminosity_distance(redshift, label):
    """Return the luminosity distance in Mpc that a COSMOLOGIES entry gives a redshift."""
    return COSMOLOGIES[label].luminosity_distance(redshift).to_value(u.Mpc)


def test_clean_tns_catalog_keeps_stripped_envelope_types() -> None:
    """Verify `clean_tns_catalog` keeps only types matching STRIPPED_ENVELOPE_TYPE_REGEX"""
    df = make_tns_frame(
        [
            {"name": "ib", "type": "SN Ib"},
            {"name": "ic", "type": "SN Ic"},
            {"name": "iib", "type": "SN IIb"},
            {"name": "ic-bl", "type": "SN Ic-BL"},
            {"name": "ibn", "type": "SN Ibn"},
            {"name": "ia", "type": "SN Ia"},
            {"name": "ii", "type": "SN II"},
            {"name": "iin", "type": "SN IIn"},
        ]
    )

    cleaned = tns_catalog.clean_tns_catalog(df)

    assert cleaned["name"].tolist() == ["ib", "ic", "iib", "ic-bl", "ibn"]


def test_clean_tns_catalog_drops_untyped_objects() -> None:
    """Verify `clean_tns_catalog` drops objects TNS has not classified"""
    df = make_tns_frame([{"name": "typed"}, {"name": "untyped", "type": np.nan}])

    assert tns_catalog.clean_tns_catalog(df)["name"].tolist() == ["typed"]


def test_clean_tns_catalog_matches_types_case_sensitively() -> None:
    """Verify `clean_tns_catalog` matches the regex as written, so lowercase types are dropped"""
    df = make_tns_frame([{"name": "upper", "type": "SN Ib"}, {"name": "lower", "type": "sn ib"}])

    assert tns_catalog.clean_tns_catalog(df)["name"].tolist() == ["upper"]


def test_clean_tns_catalog_keeps_superluminous_ic() -> None:
    """Verify `clean_tns_catalog` currently keeps SLSN-I types, which the regex matches as a substring

    STRIPPED_ENVELOPE_TYPE_REGEX carries an open TODO about exactly this: "SLSN-Ic" and
    "SLSN-Ib" contain "Ic" and "Ib", so superluminous supernovae come through alongside the
    ordinary stripped-envelope ones. "SLSN-II" does not match and is dropped.
    """
    df = make_tns_frame(
        [
            {"name": "slsn-ic", "type": "SLSN-Ic"},
            {"name": "slsn-i", "type": "SLSN-I"},
            {"name": "slsn-ii", "type": "SLSN-II"},
        ]
    )

    assert tns_catalog.clean_tns_catalog(df)["name"].tolist() == ["slsn-ic"]


def test_clean_tns_catalog_keeps_only_the_expected_columns() -> None:
    """Verify `clean_tns_catalog` drops every column it does not use and adds one per cosmology"""
    cleaned = tns_catalog.clean_tns_catalog(make_tns_frame([{}]))

    expected = KEPT_COLUMNS + [f"dist_mpc_{label}" for label in COSMOLOGIES]
    assert cleaned.columns.tolist() == expected


def test_clean_tns_catalog_parses_the_string_columns() -> None:
    """Verify `clean_tns_catalog` coerces the TNS strings to numbers and a tz-aware UTC time"""
    cleaned = tns_catalog.clean_tns_catalog(make_tns_frame([{}]))

    row = cleaned.iloc[0]
    assert row["ra"] == pytest.approx(255.326411)
    assert row["declination"] == pytest.approx(-7.002923)
    assert row["redshift"] == pytest.approx(NEARBY_Z)
    assert row["discoverydate"] == pd.Timestamp("2019-04-25 12:11:31", tz="UTC")
    assert str(cleaned["discoverydate"].dt.tz) == "UTC"


def test_clean_tns_catalog_adds_a_distance_per_cosmology() -> None:
    """Verify `clean_tns_catalog` records each cosmology's luminosity distance in Mpc"""
    cleaned = tns_catalog.clean_tns_catalog(make_tns_frame([{}]))

    for label in COSMOLOGIES:
        assert cleaned[f"dist_mpc_{label}"].iloc[0] == pytest.approx(luminosity_distance(NEARBY_Z, label))


@pytest.mark.parametrize(
    "column,value",
    [
        ("ra", "not a number"),
        ("declination", ""),
        ("redshift", "unknown"),
        ("discoverydate", "0000-00-00"),
        ("ra", np.nan),
        ("redshift", None),
    ],
)
def test_clean_tns_catalog_drops_unparseable_rows(column, value) -> None:
    """Verify `clean_tns_catalog` drops a row when any required field fails to parse"""
    df = make_tns_frame([{"name": "good"}, {"name": "bad", column: value}])

    assert tns_catalog.clean_tns_catalog(df)["name"].tolist() == ["good"]


def test_clean_tns_catalog_applies_the_distance_cut() -> None:
    """Verify `clean_tns_catalog` keeps objects inside MAX_STRIPPED_ENVELOPE_DISTANCE_MPC"""
    df = make_tns_frame(
        [
            {"name": "near", "redshift": NEARBY_Z},
            {"name": "far", "redshift": DISTANT_Z},
        ]
    )

    cleaned = tns_catalog.clean_tns_catalog(df)

    assert cleaned["name"].tolist() == ["near"]
    for label in COSMOLOGIES:
        assert luminosity_distance(NEARBY_Z, label) < tns_catalog.MAX_STRIPPED_ENVELOPE_DISTANCE_MPC
        assert luminosity_distance(DISTANT_Z, label) > tns_catalog.MAX_STRIPPED_ENVELOPE_DISTANCE_MPC


def test_clean_tns_catalog_keeps_objects_near_under_any_cosmology() -> None:
    """Verify `clean_tns_catalog` keeps an object inside the cut under only one cosmology"""
    max_mpc = tns_catalog.MAX_STRIPPED_ENVELOPE_DISTANCE_MPC
    assert luminosity_distance(SPLIT_Z, "SHOES") < max_mpc <= luminosity_distance(SPLIT_Z, "Planck18")

    cleaned = tns_catalog.clean_tns_catalog(make_tns_frame([{"name": "split", "redshift": SPLIT_Z}]))

    assert cleaned["name"].tolist() == ["split"]


def test_clean_tns_catalog_resets_the_index() -> None:
    """Verify `clean_tns_catalog` returns a contiguous index after dropping rows"""
    df = make_tns_frame(
        [
            {"name": "ia", "type": "SN Ia"},
            {"name": "ib", "type": "SN Ib"},
            {"name": "far", "redshift": DISTANT_Z},
            {"name": "ic", "type": "SN Ic"},
        ]
    )

    cleaned = tns_catalog.clean_tns_catalog(df)

    assert cleaned.index.tolist() == [0, 1]
    assert cleaned["name"].tolist() == ["ib", "ic"]


def test_clean_tns_catalog_returns_empty_frames() -> None:
    """Verify `clean_tns_catalog` survives an input where nothing is left to keep"""
    empty = tns_catalog.clean_tns_catalog(make_tns_frame([{"type": "SN Ia"}]))
    assert empty.empty
    assert f"dist_mpc_{next(iter(COSMOLOGIES))}" in empty.columns

    no_rows = tns_catalog.clean_tns_catalog(make_tns_frame([{}]).iloc[:0])
    assert no_rows.empty


def test_clean_tns_catalog_does_not_mutate_its_input() -> None:
    """Verify `clean_tns_catalog` leaves the caller's raw catalog untouched"""
    df = make_tns_frame([{}])
    before = df.copy()

    tns_catalog.clean_tns_catalog(df)

    pd.testing.assert_frame_equal(df, before)


def test_clean_tns_catalog_requires_its_columns() -> None:
    """Verify `clean_tns_catalog` raises when the raw catalog is missing a column it reads"""
    df = make_tns_frame([{}]).drop(columns=["internal_names"])

    with pytest.raises(KeyError):
        tns_catalog.clean_tns_catalog(df)


def test_clean_tns_catalog_keeps_unmeasured_redshifts() -> None:
    """Verify `clean_tns_catalog` passes z = 0 through, as its Notes describe

    TNS carries z = 0 where no redshift was measured. Neither the dropna nor the distance
    cut removes it, so the object survives at 0 Mpc, and its credible level downstream is
    meaningless rather than absent. Documented as an open question for Xander.
    """
    cleaned = tns_catalog.clean_tns_catalog(make_tns_frame([{"name": "no-z", "redshift": 0.0}]))

    assert cleaned["name"].tolist() == ["no-z"]
    for label in COSMOLOGIES:
        assert cleaned[f"dist_mpc_{label}"].iloc[0] == 0.0


def test_clean_tns_catalog_keeps_blueshifted_hosts() -> None:
    """Verify `clean_tns_catalog` passes a small negative z through as a negative distance

    The other half of the same open question. A negative distance clears the cut here and
    then raises ValueError out of SkyCoord in run_3d_spatial_crossmatch.
    """
    cleaned = tns_catalog.clean_tns_catalog(make_tns_frame([{"name": "blue", "redshift": -0.001}]))

    assert cleaned["name"].tolist() == ["blue"]
    for label in COSMOLOGIES:
        assert cleaned[f"dist_mpc_{label}"].iloc[0] < 0


def test_clean_tns_catalog_never_returns_a_deeply_negative_redshift() -> None:
    """Verify a z below -1 never comes back as an ordinary row

    How it fails is astropy-version-dependent, so this asserts only that it does. On
    astropy 7.1.1 Planck18's integrand goes complex below z = -1 and luminosity_distance
    raises TypeError, aborting the whole call; on astropy 5.3 it returns NaN instead, and
    the row is then removed by the distance cut, since NaN < MAX compares False. Either is
    acceptable. Silently returning a row with a plausible-looking distance is not, which is
    the real risk: SHOES gives z = -1.2 a finite 1132 Mpc on both versions.
    """
    try:
        cleaned = tns_catalog.clean_tns_catalog(make_tns_frame([{"redshift": -1.2}]))
    except (TypeError, ZeroDivisionError):
        return
    assert cleaned.empty


CREDENTIAL_ENV_NAMES = (
    tns_catalog.TNS_API_KEY_ENV,
    tns_catalog.TNS_BOT_ID_ENV,
    tns_catalog.TNS_BOT_NAME_ENV,
)


class FakeResponse:
    """Stand-in for the requests.Response that requests.post returns."""

    def __init__(self, content=b"zip-bytes", error=None):
        self.content = content
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        """Raise the configured error, mirroring a 4xx/5xx from TNS."""
        if self.error is not None:
            raise self.error


@pytest.fixture
def tns_env(monkeypatch):
    """Set the three TNS credential variables to known test values.

    Also isolates the tests from a developer's real exported credentials, so a machine with
    a live bot configured behaves the same as CI.
    """
    monkeypatch.setenv(tns_catalog.TNS_API_KEY_ENV, "test-key")
    monkeypatch.setenv(tns_catalog.TNS_BOT_ID_ENV, "424242")
    monkeypatch.setenv(tns_catalog.TNS_BOT_NAME_ENV, "TestBot")


def install_fake_post(monkeypatch, response=None):
    """Point requests.post at a fake and record the arguments download_tns_table passes."""
    calls = []

    def fake_post(url, headers=None, data=None):
        calls.append({"url": url, "headers": headers, "data": data})
        return FakeResponse() if response is None else response

    monkeypatch.setattr(tns_catalog.requests, "post", fake_post)
    return calls


def test_tns_credentials_reads_the_environment(tns_env) -> None:
    """Verify `tns_credentials` returns the three variables, with the bot id as a string"""
    assert tns_catalog.tns_credentials() == ("test-key", "424242", "TestBot")


@pytest.mark.parametrize("name", CREDENTIAL_ENV_NAMES)
def test_tns_credentials_names_the_missing_variable(monkeypatch, tns_env, name) -> None:
    """Verify `tns_credentials` reports which variable is unset rather than failing opaquely"""
    monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match=name):
        tns_catalog.tns_credentials()


def test_tns_credentials_rejects_a_blank_variable(monkeypatch, tns_env) -> None:
    """Verify `tns_credentials` treats a whitespace-only value as unset

    An empty GitHub secret expands to an empty string rather than being absent, so a
    presence check alone would let it through to TNS as a 401.
    """
    monkeypatch.setenv(tns_catalog.TNS_API_KEY_ENV, "   ")
    with pytest.raises(RuntimeError, match=tns_catalog.TNS_API_KEY_ENV):
        tns_catalog.tns_credentials()


def test_tns_credentials_strips_surrounding_whitespace(monkeypatch, tns_env) -> None:
    """Verify `tns_credentials` strips a trailing newline, as a pasted secret often carries"""
    monkeypatch.setenv(tns_catalog.TNS_API_KEY_ENV, "  test-key\n")
    assert tns_catalog.tns_credentials()[0] == "test-key"


def test_tns_credentials_rejects_a_non_integer_bot_id(monkeypatch, tns_env) -> None:
    """Verify `tns_credentials` catches TNS_BOT_ID and TNS_BOT_NAME being swapped"""
    monkeypatch.setenv(tns_catalog.TNS_BOT_ID_ENV, "NotAnId")
    with pytest.raises(RuntimeError, match="swapped"):
        tns_catalog.tns_credentials()


def test_download_tns_table_sends_the_bot_marker(monkeypatch, tns_env) -> None:
    """Verify `download_tns_table` posts the API key under a well-formed tns_marker header"""
    calls = install_fake_post(monkeypatch)

    assert tns_catalog.download_tns_table() == b"zip-bytes"

    assert len(calls) == 1
    assert calls[0]["url"] == tns_catalog.CATALOG_URL
    marker = calls[0]["headers"]["user-agent"]
    assert marker.startswith("tns_marker")
    assert json.loads(marker.removeprefix("tns_marker")) == {
        "tns_id": "424242",
        "type": "bot",
        "name": "TestBot",
    }
    assert calls[0]["data"] == {"api_key": (None, "test-key")}


def test_download_tns_table_does_not_request_without_credentials(monkeypatch) -> None:
    """Verify `download_tns_table` fails before issuing a request when unconfigured"""
    for name in CREDENTIAL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    calls = install_fake_post(monkeypatch)

    with pytest.raises(RuntimeError, match=tns_catalog.TNS_API_KEY_ENV):
        tns_catalog.download_tns_table()

    assert calls == []


def test_download_tns_table_propagates_http_errors(monkeypatch, tns_env) -> None:
    """Verify `download_tns_table` lets a TNS rejection surface rather than returning junk"""
    install_fake_post(monkeypatch, FakeResponse(error=requests.HTTPError("401 Unauthorized")))
    with pytest.raises(requests.HTTPError):
        tns_catalog.download_tns_table()
