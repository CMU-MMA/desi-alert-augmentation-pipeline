import json
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
import requests
from astropy import units as u

from desi_aap import tns_catalog
from desi_aap.cosmology import COSMOLOGIES

# Defaults of clean_tns_catalog, restated here so a change to either is a deliberate test
# edit rather than something the suite silently follows.
DEFAULT_MAX_DISTANCE_MPC = 500
DEFAULT_MIN_REDSHIFT = 0.0002

# Redshifts chosen against the 500 Mpc distance cut. The middle one is
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
    """Verify `clean_tns_catalog` keeps only types matching stripped_env_type_regex"""
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

    stripped_env_type_regex carries an open TODO about exactly this: "SLSN-Ic" and
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
    """Verify `clean_tns_catalog` keeps objects inside max_stripped_env_distance_mpc"""
    df = make_tns_frame(
        [
            {"name": "near", "redshift": NEARBY_Z},
            {"name": "far", "redshift": DISTANT_Z},
        ]
    )

    cleaned = tns_catalog.clean_tns_catalog(df)

    assert cleaned["name"].tolist() == ["near"]
    for label in COSMOLOGIES:
        assert luminosity_distance(NEARBY_Z, label) < DEFAULT_MAX_DISTANCE_MPC
        assert luminosity_distance(DISTANT_Z, label) > DEFAULT_MAX_DISTANCE_MPC


def test_clean_tns_catalog_keeps_objects_near_under_any_cosmology() -> None:
    """Verify `clean_tns_catalog` keeps an object inside the cut under only one cosmology"""
    max_mpc = DEFAULT_MAX_DISTANCE_MPC
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


def test_clean_tns_catalog_drops_non_positive_redshifts() -> None:
    """Verify `clean_tns_catalog` drops every kind of non-positive redshift in one pass

    Each row below is a distinct failure the floor exists to prevent, and all of them are
    checked together because the interesting property is that the healthy rows survive
    alongside them: a single unusable redshift must not take the rest of the catalog with it.

    The two healthy rows sit at different redshifts, interleaved with the bad ones, so the
    survivors reach the cosmology loop with a gappy index [0, 3]. The distance assertion
    below therefore catches distances aligned to the pre-filter rows.
    """
    df = make_tns_frame(
        [
            # Healthy control (1/2): Everything else in this frame must be dropped around the controls.
            {"name": "measured", "redshift": NEARBY_Z},
            # Zero distance, so no meaningful credible level downstream. Nothing else in the
            # function removes it: it passes dropna and clears the distance cut at 0 Mpc.
            {"name": "zero", "redshift": 0.0},
            # A real blueshift, where nearby peculiar velocity outruns the Hubble flow. This
            # is the one that bites in practice: it yields a negative distance, which if allowed
            # to reach run_3d_spatial_crossmatch would raise ValueError out of SkyCoord.
            {"name": "blueshifted", "redshift": -0.001},
            # Healthy control (2/2)
            {"name": "also-measured", "redshift": SPLIT_Z},
            # Below -1 the floor has to act early. Left to reach luminosity_distance, this
            # aborts the whole call on astropy 7.1.1, where Planck18's integrand goes complex
            # and raises TypeError, so a filter placed after the cosmology loop is too late.
            {"name": "deeply-negative", "redshift": -1.2},
            # z == -2 exactly is its own case: ZeroDivisionError rather than TypeError.
            {"name": "at-minus-two", "redshift": -2.0},
        ]
    )

    cleaned = tns_catalog.clean_tns_catalog(df)

    assert cleaned["name"].tolist() == ["measured", "also-measured"]
    for label in COSMOLOGIES:
        assert cleaned[f"dist_mpc_{label}"].tolist() == pytest.approx(
            [luminosity_distance(NEARBY_Z, label), luminosity_distance(SPLIT_Z, label)]
        )


@pytest.mark.parametrize(
    "redshift,kept",
    [
        (DEFAULT_MIN_REDSHIFT, True),
        (DEFAULT_MIN_REDSHIFT * 1.01, True),
        (DEFAULT_MIN_REDSHIFT * 0.99, False),
    ],
    ids=["at_floor", "just_above", "just_below"],
)
def test_clean_tns_catalog_applies_the_redshift_floor(redshift, kept) -> None:
    """Verify min_redshift is the smallest redshift kept, that is, that the floor is inclusive"""
    cleaned = tns_catalog.clean_tns_catalog(make_tns_frame([{"name": "sn", "redshift": redshift}]))

    assert cleaned["name"].tolist() == (["sn"] if kept else [])


# Spelled out rather than read from tns_catalog: these names are the contract with CI
# secrets and every user's shell, so a rename must fail here instead of being followed.
TNS_API_KEY_ENV = "TNS_API_KEY"
TNS_BOT_ID_ENV = "TNS_BOT_ID"
TNS_BOT_NAME_ENV = "TNS_BOT_NAME"
CREDENTIAL_ENV_NAMES = (TNS_API_KEY_ENV, TNS_BOT_ID_ENV, TNS_BOT_NAME_ENV)


def test_clean_tns_catalog_honors_the_type_regex_argument() -> None:
    """Verify `clean_tns_catalog` selects types with a caller-supplied stripped_env_type_regex"""
    df = make_tns_frame([{"name": "ib", "type": "SN Ib"}, {"name": "ia", "type": "SN Ia"}])

    assert tns_catalog.clean_tns_catalog(df, stripped_env_type_regex="Ia")["name"].tolist() == ["ia"]


def test_clean_tns_catalog_honors_the_distance_cut_argument() -> None:
    """Verify `clean_tns_catalog` applies a caller-supplied max_stripped_env_distance_mpc"""
    df = make_tns_frame([{"name": "near", "redshift": NEARBY_Z}])
    nearest = min(luminosity_distance(NEARBY_Z, label) for label in COSMOLOGIES)

    assert tns_catalog.clean_tns_catalog(df)["name"].tolist() == ["near"]
    tightened = tns_catalog.clean_tns_catalog(df, max_stripped_env_distance_mpc=nearest)
    assert tightened.empty


def test_clean_tns_catalog_honors_the_redshift_floor_argument() -> None:
    """Verify `clean_tns_catalog` applies a caller-supplied min_redshift"""
    df = make_tns_frame([{"name": "sn", "redshift": DEFAULT_MIN_REDSHIFT}])

    assert tns_catalog.clean_tns_catalog(df)["name"].tolist() == ["sn"]
    raised = tns_catalog.clean_tns_catalog(df, min_redshift=DEFAULT_MIN_REDSHIFT * 1.01)
    assert raised.empty


CATALOG_URL = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"


def make_tns_zip(rows, timestamp_line="2026-08-06 00:00:00"):
    """Build a zipped TNS CSV payload, including the timestamp line above the header.

    TNS writes the file's generation time as the first line, so a reader that does not skip
    it takes the timestamp for the header. Building the payload that way here is what lets
    the download tests prove the skip actually happens.
    """
    csv = timestamp_line + "\n" + pd.DataFrame(rows).to_csv(index=False)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("tns_public_objects.csv", csv)
    return buffer.getvalue()


class FakeResponse:
    """Stand-in for the requests.Response that requests.post returns."""

    def __init__(self, content=None, error=None):
        if content is None:
            content = make_tns_zip([{"name": "2019ebq", "type": "SN Ib", "redshift": 0.037}])
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
    monkeypatch.setenv(TNS_API_KEY_ENV, "test-key")
    monkeypatch.setenv(TNS_BOT_ID_ENV, "424242")
    monkeypatch.setenv(TNS_BOT_NAME_ENV, "TestBot")


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
    monkeypatch.setenv(TNS_API_KEY_ENV, "   ")
    with pytest.raises(RuntimeError, match=TNS_API_KEY_ENV):
        tns_catalog.tns_credentials()


def test_tns_credentials_strips_surrounding_whitespace(monkeypatch, tns_env) -> None:
    """Verify `tns_credentials` strips a trailing newline, as a pasted secret often carries"""
    monkeypatch.setenv(TNS_API_KEY_ENV, "  test-key\n")
    assert tns_catalog.tns_credentials()[0] == "test-key"


def test_tns_credentials_rejects_a_non_integer_bot_id(monkeypatch, tns_env) -> None:
    """Verify `tns_credentials` catches TNS_BOT_ID and TNS_BOT_NAME being swapped"""
    monkeypatch.setenv(TNS_BOT_ID_ENV, "NotAnId")
    with pytest.raises(RuntimeError, match="swapped"):
        tns_catalog.tns_credentials()


def test_download_tns_table_sends_the_bot_marker(monkeypatch, tns_env) -> None:
    """Verify `download_tns_table` posts the API key under a well-formed tns_marker header"""
    calls = install_fake_post(monkeypatch)

    assert tns_catalog.download_tns_table()["name"].tolist() == ["2019ebq"]

    assert len(calls) == 1
    assert calls[0]["url"] == CATALOG_URL
    marker = calls[0]["headers"]["user-agent"]
    assert marker.startswith("tns_marker")
    assert json.loads(marker.removeprefix("tns_marker")) == {
        "tns_id": "424242",
        "type": "bot",
        "name": "TestBot",
    }
    assert calls[0]["data"] == {"api_key": (None, "test-key")}


def test_download_tns_table_skips_the_generation_timestamp(monkeypatch, tns_env) -> None:
    """Verify `download_tns_table` reads TNS's first line as a timestamp, not as the header"""
    payload = make_tns_zip([{"name": "2019ebq"}], timestamp_line="2026-08-06 00:00:00")
    install_fake_post(monkeypatch, FakeResponse(payload))

    df = tns_catalog.download_tns_table()

    assert df.columns.tolist() == ["name"]
    assert df["name"].tolist() == ["2019ebq"]


def test_download_tns_table_does_not_request_without_credentials(monkeypatch) -> None:
    """Verify `download_tns_table` fails before issuing a request when unconfigured"""
    for name in CREDENTIAL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    calls = install_fake_post(monkeypatch)

    with pytest.raises(RuntimeError, match=TNS_API_KEY_ENV):
        tns_catalog.download_tns_table()

    assert calls == []


def test_download_tns_table_propagates_http_errors(monkeypatch, tns_env) -> None:
    """Verify `download_tns_table` lets a TNS rejection surface rather than returning junk"""
    install_fake_post(monkeypatch, FakeResponse(error=requests.HTTPError("401 Unauthorized")))
    with pytest.raises(requests.HTTPError):
        tns_catalog.download_tns_table()
