import json
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from desi_aap import gracedb_tools
from desi_aap.cosmology import COSMOLOGIES

# GW190425: t_0 in GPS seconds and the true UTC time of the merger.
GW190425_GPS = 1240215503.017147
GW190425_UTC = pd.Timestamp("2019-04-25 08:18:05.017147", tz="UTC")

# FAR values in Hz that land either side of the default far_threshold_per_year of 2.0.
QUIET_FAR_HZ = 1e-9  # ~0.03 per year, passes the cut
LOUD_FAR_HZ = 1e-6  # ~32 per year, fails the cut

# Credible level these tests crossmatch at, and the default run_3d_spatial_crossmatch applies
# when a caller does not name one. Kept separate so a change to the default is a visible
# one-line test edit rather than something the suite silently follows.
TEST_CREDIBLE_LEVEL = 0.5
RUN_3D_DEFAULT_CREDIBLE_LEVEL = 0.5

BNS_PASTRO = {"BNS": 0.95, "NSBH": 0.01, "BBH": 0.01, "Terrestrial": 0.03}
BBH_PASTRO = {"BNS": 0.0, "NSBH": 0.0, "BBH": 0.99, "Terrestrial": 0.01}


class FakeJsonResponse:
    """Stand-in for the listing object returned by GraceDb.files(superevent_id)."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        """Return the file listing as a dict of file name to URL."""
        return self._payload


class FakeFileResponse:
    """Stand-in for the file-like object returned by GraceDb.files(superevent_id, name)."""

    def __init__(self, data):
        self._data = data

    def read(self):
        """Return the file body."""
        return self._data


class FakeGraceDbClient:
    """Stand-in for ligo.gracedb.rest.GraceDb, driven by in-memory dicts.

    Parameters
    ----------
    superevents : list of dict
        Superevent dicts yielded by the superevents() query.
    files_by_id : dict
        Maps superevent_id to its file listing.
    payloads : dict, optional
        Maps (superevent_id, file_name) to the bytes that file downloads as.
    errors : dict, optional
        Maps superevent_id (for listings) or (superevent_id, file_name) (for downloads)
        to an exception instance to raise instead of answering.
    """

    def __init__(self, superevents, files_by_id, payloads=None, errors=None):
        self._superevents = superevents
        self._files_by_id = files_by_id
        self._payloads = payloads or {}
        self._errors = errors or {}
        self.queries = []
        self.downloads = []

    def superevents(self, query=None, max_results=None):
        """Record the query and yield the canned superevents."""
        self.queries.append((query, max_results))
        return iter(self._superevents)

    def files(self, superevent_id, filename=None):
        """Return a file listing, or a single file's contents when filename is given."""
        key = superevent_id if filename is None else (superevent_id, filename)
        if key in self._errors:
            raise self._errors[key]
        if filename is None:
            return FakeJsonResponse(self._files_by_id[superevent_id])
        self.downloads.append((superevent_id, filename))
        return FakeFileResponse(self._payloads[(superevent_id, filename)])


class FakeSkymap:
    """Stand-in for the table returned by ligo.skymap.io.read_sky_map."""

    def __init__(self, colnames=("UNIQ", "PROBDENSITY", "DISTMU", "DISTSIGMA", "DISTNORM")):
        self.colnames = list(colnames)


def make_superevent(superevent_id="S190425z", far=QUIET_FAR_HZ, t_0=GW190425_GPS, **overrides):
    """Build a minimal GraceDB superevent dict, overriding any top-level key."""
    superevent = {
        "superevent_id": superevent_id,
        "far": far,
        "t_0": t_0,
        "labels": ["PE_READY", "SKYMAP_READY"],
        "preferred_event_data": {
            "graceid": "G330561",
            "pipeline": "gstlal",
            "search": "AllSky",
            "instruments": "H1,L1",
            "far": far,
            "gpstime": t_0,
        },
    }
    superevent.update(overrides)
    return superevent


def make_crossmatch_result(n_coords, searched_prob=0.1, searched_prob_vol=0.2, contours=True):
    """Build a stand-in for ligo.skymap's CrossmatchResult with n_coords per-SN values."""
    return SimpleNamespace(
        searched_area=np.full(n_coords, 12.5),
        searched_prob=np.full(n_coords, searched_prob),
        offset=np.full(n_coords, 1.5),
        searched_prob_dist=np.full(n_coords, 0.3),
        searched_vol=np.full(n_coords, 1.0e6),
        searched_prob_vol=np.full(n_coords, searched_prob_vol),
        probdensity_vol=np.full(n_coords, 4.0e-9),
        contour_vols=[2.0e6] if contours else [],
        contour_areas=[300.0] if contours else [],
    )


@pytest.fixture(name="df_sesn")
def df_sesn_fixture():
    """Two SNe discovered within a day of GW190425, with distances for every cosmology."""
    frame = pd.DataFrame(
        {
            "name": ["2019ebq", "2019eff"],
            "discoverydate": [
                GW190425_UTC + pd.Timedelta(hours=4),
                GW190425_UTC + pd.Timedelta(hours=6),
            ],
            "ra": [240.0, 241.0],
            "declination": [-20.0, -21.0],
        }
    )
    for i, label in enumerate(COSMOLOGIES):
        frame[f"dist_mpc_{label}"] = [150.0 + i, 160.0 + i]
    return frame


@pytest.fixture(name="gw_events")
def gw_events_fixture():
    """A single-row superevent table shaped like fetch_gracedb_superevents output."""
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
                "skymap_path": "S190425z__bayestar.multiorder.fits",
                "status": "ok",
            }
        ]
    )


def test_as_float_converts_numeric_strings() -> None:
    """Verify `as_float` coerces the numeric strings in GraceDB JSON to floats"""
    assert gracedb_tools.as_float("1.5") == 1.5
    assert gracedb_tools.as_float(2) == 2.0
    assert gracedb_tools.as_float(3.5) == 3.5


def test_as_float_falls_back_on_bad_values() -> None:
    """Verify `as_float` returns the default for None and unparseable values"""
    assert np.isnan(gracedb_tools.as_float(None))
    assert np.isnan(gracedb_tools.as_float("not a number"))
    assert np.isnan(gracedb_tools.as_float([1, 2]))
    assert gracedb_tools.as_float(None, 0.0) == 0.0
    assert gracedb_tools.as_float("bad", -1.0) == -1.0


def test_response_to_bytes_reads_file_like_objects() -> None:
    """Verify `response_to_bytes` prefers read() and encodes text payloads as UTF-8"""
    assert gracedb_tools.response_to_bytes(FakeFileResponse(b"raw")) == b"raw"
    assert gracedb_tools.response_to_bytes(FakeFileResponse("téxt")) == "téxt".encode()


def test_response_to_bytes_reads_requests_style_responses() -> None:
    """Verify `response_to_bytes` falls back to the content attribute"""
    assert gracedb_tools.response_to_bytes(SimpleNamespace(content=b"raw")) == b"raw"
    assert gracedb_tools.response_to_bytes(SimpleNamespace(content="text")) == b"text"


def test_unversioned_file_names_drops_version_suffixes() -> None:
    """Verify `unversioned_file_names` keeps only unsuffixed names, in input order"""
    # TODO this may not be the behavior we end up preserving - come back to this later.
    files = [
        "bayestar.multiorder.fits",
        "bayestar.multiorder.fits,0",
        "bayestar.multiorder.fits,12",
        "p_astro.json",
        "p_astro.json,3",
    ]
    assert gracedb_tools.unversioned_file_names(files) == [
        "bayestar.multiorder.fits",
        "p_astro.json",
    ]


def test_unversioned_file_names_keeps_non_numeric_commas() -> None:
    """Verify `unversioned_file_names` only treats a trailing ",N" as a version"""
    assert gracedb_tools.unversioned_file_names(["a,b.json", "x,1y.fits"]) == ["a,b.json", "x,1y.fits"]


def test_choose_pastro_file_prefers_the_preferred_pipeline() -> None:
    """Verify `choose_pastro_file` picks the preferred event's pipeline file over the generic one"""
    superevent = make_superevent()
    files = ["p_astro.json", "gstlal.p_astro.json", "mbta.p_astro.json"]
    assert gracedb_tools.choose_pastro_file(superevent, files) == "gstlal.p_astro.json"


def test_choose_pastro_file_matches_pipeline_case_insensitively() -> None:
    """Verify `choose_pastro_file` matches the pipeline name regardless of case"""
    superevent = make_superevent()
    superevent["preferred_event_data"]["pipeline"] = "GstLAL"
    assert gracedb_tools.choose_pastro_file(superevent, ["GSTLAL.p_astro.json"]) == "GSTLAL.p_astro.json"


def test_choose_pastro_file_falls_back_to_generic_names() -> None:
    """Verify `choose_pastro_file` uses p_astro.json, then pastro.json, when no pipeline file exists"""
    superevent = make_superevent()
    assert gracedb_tools.choose_pastro_file(superevent, ["p_astro.json", "pastro.json"]) == "p_astro.json"
    assert gracedb_tools.choose_pastro_file(superevent, ["pastro.json"]) == "pastro.json"


def test_choose_pastro_file_falls_back_to_any_pipeline_file() -> None:
    """Verify `choose_pastro_file` takes the alphabetically first p_astro file as a last resort"""
    superevent = make_superevent()
    superevent["preferred_event_data"]["pipeline"] = ""
    files = ["spiir.p_astro.json", "mbta.p_astro.json"]
    assert gracedb_tools.choose_pastro_file(superevent, files) == "mbta.p_astro.json"


def test_choose_pastro_file_ignores_versioned_copies() -> None:
    """Verify `choose_pastro_file` never returns a ",N" revision of a p_astro file"""
    superevent = make_superevent()
    assert gracedb_tools.choose_pastro_file(superevent, ["gstlal.p_astro.json,0"]) is None


def test_choose_pastro_file_returns_none_without_a_candidate() -> None:
    """Verify `choose_pastro_file` returns None when the listing has no p_astro file"""
    superevent = make_superevent()
    assert gracedb_tools.choose_pastro_file(superevent, ["bayestar.multiorder.fits"]) is None
    assert gracedb_tools.choose_pastro_file(superevent, []) is None


def test_choose_pastro_file_tolerates_a_missing_pipeline() -> None:
    """Verify `choose_pastro_file` handles a superevent with no preferred event data"""
    assert gracedb_tools.choose_pastro_file({}, ["p_astro.json"]) == "p_astro.json"
    assert gracedb_tools.choose_pastro_file({"preferred_event_data": {}}, ["p_astro.json"]) == "p_astro.json"


def test_load_classification_parses_the_payload() -> None:
    """Verify `load_classification` downloads and parses the p_astro JSON"""
    payload = json.dumps(BNS_PASTRO).encode()
    client = FakeGraceDbClient([], {}, payloads={("S190425z", "p_astro.json"): payload})
    classification, name = gracedb_tools.load_classification(client, "S190425z", "p_astro.json")
    assert classification == BNS_PASTRO
    assert name == "p_astro.json"
    assert client.downloads == [("S190425z", "p_astro.json")]


def test_load_classification_skips_a_missing_file() -> None:
    """Verify `load_classification` short-circuits without downloading when given no file name"""
    client = FakeGraceDbClient([], {})
    assert gracedb_tools.load_classification(client, "S190425z", None) == ({}, None)
    assert gracedb_tools.load_classification(client, "S190425z", "") == ({}, None)
    assert client.downloads == []


def test_load_classification_propagates_errors() -> None:
    """Verify `load_classification` lets download failures reach the caller"""
    client = FakeGraceDbClient([], {}, errors={("S190425z", "p_astro.json"): RuntimeError("503")})
    with pytest.raises(RuntimeError, match="503"):
        gracedb_tools.load_classification(client, "S190425z", "p_astro.json")


def test_skymap_priority_ranks_pipelines_and_formats() -> None:
    """Verify `skymap_priority` ranks by pipeline and format, and versioned names below all others"""
    # Asserted as an ordering rather than against the individual rank values, so that
    # renumbering the table in skymap_priority does not need a matching edit here.
    best_first = [
        "bilby.multiorder.fits",
        "bayestar.multiorder.fits",
        "cwb.multiorder.fits",
        "bayestar.fits.gz",
        "LALInference.fits.gz",
        "skymap.fits",
        # The version penalty is larger than the whole span of the ranking table, so every
        # versioned name sorts below every unversioned one: a ",N" revision of the best
        # format loses to an unversioned copy of the worst. Ordering still holds among them.
        "bilby.multiorder.fits,0",
        "bayestar.multiorder.fits,0",
    ]
    priorities = [gracedb_tools.skymap_priority(name) for name in best_first]

    assert all(better < worse for better, worse in zip(priorities, priorities[1:], strict=False))
    assert max(priorities) < gracedb_tools.SKYMAP_PRIORITY_IGNORE


def test_skymap_priority_is_case_insensitive() -> None:
    """Verify `skymap_priority` ranks names regardless of case"""
    assert gracedb_tools.skymap_priority("Bilby.MultiOrder.FITS") == (
        gracedb_tools.skymap_priority("bilby.multiorder.fits")
    )


def test_skymap_priority_ignores_non_skymaps() -> None:
    """Verify `skymap_priority` ranks non-FITS files at exactly SKYMAP_PRIORITY_IGNORE"""
    assert gracedb_tools.skymap_priority("p_astro.json") == gracedb_tools.SKYMAP_PRIORITY_IGNORE
    assert gracedb_tools.skymap_priority("coinc.xml") == gracedb_tools.SKYMAP_PRIORITY_IGNORE
    # The Bilby and BAYESTAR rules match anywhere in the name, so a skymap suffix that is
    # not the last one must still be ignored.
    assert gracedb_tools.skymap_priority("bilby.multiorder.fits.txt") == (
        gracedb_tools.SKYMAP_PRIORITY_IGNORE
    )


def test_skymap_priority_ignores_without_a_version_penalty() -> None:
    """Verify `skymap_priority` gives every ignored name the same rank, versioned or not"""
    # choose_skymap_file discards anything at SKYMAP_PRIORITY_IGNORE without ordering them
    # against each other, so a version penalty on top would be meaningless precision.
    assert gracedb_tools.skymap_priority("p_astro.json,0") == gracedb_tools.SKYMAP_PRIORITY_IGNORE
    assert gracedb_tools.skymap_priority("coinc.xml,2") == gracedb_tools.SKYMAP_PRIORITY_IGNORE


def test_skymap_priority_warns_on_unexpected_name_shapes() -> None:
    """Verify `skymap_priority` warns when a name is not a bare name or a single ",N" revision"""
    for name in ("bayestar,extra.multiorder.fits", "bilby.multiorder.fits,0,1", "a,b,c.fits"):
        with pytest.warns(UserWarning, match="neither a bare name nor a single"):
            gracedb_tools.skymap_priority(name)


def test_skymap_priority_stays_quiet_on_well_formed_names() -> None:
    """Verify `skymap_priority` does not warn about the name shapes GraceDB is expected to use"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ("bayestar.multiorder.fits", "bilby.multiorder.fits,0", "p_astro.json"):
            gracedb_tools.skymap_priority(name)


def test_choose_skymap_file_falls_back_to_a_versioned_skymap() -> None:
    """Verify `choose_skymap_file` uses a ",N" revision when it is the only skymap available"""
    files = ["p_astro.json", "bayestar.multiorder.fits,0"]
    assert gracedb_tools.choose_skymap_file(files) == "bayestar.multiorder.fits,0"


def test_choose_skymap_file_takes_the_best_available() -> None:
    """Verify `choose_skymap_file` returns the lowest-priority skymap in the listing"""
    files = [
        "p_astro.json",
        "bayestar.fits.gz",
        "bayestar.multiorder.fits",
        "bilby.multiorder.fits",
    ]
    assert gracedb_tools.choose_skymap_file(files) == "bilby.multiorder.fits"
    assert gracedb_tools.choose_skymap_file(files[:-1]) == "bayestar.multiorder.fits"


def test_choose_skymap_file_breaks_ties_alphabetically() -> None:
    """Verify `choose_skymap_file` breaks equal priorities case-insensitively by name"""
    assert gracedb_tools.choose_skymap_file(["Zeta.multiorder.fits", "alpha.multiorder.fits"]) == (
        "alpha.multiorder.fits"
    )


def test_choose_skymap_file_returns_none_without_a_skymap() -> None:
    """Verify `choose_skymap_file` returns None when nothing in the listing is a skymap"""
    assert gracedb_tools.choose_skymap_file(["p_astro.json", "coinc.xml"]) is None
    assert gracedb_tools.choose_skymap_file([]) is None


def test_safe_file_part_collapses_unsafe_characters() -> None:
    """Verify `safe_file_part` replaces each run of unsafe characters with one underscore"""
    assert gracedb_tools.safe_file_part("bayestar.multiorder.fits") == "bayestar.multiorder.fits"
    assert gracedb_tools.safe_file_part("bayestar.fits,0") == "bayestar.fits_0"
    assert gracedb_tools.safe_file_part("a/b  c") == "a_b_c"
    assert gracedb_tools.safe_file_part(42) == "42"


def test_download_gracedb_file_writes_a_namespaced_copy(tmp_path) -> None:
    """Verify `download_gracedb_file` writes "<superevent>__<file>" into the output directory"""
    payloads = {("S190425z", "bayestar.multiorder.fits"): b"FITS"}
    client = FakeGraceDbClient([], {}, payloads=payloads)
    outdir = tmp_path / "skymaps"
    path = gracedb_tools.download_gracedb_file(client, "S190425z", "bayestar.multiorder.fits", outdir=outdir)
    assert path == outdir / "S190425z__bayestar.multiorder.fits"
    assert path.read_bytes() == b"FITS"


def test_download_gracedb_file_reuses_an_existing_copy(tmp_path) -> None:
    """Verify `download_gracedb_file` trusts a cached file and does not download it again"""
    payloads = {("S190425z", "bayestar.multiorder.fits"): b"FITS"}
    client = FakeGraceDbClient([], {}, payloads=payloads)
    for _ in range(2):
        gracedb_tools.download_gracedb_file(client, "S190425z", "bayestar.multiorder.fits", outdir=tmp_path)
    assert client.downloads == [("S190425z", "bayestar.multiorder.fits")]


def test_gps_to_utc_converts_a_known_event() -> None:
    """Verify `gps_to_utc` converts GPS seconds to the true UTC time, leap seconds included"""
    converted = gracedb_tools.gps_to_utc(GW190425_GPS)
    assert converted == GW190425_UTC
    assert str(converted.tz) == "UTC"
    assert gracedb_tools.gps_to_utc(0.0) == pd.Timestamp("1980-01-06 00:00:00", tz="UTC")


def test_gps_to_utc_passes_through_missing_values() -> None:
    """Verify `gps_to_utc` maps missing GPS times to NaT"""
    assert gracedb_tools.gps_to_utc(np.nan) is pd.NaT
    assert gracedb_tools.gps_to_utc(None) is pd.NaT


def install_fake_client(monkeypatch, client):
    """Point gracedb_tools at a fake GraceDb client and record the constructor arguments."""
    constructed = []

    def fake_gracedb(*args, **kwargs):
        constructed.append((args, kwargs))
        return client

    monkeypatch.setattr(gracedb_tools, "GraceDb", fake_gracedb)
    return constructed


def test_fetch_gracedb_superevents_builds_a_row(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` returns one row per passing superevent"""
    monkeypatch.chdir(tmp_path)
    superevent = make_superevent()
    files = {"gstlal.p_astro.json": "url", "bilby.multiorder.fits": "url"}
    payloads = {
        ("S190425z", "gstlal.p_astro.json"): json.dumps(BNS_PASTRO).encode(),
        ("S190425z", "bilby.multiorder.fits"): b"FITS",
    }
    client = FakeGraceDbClient([superevent], {"S190425z": files}, payloads=payloads)
    constructed = install_fake_client(monkeypatch, client)

    df = gracedb_tools.fetch_gracedb_superevents(["bns"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["superevent_id"] == "S190425z"
    assert row["gw_time"] == GW190425_UTC
    assert row["gps_time"] == GW190425_GPS
    assert row["far_hz"] == QUIET_FAR_HZ
    assert row["far_per_year"] == pytest.approx(QUIET_FAR_HZ * gracedb_tools.JULIAN_YEAR_SECONDS)
    assert row["p_bns"] == 0.95
    assert row["p_terrestrial"] == 0.03
    assert row["classification_file"] == "gstlal.p_astro.json"
    assert row["preferred_event"] == "G330561"
    assert row["pipeline"] == "gstlal"
    assert row["labels"] == "PE_READY,SKYMAP_READY"
    assert row["skymap_file"] == "bilby.multiorder.fits"
    assert row["status"] == "ok"
    expected_path = gracedb_tools.SKYMAP_DIR / "S190425z__bilby.multiorder.fits"
    assert row["skymap_path"] == str(expected_path)
    assert expected_path.read_bytes() == b"FITS"
    assert constructed == [((), {"service_url": "https://gracedb.ligo.org/api/"})]
    expected_far_hz = 2.0 / gracedb_tools.JULIAN_YEAR_SECONDS
    assert client.queries == [(f"category: Production far < {expected_far_hz:.12g}", None)]


def test_fetch_gracedb_superevents_drops_loud_events(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` applies the FAR cut before reading any file"""
    monkeypatch.chdir(tmp_path)
    superevents = [
        make_superevent("S190425z", far=LOUD_FAR_HZ),
        make_superevent("S190426c", far=None, preferred_event_data={}),
    ]
    client = FakeGraceDbClient(superevents, {})
    install_fake_client(monkeypatch, client)

    assert gracedb_tools.fetch_gracedb_superevents(["bns"]).empty
    assert client.downloads == []


def test_fetch_gracedb_superevents_falls_back_to_the_preferred_far(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` uses the preferred event's FAR when the superevent's is None"""
    monkeypatch.chdir(tmp_path)
    superevent = make_superevent(far=QUIET_FAR_HZ)
    superevent["far"] = None
    superevent["t_0"] = None
    files = {"p_astro.json": "url"}
    payloads = {("S190425z", "p_astro.json"): json.dumps(BNS_PASTRO).encode()}
    client = FakeGraceDbClient([superevent], {"S190425z": files}, payloads=payloads)
    install_fake_client(monkeypatch, client)

    df = gracedb_tools.fetch_gracedb_superevents(["bns"])

    assert df.iloc[0]["far_hz"] == QUIET_FAR_HZ
    assert df.iloc[0]["gps_time"] == GW190425_GPS
    assert df.iloc[0]["gw_time"] == GW190425_UTC


def test_fetch_gracedb_superevents_applies_the_classification_cut(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` keeps only superevents clearing the classification cut"""
    monkeypatch.chdir(tmp_path)
    superevents = [make_superevent("S190425z"), make_superevent("S190521g")]
    files_by_id = {sid: {"p_astro.json": "url"} for sid in ("S190425z", "S190521g")}
    payloads = {
        ("S190425z", "p_astro.json"): json.dumps(BNS_PASTRO).encode(),
        ("S190521g", "p_astro.json"): json.dumps(BBH_PASTRO).encode(),
    }
    client = FakeGraceDbClient(superevents, files_by_id, payloads=payloads)
    install_fake_client(monkeypatch, client)

    assert gracedb_tools.fetch_gracedb_superevents(["bns"])["superevent_id"].tolist() == ["S190425z"]
    assert gracedb_tools.fetch_gracedb_superevents(["bbh"])["superevent_id"].tolist() == ["S190521g"]


def test_fetch_gracedb_superevents_honors_the_classification_cut_argument(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` applies a caller-supplied min_classification_prob_sum"""
    monkeypatch.chdir(tmp_path)
    client = FakeGraceDbClient(
        [make_superevent()],
        {"S190425z": {"p_astro.json": "url"}},
        payloads={("S190425z", "p_astro.json"): json.dumps(BNS_PASTRO).encode()},
    )
    install_fake_client(monkeypatch, client)

    # BNS_PASTRO has p_bns 0.95, so it clears the 0.9 default but not a 0.99 cut.
    assert len(gracedb_tools.fetch_gracedb_superevents(["bns"], min_classification_prob_sum=0.5)) == 1
    assert gracedb_tools.fetch_gracedb_superevents(["bns"], min_classification_prob_sum=0.99).empty


def test_fetch_gracedb_superevents_honors_the_far_threshold_argument(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` applies far_threshold_per_year to the query and the rows"""
    monkeypatch.chdir(tmp_path)
    client = FakeGraceDbClient(
        [make_superevent(far=QUIET_FAR_HZ)],
        {"S190425z": {"p_astro.json": "url"}},
        payloads={("S190425z", "p_astro.json"): json.dumps(BNS_PASTRO).encode()},
    )
    install_fake_client(monkeypatch, client)

    # QUIET_FAR_HZ is ~0.03 per year, so a cut below that drops it locally as well as in
    # the query string the fake client records but does not act on.
    assert gracedb_tools.fetch_gracedb_superevents(["bns"], far_threshold_per_year=0.001).empty
    expected_far_hz = 0.001 / gracedb_tools.JULIAN_YEAR_SECONDS
    assert client.queries == [(f"category: Production far < {expected_far_hz:.12g}", None)]


def test_fetch_gracedb_superevents_passes_max_results_through(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` forwards max_results to the GraceDB query"""
    monkeypatch.chdir(tmp_path)
    client = FakeGraceDbClient([], {})
    install_fake_client(monkeypatch, client)

    gracedb_tools.fetch_gracedb_superevents(["bns"], max_results=5)

    assert client.queries[0][1] == 5


def test_fetch_gracedb_superevents_sums_requested_types(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` adds the probabilities of every requested type"""
    monkeypatch.chdir(tmp_path)
    split = {"BNS": 0.5, "NSBH": 0.45, "BBH": 0.02, "Terrestrial": 0.03}
    client = FakeGraceDbClient(
        [make_superevent()],
        {"S190425z": {"p_astro.json": "url"}},
        payloads={("S190425z", "p_astro.json"): json.dumps(split).encode()},
    )
    install_fake_client(monkeypatch, client)

    assert gracedb_tools.fetch_gracedb_superevents(["bns"]).empty
    assert len(gracedb_tools.fetch_gracedb_superevents(["bns", "nsbh"])) == 1


def test_fetch_gracedb_superevents_records_a_file_list_failure(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` keeps a stub row when the file listing cannot be read"""
    monkeypatch.chdir(tmp_path)
    client = FakeGraceDbClient([make_superevent()], {}, errors={"S190425z": RuntimeError("gracedb is down")})
    install_fake_client(monkeypatch, client)

    df = gracedb_tools.fetch_gracedb_superevents(["bns"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["superevent_id"] == "S190425z"
    assert row["status"] == "file_list_failed: gracedb is down"
    assert row["far_hz"] == QUIET_FAR_HZ
    assert pd.isna(row["gw_time"])
    assert "p_bns" not in df.columns


def test_fetch_gracedb_superevents_sorts_a_stub_row_last(monkeypatch, tmp_path) -> None:
    """Verify a file_list_failed stub survives the gw_time sort next to a normal row"""
    monkeypatch.chdir(tmp_path)
    superevents = [make_superevent("S190426c"), make_superevent("S190425z")]
    client = FakeGraceDbClient(
        superevents,
        {"S190425z": {"p_astro.json": "url"}},
        payloads={("S190425z", "p_astro.json"): json.dumps(BNS_PASTRO).encode()},
        errors={"S190426c": RuntimeError("gracedb is down")},
    )
    install_fake_client(monkeypatch, client)

    df = gracedb_tools.fetch_gracedb_superevents(["bns"])

    assert df["superevent_id"].tolist() == ["S190425z", "S190426c"]
    assert pd.isna(df.iloc[1]["gw_time"])
    assert pd.isna(df.iloc[1]["p_bns"])


def test_fetch_gracedb_superevents_drops_unreadable_classifications(monkeypatch, tmp_path) -> None:
    """Verify a superevent whose p_astro cannot be read is dropped by the probability cut"""
    monkeypatch.chdir(tmp_path)
    client = FakeGraceDbClient(
        [make_superevent()],
        {"S190425z": {"p_astro.json": "url"}},
        errors={("S190425z", "p_astro.json"): RuntimeError("truncated")},
    )
    install_fake_client(monkeypatch, client)

    assert gracedb_tools.fetch_gracedb_superevents(["bns"]).empty


def test_fetch_gracedb_superevents_records_a_skymap_failure(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` keeps the row when only the skymap download fails"""
    monkeypatch.chdir(tmp_path)
    files = {"p_astro.json": "url", "bilby.multiorder.fits": "url"}
    client = FakeGraceDbClient(
        [make_superevent()],
        {"S190425z": files},
        payloads={("S190425z", "p_astro.json"): json.dumps(BNS_PASTRO).encode()},
        errors={("S190425z", "bilby.multiorder.fits"): RuntimeError("404")},
    )
    install_fake_client(monkeypatch, client)

    row = gracedb_tools.fetch_gracedb_superevents(["bns"]).iloc[0]

    assert row["status"] == "skymap_download_failed: 404"
    assert row["skymap_file"] == "bilby.multiorder.fits"
    assert row["skymap_path"] is None
    assert row["p_bns"] == 0.95


def test_fetch_gracedb_superevents_handles_a_missing_skymap(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` reports status ok when the listing has no skymap"""
    monkeypatch.chdir(tmp_path)
    client = FakeGraceDbClient(
        [make_superevent()],
        {"S190425z": {"p_astro.json": "url"}},
        payloads={("S190425z", "p_astro.json"): json.dumps(BNS_PASTRO).encode()},
    )
    install_fake_client(monkeypatch, client)

    row = gracedb_tools.fetch_gracedb_superevents(["bns"]).iloc[0]

    assert row["status"] == "ok"
    assert row["skymap_file"] is None
    assert row["skymap_path"] is None


def test_fetch_gracedb_superevents_sorts_by_gw_time(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` returns rows in chronological order"""
    monkeypatch.chdir(tmp_path)
    superevents = [
        make_superevent("S190814bv", t_0=GW190425_GPS + 9_000_000),
        make_superevent("S190425z", t_0=GW190425_GPS),
    ]
    files_by_id = {sid: {"p_astro.json": "url"} for sid in ("S190425z", "S190814bv")}
    payloads = {(sid, "p_astro.json"): json.dumps(BNS_PASTRO).encode() for sid in files_by_id}
    client = FakeGraceDbClient(superevents, files_by_id, payloads=payloads)
    install_fake_client(monkeypatch, client)

    df = gracedb_tools.fetch_gracedb_superevents(["bns"])

    assert df["superevent_id"].tolist() == ["S190425z", "S190814bv"]
    assert df.index.tolist() == [0, 1]


def test_fetch_gracedb_superevents_returns_an_empty_frame(monkeypatch, tmp_path) -> None:
    """Verify `fetch_gracedb_superevents` returns an empty DataFrame when nothing passes"""
    monkeypatch.chdir(tmp_path)
    install_fake_client(monkeypatch, FakeGraceDbClient([], {}))

    df = gracedb_tools.fetch_gracedb_superevents(["bns"])

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_temporal_crossmatch_matches_inside_the_window(df_sesn, gw_events) -> None:
    """Verify `temporal_crossmatch_sesn_to_gw` copies the event's fields onto each match"""
    matches = gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events)

    assert matches["name"].tolist() == ["2019ebq", "2019eff"]
    assert (matches["superevent_id"] == "S190425z").all()
    assert (matches["gw_time"] == GW190425_UTC).all()
    assert matches["days_from_gw"].tolist() == pytest.approx([4 / 24, 6 / 24])
    assert (matches["gw_p_bns"] == 0.95).all()
    assert (matches["gw_status"] == "ok").all()
    assert matches["gw_skymap_file"].tolist() == ["bayestar.multiorder.fits"] * 2


def test_temporal_crossmatch_respects_the_window_edges(df_sesn, gw_events) -> None:
    """Verify `temporal_crossmatch_sesn_to_gw` includes the window edges and excludes beyond them"""
    window_days = 14
    window = pd.Timedelta(days=window_days)
    df_sesn.loc[0, "discoverydate"] = GW190425_UTC - window
    df_sesn.loc[1, "discoverydate"] = GW190425_UTC + window + pd.Timedelta(seconds=1)

    matches = gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events, window_days=window_days)

    assert matches["name"].tolist() == ["2019ebq"]
    assert matches["days_from_gw"].iloc[0] == pytest.approx(-window_days)


def test_temporal_crossmatch_honors_the_window_argument(df_sesn, gw_events) -> None:
    """Verify `temporal_crossmatch_sesn_to_gw` widens and narrows with window_days"""
    df_sesn.loc[0, "discoverydate"] = GW190425_UTC + pd.Timedelta(days=20)
    df_sesn.loc[1, "discoverydate"] = GW190425_UTC + pd.Timedelta(days=40)

    assert gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events).empty
    assert gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events, window_days=30)[
        "name"
    ].tolist() == ["2019ebq"]


def test_temporal_crossmatch_repeats_overlapping_events(df_sesn, gw_events) -> None:
    """Verify `temporal_crossmatch_sesn_to_gw` emits one row per (SN, event) pair"""
    second = gw_events.iloc[0].copy()
    second["superevent_id"] = "S190426c"
    second["gw_time"] = GW190425_UTC + pd.Timedelta(days=1)
    gw_events = pd.concat([gw_events, second.to_frame().T], ignore_index=True)

    matches = gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events)

    assert len(matches) == 4
    assert matches["superevent_id"].tolist() == ["S190425z"] * 2 + ["S190426c"] * 2
    assert sorted(matches["name"].unique()) == ["2019ebq", "2019eff"]


def test_temporal_crossmatch_skips_events_without_a_time(df_sesn, gw_events) -> None:
    """Verify `temporal_crossmatch_sesn_to_gw` ignores events whose gw_time is missing"""
    gw_events.loc[0, "gw_time"] = pd.NaT

    assert gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events).empty


def test_temporal_crossmatch_returns_empty_frames(df_sesn, gw_events) -> None:
    """Verify `temporal_crossmatch_sesn_to_gw` returns an empty frame for empty or unmatched input"""
    assert gracedb_tools.temporal_crossmatch_sesn_to_gw(pd.DataFrame(), gw_events).empty
    assert gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, pd.DataFrame()).empty

    df_sesn["discoverydate"] = GW190425_UTC + pd.Timedelta(days=365)
    assert gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events).empty


def test_temporal_crossmatch_prints_when_verbose(df_sesn, gw_events, capsys) -> None:
    """Verify `temporal_crossmatch_sesn_to_gw` reports each event it checks when verbose"""
    gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events, verbose=True)

    assert "S190425z" in capsys.readouterr().out


def test_add_crossmatch_columns_copies_the_result(df_sesn) -> None:
    """Verify `add_crossmatch_columns` attaches every crossmatch field to the SN rows"""
    result = make_crossmatch_result(len(df_sesn))

    out = gracedb_tools.add_crossmatch_columns(
        df_sesn,
        result,
        cosmology_label="Planck18",
        distance_column="dist_mpc_Planck18",
        credible_level=TEST_CREDIBLE_LEVEL,
    )

    assert out["cosmology"].tolist() == ["Planck18"] * 2
    assert out["distance_column"].tolist() == ["dist_mpc_Planck18"] * 2
    assert out["sn_dist_mpc"].tolist() == df_sesn["dist_mpc_Planck18"].tolist()
    assert out["searched_area_deg2"].tolist() == [12.5] * 2
    assert out["searched_prob_2d"].tolist() == [0.1] * 2
    assert out["offset_deg"].tolist() == [1.5] * 2
    assert out["searched_prob_dist"].tolist() == [0.3] * 2
    assert out["searched_vol_mpc3"].tolist() == [1.0e6] * 2
    assert out["searched_prob_vol"].tolist() == [0.2] * 2
    assert out["searched_prob_3d_density_rank"].tolist() == out["searched_prob_vol"].tolist()
    assert out["probdensity_vol"].tolist() == [4.0e-9] * 2
    assert out["credible_volume_mpc3"].tolist() == [2.0e6] * 2
    assert out["credible_area_deg2"].tolist() == [300.0] * 2
    assert "searched_area_deg2" not in df_sesn.columns


def test_add_crossmatch_columns_flags_the_credible_level(df_sesn) -> None:
    """Verify `add_crossmatch_columns` treats credible_level as inclusive for both flags"""
    level = TEST_CREDIBLE_LEVEL
    result = make_crossmatch_result(len(df_sesn))
    result.searched_prob = np.array([level, level + 0.01])
    result.searched_prob_vol = np.array([level + 0.01, level])

    out = gracedb_tools.add_crossmatch_columns(
        df_sesn,
        result,
        cosmology_label="SHOES",
        distance_column="dist_mpc_SHOES",
        credible_level=TEST_CREDIBLE_LEVEL,
    )

    assert out["inside_2d_credible_level"].tolist() == [True, False]
    assert out["inside_3d_credible_level"].tolist() == [False, True]


def test_add_crossmatch_columns_handles_missing_contours(df_sesn) -> None:
    """Verify `add_crossmatch_columns` records NaN sizes when crossmatch returned no contours"""
    result = make_crossmatch_result(len(df_sesn), contours=False)

    out = gracedb_tools.add_crossmatch_columns(
        df_sesn,
        result,
        cosmology_label="SHOES",
        distance_column="dist_mpc_SHOES",
        credible_level=TEST_CREDIBLE_LEVEL,
    )

    assert out["credible_volume_mpc3"].isna().all()
    assert out["credible_area_deg2"].isna().all()


def test_add_crossmatch_columns_resets_the_index(df_sesn) -> None:
    """Verify `add_crossmatch_columns` lines the per-SN result arrays up positionally"""
    subset = df_sesn.iloc[[1]]
    result = make_crossmatch_result(1)

    out = gracedb_tools.add_crossmatch_columns(
        subset,
        result,
        cosmology_label="SHOES",
        distance_column="dist_mpc_SHOES",
        credible_level=TEST_CREDIBLE_LEVEL,
    )

    assert out.index.tolist() == [0]
    assert out["searched_prob_2d"].tolist() == [0.1]


def test_failed_spatial_rows_flags_the_failure(df_sesn) -> None:
    """Verify `failed_spatial_rows` marks rows as outside both credible levels"""
    out = gracedb_tools.failed_spatial_rows(df_sesn, "missing_skymap")

    assert out["spatial_status"].tolist() == ["missing_skymap"] * 2
    assert out["cosmology"].isna().all()
    assert not out["inside_2d_credible_level"].any()
    assert not out["inside_3d_credible_level"].any()
    assert "spatial_status" not in df_sesn.columns


def test_failed_spatial_rows_records_a_cosmology(df_sesn) -> None:
    """Verify `failed_spatial_rows` keeps the cosmology label for in-loop failures"""
    out = gracedb_tools.failed_spatial_rows(df_sesn, "crossmatch_failed: boom", "Planck18")

    assert out["cosmology"].tolist() == ["Planck18"] * 2


def install_fake_skymap(monkeypatch, skymap=None, read_error=None, crossmatch_error=None):
    """Replace read_sky_map and crossmatch with recording fakes; return their call logs."""
    reads = []
    crossmatches = []

    def fake_read_sky_map(path, moc=False):
        reads.append((str(path), moc))
        if read_error is not None:
            raise read_error
        return skymap if skymap is not None else FakeSkymap()

    def fake_crossmatch(read_skymap, coords, contours=None, cosmology=None):
        crossmatches.append((read_skymap, coords, contours, cosmology))
        if crossmatch_error is not None:
            raise crossmatch_error
        return make_crossmatch_result(len(coords))

    monkeypatch.setattr(gracedb_tools, "read_sky_map", fake_read_sky_map)
    monkeypatch.setattr(gracedb_tools, "crossmatch", fake_crossmatch)
    return reads, crossmatches


@pytest.fixture(name="temporal_matches")
def temporal_matches_fixture(df_sesn, gw_events, tmp_path):
    """Temporal matches against one superevent whose skymap exists on disk."""
    skymap_path = tmp_path / "S190425z__bayestar.multiorder.fits"
    skymap_path.write_bytes(b"FITS")
    gw_events.loc[0, "skymap_path"] = str(skymap_path)
    matches = gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events)
    return matches, gw_events


def test_run_3d_spatial_crossmatch_runs_every_cosmology(monkeypatch, temporal_matches) -> None:
    """Verify `run_3d_spatial_crossmatch` emits one row per (SN, event, cosmology)"""
    matches, gw_events = temporal_matches
    reads, crossmatches = install_fake_skymap(monkeypatch)

    df = gracedb_tools.run_3d_spatial_crossmatch(matches, gw_events)

    assert len(df) == 2 * len(COSMOLOGIES)
    assert sorted(df["cosmology"].unique()) == sorted(COSMOLOGIES)
    assert (df["spatial_status"] == "ok").all()
    assert df["name"].tolist() == ["2019ebq"] * len(COSMOLOGIES) + ["2019eff"] * len(COSMOLOGIES)
    assert df.index.tolist() == list(range(len(df)))
    assert len(reads) == 1
    assert reads[0][1] is True
    assert len(crossmatches) == len(COSMOLOGIES)
    assert crossmatches[0][2] == (RUN_3D_DEFAULT_CREDIBLE_LEVEL,)
    assert crossmatches[0][3] == gracedb_tools.USE_COMOVING_VOLUME_RANKING


def test_run_3d_spatial_crossmatch_uses_the_cosmology_distance(monkeypatch, temporal_matches) -> None:
    """Verify `run_3d_spatial_crossmatch` places each SN at the distance its cosmology gives it"""
    matches, gw_events = temporal_matches
    _, crossmatches = install_fake_skymap(monkeypatch)

    df = gracedb_tools.run_3d_spatial_crossmatch(matches, gw_events)

    for label in COSMOLOGIES:
        rows = df[df["cosmology"] == label]
        assert rows["distance_column"].tolist() == [f"dist_mpc_{label}"] * 2
        assert rows["sn_dist_mpc"].tolist() == rows[f"dist_mpc_{label}"].tolist()
    coords = crossmatches[0][1]
    assert coords.ra.deg.tolist() == pytest.approx([240.0, 241.0])
    assert coords.dec.deg.tolist() == pytest.approx([-20.0, -21.0])


def test_run_3d_spatial_crossmatch_skips_non_finite_distances(monkeypatch, temporal_matches) -> None:
    """Verify `run_3d_spatial_crossmatch` drops SNe with no distance under a given cosmology"""
    matches, gw_events = temporal_matches
    label = next(iter(COSMOLOGIES))
    matches.loc[0, f"dist_mpc_{label}"] = np.nan
    install_fake_skymap(monkeypatch)

    df = gracedb_tools.run_3d_spatial_crossmatch(matches, gw_events)

    assert df[df["cosmology"] == label]["name"].tolist() == ["2019eff"]
    assert len(df) == 2 * len(COSMOLOGIES) - 1


def test_run_3d_spatial_crossmatch_caches_skymaps(monkeypatch, temporal_matches) -> None:
    """Verify `run_3d_spatial_crossmatch` reads a shared skymap file only once"""
    matches, gw_events = temporal_matches
    second_event = gw_events.iloc[0].copy()
    second_event["superevent_id"] = "S190426c"
    gw_events = pd.concat([gw_events, second_event.to_frame().T], ignore_index=True)
    second_matches = matches.copy()
    second_matches["superevent_id"] = "S190426c"
    matches = pd.concat([matches, second_matches], ignore_index=True)
    reads, _ = install_fake_skymap(monkeypatch)

    df = gracedb_tools.run_3d_spatial_crossmatch(matches, gw_events)

    assert len(reads) == 1
    assert len(df) == 4 * len(COSMOLOGIES)


@pytest.mark.parametrize("missing", [None, np.nan, "", "does/not/exist.fits"])
def test_run_3d_spatial_crossmatch_reports_a_missing_skymap(monkeypatch, temporal_matches, missing) -> None:
    """Verify `run_3d_spatial_crossmatch` keeps rows for events with no usable skymap

    A skymap_path can be unusable in four ways, and all four are exercised here. It can be
    missing as None, which is what fetch_gracedb_superevents writes when there was no skymap
    or the download failed. It can be missing as NaN, which is what pandas fills in for a row
    that failed its file listing and so has no skymap_path key at all, and what pandas from
    3.0 stores for an assigned None even in an object column. Or it can be present but
    unusable: an empty string, or a path to a file that is not there.

    The NaN case is the one worth guarding. NaN is truthy, so a "not skymap_path" check
    passes it straight through to Path(), which raises TypeError on a float.
    """
    matches, gw_events = temporal_matches
    reads, _ = install_fake_skymap(monkeypatch)
    gw_events["skymap_path"] = gw_events["skymap_path"].astype(object)
    gw_events.loc[0, "skymap_path"] = missing

    df = gracedb_tools.run_3d_spatial_crossmatch(matches, gw_events)

    assert df["spatial_status"].tolist() == ["missing_skymap"] * 2
    assert not df["inside_2d_credible_level"].any()
    assert not df["inside_3d_credible_level"].any()
    assert reads == []


def test_run_3d_spatial_crossmatch_reports_a_read_failure(monkeypatch, temporal_matches) -> None:
    """Verify `run_3d_spatial_crossmatch` records the reason a skymap could not be read"""
    matches, gw_events = temporal_matches
    install_fake_skymap(monkeypatch, read_error=OSError("corrupt FITS"))

    df = gracedb_tools.run_3d_spatial_crossmatch(matches, gw_events)

    assert df["spatial_status"].tolist() == ["skymap_read_failed: corrupt FITS"] * 2
    assert df["cosmology"].isna().all()


def test_run_3d_spatial_crossmatch_requires_distance_columns(monkeypatch, temporal_matches) -> None:
    """Verify `run_3d_spatial_crossmatch` rejects a skymap without DISTMU"""
    matches, gw_events = temporal_matches
    install_fake_skymap(monkeypatch, skymap=FakeSkymap(colnames=("UNIQ", "PROBDENSITY")))

    df = gracedb_tools.run_3d_spatial_crossmatch(matches, gw_events)

    assert df["spatial_status"].tolist() == ["skymap_has_no_distance_columns"] * 2


def test_run_3d_spatial_crossmatch_reports_a_crossmatch_failure(monkeypatch, temporal_matches) -> None:
    """Verify `run_3d_spatial_crossmatch` records a per-cosmology crossmatch failure"""
    matches, gw_events = temporal_matches
    install_fake_skymap(monkeypatch, crossmatch_error=ValueError("bad skymap"))

    df = gracedb_tools.run_3d_spatial_crossmatch(matches, gw_events)

    assert len(df) == 2 * len(COSMOLOGIES)
    assert (df["spatial_status"] == "crossmatch_failed: bad skymap").all()
    assert sorted(df["cosmology"].unique()) == sorted(COSMOLOGIES)
    for label in COSMOLOGIES:
        rows = df[df["cosmology"] == label]
        assert rows["distance_column"].tolist() == [f"dist_mpc_{label}"] * 2
    assert not df["inside_2d_credible_level"].any()


def test_run_3d_spatial_crossmatch_skips_unknown_superevents(monkeypatch, temporal_matches) -> None:
    """Verify `run_3d_spatial_crossmatch` ignores matches whose event is not in the event table"""
    matches, gw_events = temporal_matches
    matches["superevent_id"] = "S000000a"
    install_fake_skymap(monkeypatch)

    assert gracedb_tools.run_3d_spatial_crossmatch(matches, gw_events).empty


def test_run_3d_spatial_crossmatch_returns_empty_frames(temporal_matches) -> None:
    """Verify `run_3d_spatial_crossmatch` returns an empty frame when either input is empty"""
    matches, gw_events = temporal_matches

    assert gracedb_tools.run_3d_spatial_crossmatch(pd.DataFrame(), gw_events).empty
    assert gracedb_tools.run_3d_spatial_crossmatch(matches, pd.DataFrame()).empty


def test_summarize_temporal_matches_counts_each_event(df_sesn, gw_events) -> None:
    """Verify `summarize_temporal_matches` counts the SNe that matched each superevent"""
    matches = gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events)

    summary = gracedb_tools.summarize_temporal_matches(matches, gw_events)

    assert summary["n_temporal_sesn"].tolist() == [2]
    assert summary["superevent_id"].tolist() == gw_events["superevent_id"].tolist()


def test_summarize_temporal_matches_keeps_unmatched_events(df_sesn, gw_events) -> None:
    """Verify `summarize_temporal_matches` keeps an event no SN matched, counted as zero"""
    quiet = gw_events.copy()
    quiet.loc[0, "superevent_id"] = "S190814bv"
    quiet.loc[0, "gw_time"] = GW190425_UTC + pd.Timedelta(days=365)
    events = pd.concat([gw_events, quiet], ignore_index=True)
    matches = gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, events)

    summary = gracedb_tools.summarize_temporal_matches(matches, events)

    assert summary["n_temporal_sesn"].tolist() == [2, 0]
    assert summary["superevent_id"].tolist() == ["S190425z", "S190814bv"]


def test_summarize_temporal_matches_preserves_the_event_columns(df_sesn, gw_events) -> None:
    """Verify `summarize_temporal_matches` adds its column without disturbing the others"""
    matches = gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events)

    summary = gracedb_tools.summarize_temporal_matches(matches, gw_events)

    assert list(summary.columns) == [*gw_events.columns, "n_temporal_sesn"]
    pd.testing.assert_frame_equal(summary[gw_events.columns], gw_events)


def test_summarize_temporal_matches_counts_zero_when_nothing_matched(gw_events) -> None:
    """Verify `summarize_temporal_matches` handles the bare frame an empty match returns"""
    summary = gracedb_tools.summarize_temporal_matches(pd.DataFrame(), gw_events)

    assert summary["n_temporal_sesn"].tolist() == [0]
    assert summary["n_temporal_sesn"].dtype == int


def test_summarize_temporal_matches_returns_an_empty_frame(df_sesn, gw_events) -> None:
    """Verify `summarize_temporal_matches` returns an empty frame when there are no events"""
    matches = gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events)

    assert gracedb_tools.summarize_temporal_matches(matches, pd.DataFrame()).empty


@pytest.fixture(name="spatial_matches")
def spatial_matches_fixture():
    """Four crossmatched rows spanning every combination of the two credible-level flags."""
    return pd.DataFrame(
        {
            "name": ["inside_both", "inside_3d_only", "inside_2d_only", "outside_both"],
            "spatial_status": ["ok"] * 4,
            "inside_2d_credible_level": [True, False, True, False],
            "inside_3d_credible_level": [True, True, False, False],
        }
    )


def test_select_coincidences_keeps_the_3d_matches(spatial_matches) -> None:
    """Verify `select_coincidences` cuts on the 3D flag alone by default"""
    kept = gracedb_tools.select_coincidences(spatial_matches)

    assert kept["name"].tolist() == ["inside_both", "inside_3d_only"]
    assert kept.index.tolist() == list(range(len(kept)))


def test_select_coincidences_can_require_the_2d_level(spatial_matches) -> None:
    """Verify `select_coincidences` also cuts on the 2D flag when asked to"""
    kept = gracedb_tools.select_coincidences(spatial_matches, require_2d_credible_level=True)

    assert kept["name"].tolist() == ["inside_both"]


def test_select_coincidences_drops_failed_crossmatches(spatial_matches) -> None:
    """Verify `select_coincidences` drops rows whose spatial_status is not "ok" """
    spatial_matches.loc[0, "spatial_status"] = "missing_skymap"

    kept = gracedb_tools.select_coincidences(spatial_matches)

    assert kept["name"].tolist() == ["inside_3d_only"]


def test_select_coincidences_treats_an_unmeasured_flag_as_outside(spatial_matches) -> None:
    """Verify `select_coincidences` reads the NaN left by a missing column as outside"""
    spatial_matches["inside_3d_credible_level"] = [True, np.nan, np.nan, np.nan]

    kept = gracedb_tools.select_coincidences(spatial_matches)

    assert kept["name"].tolist() == ["inside_both"]


def test_select_coincidences_returns_an_empty_frame(spatial_matches) -> None:
    """Verify `select_coincidences` returns an empty frame when the cut columns are absent"""
    assert gracedb_tools.select_coincidences(pd.DataFrame()).empty
    assert gracedb_tools.select_coincidences(spatial_matches.drop(columns="spatial_status")).empty
    assert gracedb_tools.select_coincidences(spatial_matches.drop(columns="inside_3d_credible_level")).empty
    # The 2D column is only required when the cut actually reads it.
    without_2d = spatial_matches.drop(columns="inside_2d_credible_level")
    assert not gracedb_tools.select_coincidences(without_2d).empty
    assert gracedb_tools.select_coincidences(without_2d, require_2d_credible_level=True).empty


def test_display_temporal_summary_shows_the_summary_columns(df_sesn, gw_events) -> None:
    """Verify `display_temporal_summary` keeps the columns it finds, in its own order"""
    matches = gracedb_tools.temporal_crossmatch_sesn_to_gw(df_sesn, gw_events)
    summary = gracedb_tools.summarize_temporal_matches(matches, gw_events)

    shown = gracedb_tools.display_temporal_summary(summary)

    assert list(shown.columns) == [
        "superevent_id",
        "gw_time",
        "far_per_year",
        "p_bns",
        "p_nsbh",
        "pipeline",
        "search",
        "skymap_file",
        "n_temporal_sesn",
        "status",
    ]
    assert shown["n_temporal_sesn"].tolist() == [2]


def test_display_temporal_summary_skips_absent_names(gw_events) -> None:
    """Verify `display_temporal_summary` drops a name the frame lacks rather than raising"""
    summary = gracedb_tools.summarize_temporal_matches(pd.DataFrame(), gw_events)

    shown = gracedb_tools.display_temporal_summary(summary.drop(columns=["pipeline", "status"]))

    assert "pipeline" not in shown.columns
    assert "status" not in shown.columns
    assert "n_temporal_sesn" in shown.columns
    assert gracedb_tools.display_temporal_summary(pd.DataFrame()).empty


def test_display_coincidences_shows_the_coincidence_columns(spatial_matches) -> None:
    """Verify `display_coincidences` keeps the columns it finds and drops the rest"""
    kept = gracedb_tools.select_coincidences(spatial_matches)

    shown = gracedb_tools.display_coincidences(kept)

    # spatial_status and the two flags are in the frame; only the flags are worth showing.
    assert list(shown.columns) == ["name", "inside_2d_credible_level", "inside_3d_credible_level"]
    assert gracedb_tools.display_coincidences(pd.DataFrame()).empty
