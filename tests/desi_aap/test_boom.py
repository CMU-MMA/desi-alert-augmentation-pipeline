"""Tests for the BOOM alert-query module.

The unit tests run fully offline by stubbing out the network. The
``test_gold_standard`` regression test only runs when live BOOM credentials are
available (e.g. injected from GitHub secrets in CI) and the committed snapshot
exists; otherwise it is skipped.
"""

import os
from datetime import UTC, datetime, timedelta

import gold_standard
import nested_pandas as npd
import pytest
from astropy.time import Time

from desi_aap import boom

# A pair of records shaped like a /filters/test response: dotted sub-documents
# plus the cross_matches.LSPSC list of structs.
RECORDS = [
    {
        "objectId": "A",
        "candidate": {"ra": 10.0, "dec": -5.0},
        "cross_matches": {
            "LSPSC": [
                {"_id": 1, "mag_white": 20.5, "score": 0.1},
                {"_id": 2, "mag_white": 21.5, "score": 0.2},
            ]
        },
    },
    {
        "objectId": "B",
        "candidate": {"ra": 20.0, "dec": 5.0},
        "cross_matches": {"LSPSC": []},
    },
]


def test_to_jd_accepts_various_types() -> None:
    """`_to_jd` should handle JD floats, Time, datetime, and ISO strings."""
    assert boom._to_jd(2461187.0) == 2461187.0

    t = Time("2026-06-02T00:00:00", format="isot")
    assert boom._to_jd(t) == pytest.approx(t.jd)

    dt = datetime(2026, 6, 2, tzinfo=UTC)
    assert boom._to_jd(dt) == pytest.approx(Time(dt).jd)

    assert boom._to_jd("2026-06-02T00:00:00") == pytest.approx(t.jd)


def test_parse_timedelta_accepts_units_and_combinations() -> None:
    """Durations are given as a value plus a unit, optionally concatenated."""
    assert boom.parse_timedelta("30s") == timedelta(seconds=30)
    assert boom.parse_timedelta("90m") == timedelta(minutes=90)
    assert boom.parse_timedelta("2h") == timedelta(hours=2)
    assert boom.parse_timedelta("7d") == timedelta(days=7)
    assert boom.parse_timedelta("1w") == timedelta(weeks=1)
    assert boom.parse_timedelta("1.5h") == timedelta(hours=1, minutes=30)
    assert boom.parse_timedelta("1d12h30m") == timedelta(days=1, hours=12, minutes=30)
    assert boom.parse_timedelta(" 2H ") == timedelta(hours=2)


@pytest.mark.parametrize("text", ["", "7", "7x", "h", "-1h", "two hours"])
def test_parse_timedelta_rejects_junk(text: str) -> None:
    """A bare number, an unknown unit, or a negative value is an error."""
    with pytest.raises(ValueError, match="duration"):
        boom.parse_timedelta(text)


def test_resolve_window_defaults_to_trailing_window() -> None:
    """With no bounds, the window is `window` wide and ends ~now."""
    start_jd, end_jd = boom._resolve_window(None, None, timedelta(hours=1))
    assert end_jd - start_jd == pytest.approx(1.0 / 24.0)
    assert end_jd == pytest.approx(Time.now().jd, abs=1e-3)


def test_resolve_window_fills_in_the_missing_bound() -> None:
    """`window` extends backwards from `end` and forwards from `start`."""
    start_jd, end_jd = boom._resolve_window(None, 2461194.0, timedelta(days=7))
    assert (start_jd, end_jd) == (2461187.0, 2461194.0)

    start_jd, end_jd = boom._resolve_window(2461187.0, None, timedelta(days=7))
    assert (start_jd, end_jd) == (2461187.0, 2461194.0)


def test_resolve_window_ignores_window_when_both_bounds_given() -> None:
    """Explicit bounds win; `window` is unused."""
    assert boom._resolve_window(2461187.0, 2461194.0, timedelta(hours=1)) == (2461187.0, 2461194.0)


@pytest.mark.parametrize("window", [timedelta(0), timedelta(hours=-1)])
def test_resolve_window_rejects_nonpositive_window(window: timedelta) -> None:
    """A zero or negative window is an error."""
    with pytest.raises(ValueError, match="positive"):
        boom._resolve_window(None, None, window)


def test_resolve_window_rejects_non_timedelta_window() -> None:
    """`window` must be a timedelta, not e.g. a number of hours."""
    with pytest.raises(TypeError, match="timedelta"):
        boom._resolve_window(None, None, 1)


def test_resolve_window_rejects_inverted_range() -> None:
    """start after end is an error."""
    with pytest.raises(ValueError, match="must not be after"):
        boom._resolve_window(2461194.0, 2461187.0, timedelta(hours=1))


def test_extract_results_handles_response_shapes() -> None:
    """Results may be nested under data.results or live at the top level."""
    assert boom._extract_results({"data": {"results": [{"a": 1}]}}) == [{"a": 1}]
    assert boom._extract_results({"results": [{"b": 2}]}) == [{"b": 2}]
    assert boom._extract_results({"data": [{"c": 3}]}) == [{"c": 3}]
    assert boom._extract_results({}) == []


def test_nested_column_name() -> None:
    """Known fields get a short name; anything else keeps its (de-dotted) path."""
    assert boom.nested_column_name("cross_matches.LSPSC") == "lspsc"
    assert boom.nested_column_name("cross_matches.OTHER") == "cross_matches_OTHER"
    assert boom.nested_column_name("cross_matches.LSPSC", {"cross_matches.LSPSC": "xm"}) == "xm"


def test_to_nested_frame_packs_list_columns() -> None:
    """The LSPSC list of structs becomes a queryable nested column."""
    nested = boom.to_nested_frame(RECORDS)

    assert isinstance(nested, npd.NestedFrame)
    assert nested.nested_columns == ["lspsc"]
    assert "cross_matches.LSPSC" not in nested.columns
    assert list(nested["objectId"]) == ["A", "B"]
    assert "candidate.ra" in nested.columns

    # Sub-columns are addressed as "<nested>.<field>" and only the two
    # cross-matches of record A contribute rows.
    assert list(nested["lspsc.score"]) == [0.1, 0.2]
    assert len(nested["lspsc"].nest.to_flat()) == 2
    assert list(nested["lspsc"].nest.to_flat().columns) == ["_id", "mag_white", "score"]


def test_to_nested_frame_honors_custom_names() -> None:
    """`nested_names` overrides the nested column name."""
    nested = boom.to_nested_frame(RECORDS, nested_names={"cross_matches.LSPSC": "xmatch"})
    assert nested.nested_columns == ["xmatch"]
    assert list(nested["xmatch.score"]) == [0.1, 0.2]


def test_to_nested_frame_handles_missing_matches() -> None:
    """A record with no cross_matches key at all is still packed (as null)."""
    nested = boom.to_nested_frame([*RECORDS, {"objectId": "C", "candidate": {"ra": 1.0, "dec": 2.0}}])
    assert nested.nested_columns == ["lspsc"]
    assert nested["lspsc"].isna().tolist() == [False, False, True]


def test_to_nested_frame_warns_when_nothing_to_pack() -> None:
    """With no non-empty list anywhere there is no struct schema to infer."""
    records = [{"objectId": "A", "cross_matches": {"LSPSC": []}}]
    with pytest.warns(UserWarning, match="Could not pack"):
        nested = boom.to_nested_frame(records)
    assert nested.nested_columns == []
    assert nested["lspsc"].tolist() == [[]]


def test_to_nested_frame_of_empty_results() -> None:
    """No matching alerts gives an empty frame rather than an error."""
    nested = boom.to_nested_frame([])
    assert isinstance(nested, npd.NestedFrame)
    assert nested.empty
    assert list(nested.columns) == []


def test_query_alerts_requires_credentials(monkeypatch) -> None:
    """Missing username/password raises rather than calling the network."""
    monkeypatch.delenv("BOOM_USERNAME", raising=False)
    monkeypatch.delenv("BOOM_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="credentials"):
        boom.query_alerts()


@pytest.fixture
def stub_boom(monkeypatch):
    """Stub out auth and the data query, capturing the resolved arguments."""
    captured: dict = {}

    def fake_get_access_token(username, password, **kwargs):
        assert (username, password) == ("user", "pass")
        return "fake-token"

    def fake_run_filter_pipeline(*, start_jd, end_jd, pipeline, **kwargs):
        captured["start_jd"] = start_jd
        captured["end_jd"] = end_jd
        captured["pipeline"] = pipeline
        return {"data": {"results": RECORDS}}

    monkeypatch.setattr(boom, "get_access_token", fake_get_access_token)
    monkeypatch.setattr(boom, "_run_filter_pipeline", fake_run_filter_pipeline)
    return captured


def test_query_alerts_normalizes_results(stub_boom) -> None:
    """End-to-end with the network stubbed: returns a normalized NestedFrame."""
    alerts = boom.query_alerts(start=2461187.0, end=2461194.0, username="user", password="pass")

    assert isinstance(alerts, npd.NestedFrame)
    assert list(alerts["objectId"]) == ["A", "B"]
    assert "candidate.ra" in alerts.columns
    assert alerts.nested_columns == ["lspsc"]
    assert stub_boom["start_jd"] == 2461187.0
    assert stub_boom["end_jd"] == 2461194.0
    assert stub_boom["pipeline"] == boom.load_default_pipeline()


def test_query_alerts_accepts_a_timedelta_window(stub_boom) -> None:
    """`window` sets the width of the queried range."""
    boom.query_alerts(end=2461194.0, window=timedelta(days=7), username="user", password="pass")
    assert (stub_boom["start_jd"], stub_boom["end_jd"]) == (2461187.0, 2461194.0)

    boom.query_alerts(start=2461187.0, window=timedelta(days=7), username="user", password="pass")
    assert (stub_boom["start_jd"], stub_boom["end_jd"]) == (2461187.0, 2461194.0)

    boom.query_alerts(window=timedelta(minutes=30), username="user", password="pass")
    assert stub_boom["end_jd"] - stub_boom["start_jd"] == pytest.approx(1.0 / 48.0)


def test_query_alerts_round_trips_through_parquet(stub_boom, tmp_path) -> None:
    """The returned frame, nested column included, survives a parquet round trip."""
    alerts = boom.query_alerts(start=2461187.0, end=2461194.0, username="user", password="pass")
    out = tmp_path / "alerts.parquet"
    alerts.to_parquet(out)

    back = npd.read_parquet(out)
    assert back.nested_columns == ["lspsc"]
    assert list(back["objectId"]) == ["A", "B"]
    assert list(back["lspsc.score"]) == [0.1, 0.2]
    gold_standard.assert_alerts_equal(
        gold_standard.normalize_for_compare(alerts),
        gold_standard.normalize_for_compare(back),
    )


def test_assert_alerts_equal_detects_nested_differences() -> None:
    """The gold-standard comparison must not be blind to the nested column."""
    fresh = gold_standard.normalize_for_compare(boom.to_nested_frame(RECORDS))
    perturbed = [{**RECORDS[0], "cross_matches": {"LSPSC": [{"_id": 1, "mag_white": 20.5, "score": 0.9}]}}]
    other = gold_standard.normalize_for_compare(boom.to_nested_frame([*perturbed, RECORDS[1]]))

    with pytest.raises(AssertionError):
        gold_standard.assert_alerts_equal(fresh, other)


def test_load_default_pipeline() -> None:
    """The default pipeline loads from JSON with booleans intact."""
    pipeline = boom.load_default_pipeline()
    assert isinstance(pipeline, list)
    assert len(pipeline) == 3
    # JSON `true`/`false` must round-trip to Python bools for the API payload.
    match_clauses = pipeline[1]["$match"]["$and"][0]["$and"]
    isdiffpos = next(c["candidate.isdiffpos"] for c in match_clauses if "candidate.isdiffpos" in c)
    assert isdiffpos == {"$in": [True]}


def test_get_access_token_parses_response(monkeypatch) -> None:
    """`get_access_token` posts credentials and returns the access_token."""

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "abc123", "token_type": "bearer"}

    def fake_post(url, headers=None, data=None, timeout=None):
        assert url.endswith("/auth")
        assert data == {"username": "user", "password": "pass"}
        return FakeResponse()

    monkeypatch.setattr(boom.requests, "post", fake_post)
    assert boom.get_access_token("user", "pass") == "abc123"


def test_gold_standard_snapshot_is_nested() -> None:
    """The committed snapshot is parquet with the LSPSC matches still nested."""
    gold = gold_standard.load_gold_standard()
    assert gold.nested_columns == ["lspsc"]
    assert len(gold) > 0
    assert {"objectId", "candidate.jd", "candidate.ra"} <= set(gold.columns)
    assert not gold["lspsc"].nest.to_flat().empty


_HAVE_CREDS = bool(os.environ.get("BOOM_USERNAME") and os.environ.get("BOOM_PASSWORD"))


@pytest.mark.skipif(not _HAVE_CREDS, reason="BOOM credentials not configured")
@pytest.mark.skipif(
    not gold_standard.GOLD_PARQUET.exists(),
    reason="gold standard snapshot not generated; run tests/desi_aap/gold_standard.py",
)
def test_gold_standard() -> None:
    """Re-querying the fixed window should match the committed snapshot."""
    fresh = gold_standard.normalize_for_compare(gold_standard.fetch_gold_standard())
    gold = gold_standard.normalize_for_compare(gold_standard.load_gold_standard())

    gold_standard.assert_alerts_equal(fresh, gold)
