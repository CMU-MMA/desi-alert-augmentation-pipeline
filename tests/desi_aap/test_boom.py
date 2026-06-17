"""Tests for the BOOM alert-query module.

The unit tests run fully offline by stubbing out the network. The
``test_gold_standard`` regression test only runs when live BOOM credentials are
available (e.g. injected from GitHub secrets in CI) and the committed snapshot
exists; otherwise it is skipped.
"""

import os
from datetime import UTC, datetime

import gold_standard
import pandas as pd
import pytest
from astropy.time import Time
from desi_aap import boom


def test_to_jd_accepts_various_types() -> None:
    """`_to_jd` should handle JD floats, Time, datetime, and ISO strings."""
    assert boom._to_jd(2461187.0) == 2461187.0

    t = Time("2026-06-02T00:00:00", format="isot")
    assert boom._to_jd(t) == pytest.approx(t.jd)

    dt = datetime(2026, 6, 2, tzinfo=UTC)
    assert boom._to_jd(dt) == pytest.approx(Time(dt).jd)

    assert boom._to_jd("2026-06-02T00:00:00") == pytest.approx(t.jd)


def test_resolve_window_defaults_to_trailing_window() -> None:
    """With no bounds, the window is `default_window` wide and ends ~now."""
    from datetime import timedelta

    start_jd, end_jd = boom._resolve_window(None, None, timedelta(hours=1))
    assert end_jd - start_jd == pytest.approx(1.0 / 24.0)
    assert end_jd == pytest.approx(Time.now().jd, abs=1e-3)


def test_resolve_window_rejects_inverted_range() -> None:
    """start after end is an error."""
    from datetime import timedelta

    with pytest.raises(ValueError):
        boom._resolve_window(2461194.0, 2461187.0, timedelta(hours=1))


def test_extract_results_handles_response_shapes() -> None:
    """Results may be nested under data.results or live at the top level."""
    assert boom._extract_results({"data": {"results": [{"a": 1}]}}) == [{"a": 1}]
    assert boom._extract_results({"results": [{"b": 2}]}) == [{"b": 2}]
    assert boom._extract_results({"data": [{"c": 3}]}) == [{"c": 3}]
    assert boom._extract_results({}) == []


def test_query_alerts_requires_credentials(monkeypatch) -> None:
    """Missing username/password raises rather than calling the network."""
    monkeypatch.delenv("BOOM_USERNAME", raising=False)
    monkeypatch.delenv("BOOM_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="credentials"):
        boom.query_alerts()


def test_query_alerts_normalizes_results(monkeypatch) -> None:
    """End-to-end with the network stubbed: returns a normalized DataFrame."""

    def fake_get_access_token(username, password, **kwargs):
        assert (username, password) == ("user", "pass")
        return "fake-token"

    captured = {}

    def fake_run_filter_pipeline(*, start_jd, end_jd, pipeline, **kwargs):
        captured["start_jd"] = start_jd
        captured["end_jd"] = end_jd
        captured["pipeline"] = pipeline
        return {
            "data": {
                "results": [
                    {"objectId": "A", "candidate": {"ra": 10.0, "dec": -5.0}},
                    {"objectId": "B", "candidate": {"ra": 20.0, "dec": 5.0}},
                ]
            }
        }

    monkeypatch.setattr(boom, "get_access_token", fake_get_access_token)
    monkeypatch.setattr(boom, "_run_filter_pipeline", fake_run_filter_pipeline)

    df = boom.query_alerts(start=2461187.0, end=2461194.0, username="user", password="pass")

    assert list(df["objectId"]) == ["A", "B"]
    assert "candidate.ra" in df.columns
    assert captured["start_jd"] == 2461187.0
    assert captured["end_jd"] == 2461194.0
    assert captured["pipeline"] is boom.DEFAULT_PIPELINE


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


_HAVE_CREDS = bool(os.environ.get("BOOM_USERNAME") and os.environ.get("BOOM_PASSWORD"))


@pytest.mark.skipif(not _HAVE_CREDS, reason="BOOM credentials not configured")
@pytest.mark.skipif(
    not gold_standard.GOLD_CSV.exists(),
    reason="gold standard snapshot not generated; run tests/desi_aap/gold_standard.py",
)
def test_gold_standard() -> None:
    """Re-querying the fixed window should match the committed snapshot."""
    fresh = gold_standard.normalize_for_compare(gold_standard.fetch_gold_standard())
    gold = gold_standard.normalize_for_compare(pd.read_csv(gold_standard.GOLD_CSV))

    pd.testing.assert_frame_equal(fresh, gold, check_dtype=False, check_like=True, rtol=1e-6)
