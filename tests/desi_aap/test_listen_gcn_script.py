"""Tests for the scripts/listen_gcn.py command line.

The script is not importable as a module of the package, so it is loaded from its path. The
wiring is worth testing on its own: a flag that inverts the wrong way would silently make the
listener skip everything buffered, or replay it, with no error to notice.
"""

import importlib.util
from pathlib import Path

import pytest
from desi_aap import gcn_listener, gcn_notices, gcn_store

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "listen_gcn.py"


@pytest.fixture(scope="module")
def script():
    """Load scripts/listen_gcn.py as a module.

    Returns
    -------
    module
        The loaded script.
    """
    spec = importlib.util.spec_from_file_location("listen_gcn", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defaults_match_the_listener_module(script):
    """The CLI must not drift from the module's defaults, especially the group id."""
    args = script.parse_args([])
    assert args.store_root == gcn_store.STORE_ROOT
    assert args.topics == list(gcn_notices.DEFAULT_TOPICS)
    assert args.group_id == gcn_listener.CONSUMER_GROUP_ID
    assert args.domain == gcn_listener.GCN_DOMAIN
    assert args.once is False
    assert args.from_latest is False


def test_from_latest_flag_inverts_into_from_earliest(script, monkeypatch):
    """The flag is negative on the command line and positive in the API; check the inversion."""
    captured = {}

    def fake_run_listener(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(script, "run_listener", fake_run_listener)

    assert script.main([]) == 0
    assert captured["from_earliest"] is True

    assert script.main(["--from-latest"]) == 0
    assert captured["from_earliest"] is False


def test_arguments_are_passed_through(script, monkeypatch, tmp_path):
    """Every flag has to reach run_listener, or it is decoration."""
    captured = {}
    monkeypatch.setattr(script, "run_listener", lambda **kwargs: captured.update(kwargs) or {})
    script.main(
        [
            "--store-root",
            str(tmp_path),
            "--topics",
            gcn_notices.TOPIC_IGWN_GWALERT,
            "--group-id",
            "some-other-group",
            "--domain",
            gcn_listener.GCN_DOMAIN_TEST,
            "--once",
        ]
    )
    assert captured["root"] == tmp_path
    assert captured["topics"] == [gcn_notices.TOPIC_IGWN_GWALERT]
    assert captured["group_id"] == "some-other-group"
    assert captured["domain"] == gcn_listener.GCN_DOMAIN_TEST
    assert captured["once"] is True


def test_missing_credentials_exit_nonzero_without_a_traceback(script, monkeypatch, caplog):
    """The first-run failure should read as an instruction, not a stack trace."""

    def fail(**kwargs):
        raise RuntimeError("missing GCN credentials in environment: GCN_CLIENT_ID")

    monkeypatch.setattr(script, "run_listener", fail)
    assert script.main([]) == 1
    assert "missing GCN credentials" in caplog.text
