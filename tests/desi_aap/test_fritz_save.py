"""Saving candidates to Fritz: the order of calls, the redshift rule, and isolation of failures."""

import nested_pandas as npd
import pandas as pd
import pytest
import requests

from desi_aap import pipeline
from desi_aap.config import ConfigError, FritzConfig, Scotty750Config
from desi_aap.fritz import FritzClient, FritzError
from desi_aap.stages import fritz_save
from desi_aap.stages.base import StageResult
from desi_aap.stages.fritz_save import annotation_data, redshift_decision, run_fritz_save

GROUP = 2064


class FakeClient:
    """Records every call; `source` is what GET /api/sources returns."""

    def __init__(self, *, source=None, annotations=None, fail=()):
        self.calls = []
        self.source = source if source is not None else {"redshift": None}
        self.annotations = annotations or []
        self.fail = set(fail)

    def _record(self, name, *args):
        self.calls.append((name, *args))
        if name in self.fail:
            raise FritzError("POST", f"/{name}", 400, f"{name} refused")

    def save_from_broker(self, broker_id, obj_id, group_ids):
        """Record a broker save."""
        self._record("save_from_broker", broker_id, obj_id, tuple(group_ids))

    def save_source(self, obj_id, group_ids, *, ra=None, dec=None):
        """Record a group save."""
        self._record("save_source", obj_id, tuple(group_ids), ra, dec)

    def get_source(self, obj_id):
        """Record a read and return the scripted source."""
        self._record("get_source", obj_id)
        return self.source

    def set_redshift(self, obj_id, redshift, error, origin):
        """Record a redshift update."""
        self._record("set_redshift", obj_id, redshift, error, origin)

    def get_annotations(self, obj_id):
        """Record a read and return the scripted annotations."""
        self._record("get_annotations", obj_id)
        return self.annotations

    def post_annotation(self, obj_id, origin, data, group_ids):
        """Record a new annotation."""
        self._record("post_annotation", obj_id, origin, data, tuple(group_ids))

    def put_annotation(self, obj_id, annotation_id, origin, data, group_ids):
        """Record an annotation replacement."""
        self._record("put_annotation", obj_id, annotation_id, origin, data, tuple(group_ids))

    def names(self):
        """The names of the calls made, in order."""
        return [call[0] for call in self.calls]


def candidate(**overrides):
    row = {
        "objectId": "ZTF26aaaaaaa",
        "survey": "ZTF",
        "candidate.ra": 150.1,
        "candidate.dec": 2.2,
        "candidate.jd": 2461340.5,
        "candidate.magpsf": 20.5,
        "candidate.band": "g",
        "host_catalog": "desi_dr2",
        "host_redshift": 0.05,
        "host_redshift_err": float("nan"),
        "host_redshift_kind": "spec",
        "host_sep_arcsec": 0.2,
        "dist_mpc_SHOES": 225.0,
        "g_minus_r": -0.2,
        "annotations.drb": 0.99,
    }
    row.update(overrides)
    return row


@pytest.fixture
def fritz_config(pipeline_config):
    return pipeline_config.model_copy(
        update={
            "scotty750": Scotty750Config(),
            "fritz": FritzConfig(group_ids=[GROUP], filters=["scotty750"]),
        }
    )


def inputs_with(*rows):
    frame = npd.NestedFrame(pd.DataFrame(list(rows)))
    return {"scotty750": StageResult(stage="scotty750", frame=frame)}


def run(cfg, client, *rows):
    return run_fritz_save(cfg, inputs=inputs_with(*rows), stamp="20261027T000000Z", client=client)


def test_a_new_source_is_saved_given_a_redshift_and_annotated(fritz_config):
    client = FakeClient()
    result = run(fritz_config, client, candidate())

    assert client.names() == [
        "save_from_broker",
        "save_source",
        "get_source",
        "set_redshift",
        "get_annotations",
        "post_annotation",
    ]
    assert client.calls[0] == ("save_from_broker", 1, "ZTF26aaaaaaa", (GROUP,))
    _, _, z, err, origin = client.calls[3]
    assert z == pytest.approx(0.05)
    assert err is None
    assert origin == "DESI_AAP:desi_dr2:spec"
    record = result.frame.iloc[0]
    assert record["group_saved"] and record["broker_saved"]
    assert record["redshift_action"] == "set"
    assert record["annotation_action"] == "created"
    assert result.output_path.exists()


def test_the_group_save_follows_the_broker_save_even_for_an_existing_object(fritz_config):
    """The broker save adds only *new* objects to groups."""
    client = FakeClient(source={"redshift": 0.1, "redshift_origin": "someone"})
    run(fritz_config, client, candidate())
    assert client.names()[:2] == ["save_from_broker", "save_source"]


def test_someone_elses_redshift_is_left_alone(fritz_config):
    client = FakeClient(source={"redshift": 0.07, "redshift_origin": "TNS"})
    result = run(fritz_config, client, candidate())
    assert "set_redshift" not in client.names()
    assert result.frame.iloc[0]["redshift_action"] == "kept"


def test_our_photo_z_is_upgraded_to_a_spectrum(fritz_config):
    client = FakeClient(source={"redshift": 0.06, "redshift_origin": "DESI_AAP:dr11_photoz:photo"})
    result = run(fritz_config, client, candidate())
    assert "set_redshift" in client.names()
    assert result.frame.iloc[0]["redshift_action"] == "upgrade"


def test_our_photo_z_is_not_replaced_by_another_photo_z(fritz_config):
    source = {"redshift": 0.06, "redshift_origin": "DESI_AAP:dr11_photoz:photo"}
    row = pd.Series(candidate(host_redshift_kind="photo", host_catalog="dr11_photoz"))
    assert redshift_decision(source, row, fritz_config.fritz) is None


def test_redshifts_can_be_switched_off(fritz_config):
    cfg = fritz_config.model_copy(
        update={"fritz": FritzConfig(group_ids=[GROUP], filters=["scotty750"], set_redshift=False)}
    )
    client = FakeClient()
    run(cfg, client, candidate())
    assert "get_source" not in client.names()
    assert "set_redshift" not in client.names()


def test_an_existing_annotation_is_replaced(fritz_config):
    client = FakeClient(annotations=[{"id": 7, "origin": "other"}, {"id": 9, "origin": "DESI_AAP"}])
    result = run(fritz_config, client, candidate())
    put = [call for call in client.calls if call[0] == "put_annotation"]
    assert put and put[0][2] == 9
    assert result.frame.iloc[0]["annotation_action"] == "updated"


def test_a_failed_broker_save_still_saves_the_source_bare(fritz_config):
    client = FakeClient(fail={"save_from_broker"})
    result = run(fritz_config, client, candidate())
    assert client.calls[1] == ("save_source", "ZTF26aaaaaaa", (GROUP,), 150.1, 2.2)
    record = result.frame.iloc[0]
    assert not record["broker_saved"] and record["group_saved"]
    assert "broker save" in record["error"]


def test_one_failed_candidate_does_not_stop_the_others(fritz_config):
    client = FakeClient(fail={"save_source"})
    result = run(fritz_config, client, candidate(), candidate(objectId="ZTF26bbbbbbb"))
    assert len(result.frame) == 2
    assert result.frame["error"].notna().all()
    assert result.summary["n_errors"] == 2


def test_an_object_in_several_runs_rows_is_saved_once_from_its_latest_alert(fritz_config):
    client = FakeClient()
    result = run(fritz_config, client, candidate(**{"candidate.jd": 1.0}), candidate(**{"candidate.jd": 2.0}))
    assert len(result.frame) == 1
    posted = [call for call in client.calls if call[0] == "post_annotation"][0]
    assert posted[3]["jd_latest"] == 2.0


def test_the_annotation_is_json_safe_and_carries_the_boom_fields():
    data = annotation_data(pd.Series(candidate(g_minus_r=float("nan"))), ["scotty750"])
    assert "g_minus_r" not in data
    assert "host_z_err" not in data
    assert data["drb"] == 0.99
    assert data["host_z"] == 0.05
    assert data["filters"] == "scotty750"


def test_a_dry_run_makes_no_request(fritz_config):
    client = FakeClient()
    result = run_fritz_save(fritz_config, dry_run=True, inputs=inputs_with(candidate()), client=client)
    assert client.calls == []
    assert result.frame is None


def test_without_a_section_nothing_is_saved(pipeline_config):
    result = run_fritz_save(pipeline_config, inputs={})
    assert result.summary == {"skipped": "no [fritz] section"}


def test_a_missing_token_says_how_to_set_it(fritz_config, monkeypatch):
    monkeypatch.delenv("FRITZ_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="FRITZ_TOKEN"):
        run_fritz_save(fritz_config, inputs=inputs_with(candidate()))


def test_a_filter_that_does_not_exist_fails_the_run_before_it_starts(pipeline_config):
    cfg = pipeline_config.model_copy(update={"fritz": FritzConfig(group_ids=[GROUP], filters=["scoty750"])})
    with pytest.raises(ConfigError, match="scoty750"):
        pipeline.stages_for(cfg)


def test_the_stage_comes_last_and_requires_the_filters_it_saves(fritz_config):
    specs = pipeline.stages_for(fritz_config)
    assert specs[-1].name == fritz_save.STAGE
    assert specs[-1].requires == ("scotty750",)


# --- the client --------------------------------------------------------------


class FakeResponse:
    """A requests.Response stand-in: a status code and a JSON body (None for none)."""

    def __init__(self, status_code, body, reason="Bad Request"):
        self.status_code = status_code
        self._body = body
        self.reason = reason

    def json(self):
        """The body, or ValueError when there is none."""
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """A requests.Session stand-in that returns, or raises, one fixed response."""

    def __init__(self, response):
        self.headers = {}
        self.response = response
        self.requests = []

    def request(self, method, url, **kwargs):
        """Record the request and answer it."""
        self.requests.append((method, url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_the_client_returns_data_and_authenticates_with_the_token_scheme():
    session = FakeSession(FakeResponse(200, {"status": "success", "data": {"id": "x"}}))
    client = FritzClient("https://fritz.example/", "secret", min_request_interval_s=0, session=session)
    assert client.get_source("x") == {"id": "x"}
    assert session.headers["Authorization"] == "token secret"
    assert session.requests[0][1] == "https://fritz.example/api/sources/x"


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(400, {"status": "error", "message": "Annotation already exists"}),
        FakeResponse(200, {"status": "error", "message": "nope"}),
        FakeResponse(502, None),
        requests.ConnectionError("down"),
    ],
)
def test_the_client_raises_without_leaking_the_token(response):
    client = FritzClient(
        "https://fritz.example", "secret", min_request_interval_s=0, session=FakeSession(response)
    )
    with pytest.raises(FritzError) as raised:
        client.post_annotation("x", "DESI_AAP", {}, [GROUP])
    assert "secret" not in str(raised.value)
