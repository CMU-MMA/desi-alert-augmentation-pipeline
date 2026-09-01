"""The slack_publish stage: credentials, message formatting, and when it posts."""

import logging

import pytest
from slack_sdk.errors import SlackApiError

from desi_aap.config import PipelineConfig, SlackConfig
from desi_aap.stages import slack_publish
from desi_aap.stages.base import StageResult
from desi_aap.stages.crossmatch import STAGE as CROSSMATCH_STAGE
from desi_aap.stages.crossmatch import run_crossmatch
from desi_aap.stages.slack_publish import (
    STAGE,
    format_message,
    load_bot_token,
    post_message,
    run_slack_publish,
)

STAMP = "20260807T120000Z"
DEFAULT_COLUMNS = ["objectId", "candidate.ra", "candidate.dec"]


@pytest.fixture
def slack_credentials(tmp_path):
    """A credentials file holding a fake bot token."""
    path = tmp_path / "slack.toml"
    path.write_text('bot_token = "xoxb-test-token"\n')
    return path


@pytest.fixture
def slack_config(pipeline_config, slack_credentials):
    """The shared config, with a [slack] section pointing at the fake credentials."""
    section = SlackConfig(credentials=slack_credentials, channel="#desi-alerts", max_rows=5)
    return pipeline_config.model_copy(update={"slack": section})


@pytest.fixture
def matches(pipeline_config, gold_standard_alerts):
    """A real crossmatch result to publish, nested match column and all."""
    inputs = {"query": StageResult(stage="query", frame=gold_standard_alerts)}
    return run_crossmatch(pipeline_config, dry_run=True, inputs=inputs, stamp=STAMP)


@pytest.fixture
def match_inputs(matches):
    """What the crossmatch stage would have handed this one."""
    return {CROSSMATCH_STAGE: matches}


@pytest.fixture
def posted(monkeypatch):
    """Capture what would have gone to Slack instead of calling the Web API."""
    calls = []

    class FakeWebClient:
        def __init__(self, token):
            self.token = token

        def chat_postMessage(self, **kwargs):  # noqa: N802 -- the slack_sdk method name
            calls.append({"token": self.token, **kwargs})

    monkeypatch.setattr(slack_publish, "WebClient", FakeWebClient)
    return calls


def test_run_posts_the_matches(slack_config, match_inputs, posted):
    result = run_slack_publish(slack_config, inputs=match_inputs, stamp=STAMP)

    assert result.summary["posted"] is True
    (call,) = posted
    assert call["token"] == "xoxb-test-token"
    assert call["channel"] == "#desi-alerts"
    assert STAMP in call["text"]
    assert call["blocks"][1]["type"] == "table"


def test_the_message_lists_rows_and_cuts_off(matches):
    message = format_message(matches, max_rows=5, display_columns=DEFAULT_COLUMNS)

    section, table = message["blocks"][:2]
    assert "8 candidates found. Showing the first 5:" in section["text"]["text"]
    # One row per shown alert, plus the header row.
    rows = table["rows"]
    assert len(rows) == 6
    assert [cell["text"] for cell in rows[0]] == ["objectId", "candidate.ra", "candidate.dec", "desi_dr1"]
    # Every cell is raw_text -- Slack rejects its documented raw_number shape --
    # with measures right-aligned per column instead.
    assert all(cell["type"] == "raw_text" for row in rows for cell in row)
    assert [setting["align"] for setting in table["column_settings"]] == ["left", "right", "right", "right"]


def test_a_short_message_has_no_cutoff_line(matches):
    message = format_message(matches, max_rows=20, display_columns=DEFAULT_COLUMNS)

    section, table = message["blocks"][:2]
    assert "8 candidates found:" in section["text"]["text"]
    assert "Showing the first" not in section["text"]["text"]
    assert len(table["rows"]) == 9


def test_the_message_names_the_output_file(matches, tmp_path):
    written = tmp_path / "matches.parquet"
    with_path = StageResult(
        stage=matches.stage,
        frame=matches.frame,
        output_path=written,
        stamp=matches.stamp,
        summary=matches.summary,
    )

    context = format_message(with_path, max_rows=5, display_columns=DEFAULT_COLUMNS)["blocks"][-1]
    assert context["type"] == "context"
    assert f"Full results: `{written}`" in context["elements"][0]["text"]
    # The dry-run result wrote nothing, so there is no path to point at.
    blocks = format_message(matches, max_rows=5, display_columns=DEFAULT_COLUMNS)["blocks"]
    assert all(block["type"] != "context" for block in blocks)


def test_configured_columns_choose_and_order_the_table(matches):
    message = format_message(matches, max_rows=5, display_columns=["candidate.dec", "objectId"])

    rows = message["blocks"][1]["rows"]
    assert [cell["text"] for cell in rows[0]] == ["candidate.dec", "objectId", "desi_dr1"]


def test_a_configured_column_the_frame_lacks_warns_and_is_skipped(matches, caplog):
    with caplog.at_level(logging.WARNING):
        message = format_message(matches, max_rows=5, display_columns=["objectId", "no_such_column"])

    assert "no_such_column" in caplog.text
    rows = message["blocks"][1]["rows"]
    assert [cell["text"] for cell in rows[0]] == ["objectId", "desi_dr1"]


def test_run_uses_the_configured_columns(slack_config, match_inputs, posted):
    narrowed = slack_config.slack.model_copy(update={"columns": ["objectId"]})
    cfg = slack_config.model_copy(update={"slack": narrowed})

    run_slack_publish(cfg, inputs=match_inputs, stamp=STAMP)

    (call,) = posted
    header_row = call["blocks"][1]["rows"][0]
    assert [cell["text"] for cell in header_row] == ["objectId", "desi_dr1"]


def test_no_slack_section_skips(pipeline_config, match_inputs, posted, caplog):
    with caplog.at_level(logging.INFO):
        result = run_slack_publish(pipeline_config, inputs=match_inputs, stamp=STAMP)

    assert posted == []
    assert result.summary["posted"] is False
    assert "skipping" in caplog.text
    # The frame passes through, so this stage never reads as the one that ended the run.
    assert not result.is_empty


def test_dry_run_logs_the_message_without_posting(slack_config, match_inputs, posted, caplog):
    with caplog.at_level(logging.INFO):
        result = run_slack_publish(slack_config, dry_run=True, inputs=match_inputs, stamp=STAMP)

    assert posted == []
    assert result.summary["posted"] is False
    assert "8 candidates found" in caplog.text


def test_an_empty_upstream_posts_nothing(slack_config, posted):
    empty = {CROSSMATCH_STAGE: StageResult(stage=CROSSMATCH_STAGE, frame=None)}
    result = run_slack_publish(slack_config, inputs=empty, stamp=STAMP)

    assert posted == []
    assert result.summary == {"posted": False, "n_rows": 0}


def test_a_missing_upstream_names_the_stage_that_must_run_first(slack_config):
    with pytest.raises(KeyError, match=CROSSMATCH_STAGE):
        run_slack_publish(slack_config, inputs=None, stamp=STAMP)


def test_a_missing_credentials_file_says_how_to_make_one(tmp_path):
    with pytest.raises(ValueError, match="bot_token"):
        load_bot_token(tmp_path / "nowhere.toml")


def test_credentials_without_a_token_are_rejected(tmp_path):
    path = tmp_path / "slack.toml"
    path.write_text('other_key = "value"\n')

    with pytest.raises(ValueError, match="bot_token"):
        load_bot_token(path)


def test_the_token_is_read_from_the_file(slack_credentials):
    assert load_bot_token(slack_credentials) == "xoxb-test-token"


def test_a_slack_error_names_the_code_and_the_fix(monkeypatch):
    class RejectingWebClient:
        def __init__(self, token):
            pass

        def chat_postMessage(self, **kwargs):  # noqa: N802 -- the slack_sdk method name
            raise SlackApiError("rejected", {"error": "not_in_channel"})

    monkeypatch.setattr(slack_publish, "WebClient", RejectingWebClient)

    with pytest.raises(RuntimeError, match="not_in_channel.*invited"):
        post_message("xoxb-test-token", "#desi-alerts", "hello")


def test_a_slack_error_surfaces_the_schema_details(monkeypatch):
    details = [f"[ERROR] problem {i} [json-pointer:/blocks/1]" for i in range(7)]
    response = {"error": "invalid_blocks", "response_metadata": {"messages": details}}

    class RejectingWebClient:
        def __init__(self, token):
            pass

        def chat_postMessage(self, **kwargs):  # noqa: N802 -- the slack_sdk method name
            raise SlackApiError("rejected", response)

    monkeypatch.setattr(slack_publish, "WebClient", RejectingWebClient)

    with pytest.raises(RuntimeError, match=r"invalid_blocks.*problem 0.*problem 4.*\(\+2 more\)"):
        post_message("xoxb-test-token", "#desi-alerts", "hello")


def test_max_rows_must_be_positive(slack_credentials):
    with pytest.raises(ValueError, match="max_rows"):
        PipelineConfig.model_validate(
            {
                "run": {"output_dir": "out"},
                "query": {"boom": {"survey": "LSST"}, "window": {"lookback": "1h"}},
                "slack": {"credentials": str(slack_credentials), "channel": "#c", "max_rows": 0},
            }
        )


# TODO(#45): localize now runs before this stage and is empty on most runs, so
# run_pipeline's stop-on-empty ends the run before slack_publish is reached. The
# assertion below is left as written -- it is the invariant we still want -- so
# strict xfail reports XPASS the moment the stage order is settled.
@pytest.mark.xfail(
    reason="#45: localize precedes slack_publish and stops the run when it finds no coincidence",
    strict=True,
)
def test_the_stage_runs_last(slack_config, stub_boom, posted):
    from desi_aap import pipeline

    results = pipeline.run_pipeline(slack_config, stamp=STAMP)

    assert list(results)[-1] == STAGE
    (call,) = posted
    assert "8 candidates found" in call["text"]
