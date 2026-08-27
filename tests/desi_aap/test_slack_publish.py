"""The slack_publish stage: credentials, message formatting, and when it posts."""

import logging

import pytest
from slack_sdk.errors import SlackApiError

from desi_aap import pipeline
from desi_aap.config import PipelineConfig
from desi_aap.stages import slack_publish
from desi_aap.stages.base import StageResult
from desi_aap.stages.crossmatch import run_crossmatch
from desi_aap.stages.localize import SLACK_DISPLAY as LOCALIZE_DISPLAY
from desi_aap.stages.localize import STAGE as LOCALIZE_STAGE
from desi_aap.stages.slack_publish import (
    STAGE,
    format_message,
    load_bot_token,
    post_message,
    run_slack_publish,
)

STAMP = "20260807T120000Z"


@pytest.fixture
def matches(pipeline_config, gold_standard_alerts):
    """A real frame to publish, nested match column and all.

    A crossmatch result rather than a filter's own output: what this stage does
    with a frame does not depend on which filter produced it, and building one
    without reaching GraceDB keeps the formatting tests independent of the GW
    machinery. It is labelled as the filter's below, which is all this stage
    reads.
    """
    inputs = {"query": StageResult(stage="query", frame=gold_standard_alerts)}
    result = run_crossmatch(pipeline_config, dry_run=True, inputs=inputs, stamp=STAMP)
    return StageResult(stage=LOCALIZE_STAGE, frame=result.frame, stamp=result.stamp)


@pytest.fixture
def match_inputs(matches):
    """What the localize filter would have handed this one."""
    return {LOCALIZE_STAGE: matches}


def test_run_posts_the_matches(slack_config, match_inputs, posted):
    result = run_slack_publish(slack_config, inputs=match_inputs, stamp=STAMP)

    assert result.summary["n_posted"] == 1
    (call,) = posted
    assert call["token"] == "xoxb-test-token"
    assert call["channel"] == "#desi-alerts"
    assert STAMP in call["text"]
    assert call["blocks"][1]["type"] == "table"


def test_the_message_lists_rows_and_cuts_off(matches):
    message = format_message(matches, LOCALIZE_DISPLAY, max_rows=5)

    section, table = message["blocks"][:2]
    assert "8 GW coincidence candidates found. Showing the first 5:" in section["text"]["text"]
    # One row per shown alert, plus the header row.
    rows = table["rows"]
    assert len(rows) == 6
    assert [cell["text"] for cell in rows[0]] == [
        "objectId",
        "candidate.ra",
        "candidate.dec",
        "candidate.magpsf",
        "candidate.band",
        "desi_dr1",
    ]
    # Every cell is raw_text -- Slack rejects its documented raw_number shape --
    # with measures right-aligned per column instead. The band is a letter, so
    # it stays left like the identifier rather than right like the magnitude.
    assert all(cell["type"] == "raw_text" for row in rows for cell in row)
    assert [setting["align"] for setting in table["column_settings"]] == [
        "left",
        "right",
        "right",
        "right",
        "left",
        "right",
    ]


def test_a_short_message_has_no_cutoff_line(matches):
    message = format_message(matches, LOCALIZE_DISPLAY, max_rows=20)

    section, table = message["blocks"][:2]
    assert "8 GW coincidence candidates found:" in section["text"]["text"]
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

    context = format_message(with_path, LOCALIZE_DISPLAY, max_rows=5)["blocks"][-1]
    assert context["type"] == "context"
    assert f"Full results: `{written}`" in context["elements"][0]["text"]
    # The dry-run result wrote nothing, so there is no path to point at.
    blocks = format_message(matches, LOCALIZE_DISPLAY, max_rows=5)["blocks"]
    assert all(block["type"] != "context" for block in blocks)


def test_no_slack_section_skips(pipeline_config, match_inputs, posted, caplog):
    with caplog.at_level(logging.INFO):
        result = run_slack_publish(pipeline_config, inputs=match_inputs, stamp=STAMP)

    assert posted == []
    assert result.summary["n_posted"] == 0
    assert "skipping" in caplog.text
    # It still reports what each filter would have contributed.
    assert result.summary["rows_by_filter"] == {LOCALIZE_STAGE: 8}


def test_dry_run_logs_the_message_without_posting(slack_config, match_inputs, posted, caplog):
    with caplog.at_level(logging.INFO):
        result = run_slack_publish(slack_config, dry_run=True, inputs=match_inputs, stamp=STAMP)

    assert posted == []
    assert result.summary["n_posted"] == 0
    assert "8 GW coincidence candidates found" in caplog.text


def test_an_empty_filter_posts_nothing(slack_config, posted):
    empty = {LOCALIZE_STAGE: StageResult(stage=LOCALIZE_STAGE, frame=None)}
    result = run_slack_publish(slack_config, inputs=empty, stamp=STAMP)

    assert posted == []
    # The filter ran and found nothing, which is not the same as not running:
    # it is present with a count of zero.
    assert result.summary["n_posted"] == 0
    assert result.summary["rows_by_filter"] == {LOCALIZE_STAGE: 0}


def test_a_filter_that_did_not_run_is_passed_over(slack_config, posted):
    """Verify a skipped or switched-off filter is silence, not an error.

    Unlike the data stages, a missing filter is normal here: the pipeline skips
    one whose input was empty, and switching one off is how a nightly run avoids
    repeating the hourly one's work.
    """
    result = run_slack_publish(slack_config, inputs={}, stamp=STAMP)

    assert posted == []
    # Absent entirely, rather than present with a zero: it never ran.
    assert result.summary["rows_by_filter"] == {}


def test_one_rejected_message_does_not_withhold_the_others(slack_config, match_inputs, monkeypatch):
    """Verify a Slack rejection is loud but does not silence the filters that could post.

    The filters are independent, so a rate limit hit while announcing one is no
    reason to withhold another -- but a partial post must not report as success
    either, so the failures are raised together once the postable ones are out.
    """
    attempted = []

    def flaky(token, channel, text, blocks=None):
        attempted.append(text)
        raise RuntimeError("Slack rejected the message: ratelimited.")

    monkeypatch.setattr(slack_publish, "post_message", flaky)

    with pytest.raises(RuntimeError, match=r"Posted 0 of 1 filter message\(s\).*ratelimited"):
        run_slack_publish(slack_config, inputs=match_inputs, stamp=STAMP)

    # It got as far as trying, rather than bailing out before the attempt.
    assert len(attempted) == 1


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


def test_the_stage_runs_last_and_a_quiet_run_reaches_it(slack_config, stub_boom, stub_gracedb, posted):
    """Verify the run gets all the way here rather than ending at an empty filter.

    This is what the fan-out bought. Under the old stop-on-empty pipeline,
    localize ran before this stage and was empty on most runs, so the run ended
    early and never reached slack_publish at all. Now every stage is accounted
    for, and this one is still last.

    With one filter configured, every filter being empty does leave nothing to
    announce, so the stage is skipped rather than run -- but it is *reached*,
    which is the part that used to be untrue, and the skip is recorded rather
    than being the silent end of the run.
    """
    stub_gracedb.searched_prob_vol = 0.99  # nothing lands inside the credible volume

    results = pipeline.run_pipeline(slack_config, stamp=STAMP)

    assert list(results)[-1] == STAGE
    assert set(results) == set(pipeline.STAGE_ORDER)
    assert results[LOCALIZE_STAGE].is_empty
    assert results[STAGE].summary["skipped"] == "no input"
    assert posted == []
