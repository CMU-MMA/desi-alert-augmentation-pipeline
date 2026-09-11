"""The slack_publish stage: credentials, message formatting, and when it posts."""

import logging

import pytest
from slack_sdk.errors import SlackApiError

from desi_aap import pipeline
from desi_aap.config import PipelineConfig, SlackConfig
from desi_aap.stages import slack_publish
from desi_aap.stages.base import SlackDisplay, StageResult
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
# What every filter's table starts with when [slack] columns is left alone.
DEFAULT_COLUMNS = ["objectId", "candidate.ra", "candidate.dec", "candidate.magpsf", "candidate.band"]


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
    message = format_message(matches, LOCALIZE_DISPLAY, max_rows=5, display_columns=DEFAULT_COLUMNS)

    section, table = message["blocks"][:2]
    assert "8 GW coincidence candidates found. Showing the first 5:" in section["text"]["text"]
    # One row per shown alert, plus the header row.
    rows = table["rows"]
    assert len(rows) == 6
    # The configured columns; the filter's own (superevent, host redshift,
    # distance) are not on a crossmatch frame, so they are skipped.
    assert [cell["text"] for cell in rows[0]] == DEFAULT_COLUMNS
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
    ]


def test_a_short_message_has_no_cutoff_line(matches):
    message = format_message(matches, LOCALIZE_DISPLAY, max_rows=20, display_columns=DEFAULT_COLUMNS)

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

    context = format_message(with_path, LOCALIZE_DISPLAY, max_rows=5, display_columns=DEFAULT_COLUMNS)[
        "blocks"
    ][-1]
    assert context["type"] == "context"
    assert f"Full results: `{written}`" in context["elements"][0]["text"]
    # The dry-run result wrote nothing, so there is no path to point at.
    blocks = format_message(matches, LOCALIZE_DISPLAY, max_rows=5, display_columns=DEFAULT_COLUMNS)["blocks"]
    assert all(block["type"] != "context" for block in blocks)


def test_configured_columns_choose_and_order_the_table(matches):
    message = format_message(
        matches, LOCALIZE_DISPLAY, max_rows=5, display_columns=["candidate.dec", "objectId"]
    )

    rows = message["blocks"][1]["rows"]
    assert [cell["text"] for cell in rows[0]] == ["candidate.dec", "objectId"]


def test_the_filters_own_columns_follow_the_configured_ones(matches, caplog):
    display = SlackDisplay(title="candidate", columns=("candidate.band", "only_some_runs_have_this"))

    with caplog.at_level(logging.INFO):
        message = format_message(matches, display, max_rows=5, display_columns=["objectId", "candidate.ra"])

    rows = message["blocks"][1]["rows"]
    assert [cell["text"] for cell in rows[0]] == ["objectId", "candidate.ra", "candidate.band"]
    # A filter may name a column only some runs produce, so that is a note, not
    # the warning a mis-configured [slack] column gets.
    assert "only_some_runs_have_this" in caplog.text
    assert "Configured [slack] column(s)" not in caplog.text


def test_a_configured_column_the_frame_lacks_warns_and_is_skipped(matches, caplog):
    with caplog.at_level(logging.WARNING):
        message = format_message(
            matches,
            LOCALIZE_DISPLAY,
            max_rows=5,
            display_columns=["objectId", "no_such_column", "desi_dr1.NO_SUCH_FIELD"],
        )

    assert "no_such_column, desi_dr1.NO_SUCH_FIELD" in caplog.text
    rows = message["blocks"][1]["rows"]
    assert [cell["text"] for cell in rows[0]] == ["objectId"]


def _lines(cell):
    """The lines of a rich_text cell, each as (text, bold).

    Line breaks inside a table cell come from newlines in the text elements
    of a single section -- Slack runs separate sections together -- so a
    line's boldness is that of the element it came from.
    """
    assert cell["type"] == "rich_text"
    (section,) = cell["elements"]
    lines = []
    for element in section["elements"]:
        bold = bool(element.get("style", {}).get("bold"))
        lines += [(line, bold) for line in element["text"].split("\n") if line]
    return lines


def test_a_nested_field_lists_each_rows_values_one_per_line(matches):
    # lspsc has three sources per alert; desi_dr1 one match per alert.
    message = format_message(
        matches, LOCALIZE_DISPLAY, max_rows=2, display_columns=["lspsc.ra", "desi_dr1.Z"]
    )

    table = message["blocks"][1]
    (header, first, second) = table["rows"]
    assert [cell["text"] for cell in header] == ["lspsc.ra", "desi_dr1.Z"]
    frame = matches.frame
    assert _lines(first[0]) == [(f"{ra:.4f}", False) for ra in frame["lspsc"].iloc[0]["ra"]]
    assert _lines(first[1]) == [(f"{frame['desi_dr1'].iloc[0]['Z'].iloc[0]:.4f}", False)]
    assert _lines(second[1]) == [(f"{frame['desi_dr1'].iloc[1]['Z'].iloc[0]:.4f}", False)]
    # Wrapped, so every line shows; right-aligned, like the numbers they are.
    assert table["column_settings"] == [{"align": "right", "is_wrapped": True}] * 2


def test_a_whole_nested_column_lists_sub_rows_under_a_header_line(matches):
    message = format_message(matches, LOCALIZE_DISPLAY, max_rows=1, display_columns=["objectId", "lspsc"])

    table = message["blocks"][1]
    (header, row) = table["rows"]
    assert [cell["text"] for cell in header] == ["objectId", "lspsc"]
    assert row[0]["type"] == "raw_text"
    sub = matches.frame["lspsc"].iloc[0]
    lines = _lines(row[1])
    assert lines[0] == ("_id, ra, dec, mag_white, score, distance_arcsec", True)
    assert len(lines) == 1 + len(sub)
    first_source = next(sub.itertuples(index=False))
    assert lines[1] == (
        ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in first_source),
        False,
    )
    assert lines[1][0].startswith(f"{sub['_id'].iloc[0]}, {sub['ra'].iloc[0]:.4f}")
    assert table["column_settings"] == [{"align": "left"}, {"align": "left", "is_wrapped": True}]


def test_a_nested_cell_cuts_off_at_max_nested_rows(matches):
    # lspsc has three sources per alert; two fit, so the third becomes a count.
    message = format_message(
        matches, LOCALIZE_DISPLAY, max_rows=1, display_columns=["lspsc.ra", "lspsc"], max_nested_rows=2
    )

    (_, row) = message["blocks"][1]["rows"]
    ras = [f"{ra:.4f}" for ra in matches.frame["lspsc"].iloc[0]["ra"]]
    assert _lines(row[0]) == [(ras[0], False), (ras[1], False), ("... +1 more", False)]
    whole = _lines(row[1])
    assert whole[0][1] is True  # the header line stays
    assert len(whole) == 4
    assert whole[-1] == ("... +1 more", False)
    # Exactly at the limit, nothing is cut and no count is shown.
    exact = format_message(
        matches, LOCALIZE_DISPLAY, max_rows=1, display_columns=["lspsc.ra"], max_nested_rows=3
    )
    assert len(_lines(exact["blocks"][1]["rows"][1][0])) == 3


def test_run_uses_the_configured_nested_cutoff(slack_config, match_inputs, posted):
    section = slack_config.slack.model_copy(update={"columns": ["lspsc.ra"], "max_nested_rows": 1})
    cfg = slack_config.model_copy(update={"slack": section})

    run_slack_publish(cfg, inputs=match_inputs, stamp=STAMP)

    (call,) = posted
    first_cell = call["blocks"][1]["rows"][1][0]
    assert _lines(first_cell)[-1] == ("... +2 more", False)


def test_a_row_with_no_sub_rows_gets_an_empty_cell(matches):
    frame = matches.frame.copy()
    # Empty every alert's DESI matches while leaving the column nested.
    frame = frame.query("desi_dr1.Z < 0")
    assert len(frame) == len(matches.frame)
    emptied = StageResult(stage=matches.stage, frame=frame, stamp=matches.stamp, summary=matches.summary)

    message = format_message(
        emptied, LOCALIZE_DISPLAY, max_rows=1, display_columns=["desi_dr1.Z", "desi_dr1"]
    )

    (_, row) = message["blocks"][1]["rows"]
    assert row == [{"type": "raw_text", "text": ""}] * 2


def test_run_uses_the_configured_columns(slack_config, match_inputs, posted):
    narrowed = slack_config.slack.model_copy(update={"columns": ["objectId"]})
    cfg = slack_config.model_copy(update={"slack": narrowed})

    run_slack_publish(cfg, inputs=match_inputs, stamp=STAMP)

    (call,) = posted
    header_row = call["blocks"][1]["rows"][0]
    assert [cell["text"] for cell in header_row] == ["objectId"]


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


def test_a_skipped_filter_is_passed_over_but_not_counted_as_quiet(slack_config, posted):
    """Verify a switched-off filter reads as "did not run", never as "found nothing".

    The pipeline records a result for a skipped filter, marked as such; this
    stage must keep it out of the per-filter counts, because "the GW search was
    off" and "the GW search found nothing" must not read the same.
    """
    skipped = StageResult(stage=LOCALIZE_STAGE, stamp=STAMP, summary={"skipped": "disabled"})

    result = run_slack_publish(slack_config, inputs={LOCALIZE_STAGE: skipped}, stamp=STAMP)

    assert posted == []
    # Absent entirely, rather than present with a zero: it never ran.
    assert result.summary["rows_by_filter"] == {}


def test_a_filter_missing_from_the_inputs_entirely_fails_loudly(slack_config, posted):
    """Verify mis-keyed inputs cannot silently drop real candidates.

    Every run records a result for every filter, so an absent entry is a typo
    in a programmatic call, and "exit 0, post nothing" would be the worst
    possible reading of it.
    """
    with pytest.raises(KeyError, match=LOCALIZE_STAGE):
        run_slack_publish(slack_config, inputs={}, stamp=STAMP)

    assert posted == []


def test_a_transport_error_is_isolated_like_a_rejection(slack_config, match_inputs, monkeypatch):
    """Verify a network-level failure, not just a Slack rejection, spares the siblings.

    post_message wraps Slack's own rejections in RuntimeError, but an SSL or
    connection error raises something else entirely, and one filter's network
    mishap is no more a reason to withhold the others than a rejection is.
    """

    def unplugged(token, channel, text, blocks=None):
        raise ConnectionResetError("wire fell out")

    monkeypatch.setattr(slack_publish, "post_message", unplugged)

    with pytest.raises(RuntimeError, match=r"Posted 0 of 1.*wire fell out"):
        run_slack_publish(slack_config, inputs=match_inputs, stamp=STAMP)


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


@pytest.mark.parametrize("field", ["max_rows", "max_nested_rows"])
def test_row_limits_must_be_positive(slack_credentials, field):
    with pytest.raises(ValueError, match=field):
        PipelineConfig.model_validate(
            {
                "run": {"output_dir": "out"},
                "query": {"boom": {"survey": "LSST"}, "window": {"lookback": "1h"}},
                "slack": {"credentials": str(slack_credentials), "channel": "#c", field: 0},
            }
        )


def test_the_default_columns_are_the_identifying_ones(slack_credentials):
    section = SlackConfig(credentials=slack_credentials, channel="#c")
    assert section.columns == DEFAULT_COLUMNS
    assert section.max_nested_rows == 3


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
    assert set(results) == set(pipeline.stage_order(slack_config))
    assert results[LOCALIZE_STAGE].is_empty
    assert results[STAGE].summary["skipped"] == "no input"
    assert posted == []
