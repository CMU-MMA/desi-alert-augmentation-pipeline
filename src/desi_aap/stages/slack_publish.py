"""Post each filter's candidates to a Slack channel, one message per filter.

Each filter in :func:`desi_aap.stages.filters.filter_descriptors` that found
anything gets its own message -- a header naming the run and what the filter
found, the first ``[slack].max_rows`` candidates as a native Block Kit table,
and a pointer to the full parquet output -- posted with the Slack Web API's
``chat.postMessage``.

One message per filter rather than one per run because the filters answer
different questions and are read by different people: a GW coincidence wants
looking at tonight, while a superluminous supernova candidate can wait for the
morning. Each filter says how its own candidates are announced, via the
:class:`~desi_aap.stages.base.SlackDisplay` its module declares, so this module
never learns what any particular filter means.

A filter that found nothing is passed over in silence rather than posting an
empty message; a run where every filter found nothing posts nothing at all.
That is the normal outcome for most hours, not a sign that anything is wrong.

Posting needs a *bot token*: register an app on https://api.slack.com/apps
with the ``chat:write`` scope, install it to the workspace, and put the
resulting ``xoxb-`` token in a TOML file (``bot_token = "xoxb-..."``) outside
the repository. The ``[slack]`` section names that file, the channel, and the
row cutoff; the section is optional, and the stage skips itself when it is
absent. The bot must be invited to the channel once (``/invite @<bot>``).
"""

import json
import logging
import tomllib
from pathlib import Path
from typing import Any

import nested_pandas as npd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from desi_aap.config import PipelineConfig
from desi_aap.stages.base import SlackDisplay, StageInputs, StageResult
from desi_aap.stages.filters import FilterDescriptor, filter_descriptors
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)

STAGE = "slack_publish"

# This stage requires every filter, and the filters come from the config --
# each JSON file in the filters directory is one -- so its dependencies cannot
# be a module constant. desi_aap.pipeline.stages_for builds them per run, from
# the same filter_descriptors this module announces. It is also the one stage
# that tolerates a required stage not having run: see run_slack_publish.
#
# Which columns every filter's table shows is likewise not a constant here but
# `[slack].columns` in the config (see desi_aap.config.SlackConfig for the
# default and why). Whatever else makes a particular filter's result worth
# reading comes from its own SlackDisplay.columns, appended after those.


def load_bot_token(path: Path) -> str:
    """Read the Slack bot token from a TOML credentials file.

    Parameters
    ----------
    path : Path
        A TOML file holding ``bot_token = "xoxb-..."``. ``~`` is expanded, so
        the config can point into a home directory on any machine.

    Returns
    -------
    str
        The token.

    Raises
    ------
    ValueError
        If the file is missing, is not valid TOML, or has no ``bot_token``.
    """
    path = path.expanduser()
    try:
        with path.open("rb") as handle:
            credentials = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Slack credentials file not found: {path}. Create it with one line, "
            'bot_token = "xoxb-...", or point [slack].credentials elsewhere.'
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Slack credentials file {path} is not valid TOML: {exc}") from exc

    token = credentials.get("bot_token")
    if not token or not isinstance(token, str):
        raise ValueError(f'Slack credentials file {path} must set bot_token = "xoxb-...".')
    return token


def _format_cell(value: object) -> str:
    """Render one table cell, keeping coordinates readable but compact."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _raw_cell(text: str) -> dict[str, Any]:
    """A plain table cell."""
    return {"type": "raw_text", "text": text}


def _lines_cell(lines: list[str], header: str | None = None) -> dict[str, Any]:
    """A table cell showing each line on its own, under an optional bold header.

    A ``rich_text`` cell honors newlines inside its text elements, where a
    ``raw_text`` cell's handling of them is undocumented; separate
    ``rich_text_section`` elements, which stack as paragraphs elsewhere, run
    together on one line inside a table cell. Empty, it falls back to a
    blank plain cell, since a rich_text cell needs at least one element.
    """
    elements: list[dict[str, Any]] = []
    if header is not None:
        elements.append({"type": "text", "text": header, "style": {"bold": True}})
    if lines:
        body = "\n".join(lines)
        elements.append({"type": "text", "text": f"\n{body}" if header is not None else body})
    if not elements:
        return _raw_cell("")
    return {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": elements}]}


def _cut_off(lines: list[str], limit: int) -> list[str]:
    """The first ``limit`` lines, plus one saying how many were left out."""
    if len(lines) <= limit:
        return lines
    return [*lines[:limit], f"... +{len(lines) - limit} more"]


def _column_cells(
    shown: npd.NestedFrame, name: str, max_nested_rows: int
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """The cells of one configured column, one per row of ``shown``.

    A name resolves, in order, as a column of the frame -- flat, or nested,
    in which case each cell lists the row's sub-rows one per line under a
    bold header line naming the fields -- or as a ``nested.field`` path, in
    which case each cell lists that field's values for the row, one per
    line. Checking the frame's own columns first keeps BOOM's flat alert
    fields, whose names contain dots like ``candidate.ra``, working as they
    are. A row with no sub-rows gets an empty cell; one with more than
    ``max_nested_rows`` lists that many, then a line saying how many more
    there are.

    Returns
    -------
    tuple of (list of dict, dict) or None
        The Block Kit cells and the column's ``column_settings`` entry --
        measures right-align like numbers, the integer identifiers stay
        left like labels, and nested cells wrap so their lines show -- or
        None when the frame has no such column or field.
    """
    if name in shown.columns:
        series = shown[name]
        if isinstance(series.dtype, npd.NestedDtype):
            header = ", ".join(series.nest.columns)
            # Iterating a nested column yields each row's sub-rows as a
            # DataFrame, or None when it has none.
            cells = [
                _raw_cell("")
                if sub is None
                else _lines_cell(
                    _cut_off(
                        [", ".join(_format_cell(v) for v in row) for row in sub.itertuples(index=False)],
                        max_nested_rows,
                    ),
                    header,
                )
                for sub in series
            ]
            return cells, {"align": "left", "is_wrapped": True}
        align = "right" if series.dtype.kind == "f" else "left"
        return [_raw_cell(_format_cell(value)) for value in series], {"align": align}

    nested, _, field = name.partition(".")
    if nested in shown.nested_columns and field in shown[nested].nest.columns:
        align = "right" if shown[nested].nest[field].dtype.kind == "f" else "left"
        cells = [
            _lines_cell(
                []
                if sub is None
                else _cut_off([_format_cell(value) for value in sub[field]], max_nested_rows)
            )
            for sub in shown[nested]
        ]
        return cells, {"align": align, "is_wrapped": True}
    return None


def format_message(
    result: StageResult,
    display: SlackDisplay,
    max_rows: int,
    display_columns: list[str],
    max_nested_rows: int = 3,
) -> dict[str, Any]:
    """Render one filter's non-empty frame as one Slack message.

    Parameters
    ----------
    result : StageResult
        The result to publish. Its ``frame`` must have at least one row; the
        message also names its ``stamp`` and, when set, its ``output_path``.
    display : desi_aap.stages.base.SlackDisplay
        How this filter's candidates are named, and which of its own columns to
        show after ``display_columns``, as its module declares.
    max_rows : int
        How many rows the table lists before cutting off.
    display_columns : list of str
        Columns every filter's table shows first, in this order, normally
        ``[slack].columns`` from the config. Each -- and each of
        ``display.columns`` -- may be a flat column, a nested column, or a
        ``nested.field`` path; the last two render the row's sub-rows as
        lines of one cell, as :func:`_column_cells` describes. A name the
        frame lacks is skipped: with a warning when it came from the config,
        which was written for every filter, and quietly when it came from the
        filter, which may name a column only some runs produce.
    max_nested_rows : int
        How many sub-rows a nested cell lists before cutting off, normally
        ``[slack].max_nested_rows`` from the config.

    Returns
    -------
    dict
        Keyword arguments for ``chat.postMessage``: a plain ``text`` fallback
        for notifications, and ``blocks`` holding a header section naming the
        run and how many candidates it found, a native table block, and,
        when the results were written, where.
    """
    frame = result.frame
    n_rows = len(frame)

    title = f"DESI Alert Augmentation Pipeline run {result.stamp}"
    plural = "" if n_rows == 1 else "s"
    found = f"{n_rows} {display.title}{plural} found"
    cutoff = f". Showing the first {max_rows}:" if n_rows > max_rows else ":"

    # Each column is (name, its Block Kit cells, its column_settings entry).
    shown = frame.head(max_rows)
    columns = []
    missing_configured = []
    missing_filter = []
    for name in (*display_columns, *display.columns):
        resolved = _column_cells(shown, name, max_nested_rows)
        if resolved is None:
            (missing_configured if name in display_columns else missing_filter).append(name)
        else:
            columns.append((name, *resolved))
    if missing_configured:
        logger.warning(
            "Configured [slack] column(s) not in the frame, skipping: %s", ", ".join(missing_configured)
        )
    if missing_filter:
        logger.info(
            "Column(s) %s names that are not in its frame, skipping: %s",
            display.title,
            ", ".join(missing_filter),
        )

    # Flat cells are raw_text holding our own formatting: as of 2026-08
    # Slack's validator rejects the raw_number cells its docs describe (it
    # wants an undocumented `value` field), and a preformatted string renders
    # the same. Nested cells are rich_text, for their line breaks.
    header_row = [_raw_cell(name) for name, _, _ in columns]
    value_rows = [[cells[i] for _, cells, _ in columns] for i in range(len(shown))]
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*\n{found}{cutoff}"}},
        {
            "type": "table",
            "column_settings": [settings for _, _, settings in columns],
            "rows": [header_row, *value_rows],
        },
    ]
    if result.output_path is not None:
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"Full results: `{result.output_path}`"}],
            }
        )
    return {"text": f"{title}: {found}.", "blocks": blocks}


def post_message(token: str, channel: str, text: str, blocks: list[dict[str, Any]] | None = None) -> None:
    """Post one message to a channel with the Slack Web API.

    Parameters
    ----------
    token : str
        A bot token, as :func:`load_bot_token` returns.
    channel : str
        The channel to post to, such as ``"#desi-alerts"``.
    text : str
        Plain text. With ``blocks`` it only feeds notifications and clients
        that cannot render them; alone it is the whole message.
    blocks : list of dict, optional
        Block Kit blocks, as :func:`format_message` builds.

    Raises
    ------
    RuntimeError
        If Slack rejects the message, naming its error code and whatever
        detail messages came with it -- and, for the two codes that mean the
        bot cannot see the channel, the fix.
    """
    try:
        WebClient(token=token).chat_postMessage(channel=channel, text=text, blocks=blocks)
    except SlackApiError as exc:
        error = exc.response.get("error", "unknown error")
        hint = ""
        if error in ("not_in_channel", "channel_not_found"):
            hint = f" Make sure the channel exists and the bot has been invited: /invite in {channel}."
        # invalid_blocks and friends come with per-field schema messages;
        # surface a few so the failure can be read without a debugger.
        messages = (exc.response.get("response_metadata") or {}).get("messages") or []
        detail = ""
        if messages:
            more = f" (+{len(messages) - 5} more)" if len(messages) > 5 else ""
            detail = f" Details: {'; '.join(messages[:5])}{more}"
        raise RuntimeError(f"Slack rejected the message: {error}.{hint}{detail}") from exc


def run_slack_publish(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    inputs: StageInputs | None = None,
    stamp: str | None = None,
    descriptors: tuple[FilterDescriptor, ...] | None = None,
) -> StageResult:
    """Run the stage: post one message for each filter that found something.

    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration. Without a ``[slack]`` section the stage
        logs that it is skipping and posts nothing.
    dry_run : bool
        Build each message and log it instead of posting it. Also the way to
        preview the formatting without a workspace.
    inputs : dict of str to StageResult, optional
        Results of the stages that already ran. The frames to publish come from
        the filters in :func:`desi_aap.stages.filters.filter_descriptors`. A filter
        that was switched off or skipped this run is passed over, since the
        pipeline records an empty result for it either way.
    stamp : str, optional
        This run's timestamp. Defaults to now.
    descriptors : tuple of FilterDescriptor, optional
        The filters to announce. The pipeline binds the ones the run was built
        from, so the stage set cannot drift between building the run and
        announcing it -- a filter file edited or deleted mid-run changes the
        *next* run. Left out (a direct call), they are read from the config.

    Returns
    -------
    StageResult
        ``frame`` is ``None``: this stage announces results rather than
        producing any, and nothing runs after it. ``summary`` holds
        ``n_posted``, how many messages went out, and ``rows_by_filter``, the
        candidate count of every filter that has a result -- zero included, so
        that a filter which ran and found nothing is told apart from one that
        never ran.

    Raises
    ------
    KeyError
        If a filter has no entry in ``inputs`` at all. Every run records a
        result for every filter -- a skipped one gets a ``skipped`` summary --
        so an absent entry means the inputs were mis-keyed, and treating that
        as "found nothing" would silently drop real candidates.
    ValueError
        If the credentials file is missing or malformed.
    RuntimeError
        If Slack rejects any message. The filters that posted successfully stay
        posted, and the error names every one that did not.
    """
    stamp = stamp or run_stamp()
    if descriptors is None:
        descriptors = filter_descriptors(cfg)

    # Every filter that ran, and how many candidates it contributed --
    # including the ones that contributed none, so the summary distinguishes a
    # filter that ran and found nothing from one that was switched off or
    # skipped. The skipped ones are absent from the counts and named in the
    # log, because "the GW search was off" and "the GW search found nothing"
    # must not read the same.
    rows_by_filter: dict[str, int] = {}
    published: list[tuple[FilterDescriptor, StageResult]] = []
    for descriptor in descriptors:
        result = (inputs or {}).get(descriptor.stage)
        if result is None:
            raise KeyError(
                f"Filter {descriptor.stage!r} has no result in this run's inputs. Every run "
                "records one, even for a skipped filter, so a missing entry means the inputs "
                "are mis-keyed rather than that the filter found nothing."
            )
        if result.summary.get("skipped"):
            logger.info("Filter %r did not run (%s).", descriptor.stage, result.summary["skipped"])
            continue
        rows_by_filter[descriptor.stage] = 0 if result.frame is None else len(result.frame)
        if result.is_empty:
            logger.info("Filter %r found nothing to publish.", descriptor.stage)
            continue
        published.append((descriptor, result))

    summary: dict[str, Any] = {"n_posted": 0, "rows_by_filter": rows_by_filter}
    outcome = StageResult(stage=STAGE, frame=None, stamp=stamp, summary=summary)

    if cfg.slack is None:
        logger.info("No [slack] section configured; skipping.")
        return outcome
    if not published:
        logger.info("No filter produced candidates; posting nothing.")
        return outcome

    # Each filter is posted on its own, and one Slack rejection does not stop
    # the rest: the filters are independent, and a rate limit hit while
    # announcing the first is no reason to withhold the second. The failures are
    # collected and raised together at the end, so the run still fails loudly
    # rather than reporting a partial post as a success.
    token = None if dry_run else load_bot_token(cfg.slack.credentials)
    failed_filters: list[str] = []
    failures: list[str] = []
    for descriptor, result in published:
        message = format_message(
            result,
            descriptor.slack_display,
            cfg.slack.max_rows,
            cfg.slack.columns,
            cfg.slack.max_nested_rows,
        )
        if dry_run:
            logger.info(
                "Dry run: not posting %r to Slack. %s Blocks payload:\n%s",
                descriptor.stage,
                message["text"],
                json.dumps(message["blocks"], indent=2),
            )
            continue
        try:
            post_message(token, cfg.slack.channel, message["text"], message["blocks"])
        # Not just the RuntimeError post_message wraps Slack's rejections in:
        # a transport-level error (SSL, connection reset, DNS) raises something
        # else entirely, and one filter's network mishap is no more a reason to
        # withhold its siblings' messages than a rejection is.
        except Exception as exc:
            logger.error("Could not post %r to %s: %s", descriptor.stage, cfg.slack.channel, exc)
            failed_filters.append(descriptor.stage)
            failures.append(f"{descriptor.stage}: {exc}")
            continue
        summary["n_posted"] += 1
        logger.info(
            "Posted %d of %d %s row(s) to %s.",
            min(len(result.frame), cfg.slack.max_rows),
            len(result.frame),
            descriptor.stage,
            cfg.slack.channel,
        )

    summary["failed_filters"] = failed_filters
    if failures:
        logger.info("Slack summary: %s", summary)
        raise RuntimeError(
            f"Posted {summary['n_posted']} of {len(published)} filter message(s); "
            f"{len(failures)} failed. {' | '.join(failures)}"
        )
    return outcome
