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

from desi_aap.boom import (
    ALERT_BAND_COLUMN,
    ALERT_DEC_COLUMN,
    ALERT_ID_COLUMN,
    ALERT_MAG_COLUMN,
    ALERT_RA_COLUMN,
)
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

# Flat columns every filter's table shows, in order, skipping any the frame
# lacks. What identifies an alert, where to point a telescope, and how bright it
# was in which band -- the last two together, since a magnitude without its band
# is not a brightness anyone can act on. Whatever else makes a particular
# filter's result worth reading comes from its own SlackDisplay.columns,
# appended after these.
DISPLAY_COLUMNS = [
    ALERT_ID_COLUMN,
    ALERT_RA_COLUMN,
    ALERT_DEC_COLUMN,
    ALERT_MAG_COLUMN,
    ALERT_BAND_COLUMN,
]


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


def _match_columns(frame: npd.NestedFrame) -> list[str]:
    """The nested columns a crossmatch left, one per catalog.

    Recognized by the ``_dist_arcsec`` field ``crossmatch_nested`` adds, which
    keeps the alerts' own nested columns (BOOM's ``lspsc``) out of the table.
    """
    return [
        column
        for column, dtype in frame.dtypes.items()
        if isinstance(dtype, npd.NestedDtype) and "_dist_arcsec" in frame[column].nest.columns
    ]


def format_message(result: StageResult, display: SlackDisplay, max_rows: int) -> dict[str, Any]:
    """Render one filter's non-empty frame as one Slack message.

    Parameters
    ----------
    result : StageResult
        The result to publish. Its ``frame`` must have at least one row; the
        message also names its ``stamp`` and, when set, its ``output_path``.
    display : desi_aap.stages.base.SlackDisplay
        How this filter's candidates are named, and which of its own columns to
        show, as its module declares.
    max_rows : int
        How many rows the table lists before cutting off.

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

    # Each column is (name, its cells as strings, whether it right-aligns).
    # Measures right-align like numbers; the integer identifiers stay left,
    # like labels.
    shown = frame.head(max_rows)
    columns = [
        (name, [_format_cell(value) for value in shown[name]], shown[name].dtype.kind == "f")
        for name in (*DISPLAY_COLUMNS, *display.columns)
        if name in frame.columns
    ]
    # Each catalog's column shows how many of its sources matched the alert.
    columns += [
        (name, [str(count) for count in shown[name].array.list_lengths], True)
        for name in _match_columns(frame)
    ]

    # Every cell is raw_text, holding our own formatting: as of 2026-08 Slack's
    # validator rejects the raw_number cells its docs describe (it wants an
    # undocumented `value` field), and a preformatted string renders the same.
    header_row = [{"type": "raw_text", "text": name} for name, _, _ in columns]
    value_rows = [
        [{"type": "raw_text", "text": cells[i]} for _, cells, _ in columns] for i in range(len(shown))
    ]
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*\n{found}{cutoff}"}},
        {
            "type": "table",
            "column_settings": [{"align": "right" if numeric else "left"} for _, _, numeric in columns],
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
        message = format_message(result, descriptor.slack_display, cfg.slack.max_rows)
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
