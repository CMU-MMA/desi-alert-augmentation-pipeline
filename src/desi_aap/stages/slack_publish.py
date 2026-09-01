"""Post each run's matched alerts to a Slack channel.

The stage takes the frame the previous stage produced, renders it as a short
message -- a header naming the run, how many candidates it found, the first
``[slack].max_rows`` of them as a native Block Kit table, and a pointer to
the full parquet output -- and posts it with the Slack Web API's
``chat.postMessage``.

Posting needs a *bot token*: register an app on https://api.slack.com/apps
with the ``chat:write`` scope, install it to the workspace, and put the
resulting ``xoxb-`` token in a TOML file (``bot_token = "xoxb-..."``) outside
the repository. The ``[slack]`` section names that file, the channel, and the
row cutoff; the section is optional, and the stage skips itself when it is
absent. The bot must be invited to the channel once (``/invite @<bot>``).

The pipeline stops before this stage when an earlier one produces no rows, so
a run with nothing to report posts nothing by design.
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
from desi_aap.stages.base import StageInputs, StageResult, input_result
from desi_aap.stages.crossmatch import STAGE as CROSSMATCH_STAGE
from desi_aap.utils import run_stamp

logger = logging.getLogger(__name__)

STAGE = "slack_publish"

# The stage whose frame gets published. As stages are added between crossmatch
# and this one, point this at the new last data stage.
INPUT_STAGE = CROSSMATCH_STAGE


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


def format_message(result: StageResult, max_rows: int, display_columns: list[str]) -> dict[str, Any]:
    """Render a stage's non-empty frame as one Slack message.

    Parameters
    ----------
    result : StageResult
        The result to publish. Its ``frame`` must have at least one row; the
        message also names its ``stamp`` and, when set, its ``output_path``.
    max_rows : int
        How many rows the table lists before cutting off.
    display_columns : list of str
        Columns the table shows, in this order, normally ``[slack].columns``
        from the config. A name the frame lacks is skipped, with a warning. A
        match-count column per crossmatched catalog is always appended.

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
    found = f"{n_rows} candidate{'' if n_rows == 1 else 's'} found"
    cutoff = f". Showing the first {max_rows}:" if n_rows > max_rows else ":"

    missing = [name for name in display_columns if name not in frame.columns]
    if missing:
        logger.warning("Configured [slack] column(s) not in the frame, skipping: %s", ", ".join(missing))

    # Each column is (name, its cells as strings, whether it right-aligns).
    # Measures right-align like numbers; the integer identifiers stay left,
    # like labels.
    shown = frame.head(max_rows)
    columns = [
        (name, [_format_cell(value) for value in shown[name]], shown[name].dtype.kind == "f")
        for name in display_columns
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
) -> StageResult:
    """Run the stage: render the previous stage's frame and post it to Slack.

    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration. Without a ``[slack]`` section the stage
        logs that it is skipping and posts nothing.
    dry_run : bool
        Build the message and log it instead of posting it. Also the way to
        preview the formatting without a workspace.
    inputs : dict of str to StageResult, optional
        Results of the stages that already ran. The frame to publish comes
        from :data:`INPUT_STAGE`.
    stamp : str, optional
        This run's timestamp. Defaults to now.

    Returns
    -------
    StageResult
        The input frame, passed through unchanged so this stage never reads
        as the one that ended the run; ``summary["posted"]`` says whether a
        message went out.

    Raises
    ------
    KeyError
        If :data:`INPUT_STAGE` has not run.
    ValueError
        If the credentials file is missing or malformed.
    RuntimeError
        If Slack rejects the message.
    """
    stamp = stamp or run_stamp()
    upstream = input_result(inputs, INPUT_STAGE)
    frame = upstream.frame
    summary = {"posted": False, "n_rows": 0 if frame is None else len(frame)}
    result = StageResult(stage=STAGE, frame=frame, stamp=stamp, summary=summary)

    if cfg.slack is None:
        logger.info("No [slack] section configured; skipping.")
        return result
    if frame is None or frame.empty:
        logger.info("No rows to publish; posting nothing.")
        return result

    message = format_message(upstream, cfg.slack.max_rows, cfg.slack.columns)
    if dry_run:
        logger.info(
            "Dry run: not posting to Slack. %s Blocks payload:\n%s",
            message["text"],
            json.dumps(message["blocks"], indent=2),
        )
        return result

    token = load_bot_token(cfg.slack.credentials)
    post_message(token, cfg.slack.channel, message["text"], message["blocks"])
    summary["posted"] = True
    logger.info(
        "Posted %d of %d row(s) to %s.", min(len(frame), cfg.slack.max_rows), len(frame), cfg.slack.channel
    )
    return result
