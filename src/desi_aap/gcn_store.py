"""On-disk store for GCN notices and their localization maps.

Layout, rooted at STORE_ROOT:

    <root>/<category>/<source>/<event_id>/
        notices/<stem>.json                     the notice, verbatim
        skymaps/<stem>.multiorder.fits          real map the mission supplied
        skymaps/<stem>.synthetic.multiorder.fits  map synthesized from a quoted error region
        history.jsonl                           one line per notice received for this event
        latest.json                             pointer to the current best notice
        latest.fits                             symlink to the current best map, if any
    <root>/index.jsonl                          one line per notice, across all events
    <root>/_quarantine/                         payloads that failed to parse or store

GW maps live under ``gw/`` and the GRB and neutrino localizations under ``grb/`` and
``neutrino/``, so the two never mix.

Every version of an event is kept, because a notice sequence is the record of what was known
when: the early_warning map that a follow-up was triggered on is not recoverable from the
final update that replaced it. ``latest.json`` is recomputed from history.jsonl on each
write, so it stays correct even when notices arrive out of order.
"""

import contextlib
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

# safe_file_part lives in gracedb_cache rather than gracedb_tools, which only re-exports it.
# Taking it from its own module keeps this store off the GraceDB REST and ligo.skymap import
# chain, which it otherwise has no use for.
from desi_aap.gracedb_cache import safe_file_part

# Root of the store, relative to the working directory unless overridden.
STORE_ROOT = Path("gcn_localizations")

# Names inside an event directory.
NOTICE_SUBDIR = "notices"
SKYMAP_SUBDIR = "skymaps"
HISTORY_NAME = "history.jsonl"
LATEST_POINTER_NAME = "latest.json"
LATEST_SKYMAP_LINK_NAME = "latest.fits"

# Names inside the store root.
INDEX_NAME = "index.jsonl"
QUARANTINE_SUBDIR = "_quarantine"

# Length of the payload digest appended to every file stem. Its job is idempotency: a
# redelivered message hashes to the same stem and is skipped, while a genuinely different
# notice that happens to share an alert type and record number lands on its own file instead
# of silently overwriting the first.
DIGEST_CHARS = 8

# Timestamps are compacted into file stems as 20260806T101122Z.
STEM_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

# Sort key components for a missing record number and a missing timestamp. Both sort below
# any real value, so a notice that omits them never displaces one that does not.
MISSING_RECORD_NUMBER = -1
MISSING_TIMESTAMP = ""

# Temporary suffix for atomic writes. A reader polling the store must never see a
# half-written notice or pointer, so every file is written aside and renamed into place.
TEMP_SUFFIX = ".tmp"

# Stands in for a directory name that sanitizes to nothing usable. safe_file_part() strips
# path separators but keeps dots, so an event id of ".." would survive it and then resolve to
# the parent directory. Identifiers come off the network, so that has to be closed.
UNSAFE_PATH_PART = "_"


def utc_now_iso():
    """Return the current UTC time as an ISO-8601 string with a trailing Z.

    Returns
    -------
    str
        Timestamp such as "2026-08-06T10:11:22.123456Z".
    """
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def payload_digest(raw):
    """Hash a raw notice body, for idempotent file naming.

    Parameters
    ----------
    raw : bytes or str
        Message body exactly as received.

    Returns
    -------
    str
        The first DIGEST_CHARS characters of the SHA-256 hex digest.
    """
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    return hashlib.sha256(data).hexdigest()[:DIGEST_CHARS]


def compact_timestamp(value):
    """Compact an ISO-8601 timestamp into a file-name-safe stamp.

    Parameters
    ----------
    value : str or None
        ISO-8601 timestamp from a notice, with or without a trailing Z.

    Returns
    -------
    str or None
        Stamp such as "20260806T101122Z", or None if the value is missing or unparseable.
    """
    if not value:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        # Keep something traceable rather than dropping an unrecognized format entirely.
        return safe_file_part(text)[:20] or None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime(STEM_TIMESTAMP_FORMAT)


def safe_path_part(value):
    """Sanitize a value for use as a single directory name.

    safe_file_part() already removes path separators, which is what stops an identifier from
    naming a directory elsewhere. This additionally rejects names made only of dots, since
    "." and ".." are valid file-name characters but resolve to directories of their own.

    Parameters
    ----------
    value : object
        Value to sanitize; converted with str() first.

    Returns
    -------
    str
        A name safe to use as one path component.
    """
    part = safe_file_part(value)
    if part.strip(".") == "":
        return UNSAFE_PATH_PART
    return part


def notice_stem(record, digest):
    """Build the file stem shared by a notice and its maps.

    The stem reads as the event's history: identifier, sequence number where the mission
    provides one, alert type, notice time, and the payload digest.

    Parameters
    ----------
    record : desi_aap.gcn_notices.NoticeRecord
        Normalized notice.
    digest : str
        Payload digest from payload_digest().

    Returns
    -------
    str
        Sanitized file stem, e.g. "S250101a__preliminary__20260806T101122Z__a1b2c3d4".
    """
    parts = [safe_file_part(record.event_id)]
    if record.record_number is not None:
        parts.append(f"r{record.record_number:03d}")
    if record.alert_type:
        parts.append(safe_file_part(record.alert_type))
    stamp = compact_timestamp(record.notice_time) or compact_timestamp(record.trigger_time)
    if stamp:
        parts.append(stamp)
    parts.append(digest)
    return "__".join(parts)


def skymap_stem(stem, label):
    """Extend a notice stem with a localization label.

    Parameters
    ----------
    stem : str
        Notice stem from notice_stem().
    label : str or None
        Localization label, or None for the notice's single or primary region.

    Returns
    -------
    str
        Stem for that localization's map file.
    """
    if label is None:
        return stem
    return f"{stem}__{safe_file_part(label)}"


def event_dir(record, root=STORE_ROOT):
    """Return the directory a notice's event is stored under.

    Parameters
    ----------
    record : desi_aap.gcn_notices.NoticeRecord
        Normalized notice.
    root : pathlib.Path, optional
        Store root.

    Returns
    -------
    pathlib.Path
        ``<root>/<category>/<source>/<event_id>``, with each part sanitized.
    """
    return (
        Path(root)
        / safe_path_part(record.category)
        / safe_path_part(record.source)
        / safe_path_part(record.event_id)
    )


def write_atomic_bytes(path, payload):
    """Write bytes to a path atomically, so readers never see a partial file.

    Parameters
    ----------
    path : pathlib.Path
        Destination.
    payload : bytes
        Contents to write.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + TEMP_SUFFIX)
    temp.write_bytes(payload)
    os.replace(temp, path)
    return path


def write_atomic_json(path, obj):
    """Write an object as pretty-printed JSON, atomically.

    Parameters
    ----------
    path : pathlib.Path
        Destination.
    obj : object
        JSON-serializable value.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    return write_atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def append_jsonl(path, entry):
    """Append one JSON object as a line to a JSON Lines file.

    Parameters
    ----------
    path : pathlib.Path
        JSON Lines file, created if absent.
    entry : dict
        Object to append.

    Returns
    -------
    pathlib.Path
        The path appended to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def read_jsonl(path):
    """Read a JSON Lines file, skipping blank and malformed lines.

    Parameters
    ----------
    path : pathlib.Path
        JSON Lines file; a missing file reads as empty.

    Returns
    -------
    list of dict
        Parsed entries, in file order.
    """
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def entry_sort_key(entry):
    """Rank a history entry so the greatest is the event's current best notice.

    Ordered by the mission's sequence number first, since that is what a mission increments
    when it supersedes a notice. Alert type breaks ties for IGWN, which publishes no sequence
    number, and puts a retraction last so a withdrawn event never reads as live. Notice time
    and then arrival order break remaining ties.

    Parameters
    ----------
    entry : dict
        History entry written by store_notice().

    Returns
    -------
    tuple
        Sort key; compare with max().
    """
    record_number = entry.get("record_number")
    return (
        MISSING_RECORD_NUMBER if record_number is None else int(record_number),
        entry.get("alert_type_rank", MISSING_RECORD_NUMBER),
        entry.get("notice_time") or entry.get("trigger_time") or MISSING_TIMESTAMP,
        entry.get("received_time") or MISSING_TIMESTAMP,
        entry.get("sequence", 0),
    )


def latest_entry(entries):
    """Pick the current best notice from a list of history entries.

    Parameters
    ----------
    entries : list of dict
        History entries for one event.

    Returns
    -------
    dict or None
        The highest-ranked entry, or None if the list is empty.
    """
    if not entries:
        return None
    return max(entries, key=entry_sort_key)


def latest_skymap_entry(entries):
    """Pick the newest notice that actually carried a map, honouring retractions.

    The newest notice for an event is not always the one with a map. Swift GUANO's first
    record carries none and its later ones do, and an update can supersede a localized notice
    without repeating the map. A retraction is the exception: once an event is withdrawn,
    pointing at its last map would present a withdrawn localization as current.

    Parameters
    ----------
    entries : list of dict
        History entries for one event.

    Returns
    -------
    dict or None
        The highest-ranked entry holding at least one map, or None if the event is retracted
        or no notice carried one.
    """
    newest = latest_entry(entries)
    if newest is None or newest.get("is_retraction"):
        return None
    return latest_entry([entry for entry in entries if entry.get("skymaps")])


def update_latest_pointer(directory):
    """Recompute an event's latest.json pointer, and its latest.fits symlink, from history.

    Recomputing from the full history rather than trusting arrival order means an update that
    reaches us before the preliminary it supersedes still ends up as latest, and a replayed
    old notice does not demote a newer one.

    Parameters
    ----------
    directory : pathlib.Path
        Event directory.

    Returns
    -------
    dict or None
        The pointer written, or None if the event has no history. It is the latest history
        entry plus "latest_skymap", the best map currently available for the event, which may
        come from an earlier notice than the latest one.
    """
    directory = Path(directory)
    entries = read_jsonl(directory / HISTORY_NAME)
    entry = latest_entry(entries)
    if entry is None:
        return None

    map_entry = latest_skymap_entry(entries)
    skymaps = (map_entry or {}).get("skymaps") or []
    target = skymaps[0]["path"] if skymaps else None
    pointer = {
        **entry,
        "latest_skymap": target,
        "latest_skymap_stem": (map_entry or {}).get("stem"),
        "latest_skymap_source": skymaps[0]["source"] if skymaps else None,
    }
    write_atomic_json(directory / LATEST_POINTER_NAME, pointer)

    link = directory / LATEST_SKYMAP_LINK_NAME
    if link.is_symlink() or link.exists():
        link.unlink()
    if target:
        # A filesystem without symlinks still gets latest.json, which is authoritative.
        with contextlib.suppress(OSError):
            link.symlink_to(Path(target))
    return pointer


def stored_digests(directory):
    """Return the payload digests already stored for an event.

    Parameters
    ----------
    directory : pathlib.Path
        Event directory.

    Returns
    -------
    set of str
        Digests from the event's history.
    """
    return {entry["digest"] for entry in read_jsonl(Path(directory) / HISTORY_NAME) if "digest" in entry}


def store_notice(record, raw, root=STORE_ROOT, resolve=None, received_time=None):
    """Store a notice, its localization maps, and its index entries.

    A notice whose payload digest is already in the event's history is a redelivery and is
    skipped, which is what makes an at-least-once Kafka consumer safe to restart.

    Parameters
    ----------
    record : desi_aap.gcn_notices.NoticeRecord
        Normalized notice.
    raw : bytes or str
        Message body exactly as received, stored verbatim.
    root : pathlib.Path, optional
        Store root.
    resolve : callable, optional
        ``resolve(path_stem, localization, record) -> (path, source)`` used to write each
        map. Defaults to desi_aap.gcn_skymaps.resolve_skymap. Injectable so the store can be
        tested without building real FITS files.
    received_time : str, optional
        ISO-8601 arrival time; defaults to now. Injectable to keep tests deterministic.

    Returns
    -------
    dict
        The history entry for this notice, with an added "stored" key that is False when the
        notice was a redelivery and nothing was written.
    """
    if resolve is None:
        from desi_aap.gcn_skymaps import resolve_skymap

        resolve = resolve_skymap

    root = Path(root)
    directory = event_dir(record, root=root)
    digest = payload_digest(raw)
    history_path = directory / HISTORY_NAME
    history = read_jsonl(history_path)

    for existing in history:
        if existing.get("digest") == digest:
            return {**existing, "stored": False}

    stem = notice_stem(record, digest)
    notice_path = directory / NOTICE_SUBDIR / f"{stem}.json"
    write_atomic_bytes(notice_path, raw.encode("utf-8") if isinstance(raw, str) else raw)

    skymaps = []
    for localization in record.localizations:
        path_stem = directory / SKYMAP_SUBDIR / skymap_stem(stem, localization.label)
        path, source = resolve(path_stem, localization, record)
        if path is None:
            continue
        skymaps.append(
            {
                "path": str(Path(path).relative_to(directory)),
                "source": source,
                **localization.summary(),
            }
        )

    entry = {
        **record.summary(),
        "digest": digest,
        "stem": stem,
        "alert_type_rank": record.alert_type_rank,
        "received_time": received_time or utc_now_iso(),
        "sequence": len(history),
        "notice_path": str(notice_path.relative_to(directory)),
        "event_dir": str(directory.relative_to(root)),
        "skymaps": skymaps,
        "localizations_without_map": len(record.localizations) - len(skymaps),
    }
    append_jsonl(history_path, entry)
    append_jsonl(root / INDEX_NAME, entry)
    update_latest_pointer(directory)
    return {**entry, "stored": True}


def quarantine_payload(topic, raw, error, root=STORE_ROOT, received_time=None):
    """Set aside a payload that could not be parsed or stored, and why.

    The listener commits the Kafka offset for such a message so one bad payload cannot stall
    the stream, which only stays safe if the payload itself is kept for inspection.

    Parameters
    ----------
    topic : str
        Topic the message arrived on.
    raw : bytes or str
        Message body exactly as received.
    error : BaseException or str
        The failure.
    root : pathlib.Path, optional
        Store root.
    received_time : str, optional
        ISO-8601 arrival time; defaults to now.

    Returns
    -------
    pathlib.Path
        Path of the quarantined payload.
    """
    root = Path(root)
    stamp = received_time or utc_now_iso()
    digest = payload_digest(raw)
    stem = f"{safe_file_part(topic)}__{safe_file_part(stamp)}__{digest}"
    directory = root / QUARANTINE_SUBDIR
    path = directory / f"{stem}.payload"
    write_atomic_bytes(path, raw.encode("utf-8") if isinstance(raw, str) else raw)
    write_atomic_json(
        directory / f"{stem}.error.json",
        {
            "topic": topic,
            "received_time": stamp,
            "digest": digest,
            "error_type": type(error).__name__ if isinstance(error, BaseException) else "str",
            "error": str(error),
            "payload_path": path.name,
        },
    )
    return path


def iter_index(root=STORE_ROOT):
    """Read the store-wide index.

    Parameters
    ----------
    root : pathlib.Path, optional
        Store root.

    Returns
    -------
    list of dict
        Index entries, oldest first.
    """
    return read_jsonl(Path(root) / INDEX_NAME)


def latest_skymap_path(category, source, event_id, root=STORE_ROOT):
    """Find the current best map for one event.

    Parameters
    ----------
    category, source, event_id : str
        Store route, as on the NoticeRecord.
    root : pathlib.Path, optional
        Store root.

    Returns
    -------
    pathlib.Path or None
        Path to the map, or None if the event is unknown, retracted, or never carried a map.
    """
    directory = Path(root) / safe_path_part(category) / safe_path_part(source) / safe_path_part(event_id)
    pointer = directory / LATEST_POINTER_NAME
    if not pointer.exists():
        return None
    entry = json.loads(pointer.read_text(encoding="utf-8"))
    target = entry.get("latest_skymap")
    return directory / target if target else None


def find_events(root=STORE_ROOT, category=None, source=None, event_id=None, related_id=None):
    """Search the index for events matching a route or a related identifier.

    ``related_id`` is the useful one for follow-up: it finds the Swift GUANO records and BOOM
    counterparts that name a given GW superevent, which is how a non-GW notice is tied back
    to the GW event it belongs to.

    Parameters
    ----------
    root : pathlib.Path, optional
        Store root.
    category, source, event_id : str, optional
        Exact-match filters on the store route.
    related_id : str, optional
        Match notices whose related_ids contain this identifier, matched case-insensitively
        and as a substring so "S230914ak" also finds "S230914ak-2-Preliminary".

    Returns
    -------
    list of dict
        Matching index entries, oldest first.
    """
    needle = related_id.lower() if related_id else None
    matches = []
    for entry in iter_index(root=root):
        if category is not None and entry.get("category") != category:
            continue
        if source is not None and entry.get("source") != source:
            continue
        if event_id is not None and entry.get("event_id") != event_id:
            continue
        if needle is not None and not any(
            needle in str(value).lower() for value in entry.get("related_ids") or []
        ):
            continue
        matches.append(entry)
    return matches


def event_directories(root=STORE_ROOT):
    """List every event directory in the store.

    Parameters
    ----------
    root : pathlib.Path, optional
        Store root.

    Returns
    -------
    list of pathlib.Path
        Directories holding a history.jsonl, sorted by path.
    """
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path.parent for path in root.glob(f"*/*/*/{HISTORY_NAME}"))
