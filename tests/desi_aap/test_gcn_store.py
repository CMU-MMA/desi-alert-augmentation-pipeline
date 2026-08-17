"""Tests for the on-disk store of notices and localization maps."""

import json
from pathlib import Path

import pytest
from gcn_examples import (
    BOOM_CROSSMATCH_ID,
    GUANO_TRIGGER_ID,
    ICECUBE_EVENT_NAME,
    IGWN_SUPEREVENT_ID,
    boom_alert,
    icecube_gold_bronze,
    igwn_gwalert,
    swift_bat_guano,
)

from desi_aap import gcn_notices, gcn_store

# A fixed arrival time keeps stems and history entries deterministic across runs.
RECEIVED_TIME = "2026-08-06T12:00:00Z"


def fake_resolve(suffix=".synthetic.multiorder.fits"):
    """Build a stand-in for resolve_skymap that writes a placeholder instead of real FITS.

    The store's job is layout, ordering and idempotency, none of which depend on a map's
    contents, so the tests that check those do not pay to build real HEALPix files.

    Parameters
    ----------
    suffix : str, optional
        Suffix to give the written file.

    Returns
    -------
    callable
        A resolve(path_stem, localization, record) -> (path, source) function.
    """

    def resolve(path_stem, localization, record):
        if not (localization.has_real_map or localization.has_error_region):
            return None, None
        path = path_stem.with_name(path_stem.name + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PLACEHOLDER")
        return path, "test"

    return resolve


def store(payload, topic, root, received_time=RECEIVED_TIME, resolve=None):
    """Parse and store one payload, returning the store's history entry.

    Parameters
    ----------
    payload : dict
        Notice payload.
    topic : str
        Topic to parse it as.
    root : pathlib.Path
        Store root.
    received_time : str, optional
        Fixed arrival time.
    resolve : callable, optional
        Skymap writer; defaults to fake_resolve().

    Returns
    -------
    dict
        The history entry.
    """
    record = gcn_notices.parse_notice(topic, payload)
    raw = json.dumps(payload).encode("utf-8")
    return gcn_store.store_notice(
        record,
        raw,
        root=root,
        resolve=resolve or fake_resolve(),
        received_time=received_time,
    )


def test_gw_grb_and_neutrino_land_in_separate_trees(tmp_path):
    """The layout requirement from the issue: GRB and neutrino maps in their own folders."""
    store(igwn_gwalert(), gcn_notices.TOPIC_IGWN_GWALERT, tmp_path)
    store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    store(icecube_gold_bronze(0), gcn_notices.TOPIC_ICECUBE_GOLD_BRONZE, tmp_path)
    store(boom_alert(), gcn_notices.TOPIC_BOOM, tmp_path)

    assert (tmp_path / "gw" / "lvk" / IGWN_SUPEREVENT_ID).is_dir()
    assert (tmp_path / "grb" / "swift_bat_guano" / GUANO_TRIGGER_ID).is_dir()
    assert (tmp_path / "neutrino" / "icecube_gold_bronze" / ICECUBE_EVENT_NAME).is_dir()
    assert (tmp_path / "optical" / "boom").is_dir()
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_dir()) == [
        "grb",
        "gw",
        "neutrino",
        "optical",
    ]


def test_stored_notice_keeps_the_payload_verbatim(tmp_path):
    """The notice on disk has to be the notice we received, not our re-serialization of it."""
    payload = swift_bat_guano(3)
    raw = json.dumps(payload).encode("utf-8")
    record = gcn_notices.parse_notice(gcn_notices.TOPIC_SWIFT_BAT_GUANO, payload)
    entry = gcn_store.store_notice(
        record, raw, root=tmp_path, resolve=fake_resolve(), received_time=RECEIVED_TIME
    )
    directory = gcn_store.event_dir(record, root=tmp_path)
    assert (directory / entry["notice_path"]).read_bytes() == raw


def test_stem_reads_as_the_event_history(tmp_path):
    """Stems carry identifier, sequence, alert type, time and digest, in that order."""
    entry = store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    stem = entry["stem"]
    assert stem.startswith(f"{GUANO_TRIGGER_ID}__r003__update__20220130T070300Z__")
    assert stem.endswith(entry["digest"])
    assert len(entry["digest"]) == gcn_store.DIGEST_CHARS


def test_igwn_stem_falls_back_to_the_notice_time(tmp_path):
    """IGWN publishes no record_number, so the stem has only time_created to order by."""
    entry = store(igwn_gwalert(), gcn_notices.TOPIC_IGWN_GWALERT, tmp_path)
    assert "__r" not in entry["stem"]
    assert entry["stem"].startswith(f"{IGWN_SUPEREVENT_ID}__preliminary__20181101T223449Z__")


def test_einstein_probe_stem_falls_back_to_the_trigger_time(tmp_path):
    """WXT has no alert_datetime at all, so the trigger time is the only stamp available."""
    from gcn_examples import einstein_probe_wxt

    entry = store(einstein_probe_wxt(), gcn_notices.TOPIC_EINSTEIN_PROBE_WXT, tmp_path)
    assert "20240226T053126Z" in entry["stem"]


def test_redelivered_message_is_recognized_and_not_rewritten(tmp_path):
    """An at-least-once consumer replays messages on restart; that must be a no-op."""
    first = store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    second = store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    assert first["stored"] is True
    assert second["stored"] is False
    assert second["stem"] == first["stem"]
    directory = tmp_path / "grb" / "swift_bat_guano" / GUANO_TRIGGER_ID
    assert len(gcn_store.read_jsonl(directory / gcn_store.HISTORY_NAME)) == 1
    assert len(gcn_store.iter_index(root=tmp_path)) == 1
    assert len(list((directory / gcn_store.NOTICE_SUBDIR).iterdir())) == 1


def test_every_version_of_an_event_is_kept(tmp_path):
    """A notice sequence is the record of what was known when, so nothing is overwritten."""
    for record_number in (1, 2, 3):
        store(swift_bat_guano(record_number), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    directory = tmp_path / "grb" / "swift_bat_guano" / GUANO_TRIGGER_ID
    history = gcn_store.read_jsonl(directory / gcn_store.HISTORY_NAME)
    assert [entry["record_number"] for entry in history] == [1, 2, 3]
    assert len(list((directory / gcn_store.NOTICE_SUBDIR).iterdir())) == 3
    # Record 1 has no localization, so only records 2 and 3 wrote a map.
    assert len(list((directory / gcn_store.SKYMAP_SUBDIR).iterdir())) == 2


def test_latest_pointer_tracks_the_highest_record_number(tmp_path):
    """The mission's sequence number is what says which notice supersedes which."""
    for record_number in (1, 2, 3):
        store(swift_bat_guano(record_number), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    directory = tmp_path / "grb" / "swift_bat_guano" / GUANO_TRIGGER_ID
    pointer = json.loads((directory / gcn_store.LATEST_POINTER_NAME).read_text())
    assert pointer["record_number"] == 3
    assert pointer["latest_skymap"].startswith(gcn_store.SKYMAP_SUBDIR)
    assert (directory / pointer["latest_skymap"]).exists()


def test_latest_pointer_is_recomputed_so_out_of_order_arrival_is_safe(tmp_path):
    """GCN replays and network reordering must not leave an old notice as latest."""
    store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    store(swift_bat_guano(1), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    directory = tmp_path / "grb" / "swift_bat_guano" / GUANO_TRIGGER_ID
    pointer = json.loads((directory / gcn_store.LATEST_POINTER_NAME).read_text())
    assert pointer["record_number"] == 3


def test_latest_skymap_survives_a_newer_notice_that_carries_no_map(tmp_path):
    """An update need not repeat the map, and losing the map we have would be a regression."""
    store(swift_bat_guano(2), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    later = swift_bat_guano(3)
    for key in ("ra", "dec", "ra_dec_error"):
        later.pop(key)
    store(later, gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)

    directory = tmp_path / "grb" / "swift_bat_guano" / GUANO_TRIGGER_ID
    pointer = json.loads((directory / gcn_store.LATEST_POINTER_NAME).read_text())
    assert pointer["record_number"] == 3
    assert pointer["skymaps"] == []
    # The pointer still names the map from record 2, which is the best we have.
    assert pointer["latest_skymap"] is not None
    assert "r002" in pointer["latest_skymap_stem"]


def test_retraction_becomes_latest_and_withdraws_the_map(tmp_path):
    """Once an event is withdrawn, presenting its last map as current would be wrong."""
    store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    store(swift_bat_guano(4), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    directory = tmp_path / "grb" / "swift_bat_guano" / GUANO_TRIGGER_ID
    pointer = json.loads((directory / gcn_store.LATEST_POINTER_NAME).read_text())
    assert pointer["is_retraction"] is True
    assert pointer["latest_skymap"] is None
    assert not (directory / gcn_store.LATEST_SKYMAP_LINK_NAME).exists()
    # The retracted map itself is still on disk, because the history is the point.
    assert len(list((directory / gcn_store.SKYMAP_SUBDIR).iterdir())) == 1
    assert (
        gcn_store.latest_skymap_path(
            gcn_notices.CATEGORY_GRB, "swift_bat_guano", GUANO_TRIGGER_ID, root=tmp_path
        )
        is None
    )


def test_igwn_versions_order_by_alert_type_when_no_record_number_exists(tmp_path):
    """IGWN's only ordering signals are alert_type and time_created; both must be used."""
    for alert_type in ("EARLYWARNING", "PRELIMINARY", "INITIAL", "UPDATE"):
        store(igwn_gwalert(alert_type=alert_type), gcn_notices.TOPIC_IGWN_GWALERT, tmp_path)
    directory = tmp_path / "gw" / "lvk" / IGWN_SUPEREVENT_ID
    pointer = json.loads((directory / gcn_store.LATEST_POINTER_NAME).read_text())
    assert pointer["alert_type"] == "update"


def test_latest_symlink_points_at_the_current_map(tmp_path):
    """A stable path is what lets the crossmatch read "the current map" without an index."""
    store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    directory = tmp_path / "grb" / "swift_bat_guano" / GUANO_TRIGGER_ID
    link = directory / gcn_store.LATEST_SKYMAP_LINK_NAME
    assert link.is_symlink()
    assert link.resolve().read_bytes() == b"PLACEHOLDER"
    assert gcn_store.latest_skymap_path(
        gcn_notices.CATEGORY_GRB, "swift_bat_guano", GUANO_TRIGGER_ID, root=tmp_path
    ).exists()


def test_multiple_localizations_get_one_map_each(tmp_path):
    """IGWN's combined external map and IceCube's per-neutrino regions must not collide."""
    entry = store(igwn_gwalert(with_external_coinc=True), gcn_notices.TOPIC_IGWN_GWALERT, tmp_path)
    assert len(entry["skymaps"]) == 2
    paths = {skymap["path"] for skymap in entry["skymaps"]}
    assert len(paths) == 2
    assert any(gcn_notices.COMBINED_SKYMAP_LABEL in path for path in paths)


def test_index_records_the_source_of_every_map(tmp_path):
    """A synthesized region must never be mistaken for an observed one downstream."""
    store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    entries = gcn_store.iter_index(root=tmp_path)
    assert [skymap["source"] for skymap in entries[0]["skymaps"]] == ["test"]
    assert entries[0]["skymaps"][0]["containment_probability"] == 0.9
    assert entries[0]["skymaps"][0]["semi_major_deg"] == 0.5


def test_notice_with_no_localization_is_still_recorded(tmp_path):
    """GUANO record 1 and IGWN retractions carry no map, and both still belong in the store."""
    entry = store(swift_bat_guano(1), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    assert entry["skymaps"] == []
    assert entry["localizations_without_map"] == 0
    boom = store(boom_alert(), gcn_notices.TOPIC_BOOM, tmp_path)
    # BOOM quotes a position but no error region, so it is a localization we cannot map.
    assert boom["localizations_without_map"] == 1


def test_find_events_joins_a_counterpart_back_to_its_gw_event(tmp_path):
    """This is how a GRB or optical counterpart is tied to the superevent it belongs to."""
    store(boom_alert(), gcn_notices.TOPIC_BOOM, tmp_path)
    store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    matches = gcn_store.find_events(root=tmp_path, related_id=BOOM_CROSSMATCH_ID)
    assert [entry["source"] for entry in matches] == ["boom"]
    assert gcn_store.find_events(root=tmp_path, category=gcn_notices.CATEGORY_GRB)
    assert gcn_store.find_events(root=tmp_path, related_id="S000000zz") == []


def test_find_events_matches_a_superevent_inside_a_versioned_reference(tmp_path):
    """IceCube references a GW map revision as "S230914ak-2-Preliminary"; the id still matches."""
    from gcn_examples import LVK_NU_REFERENCE_ID, icecube_lvk_nu_track_search

    store(
        icecube_lvk_nu_track_search(),
        gcn_notices.TOPIC_ICECUBE_LVK_NU_TRACK_SEARCH,
        tmp_path,
    )
    assert gcn_store.find_events(root=tmp_path, related_id=LVK_NU_REFERENCE_ID.lower())


def test_event_directories_lists_every_stored_event(tmp_path):
    """A directory walk has to agree with the index, or reconciliation is impossible."""
    store(igwn_gwalert(), gcn_notices.TOPIC_IGWN_GWALERT, tmp_path)
    store(swift_bat_guano(3), gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    directories = gcn_store.event_directories(root=tmp_path)
    assert len(directories) == 2
    assert {path.name for path in directories} == {IGWN_SUPEREVENT_ID, GUANO_TRIGGER_ID}
    assert gcn_store.event_directories(root=tmp_path / "missing") == []


def test_quarantine_keeps_the_payload_and_the_reason(tmp_path):
    """The listener commits a bad message's offset, which is only safe if we kept the payload."""
    path = gcn_store.quarantine_payload(
        gcn_notices.TOPIC_BOOM,
        b"not json at all",
        ValueError("boom"),
        root=tmp_path,
        received_time=RECEIVED_TIME,
    )
    assert path.read_bytes() == b"not json at all"
    error_path = path.with_name(path.name.replace(".payload", ".error.json"))
    details = json.loads(error_path.read_text())
    assert details["error_type"] == "ValueError"
    assert details["topic"] == gcn_notices.TOPIC_BOOM
    assert details["error"] == "boom"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-08-06T12:00:00Z", "20260806T120000Z"),
        ("2026-08-06T12:00:00+00:00", "20260806T120000Z"),
        ("2026-08-06T13:00:00+01:00", "20260806T120000Z"),
        ("2026-08-06T12:00:00", "20260806T120000Z"),
        (None, None),
        ("", None),
    ],
)
def test_compact_timestamp_normalizes_to_utc(value, expected):
    """Stems have to sort chronologically, which they only do if every stamp is UTC."""
    assert gcn_store.compact_timestamp(value) == expected


def test_compact_timestamp_keeps_something_traceable_for_junk():
    """An unrecognized format should still yield a usable file name rather than vanish."""
    stamp = gcn_store.compact_timestamp("not a timestamp")
    assert stamp
    assert "/" not in stamp


@pytest.mark.parametrize("hostile_id", ["../../etc/passwd", "..", ".", "/absolute", "..."])
def test_event_id_cannot_escape_the_store(tmp_path, hostile_id):
    """Identifiers come off the network, so they must never be trusted as path components."""
    payload = swift_bat_guano(3)
    payload["id"] = [hostile_id]
    entry = store(payload, gcn_notices.TOPIC_SWIFT_BAT_GUANO, tmp_path)
    directory = tmp_path / entry["event_dir"]
    assert directory.resolve().is_relative_to(tmp_path.resolve())
    # Three components exactly: nothing collapsed away and nothing added.
    assert len(Path(entry["event_dir"]).parts) == 3
    assert directory.is_dir()


def test_dotted_event_id_is_replaced_rather_than_resolved(tmp_path):
    """A name of only dots is a valid file name but resolves to another directory."""
    assert gcn_store.safe_path_part("..") == gcn_store.UNSAFE_PATH_PART
    assert gcn_store.safe_path_part(".") == gcn_store.UNSAFE_PATH_PART
    assert gcn_store.safe_path_part("/") == gcn_store.UNSAFE_PATH_PART
    # An ordinary identifier is untouched, dots and all.
    assert gcn_store.safe_path_part("IceCube-260425A") == "IceCube-260425A"
    assert gcn_store.safe_path_part("v1.2") == "v1.2"


def test_read_jsonl_skips_malformed_lines(tmp_path):
    """A truncated final line from an interrupted write must not break the whole history."""
    path = tmp_path / "history.jsonl"
    path.write_text('{"a": 1}\n\n{"b": broken\n{"c": 3}\n')
    assert gcn_store.read_jsonl(path) == [{"a": 1}, {"c": 3}]
    assert gcn_store.read_jsonl(tmp_path / "absent.jsonl") == []


def test_write_atomic_leaves_no_temporary_files(tmp_path):
    """A reader polling the store must never see a half-written file."""
    gcn_store.write_atomic_json(tmp_path / "nested" / "out.json", {"a": 1})
    assert json.loads((tmp_path / "nested" / "out.json").read_text()) == {"a": 1}
    assert not list((tmp_path / "nested").glob(f"*{gcn_store.TEMP_SUFFIX}"))
