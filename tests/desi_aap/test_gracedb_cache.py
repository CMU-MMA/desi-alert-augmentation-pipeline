import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from desi_aap import gracedb_cache
from desi_aap.config import PipelineConfig
from desi_aap.gracedb_cache import GraceDbCache

# A merger time comfortably outside any recheck window, and the clock the tests judge it against.
OLD_GW_TIME = pd.Timestamp("2019-04-25 08:18:05", tz="UTC")
NOW = pd.Timestamp("2026-08-11 00:00:00", tz="UTC")


def make_superevent(superevent_id="S190425z", **overrides):
    """Build a minimal GraceDB superevent dict, overriding any top-level key."""
    superevent = {
        "superevent_id": superevent_id,
        "far": 1e-9,
        "t_0": 1240215503.017147,
        "labels": ["PE_READY", "SKYMAP_READY"],
        "preferred_event_data": {"graceid": "G330561", "pipeline": "gstlal"},
    }
    superevent.update(overrides)
    return superevent


def test_safe_file_part_collapses_unsafe_characters() -> None:
    """Verify `safe_file_part` replaces each run of unsafe characters with one underscore"""
    assert gracedb_cache.safe_file_part("bayestar.multiorder.fits") == "bayestar.multiorder.fits"
    assert gracedb_cache.safe_file_part("bayestar.fits,0") == "bayestar.fits_0"
    assert gracedb_cache.safe_file_part("a/b  c") == "a_b_c"


def test_atomic_write_bytes_creates_parents_and_leaves_no_temp_file(tmp_path) -> None:
    """Verify `atomic_write_bytes` writes through a temporary file it then cleans up"""
    path = tmp_path / "deep" / "nested" / "skymap.fits"
    assert gracedb_cache.atomic_write_bytes(path, b"FITS") == path
    assert path.read_bytes() == b"FITS"
    assert list(path.parent.glob(f"*{gracedb_cache.TEMP_SUFFIX}")) == []


def test_atomic_write_bytes_replaces_an_existing_file(tmp_path) -> None:
    """Verify `atomic_write_bytes` overwrites a file already at the destination"""
    path = tmp_path / "skymap.fits"
    gracedb_cache.atomic_write_bytes(path, b"OLD")
    gracedb_cache.atomic_write_bytes(path, b"NEWER PAYLOAD")
    assert path.read_bytes() == b"NEWER PAYLOAD"


def test_atomic_write_bytes_removes_the_temp_file_when_writing_fails(tmp_path) -> None:
    """Verify `atomic_write_bytes` leaves nothing behind when the payload cannot be written"""
    path = tmp_path / "skymap.fits"
    with pytest.raises(TypeError):
        gracedb_cache.atomic_write_bytes(path, "not bytes")
    assert not path.exists()
    assert list(tmp_path.glob(f"*{gracedb_cache.TEMP_SUFFIX}")) == []


def test_superevent_fingerprint_reads_the_mutable_listing_fields() -> None:
    """Verify `superevent_fingerprint` captures labels, preferred event, FAR and t_0"""
    assert gracedb_cache.superevent_fingerprint(make_superevent()) == {
        "labels": ["PE_READY", "SKYMAP_READY"],
        "preferred_event": "G330561",
        "far": 1e-9,
        "t_0": 1240215503.017147,
    }


def test_superevent_fingerprint_ignores_label_order() -> None:
    """Verify `superevent_fingerprint` sorts labels, since GraceDB does not return them in order"""
    reordered = make_superevent(labels=["SKYMAP_READY", "PE_READY"])
    assert gracedb_cache.superevent_fingerprint(reordered) == gracedb_cache.superevent_fingerprint(
        make_superevent()
    )


def test_superevent_fingerprint_tolerates_missing_fields() -> None:
    """Verify `superevent_fingerprint` handles a superevent with no labels or preferred event"""
    bare = {"superevent_id": "S190426c", "preferred_event_data": {}}
    assert gracedb_cache.superevent_fingerprint(bare) == {
        "labels": [],
        "preferred_event": None,
        "far": None,
        "t_0": None,
    }
    assert gracedb_cache.superevent_fingerprint({"preferred_event_data": None})["preferred_event"] is None


def test_superevent_fingerprint_survives_a_json_round_trip() -> None:
    """Verify a fingerprint compares equal after being stored and read back as JSON"""
    fingerprint = gracedb_cache.superevent_fingerprint(make_superevent())
    assert json.loads(json.dumps(fingerprint)) == fingerprint


def test_latest_revision_finds_the_highest_numbered_copy() -> None:
    """Verify `latest_revision` returns the largest ",N" GraceDB lists for a file"""
    # The shape S250206dm has on the live API: the unversioned name plus two revisions.
    files = [
        "Bilby.multiorder.fits",
        "Bilby.multiorder.fits,0",
        "Bilby.multiorder.fits,1",
        "bayestar.multiorder.fits",
        "bayestar.multiorder.fits,0",
    ]
    assert gracedb_cache.latest_revision(files, "Bilby.multiorder.fits") == 1
    assert gracedb_cache.latest_revision(files, "bayestar.multiorder.fits") == 0


def test_latest_revision_returns_none_without_a_versioned_copy() -> None:
    """Verify `latest_revision` reports None when the listing holds no ",N" entry"""
    assert gracedb_cache.latest_revision(["bayestar.multiorder.fits"], "bayestar.multiorder.fits") is None
    assert gracedb_cache.latest_revision([], "bayestar.multiorder.fits") is None


def test_latest_revision_ignores_a_versioned_basename() -> None:
    """Verify `latest_revision` reports None for a name that already names one fixed revision"""
    files = ["bayestar.fits", "bayestar.fits,0", "bayestar.fits,1"]
    assert gracedb_cache.latest_revision(files, "bayestar.fits,0") is None


def test_latest_revision_ignores_a_non_numeric_suffix() -> None:
    """Verify `latest_revision` skips entries whose suffix after the comma is not a number"""
    assert gracedb_cache.latest_revision(["a.fits,draft", "a.fits,2"], "a.fits") == 2


def test_cache_lays_out_its_subdirectories(superevent_cache) -> None:
    """Verify a `GraceDbCache` puts entries and skymaps in named subdirectories of its root"""
    assert superevent_cache.entry_dir == superevent_cache.cache_dir / "superevents"
    assert superevent_cache.skymap_dir == superevent_cache.cache_dir / "skymaps"
    assert superevent_cache.entry_path("S190425z") == superevent_cache.entry_dir / "S190425z.json"


def test_cache_resolves_a_stored_path_against_its_root(superevent_cache) -> None:
    """Verify `resolve` turns an entry's relative path into an absolute one"""
    resolved = superevent_cache.resolve("skymaps/S190425z__bayestar.multiorder.fits")
    assert resolved == (superevent_cache.skymap_dir / "S190425z__bayestar.multiorder.fits").resolve()
    assert resolved.is_absolute()
    assert superevent_cache.resolve(None) is None


def test_cache_round_trips_an_entry(superevent_cache) -> None:
    """Verify an entry written to the cache reads back unchanged"""
    entry = {"superevent_id": "S190425z", "files": ["a.fits"], "skymap_revision": 1}
    superevent_cache.write_entry("S190425z", entry)
    assert superevent_cache.read_entry("S190425z") == entry


def test_cache_reports_a_missing_entry_as_absent(superevent_cache) -> None:
    """Verify `read_entry` returns None when nothing has been cached for a superevent"""
    assert superevent_cache.read_entry("S190425z") is None


@pytest.mark.parametrize("damage", [b"", b"{not json", b'"a string, not an object"'])
def test_cache_treats_a_damaged_entry_as_absent(superevent_cache, damage) -> None:
    """Verify `read_entry` reports an empty, truncated or non-object entry as absent, not an error"""
    path = superevent_cache.entry_path("S190425z")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(damage)
    assert superevent_cache.read_entry("S190425z") is None


def test_cache_ignores_a_temp_file_left_by_a_killed_run(superevent_cache) -> None:
    """Verify a leftover temporary file is not mistaken for a cache entry"""
    superevent_cache.write_entry("S190425z", {"files": []})
    leftover = superevent_cache.entry_path("S190425z").with_suffix(f".json{gracedb_cache.TEMP_SUFFIX}")
    leftover.write_bytes(b"{half wri")
    assert superevent_cache.read_entry("S190425z") == {"files": []}


def test_cache_status_reports_a_miss_without_an_entry(superevent_cache) -> None:
    """Verify `status` reports "miss" when nothing is cached"""
    superevent = make_superevent()
    assert superevent_cache.status(None, superevent, gw_time=OLD_GW_TIME, now=NOW) == "miss"
    assert superevent_cache.status({}, superevent, gw_time=OLD_GW_TIME, now=NOW) == "miss"


def test_cache_status_reports_a_hit_for_a_settled_superevent(superevent_cache) -> None:
    """Verify `status` reports "hit" when the fingerprint matches and the event is old"""
    superevent = make_superevent()
    entry = {"fingerprint": gracedb_cache.superevent_fingerprint(superevent)}
    assert superevent_cache.status(entry, superevent, gw_time=OLD_GW_TIME, now=NOW) == "hit"


@pytest.mark.parametrize(
    "overrides",
    [
        {"labels": ["PE_READY", "SKYMAP_READY", "ADVOK"]},
        {"preferred_event_data": {"graceid": "G999999"}},
        {"far": 2e-9},
        {"t_0": 1240215504.0},
    ],
    ids=["labels", "preferred_event", "far", "t_0"],
)
def test_cache_status_reports_a_moved_field_as_stale(superevent_cache, overrides) -> None:
    """Verify `status` reports "stale_fingerprint" when any listed field has changed"""
    entry = {"fingerprint": gracedb_cache.superevent_fingerprint(make_superevent())}
    changed = make_superevent(**overrides)
    assert superevent_cache.status(entry, changed, gw_time=OLD_GW_TIME, now=NOW) == "stale_fingerprint"


def test_cache_status_ignores_a_label_reordering(superevent_cache) -> None:
    """Verify `status` still reports "hit" when only the order of the labels changed"""
    entry = {"fingerprint": gracedb_cache.superevent_fingerprint(make_superevent())}
    reordered = make_superevent(labels=["SKYMAP_READY", "PE_READY"])
    assert superevent_cache.status(entry, reordered, gw_time=OLD_GW_TIME, now=NOW) == "hit"


def test_cache_status_rechecks_a_recent_superevent(superevent_cache) -> None:
    """Verify `status` reports "stale_age" inside the recheck window even on an intact fingerprint"""
    superevent = make_superevent()
    entry = {"fingerprint": gracedb_cache.superevent_fingerprint(superevent)}
    recent = NOW - timedelta(days=3)
    assert superevent_cache.status(entry, superevent, gw_time=recent, now=NOW) == "stale_age"


def test_cache_status_honors_the_recheck_window(tmp_path) -> None:
    """Verify the recheck window is the boundary between a re-check and a hit"""
    superevent_cache = GraceDbCache(cache_dir=tmp_path, recheck_window=timedelta(days=7))
    superevent = make_superevent()
    entry = {"fingerprint": gracedb_cache.superevent_fingerprint(superevent)}
    inside = NOW - timedelta(days=6, hours=23)
    outside = NOW - timedelta(days=7, hours=1)
    assert superevent_cache.status(entry, superevent, gw_time=inside, now=NOW) == "stale_age"
    assert superevent_cache.status(entry, superevent, gw_time=outside, now=NOW) == "hit"


def test_cache_status_rechecks_a_superevent_with_no_time(superevent_cache) -> None:
    """Verify `status` re-checks rather than trusts a superevent whose merger time is missing"""
    superevent = make_superevent()
    entry = {"fingerprint": gracedb_cache.superevent_fingerprint(superevent)}
    assert superevent_cache.status(entry, superevent, gw_time=pd.NaT, now=NOW) == "stale_age"


def test_cache_defaults_its_recheck_window_to_thirty_days(superevent_cache) -> None:
    """Verify the recheck window defaults to 30 days, the one place that number is written"""
    assert superevent_cache.recheck_window == timedelta(days=30)


# ---------------------------------------------------------------------------
# The [gracedb] config section, which is how a cache is built in production.
# ---------------------------------------------------------------------------

# The two sections PipelineConfig requires, so a [gracedb] table can be validated on its own.
REQUIRED_SECTIONS = {
    "run": {"output_dir": "out"},
    "query": {"boom": {"survey": "LSST"}, "window": {"lookback": "1h"}},
}


def load_gracedb_section(**gracedb):
    """Validate a PipelineConfig carrying just the given [gracedb] keys, and return that section."""
    return PipelineConfig.model_validate({**REQUIRED_SECTIONS, "gracedb": gracedb}).gracedb


def test_config_without_a_gracedb_section_still_validates() -> None:
    """Verify a config file predating the [gracedb] section is still accepted"""
    config = PipelineConfig.model_validate(REQUIRED_SECTIONS)
    assert config.gracedb.cache_dir == Path("gracedb_cache")


def test_gracedb_config_builds_a_cache() -> None:
    """Verify `to_cache` turns the section into the cache it describes"""
    section = load_gracedb_section(cache_dir="/data/gdb", recheck_window="7d")
    assert section.to_cache() == GraceDbCache(cache_dir=Path("/data/gdb"), recheck_window=timedelta(days=7))


def test_gracedb_config_leaves_the_recheck_window_to_the_cache() -> None:
    """Verify an unset recheck_window falls through to GraceDbCache's own default"""
    section = load_gracedb_section(cache_dir="/data/gdb")
    assert section.recheck_window is None
    assert section.to_cache().recheck_window == GraceDbCache(cache_dir=Path(".")).recheck_window


def test_gracedb_config_rejects_an_unknown_key() -> None:
    """Verify a typo in the [gracedb] section is an error rather than being ignored"""
    with pytest.raises(ValidationError):
        load_gracedb_section(cache_directory="/data/gdb")
