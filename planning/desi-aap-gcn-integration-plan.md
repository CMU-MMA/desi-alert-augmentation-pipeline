# Integrate the GCN store into the localize read path

## Context

`desi-alert-augmentation-pipeline` grew two independent on-disk stores that never meet:

- **`gracedb_cache`** (PR #31) — `superevents/<sid>.json` plus `skymaps/`, keyed by superevent id,
  written only by `fetch_gracedb_superevents`, read only by the `localize` stage.
- **`gcn_localizations`** (PR #23) — `<category>/<source>/<event_id>/` with `history.jsonl`,
  `latest.json` and MOC FITS maps, written only by `gcn_listener`. Nothing outside
  `scripts/listen_gcn.py` and its tests imports it.

So the live GCN Kafka feed archives LVK alerts that the pipeline never sees. A public LVK alert
reaches the Kafka topic well before the next scheduled GraceDB scan, and it carries the
early-warning and preliminary maps that GraceDB later replaces — data the pipeline currently
cannot reach at all.

The intended outcome: `localize` consumes both sources through one call, each row says where its
data and its skymap came from, and neither store changes shape.

### The design decision, and why not the obvious one

The first instinct was to *promote* GCN data into `gracedb_cache` — write entries, copy skymaps.
Rejected, for three reasons:

1. **It breaks the cache's invariant.** The README states "To start over, delete the directory:
   all of it is re-derivable." A GCN-carried preliminary map is *not* re-derivable — GraceDB has
   since replaced it. Promotion would make `rm -rf gracedb_cache` destroy unrecoverable data.
   A cache and an archive have different safety levels; mixing them demotes the archive.
2. **It saves no work on the read path.** `fetch_gracedb_superevents` has to learn about a second
   source either way, because the *live listing* drives its loop — a cache entry the listing does
   not name is never read. Given it must read something extra, it can read the GCN store directly.
3. **Downstream does not need the copy.** `run_3d_spatial_crossmatch` calls
   `read_sky_map(skymap_path)` on an absolute path (`gracedb_tools.py:1003-1014`). That path can
   point into `gcn_localizations/` as easily as into `gracedb_cache/skymaps/`.

Three problems disappear with the copy: retractions are already handled by
`latest_skymap_entry` (`gcn_store.py:378-381`, returns `None` once the newest notice is a
retraction); GCN entries have no `files` list and an incomplete `fingerprint`, so promoted
entries would sit permanently `stale_fingerprint`; and no backfill script or listener hook is
needed. The dependency arrow also points the right way — `gracedb_tools` already carries the
`ligo.skymap` chain, whereas a listener hook would have dragged `gracedb_cache` into
`gcn_listener`, against the note at `gcn_store.py:31-33`.

### Scope: `gw/lvk` only

Filter on `category == "gw" and source == "lvk"`, an allowlist rather than an exclusion. This is
not merely tidiness: `parse_icecube_lvk_nu_track_search` (`gcn_notices.py:552`) keys that topic on
`ref_ID`, so a **neutrino** notice is filed under a superevent id
(`neutrino/icecube_lvk_nu_track_search/S230914ak/`). Selecting on "the event id looks like a
superevent" would pull a neutrino error circle in as a superevent.

GRB, neutrino and optical notices are deferred to follow-up issues (drafted below). They need a
key space that is not `superevent_id`, and their maps are *synthesized* from an error ellipse, so
they carry no `DISTMU` and are skipped at `gracedb_tools.py:1020` — consuming them needs a 2D
credible-region path that does not exist yet.

---

## Changes

### 1. New module: `src/desi_aap/gcn_superevents.py`

The projection from GCN notices to superevent-shaped rows. A separate module rather than more
lines in `gracedb_tools.py` (already ~1500) and rather than in `gcn_store.py`, which stays free of
GraceDB concepts.

Reuses `gcn_store.event_directories`, `gcn_store.latest_skymap_path`, `gcn_store.read_jsonl`,
and `gracedb_tools.JULIAN_YEAR_SECONDS` / `as_float`.

```python
# Only igwn.gwalert notices are projected: they are the ones already keyed by superevent id.
# An allowlist rather than an exclusion, because parse_icecube_lvk_nu_track_search keys that
# topic on ref_ID, so a neutrino notice is also filed under a superevent id.
# TODO: support the other categories here. They are skipped only because the cache is
# superevent-keyed and their maps are synthesized, hence carry no DISTMU -- not because
# they are uninteresting.
GCN_CATEGORY = "gw"
GCN_SOURCE = "lvk"

# latest.json's map source -> the skymap_origin reported on a row.
SKYMAP_ORIGIN_BY_SOURCE = {
    "notice_file": "gcn_notice_file",
    "notice_url": "gcn_notice_url",
    "synthesized": "gcn_synthesized",
}

def iter_gw_pointers(root):
    """Yield (event_dir, latest.json) for every gw/lvk event in the store."""

def load_notice_payload(event_dir, pointer):
    """Read the verbatim notice latest.json points at, via its notice_path."""

def gcn_superevent_row(event_dir, pointer, payload, root):
    """Build one fetch_gracedb_superevents-shaped row from a notice."""

def gcn_superevent_rows(root, *, se_types, far_threshold_per_year,
                        min_classification_prob_sum, exclude=frozenset()):
    """Rows for gw/lvk events passing the same cuts, excluding ids already listed."""
```

Field mapping — every value below is present in the `igwn_gwalert` fixture at
`tests/desi_aap/gcn_examples.py:104-140`:

| Row column | Source |
|---|---|
| `superevent_id` | `pointer["event_id"]` |
| `gw_time` | `pd.Timestamp(payload["event"]["time"])` |
| `gps_time` | `Time(gw_time).gps`, so the column stays meaningful downstream |
| `far_hz`, `far_per_year` | `payload["event"]["far"]`, × `JULIAN_YEAR_SECONDS` — same units as the listing's `far` |
| `p_bns`, `p_nsbh`, `p_bbh`, `p_terrestrial` | `payload["event"]["classification"]` — same class keys as `p_astro.json` |
| `classification_file` | `None` — the notice carries the classification inline |
| `pipeline`, `search`, `instruments` | `payload["event"]` |
| `preferred_event`, `labels` | `None` / `""` — a notice carries neither |
| `skymap_file` | `payload["event"]["skymap_filename"]` |
| `skymap_path` | `gcn_store.latest_skymap_path(GCN_CATEGORY, GCN_SOURCE, event_id, root)`, absolute |
| `status` | `"ok"` |
| `cache_status` | `"gcn_only"` |
| `origin` | `"gcn"` |
| `skymap_origin` | `SKYMAP_ORIGIN_BY_SOURCE[pointer["latest_skymap_source"]]`, or `None` |

**Retractions** need no field of their own: skip the event when `pointer["is_retraction"]` is
true. A retracted superevent should not be localized at all, and `latest_skymap_path` already
returns `None` for one.

An event whose notice carried no map keeps its row with `skymap_path = None`, matching how a
listing row survives a failed skymap download.

Note the `latest.json` shape: it holds only `NoticeRecord.summary()`, which has no `far` and no
`classification` (`gcn_notices.py:332-345`). Those live in `payload`, written verbatim to
`notices/<stem>.json`, which `latest.json` names via `notice_path`. Hence `load_notice_payload`.

### 2. `src/desi_aap/gracedb_tools.py` — `fetch_gracedb_superevents`

- New **required** keyword-only `gcn_store_root`, with no default, mirroring `cache`. The
  reasoning at `gracedb_tools.py:404-407` and `gracedb_cache.py:182-187` — "the location is always
  an explicit decision" — applies unchanged, and a silent default here would mean a caller who
  forgets the argument quietly loses the whole GCN feed. Opting out is `gcn_store_root=None`,
  spelled out. A `None` root, or a directory that does not exist, means no GCN rows:
  `gcn_store.event_directories` already returns `[]` for a missing root, so degradation is free.
  Four call sites to update: `stages/localize.py:505`, the README snippet at line 299, and
  `docs/pre_executed/benchmark_loc_map_xmatch.ipynb` / `gracedb_sesn_refactor.ipynb`.
- Every listing row gains `origin: "gracedb"` and `skymap_origin: "gracedb"` (or `None` when the
  row has no map).
- **Collect `listed_ids`** — every `sid` the listing returns, recorded at the top of the loop
  immediately after `sid = superevent.get("superevent_id")`, *before* any cut. This is not the same
  as the ids in `rows`: the FAR cut (`line 503`) and the p_astro cut (`line 580`) both `continue`
  without appending a row. Excluding on `rows` alone would let the sweep resurrect a superevent
  GraceDB has ruled out, judging it on the notice's inline classification instead of the
  authoritative `p_astro.json`. If the listing named it, GraceDB has spoken.
- After the listing loop, sweep the GCN store with `exclude=listed_ids` and extend `rows`. The
  exclusion prevents a *duplicate row* for a superevent both sources hold — two rows would pair
  every SN with that event twice in `temporal_crossmatch_sesn_to_gw`. GCN-only superevents are
  exactly what the sweep is for.
- **Merge rule for a superevent in both** — GraceDB wins on metadata, GCN fills only a missing
  map. Applied to the listing row before the sweep excludes that id:
  - GraceDB has a `skymap_path` → unchanged, `origin` stays `"gracedb"`.
  - GraceDB has none and the GCN store does → adopt the GCN `skymap_path` and its
    `skymap_origin`; set `origin = "gracedb+gcn"`.
- Docstring: document `gcn_store_root`, the two new columns, the `"gcn_only"` value of
  `cache_status`, and that the frame is now the union of the listing and the GCN store rather than
  the listing alone.

### 3. `temporal_crossmatch_sesn_to_gw` (`gracedb_tools.py:713-726`)

Add `"origin"` and `"skymap_origin"` to the copied column list, giving `gw_origin` and
`gw_skymap_origin`. The `gw_` prefix is on the *column name*, not the value — `gw_origin` holds
`"gcn"`, not `"gw_gcn"`. That function's output merges a transient catalog with the event table,
so each of the twelve event fields it copies is prefixed to avoid colliding with a catalog column
of the same name; `origin` and `status` are exactly the names that could collide.

Worth doing because `gw_skymap_path` is already in that list, and without `gw_skymap_origin` a
reader cannot tell whether that path is a real LVK map or a synthesized ellipse. On the live path:
`stages/localize.py:354`. Update the docstring's `gw_*` list at `gracedb_tools.py:681-684`.

### 4. `src/desi_aap/config.py` — new `[gcn]` section

Mirror `GraceDbConfig` exactly, including the relative-path-against-a-root behaviour:

```python
class GcnConfig(_Section):
    """The ``[gcn]`` section: where the GCN notice store lives."""
    store_root: Path = Path("gcn_localizations")   # matches gcn_store.STORE_ROOT

    def to_store_root(self, root: Path | None = None) -> Path: ...
```

Add `gcn: GcnConfig = GcnConfig()` to `PipelineConfig` (`config.py:216-228`), and a commented
`[gcn]` block to `config.toml` next to `[gracedb]`, carrying the same working-directory warning.

### 5. `src/desi_aap/stages/localize.py:505-510`

```python
events = fetch_gracedb_superevents(
    settings.se_types,
    cache=cfg.gracedb.to_cache(),
    gcn_store_root=cfg.gcn.to_store_root(),
    ...
)
```

Add `n_superevents_gcn_only` to `summary` from the `cache_status` counts, so a run's log says how
much came from the live feed.

### 6. Docs

`README.md`, the `## GraceDB` section (lines 288-360): a short subsection on the union — that
`gcn_localizations` is an archive rather than a cache and is *not* re-derivable, that GraceDB wins
where both have data, and what `origin` / `skymap_origin` / `gcn_only` mean.

---

## Conventions to hold to

- Numpydoc docstrings throughout, matching the density of `gracedb_cache.py` — every module,
  function and non-obvious constant carries its reasoning.
- Per `CLAUDE.md`/memory: do not assert unsourced facts about the GCN wire format in comments.
  Anchor each claim to the fixture in `tests/desi_aap/gcn_examples.py` or to the parser in
  `gcn_notices.py` that demonstrably reads the field.
- Any bug found while testing: fix the source, document it in the docstring, leave a plain
  `#TODO: <the specific thing to confirm>` — never pin it in a test. TODOs are not addressed to a
  person; whoever picks the work up owns it.

---

## Verification

1. **New unit tests**, `tests/desi_aap/test_gcn_superevents.py`, building a store with
   `gcn_store.store_notice` over the `gcn_examples.igwn_gwalert()` fixtures:
   - a `PRELIMINARY` with an inline map yields a row with `origin == "gcn"`,
     `skymap_origin == "gcn_notice_file"`, an existing `skymap_path`;
   - `with_skymap=False` yields a row with `skymap_path is None`;
   - a `RETRACTION` after a preliminary yields **no** row;
   - the FAR and p_astro cuts drop a notice whose own values fail them;
   - a `neutrino/icecube_lvk_nu_track_search/S230914ak/` event in the same store is never picked
     up — the regression guarding the key collision. Carries a
     `#TODO: support icecube_lvk_nu_track_search here; it is skipped only because the cache is
     superevent-keyed and its maps are synthesized` so the exclusion reads as deferred work
     rather than a decision that these events do not matter;
   - a missing / empty store root yields `[]`.
2. **Merge tests** in `tests/desi_aap/test_gracedb_tools.py`, alongside the existing cache
   integration tests (~1355-1505), using the `superevent_cache` fixture and a mocked client:
   - a superevent in the listing with a skymap keeps it, `origin == "gracedb"`;
   - one in the listing whose skymap download failed adopts the GCN map,
     `origin == "gracedb+gcn"`;
   - one only in the GCN store appears with `cache_status == "gcn_only"`;
   - **a superevent the listing named but that failed the FAR or p_astro cut does not reappear**
     via the sweep, even when the GCN store holds it with a passing inline classification — the
     regression guarding the `listed_ids` fix;
   - `gcn_store_root=None` reproduces today's frame exactly, columns aside.
3. `pytest tests/desi_aap/ -q` — the full suite, confirming the golden-alert and localize stage
   tests are unmoved.
4. **End to end against real data.** The repo already has a populated `gracedb_cache/` (349
   entries, 15 skymaps) but no `gcn_localizations/`. Populate one with a short live capture:
   ```
   GCN_CLIENT_ID=... GCN_CLIENT_SECRET=... \
     python scripts/listen_gcn.py --topics igwn.gwalert --once --log-level INFO
   ```
   then run the stage and confirm the new columns and the `gcn_only` count in the summary:
   ```
   desi-aap run --stage localize --dry-run
   ```
   `--dry-run` still queries GraceDB and writes the cache, so it exercises the full path without
   producing results.

---

## Follow-up issues — drafts for you to revise and post

Text only. Nothing here gets filed by me: revise the wording and post them yourself, or tell me
explicitly to run `gh issue create` on a specific one.

**"Consume GRB and optical GCN localizations in the localize stage"** — Swift BAT-GUANO, Einstein
Probe WXT, IceCube gold/bronze and BOOM notices are archived but unread. Two blockers, both real
design work: they have no superevent key (GUANO trigger ids, `IceCube-260425A`, ZTF names), and
their maps are synthesized from a quoted error ellipse by `gcn_skymaps.write_synthetic_skymap`, so
they carry no `DISTMU` and `run_3d_spatial_crossmatch` skips them at `gracedb_tools.py:1020`.
Needs a 2D credible-region match alongside the 3D one, and a row identity that is not
`superevent_id`.

**"Join non-GW GCN notices to their superevents"** — `gcn_store.find_events(related_id=...)` already
ties a GUANO record, an IceCube LVK-nu track or a BOOM optical counterpart back to the superevent
it names, matching as a case-insensitive substring so `S230914ak` also finds
`S230914ak-2-Preliminary`. Surface those as multi-messenger context on a localize row. Distinct
from the issue above: this is a join onto rows that already exist, not a new row source.

**Considered and dropped: renaming `gracedb_cache`.** A rename to `superevent_cache` was discussed
while the design still promoted GCN data *into* the cache. The read-time union writes nothing
GCN-sourced there, so the cache holds exactly what its name says. `gracedb_cache` / `gcn_store` is
also a useful pair as it stands: each name carries the source and the storage kind, and the
storage kind — re-derivable cache versus unrecoverable archive — is the real distinction.
