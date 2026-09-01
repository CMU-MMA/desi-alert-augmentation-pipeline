# PR description — integrate the GCN store into the localize read path

Running draft. Decisions get recorded here as they are settled, so the final PR description is
assembled rather than written from memory. Companion to `desi-aap-gcn-integration-plan.md`
(the design) and `followup_issues.md` (what is deferred).

**Status:** design settled, not yet implemented.

---

## Summary

`localize` reads superevents from GraceDB only. The live GCN Kafka feed has been archiving LVK
alerts into `gcn_localizations/` since PR #23, and nothing outside `scripts/listen_gcn.py`
reads them. This makes `fetch_gracedb_superevents` return the union of both sources, with each
row saying where its metadata and its skymap came from. Neither store changes shape.

What that buys, concretely:

- **Early-warning and preliminary maps that GraceDB has since replaced.** These are not
  recoverable from GraceDB at all — the archive is the only copy. This is the main win.
- **Superevents the live listing does not return**, for whatever reason, that the Kafka feed
  did deliver.

What it does *not* buy, to be clear about it: the sweep runs inside a scheduled scan and
behind the same `client.superevents()` call, so no alert is seen sooner than it would have
been, and if the listing query raises, the GCN rows are lost with it.

## New columns

| Column | Values |
|---|---|
| `origin` | `"gracedb"`, `"gcn"`, `"gracedb+gcn"` |
| `skymap_origin` | `"gracedb"`, `"gcn_notice_file"`, `"gcn_notice_url"`, `"gcn_synthesized"`, or `None` |

Both are also copied by `temporal_crossmatch_sesn_to_gw` as `gw_origin` / `gw_skymap_origin`,
since without the latter a reader cannot tell whether `gw_skymap_path` is a real LVK map or a
synthesized error ellipse.

---

## Decisions

### GCN data is read, never promoted into `gracedb_cache`

The cache's invariant is that all of it is re-derivable — the README says so, and says to
delete the directory to start over. A GCN-carried preliminary map is *not* re-derivable, since
GraceDB has replaced it. Promotion would make `rm -rf gracedb_cache` destroy unrecoverable
data; a cache and an archive have different safety levels, and mixing them demotes the archive.

It also saves nothing. The live listing drives the read loop, so a cache entry the listing does
not name is never read — `fetch_gracedb_superevents` has to learn about the second source
either way. Given that, it can read the GCN store directly.

### GraceDB wins on metadata; GCN fills only gaps

Where both sources hold a superevent, every metadata column — time, FAR, classification,
pipeline, search — comes from GraceDB. The GCN notice contributes only a skymap, and only when
GraceDB has none.

The map lookup deliberately applies **no cuts**, which is why it is a separate function from
the sweep rather than a reuse of it. Consider S123: GraceDB lists it, its authoritative
`p_astro.json` clears the 0.9 cut, but the skymap download failed. The GCN store holds an early
preliminary whose *inline* classification was 0.6. Gating the map on the notice's own cuts
would drop it, leaving S123 with `missing_skymap` and failing every SN near it spatially — even
though GraceDB had already ruled the event in.

**Cuts decide whether to admit an event. They do not decide whether to trust a map.**

The one place GCN metadata is used for a superevent GraceDB listed is a row whose *file
listing* failed. That row keeps its `file_list_failed` status and its listing-derived FAR,
takes its time and classification from the notice, and reports `origin = "gracedb+gcn"`.
GraceDB is not overridden there — it produced nothing to override. The distinction that matters
is between a superevent GraceDB *ruled out* (failed the FAR or p_astro cut — the sweep must not
resurrect it on the notice's inline classification) and one it simply *failed to answer about*.

### MDC alerts are excluded, with no option to include them

`fetch_gracedb_superevents` already queries GraceDB with `category: Production`, hardcoded and
unparameterized. The pipeline has therefore always excluded MDC superevents from its primary
source. Excluding them from the GCN path is not a new policy — it is making the second source
obey the one the first already enforces.

Leaving it out would give the incoherent version: the same MDC superevent dropped when it
arrives via GraceDB and admitted when it arrives via Kafka, so whether a mock event enters the
science frame depends on which pipe it came down. If MDC data is ever wanted, that is one
decision to make at `category`, for both sources.

The sweep logs a count of what it skipped, so the exclusion is visible rather than silent.

### `significant` is not cut on

The FAR cut is expected to subsume it: `far_threshold_per_year = 2.0`, and low-significance
alerts are published at far higher FAR. Adding a second cut would be a second place for the
same decision to live, and `test_no_scientific_cut_carries_a_default` pins the convention that
this module states no scientific cut of its own.

Whether an alert can pass a 2/yr FAR cut while still being flagged `significant: false` is
unconfirmed — see follow-up issue 4.

### `cache_status` is `None` for GCN rows

`cache_status` documents what the *cache* did, across five values. A GCN row never touched the
cache, so a sixth value there would conflate two axes. `origin` already carries this, and the
`n_superevents_gcn_only` summary count reads from it.

### `skymap_file` names the map the row actually points at

`skymap_file` is a display column — it feeds `summarize_temporal_matches`' reading view and
nothing computes from it. For GraceDB rows it holds the *remote* name from the file listing
(`bilby.multiorder.fits`), not a path. The GCN analogue is therefore the notice's own
`event.skymap_filename`, which is what that source calls the map.

Taken from **the notice that supplied the stored map**, not from the newest notice for the
event. Those differ: `latest_skymap_entry` deliberately returns the newest notice *carrying a
map*, which an update can supersede without repeating. Reading `skymap_filename` off the newest
notice would name a file that `skymap_path` does not point at.

This keeps `None` meaning exactly one thing — *no map was named* — on both sides. It is already
taken: a GraceDB superevent whose listing has no skymap gets `skymap_file = None`, pinned by
`test_fetch_gracedb_superevents_handles_a_missing_skymap`. Blanking the column for every GCN
row would have given it a second meaning, and a sentinel like `"<gcn>"` would put a
non-filename into a column of filenames while duplicating what `skymap_origin` already says.

Two details: read it with `.get`, since a notice carrying a map need not name it, and `None`
there is the correct answer anyway; and when the stored map is the joint external-coincidence
map (`label == "combined"`), the name is `external_coinc.combined_skymap_filename`.

**Where the map notice comes from.** `latest.json` already carries `latest_skymap_stem`, and
`store_notice` stores every notice at `notices/{stem}.json`, so the map's notice is reachable
from the pointer without reading `history.jsonl`. The reader stays on `latest.json` — the
documented entry point, a fixed-size read, and the cached answer to `entry_sort_key`'s
out-of-order and retraction handling, which nothing else should re-derive.

The one gap that closes with it: reconstructing `notices/{stem}.json` in a second place would
couple this module to a naming rule `gcn_store` owns. Add a small `gcn_store.notice_path(directory, stem)`
used by both, so the rule has exactly one definition.

### Scope is `gw`/`lvk` only

An allowlist, not an exclusion, and not merely tidiness:
`parse_icecube_lvk_nu_track_search` keys its topic on `ref_ID`, so a **neutrino** notice is
filed under a superevent id (`neutrino/icecube_lvk_nu_track_search/S230914ak/`). Selecting on
"the event id looks like a superevent" would pull a neutrino error circle in as a superevent.
There is a regression test on this.

GRB, neutrino and optical notices need a key space that is not `superevent_id`, and their maps
are synthesized from an error ellipse so they carry no `DISTMU` and are skipped by
`run_3d_spatial_crossmatch`. Deferred — see `followup_issues.md`.

---

## Open

- **Which signal identifies an MDC alert** — `search == "MDC"`, the `M` id prefix, or both. The
  id prefix is filterable from the history entry without opening a payload, and would also
  catch test-domain events. Needs one check against the schema before it goes in a comment.

## Notes for reviewers

- A store captured from `--domain test.gcn.nasa.gov` is indistinguishable from a production one
  once written: `store_notice` records the topic but not the domain. Use a separate
  `--store-root` when rehearsing. Worth a README line.
- `gcn_store_root` is a required keyword with no default, mirroring `cache`. Opting out is
  `gcn_store_root=None`, spelled out. This touches ~25 existing test call sites.
