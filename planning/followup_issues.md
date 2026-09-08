# Follow-up issues — desi-alert-augmentation-pipeline

Deferred from the GCN/GraceDB read-time union (see `desi-aap-gcn-integration-plan.md`).
Each section below is one issue, body included. Revise before posting.

Issues 1 and 2 add **new rows**; issue 3 adds **context to rows that already exist**; issues 4
and 5 are questions to settle about filters, not features. Both 4 and 5 are answerable from a few
weeks of captured notices rather than from the spec, so they get cheaper after this PR ships and
the archive fills.

---

## 1. Consume GRB and optical GCN localizations in the localize stage

**Problem.** `gcn_listener` archives Swift BAT-GUANO, Einstein Probe WXT and BOOM notices
into `gcn_localizations/`, but nothing reads them. The localize stage sees only GW
superevents.

**Blockers.**

- **No superevent key.** These events are keyed on their own identifiers — GUANO trigger
  ids (`694215995`), EP trigger ids, ZTF names (`ZTF25acffkdr`). The localize row identity
  is `superevent_id` throughout, including the temporal and spatial crossmatch.
- **No distance information.** Their maps are synthesized from a quoted error ellipse by
  `gcn_skymaps.write_synthetic_skymap`, so they carry no `DISTMU` and are skipped at
  `gracedb_tools.py:1020`. A 3D volume match is not possible.

**Approach.** Needs a 2D credible-region match alongside the existing 3D one, and a row
identity that is not `superevent_id`. Worth deciding first whether these become rows in the
same frame or a parallel output.

---

## 2. Consume IceCube neutrino localizations in the localize stage

**Problem.** Both IceCube topics in `gcn_notices.TOPIC_ROUTING` are archived and unread.
They need different handling, which is why they are called out separately from issue 1.

**The two topics.**

- `icecube.gold_bronze_track_alerts` — a standalone neutrino event keyed on its own name
  (`IceCube-260425A`). Same two blockers as issue 1: new key space, and a synthesized map
  with no `DISTMU`.
- `icecube.lvk_nu_track_search` — reports neutrinos in the ±500 s window around an LVK
  superevent. `parse_icecube_lvk_nu_track_search` keys it on `ref_ID`, so it is **already
  filed under a superevent id** (`neutrino/icecube_lvk_nu_track_search/S230914ak/`). Each
  entry in `coincident_events` carries its own error circle labelled by `RUNID_EVENTID`.

**Note.** Because of that key, the second topic is join-shaped rather than row-shaped: it
attaches to a superevent row that already exists. The join mechanism itself is issue 3;
this issue covers what the pipeline does with the neutrino data once joined.

**Watch out.** The shared key space is a hazard, not a convenience. Selecting GCN events by
"the event id looks like a superevent" would pull a neutrino error circle in as a
superevent. The current allowlist (`category == "gw" and source == "lvk"`) exists to
prevent exactly that; there is a regression test on it.

---

## 3. Join non-GW GCN notices to their superevents

**Problem.** When a GW superevent has a Swift GUANO follow-up or a BOOM optical counterpart
in the store, the localize output does not say so. The multi-messenger context is on disk
and invisible.

**Approach.** `gcn_store.find_events(related_id=...)` (`gcn_store.py:609`) already does the
lookup — it matches case-insensitively and as a substring, so `S230914ak` also finds
`S230914ak-2-Preliminary`. Surface the matches as columns or a nested field on the
superevent rows the localize stage already produces.

**Scope.** Adds no new rows and needs no new key space or 2D matching, so it is independent
of issues 1 and 2 and considerably cheaper. Could land first.

---

## 4. Confirm whether the FAR cut already subsumes `significant`

**Problem.** `igwn.gwalert` notices carry `event.significant`, and the GCN read path
deliberately ignores it. The design assumes `far_threshold_per_year` already excludes
everything a significance cut would have, so adding a second cut would be a second place for
the same decision to live.

That assumption is untested. `[localize] far_threshold_per_year = 2.0` (`config.toml:76`), and
low-significance alerts are published at far higher FAR, so in the normal case the FAR cut
drops them first and the flag never matters. Nobody has confirmed that an alert cannot pass a
2/yr FAR cut while still being flagged `significant: false` — after an offline FAR revision,
say, or on a search whose significance criterion is not purely a FAR threshold.

**What to check.**

- Whether LVK's significance criterion is purely a FAR threshold, and whether it is applied
  per-pipeline or per-search. If it is strictly FAR-based and stricter than 2/yr, this closes
  with a comment anchored to the schema and no code change.
- Whether `significant` ever disagrees with our own FAR cut in the captured archive. Once
  `gcn_localizations/` holds a few weeks of real notices this is a one-liner over the store,
  and it answers the question with our own data rather than from the spec.

**If it can happen,** the cut belongs in `[localize]` config, not as a default in
`gracedb_tools`. `test_no_scientific_cut_carries_a_default` (`test_gracedb_tools.py:1507`)
pins that convention: the module states no scientific cut of its own, so a
`require_significant=True` default would be a second definition with nothing keeping it in
step with the config the pipeline actually runs from.

**Not the same as the MDC exclusion.** MDC is hardcoded with no knob, because it is not a
science cut but a "this is not real data" fact. Only `significant` is in question here.

---

## 5. Confirm which signal authoritatively identifies an MDC alert

**Problem.** The GCN read path excludes MDC alerts, to match the `category: Production` filter the
GraceDB listing query already applies (`gracedb_tools.py:478`). It currently excludes on **two**
signals, either sufficient: `event.search == "MDC"`, or a superevent id beginning with `M`.

That is a deliberate best guess, not a confirmed rule. The only payload we have carries both
signals at once (`gcn_examples.py:39,118` — `MS181101ab` with `"search": "MDC"`), so neither has
been shown sufficient on its own, and neither has been shown necessary. The belt-and-braces
version was chosen for its failure mode: over-excluding a real event shows up in the logged count
and is recoverable, while under-excluding puts mock data silently into science results.

**What to check.**

- Whether the `S` / `MS` / `TS` superevent-id prefix convention is documented anywhere
  authoritative, and whether it is guaranteed rather than customary. If it is, an allowlist
  (`id.startswith("S")`) is stronger than the current blocklist and is the likely end state —
  consistent with the allowlist reasoning used for the `gw`/`lvk` scope filter.
- Whether `search` takes values other than `MDC` that also indicate non-astrophysical data.
- Whether the two signals ever disagree in the captured archive. The sweep logs which rule fired
  for each exclusion precisely so this can be answered from our own data.

**Also worth settling here.** A store captured from `--domain test.gcn.nasa.gov` is
indistinguishable from a production one once written — `store_notice` records the topic but not
the domain (`gcn_listener.py:198`). The mitigation today is documentation: use a separate
`--store-root` when rehearsing. If the id-prefix rule turns out to be reliable it may cover this
too; if not, recording the domain on the history entry is the alternative, at the cost of every
already-captured entry reading `None`.

**Not the same as issue 4.** That one asks whether a *science* cut (`significant`) is redundant.
This one asks how to recognise data that is not science at all.

---

## Considered and dropped: renaming `gracedb_cache`

A rename to `superevent_cache` was discussed while the design still promoted GCN data
*into* the cache, which would have left the name describing only half its contents. The
read-time union does not write anything GCN-sourced there, so the cache holds exactly what
its name says: GraceDB's file listing, `p_astro`, and skymaps from the REST API.

`gracedb_cache` / `gcn_store` is also a useful pair as it stands — each name carries the
source and the storage kind, and the storage kind is the real distinction: a re-derivable
cache versus an unrecoverable archive. `superevent_cache` would drop the source, which is
what differs between the two.
