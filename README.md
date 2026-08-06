# desi-alert-augmentation-pipeline


[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/desi_aap?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/{{desi_aap}}/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/CMU-MMA/desi-alert-augmentation-pipeline/smoke-test.yml)](https://github.com/CMU-MMA/desi-alert-augmentation-pipeline/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/CMU-MMA/desi-alert-augmentation-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/CMU-MMA/desi-alert-augmentation-pipeline)

## Live GCN localization maps

`desi_aap.gracedb_tools` fetches GW skymaps from GraceDB after the fact. The GCN listener
gets them, and the GRB and neutrino localizations, as they are published.

```bash
export GCN_CLIENT_ID=...
export GCN_CLIENT_SECRET=...

# Run as a daemon.
python scripts/listen_gcn.py --store-root /path/to/gcn_localizations

# Or drain whatever is buffered and exit, for a scheduled catch-up run.
python scripts/listen_gcn.py --once
```

Credentials come from a client created at <https://gcn.nasa.gov/quickstart> and are read from
the environment, never stored in the repository; in CI they come from repository secrets of
the same names. The test suite does not need them, because every listener test drives the loop
through a fake consumer rather than a connection. Pass `--domain test.gcn.nasa.gov` to rehearse
against GCN's synthetic events first.

Subscribed topics, and where each lands in the store:

| Topic | Category | Localization as published |
| --- | --- | --- |
| `igwn.gwalert` | `gw/lvk` | base64 multi-order FITS skymap |
| `gcn.notices.swift.bat.guano` | `grb/swift_bat_guano` | none, then an inline map or an error circle |
| `gcn.notices.einstein_probe.wxt.alert` | `grb/einstein_probe_wxt` | ~arcminute error circle |
| `gcn.notices.icecube.gold_bronze_track_alerts` | `neutrino/icecube_gold_bronze` | error circle, plus a hosted map once revised |
| `gcn.notices.icecube.lvk_nu_track_search` | `neutrino/icecube_lvk_nu_track_search` | one error circle per coincident neutrino |
| `gcn.notices.boom.alert` | `optical/boom` | arcsecond optical positions, no error region |

### Store layout

```
<root>/<category>/<source>/<event_id>/
    notices/<stem>.json                        the notice, verbatim
    skymaps/<stem>.multiorder.fits             real map the mission supplied
    skymaps/<stem>.synthetic.multiorder.fits   map synthesized from a quoted error region
    history.jsonl                              one line per notice received
    latest.json                                pointer to the current best notice and map
    latest.fits                                symlink to the current best map
<root>/index.jsonl                             one line per notice, across all events
<root>/_quarantine/                            payloads that failed to parse or store
```

Every version of an event is kept: a notice sequence records what was known when, and the
early-warning map a follow-up was triggered on is not recoverable from the update that
replaced it. `latest.json` is recomputed from the full history on each write, so an update
arriving before the preliminary it supersedes still ends up as latest, and a retraction
withdraws the map rather than leaving a stale one current.

Every stored map reads with `ligo.skymap.io.read_sky_map(path, moc=True)`, whatever its
origin, so `ligo.skymap.postprocess.crossmatch` treats the whole store uniformly.

### Synthesized maps are models, not measurements

Most GRB and neutrino notices publish a position and an error radius rather than a map.
`desi_aap.gcn_skymaps` prefers a real map whenever the mission supplied one — inline, or by
URL — and only otherwise synthesizes a multi-order HEALPix map, spreading probability as a 2D
Gaussian calibrated so the quoted radius encloses the quoted containment probability (0.9
when the notice omits it, per the GCN core schema).

Such a file is marked three ways: the `.synthetic.multiorder.fits` suffix, the FITS `CREATOR`
card, and `"source": "synthesized"` in the store index. Treat it accordingly. IceCube in
particular warns that its circularized error is only an approximation of a region that is
often irregular or multi-modal, which is why the revised alert's hosted map is preferred.
A position with no quoted error — BOOM's optical targets — gets no map at all rather than an
invented one.

Two limits worth knowing. Synthesized maps are 2D only, so the 3D volume crossmatch in
`gracedb_tools` does not apply to them. And GCN's buffers hold only the past few days, so a
listener down longer than that leaves a gap: backfill GW events through GraceDB, and the rest
through GCN's notices archive.

## TNS credentials

Downloading the TNS public object catalog requires a registered TNS bot. The credentials
are read from the environment, never stored in the repository:

```bash
export TNS_API_KEY=...
export TNS_BOT_ID=...
export TNS_BOT_NAME=...
```

Set these before calling `desi_aap.tns_catalog.download_tns_table`, including in the kernel
running `docs/notebooks/gracedb_sesn_refactor.ipynb`. Without them the call raises a
`RuntimeError` naming the missing variables; nothing else in the package needs them, so the
rest of the pipeline and the test suite run fine unset.

In CI they come from repository secrets of the same names.
