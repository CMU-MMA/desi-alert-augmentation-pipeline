# desi-alert-augmentation-pipeline


[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/desi-aap?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/desi-aap/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/CMU-MMA/desi-alert-augmentation-pipeline/smoke-test.yml)](https://github.com/CMU-MMA/desi-alert-augmentation-pipeline/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/CMU-MMA/desi-alert-augmentation-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/CMU-MMA/desi-alert-augmentation-pipeline)

## Pipeline

`desi-aap` runs a sequence of stages, configured by TOML. Three stages exist
today:

| Stage | Does | Writes |
|---|---|---|
| `query` | Queries the [BOOM](https://api.kaboom.caltech.edu) broker for alerts in a time window (`desi_aap.boom`) | `query/alerts_<stamp>.parquet` |
| `crossmatch` | Wraps those alerts in an LSDB catalog and cross-matches them against the DESI spectroscopic catalogs (`desi_aap.stages.crossmatch`), keeping the alerts that matched | `crossmatch/matches_<stamp>.parquet` |
| `slack_publish` | Posts the matched alerts to a Slack channel (`desi_aap.stages.slack_publish`). Skips itself unless a `[slack]` section is configured — see [Slack publishing](#slack-publishing) | nothing |

### Install and run

```bash
pip install -e '.[dev]'

export BOOM_USERNAME=... BOOM_PASSWORD=...
desi-aap run --config config.toml

desi-aap stages          # list the stages, in the order they run
```

### Configuration

`config.toml` at the root of the repo is the main configuration file. The code
carries no defaults, so every value the pipeline uses is written there.

`output_dir` and the GraceDB `cache_dir` are relative, so a fresh clone works
anywhere with no setup. Both therefore follow the working directory: see
[Scheduled runs](#scheduled-runs) before putting this on a timer.

`--config` is repeatable and later files merge over earlier ones, table by
table. Write your own overlay for anything that depends on where you are
running, and layer it on:

```bash
desi-aap run -c config.toml -c my-overlay.toml
```

```toml
[run]
output_dir = "output"

[dask]
n_workers = 4
threads_per_worker = 1
memory_limit = "2GB"

[query.boom]
survey = "LSST"

[query.window]
lookback = "1h"

[crossmatch.catalogs.desi_dr1]
catalog = "/ocean/projects/phy250012p/shared/3DTS/DESI/dr1/desi_dr1_zcat"
radius_arcsec = 5.0
n_neighbors = 1

[crossmatch.catalogs.desi_dr2]
catalog = "/ocean/projects/phy250012p/shared/3DTS/DESI/dr2/desi_dr2_zcat"
radius_arcsec = 5.0
n_neighbors = 1

[gracedb]
cache_dir = "gracedb_cache"
recheck_window = "30d"

[slack]
credentials = "~/.config/desi_aap/slack.toml"
channel = "#desi-alerts"
max_rows = 20
```

Everything else is a property of one invocation rather than of the pipeline's
settings, so it stays on the command line:

| Flag | Does                                                                                                                                           |
|---|------------------------------------------------------------------------------------------------------------------------------------------------|
| `--config` / `-c` | The config file. Required, and repeatable: later files are merged over earlier ones. |
| `--dry-run` | Does all the work but writes no results to disk.                                                                                               |                                                                                                                                                   |
| `--verbose` / `-v` | Logs at DEBUG instead of INFO. Only for this package's own logger, so dependencies stay quiet.                                                 |
| `--log-file` | Writes the log here instead of `<output_dir>/logs/<stamp>.log`.                                                                                |
| `--from-stage` / `--input` | Start at a later stage, with a parquet file standing in for the previous stage's output. Useful for re-running the tail of the pipeline, or testing one stage on known input. |

```bash
desi-aap run --config config.toml --config backfill.toml
desi-aap run --config config.toml --dry-run -v
desi-aap run --config config.toml --log-file /var/log/desi_aap.log
```

A dry run still writes its log - it is the point of the exercise:

```
BOOM returned 327 alerts.
Dry run: not writing the alerts.
Crossmatch summary: {'n_alerts': 327, 'n_matches_desi_dr1': 8, 'n_alerts_matched': 8}
Dry run: not writing the matches.
```

### Output

The pipeline writes intermediate results, per stage, in parquet. Each stage
gets its own subdirectory of `run.output_dir`, and one run leaves three files:

```
/output/
├── query/
│   └── alerts_20260807T182718Z.parquet     # every alert BOOM returned
├── crossmatch/
│   └── matches_20260807T182718Z.parquet    # the alerts that matched a catalog
└── logs/
    └── 20260807T182718Z.log                # what the run did
```

One timestamp, taken when the run starts, names all three - so a run's files
group together, and the log sits beside the results it describes.

### Scheduled runs

`output_dir` and `[gracedb] cache_dir` are both relative, so they resolve
against the working directory. cron does not inherit yours - it starts in
`$HOME` - so a scheduled run must `cd` into the checkout first, or it will
write its results somewhere else and rebuild the GraceDB cache from scratch
every time. Neither failure raises anything; you just get a second cache and
scattered output.

So `cd` in the crontab entry rather than relying on the environment:

```cron
# Hourly through the night, from the checkout so the relative paths land in it.
0 20-23,0-6 * * *  cd /path/to/desi-alert-augmentation-pipeline && \
                   desi-aap run -c config.toml >> cron.log 2>&1
```

If you would rather not depend on that, give absolute paths in an overlay:

```toml
# my-overlay.toml
[run]
output_dir = "/ocean/projects/phy250012p/shared/3DTS/output"

[gracedb]
cache_dir = "/ocean/projects/phy250012p/shared/3DTS/gracedb_cache"
```

### Slack publishing

The `slack_publish` stage posts each run's candidates to a channel: a header
naming the run, how many candidates it found, the first `max_rows` of them
with their coordinates and per-catalog match counts, and the path to the
full parquet output. The
pipeline stops before this stage when an earlier one produces no rows, so a
run with nothing to report posts nothing.

The `[slack]` section is optional — without it the stage logs that it is
skipping, so a fresh clone runs with no Slack setup. To turn it on:

1. Create a Slack app at https://api.slack.com/apps ("From scratch" is fine —
   the only scope it needs is `chat:write`, under **OAuth & Permissions** >
   **Bot Token Scopes**).
2. Install the app to the workspace on the same page and copy the
   **Bot User OAuth Token** (it starts with `xoxb-`).
3. Put the token in a TOML file *outside the repository*, and make it
   readable only by you:

   ```bash
   mkdir -p ~/.config/desi_aap
   echo 'bot_token = "xoxb-your-token-here"' > ~/.config/desi_aap/slack.toml
   chmod 600 ~/.config/desi_aap/slack.toml
   ```

4. Invite the bot to the target channel (`/invite @<bot name>` in the
   channel) and fill in the `[slack]` section: `credentials` is the path from
   step 3, `channel` is where it posts, and `max_rows` is how many candidates
   the message lists before cutting off.

`--dry-run` builds the message and logs it instead of posting, which is the
way to preview the formatting before pointing it at a real channel.

To exercise just this stage on known input, start the pipeline at it. Any
previous run's matches file works, or build one from the committed test data
(`python scripts/make_test_matches.py test_matches.parquet`, with `--rows` to
tile it bigger):

```bash
# Preview the message this file would produce, without posting:
desi-aap run -c config.toml --from-stage slack_publish \
             --input output/crossmatch/matches_<stamp>.parquet --dry-run

# Post it for real, once [slack] is configured:
desi-aap run -c config.toml --from-stage slack_publish \
             --input output/crossmatch/matches_<stamp>.parquet
```

## GraceDB

`desi_aap.gracedb_tools` matches supernovae from TNS against public LIGO/Virgo
superevents. It is not a pipeline stage yet — it is driven from
`docs/pre_executed/gracedb_sesn_refactor.ipynb` — so `desi-aap run` does not
touch it, and the `[gracedb]` section only takes effect for code that loads the
config and passes the cache in:

```python
from desi_aap.config import load_config
from desi_aap.gracedb_tools import fetch_gracedb_superevents

cfg = load_config("config.toml")
events = fetch_gracedb_superevents(se_types=["BNS", "NSBH"], cache=cfg.gracedb.to_cache())
```

`cache` is required and has no default anywhere in the call chain, so where the
cache lives is always an explicit decision.

### Why it is cached

A scan does one paginated `superevents()` query, then two requests per
superevent behind it. Against the live API with the default FAR threshold that
is 349 superevents and roughly 700 requests, about two minutes. Run hourly
overnight, uncached, that is some 7,000 requests a night at a public service to
rebuild a table that changes by a few rows.

So the listing is always fetched live — it is a handful of requests, and it is
the signal every freshness decision is made from, which means a retracted or
backfilled superevent is noticed with no way for the cache to drift. Everything
behind it is served from disk unless something says otherwise.

### What invalidates an entry

GraceDB's superevent payload carries no modification timestamp, so an entry is
re-fetched when either:

- **the fingerprint moved** — the labels, preferred event, FAR or `t_0` that the
  listing reports differ from what was stored; or
- **the event is recent** — younger than `recheck_window`, because a file can be
  uploaded without moving any field the listing reports.

A skymap is re-downloaded when GraceDB lists a higher `,N` revision than the one
recorded. That matters because the unversioned name is repointed at each new
revision, so a copy taken beforehand is stale while still matching by name.

Past the recheck window, a superevent whose fingerprint has not moved is trusted
and not re-listed at all. A quietly reissued skymap on a settled event is
therefore not noticed — that is the trade the window makes. After a bulk
reprocessing campaign, pass `force_refresh=True` once.

The `cache_status` column reports what happened per superevent: `hit`, `miss`,
`stale_fingerprint`, `stale_age` or `forced`.

### Layout, and clearing it

```
<cache_dir>/
├── superevents/S190425z.json                     # listing, p_astro, skymap revision
└── skymaps/S190425z__bilby.multiorder.fits
```

Entries record the skymap path relative to the cache root, so the whole
directory can be moved between a laptop, `$HOME` and a project filesystem
without invalidating it. Every write goes through a temporary file and a rename,
so a run killed partway through leaves no half-written file for the next one to
trust.

To force one superevent's metadata to be re-read, delete its
`superevents/<id>.json`. That does *not* replace its skymap — nothing in a
re-read listing says the local copy is damaged rather than merely old — so to
repair bytes that are actually wrong, pass `force_refresh=True`, or delete the
skymap file as well. To start over, delete the directory: all of it is
re-derivable.
