# desi-alert-augmentation-pipeline


[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/desi_aap?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/{{desi_aap}}/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/CMU-MMA/desi-alert-augmentation-pipeline/smoke-test.yml)](https://github.com/CMU-MMA/desi-alert-augmentation-pipeline/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/CMU-MMA/desi-alert-augmentation-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/CMU-MMA/desi-alert-augmentation-pipeline)

## Pipeline

`desi-aap` runs a sequence of stages, configured by TOML. Two stages exist
today:

| Stage | Does | Writes |
|---|---|---|
| `query` | Queries the [BOOM](https://api.kaboom.caltech.edu) broker for alerts in a time window (`desi_aap.boom`) | `query/alerts_<stamp>.parquet` |
| `crossmatch` | Wraps those alerts in an LSDB catalog and cross-matches them against the DESI spectroscopic catalogs (`desi_aap.stages.crossmatch`), keeping the alerts that matched | `crossmatch/matches_<stamp>.parquet` |

### Install and run

```bash
pip install -e '.[dev]'

export BOOM_USERNAME=... BOOM_PASSWORD=...
desi-aap run --config config.toml

desi-aap stages          # list the stages, in the order they run
```

### Configuration

`config.toml` at the root of the repo is the only configuration file. The code
carries no defaults, so every value the pipeline uses is written there.


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
```

Everything else is a property of one invocation rather than of the pipeline's
settings, so it stays on the command line:

| Flag | Does                                                                                                                                           |
|---|------------------------------------------------------------------------------------------------------------------------------------------------|
| `--config` / `-c` | The config file. Required, and repeatable: later files are merged over earlier ones. |
| `--dry-run` | Does all the work but writes no results to disk.                                                                                               |                                                                                                                                                   |
| `--verbose` / `-v` | Logs at DEBUG instead of INFO. Only for this package's own logger, so dependencies stay quiet.                                                 |
| `--log-file` | Writes the log here instead of `<output_dir>/logs/<stamp>.log`.                                                                                |

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
