# desi-alert-augmentation-pipeline


[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/desi_aap?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/{{desi_aap}}/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/CMU-MMA/desi-alert-augmentation-pipeline/smoke-test.yml)](https://github.com/CMU-MMA/desi-alert-augmentation-pipeline/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/CMU-MMA/desi-alert-augmentation-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/CMU-MMA/desi-alert-augmentation-pipeline)

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
