# Test data

`gold_standard_alerts.parquet` is a snapshot of the BOOM alerts returned for a
fixed historical Julian-date window (see `GOLD_START_JD` / `GOLD_END_JD` in
`../gold_standard.py`). `test_boom.py::test_gold_standard` re-queries that same
window and diffs the result against this file.

It is stored as parquet rather than CSV so that the `cross_matches.LSPSC` list
of structs stays a real nested column (`lspsc`) instead of a stringified Python
literal. Read it with `nested_pandas.read_parquet` (or
`gold_standard.load_gold_standard()`).

To (re)generate the snapshot you need live BOOM credentials and network access:

```bash
BOOM_USERNAME=<your-username> BOOM_PASSWORD=<your-password> \
    python tests/desi_aap/gold_standard.py
```

Then commit the regenerated `gold_standard_alerts.parquet`.
