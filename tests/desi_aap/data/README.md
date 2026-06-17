# Test data

`gold_standard_alerts.csv` is a snapshot of the BOOM alerts returned for a
fixed historical Julian-date window (see `GOLD_START_JD` / `GOLD_END_JD` in
`../gold_standard.py`). `test_boom.py::test_gold_standard` re-queries that same
window and diffs the result against this file.

To (re)generate the snapshot you need live BOOM credentials and network access:

```bash
BOOM_USERNAME=<your-username> BOOM_PASSWORD=<your-password> \
    python tests/desi_aap/gold_standard.py
```

Then commit the regenerated `gold_standard_alerts.csv`.
