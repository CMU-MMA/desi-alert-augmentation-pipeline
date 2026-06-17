"""Helpers for the BOOM query "gold standard" regression test.

The gold standard is a CSV snapshot of the alerts returned for a *fixed*,
historical Julian-date window. Because the window is fixed in the past, BOOM
should return the same records every time, so a later run can be diffed against
the committed snapshot to catch regressions in the query/normalization code.

Run this file directly to (re)generate the snapshot::

    BOOM_USERNAME=... BOOM_PASSWORD=... python tests/desi_aap/gold_standard.py

This requires live BOOM credentials and network access, so it is a manual step
rather than something pytest does automatically.
"""

import io
from pathlib import Path

import pandas as pd
from desi_aap.boom import query_alerts

# Fixed historical window ported from old_code/LSST_For_COSMOS.ipynb. Keeping
# this in the past makes the query reproducible.
GOLD_START_JD = 2461187.1383197717
GOLD_END_JD = 2461194.1383197717

DATA_DIR = Path(__file__).parent / "data"
GOLD_CSV = DATA_DIR / "gold_standard_alerts.csv"


def fetch_gold_standard() -> pd.DataFrame:
    """Query BOOM for the fixed gold-standard window."""
    return query_alerts(
        start=GOLD_START_JD,
        end=GOLD_END_JD,
        survey="LSST",
        sort_by="candidate.jd",
        sort_order="Descending",
    )


def normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame comparable to the gold CSV.

    Round-trips through CSV so that nested columns (lists of dicts) and numeric
    precision are serialized identically to the on-disk snapshot, then sorts by
    ``_id`` for a stable row order.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame, either freshly queried or loaded from the gold CSV.

    Returns
    -------
    pandas.DataFrame
        A normalized copy with a stable row order and reset index.
    """
    roundtripped = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    sort_key = "_id" if "_id" in roundtripped.columns else roundtripped.columns[0]
    return roundtripped.sort_values(sort_key).reset_index(drop=True)


def generate(out: Path = GOLD_CSV) -> Path:
    """Fetch the gold-standard window from BOOM and write it to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_gold_standard()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} alerts to {out}")
    return out


if __name__ == "__main__":
    generate()
