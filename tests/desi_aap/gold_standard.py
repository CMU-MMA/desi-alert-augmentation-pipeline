"""Helpers for the BOOM query "gold standard" regression test.

The gold standard is a parquet snapshot of the alerts returned for a *fixed*,
historical Julian-date window. Because the window is fixed in the past, BOOM
should return the same records every time, so a later run can be diffed against
the committed snapshot to catch regressions in the query/normalization code.

Parquet (rather than CSV) keeps the nested ``lspsc`` column -- the
``cross_matches.LSPSC`` list of structs -- as real nested data instead of
stringified Python literals.

Run this file directly to (re)generate the snapshot::

    BOOM_USERNAME=... BOOM_PASSWORD=... python tests/desi_aap/gold_standard.py

This requires live BOOM credentials and network access, so it is a manual step
rather than something pytest does automatically.
"""

import io
from pathlib import Path

import nested_pandas as npd
import pandas as pd

from desi_aap.boom import query_alerts

# Fixed historical window ported from old_code/LSST_For_COSMOS.ipynb. Keeping
# this in the past makes the query reproducible.
GOLD_START_JD = 2461187.1383197717
GOLD_END_JD = 2461194.1383197717

DATA_DIR = Path(__file__).parent / "data"
GOLD_PARQUET = DATA_DIR / "gold_standard_alerts.parquet"


def fetch_gold_standard() -> npd.NestedFrame:
    """Query BOOM for the fixed gold-standard window."""
    return query_alerts(
        start=GOLD_START_JD,
        end=GOLD_END_JD,
        survey="LSST",
        sort_by="candidate.jd",
        sort_order="Descending",
    )


def load_gold_standard(path: Path = GOLD_PARQUET) -> npd.NestedFrame:
    """Read the committed gold-standard snapshot."""
    return npd.read_parquet(path)


def normalize_for_compare(alerts: npd.NestedFrame) -> npd.NestedFrame:
    """Make a NestedFrame comparable to the gold parquet file.

    Round-trips through an in-memory parquet file so that dtypes match the
    on-disk snapshot (reading parquet yields pyarrow-backed columns), then sorts
    by ``_id`` for a stable row order.

    Parameters
    ----------
    alerts : nested_pandas.NestedFrame
        A frame, either freshly queried or loaded from the gold parquet file.

    Returns
    -------
    nested_pandas.NestedFrame
        A normalized copy with a stable row order and reset index.
    """
    buffer = io.BytesIO()
    alerts.to_parquet(buffer)
    buffer.seek(0)
    roundtripped = npd.read_parquet(buffer)
    sort_key = "_id" if "_id" in roundtripped.columns else roundtripped.columns[0]
    return roundtripped.sort_values(sort_key).reset_index(drop=True)


def assert_alerts_equal(fresh: npd.NestedFrame, gold: npd.NestedFrame) -> None:
    """Assert two normalized alert frames are equal, nested columns included.

    ``pandas.testing.assert_frame_equal`` cannot diff a nested column (it tries
    to compare the per-row sub-frames as scalars), so base columns and nested
    columns are checked separately: the nested ones via their flattened form,
    plus a null-mask check to separate a missing sub-frame from an empty one.

    Parameters
    ----------
    fresh, gold : nested_pandas.NestedFrame
        Frames as returned by :func:`normalize_for_compare`.
    """
    fresh_nested = sorted(fresh.nested_columns)
    gold_nested = sorted(gold.nested_columns)
    nested_columns = sorted(set(fresh_nested) | set(gold_nested))
    pd.testing.assert_frame_equal(
        fresh.drop(columns=nested_columns, errors="ignore"),
        gold.drop(columns=nested_columns, errors="ignore"),
        check_dtype=False,
        check_like=True,
        rtol=1e-6,
    )
    assert fresh_nested == gold_nested, f"nested columns differ: {fresh_nested} != {gold_nested}"
    for column in nested_columns:
        pd.testing.assert_series_equal(
            fresh[column].isna(),
            gold[column].isna(),
            check_dtype=False,
            obj=f"{column} null mask",
        )
        pd.testing.assert_frame_equal(
            fresh[column].nest.to_flat(),
            gold[column].nest.to_flat(),
            check_dtype=False,
            check_like=True,
            rtol=1e-6,
            obj=f"{column} (flattened)",
        )


def generate(out: Path = GOLD_PARQUET) -> Path:
    """Fetch the gold-standard window from BOOM and write it to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    alerts = fetch_gold_standard()
    alerts.to_parquet(out)
    print(f"Wrote {len(alerts)} alerts to {out}")
    return out


if __name__ == "__main__":
    generate()
