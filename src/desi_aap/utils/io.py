"""Reading and writing the parquet files stages hand to each other."""

from pathlib import Path

import nested_pandas as npd


def write_frame(frame: npd.NestedFrame, path: Path) -> Path:
    """Write a frame to parquet, preserving its Arrow dtypes.

    Parameters
    ----------
    frame : nested_pandas.NestedFrame
        The table to write.
    path : Path
        Destination file. Parent directories are created.

    Returns
    -------
    Path
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


def read_frame(path: Path | str) -> npd.NestedFrame:
    """Read a frame back, keeping the Arrow-backed dtypes it was written with.

    Parameters
    ----------
    path : Path or str
        A parquet file written by :func:`write_frame`.

    Returns
    -------
    nested_pandas.NestedFrame
        The stored table.
    """
    return npd.read_parquet(path)
