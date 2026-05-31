"""
storage.py
----------
Parquet read/write helpers and filesystem utilities.

Usage
-----
    from taig.utils.storage import read_parquet, write_parquet, ensure_dir

    df = read_parquet("data/processed/conti_cleaned.parquet")
    write_parquet(df, "data/processed/conti_entities.parquet")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from taig.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if it does not exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def read_parquet(
    path: Path | str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame.

    Parameters
    ----------
    path:
        Path to the .parquet file.
    columns:
        If provided, only these columns are loaded (projection pushdown).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")

    df = pd.read_parquet(p, columns=list(columns) if columns else None)
    log.debug("Loaded %d rows from %s", len(df), p)
    return df


def write_parquet(
    df: pd.DataFrame,
    path: Path | str,
    compression: str = "snappy",
    overwrite: bool = True,
) -> Path:
    """Write a DataFrame to a Parquet file.

    Parameters
    ----------
    df:
        DataFrame to persist.
    path:
        Destination path. Parent directories are created automatically.
    compression:
        Parquet compression codec. "snappy" balances speed and size well.
    overwrite:
        If False and *path* exists, raises FileExistsError.

    Returns
    -------
    Path
        The resolved path that was written to.
    """
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite=False: {p}")

    ensure_dir(p.parent)
    df.to_parquet(p, compression=compression, index=False)
    log.info("Wrote %d rows to %s", len(df), p)
    return p


def list_parquet_files(directory: Path | str) -> list[Path]:
    """Return all .parquet files under *directory*, sorted by modification time."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")
    files = sorted(d.rglob("*.parquet"), key=lambda f: f.stat().st_mtime)
    return files


def parquet_info(path: Path | str) -> dict:
    """Return basic metadata for a Parquet file without loading all data.

    Returns a dict with keys: path, rows, columns, size_mb.
    """
    import pyarrow.parquet as pq

    p = Path(path)
    pf = pq.ParquetFile(p)
    meta = pf.metadata
    return {
        "path": str(p),
        "rows": meta.num_rows,
        "columns": [pf.schema_arrow.field(i).name for i in range(len(pf.schema_arrow))],
        "size_mb": round(p.stat().st_size / 1_048_576, 2),
        "row_groups": meta.num_row_groups,
    }
