"""
Portable Parquet write/repack for cross-platform (WSL <-> Windows) compatibility.
Uses conservative settings to avoid "Repetition level histogram size mismatch" on Windows.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


def write_portable_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Write a DataFrame to Parquet with conservative, Windows-friendly settings.
    Converts to PyArrow Table with preserve_index=False, then write_table with:
    compression=snappy, use_dictionary=False, write_statistics=False,
    version=1.0. Writes atomically (temp file then replace).
    """
    path = Path(path)
    if pq is None or pa is None:
        raise ImportError("pyarrow is required for write_portable_parquet")
    # Ensure no index in the table
    table = pa.Table.from_pandas(df, preserve_index=False)
    kwargs = {
        "compression": "snappy",
        "use_dictionary": False,
        "write_statistics": False,
        "version": "1.0",
    }
    if "coerce_timestamps" in _get_write_table_params():
        kwargs["coerce_timestamps"] = "ms"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=path.parent, prefix=".portable_")
    try:
        os.close(fd)
        pq.write_table(table, tmp_path, **kwargs)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def _get_write_table_params() -> set:
    """Return set of parameter names accepted by pq.write_table (best-effort)."""
    try:
        import inspect
        return set(inspect.signature(pq.write_table).parameters)
    except Exception:
        return set()


def repack_parquet_file(src_path: Path, dst_path: Path, csv_path: Optional[Path] = None) -> None:
    """
    Read Parquet at src_path (pandas/pyarrow), write portable Parquet to dst_path
    and optionally CSV to csv_path (same stem as dst_path with .csv if not given).
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")
    df = pd.read_parquet(src_path)
    write_portable_parquet(df, dst_path)
    out_csv = csv_path if csv_path is not None else dst_path.with_suffix(".csv")
    df.to_csv(out_csv, index=False)
    logger.info("Repacked %s -> %s and %s", src_path.name, dst_path.name, out_csv.name)
