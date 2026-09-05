"""
Phase 13 v3 — Utilities: FDR, memory, timing, binning.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def bh_fdr(p_values: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Benjamini-Hochberg FDR; returns q-values (same length as p_values)."""
    p = np.asarray(p_values, dtype=float).ravel()
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.zeros(n)
    q[order] = np.minimum(1.0, np.minimum.accumulate((n / np.arange(1, n + 1)) * p[order]))
    return q


def get_memory_mb() -> Optional[float]:
    """Current process memory in MB if available."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def get_git_commit(root: Optional[Path] = None) -> Optional[str]:
    root = root or Path(".").resolve()
    if not (root / ".git").exists():
        root = root.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to a bin 0..n_bins-1 by quantile. Ties get same bin."""
    x = np.asarray(x, dtype=float).ravel()
    nan_mask = np.isnan(x)
    out = np.full(len(x), -1, dtype=np.int32)
    if np.all(nan_mask):
        return out
    valid = x[~nan_mask]
    if len(valid) == 0:
        return out
    q = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.percentile(valid, q * 100)
    edges = np.unique(edges)
    if len(edges) < 2:
        out[~nan_mask] = 0
        return out
    idx = np.searchsorted(edges, valid, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    out[~nan_mask] = idx
    return out


def run_manifest_dict(
    output_dir: Path,
    config: Any,
    run_id: str,
) -> Dict[str, Any]:
    """Build run_manifest.json content."""
    return {
        "run_id": run_id,
        "timestamp": _ts(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "git_commit": get_git_commit(output_dir.parent.parent.parent),
        "memory_mb": get_memory_mb(),
    }


class Timer:
    """Simple context manager for elapsed time."""

    def __init__(self, name: str = "", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start
        if self.verbose and self.name:
            mem = get_memory_mb()
            mem_s = f" | mem={mem:.0f} MB" if mem is not None else ""
            print(f"  [{_ts()}] END {self.name} — elapsed={self.elapsed:.1f}s{mem_s}")


def safe_divide(num: np.ndarray, denom: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Element-wise division, fill where denom is 0 or nan."""
    denom = np.asarray(denom, dtype=float)
    num = np.asarray(num, dtype=float)
    out = np.full_like(num, fill, dtype=float)
    ok = (denom != 0) & ~np.isnan(denom)
    out[ok] = num[ok] / denom[ok]
    return out


def stratum_did_means(
    y: np.ndarray,
    group_both: np.ndarray,
    group_a_only: np.ndarray,
    group_b_only: np.ndarray,
    group_none: np.ndarray,
    stratum_ids: np.ndarray,
    n_strata: int,
    min_group: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each stratum, compute counts and means for the four groups.
    y: [n_recipes], group_*: boolean masks [n_recipes], stratum_ids: [n_recipes] int 0..n_strata-1.
    Returns:
        n11, n10, n01, n00: [n_strata] counts
        m11, m10, m01, m00: [n_strata] means (nan if count < min_group)
    """
    n11 = np.bincount(stratum_ids[group_both], minlength=n_strata)
    n10 = np.bincount(stratum_ids[group_a_only], minlength=n_strata)
    n01 = np.bincount(stratum_ids[group_b_only], minlength=n_strata)
    n00 = np.bincount(stratum_ids[group_none], minlength=n_strata)

    # Sum of y per stratum per group (using float weights in bincount)
    # y * group gives value where group else 0
    s11 = np.bincount(stratum_ids[group_both], weights=y[group_both], minlength=n_strata)
    s10 = np.bincount(stratum_ids[group_a_only], weights=y[group_a_only], minlength=n_strata)
    s01 = np.bincount(stratum_ids[group_b_only], weights=y[group_b_only], minlength=n_strata)
    s00 = np.bincount(stratum_ids[group_none], weights=y[group_none], minlength=n_strata)

    m11 = safe_divide(s11, n11, np.nan)
    m10 = safe_divide(s10, n10, np.nan)
    m01 = safe_divide(s01, n01, np.nan)
    m00 = safe_divide(s00, n00, np.nan)

    # Mask strata where any group has count < min_group
    use = (n11 >= min_group) & (n10 >= min_group) & (n01 >= min_group) & (n00 >= min_group)
    m11 = np.where(use, m11, np.nan)
    m10 = np.where(use, m10, np.nan)
    m01 = np.where(use, m01, np.nan)
    m00 = np.where(use, m00, np.nan)

    return n11, n10, n01, n00, m11, m10, m01, m00
