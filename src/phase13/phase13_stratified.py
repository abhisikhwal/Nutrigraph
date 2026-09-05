"""
Phase 13 v3 — Stratified difference-in-differences engine (sharded, resume, ETA).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .phase13_config import Phase13Config
from .phase13_utils import (
    Timer,
    _ts,
    get_memory_mb,
    quantile_bins,
    safe_divide,
)


def _stratum_means_and_vars(
    y: np.ndarray,
    group_mask: np.ndarray,
    stratum_ids: np.ndarray,
    n_strata: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum and sum-of-squares of y per stratum for group_mask. Returns mean, var (nan where n<2)."""
    # stratum_ids: [n_recipes], group_mask: bool [n_recipes], y: [n_recipes]
    idx = np.where(group_mask)[0]
    if len(idx) == 0:
        return np.full(n_strata, np.nan), np.full(n_strata, np.nan)
    s = stratum_ids[idx]
    yy = y[idx]
    n_s = np.bincount(s, minlength=n_strata)
    sum_s = np.bincount(s, weights=yy, minlength=n_strata)
    sum2_s = np.bincount(s, weights=yy * yy, minlength=n_strata)
    mean_s = safe_divide(sum_s, n_s, np.nan)
    # var = E[X^2] - E[X]^2, sample var = n/(n-1) * (sum2/n - (sum/n)^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_s = np.where(n_s >= 2, n_s / (n_s - 1) * (safe_divide(sum2_s, n_s, 0) - mean_s * mean_s), np.nan)
    return mean_s, var_s


def compute_pair_delta_stratified(
    recipe_idx_both: np.ndarray,
    recipe_idx_a_only: np.ndarray,
    recipe_idx_b_only: np.ndarray,
    recipe_idx_none: np.ndarray,
    stratum_ids: np.ndarray,
    n_strata: int,
    y_matrix: np.ndarray,
    min_group: int,
    weighting: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    For each category (column of y_matrix), compute stratified DiD delta, SE, and p-value.
    Returns: delta [n_cat], se [n_cat], p [n_cat], n_strata_used.
    """
    n_cat = y_matrix.shape[1]
    n_recipes = len(stratum_ids)
    # Boolean masks for recipe indices
    both_mask = np.zeros(n_recipes, dtype=bool)
    both_mask[recipe_idx_both] = True
    a_only_mask = np.zeros(n_recipes, dtype=bool)
    a_only_mask[recipe_idx_a_only] = True
    b_only_mask = np.zeros(n_recipes, dtype=bool)
    b_only_mask[recipe_idx_b_only] = True
    none_mask = np.zeros(n_recipes, dtype=bool)
    none_mask[recipe_idx_none] = True

    n11 = np.bincount(stratum_ids[both_mask], minlength=n_strata)
    n10 = np.bincount(stratum_ids[a_only_mask], minlength=n_strata)
    n01 = np.bincount(stratum_ids[b_only_mask], minlength=n_strata)
    n00 = np.bincount(stratum_ids[none_mask], minlength=n_strata)
    use_stratum = (n11 >= min_group) & (n10 >= min_group) & (n01 >= min_group) & (n00 >= min_group)
    n_strata_used = int(np.sum(use_stratum))

    delta_per_cat = np.full(n_cat, np.nan, dtype=np.float64)
    se_per_cat = np.full(n_cat, np.nan, dtype=np.float64)
    p_per_cat = np.full(n_cat, np.nan, dtype=np.float64)

    for c in range(n_cat):
        y = y_matrix[:, c].astype(np.float64)
        m11, v11 = _stratum_means_and_vars(y, both_mask, stratum_ids, n_strata)
        m10, v10 = _stratum_means_and_vars(y, a_only_mask, stratum_ids, n_strata)
        m01, v01 = _stratum_means_and_vars(y, b_only_mask, stratum_ids, n_strata)
        m00, v00 = _stratum_means_and_vars(y, none_mask, stratum_ids, n_strata)

        delta_s = m11 - m10 - m01 + m00
        # var(Delta_s) = var(y11)/n11 + var(y10)/n10 + var(y01)/n01 + var(y00)/n00
        with np.errstate(divide="ignore", invalid="ignore"):
            var_delta_s = safe_divide(v11, n11, np.nan) + safe_divide(v10, n10, np.nan) + safe_divide(v01, n01, np.nan) + safe_divide(v00, n00, np.nan)
        var_delta_s = np.where(use_stratum, var_delta_s, np.nan)

        if weighting == "size":
            w_s = (n11 + n10 + n01 + n00).astype(float)
        else:
            # inv_var
            w_s = safe_divide(1.0, var_delta_s, 0.0)
        w_s = np.where(use_stratum, w_s, 0.0)
        total_w = np.sum(w_s)
        if total_w <= 0:
            continue
        delta = np.nansum(w_s * delta_s) / total_w
        # var(Delta) = sum w_s^2 var(Delta_s) / (sum w_s)^2
        var_delta = np.nansum(w_s * w_s * var_delta_s) / (total_w * total_w)
        if var_delta <= 0 or np.isnan(var_delta):
            continue
        se = np.sqrt(var_delta)
        delta_per_cat[c] = delta
        se_per_cat[c] = se
        if se > 0:
            z = delta / se
            p_per_cat[c] = 2 * (1 - _norm_cdf(np.abs(z)))
        else:
            p_per_cat[c] = 1.0

    return delta_per_cat, se_per_cat, p_per_cat, n_strata_used


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from scipy import stats
    return stats.norm.cdf(x)


def compute_pair_interactions_sharded(
    config: Phase13Config,
    y_cols: List[str],
    ingredient_to_recipe_idx: Dict[str, np.ndarray],
    stratum_ids: np.ndarray,
    recipe_y_matrix: np.ndarray,
    candidate_pairs: List[Tuple[str, str]],
    output_dir: Path,
    *,
    progress_callback: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Run stratified DiD for each pair in shards. Skip shard if file exists and has >0 rows.
    Returns concatenated raw results.
    """
    n_strata = int(np.max(stratum_ids)) + 1
    n_recipes = recipe_y_matrix.shape[0]
    all_recipe_idx = np.arange(n_recipes, dtype=np.int64)
    min_group = getattr(config, "min_group_per_stratum", 10)
    weighting = config.weighting
    shard_size = config.shard_size_pairs
    total_pairs = len(candidate_pairs)
    n_shards = (total_pairs + shard_size - 1) // shard_size
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output_dir / "heartbeat.json"

    rows: List[Dict[str, Any]] = []
    start_global = time.perf_counter()
    pairs_done = 0

    for shard_idx in range(n_shards):
        start_shard = (shard_idx * shard_size)
        end_shard = min(start_shard + shard_size, total_pairs)
        pair_slice = candidate_pairs[start_shard:end_shard]
        shard_path = shards_dir / f"interactions_shard_{shard_idx + 1:04d}.parquet"

        # Resume: skip only if exists and non-empty
        skip_shard = False
        if shard_path.exists():
            try:
                df_ex = pd.read_parquet(shard_path)
                if len(df_ex) > 0:
                    skip_shard = True
                    rows.extend(df_ex.to_dict("records"))
                    pairs_done += len(pair_slice)
                    if progress_callback:
                        progress_callback(f"Skip shard {shard_idx + 1}/{n_shards} (rows={len(df_ex)})")
            except Exception:
                pass

        if skip_shard:
            continue

        shard_rows: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        for i, (ing_a, ing_b) in enumerate(pair_slice):
            rec_a = ingredient_to_recipe_idx.get(ing_a)
            rec_b = ingredient_to_recipe_idx.get(ing_b)
            if rec_a is None or rec_b is None:
                continue
            rec_a = np.unique(rec_a)
            rec_b = np.unique(rec_b)
            both_idx = np.intersect1d(rec_a, rec_b)
            a_only_idx = np.setdiff1d(rec_a, rec_b)
            b_only_idx = np.setdiff1d(rec_b, rec_a)
            union_ab = np.union1d(rec_a, rec_b)
            none_idx = np.setdiff1d(all_recipe_idx, union_ab)

            n_both = len(both_idx)
            n_a_only = len(a_only_idx)
            n_b_only = len(b_only_idx)
            n_none = len(none_idx)

            delta_arr, se_arr, p_arr, n_strata_used = compute_pair_delta_stratified(
                both_idx, a_only_idx, b_only_idx, none_idx,
                stratum_ids, n_strata, recipe_y_matrix, min_group, weighting,
            )

            for c, cat in enumerate(y_cols):
                if np.isnan(delta_arr[c]) or np.isnan(p_arr[c]):
                    continue
                shard_rows.append({
                    "ingA_id": ing_a,
                    "ingB_id": ing_b,
                    "category": cat,
                    "delta": float(delta_arr[c]),
                    "se": float(se_arr[c]),
                    "z": float(delta_arr[c] / se_arr[c]) if se_arr[c] > 0 else 0.0,
                    "p": float(p_arr[c]),
                    "nBoth": n_both,
                    "nA_only": n_a_only,
                    "nB_only": n_b_only,
                    "nNone": n_none,
                    "n_strata_used": n_strata_used,
                    "min_group_used": min_group,
                    "weighting": weighting,
                })

            pairs_done += 1
            # Progress every 50 pairs
            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                speed = (i + 1) / (elapsed / 60.0) if elapsed > 0 else 0
                remaining = len(pair_slice) - (i + 1)
                eta_min = remaining / speed if speed > 0 else 0
                mem = get_memory_mb()
                msg = (
                    f"  [{_ts()}] Shard {shard_idx + 1}/{n_shards} — "
                    f"pairs {start_shard + i + 1}/{total_pairs} | "
                    f"speed={speed:.1f} pairs/min | ETA={eta_min:.1f} min"
                )
                if mem is not None:
                    msg += f" | mem={mem:.0f} MB"
                print(msg)
                if progress_callback:
                    progress_callback(msg)
                # Heartbeat every 2 min
                if elapsed > 120 and (i + 1) % 100 == 0:
                    with open(heartbeat_path, "w") as f:
                        json.dump({
                            "shard": shard_idx + 1,
                            "pairs_done": pairs_done,
                            "total_pairs": total_pairs,
                            "elapsed_sec": time.perf_counter() - start_global,
                            "timestamp": _ts(),
                        }, f, indent=2)

        if shard_rows:
            pd.DataFrame(shard_rows).to_parquet(shard_path, index=False)
            rows.extend(shard_rows)
            print(f"  Wrote {shard_path.name}: {len(shard_rows)} rows")
        else:
            pd.DataFrame(columns=[
                "ingA_id", "ingB_id", "category", "delta", "se", "z", "p",
                "nBoth", "nA_only", "nB_only", "nNone", "n_strata_used", "min_group_used", "weighting",
            ]).to_parquet(shard_path, index=False)
            print(f"  Wrote empty {shard_path.name} (0 rows)")

    return pd.DataFrame(rows)
