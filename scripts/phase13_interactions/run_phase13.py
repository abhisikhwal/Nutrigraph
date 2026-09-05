"""
Phase 13: Ingredient Interaction Discovery (v3).
Robust, resumable pipeline: validation, sharded regression, null tests, bootstrap stability.
Run from repo root: python scripts/phase13_interactions/run_phase13.py [--config config/phase13.yaml] [--use-v2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Repo root: parent of scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

from src.phase13.interaction_atlas import (
    bh_fdr,
    bootstrap_stability,
    build_candidate_pairs,
    build_confounders,
    _run_single_pair_regression,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("phase13")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _git_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _config_hash(cfg: Dict[str, Any]) -> str:
    """Stable hash of config keys that affect pair selection and regression."""
    keys = [
        "min_ingredient_freq", "top_k_ingredients", "min_joint", "min_a_only", "min_b_only",
        "max_overlap_ratio", "min_pairs_after_filter", "relaxed_min_joint", "relaxed_min_a_only", "relaxed_min_b_only",
        "shard_size", "seed", "min_n_both_regression", "min_n_a_only_regression", "min_n_b_only_regression",
        "use_v2", "recipe_ingredients_path", "signature_paths_v3", "signature_path_v2",
    ]
    d = {k: cfg.get(k) for k in keys if k in cfg or cfg.get(k) is not None}
    blob = json.dumps(d, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or _REPO_ROOT / "config" / "phase13.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}. Tried: {path.resolve()}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_path(p: str, root: Path) -> Path:
    if os.path.isabs(p):
        return Path(p)
    return (root / p).resolve()


def get_signature_path(cfg: Dict[str, Any], use_v2: bool, root: Path) -> Path:
    if use_v2:
        path = resolve_path(cfg.get("signature_path_v2", ""), root)
        if not path.is_file():
            raise FileNotFoundError(f"V2 signature file not found: {path}. Tried: {path}")
        return path
    for rel in cfg.get("signature_paths_v3", []):
        path = resolve_path(rel, root)
        if path.is_file():
            log.info("Using v3 signatures: %s", path)
            return path
    attempted = [resolve_path(rel, root) for rel in cfg.get("signature_paths_v3", [])]
    raise FileNotFoundError(
        f"No v3 signature file found. Tried: {[str(p) for p in attempted]}"
    )


# ---------------------------------------------------------------------------
# Validation & functional columns
# ---------------------------------------------------------------------------


def detect_functional_columns_v3(df: pd.DataFrame) -> List[str]:
    exclude = {"recipe_id", "num_ingredients", "num_compounds"}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]


def detect_functional_columns_v2(df: pd.DataFrame) -> List[str]:
    metadata = {"recipe_id", "predicted_cuisine", "cuisine", "source", "num_ingredients", "num_compounds"}
    # Assume effect columns are numeric and not metadata
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in metadata]


def drop_constant_columns(df: pd.DataFrame, columns: List[str], variance_threshold: float = 1e-8) -> List[str]:
    kept = []
    dropped = []
    for c in columns:
        if c not in df.columns:
            continue
        var = df[c].var()
        if pd.isna(var) or var < variance_threshold:
            dropped.append(c)
        else:
            kept.append(c)
    if dropped:
        log.warning("Dropped constant/near-constant functional columns: %s", dropped)
    return kept


def validate_and_prepare(
    signatures: pd.DataFrame,
    recipe_ingredients: pd.DataFrame,
    use_v2: bool,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Validate recipe_id overlap, ingredient_id, detect func cols, drop constant. Returns (signatures, func_cols, recipe_ingredients)."""
    if "recipe_id" not in signatures.columns:
        raise ValueError("Signatures must contain column 'recipe_id'. Columns: %s" % list(signatures.columns))
    if "recipe_id" not in recipe_ingredients.columns or "ingredient_id" not in recipe_ingredients.columns:
        raise ValueError(
            "recipe_ingredients must contain 'recipe_id' and 'ingredient_id'. Columns: %s"
            % list(recipe_ingredients.columns)
        )

    sig_ids = set(signatures["recipe_id"].astype(str))
    ri_ids = set(recipe_ingredients["recipe_id"].astype(str))
    missing_in_ri = sig_ids - ri_ids
    missing_in_sig = ri_ids - sig_ids
    if len(sig_ids) == 0:
        raise ValueError("Signatures have no recipe_id values.")
    overlap = sig_ids & ri_ids
    if len(overlap) == 0:
        raise ValueError(
            "No recipe_id overlap between signatures and recipe_ingredients. "
            "Signatures sample: %s; recipe_ingredients sample: %s"
            % (list(sig_ids)[:5], list(ri_ids)[:5])
        )
    log.info("Recipe overlap: %d in both; %d only in signatures; %d only in recipe_ingredients",
             len(overlap), len(missing_in_ri), len(missing_in_sig))

    # Restrict to overlapping recipes for consistency
    signatures = signatures[signatures["recipe_id"].astype(str).isin(overlap)].copy()
    recipe_ingredients = recipe_ingredients[recipe_ingredients["recipe_id"].astype(str).isin(overlap)].copy()

    if use_v2:
        func_cols = detect_functional_columns_v2(signatures)
    else:
        func_cols = detect_functional_columns_v3(signatures)
    if not func_cols:
        raise ValueError("No functional columns detected. Check signature schema.")
    func_cols = drop_constant_columns(signatures, func_cols)
    if not func_cols:
        raise ValueError("All functional columns were constant. Aborting.")
    log.info("Functional columns (%d): %s", len(func_cols), func_cols)
    return signatures, func_cols, recipe_ingredients


# ---------------------------------------------------------------------------
# Confounders
# ---------------------------------------------------------------------------


def build_confounders_phase13(
    recipe_ids: pd.Series,
    recipe_ingredients: pd.DataFrame,
    signatures: Optional[pd.DataFrame],
    confounders_path: Optional[Path],
    optional_confounder_cols: List[str],
) -> pd.DataFrame:
    """num_ingredients_actual from recipe_ingredients; cuisine dummies or cuisine_unknown=1."""
    out = pd.DataFrame({"recipe_id": recipe_ids.unique()})
    num_ing = recipe_ingredients.groupby("recipe_id").size().reset_index(name="num_ingredients_actual")
    out = out.merge(num_ing, on="recipe_id", how="left")
    out["num_ingredients_actual"] = out["num_ingredients_actual"].fillna(0).astype(int)
    # Alias for atlas which expects num_ingredients
    out["num_ingredients"] = out["num_ingredients_actual"]

    cuisine_col = None
    if confounders_path and confounders_path.is_file():
        try:
            ext = pd.read_parquet(confounders_path)
            for col in ["cuisine", "predicted_cuisine", "cuisine_or_cluster"]:
                if col in ext.columns and "recipe_id" in ext.columns:
                    cuisine_col = col
                    sub = ext[["recipe_id", col]].drop_duplicates()
                    sub = sub.rename(columns={col: "cuisine_or_cluster"})
                    out = out.merge(sub, on="recipe_id", how="left")
                    break
        except Exception as e:
            log.warning("Could not load confounders from %s: %s. Using cuisine_unknown=1.", confounders_path, e)
    if "cuisine_or_cluster" not in out.columns:
        out["cuisine_or_cluster"] = "cuisine_unknown"
    out["cuisine_or_cluster"] = out["cuisine_or_cluster"].fillna("cuisine_unknown").astype("category")

    if signatures is not None and "num_compounds" in signatures.columns:
        sub = signatures[["recipe_id", "num_compounds"]].drop_duplicates()
        out = out.merge(sub, on="recipe_id", how="left")
    else:
        out["num_compounds"] = np.nan

    for col in optional_confounder_cols:
        if col and signatures is not None and col in signatures.columns:
            sub = signatures[["recipe_id", col]].drop_duplicates()
            out = out.merge(sub, on="recipe_id", how="left")
    return out


# ---------------------------------------------------------------------------
# Counts-only diagnostic and eligibility (on overlap recipes only)
# ---------------------------------------------------------------------------

def run_diagnostic_counts(
    recipe_ingredients: pd.DataFrame,
    min_ingredient_freq: int,
    top_k_ingredients: int,
) -> pd.DataFrame:
    """
    Compute ingredient frequencies and pair co-occurrence on the given recipe_ingredients
    (must already be restricted to overlapping recipes). Returns DataFrame with one row per pair:
    ingA_id, ingB_id, nA, nB, nBoth, nA_only, nB_only, overlap_ratio = nBoth/min(nA,nB).
    """
    from itertools import combinations
    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
    ing_freq = ri.groupby("ingredient_id").size().sort_values(ascending=False)
    kept_ing = ing_freq[ing_freq >= min_ingredient_freq].index.tolist()
    if not kept_ing:
        raise ValueError("No ingredients with freq >= %d. Max freq: %s" % (min_ingredient_freq, ing_freq.max()))
    top_ing = kept_ing[:top_k_ingredients]
    top_set = set(top_ing)
    pair_counts: Dict[Tuple[str, str], int] = {}
    ing_recipe_freq: Dict[str, int] = {i: 0 for i in top_ing}
    for _rid, grp in ri.groupby("recipe_id"):
        ings = [x for x in grp["ingredient_id"].unique() if x in top_set]
        for i in ings:
            ing_recipe_freq[i] = ing_recipe_freq.get(i, 0) + 1
        if len(ings) < 2:
            continue
        for a, b in combinations(sorted(ings), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    rows = []
    for (a, b), n_both in pair_counts.items():
        n_a = ing_recipe_freq.get(a, 0)
        n_b = ing_recipe_freq.get(b, 0)
        n_a_only = n_a - n_both
        n_b_only = n_b - n_both
        mn = min(n_a, n_b)
        overlap_ratio = (n_both / mn) if mn > 0 else 0.0
        rows.append({
            "ingA_id": a, "ingB_id": b,
            "nA": n_a, "nB": n_b, "nBoth": n_both,
            "nA_only": n_a_only, "nB_only": n_b_only,
            "overlap_ratio": overlap_ratio,
        })
    return pd.DataFrame(rows)


def apply_eligibility(
    pair_stats: pd.DataFrame,
    min_joint: int,
    min_a_only: int,
    min_b_only: int,
    max_overlap_ratio: float,
    min_pairs_after_filter: int,
    relaxed_min_joint: int,
    relaxed_min_a_only: int,
    relaxed_min_b_only: int,
) -> Tuple[List[Tuple[str, str]], pd.DataFrame, pd.DataFrame]:
    """
    Apply eligibility thresholds; if survivors < min_pairs_after_filter, relax and re-apply.
    Returns (candidate_pairs, tested_df, skip_df) with explicit reasons in skip_df.
    """
    def filter_df(df: pd.DataFrame, mj: int, ma: int, mb: int, mo: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        eligible = df[
            (df["nBoth"] >= mj) &
            (df["nA_only"] >= ma) &
            (df["nB_only"] >= mb) &
            (df["overlap_ratio"] <= mo)
        ].copy()
        skip = df[~df.index.isin(eligible.index)].copy()
        reasons = []
        for _, row in skip.iterrows():
            r = []
            if row["nBoth"] < mj:
                r.append("nBoth < %d" % mj)
            if row["nA_only"] < ma:
                r.append("nA_only < %d" % ma)
            if row["nB_only"] < mb:
                r.append("nB_only < %d" % mb)
            if row["overlap_ratio"] > mo:
                r.append("overlap_ratio > %.2f" % mo)
            reasons.append("; ".join(r))
        skip["reason"] = reasons
        return eligible, skip

    eligible, skip = filter_df(pair_stats, min_joint, min_a_only, min_b_only, max_overlap_ratio)
    if len(eligible) < min_pairs_after_filter:
        log.warning("Eligible pairs %d < %d; relaxing thresholds and re-applying.",
                    len(eligible), min_pairs_after_filter)
        eligible, skip = filter_df(pair_stats, relaxed_min_joint, relaxed_min_a_only, relaxed_min_b_only, max_overlap_ratio)
        log.info("After relaxation: eligible=%d", len(eligible))
    candidate_pairs = list(zip(eligible["ingA_id"].tolist(), eligible["ingB_id"].tolist()))
    candidate_pairs.sort(key=lambda x: (x[0], x[1]))
    tested_df = eligible[["ingA_id", "ingB_id", "nA", "nB", "nBoth", "nA_only", "nB_only", "overlap_ratio"]].copy()
    skip_df = skip.copy()
    skip_df["ingredient_a"] = skip_df["ingA_id"]
    skip_df["ingredient_b"] = skip_df["ingB_id"]
    skip_df = skip_df[["ingredient_a", "ingredient_b", "reason", "nA", "nB", "nBoth"]]
    return candidate_pairs, tested_df, skip_df


# ---------------------------------------------------------------------------
# Sharded regression
# ---------------------------------------------------------------------------


def run_shard(
    pair_slice: List[Tuple[str, str]],
    recipe_ingredients: pd.DataFrame,
    signatures: pd.DataFrame,
    confounders: pd.DataFrame,
    categories: List[str],
    grouped: Dict[str, set],
    min_n_both: int = 0,
    min_n_a_only: int = 0,
    min_n_b_only: int = 0,
) -> pd.DataFrame:
    rows = []
    for (a, b) in pair_slice:
        for cat in categories:
            if cat not in signatures.columns:
                continue
            r = _run_single_pair_regression(a, b, grouped, signatures, confounders, cat, None)
            if r is not None:
                if r["n_both"] >= min_n_both and r["n_A_only"] >= min_n_a_only and r["n_B_only"] >= min_n_b_only:
                    rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_all_shards(
    candidate_pairs: List[Tuple[str, str]],
    recipe_ingredients: pd.DataFrame,
    signatures: pd.DataFrame,
    confounders: pd.DataFrame,
    categories: List[str],
    shard_size: int,
    shards_dir: Path,
    progress_every: int,
    config_hash: str,
    manifest_path: Path,
    min_n_both: int = 0,
    min_n_a_only: int = 0,
    min_n_b_only: int = 0,
) -> Path:
    """
    Run sharded regression. Skip a shard only if it exists AND rowcount > 0 AND manifest config_hash matches.
    Write manifest with config_hash and shard rowcounts. If config changed, recompute (do not reuse).
    """
    shards_dir.mkdir(parents=True, exist_ok=True)
    # If manifest exists and config_hash differs, remove existing shards so we recompute
    if manifest_path.is_file():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("config_hash") != config_hash:
                log.warning("Config hash changed; recomputing all shards (old hash=%s, new=%s)",
                            manifest.get("config_hash"), config_hash)
                for p in sorted(shards_dir.glob("interactions_shard_*.parquet")):
                    p.unlink()
                    log.info("Removed %s", p.name)
        except Exception as e:
            log.warning("Could not read manifest or config hash: %s; recomputing shards", e)
            for p in sorted(shards_dir.glob("interactions_shard_*.parquet")):
                p.unlink(missing_ok=True)

    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
    grouped = ri.groupby("recipe_id")["ingredient_id"].apply(set).to_dict()
    n_pairs = len(candidate_pairs)
    n_shards = (n_pairs + shard_size - 1) // shard_size
    shard_rowcounts: Dict[str, int] = {}
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, n_pairs)
        pair_slice = candidate_pairs[start:end]
        shard_path = shards_dir / f"interactions_shard_{shard_idx + 1:04d}.parquet"
        skip = False
        if shard_path.is_file():
            try:
                df_ex = pd.read_parquet(shard_path)
                rc = len(df_ex)
                shard_rowcounts[shard_path.name] = rc
                if rc > 0:
                    skip = True
                    log.info("[%s] Skip completed shard %d/%d: %s (rows=%d)", _ts(), shard_idx + 1, n_shards, shard_path.name, rc)
            except Exception as e:
                log.warning("Shard %s corrupted or empty: %s; recomputing.", shard_path.name, e)
        if not skip:
            log.info("[%s] Running shard %d/%d (%d pairs) -> %s", _ts(), shard_idx + 1, n_shards, len(pair_slice), shard_path.name)
            df = run_shard(pair_slice, recipe_ingredients, signatures, confounders, categories, grouped,
                           min_n_both=min_n_both, min_n_a_only=min_n_a_only, min_n_b_only=min_n_b_only)
            rc = len(df)
            shard_rowcounts[shard_path.name] = rc
            if rc > 0:
                df.to_parquet(shard_path, index=False)
                log.info("Wrote %s: shape %s", shard_path.name, df.shape)
            else:
                pd.DataFrame(columns=["ingA_id", "ingB_id", "category", "beta_int", "se", "t", "p", "n_total", "n_both", "n_A_only", "n_B_only"]).to_parquet(shard_path, index=False)
                log.warning("Shard %s produced 0 rows (pair/category combos may fail min_n_*).", shard_path.name)
    with open(manifest_path, "w") as f:
        json.dump({"config_hash": config_hash, "shard_rowcounts": shard_rowcounts, "n_shards": n_shards}, f, indent=2)
    log.info("Wrote manifest: %s", manifest_path.name)
    return shards_dir


# ---------------------------------------------------------------------------
# Null test: permute IA*IB
# ---------------------------------------------------------------------------


def run_permutation_null(
    candidate_pairs: List[Tuple[str, str]],
    recipe_ingredients: pd.DataFrame,
    signatures: pd.DataFrame,
    confounders: pd.DataFrame,
    categories: List[str],
    n_subset_pairs: int,
    seed: int,
) -> Dict[str, Any]:
    """Permute IA*IB across recipes for a subset of pairs; check p-values become ~uniform."""
    if sm is None:
        return {"error": "statsmodels not available", "n_pairs": 0}
    rng = np.random.default_rng(seed)
    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
    grouped = ri.groupby("recipe_id")["ingredient_id"].apply(set).to_dict()
    pairs_sub = candidate_pairs[:n_subset_pairs] if len(candidate_pairs) > n_subset_pairs else candidate_pairs
    p_values_perm = []
    for (a, b) in pairs_sub:
        set_a = set()
        set_b = set()
        for rid, ings in grouped.items():
            if a in ings:
                set_a.add(rid)
            if b in ings:
                set_b.add(rid)
        relevant = sorted(set_a | set_b)
        if len(relevant) < 30:
            continue
        A = np.array([1 if rec in set_a else 0 for rec in relevant], dtype=np.float64)
        B = np.array([1 if rec in set_b else 0 for rec in relevant], dtype=np.float64)
        true_interaction = A * B
        for cat in categories:
            if cat not in signatures.columns:
                continue
            perm_interaction = rng.permutation(true_interaction)
            r = _run_single_pair_regression(a, b, grouped, signatures, confounders, cat, None, interaction_override=perm_interaction)
            if r is not None:
                p_values_perm.append(r["p"])
            if len(p_values_perm) >= 500:
                break
        if len(p_values_perm) >= 500:
            break
    p_values_perm = np.array(p_values_perm)
    if len(p_values_perm) == 0:
        return {"n_permutation_pvalues": 0, "message": "No valid permutation runs"}
    # Under null, p should be uniform; fraction significant at 0.05 should be ~0.05
    frac_sig_005 = (p_values_perm <= 0.05).mean()
    frac_sig_001 = (p_values_perm <= 0.01).mean()
    return {
        "n_permutation_pvalues": int(len(p_values_perm)),
        "frac_significant_0.05": float(frac_sig_005),
        "frac_significant_0.01": float(frac_sig_001),
        "expected_under_null_0.05": 0.05,
        "expected_under_null_0.01": 0.01,
        "p_value_median": float(np.median(p_values_perm)),
        "p_value_mean": float(np.mean(p_values_perm)),
    }


def run_permutation_null_within_stratum(
    candidate_pairs: List[Tuple[str, str]],
    recipe_ingredients: pd.DataFrame,
    signatures: pd.DataFrame,
    confounders: pd.DataFrame,
    categories: List[str],
    n_subset_pairs: int,
    seed: int,
) -> Dict[str, Any]:
    """Permute IA*IB within each cuisine_or_cluster stratum; p-values should be ~uniform under null."""
    if sm is None:
        return {"error": "statsmodels not available", "n_pairs": 0}
    rng = np.random.default_rng(seed)
    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
    grouped = ri.groupby("recipe_id")["ingredient_id"].apply(set).to_dict()
    if "cuisine_or_cluster" not in confounders.columns:
        return {"message": "No cuisine_or_cluster; skipping within-stratum null", "n_permutation_pvalues": 0}
    rec_to_stratum = confounders.set_index("recipe_id")["cuisine_or_cluster"].astype(str).to_dict()
    pairs_sub = candidate_pairs[:n_subset_pairs] if len(candidate_pairs) > n_subset_pairs else candidate_pairs
    p_values_perm = []
    for (a, b) in pairs_sub:
        set_a = set()
        set_b = set()
        for rid, ings in grouped.items():
            if a in ings:
                set_a.add(rid)
            if b in ings:
                set_b.add(rid)
        relevant = sorted(set_a | set_b)
        if len(relevant) < 30:
            continue
        A = np.array([1 if rec in set_a else 0 for rec in relevant], dtype=np.float64)
        B = np.array([1 if rec in set_b else 0 for rec in relevant], dtype=np.float64)
        true_interaction = A * B
        strata = [rec_to_stratum.get(rec, "unknown") for rec in relevant]
        perm_interaction = np.empty_like(true_interaction)
        for stratum in set(strata):
            idx = [i for i, s in enumerate(strata) if s == stratum]
            if len(idx) < 2:
                perm_interaction[idx] = true_interaction[idx]
                continue
            perm_interaction[idx] = rng.permutation(true_interaction[idx])
        for cat in categories:
            if cat not in signatures.columns:
                continue
            r = _run_single_pair_regression(a, b, grouped, signatures, confounders, cat, None, interaction_override=perm_interaction)
            if r is not None:
                p_values_perm.append(r["p"])
            if len(p_values_perm) >= 500:
                break
        if len(p_values_perm) >= 500:
            break
    p_values_perm = np.array(p_values_perm)
    if len(p_values_perm) == 0:
        return {"n_permutation_pvalues_within_stratum": 0, "message": "No valid runs"}
    return {
        "n_permutation_pvalues_within_stratum": int(len(p_values_perm)),
        "frac_significant_0.05_within_stratum": float((p_values_perm <= 0.05).mean()),
        "p_value_median_within_stratum": float(np.median(p_values_perm)),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    config_path: Optional[Path] = None,
    use_v2: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    if sm is None:
        raise RuntimeError(
            "statsmodels is required for regression. Install with: pip install statsmodels"
        )
    cfg = load_config(config_path)
    root = _REPO_ROOT
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = Path(cfg.get("output_dir", "data/processed/phase13_interactions_v3")).parent
    out_dir = root / base_dir / f"phase13_interactions_v3_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / cfg.get("shards_dir", "shards")
    manifest_path = out_dir / "shard_manifest.json"
    cfg_hash = {**cfg, "use_v2": use_v2}
    config_hash = _config_hash(cfg_hash)
    log.info("[%s] RUN_ID=%s | output_dir=%s | config_hash=%s", _ts(), run_id, out_dir, config_hash)

    # --- Inputs ---
    sig_path = get_signature_path(cfg, use_v2, root)
    ri_path = resolve_path(cfg.get("recipe_ingredients_path", ""), root)
    if not ri_path.is_file():
        raise FileNotFoundError(f"recipe_ingredients not found: {ri_path}. Tried: {ri_path}")
    conf_path = None
    if cfg.get("confounders_path"):
        conf_path = resolve_path(cfg["confounders_path"], root)

    log.info("[%s] Loading signatures from %s", _ts(), sig_path)
    signatures = pd.read_parquet(sig_path)
    log.info("[%s] Loading recipe_ingredients from %s", _ts(), ri_path)
    recipe_ingredients = pd.read_parquet(ri_path)

    signatures, func_cols, recipe_ingredients = validate_and_prepare(signatures, recipe_ingredients, use_v2)
    log.info("Signatures shape after validation: %s (overlap recipes only)", signatures.shape)

    # Confounders
    confounders = build_confounders_phase13(
        signatures["recipe_id"],
        recipe_ingredients,
        signatures,
        conf_path,
        optional_confounder_cols=cfg.get("optional_confounder_cols", []),
    )
    log.info("Confounders shape: %s; columns: %s", confounders.shape, list(confounders.columns))

    # --- Counts-only diagnostic (on overlap recipes only) ---
    min_freq = int(cfg.get("min_ingredient_freq", 500))
    top_k = int(cfg.get("top_k_ingredients", 500))
    pair_stats = run_diagnostic_counts(recipe_ingredients, min_freq, top_k)
    log.info("[%s] Diagnostic: %d candidate pairs (before eligibility)", _ts(), len(pair_stats))
    if len(pair_stats) > 0:
        n_both = pair_stats["nBoth"]
        log.info("  nBoth: min=%d, max=%d, median=%.0f, mean=%.1f",
                 n_both.min(), n_both.max(), n_both.median(), n_both.mean())
        log.info("  Pairs with nBoth>=50: %d", (n_both >= 50).sum())
        log.info("  Pairs with nBoth>=20: %d", (n_both >= 20).sum())

    # Eligibility (with auto-relax if survivors < min_pairs_after_filter)
    min_joint = int(cfg.get("min_joint", 50))
    min_a_only = int(cfg.get("min_a_only", 200))
    min_b_only = int(cfg.get("min_b_only", 200))
    max_overlap_ratio = float(cfg.get("max_overlap_ratio", 0.9))
    min_pairs_after_filter = int(cfg.get("min_pairs_after_filter", 2000))
    relaxed_min_joint = int(cfg.get("relaxed_min_joint", 20))
    relaxed_min_a_only = int(cfg.get("relaxed_min_a_only", 100))
    relaxed_min_b_only = int(cfg.get("relaxed_min_b_only", 100))
    candidate_pairs, tested_df, skip_df = apply_eligibility(
        pair_stats,
        min_joint, min_a_only, min_b_only, max_overlap_ratio,
        min_pairs_after_filter,
        relaxed_min_joint, relaxed_min_a_only, relaxed_min_b_only,
    )
    log.info("[%s] Eligible pairs: %d | skipped: %d | tested_df: %s", _ts(), len(candidate_pairs), len(skip_df), tested_df.shape)
    if len(candidate_pairs) == 0:
        raise RuntimeError(
            "Zero eligible pairs after thresholds. Relax min_joint / min_a_only / min_b_only or check data. "
            "Skip reasons sample: %s" % (skip_df["reason"].value_counts().head().to_dict() if len(skip_df) > 0 else "N/A")
        )

    # Save pair outputs (explicit reasons)
    tested_path = out_dir / "ingredient_pairs_tested_v3.parquet"
    tested_df.to_parquet(tested_path, index=False)
    log.info("Wrote %s: %s", tested_path.name, tested_df.shape)
    skip_path = out_dir / "pair_skip_reasons_v3.parquet"
    skip_df.to_parquet(skip_path, index=False)
    log.info("Wrote %s: %s", skip_path.name, skip_df.shape)
    if len(skip_df) > 0:
        log.info("Skip reason counts: %s", skip_df["reason"].value_counts().to_dict())

    # Sharded regression (skip shard only if exists + rowcount > 0 + manifest config_hash matches)
    shard_size = int(cfg.get("shard_size", 5000))
    progress_every = int(cfg.get("progress_every", 1000))
    min_n_both_r = int(cfg.get("min_n_both_regression", 10))
    min_n_a_only_r = int(cfg.get("min_n_a_only_regression", 10))
    min_n_b_only_r = int(cfg.get("min_n_b_only_regression", 10))
    seed = int(cfg.get("seed", 42))
    run_all_shards(
        candidate_pairs,
        recipe_ingredients,
        signatures,
        confounders,
        func_cols,
        shard_size,
        shards_dir,
        progress_every,
        config_hash=config_hash,
        manifest_path=manifest_path,
        min_n_both=min_n_both_r,
        min_n_a_only=min_n_a_only_r,
        min_n_b_only=min_n_b_only_r,
    )

    # Merge shards -> raw
    raw_path = out_dir / "interactions_raw_v3.parquet"
    shard_files = sorted(shards_dir.glob("interactions_shard_*.parquet"))
    if not shard_files:
        raise RuntimeError("No shard files found in %s" % shards_dir)
    parts = [pd.read_parquet(p) for p in shard_files]
    raw = pd.concat(parts, ignore_index=True)
    n_raw = len(raw)
    raw.to_parquet(raw_path, index=False)
    log.info("[%s] Wrote %s: %s (tests run=%d)", _ts(), raw_path.name, raw.shape, n_raw)
    if n_raw == 0:
        with open(manifest_path) as f:
            m = json.load(f)
        log.error("CRITICAL: Zero interaction results. Shard rowcounts: %s. Fix eligibility or regression mins.", m.get("shard_rowcounts", {}))
        raise RuntimeError("interactions_raw_v3.parquet is empty. Check shard_manifest.json shard_rowcounts and eligibility thresholds.")

    # FDR per category and globally
    raw = raw.copy()
    raw["q_category"] = np.nan
    for cat in raw["category"].unique():
        mask = raw["category"] == cat
        p = raw.loc[mask, "p"].values
        raw.loc[mask, "q_category"] = bh_fdr(p)
    raw["q_global"] = bh_fdr(raw["p"].values)
    raw["significant_005"] = raw["q_global"] <= 0.05
    raw["significant_001"] = raw["q_global"] <= 0.01
    adjusted_path = out_dir / "interactions_adjusted_v3.parquet"
    raw.to_parquet(adjusted_path, index=False)
    log.info("Wrote %s: %s", adjusted_path.name, raw.shape)

    # Null tests: global permutation + within-stratum permutation
    null_n = int(cfg.get("null_n_pairs", 200))
    null_result = run_permutation_null(
        candidate_pairs, recipe_ingredients, signatures, confounders, func_cols, null_n, seed
    )
    null_within = run_permutation_null_within_stratum(
        candidate_pairs, recipe_ingredients, signatures, confounders, func_cols, null_n, seed
    )
    null_result["within_stratum"] = null_within
    null_path = out_dir / "null_test_v3.json"
    with open(null_path, "w") as f:
        json.dump(null_result, f, indent=2)
    log.info("Wrote %s: global=%s within_stratum=%s", null_path.name, null_result.get("n_permutation_pvalues"), null_result.get("within_stratum", {}).get("n_permutation_pvalues_within_stratum"))

    # Bootstrap stability (atlas expects "q" column); skip if no results
    boot_df = pd.DataFrame()
    if n_raw > 0:
        top_n_boot = int(cfg.get("bootstrap_top_n", 200))
        B_boot = int(cfg.get("bootstrap_n_samples", 200))
        adjusted = pd.read_parquet(adjusted_path).copy()
        if "q" not in adjusted.columns and "q_global" in adjusted.columns:
            adjusted["q"] = adjusted["q_global"]
        boot_df = bootstrap_stability(
            adjusted, recipe_ingredients, signatures, confounders,
            top_n=top_n_boot, B=B_boot, seed=seed, n_jobs=1,
        )
    boot_path = out_dir / "bootstrap_stability_v3.parquet"
    boot_df.to_parquet(boot_path, index=False)
    log.info("Wrote %s: %s", boot_path.name, boot_df.shape)

    # Summary JSON (raw has "category" column)
    if n_raw > 0:
        top100 = raw.nsmallest(100, "q_global")[["ingA_id", "ingB_id", "category", "beta_int", "q_global", "n_both"]].copy()
        top100 = top100.rename(columns={"category": "functional_category"})
    else:
        top100 = pd.DataFrame(columns=["ingA_id", "ingB_id", "functional_category", "beta_int", "q_global", "n_both"])
    summary = {
        "run_id": run_id,
        "n_recipes": int(signatures["recipe_id"].nunique()),
        "pairs_eligible": len(candidate_pairs),
        "pairs_skipped": int(len(skip_df)),
        "tests_run": int(len(raw)),
        "n_significant_005": int(raw["significant_005"].sum()),
        "n_significant_001": int(raw["significant_001"].sum()),
        "n_ingredients": int(recipe_ingredients["ingredient_id"].nunique()),
        "func_cols": func_cols,
        "confounders": list(confounders.columns),
        "run_timestamp": _ts(),
        "git_commit": _git_hash(),
        "use_v2": use_v2,
        "config_hash": config_hash,
        "null_test": null_result,
        "top_100_by_q": top100.to_dict(orient="records") if len(top100) > 0 else [],
    }
    summary_path = out_dir / "phase13_summary_v3.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", summary_path.name)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 13 Ingredient Interaction Discovery (v3)")
    parser.add_argument("--config", type=Path, default=None, help="Path to phase13.yaml")
    parser.add_argument("--use-v2", action="store_true", help="Use v2 signatures (55 categories)")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for output dir phase13_interactions_v3_{RUN_ID}")
    args = parser.parse_args()
    run_pipeline(config_path=args.config, use_v2=args.use_v2, run_id=args.run_id)


if __name__ == "__main__":
    main()
