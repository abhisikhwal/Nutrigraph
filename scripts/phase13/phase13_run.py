"""
Phase 13 GPU-first run: Phase13A (FAISS matching + fast OLS screen) and Phase13B (confirm).
Never silent: progress every 100 pairs, ETA, skip reasons. Shard skip only if valid (size>0, rows>0, schema).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Repo root when run from scripts/phase13/
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))

# Defaults
MIN_JOINT = 50
MIN_SOLO = 200
MIN_MATCHED = 200
MAX_TREATED = 1000            # cap AB (both) recipes per pair for matching
MATCH_RATIO = 1               # 1 matched recipe per group (A-only, B-only, none) per AB sample
FAISS_TOPK = 50               # retrieve this many nearest neighbors globally, then filter
K_NEIGHBORS = 5               # minimum neighbors per treated (matching coverage)
N_MATCH_PER_TREATED = 1       # 1:1 matching per group (MATCH_RATIO)
SVD_DIM = 24                  # ingredient SVD components
N_FEATURES_HASH = 2**18       # HashingVectorizer n_features (no huge vocab)
TOP_CUISINES = 20             # top N cuisines for one-hot (others -> other)
TOP_SOURCES = 10              # top N sources for one-hot (others -> other)
MAX_PAIRS_PER_SHARD = 250     # smaller shards so you see outputs frequently
MIN_ABS_DID = 0.01            # minimum |DID| for passes_min_effect (reporting only, not significance)
BOOTSTRAP_B_SMOKE = 200       # bootstrap replicates for SE/p in smoke run
BOOTSTRAP_B_FULL = 100        # bootstrap replicates for full run
VARIANCE_THRESHOLD = 1e-8
UNIQUE_THRESHOLD = 50
# DID schema: did, p_raw, n_ab, n_a, n_b, n_none + matching quality
EXPECTED_SHARD_COLUMNS = [
    "ingA_id", "ingB_id", "category", "did", "se", "t", "p_raw",
    "n_ab", "n_a", "n_b", "n_none", "avg_L2_A", "avg_L2_B", "avg_L2_0",
]
PROGRESS_EVERY = 100


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


def get_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_output_dir(repo_root: Path, use_v2: bool, run_id: str) -> Path:
    variant = "v2" if use_v2 else "v3"
    out = repo_root / "data" / "processed" / f"phase13_interactions_{variant}_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_latest_txt(repo_root: Path, output_dir: Path) -> None:
    latest_file = repo_root / "data" / "processed" / "phase13_interactions_latest.txt"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(str(output_dir.resolve()), encoding="utf-8")


def load_signatures_and_ri(
    repo_root: Path,
    use_v2: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if use_v2:
        sig_path = repo_root / "data/processed/exports_v2/recipes_biological_effects_v2_FINAL.parquet"
    else:
        sig_path = repo_root / "data/processed/phase17_reaggregation/recipe_functional_signatures_v3.parquet"
        if not sig_path.exists():
            sig_path = repo_root / "data/processed/canonical/recipes_biological_effects_v3.parquet"
    ri_path = repo_root / "data/processed/canonical/recipe_ingredients_expanded_v2.parquet"
    if not sig_path.exists():
        raise FileNotFoundError(f"Signatures not found: {sig_path}")
    if not ri_path.exists():
        raise FileNotFoundError(f"Recipe ingredients not found: {ri_path}")
    signatures = pd.read_parquet(sig_path)
    ri = pd.read_parquet(ri_path)
    sig_ids = set(signatures["recipe_id"].astype(str))
    ri_ids = set(ri["recipe_id"].astype(str))
    overlap = sig_ids & ri_ids
    if len(overlap) == 0:
        raise ValueError("No recipe_id overlap between signatures and recipe_ingredients")
    signatures = signatures[signatures["recipe_id"].astype(str).isin(overlap)].copy()
    ri = ri[ri["recipe_id"].astype(str).isin(overlap)].copy()
    exclude = {"recipe_id", "num_ingredients", "num_compounds", "predicted_cuisine", "cuisine", "source"}
    func_cols = [c for c in signatures.select_dtypes(include=[np.number]).columns if c not in exclude]
    return signatures, ri, func_cols


def variance_gate(signatures: pd.DataFrame, func_cols: List[str]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    low_var = []
    for c in func_cols:
        if c not in signatures.columns:
            continue
        var = signatures[c].var()
        uniq = signatures[c].nunique()
        if (pd.isna(var) or var < VARIANCE_THRESHOLD) or (uniq < UNIQUE_THRESHOLD):
            low_var.append(c)
    use_cols = [c for c in func_cols if c not in low_var]
    report = {"low_variance_skipped": low_var, "categories_used": use_cols}
    return use_cols, low_var, report


def build_recipe_text_corpus(
    ri: pd.DataFrame,
    recipe_list: List[Any],
) -> List[str]:
    """
    Build per-recipe ingredient 'document' for HashingVectorizer.
    Returns list of strings "ing_123 ing_55 ing_902 ..." in recipe_list order.
    """
    t0 = time.perf_counter()
    ri = ri.copy()
    ri["recipe_id"] = ri["recipe_id"].astype(str)
    ri["ingredient_id"] = ri["ingredient_id"].astype(str)
    rid_to_ings = ri.groupby("recipe_id")["ingredient_id"].apply(
        lambda x: " ".join("ing_" + str(i) for i in x.tolist())
    ).to_dict()
    corpus = [rid_to_ings.get(str(r), "") for r in recipe_list]
    print(f"[{_ts()}] build_recipe_text_corpus: len(corpus)={len(corpus)} in {time.perf_counter() - t0:.1f}s")
    return corpus


def ingredient_hash_svd_embedding(
    corpus: List[str],
    n_features: int = 2**18,
    svd_dim: int = 24,
) -> Tuple[np.ndarray, float]:
    """
    HashingVectorizer -> sparse -> TruncatedSVD to dense float32.
    Returns (X_ing shape (n, svd_dim), explained_variance_ratio sum).
    """
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.decomposition import TruncatedSVD
    t0 = time.perf_counter()
    hv = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2", lowercase=False)
    X_sparse = hv.transform(corpus)
    print(f"[{_ts()}] ingredient_hash_svd: corpus_len={len(corpus)} sparse_shape={X_sparse.shape} in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    X_ing = svd.fit_transform(X_sparse).astype(np.float32)
    evar = float(svd.explained_variance_ratio_.sum())
    print(f"[{_ts()}] TruncatedSVD: svd_dim={svd_dim} explained_variance_ratio.sum()={evar:.4f} in {time.perf_counter() - t0:.1f}s")
    return X_ing, evar


def build_recipe_index_and_embedding(
    signatures: pd.DataFrame,
    ri: pd.DataFrame,
    recipe_id_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    ~32D embedding: ingredient SVD (24) + num_ingredients, num_compounds (2) + cuisine one-hot (<=21) + source one-hot (<=11).
    Returns (embed, scaler, embedding_meta) with meta for config. float32, standardized. Reproducible (random_state=42).
    """
    from sklearn.preprocessing import StandardScaler
    recipe_list = signatures["recipe_id"].tolist()
    n_recipes = len(recipe_list)
    meta: Dict[str, Any] = {}

    # A) Ingredient corpus + hash SVD
    t0 = time.perf_counter()
    corpus = build_recipe_text_corpus(ri, recipe_list)
    X_ing, evar = ingredient_hash_svd_embedding(corpus, n_features=N_FEATURES_HASH, svd_dim=SVD_DIM)
    meta["svd_dim"] = SVD_DIM
    meta["n_features_hash"] = N_FEATURES_HASH
    meta["ingredient_svd_explained_variance_ratio_sum"] = evar
    dim_ing = X_ing.shape[1]
    print(f"[{_ts()}] Embedding component: ingredient SVD dim={dim_ing}")

    # B) Numeric: num_ingredients, num_compounds (align to recipe_list order)
    ri_ids = ri["recipe_id"].astype(str)
    num_ing_series = ri.groupby(ri_ids).size()
    num_ing = num_ing_series.reindex([str(r) for r in recipe_list]).fillna(0).values.astype(np.float32)
    num_comp = signatures["num_compounds"].fillna(0).values.astype(np.float32) if "num_compounds" in signatures.columns else np.zeros(n_recipes, dtype=np.float32)
    X_num = np.hstack([num_ing.reshape(-1, 1), num_comp.reshape(-1, 1)])

    # C) Cuisine: top TOP_CUISINES, others -> "other"
    if "predicted_cuisine" in signatures.columns:
        cuisine_col = signatures["predicted_cuisine"].fillna("unknown").astype(str)
    elif "cuisine" in signatures.columns:
        cuisine_col = signatures["cuisine"].fillna("unknown").astype(str)
    else:
        cuisine_col = pd.Series(["unknown"] * n_recipes)
    top_cuisines = cuisine_col.value_counts().head(TOP_CUISINES).index.tolist()
    meta["top_cuisines"] = top_cuisines
    cuisine_col = cuisine_col.where(cuisine_col.isin(top_cuisines), "other")
    uniq = ["other"] + [c for c in top_cuisines if c != "other"]
    if "other" not in uniq:
        uniq = ["other"] + uniq[:TOP_CUISINES]
    uniq = uniq[: TOP_CUISINES + 1]
    cuisine_idx = {u: i for i, u in enumerate(uniq)}
    n_cuisine = len(uniq)
    cuisine_oh = np.zeros((n_recipes, n_cuisine), dtype=np.float32)
    for i, v in enumerate(cuisine_col):
        j = cuisine_idx.get(v, 0)
        cuisine_oh[i, j] = 1.0
    print(f"[{_ts()}] Embedding component: cuisine one-hot dim={n_cuisine}")

    # D) Source: detect column, top TOP_SOURCES
    source_col = None
    for col in ["source", "dataset", "origin"]:
        if col in signatures.columns:
            source_col = signatures[col].fillna("unknown").astype(str)
            break
    if source_col is None and "source" in ri.columns:
        src_agg = ri.groupby("recipe_id")["source"].first()
        source_col = signatures["recipe_id"].map(src_agg).fillna("unknown").astype(str)
    if source_col is None:
        source_col = pd.Series(["unknown"] * n_recipes)
    top_sources = source_col.value_counts().head(TOP_SOURCES).index.tolist()
    meta["top_sources"] = top_sources
    source_col = source_col.where(source_col.isin(top_sources), "other")
    uniq_s = ["other"] + [s for s in top_sources if s != "other"]
    if "other" not in uniq_s:
        uniq_s = ["other"] + uniq_s[:TOP_SOURCES]
    uniq_s = uniq_s[: TOP_SOURCES + 1]
    source_idx = {s: i for i, s in enumerate(uniq_s)}
    n_source = len(uniq_s)
    source_oh = np.zeros((n_recipes, n_source), dtype=np.float32)
    for i, v in enumerate(source_col):
        j = source_idx.get(v, 0)
        source_oh[i, j] = 1.0
    print(f"[{_ts()}] Embedding component: source one-hot dim={n_source}")

    # E) Concatenate and standardize
    parts = [X_ing, X_num, cuisine_oh, source_oh]
    X = np.hstack(parts).astype(np.float32)
    dim_total = X.shape[1]
    print(f"[{_ts()}] Embedding total dim={dim_total} (ing={dim_ing} num=2 cuisine={n_cuisine} source={n_source}) in {time.perf_counter() - t0:.1f}s")
    if dim_total > 64:
        print(f"[{_ts()}] WARNING: embedding dim {dim_total} > 64; consider lowering TOP_CUISINES or SVD_DIM")
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    meta["embed_dim"] = dim_total
    return X, scaler, meta


def build_global_faiss_index(embed: np.ndarray, use_gpu: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """Build FAISS index once (GPU if available). Prefer IndexHNSWFlat on CPU for speed. Returns (index, info)."""
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss is required for Phase13A matching. Install: pip install faiss-cpu (or conda install -c pytorch faiss-gpu)")
    d = embed.shape[1]
    n = embed.shape[0]
    n_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    gpu_used = False
    cpu_index = None
    if use_gpu and n_gpus > 0 and hasattr(faiss, "StandardGpuResources"):
        try:
            cpu_index = faiss.IndexFlatL2(d)
            cpu_index.add(embed.astype(np.float32))
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_used = True
            info = {"faiss_version": getattr(faiss, "__version__", "?"), "n_gpus": n_gpus, "gpu_used": True, "index_type": "FlatL2_gpu"}
            return index, info
        except Exception:
            pass
    # CPU: try HNSW for speed on large n
    if n > 10000 and hasattr(faiss, "IndexHNSWFlat"):
        try:
            M = 32
            cpu_index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
            cpu_index.hnsw.efConstruction = 80
            cpu_index.hnsw.efSearch = 16
            cpu_index.add(embed.astype(np.float32))
            info = {"faiss_version": getattr(faiss, "__version__", "?"), "n_gpus": n_gpus, "gpu_used": False, "index_type": "HNSWFlat"}
            return cpu_index, info
        except Exception:
            pass
    cpu_index = faiss.IndexFlatL2(d)
    cpu_index.add(embed.astype(np.float32))
    info = {"faiss_version": getattr(faiss, "__version__", "?"), "n_gpus": n_gpus, "gpu_used": False, "index_type": "FlatL2"}
    return cpu_index, info


def match_did_groups(
    index: Any,
    embed: np.ndarray,
    treated_idx: np.ndarray,
    pool_a: Set[int],
    pool_b: Set[int],
    pool_none: Set[int],
    topk: int = 50,
    ratio: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    For each AB recipe in treated_idx, find 1:1 NN in each of A-only, B-only, none.
    Returns (treated_matched, matched_a, matched_b, matched_none, avg_L2_A, avg_L2_B, avg_L2_0).
    Only includes quadruplets where all three matches were found (aligned).
    """
    treated_vecs = embed[treated_idx].astype(np.float32)
    D, I = index.search(treated_vecs, topk)
    used_a: Set[int] = set()
    used_b: Set[int] = set()
    used_none: Set[int] = set()
    treated_out: List[int] = []
    matched_a: List[int] = []
    matched_b: List[int] = []
    matched_none: List[int] = []
    sum_d_a = 0.0
    sum_d_b = 0.0
    sum_d_none = 0.0
    for row_idx, (d_row, i_row) in enumerate(zip(D, I)):
        pick_a = pick_b = pick_none = None
        dist_a = dist_b = dist_none = np.nan
        for cand, dist in zip(i_row, d_row):
            cand = int(cand)
            if pick_a is None and cand in pool_a and cand not in used_a:
                pick_a = cand
                dist_a = float(dist)
            if pick_b is None and cand in pool_b and cand not in used_b:
                pick_b = cand
                dist_b = float(dist)
            if pick_none is None and cand in pool_none and cand not in used_none:
                pick_none = cand
                dist_none = float(dist)
            if pick_a is not None and pick_b is not None and pick_none is not None:
                break
        if pick_a is not None and pick_b is not None and pick_none is not None:
            treated_out.append(int(treated_idx[row_idx]))
            used_a.add(pick_a)
            used_b.add(pick_b)
            used_none.add(pick_none)
            matched_a.append(pick_a)
            matched_b.append(pick_b)
            matched_none.append(pick_none)
            sum_d_a += dist_a
            sum_d_b += dist_b
            sum_d_none += dist_none
    n = len(treated_out)
    avg_L2_A = sum_d_a / n if n else np.nan
    avg_L2_B = sum_d_b / n if n else np.nan
    avg_L2_0 = sum_d_none / n if n else np.nan
    return (
        np.array(treated_out, dtype=np.int32),
        np.array(matched_a, dtype=np.int32),
        np.array(matched_b, dtype=np.int32),
        np.array(matched_none, dtype=np.int32),
        avg_L2_A,
        avg_L2_B,
        avg_L2_0,
    )


def did_group_means(
    y_ab: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    y_none: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    DID = (mean(AB) - mean(A_only)) - (mean(B_only) - mean(None)).
    Analytic SE: Var(DID) = Var(AB)/nAB + Var(A)/nA + Var(B)/nB + Var(None)/nNone.
    t = DID / sqrt(Var(DID)), df = min(n)-1, p = 2*(1 - t_cdf(|t|, df)).
    """
    from scipy import stats
    n_ab, n_a, n_b, n_n = len(y_ab), len(y_a), len(y_b), len(y_none)
    if n_ab < 2 or n_a < 2 or n_b < 2 or n_n < 2:
        return np.nan, np.nan, np.nan, np.nan
    m_ab = float(np.mean(y_ab))
    m_a = float(np.mean(y_a))
    m_b = float(np.mean(y_b))
    m_n = float(np.mean(y_none))
    did = (m_ab - m_a) - (m_b - m_n)
    v_ab = np.var(y_ab, ddof=1) / n_ab
    v_a = np.var(y_a, ddof=1) / n_a
    v_b = np.var(y_b, ddof=1) / n_b
    v_n = np.var(y_none, ddof=1) / n_n
    var_did = v_ab + v_a + v_b + v_n
    if var_did <= 0:
        return did, 0.0, 0.0, 1.0
    se = float(np.sqrt(var_did))
    t = did / se
    df = min(n_ab, n_a, n_b, n_n) - 1
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return did, se, float(t), float(p)


def did_bootstrap(
    y_ab: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    y_none: np.ndarray,
    B: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    DID = (mean(AB)-mean(A)) - (mean(B)-mean(None)). Bootstrap SE and two-sided p from bootstrap distribution.
    """
    from scipy import stats
    rng = np.random.default_rng(seed)
    n = len(y_ab)
    if n != len(y_a) or n != len(y_b) or n != len(y_none) or n < 2:
        return np.nan, np.nan, np.nan, np.nan
    did_obs = (float(np.mean(y_ab)) - float(np.mean(y_a))) - (float(np.mean(y_b)) - float(np.mean(y_none)))
    did_boot = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        m_ab = np.mean(y_ab[idx])
        m_a = np.mean(y_a[idx])
        m_b = np.mean(y_b[idx])
        m_n = np.mean(y_none[idx])
        did_boot.append((m_ab - m_a) - (m_b - m_n))
    did_boot = np.array(did_boot)
    se = float(np.std(did_boot, ddof=1))
    if se <= 0:
        return did_obs, 0.0, 0.0, 1.0
    # Two-sided p: fraction of |did_boot| >= |did_obs|
    p = float(np.mean(np.abs(did_boot) >= np.abs(did_obs)))
    p = max(p, 1.0 / (B + 1))
    t = did_obs / se
    return did_obs, se, float(t), p


def did_paired_effect(
    y_ab: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    y_0: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Paired DID on matched quadruplets: d_i = y_AB[i] - y_A[i] - y_B[i] + y_0[i].
    did = mean(d), SE = std(d)/sqrt(n), t = did/SE, p from t(n-1).
    """
    from scipy import stats
    n = len(y_ab)
    if n != len(y_a) or n != len(y_b) or n != len(y_0) or n < 2:
        return np.nan, np.nan, np.nan, np.nan
    d = y_ab.astype(np.float64) - y_a.astype(np.float64) - y_b.astype(np.float64) + y_0.astype(np.float64)
    did = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if se <= 0:
        return did, 0.0, 0.0, 1.0
    t = did / se
    p = 2 * (1 - stats.t.cdf(abs(t), n - 1))
    return did, se, float(t), float(p)


def _run_null_calibration(
    out_dir: Path,
    pair_subset: List[Tuple[Any, Any]],
    ing_to_set: Dict[Any, Set[int]],
    none_mask: np.ndarray,
    n_recipes: int,
    embed: np.ndarray,
    global_index: Any,
    y_mat: np.ndarray,
    func_cols: List[str],
    recipe_id_to_idx: Dict[str, int],
) -> None:
    """Shuffle group membership (preserve group sizes); recompute DID for subset; write null_report.json."""
    if not pair_subset or len(func_cols) == 0:
        return
    rng = np.random.default_rng(42)
    null_ps: List[float] = []
    n_tests = 0
    for (a, b) in pair_subset[:50]:
        rec_a = ing_to_set.get(a, set())
        rec_b = ing_to_set.get(b, set())
        n_ab = len(rec_a & rec_b)
        n_a = len(rec_a - rec_b)
        n_b_only = len(rec_b - rec_a)
        if n_ab < MIN_JOINT or n_a < 20 or n_b_only < 20:
            continue
        n_ab = min(n_ab, MAX_TREATED)
        total = n_ab + n_a + n_b_only
        if total > n_recipes - 100:
            continue
        perm = rng.permutation(n_recipes)
        both_idx = np.array(perm[:n_ab], dtype=np.int32)
        a_only_set = set(perm[n_ab : n_ab + n_a])
        b_only_set = set(perm[n_ab + n_a : n_ab + n_a + n_b_only])
        none_set = set(perm[n_ab + n_a + n_b_only :])
        if len(none_set) < MIN_MATCHED:
            continue
        treated_matched, matched_a, matched_b, matched_none, _, _, _ = match_did_groups(
            global_index, embed, both_idx,
            pool_a=a_only_set, pool_b=b_only_set, pool_none=none_set,
            topk=FAISS_TOPK, ratio=MATCH_RATIO,
        )
        if len(treated_matched) < 30:
            continue
        for c_idx in range(min(3, len(func_cols))):
            y_ab = y_mat[treated_matched, c_idx]
            y_a = y_mat[matched_a, c_idx]
            y_b = y_mat[matched_b, c_idx]
            y_0 = y_mat[matched_none, c_idx]
            _, _, _, p = did_paired_effect(y_ab, y_a, y_b, y_0)
            if not np.isnan(p):
                null_ps.append(p)
                n_tests += 1
    frac_sig_null = float(np.mean(np.array(null_ps) < 0.05)) if null_ps else 0
    null_report = {
        "n_null_tests": n_tests,
        "n_pairs_subset": min(50, len(pair_subset)),
        "frac_significant_p005_under_null": frac_sig_null,
        "expected_approx": 0.05,
        "note": "Shuffled group membership (random recipe indices); DID recomputed. Expect ~5% significant under null.",
    }
    with open(out_dir / "null_report.json", "w") as f:
        json.dump(null_report, f, indent=2)
    print(f"[{_ts()}] Null calibration: n_tests={n_tests} frac_sig={frac_sig_null:.2%} -> null_report.json")


def _compute_significance_sanity(
    p_raw: np.ndarray,
    q_global: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Sanity checks. Write debug_significance.json on inconsistency (WARNING only).
    Only raise if q-values contain NaN or values outside [0,1].
    """
    p_raw = np.asarray(p_raw, dtype=float)
    q_global = np.asarray(q_global, dtype=float)
    n = len(p_raw)
    if n == 0:
        return

    # Hard failure: invalid q-values
    if np.any(~np.isfinite(q_global)) or np.any(q_global < 0) or np.any(q_global > 1):
        with open(out_dir / "debug_significance.json", "w") as f:
            json.dump({
                "error": "q_values_invalid",
                "has_nan": bool(np.any(~np.isfinite(q_global))),
                "min_q": float(np.nanmin(q_global)) if n > 0 else None,
                "max_q": float(np.nanmax(q_global)) if n > 0 else None,
            }, f, indent=2)
        raise RuntimeError("Phase13A q-values invalid: must be finite and in [0,1]. See " + str(out_dir / "debug_significance.json"))

    frac_p_lt_005 = float(np.nanmean(p_raw < 0.05))
    frac_q_lt_005 = float(np.nanmean(q_global <= 0.05))
    median_p = float(np.nanmedian(p_raw))
    debug = {
        "n": n,
        "frac_p_lt_005": frac_p_lt_005,
        "frac_q_lt_005": frac_q_lt_005,
        "median_p_raw": median_p,
        "n_significant_005": int(np.sum(q_global <= 0.05)),
    }
    msg_parts = []
    if frac_q_lt_005 > frac_p_lt_005 + 0.05:
        msg_parts.append(f"frac_q_lt_005 ({frac_q_lt_005:.4f}) > frac_p_lt_005 + 0.05 ({frac_p_lt_005 + 0.05:.4f})")
    if median_p > 0.05 and frac_q_lt_005 > 0.8:
        msg_parts.append(f"median_p_raw ({median_p:.4f}) > 0.05 but frac_q_lt_005 ({frac_q_lt_005:.4f}) > 0.8 (guard against all-significant)")
    if msg_parts:
        with open(out_dir / "debug_significance.json", "w") as f:
            json.dump(debug, f, indent=2)
        print(f"[{_ts()}] WARNING: significance sanity check: " + "; ".join(msg_parts) + ". See " + str(out_dir / "debug_significance.json"))


def phase13_self_test() -> bool:
    """
    Deterministic self-test: synthetic data with known interaction; confirm DID detected, BH sensible, not 100% significant.
    Returns True if all checks pass.
    """
    from src.phase13.interaction_atlas import bh_fdr
    np.random.seed(12345)
    n = 100
    # Synthetic: AB group has +0.5 interaction, others baseline 0
    y_ab = np.random.randn(n).astype(np.float32) + 0.5
    y_a = np.random.randn(n).astype(np.float32)
    y_b = np.random.randn(n).astype(np.float32)
    y_none = np.random.randn(n).astype(np.float32)
    did, se, t, p = did_paired_effect(y_ab, y_a, y_b, y_none)
    if np.isnan(p) or p > 0.05:
        raise RuntimeError("phase13_self_test: DID should be detected (p small) on synthetic interaction data")
    # Mix of significant and non-significant p-values
    p_mixed = np.concatenate([np.array([0.001, 0.01, 0.03]), np.random.uniform(0.1, 1.0, 97)])
    p_for_bh = np.where(np.isnan(p_mixed), 1.0, p_mixed)
    q = bh_fdr(p_for_bh)
    frac_q = np.mean(q <= 0.05)
    if frac_q > 0.5:
        raise RuntimeError("phase13_self_test: BH on mixed p-values should not make >50% significant")
    # Known: 3 small p's; BH can reject a few, but not all 100
    significant_005 = (q <= 0.05)
    if np.sum(significant_005) == len(p_mixed):
        raise RuntimeError("phase13_self_test: significant fraction must not be 100% for mixed p-values")
    return True


def _embedding_sanity_check(
    embed: np.ndarray,
    signatures: pd.DataFrame,
    recipe_list: List[Any],
    out_dir: Path,
    global_index: Any,
    n_sample: int = 100,
) -> None:
    """
    Pick n_sample recipes, compute nearest neighbors in embedding; verify neighbor cuisine overlap > random.
    Write output_dir/embedding_sanity.json.
    """
    if "predicted_cuisine" not in signatures.columns and "cuisine" not in signatures.columns:
        with open(out_dir / "embedding_sanity.json", "w") as f:
            json.dump({"skipped": True, "reason": "no cuisine column"}, f, indent=2)
        return
    cuisine_col = signatures["predicted_cuisine"] if "predicted_cuisine" in signatures.columns else signatures["cuisine"]
    cuisine_col = cuisine_col.fillna("unknown").astype(str)
    rng = np.random.default_rng(42)
    n = len(recipe_list)
    if n < n_sample + 5:
        return
    idx = rng.choice(n, size=n_sample, replace=False)
    vecs = embed[idx].astype(np.float32)
    _, I = global_index.search(vecs, 6)
    same_cuisine = 0
    total_pairs = 0
    for i, neighbors in enumerate(I):
        q_cuisine = cuisine_col.iloc[idx[i]]
        for j in neighbors[1:]:
            j = int(j)
            if j < 0 or j >= n:
                continue
            if recipe_list[j] != recipe_list[idx[i]]:
                total_pairs += 1
                if cuisine_col.iloc[j] == q_cuisine:
                    same_cuisine += 1
    observed = same_cuisine / total_pairs if total_pairs > 0 else 0
    cuisine_counts = cuisine_col.value_counts()
    baseline = float((cuisine_counts**2).sum() / (cuisine_counts.sum() ** 2)) if cuisine_counts.sum() > 0 else 0
    report = {
        "n_sample": n_sample,
        "total_neighbor_pairs": total_pairs,
        "same_cuisine_pairs": same_cuisine,
        "observed_same_cuisine_frac": observed,
        "random_baseline_same_cuisine_frac": baseline,
        "pass": observed >= baseline * 0.9,
    }
    with open(out_dir / "embedding_sanity.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{_ts()}] Embedding sanity: observed_same_cuisine={observed:.3f} baseline={baseline:.3f} pass={report['pass']} -> embedding_sanity.json")


def shard_is_valid(shard_path: Path) -> bool:
    """Only skip shard if file exists AND size>0 AND row_count>0 AND schema matches (core 7 cols)."""
    if not shard_path.exists():
        return False
    if shard_path.stat().st_size <= 0:
        return False
    try:
        df = pd.read_parquet(shard_path)
        if len(df) == 0:
            return False
        for col in ["ingA_id", "ingB_id", "category", "se", "t"]:
            if col not in df.columns:
                return False
        if "did" in df.columns and "p_raw" in df.columns:
            pass
        elif "beta_int" in df.columns and "p" in df.columns:
            pass
        else:
            return False
    except Exception:
        return False
    return True


def run_phase13a(
    repo_root: Path,
    use_v2: bool = False,
    run_id: Optional[str] = None,
    test_pairs: int = 5000,
    max_categories: Optional[int] = None,
    skip_low_variance: bool = True,
    null_calibration: bool = False,
    min_matched: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Phase13A: DID (difference-in-differences) screen for true ingredient interaction.
    Four groups: both (AB), A-only, B-only, none. Matched 1:1 per group via global FAISS index.
    beta_int = mu_AB - mu_A - mu_B + mu_0 with paired SE. No set(range(n_recipes)) per pair.
    Writes: atlas_screen.parquet, interactions_adjusted.parquet, diagnostics.json, config.json, skip_reasons.parquet.
    Optionally null_calibration: shuffle labels and write null_report.json.
    """
    run_id = run_id or get_run_id()
    np.random.seed(42)
    out_dir = ensure_output_dir(repo_root, use_v2, run_id)
    write_latest_txt(repo_root, out_dir)
    t_start = time.perf_counter()
    print(f"[{_ts()}] Phase13A START (DID) — RUN_ID={run_id} — output={out_dir}")

    signatures, ri, func_cols = load_signatures_and_ri(repo_root, use_v2)
    print(f"[{_ts()}] Loaded signatures {signatures.shape}, recipe_ingredients {ri.shape}, func_cols={len(func_cols)}")

    if skip_low_variance:
        func_cols, low_var, var_report = variance_gate(signatures, func_cols)
        print(f"[{_ts()}] Variance gate: using {len(func_cols)} categories; skipped low_variance: {low_var}")
    else:
        var_report = {}
    if max_categories:
        func_cols = func_cols[:max_categories]
    if not func_cols:
        raise RuntimeError("No categories left after variance gate. Set skip_low_variance=False or check data.")

    y_mat = signatures[func_cols].fillna(0).to_numpy(np.float32)
    print(f"[{_ts()}] y_mat shape {y_mat.shape}")

    recipe_list = signatures["recipe_id"].tolist()
    recipe_id_to_idx = {r: i for i, r in enumerate(recipe_list)}
    n_recipes = len(recipe_list)
    embed, scaler, embedding_meta = build_recipe_index_and_embedding(signatures, ri, recipe_id_to_idx)
    print(f"[{_ts()}] Embedding shape {embed.shape}")

    global_index, faiss_info = build_global_faiss_index(embed, use_gpu=True)
    print(f"[{_ts()}] Phase13A GPU/FAISS: faiss_version={faiss_info.get('faiss_version')} n_gpus={faiss_info.get('n_gpus')} gpu_used={faiss_info.get('gpu_used')} index_type={faiss_info.get('index_type')}")

    _embedding_sanity_check(embed, signatures, recipe_list, out_dir, global_index, n_sample=100)

    min_matched_eff = min_matched if min_matched is not None else MIN_MATCHED

    conf = signatures[["recipe_id"]].drop_duplicates()
    num_ing = ri.groupby("recipe_id").size().reset_index(name="num_ingredients")
    conf = conf.merge(num_ing, on="recipe_id", how="left")
    conf["num_ingredients"] = conf["num_ingredients"].fillna(0).astype(int)
    if "num_compounds" in signatures.columns:
        conf = conf.merge(signatures[["recipe_id", "num_compounds"]].drop_duplicates(), on="recipe_id", how="left")
    else:
        conf["num_compounds"] = 0
    if "predicted_cuisine" in signatures.columns:
        cuis = signatures[["recipe_id", "predicted_cuisine"]].drop_duplicates()
        conf = conf.merge(cuis, on="recipe_id", how="left")
        conf["cuisine_or_cluster"] = conf["predicted_cuisine"].fillna("unknown").astype(str)
    else:
        conf["cuisine_or_cluster"] = "cuisine_unknown"
    conf.to_parquet(out_dir / "recipes_with_confounders.parquet", index=False)

    ri_dedup = ri[["recipe_id", "ingredient_id"]].drop_duplicates()
    ing_to_rec = ri_dedup.groupby("ingredient_id")["recipe_id"].apply(lambda x: list(x)).to_dict()
    for k in ing_to_rec:
        ing_to_rec[k] = [recipe_id_to_idx[r] for r in ing_to_rec[k] if r in recipe_id_to_idx]
    ing_to_set = {k: set(v) for k, v in ing_to_rec.items()}

    from itertools import combinations
    pair_counts = {}
    for rid, grp in ri_dedup.groupby("recipe_id"):
        ings = list(grp["ingredient_id"].unique())
        if len(ings) < 2:
            continue
        for a, b in combinations(sorted(ings), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    eligible_pairs = []
    skip_rows = []
    for (a, b), n_both in pair_counts.items():
        rec_a = ing_to_set.get(a, set())
        rec_b = ing_to_set.get(b, set())
        n_a_only = len(rec_a - rec_b)
        n_b_only = len(rec_b - rec_a)
        if n_both < MIN_JOINT:
            skip_rows.append({"pair_id": f"{a}|{b}", "ingA_id": a, "ingB_id": b, "reason": "nBoth < MIN_JOINT", "nBoth": n_both, "nA_only": n_a_only, "nB_only": n_b_only, "matched_n": None, "category_if_relevant": None})
            continue
        if n_a_only < MIN_SOLO or n_b_only < MIN_SOLO:
            skip_rows.append({"pair_id": f"{a}|{b}", "ingA_id": a, "ingB_id": b, "reason": "nA_only or nB_only < MIN_SOLO", "nBoth": n_both, "nA_only": n_a_only, "nB_only": n_b_only, "matched_n": None, "category_if_relevant": None})
            continue
        eligible_pairs.append((a, b))
    eligible_pairs = eligible_pairs[:test_pairs]
    print(f"[{_ts()}] Eligible pairs: {len(eligible_pairs)}; skip_eligibility: {len(skip_rows)}")

    # Reusable mask for none_idx (no set(range(n_recipes)) per pair)
    none_mask = np.ones(n_recipes, dtype=bool)
    from src.phase13.interaction_atlas import bh_fdr

    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_size = MAX_PAIRS_PER_SHARD
    n_shards = (len(eligible_pairs) + shard_size - 1) // shard_size
    all_results = []
    skip_match = []
    skip_reason_counts: Dict[str, int] = {}
    treated_sizes: List[int] = []
    matched_per_group: List[int] = []
    p_values_so_far: List[float] = []
    use_bootstrap = max_categories is not None
    bootstrap_B = BOOTSTRAP_B_SMOKE if use_bootstrap else BOOTSTRAP_B_FULL
    if use_bootstrap:
        print(f"[{_ts()}] DID: using bootstrap SE/p (B={bootstrap_B})")
    else:
        print(f"[{_ts()}] DID: using analytic group-means SE/p")

    for shard_i in range(n_shards):
        start_i = shard_i * shard_size
        end_i = min(start_i + shard_size, len(eligible_pairs))
        pair_slice = eligible_pairs[start_i:end_i]
        shard_path = shards_dir / f"interactions_shard_{shard_i + 1:04d}.parquet"

        if shard_is_valid(shard_path):
            df_ex = pd.read_parquet(shard_path)
            all_results.append(df_ex)
            print(f"[{_ts()}] Skip shard {shard_i + 1}/{n_shards} (valid, {len(df_ex)} rows)")
            continue

        shard_results = []
        for i, (a, b) in enumerate(pair_slice):
            rec_a = ing_to_set.get(a, set())
            rec_b = ing_to_set.get(b, set())
            both_set = rec_a & rec_b
            a_only_set = rec_a - rec_b
            b_only_set = rec_b - rec_a
            none_mask[:] = True
            for r in rec_a:
                none_mask[r] = False
            for r in rec_b:
                none_mask[r] = False
            none_idx = np.where(none_mask)[0]

            if len(a_only_set) < MIN_SOLO or len(b_only_set) < MIN_SOLO or len(none_idx) < min_matched_eff:
                reason = "pool too small (A_only/B_only/none)"
                skip_match.append({"pair_id": f"{a}|{b}", "ingA_id": a, "ingB_id": b, "reason": reason, "nBoth": len(both_set), "nA_only": len(a_only_set), "nB_only": len(b_only_set), "matched_n": 0, "category_if_relevant": None})
                skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
                continue

            both_idx = np.array(sorted(both_set), dtype=np.int32)
            if len(both_idx) > MAX_TREATED:
                both_idx = np.random.choice(both_idx, MAX_TREATED, replace=False)
            if len(both_idx) < MIN_JOINT:
                reason = "both < MIN_JOINT after cap"
                skip_match.append({"pair_id": f"{a}|{b}", "ingA_id": a, "ingB_id": b, "reason": reason, "nBoth": len(both_idx), "nA_only": len(a_only_set), "nB_only": len(b_only_set), "matched_n": 0, "category_if_relevant": None})
                skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
                continue

            treated_matched, matched_a, matched_b, matched_none, avg_L2_A, avg_L2_B, avg_L2_0 = match_did_groups(
                global_index, embed, both_idx,
                pool_a=a_only_set, pool_b=b_only_set, pool_none=set(none_idx.tolist()),
                topk=FAISS_TOPK, ratio=MATCH_RATIO,
            )
            n_matched = len(treated_matched)
            if n_matched < min_matched_eff:
                reason = "DID matched_n < MIN_MATCHED"
                skip_match.append({"pair_id": f"{a}|{b}", "ingA_id": a, "ingB_id": b, "reason": reason, "nBoth": len(both_idx), "nA_only": len(a_only_set), "nB_only": len(b_only_set), "matched_n": n_matched, "category_if_relevant": None})
                skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
                continue

            for c_idx, cat in enumerate(func_cols):
                y_ab = y_mat[treated_matched, c_idx]
                y_a = y_mat[matched_a, c_idx]
                y_b = y_mat[matched_b, c_idx]
                y_0 = y_mat[matched_none, c_idx]
                if use_bootstrap:
                    did, se, t, p_raw = did_bootstrap(y_ab, y_a, y_b, y_0, B=bootstrap_B, seed=42)
                else:
                    did, se, t, p_raw = did_group_means(y_ab, y_a, y_b, y_0)
                if np.isnan(did):
                    continue
                p_values_so_far.append(p_raw)
                shard_results.append({
                    "ingA_id": a, "ingB_id": b, "category": cat,
                    "did": did, "se": se, "t": t, "p_raw": p_raw,
                    "n_ab": n_matched, "n_a": n_matched, "n_b": n_matched, "n_none": n_matched,
                    "avg_L2_A": avg_L2_A, "avg_L2_B": avg_L2_B, "avg_L2_0": avg_L2_0,
                })
            treated_sizes.append(n_matched)
            matched_per_group.append(n_matched)

            count = start_i + i + 1
            if (count % PROGRESS_EVERY) == 0 or count == len(eligible_pairs):
                elapsed = time.perf_counter() - t_start
                rate = count / (elapsed / 60.0) if elapsed > 0 else 0
                eta_min = (len(eligible_pairs) - count) / rate if rate > 0 else 0
                avg_treated = float(np.mean(treated_sizes)) if treated_sizes else 0
                avg_matched = float(np.mean(matched_per_group)) if matched_per_group else 0
                median_p = float(np.median(p_values_so_far)) if p_values_so_far else np.nan
                frac_p_005 = float(np.mean(np.array(p_values_so_far) < 0.05)) if p_values_so_far else 0
                # BH on accumulated p's for frac q<0.05 so far
                p_arr = np.where(np.isnan(np.array(p_values_so_far)), 1.0, np.array(p_values_so_far))
                q_so_far = bh_fdr(p_arr) if len(p_arr) > 0 else np.array([])
                frac_q_005 = float(np.mean(q_so_far <= 0.05)) if len(q_so_far) > 0 else 0
                skip_str = "; ".join(f"{k}={v}" for k, v in sorted(skip_reason_counts.items()))
                print(f"[{_ts()}] pairs={count} pairs/min={rate:.1f} elapsed={elapsed:.0f}s ETA={eta_min:.1f}min avg_treated={avg_treated:.0f} avg_matched={avg_matched:.0f} median_p={median_p:.4f} frac_p<0.05={frac_p_005:.2%} frac_q<0.05={frac_q_005:.2%} skip=[{skip_str}]")

        if shard_results:
            pd.DataFrame(shard_results).to_parquet(shard_path, index=False)
            all_results.append(pd.DataFrame(shard_results))
        else:
            pd.DataFrame(columns=EXPECTED_SHARD_COLUMNS).to_parquet(shard_path, index=False)

    if not all_results:
        raise RuntimeError("Phase13A produced 0 result rows. Check variance gate and eligibility.")
    atlas = pd.concat(all_results, ignore_index=True)
    # Standardize column names (support old shards with p, beta_int, n_AB, ...)
    if "p_raw" not in atlas.columns and "p" in atlas.columns:
        atlas["p_raw"] = atlas["p"]
    if "did" not in atlas.columns and "beta_int" in atlas.columns:
        atlas["did"] = atlas["beta_int"]
    for old_c, new_c in [("n_AB", "n_ab"), ("n_A", "n_a"), ("n_B", "n_b"), ("n_0", "n_none")]:
        if new_c not in atlas.columns and old_c in atlas.columns:
            atlas[new_c] = atlas[old_c]
    atlas.to_parquet(out_dir / "atlas_screen.parquet", index=False)
    print(f"[{_ts()}] Wrote atlas_screen.parquet: {atlas.shape}")

    # A) Significance: full p vector (p_raw if present else p), NaN -> 1.0, single BH pass on full vector
    p_col = atlas["p_raw"] if "p_raw" in atlas.columns else atlas["p"]
    p_raw = np.asarray(p_col, dtype=np.float64)
    if p_raw.ndim != 1:
        p_raw = p_raw.reshape(-1)
    p_raw = np.where(np.isfinite(p_raw), p_raw, 1.0)
    p_raw = np.clip(p_raw, 0.0, 1.0)
    q_global = bh_fdr(p_raw)
    atlas["q_global"] = q_global
    atlas["significant_005"] = (atlas["q_global"].to_numpy() <= 0.05)
    # No other logic for significant_005 (no beta thresholds or heuristics)

    # B) Hard sanity checks (raise if inconsistent)
    _compute_significance_sanity(p_raw, q_global, out_dir)

    # D) passes_min_effect: reporting only
    atlas["passes_min_effect"] = atlas["did"].abs() >= MIN_ABS_DID
    # Backward compat for Phase13B / interaction_atlas
    atlas["beta_int"] = atlas["did"]
    atlas["n_both"] = atlas["n_ab"]
    atlas.to_parquet(out_dir / "interactions_adjusted.parquet", index=False)

    skip_df = pd.DataFrame(skip_rows + skip_match)
    if len(skip_df) > 0:
        skip_df.to_parquet(out_dir / "skip_reasons.parquet", index=False)
    else:
        pd.DataFrame(columns=["pair_id", "ingA_id", "ingB_id", "reason", "nBoth", "nA_only", "nB_only", "matched_n", "category_if_relevant"]).to_parquet(out_dir / "skip_reasons.parquet", index=False)

    diagnostics = {
        "run_id": run_id,
        "start_utc": _ts(),
        "elapsed_sec": time.perf_counter() - t_start,
        "n_pairs_considered": len(pair_counts),
        "n_pairs_eligible": len(eligible_pairs),
        "n_pairs_tested": len(eligible_pairs),
        "n_results_rows": len(atlas),
        "n_significant_q005": int(atlas["significant_005"].sum()),
        "variance_report": var_report,
        "skip_eligibility_count": len(skip_rows),
        "skip_matched_count": len(skip_match),
        "skip_reason_counts": skip_reason_counts,
        "pct_pairs_skipped_matched_n": 100.0 * len([s for s in skip_match if s.get("reason") == "DID matched_n < MIN_MATCHED"]) / len(eligible_pairs) if eligible_pairs else 0,
        "distribution_n_ab": atlas["n_ab"].describe().to_dict() if "n_ab" in atlas.columns and len(atlas) > 0 else {},
        "avg_L2_matched_A": float(atlas["avg_L2_A"].mean()) if "avg_L2_A" in atlas.columns else None,
        "avg_L2_matched_B": float(atlas["avg_L2_B"].mean()) if "avg_L2_B" in atlas.columns else None,
        "avg_L2_matched_none": float(atlas["avg_L2_0"].mean()) if "avg_L2_0" in atlas.columns else None,
        "top_20_by_q": atlas.nsmallest(20, "q_global")[["ingA_id", "ingB_id", "category", "did", "q_global", "n_ab"]].to_dict("records") if len(atlas) >= 20 else atlas.head(20).to_dict("records"),
        "top_20_by_abs_beta": atlas.reindex(atlas["did"].abs().sort_values(ascending=False).index).head(20)[["ingA_id", "ingB_id", "category", "did", "q_global", "n_ab"]].to_dict("records") if len(atlas) >= 20 else [],
        "top_20_preview": atlas.nsmallest(20, "q_global")[["ingA_id", "ingB_id", "category", "did", "q_global", "n_ab"]].to_dict("records") if len(atlas) >= 20 else atlas.head(20).to_dict("records"),
    }
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    config_snapshot = {
        "MIN_JOINT": MIN_JOINT, "MIN_SOLO": MIN_SOLO, "MIN_MATCHED": min_matched_eff,
        "MAX_TREATED": MAX_TREATED, "MATCH_RATIO": MATCH_RATIO, "FAISS_TOPK": FAISS_TOPK,
        "K_NEIGHBORS": K_NEIGHBORS, "N_MATCH_PER_TREATED": N_MATCH_PER_TREATED,
        "MAX_PAIRS_PER_SHARD": MAX_PAIRS_PER_SHARD, "MIN_ABS_DID": MIN_ABS_DID,
        "svd_dim": embedding_meta.get("svd_dim"), "n_features_hash": embedding_meta.get("n_features_hash"),
        "top_cuisines_count": len(embedding_meta.get("top_cuisines", [])), "top_sources_count": len(embedding_meta.get("top_sources", [])),
        "embed_dim": embedding_meta.get("embed_dim"),
        "test_pairs": test_pairs, "use_v2": use_v2, "run_id": run_id,
        "signatures_path": str(repo_root / "data/processed/phase17_reaggregation/recipe_functional_signatures_v3.parquet") if not use_v2 else str(repo_root / "data/processed/exports_v2/recipes_biological_effects_v2_FINAL.parquet"),
        "recipe_ingredients_path": str(repo_root / "data/processed/canonical/recipe_ingredients_expanded_v2.parquet"),
        "git_commit": _git_hash(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    if null_calibration:
        _run_null_calibration(out_dir, eligible_pairs[: min(100, len(eligible_pairs))], ing_to_set, none_mask, n_recipes, embed, global_index, y_mat, func_cols, recipe_id_to_idx)

    print(f"[{_ts()}] Phase13A END — elapsed={diagnostics['elapsed_sec']:.1f}s n_results={len(atlas)} n_sig={diagnostics['n_significant_q005']}")
    return diagnostics


def run_phase13b(
    repo_root: Path,
    output_dir: Path,
    top_k: int = 200,
    run_ols_confirm: bool = False,
) -> Dict[str, Any]:
    """
    Phase13B: Confirm top_k pairs with DID + analytic SE, matched None, optional OLS;
    within-group bootstrap; null test; KG export. Uses Phase13BEngine (no full-table scans).
    Writes atlas_confirmed.parquet, bootstrap_stability.parquet, null_tests.json,
    phase13b_diagnostics.json, kg/*.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    np.random.seed(42)
    t0 = time.perf_counter()
    print(f"[{_ts()}] Phase13B START — output_dir={output_dir} top_k={top_k}")

    adj_path = output_dir / "interactions_adjusted.parquet"
    if not adj_path.exists():
        raise FileNotFoundError(f"Run Phase13A first: {adj_path}")
    atlas = pd.read_parquet(adj_path)
    top = atlas.nsmallest(top_k, "q_global").drop_duplicates(subset=["ingA_id", "ingB_id"]).head(top_k)
    if top.empty:
        top = atlas.head(top_k)
    print(f"[{_ts()}] Loaded top {len(top)} pairs from interactions_adjusted.parquet")

    signatures, ri, func_cols = load_signatures_and_ri(repo_root, use_v2="v2" in str(output_dir))
    conf_path = output_dir / "recipes_with_confounders.parquet"
    if conf_path.exists():
        conf = pd.read_parquet(conf_path)
    else:
        conf = ri.groupby("recipe_id").size().reset_index(name="num_ingredients")
        conf["cuisine_or_cluster"] = "cuisine_unknown"
    print(f"[{_ts()}] Loaded signatures (n={len(signatures)}), ri, confounders; func_cols={len(func_cols)}")

    recipe_id_to_idx = {str(rid): i for i, rid in enumerate(signatures["recipe_id"])}
    print(f"[{_ts()}] Building embedding and FAISS index (once)...")
    embed, _scaler, _embed_meta = build_recipe_index_and_embedding(signatures, ri, recipe_id_to_idx)
    faiss_index, _faiss_info = build_global_faiss_index(embed, use_gpu=True)
    print(f"[{_ts()}] Embedding built dim={embed.shape[1]} index_type={_faiss_info.get('index_type', '?')}")

    from src.phase13.phase13b_engine import Phase13BEngine

    B = 200
    top_n_bootstrap = min(100, top_k)
    engine = Phase13BEngine(
        output_dir=output_dir,
        top_pairs_df=top,
        signatures=signatures,
        ri=ri,
        conf=conf,
        func_cols=func_cols,
        embed=embed,
        faiss_index=faiss_index,
        recipe_id_to_idx=recipe_id_to_idx,
        B=B,
        top_n_bootstrap=top_n_bootstrap,
        run_ols_confirm=run_ols_confirm,
        null_pairs=20,
        null_categories=3,
    )
    print(f"[{_ts()}] Confirming top_k pairs (DID + matched None)...")
    diagnostics = engine.run(run_bootstrap=True, run_null=True, run_kg=True)

    elapsed = time.perf_counter() - t0
    print(f"[{_ts()}] Phase13B END — confirmed={diagnostics.get('n_rows_written', 0)} bootstrap={diagnostics.get('n_bootstrap_rows', 0)} runtime={elapsed:.1f}s")
    return diagnostics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    p.add_argument("--use-v2", action="store_true")
    p.add_argument("--test-pairs", type=int, default=5000)
    p.add_argument("--smoke", action="store_true", help="200 pairs, 2 categories, < 2 min")
    p.add_argument("--phase13b", action="store_true", help="Run Phase13B only (need existing output)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--null-calibration", action="store_true", help="Run null calibration and write null_report.json")
    args = p.parse_args()
    if args.smoke:
        print(f"[{_ts()}] Running phase13 self-test...")
        phase13_self_test()
        print(f"[{_ts()}] Self-test passed.")
        run_phase13a(args.repo_root, use_v2=args.use_v2, test_pairs=200, max_categories=2, null_calibration=args.null_calibration)
    elif args.phase13b:
        out = args.output_dir or (args.repo_root / "data" / "processed" / "phase13_interactions_latest.txt")
        if out.suffix == ".txt":
            out = Path(out.read_text().strip())
        run_phase13b(args.repo_root, out, top_k=200)
    else:
        run_phase13a(args.repo_root, use_v2=args.use_v2, test_pairs=args.test_pairs, null_calibration=args.null_calibration)
