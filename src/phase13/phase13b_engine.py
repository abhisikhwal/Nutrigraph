"""
Phase13B confirmation engine: cached indices, DID + analytic SE, matched None, optional OLS,
within-group bootstrap, null test, KG export. No full-table scans.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from src.utils.parquet_portable import repack_parquet_file
except ImportError:
    repack_parquet_file = None

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

logger = logging.getLogger(__name__)


def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# DID and matching (self-contained to avoid circular imports)
# ---------------------------------------------------------------------------

def _did_group_means(
    y_ab: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    y_none: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    DID = (mean(AB) - mean(A_only)) - (mean(B_only) - mean(None)).
    Analytic SE from group variances. Returns (did, se, t, p).
    """
    if scipy_stats is None:
        return np.nan, np.nan, np.nan, np.nan
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
    df = max(1, min(n_ab, n_a, n_b, n_n) - 1)
    p = 2 * (1 - scipy_stats.t.cdf(abs(t), df))
    return did, se, float(t), float(p)


def _match_none_for_pair(
    index: Any,
    embed: np.ndarray,
    ab_idx: np.ndarray,
    pool_none: Set[int],
    topk: int = 50,
) -> np.ndarray:
    """
    For each AB recipe, find one NN in pool_none (no replacement).
    Returns matched_none indices (length <= len(ab_idx)).
    """
    if len(ab_idx) == 0 or len(pool_none) == 0:
        return np.array([], dtype=np.int32)
    treated_vecs = embed[ab_idx].astype(np.float32)
    D, I = index.search(treated_vecs, min(topk, index.ntotal))
    used_none: Set[int] = set()
    matched: List[int] = []
    for i_row in range(I.shape[0]):
        for cand in I[i_row]:
            cand = int(cand)
            if cand in pool_none and cand not in used_none:
                used_none.add(cand)
                matched.append(cand)
                break
    return np.array(matched, dtype=np.int32)


def _match_did_groups(
    index: Any,
    embed: np.ndarray,
    ab_idx: np.ndarray,
    pool_a: Set[int],
    pool_b: Set[int],
    pool_none: Set[int],
    topk: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1:1 NN matching: for each AB recipe get one from A_only, B_only, None.
    Returns (treated_matched, matched_a, matched_b, matched_none) of same length.
    """
    if len(ab_idx) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    treated_vecs = embed[ab_idx].astype(np.float32)
    D, I = index.search(treated_vecs, min(topk, index.ntotal))
    used_a: Set[int] = set()
    used_b: Set[int] = set()
    used_none: Set[int] = set()
    treated_out: List[int] = []
    matched_a: List[int] = []
    matched_b: List[int] = []
    matched_none: List[int] = []
    for row_idx in range(I.shape[0]):
        pick_a = pick_b = pick_none = None
        for cand in I[row_idx]:
            cand = int(cand)
            if pick_a is None and cand in pool_a and cand not in used_a:
                pick_a = cand
            if pick_b is None and cand in pool_b and cand not in used_b:
                pick_b = cand
            if pick_none is None and cand in pool_none and cand not in used_none:
                pick_none = cand
            if pick_a is not None and pick_b is not None and pick_none is not None:
                break
        if pick_a is not None and pick_b is not None and pick_none is not None:
            treated_out.append(int(ab_idx[row_idx]))
            used_a.add(pick_a)
            used_b.add(pick_b)
            used_none.add(pick_none)
            matched_a.append(pick_a)
            matched_b.append(pick_b)
            matched_none.append(pick_none)
    return (
        np.array(treated_out, dtype=np.int32),
        np.array(matched_a, dtype=np.int32),
        np.array(matched_b, dtype=np.int32),
        np.array(matched_none, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Bootstrap within groups (resample each group with replacement, then DID)
# ---------------------------------------------------------------------------

def _bootstrap_did_within_groups(
    y_ab: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    y_none: np.ndarray,
    B: int = 200,
    seed: int = 42,
    stability_threshold: float = 0.01,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Resample within each group with replacement; compute DID per replicate.
    Returns (did_obs, did_mean, did_std, p_boot, ci_low, ci_high, stability_score).
    """
    rng = np.random.default_rng(seed)
    n_ab, n_a, n_b, n_n = len(y_ab), len(y_a), len(y_b), len(y_none)
    if n_ab < 2 or n_a < 2 or n_b < 2 or n_n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0
    did_obs = (float(np.mean(y_ab)) - float(np.mean(y_a))) - (float(np.mean(y_b)) - float(np.mean(y_none)))
    did_boot: List[float] = []
    for _ in range(B):
        i_ab = rng.integers(0, n_ab, size=n_ab)
        i_a = rng.integers(0, n_a, size=n_a)
        i_b = rng.integers(0, n_b, size=n_b)
        i_n = rng.integers(0, n_n, size=n_n)
        m_ab = np.mean(y_ab[i_ab])
        m_a = np.mean(y_a[i_a])
        m_b = np.mean(y_b[i_b])
        m_n = np.mean(y_none[i_n])
        did_boot.append((m_ab - m_a) - (m_b - m_n))
    did_boot = np.array(did_boot)
    did_mean = float(np.mean(did_boot))
    did_std = float(np.std(did_boot, ddof=1))
    p_boot = float(np.mean(np.abs(did_boot) >= np.abs(did_obs)))
    p_boot = max(p_boot, 1.0 / (B + 1))
    ci_low = float(np.percentile(did_boot, 2.5))
    ci_high = float(np.percentile(did_boot, 97.5))
    stability = _stability_score(did_boot, did_obs, threshold=stability_threshold)
    return did_obs, did_mean, did_std, p_boot, ci_low, ci_high, stability


def _stability_score(did_boot: np.ndarray, did_obs: float, threshold: float = 0.01) -> float:
    """Fraction of bootstrap samples with same sign as did_obs and |did| >= threshold."""
    if len(did_boot) == 0:
        return 0.0
    same_sign = (did_boot >= threshold) if did_obs >= 0 else (did_boot <= -threshold)
    return float(np.mean(same_sign))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Phase13BEngine:
    """
    Phase13B: confirm top pairs with DID + analytic SE, matched None, optional OLS;
    within-group bootstrap; null test; KG export.
    """

    def __init__(
        self,
        output_dir: Path,
        top_pairs_df: pd.DataFrame,
        signatures: pd.DataFrame,
        ri: pd.DataFrame,
        conf: pd.DataFrame,
        func_cols: List[str],
        embed: np.ndarray,
        faiss_index: Any,
        recipe_id_to_idx: Dict[str, int],
        *,
        B: int = 200,
        top_n_bootstrap: Optional[int] = None,
        n_jobs: int = None,
        run_ols_confirm: bool = False,
        null_pairs: int = 20,
        null_categories: int = 3,
        faiss_topk: int = 50,
    ):
        self.output_dir = Path(output_dir)
        self.top_pairs_df = top_pairs_df
        self.signatures = signatures
        self.ri = ri
        self.conf = conf
        self.func_cols = [c for c in func_cols if c in signatures.columns]
        self.embed = embed
        self.faiss_index = faiss_index
        self.recipe_id_to_idx = recipe_id_to_idx
        self.B = B
        self.top_n_bootstrap = top_n_bootstrap  # None => use all confirmed pairs for bootstrap
        self.n_jobs = max(1, (os.cpu_count() or 2) - 2) if n_jobs is None else n_jobs
        self.run_ols_confirm = run_ols_confirm and (sm is not None)
        self.null_pairs = null_pairs
        self.null_categories = null_categories
        self.faiss_topk = faiss_topk

        self.n_recipes = len(signatures)
        self.y_matrix: np.ndarray = signatures[self.func_cols].to_numpy(dtype=np.float32)
        self._ing_to_recipe_idx: Dict[str, np.ndarray] = {}
        self._built = False

    def _build_indices(self) -> None:
        if self._built:
            return
        t0 = time.perf_counter()
        logger.info("[%s] Building ingredient -> recipe index...", _ts())
        ri = self.ri[["recipe_id", "ingredient_id"]].copy()
        ri["recipe_id"] = ri["recipe_id"].astype(str)
        ri["ingredient_id"] = ri["ingredient_id"].astype(str)
        ri["recipe_idx"] = ri["recipe_id"].map(self.recipe_id_to_idx)
        ri = ri.dropna(subset=["recipe_idx"])
        ri["recipe_idx"] = ri["recipe_idx"].astype(np.int32)
        grp = ri.groupby("ingredient_id")["recipe_idx"].apply(lambda x: np.sort(x.unique().astype(np.int32)))
        self._ing_to_recipe_idx = grp.to_dict()
        self._built = True
        logger.info("[%s] Ingredient index built: %d ingredients in %.1fs", _ts(), len(self._ing_to_recipe_idx), time.perf_counter() - t0)

    def _get_groups(self, ing_a: str, ing_b: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Set[int]]:
        """Return (ab_idx, a_only_idx, b_only_idx, pool_none_set)."""
        set_a = set(self._ing_to_recipe_idx.get(ing_a, np.array([], dtype=np.int32)).tolist())
        set_b = set(self._ing_to_recipe_idx.get(ing_b, np.array([], dtype=np.int32)).tolist())
        ab = np.array(sorted(set_a & set_b), dtype=np.int32)
        a_only = np.array(sorted(set_a - set_b), dtype=np.int32)
        b_only = np.array(sorted(set_b - set_a), dtype=np.int32)
        pool_none = set(range(self.n_recipes)) - set_a - set_b
        return ab, a_only, b_only, pool_none

    def _confirm_one_pair_category(
        self,
        ing_a: str,
        ing_b: str,
        category: str,
        cat_idx: int,
        q_global: float = np.nan,
        p_raw: float = np.nan,
    ) -> Optional[Dict[str, Any]]:
        ab_idx, a_only_idx, b_only_idx, pool_none = self._get_groups(ing_a, ing_b)
        if len(ab_idx) < 2 or len(a_only_idx) < 2 or len(b_only_idx) < 2 or len(pool_none) < 2:
            return None
        treated_matched, matched_a, matched_b, matched_none = _match_did_groups(
            self.faiss_index,
            self.embed,
            ab_idx,
            set(a_only_idx),
            set(b_only_idx),
            pool_none,
            topk=self.faiss_topk,
        )
        if len(treated_matched) < 2:
            return None
        y_ab = self.y_matrix[treated_matched, cat_idx]
        y_a = self.y_matrix[matched_a, cat_idx]
        y_b = self.y_matrix[matched_b, cat_idx]
        y_none = self.y_matrix[matched_none, cat_idx]
        did, se_analytic, t_analytic, p_analytic = _did_group_means(y_ab, y_a, y_b, y_none)
        if np.isnan(did):
            return None
        n_ab, n_a, n_b, n_none = len(y_ab), len(y_a), len(y_b), len(y_none)
        row: Dict[str, Any] = {
            "ingA_id": ing_a,
            "ingB_id": ing_b,
            "category": category,
            "did": did,
            "se_analytic": se_analytic,
            "t_analytic": t_analytic,
            "p_analytic": p_analytic,
            "n_ab": n_ab,
            "n_a": n_a,
            "n_b": n_b,
            "n_none": n_none,
            "q_global": q_global,
            "p_raw": p_raw,
        }
        if self.run_ols_confirm and sm is not None:
            ols_result = self._ols_on_subset(treated_matched, matched_a, matched_b, matched_none, ing_a, ing_b, category, cat_idx)
            if ols_result:
                row["ols_beta_int"] = ols_result.get("beta_int")
                row["ols_se"] = ols_result.get("se")
                row["ols_p"] = ols_result.get("p")
        return row

    def _ols_on_subset(
        self,
        treated_matched: np.ndarray,
        matched_a: np.ndarray,
        matched_b: np.ndarray,
        matched_none: np.ndarray,
        ing_a: str,
        ing_b: str,
        category: str,
        cat_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """OLS on subset: AB + A_only + B_only + None with A, B, A*B + confounders.
        Confounder rows are aligned to X by using the same concatenation order (no deduplication).
        """
        # Same concatenation order as for y and X (preserves duplicates)
        idx_all = np.concatenate([treated_matched, matched_a, matched_b, matched_none])
        rid_order = self.signatures["recipe_id"].astype(str).iloc[idx_all].tolist()
        y = self.y_matrix[idx_all, cat_idx]
        set_ab = set(treated_matched.tolist())
        set_a = set(treated_matched.tolist()) | set(matched_a.tolist())
        set_b = set(treated_matched.tolist()) | set(matched_b.tolist())
        A = np.array([1 if i in set_a else 0 for i in idx_all], dtype=np.float64)
        B = np.array([1 if i in set_b else 0 for i in idx_all], dtype=np.float64)
        A_B = np.array([1 if i in set_ab else 0 for i in idx_all], dtype=np.float64)
        n_rows = len(idx_all)
        X = pd.DataFrame({"A": A, "B": B, "A_B": A_B}, index=range(n_rows))

        conf_idx = self.conf.copy()
        conf_idx["recipe_id"] = conf_idx["recipe_id"].astype(str)
        conf_idx = conf_idx.set_index("recipe_id")
        conf_sub = conf_idx.reindex(rid_order).reset_index()
        if len(conf_sub) != len(X):
            raise RuntimeError(
                "Confounder length mismatch: len(conf_sub)=%d, len(X)=%d; first few recipe_ids: %s"
                % (len(conf_sub), len(X), rid_order[:5] if len(rid_order) >= 5 else rid_order)
            )

        if "num_ingredients" in conf_sub.columns:
            X["num_ingredients"] = conf_sub["num_ingredients"].fillna(0).to_numpy()
        if "num_compounds" in conf_sub.columns:
            X["num_compounds"] = conf_sub["num_compounds"].fillna(0).to_numpy()
        if "cuisine_or_cluster" in conf_sub.columns:
            dums = pd.get_dummies(conf_sub["cuisine_or_cluster"].fillna("unknown"), drop_first=True, dtype=float)
            X = pd.concat([X, dums], axis=1)
        X = sm.add_constant(X, has_constant="add")
        try:
            res = sm.OLS(y, X).fit()
        except Exception:
            return None
        if "A_B" not in res.params.index:
            return None
        return {
            "beta_int": float(res.params["A_B"]),
            "se": float(res.bse["A_B"]),
            "p": float(res.pvalues["A_B"]),
        }

    def run_confirm(self) -> pd.DataFrame:
        """Run DID confirm for all top pairs and categories. Returns atlas_confirmed rows."""
        self._build_indices()
        pairs_list = list(zip(self.top_pairs_df["ingA_id"], self.top_pairs_df["ingB_id"]))
        if "q_global" in self.top_pairs_df.columns:
            q_by_pair = self.top_pairs_df.groupby(["ingA_id", "ingB_id"])["q_global"].min().to_dict()
        else:
            q_by_pair = {}
        if "p_raw" in self.top_pairs_df.columns:
            p_by_pair = self.top_pairs_df.groupby(["ingA_id", "ingB_id"])["p_raw"].min().to_dict()
        else:
            p_by_pair = {}
        confirmed_rows: List[Dict[str, Any]] = []
        n_tasks = len(pairs_list) * len(self.func_cols)
        t0 = time.perf_counter()
        done = 0
        for pi, (ing_a, ing_b) in enumerate(pairs_list):
            q_global = q_by_pair.get((ing_a, ing_b), np.nan)
            p_raw = p_by_pair.get((ing_a, ing_b), np.nan)
            for ci, category in enumerate(self.func_cols):
                r = self._confirm_one_pair_category(ing_a, ing_b, category, ci, q_global=q_global, p_raw=p_raw)
                if r is not None:
                    confirmed_rows.append(r)
                done += 1
                if done % 200 == 0:
                    elapsed = time.perf_counter() - t0
                    eta = (elapsed / done) * (n_tasks - done) if done else 0
                    logger.info("[%s] Confirm progress: %d/%d pairs×cats elapsed=%.1fs ETA=%.1fs", _ts(), done, n_tasks, elapsed, eta)
        logger.info("[%s] Confirm done: %d rows in %.1fs", _ts(), len(confirmed_rows), time.perf_counter() - t0)
        return pd.DataFrame(confirmed_rows) if confirmed_rows else pd.DataFrame()

    def run_bootstrap(
        self,
        confirmed_df: pd.DataFrame,
        top_n: Optional[int] = None,
        B: Optional[int] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Bootstrap within groups for top_n pair-category rows. Returns bootstrap_stability.parquet schema."""
        self._build_indices()
        B = B or self.B
        top_n = top_n or self.top_n_bootstrap or min(100, len(confirmed_df))
        subset = confirmed_df.head(top_n) if len(confirmed_df) > top_n else confirmed_df
        if subset.empty:
            return pd.DataFrame()
        n_tasks = len(subset)
        t0 = time.perf_counter()
        rows: List[Dict[str, Any]] = []
        for i, r in subset.iterrows():
            ing_a, ing_b, category = r["ingA_id"], r["ingB_id"], r["category"]
            cat_idx = self.func_cols.index(category) if category in self.func_cols else 0
            ab_idx, a_only_idx, b_only_idx, pool_none = self._get_groups(ing_a, ing_b)
            treated_matched, matched_a, matched_b, matched_none = _match_did_groups(
                self.faiss_index, self.embed, ab_idx,
                set(a_only_idx), set(b_only_idx), pool_none, topk=self.faiss_topk,
            )
            if len(treated_matched) < 2:
                continue
            y_ab = self.y_matrix[treated_matched, cat_idx]
            y_a = self.y_matrix[matched_a, cat_idx]
            y_b = self.y_matrix[matched_b, cat_idx]
            y_none = self.y_matrix[matched_none, cat_idx]
            did_obs, did_mean, did_std, p_boot, ci_low, ci_high, stability = _bootstrap_did_within_groups(
                y_ab, y_a, y_b, y_none, B=B, seed=seed, stability_threshold=0.01
            )
            rows.append({
                "ingA_id": ing_a,
                "ingB_id": ing_b,
                "category": category,
                "did_mean": did_mean,
                "did_std": did_std,
                "p_boot": p_boot,
                "did_ci_low": ci_low,
                "did_ci_high": ci_high,
                "n_ab": len(y_ab),
                "n_a": len(y_a),
                "n_b": len(y_b),
                "n_none": len(y_none),
                "stability_score": stability,
            })
            if len(rows) % 20 == 0 and len(rows) > 0:
                elapsed = time.perf_counter() - t0
                eta = (elapsed / len(rows)) * (n_tasks - len(rows)) if len(rows) else 0
                logger.info("[%s] Bootstrap progress: %d/%d elapsed=%.1fs ETA=%.1fs", _ts(), len(rows), n_tasks, elapsed, eta)
        logger.info("[%s] Bootstrap done: %d rows in %.1fs", _ts(), len(rows), time.perf_counter() - t0)
        return pd.DataFrame(rows)

    def run_null_test(self, confirmed_df: pd.DataFrame) -> Dict[str, Any]:
        """Permutation within groups: shuffle outcome within each group; recompute DID; empirical p."""
        self._build_indices()
        subset = confirmed_df.head(self.null_pairs * self.null_categories * 2)
        if subset.empty:
            return {"n_null_tests": 0, "note": "No confirmed rows for null test."}
        null_ps: List[float] = []
        rng = np.random.default_rng(42)
        seen: Set[Tuple[str, str, str]] = set()
        for _, r in subset.iterrows():
            key = (r["ingA_id"], r["ingB_id"], r["category"])
            if key in seen:
                continue
            seen.add(key)
            if len(null_ps) >= self.null_pairs * self.null_categories:
                break
            ing_a, ing_b, category = r["ingA_id"], r["ingB_id"], r["category"]
            cat_idx = self.func_cols.index(category) if category in self.func_cols else None
            if cat_idx is None:
                continue
            ab_idx, a_only_idx, b_only_idx, pool_none = self._get_groups(ing_a, ing_b)
            if len(ab_idx) < 5 or len(pool_none) < 20:
                continue
            treated_matched, matched_a, matched_b, matched_none = _match_did_groups(
                self.faiss_index, self.embed, ab_idx,
                set(a_only_idx), set(b_only_idx), pool_none, topk=self.faiss_topk,
            )
            if len(treated_matched) < 5:
                continue
            y_ab = self.y_matrix[treated_matched, cat_idx].copy()
            y_a = self.y_matrix[matched_a, cat_idx].copy()
            y_b = self.y_matrix[matched_b, cat_idx].copy()
            y_none = self.y_matrix[matched_none, cat_idx].copy()
            did_obs, _, _, p_obs = _did_group_means(y_ab, y_a, y_b, y_none)
            if np.isnan(did_obs):
                continue
            n_perm = 50
            did_perm: List[float] = []
            for _ in range(n_perm):
                rng.shuffle(y_ab)
                rng.shuffle(y_a)
                rng.shuffle(y_b)
                rng.shuffle(y_none)
                d, _, _, _ = _did_group_means(y_ab, y_a, y_b, y_none)
                if not np.isnan(d):
                    did_perm.append(d)
            if len(did_perm) < 10:
                continue
            did_perm = np.array(did_perm)
            p_null = float(np.mean(np.abs(did_perm) >= np.abs(did_obs)))
            p_null = max(p_null, 1.0 / (n_perm + 1))
            null_ps.append(p_null)
        null_ps_arr = np.array(null_ps)
        frac_sig = float(np.mean(null_ps_arr < 0.05)) if len(null_ps_arr) > 0 else 0.0
        report = {
            "n_null_tests": int(len(null_ps_arr)),
            "n_pairs_subset": self.null_pairs,
            "n_categories_subset": self.null_categories,
            "frac_significant_p005_under_null": frac_sig,
            "expected_approx": 0.05,
            "null_p_mean": float(np.mean(null_ps_arr)) if len(null_ps_arr) > 0 else None,
            "null_p_median": float(np.median(null_ps_arr)) if len(null_ps_arr) > 0 else None,
            "note": "Permutation within groups (shuffle outcome per group); DID recomputed; empirical p = fraction |DID_perm| >= |DID_obs|.",
        }
        return report

    def export_kg(
        self,
        confirmed_df: pd.DataFrame,
        bootstrap_df: pd.DataFrame,
        run_id: str = "phase13b",
    ) -> None:
        """Write kg/ with kg_nodes.parquet, kg_edges.parquet, kg_nodes.csv, kg_edges.csv."""
        kg_dir = self.output_dir / "kg"
        kg_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] Exporting KG to %s", _ts(), kg_dir)

        all_ings: Set[str] = set()
        for _, r in confirmed_df.iterrows():
            all_ings.add(str(r["ingA_id"]))
            all_ings.add(str(r["ingB_id"]))
        nodes_ing = [
            {"node_id": nid if str(nid).startswith("ING_") else f"ING_{nid}", "label": "Ingredient", "ingredient_id": nid}
            for nid in sorted(all_ings)
        ]
        nodes_cat = [{"node_id": c, "label": "Category", "category": c} for c in self.func_cols]
        nodes_int: List[Dict[str, Any]] = []
        for _, r in confirmed_df.iterrows():
            nodes_int.append({
                "node_id": f"INT_{r['ingA_id']}_{r['ingB_id']}",
                "label": "Interaction",
                "ingA_id": r["ingA_id"],
                "ingB_id": r["ingB_id"],
            })
        nodes_df = pd.DataFrame(nodes_ing + nodes_cat + nodes_int)
        nodes_df = nodes_df.drop_duplicates(subset=["node_id"])
        edges_rows: List[Dict[str, Any]] = []
        for _, r in confirmed_df.iterrows():
            eid = f"INT_{r['ingA_id']}_{r['ingB_id']}"
            aid = r["ingA_id"] if str(r["ingA_id"]).startswith("ING_") else f"ING_{r['ingA_id']}"
            bid = r["ingB_id"] if str(r["ingB_id"]).startswith("ING_") else f"ING_{r['ingB_id']}"
            edges_rows.append({
                "source_id": aid,
                "target_id": bid,
                "edge_type": "INTERACTS_WITH",
                "did": r.get("did"),
                "p_analytic": r.get("p_analytic"),
                "q_global": r.get("q_global"),
                "n_ab": r.get("n_ab"),
                "n_none": r.get("n_none"),
                "run_id": run_id,
                "interaction_id": eid,
            })
        for _, r in confirmed_df.iterrows():
            eid = f"INT_{r['ingA_id']}_{r['ingB_id']}"
            edges_rows.append({
                "source_id": eid,
                "target_id": str(r["category"]),
                "edge_type": "AFFECTS",
                "ingA_id": r["ingA_id"],
                "ingB_id": r["ingB_id"],
                "category": r["category"],
                "did": r.get("did"),
                "p_analytic": r.get("p_analytic"),
                "p_boot": None,
                "stability_score": None,
                "n_ab": r.get("n_ab"),
                "n_none": r.get("n_none"),
                "run_id": run_id,
            })
        boot_map = bootstrap_df.set_index(["ingA_id", "ingB_id", "category"]) if not bootstrap_df.empty else pd.DataFrame()
        for er in edges_rows:
            if er["edge_type"] != "AFFECTS":
                continue
            key = (er.get("ingA_id"), er.get("ingB_id"), er.get("category"))
            if key[0] is not None and key[1] is not None and key[2] is not None and not boot_map.empty and key in boot_map.index:
                row = boot_map.loc[key]
                er["p_boot"] = row.get("p_boot")
                er["stability_score"] = row.get("stability_score")
        edges_df = pd.DataFrame(edges_rows)
        nodes_path = kg_dir / "kg_nodes.parquet"
        edges_path = kg_dir / "kg_edges.parquet"
        nodes_df.to_parquet(nodes_path, index=False)
        edges_df.to_parquet(edges_path, index=False)
        nodes_df.to_csv(kg_dir / "kg_nodes.csv", index=False)
        edges_df.to_csv(kg_dir / "kg_edges.csv", index=False)
        logger.info("[%s] KG written: nodes=%d edges=%d", _ts(), len(nodes_df), len(edges_df))

    def run(
        self,
        run_bootstrap: bool = True,
        run_null: bool = True,
        run_kg: bool = True,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: confirm -> bootstrap -> null -> diagnostics -> KG."""
        run_id = run_id or f"phase13b_{int(time.time())}"
        t0 = time.perf_counter()
        confirmed_df = self.run_confirm()
        bootstrap_df = pd.DataFrame()
        if run_bootstrap and not confirmed_df.empty:
            bootstrap_df = self.run_bootstrap(confirmed_df, B=self.B, seed=42)
        null_report = {}
        if run_null:
            null_report = self.run_null_test(confirmed_df)
        if not bootstrap_df.empty:
            merge_cols = ["ingA_id", "ingB_id", "category"]
            boot_sub = bootstrap_df[merge_cols + ["did_std", "p_boot", "stability_score"]].rename(columns={"did_std": "did_boot_se"})
            confirmed_df = confirmed_df.merge(boot_sub, on=merge_cols, how="left")
            bootstrap_df.to_parquet(self.output_dir / "bootstrap_stability.parquet", index=False)
        confirmed_path = self.output_dir / "atlas_confirmed.parquet"
        confirmed_df.to_parquet(confirmed_path, index=False)
        with open(self.output_dir / "null_tests.json", "w") as f:
            json.dump(null_report, f, indent=2)
        n_05 = int((confirmed_df["p_analytic"] < 0.05).sum()) if "p_analytic" in confirmed_df.columns and len(confirmed_df) > 0 else 0
        q_05 = int((confirmed_df["q_global"] < 0.05).sum()) if "q_global" in confirmed_df.columns and len(confirmed_df) > 0 else 0
        diagnostics = {
            "top_k": len(self.top_pairs_df),
            "B": self.B,
            "runtime_sec": round(time.perf_counter() - t0, 2),
            "n_pairs_confirmed": confirmed_df["ingA_id"].nunique() if not confirmed_df.empty else 0,
            "n_rows_written": len(confirmed_df),
            "fraction_p_under_005": n_05 / len(confirmed_df) if len(confirmed_df) > 0 else 0,
            "fraction_q_under_005": q_05 / len(confirmed_df) if len(confirmed_df) > 0 else 0,
            "n_bootstrap_rows": len(bootstrap_df),
        }
        with open(self.output_dir / "phase13b_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)
        if run_kg:
            self.export_kg(confirmed_df, bootstrap_df, run_id=run_id)
        self._repack_portable_outputs()
        logger.info("[%s] Phase13B END — confirmed=%d bootstrap=%d runtime=%.1fs", _ts(), len(confirmed_df), len(bootstrap_df), diagnostics["runtime_sec"])
        return diagnostics

    def _repack_portable_outputs(self) -> None:
        """Repack Parquet outputs to portable format and write CSVs. On failure log and continue."""
        if repack_parquet_file is None:
            return
        out = self.output_dir
        files = [
            (out / "atlas_confirmed.parquet", out / "atlas_confirmed_portable.parquet", out / "atlas_confirmed.csv"),
            (out / "bootstrap_stability.parquet", out / "bootstrap_stability_portable.parquet", out / "bootstrap_stability.csv"),
        ]
        for src, dst_portable, csv_path in files:
            if not src.exists():
                continue
            try:
                repack_parquet_file(src, dst_portable, csv_path)
            except Exception as e:
                logger.warning("[%s] Portable repack failed for %s: %s", _ts(), src.name, e)
        kg_dir = out / "kg"
        if kg_dir.exists():
            for name in ("kg_nodes.parquet", "kg_edges.parquet"):
                src = kg_dir / name
                if not src.exists():
                    continue
                dst_portable = kg_dir / name.replace(".parquet", "_portable.parquet")
                csv_path = kg_dir / name.replace(".parquet", ".csv")
                try:
                    repack_parquet_file(src, dst_portable, csv_path)
                except Exception as e:
                    logger.warning("[%s] Portable repack failed for %s: %s", _ts(), src.name, e)
