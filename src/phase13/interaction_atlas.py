"""
Phase 13: Ingredient interaction atlas — candidate pairs, confounders, regression, FDR, stability, null tests.
Memory-safe: works with grouped recipe–ingredient structures; no full recipe×ingredient matrix.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLSResults
except ImportError:
    sm = None
    OLSResults = None


# ---------------------------------------------------------------------------
# A) Candidate pair generation
# ---------------------------------------------------------------------------

def build_candidate_pairs(
    recipe_ingredients: pd.DataFrame,
    min_ing_freq: int = 5000,
    min_pair_freq: int = 300,
    max_pairs: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]], pd.Series]:
    """
    Build candidate ingredient pairs from recipe–ingredient table.
    Memory-safe: iterates by recipe groups; does not materialize full cooccurrence matrix.

    Returns:
        pair_stats: DataFrame with ingA_id, ingB_id, cooccur, freqA, freqB, jaccard, ...
        ingredient_stats: Series of ingredient -> recipe count (for kept ingredients)
        candidate_pairs: list of (ingA_id, ingB_id)
        ing_freq: Series ingredient -> count (all ingredients before filter)
    """
    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
    ing_freq = ri.groupby("ingredient_id").size()
    kept = ing_freq[ing_freq >= min_ing_freq].index.tolist()
    kept_set = set(kept)
    ingredient_stats = ing_freq.loc[kept].copy()

    # Recipe count per kept ingredient (freq = # recipes containing this ingredient)
    ing_recipe_freq: Dict[str, int] = {i: 0 for i in kept}
    # Co-occurrence: for each recipe, combinations of kept ingredients
    from itertools import combinations

    pair_counts: Dict[Tuple[str, str], int] = {}

    for recipe_id, grp in ri.groupby("recipe_id"):
        ings = [x for x in grp["ingredient_id"].unique() if x in kept_set]
        for i in ings:
            ing_recipe_freq[i] = ing_recipe_freq.get(i, 0) + 1
        if len(ings) < 2:
            continue
        for a, b in combinations(sorted(ings), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    pairs_above = [(p, c) for p, c in pair_counts.items() if c >= min_pair_freq]
    pairs_above.sort(key=lambda x: -x[1])
    if max_pairs is not None:
        pairs_above = pairs_above[:max_pairs]
    candidate_pairs = [p for p, _ in pairs_above]

    # pair_stats: freqA/freqB = recipe count containing that ingredient
    rows = []
    for (a, b), cooccur in pairs_above:
        fa = ing_recipe_freq.get(a, int(ing_freq.get(a, 0)))
        fb = ing_recipe_freq.get(b, int(ing_freq.get(b, 0)))
        jaccard = cooccur / (fa + fb - cooccur) if (fa + fb - cooccur) > 0 else 0.0
        rows.append({"ingA_id": a, "ingB_id": b, "cooccur": cooccur, "freqA": fa, "freqB": fb, "jaccard": jaccard})
    pair_stats = pd.DataFrame(rows)
    ingredient_stats = pd.Series(ing_recipe_freq)
    return pair_stats, ingredient_stats, candidate_pairs, ing_freq


# ---------------------------------------------------------------------------
# B) Confounders
# ---------------------------------------------------------------------------

def build_confounders(
    recipe_ids: pd.Series,
    recipe_ingredients: pd.DataFrame,
    signatures: Optional[pd.DataFrame] = None,
    existing_confounders: Optional[pd.DataFrame] = None,
    k_clusters: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build confounder DataFrame: recipe_id, num_ingredients, cuisine_or_cluster, [num_compounds].
    If existing_confounders has recipe_id + cuisine/predicted_cuisine/cluster, use it.
    Else build cluster from ingredient presence (mini-batch kmeans on hashed presence).
    """
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({"recipe_id": recipe_ids.unique()})

    # num_ingredients from recipe_ingredients
    num_ing = recipe_ingredients.groupby("recipe_id").size().reset_index(name="num_ingredients")
    out = out.merge(num_ing, on="recipe_id", how="left")
    out["num_ingredients"] = out["num_ingredients"].fillna(0).astype(int)

    # cuisine / cluster
    if existing_confounders is not None:
        for col in ["cuisine", "predicted_cuisine", "cuisine_or_cluster", "cluster_id"]:
            if col in existing_confounders.columns and "recipe_id" in existing_confounders.columns:
                sub = existing_confounders[["recipe_id", col]].drop_duplicates()
                sub = sub.rename(columns={col: "cuisine_or_cluster"})
                if "cuisine_or_cluster" not in out.columns:
                    out = out.merge(sub, on="recipe_id", how="left")
                break
    if "cuisine_or_cluster" not in out.columns:
        # Build recipe x ingredient presence (sparse-ish): one row per recipe, columns = top ingredients
        ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
        top_ings = ri["ingredient_id"].value_counts().head(500).index.tolist()
        rec_list = out["recipe_id"].tolist()
        rec_to_idx = {r: i for i, r in enumerate(rec_list)}
        ing_to_idx = {ing: j for j, ing in enumerate(top_ings)}
        n_rec = len(rec_list)
        n_ing = len(top_ings)
        X = np.zeros((n_rec, n_ing), dtype=np.float32)
        for _, row in ri.iterrows():
            r, ing = row["recipe_id"], row["ingredient_id"]
            if r in rec_to_idx and ing in ing_to_idx:
                X[rec_to_idx[r], ing_to_idx[ing]] = 1.0
        k = min(k_clusters, n_rec - 1, n_ing)
        if k < 2:
            out["cuisine_or_cluster"] = 0
        else:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1000, n_init=3)
            lab = km.fit_predict(X)
            out["cuisine_or_cluster"] = pd.Series(lab).map(lambda x: f"cluster_{x}").values
        out["cuisine_or_cluster"] = out["cuisine_or_cluster"].astype("category")

    if signatures is not None and "num_compounds" in signatures.columns:
        sub = signatures[["recipe_id", "num_compounds"]].drop_duplicates()
        out = out.merge(sub, on="recipe_id", how="left")
    else:
        out["num_compounds"] = np.nan

    return out


# ---------------------------------------------------------------------------
# C) Regression per pair
# ---------------------------------------------------------------------------

def _run_single_pair_regression(
    ing_a: str,
    ing_b: str,
    recipe_ingredients_grouped: Dict[str, set],
    signatures: pd.DataFrame,
    confounders: pd.DataFrame,
    category: str,
    recipe_idx: Optional[Dict[str, int]] = None,
    interaction_override: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Run y ~ A + B + A*B + confounders for one pair and one category.
    If interaction_override is provided (same length as rec_list), use it instead of A*B (e.g. for null tests).
    """
    if sm is None:
        return None
    rec_ids = set(recipe_ingredients_grouped.keys())
    set_a = set()
    set_b = set()
    for rid, ings in recipe_ingredients_grouped.items():
        if ing_a in ings:
            set_a.add(rid)
        if ing_b in ings:
            set_b.add(rid)
    relevant = set_a | set_b
    if len(relevant) < 30:
        return None
    rec_list = sorted(relevant)
    y = signatures.set_index("recipe_id").loc[rec_list, category].values
    A = np.array([1 if rec in set_a else 0 for rec in rec_list], dtype=np.float64)
    B = np.array([1 if rec in set_b else 0 for rec in rec_list], dtype=np.float64)
    A_B = interaction_override if interaction_override is not None else (A * B)
    if A_B.shape[0] != len(rec_list):
        return None
    conf = confounders.set_index("recipe_id").loc[rec_list]
    X = pd.DataFrame({"A": A, "B": B, "A_B": A_B}, index=rec_list)
    if "num_ingredients" in conf.columns:
        X["num_ingredients"] = conf["num_ingredients"].values
    if "cuisine_or_cluster" in conf.columns:
        dums = pd.get_dummies(conf["cuisine_or_cluster"], drop_first=True, dtype=float)
        X = pd.concat([X, dums], axis=1)
    if "num_compounds" in conf.columns and conf["num_compounds"].notna().any():
        X["num_compounds"] = conf["num_compounds"].fillna(0).values
    X = sm.add_constant(X, has_constant="add")
    try:
        res = sm.OLS(y, X).fit()
    except Exception:
        return None
    n_both = int((A * B).sum())
    n_a_only = int((A * (1 - B)).sum())
    n_b_only = int(((1 - A) * B).sum())
    idx_ab = res.params.index.get_loc("A_B") if "A_B" in res.params.index else None
    if idx_ab is None:
        return None
    beta_int = res.params.iloc[idx_ab]
    se = res.bse.iloc[idx_ab]
    t = res.tvalues.iloc[idx_ab]
    p = res.pvalues.iloc[idx_ab]
    return {
        "ingA_id": ing_a,
        "ingB_id": ing_b,
        "category": category,
        "beta_int": float(beta_int),
        "se": float(se),
        "t": float(t),
        "p": float(p),
        "n_total": len(rec_list),
        "n_both": n_both,
        "n_A_only": n_a_only,
        "n_B_only": n_b_only,
    }


def run_pair_regression(
    candidate_pairs: List[Tuple[str, str]],
    recipe_ingredients: pd.DataFrame,
    signatures: pd.DataFrame,
    confounders: pd.DataFrame,
    categories: List[str],
    n_jobs: int = 1,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 1000,
) -> pd.DataFrame:
    """
    Run interaction regression for all pairs and categories.
    Returns long-format DataFrame: ingA_id, ingB_id, category, beta_int, se, t, p, n_total, n_both, n_A_only, n_B_only.
    """
    from joblib import Parallel, delayed

    # Group recipe -> set(ingredient_id)
    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
    grouped = ri.groupby("recipe_id")["ingredient_id"].apply(set).to_dict()

    all_rows = []
    n_pairs = len(candidate_pairs)
    for cat in categories:
        if cat not in signatures.columns:
            continue
        if n_jobs != 1:
            jobs = [
                delayed(_run_single_pair_regression)(
                    a, b, grouped, signatures, confounders, cat, None
                )
                for a, b in candidate_pairs
            ]
            results = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
        else:
            results = []
            for i, (a, b) in enumerate(candidate_pairs):
                r = _run_single_pair_regression(a, b, grouped, signatures, confounders, cat, None)
                results.append(r)
                if checkpoint_path and (i + 1) % checkpoint_every == 0:
                    so_far = [x for x in results if x is not None]
                    if so_far:
                        pd.DataFrame(so_far).to_parquet(
                            checkpoint_path / f"checkpoint_cat_{cat}_pair_{i+1}.parquet",
                            index=False,
                        )
        for r in results:
            if r is not None:
                all_rows.append(r)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# D) FDR
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR; returns q-values (same length as pvals).
    Robust to NaN/inf: converted to 1.0. p clipped to [0,1].
    Monotonicity enforced by taking min-accumulate from largest to smallest.
    """
    p = np.asarray(pvals, dtype=np.float64)
    if p.ndim != 1:
        p = p.reshape(-1)

    # sanitize
    p = np.where(np.isfinite(p), p, 1.0)
    p = np.clip(p, 0.0, 1.0)

    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)

    q_sorted = p_sorted * (n / ranks)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q


# ---------------------------------------------------------------------------
# E–F) Holdout & null tests (helpers for notebook)
# ---------------------------------------------------------------------------

def run_null_tests(
    recipe_ingredients: pd.DataFrame,
    signatures: pd.DataFrame,
    confounders: pd.DataFrame,
    candidate_pairs: List[Tuple[str, str]],
    categories: List[str],
    n_subset_pairs: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run null tests: (1) shuffle ingredients within cluster, (2) shuffle y.
    Returns dict with null_beta_ingredient_shuffle, null_beta_label_shuffle, etc.
    """
    rng = np.random.default_rng(seed)
    results = {"ingredient_shuffle": [], "label_shuffle": []}

    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
    grouped = ri.groupby("recipe_id")["ingredient_id"].apply(set).to_dict()
    pairs_sub = candidate_pairs[:n_subset_pairs] if len(candidate_pairs) > n_subset_pairs else candidate_pairs

    for category in categories:
        if category not in signatures.columns:
            continue
        y_orig = signatures.set_index("recipe_id")[category]
        # Label shuffle: shuffle y across recipes
        y_shuf = y_orig.copy()
        y_shuf.values[:] = rng.permutation(y_shuf.values)
        sig_label = signatures.copy()
        sig_label[category] = y_shuf.values
        for (a, b) in pairs_sub[:50]:  # small subset for null
            r = _run_single_pair_regression(a, b, grouped, sig_label, confounders, category, None)
            if r is not None:
                results["label_shuffle"].append({"category": category, "beta_int": r["beta_int"], "p": r["p"]})
        # Ingredient shuffle: permute ingredient_ids within each recipe (preserve sizes)
        ri_shuf = recipe_ingredients.copy()
        for rid, grp in ri_shuf.groupby("recipe_id"):
            ings = grp["ingredient_id"].tolist()
            all_ings = recipe_ingredients["ingredient_id"].unique().tolist()
            new_ings = rng.choice(all_ings, size=len(ings), replace=True).tolist()
            ri_shuf.loc[ri_shuf["recipe_id"] == rid, "ingredient_id"] = new_ings
        grouped_shuf = ri_shuf.groupby("recipe_id")["ingredient_id"].apply(set).to_dict()
        for (a, b) in pairs_sub[:50]:
            r = _run_single_pair_regression(a, b, grouped_shuf, signatures, confounders, category, None)
            if r is not None:
                results["ingredient_shuffle"].append({"category": category, "beta_int": r["beta_int"], "p": r["p"]})

    out = {
        "null_beta_ingredient_shuffle_mean": np.mean([x["beta_int"] for x in results["ingredient_shuffle"]]) if results["ingredient_shuffle"] else None,
        "null_beta_label_shuffle_mean": np.mean([x["beta_int"] for x in results["label_shuffle"]]) if results["label_shuffle"] else None,
        "n_ingredient_shuffle": len(results["ingredient_shuffle"]),
        "n_label_shuffle": len(results["label_shuffle"]),
    }
    return out


# ---------------------------------------------------------------------------
# G) Bootstrap stability
# ---------------------------------------------------------------------------

def bootstrap_stability(
    interaction_atlas: pd.DataFrame,
    recipe_ingredients: pd.DataFrame,
    signatures: pd.DataFrame,
    confounders: pd.DataFrame,
    top_n: int = 200,
    B: int = 50,
    seed: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    For top_n interactions (by q-value), bootstrap recipes B times and get CI for beta_int.
    Returns DataFrame with ingA_id, ingB_id, category, beta_int_orig, beta_int_lower, beta_int_upper, ...
    """
    rng = np.random.default_rng(seed)
    # Top N by min q across categories (or by abs(beta_int)*significance)
    atlas = interaction_atlas.copy()
    if "q_global" in atlas.columns:
        atlas["min_q"] = atlas.groupby(["ingA_id", "ingB_id"])["q_global"].transform("min")
    elif "q" in atlas.columns:
        atlas["min_q"] = atlas.groupby(["ingA_id", "ingB_id"])["q"].transform("min")
    else:
        atlas["min_q"] = atlas.groupby(["ingA_id", "ingB_id"])["p"].transform("min")
    top_pairs = atlas.sort_values("min_q").drop_duplicates(subset=["ingA_id", "ingB_id"]).head(top_n)
    pairs_list = list(zip(top_pairs["ingA_id"], top_pairs["ingB_id"]))
    recipe_ids = signatures["recipe_id"].unique()
    n_rec = len(recipe_ids)
    rows = []
    for (ing_a, ing_b) in pairs_list:
        betas = []
        for _ in range(B):
            idx = rng.choice(n_rec, size=n_rec, replace=True)
            rec_boot = recipe_ids[idx]
            sig_boot = signatures[signatures["recipe_id"].isin(rec_boot)]
            conf_boot = confounders[confounders["recipe_id"].isin(rec_boot)]
            ri_boot = recipe_ingredients[recipe_ingredients["recipe_id"].isin(rec_boot)]
            grouped = ri_boot.groupby("recipe_id")["ingredient_id"].apply(set).to_dict()
            cat = top_pairs[(top_pairs["ingA_id"] == ing_a) & (top_pairs["ingB_id"] == ing_b)]["category"].iloc[0]
            r = _run_single_pair_regression(ing_a, ing_b, grouped, sig_boot, conf_boot, cat, None)
            if r is not None:
                betas.append(r["beta_int"])
        if betas:
            rows.append({
                "ingA_id": ing_a,
                "ingB_id": ing_b,
                "beta_int_orig": top_pairs[(top_pairs["ingA_id"] == ing_a) & (top_pairs["ingB_id"] == ing_b)]["beta_int"].iloc[0],
                "beta_int_mean": np.mean(betas),
                "beta_int_lower": np.percentile(betas, 2.5),
                "beta_int_upper": np.percentile(betas, 97.5),
            })
    return pd.DataFrame(rows)
