"""
Phase15: Triplet Interaction Layer — mine triplets from recipe_ingredients (PMI/lift),
score by pair_mediation (mechanistic + optional propagation), export INT3 nodes and edges.
Bounded pipeline for production on large (e.g. 9.45M-row) recipe_ingredients.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import phase14_config as config
from .id_normalization import to_category_id, to_ingredient_id, to_triplet_id
from .utils import timer, elapsed_since, within_time_budget

logger = logging.getLogger(__name__)


def _pair_key(ing_a: str, ing_b: str) -> Tuple[str, str]:
    a, b = to_ingredient_id(ing_a), to_ingredient_id(ing_b)
    return (a, b) if a <= b else (b, a)


def _mine_triplets_bounded(
    recipe_ingredients: pd.DataFrame,
    max_recipes: int,
    max_unique_ingredients: int,
    min_support_pairs: int,
    min_support_triplets: int,
    top_k_pairs: int,
    max_triplets_scored: int,
    time_budget_seconds: float,
    random_seed: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Bounded, memory-safe triplet mining. Returns (triplets_df, summary_dict).
    triplets_df columns: ingA_id, ingB_id, ingC_id, support_ABC, PMI_ABC, lift_ABC, source, sampled_recipes_n.
    """
    summary: Dict[str, Any] = {
        "stage0_recipes_in": None, "stage0_recipes_sampled": None, "stage0_unique_ingredients": None, "stage0_recipe_sets": None,
        "stage1_candidate_pairs": None, "stage1_top_k_pairs": None,
        "stage2_recipes_with_pairs": None, "stage2_triplets_raw": None, "stage2_triplets_after_support": None, "stage2_triplets_scored": None,
        "elapsed_seconds": None, "time_budget_seconds": time_budget_seconds, "time_exceeded": False,
        "max_recipes": max_recipes, "max_unique_ingredients": max_unique_ingredients, "min_support_pairs": min_support_pairs, "min_support_triplets": min_support_triplets,
        "top_k_pairs": top_k_pairs, "max_triplets_scored": max_triplets_scored, "random_seed": random_seed,
    }
    summary["max_recipes"] = max_recipes
    summary["max_unique_ingredients"] = max_unique_ingredients
    summary["min_support_pairs"] = min_support_pairs
    summary["min_support_triplets"] = min_support_triplets
    summary["top_k_pairs"] = top_k_pairs
    summary["max_triplets_scored"] = max_triplets_scored
    summary["random_seed"] = random_seed

    if recipe_ingredients.empty or "recipe_id" not in recipe_ingredients.columns or "ingredient_id" not in recipe_ingredients.columns:
        return pd.DataFrame(columns=["ingA_id", "ingB_id", "ingC_id", "support_ABC", "PMI_ABC", "lift_ABC", "source", "sampled_recipes_n"]), summary

    t0 = timer()
    # Only required columns
    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].copy()
    ri["ingredient_id"] = ri["ingredient_id"].astype(str).str.strip()
    ri["ingredient_id"] = ri["ingredient_id"].apply(lambda x: to_ingredient_id(x))
    ri = ri[ri["ingredient_id"].notna() & (ri["ingredient_id"] != "ING_unknown")]

    unique_recipe_series = ri["recipe_id"].drop_duplicates()
    n_recipes_in = len(unique_recipe_series)
    summary["stage0_recipes_in"] = n_recipes_in

    if n_recipes_in > max_recipes:
        recipe_sample = unique_recipe_series.sample(n=max_recipes, random_state=random_seed)
        ri = ri[ri["recipe_id"].isin(recipe_sample)]
        n_recipes_sampled = max_recipes
    else:
        n_recipes_sampled = n_recipes_in
    summary["stage0_recipes_sampled"] = n_recipes_sampled
    N = n_recipes_sampled

    # Single-ingredient supports; keep top max_unique_ingredients
    support_single = ri.groupby("ingredient_id")["recipe_id"].nunique()
    if len(support_single) > max_unique_ingredients:
        top_ings = support_single.nlargest(max_unique_ingredients).index.tolist()
        ri = ri[ri["ingredient_id"].isin(top_ings)]
        support_single = support_single.loc[support_single.index.isin(top_ings)]
    summary["stage0_unique_ingredients"] = len(support_single)

    # Recipe -> set of ingredient_id (strings)
    recipe_sets = ri.groupby("recipe_id")["ingredient_id"].apply(lambda x: set(x.unique())).to_dict()
    recipe_sets = {rid: s for rid, s in recipe_sets.items() if len(s) >= 3}
    summary["stage0_recipe_sets"] = len(recipe_sets)
    if not recipe_sets:
        return pd.DataFrame(columns=["ingA_id", "ingB_id", "ingC_id", "support_ABC", "PMI_ABC", "lift_ABC", "source", "sampled_recipes_n"]), summary

    # Integer coding
    all_ings = sorted(support_single.index.tolist())
    ing_to_code = {s: i for i, s in enumerate(all_ings)}
    n_ing = len(all_ings)

    # Stage 1: frequent pairs
    if not within_time_budget(t0, time_budget_seconds):
        summary["elapsed_seconds"] = round(elapsed_since(t0), 2)
        summary["time_exceeded"] = True
        return pd.DataFrame(columns=["ingA_id", "ingB_id", "ingC_id", "support_ABC", "PMI_ABC", "lift_ABC", "source", "sampled_recipes_n"]), summary

    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for rid, ings in recipe_sets.items():
        codes = sorted(ing_to_code[s] for s in ings if s in ing_to_code)
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                pair_counts[(codes[i], codes[j])] += 1

    pairs_freq = [(p, c) for p, c in pair_counts.items() if c >= min_support_pairs]
    summary["stage1_candidate_pairs"] = len(pairs_freq)
    if not pairs_freq:
        summary["elapsed_seconds"] = round(elapsed_since(t0), 2)
        return pd.DataFrame(columns=["ingA_id", "ingB_id", "ingC_id", "support_ABC", "PMI_ABC", "lift_ABC", "source", "sampled_recipes_n"]), summary

    pairs_freq.sort(key=lambda x: -x[1])
    candidate_pairs = set(pairs_freq[:top_k_pairs])
    summary["stage1_top_k_pairs"] = len(candidate_pairs)

    # Stage 2: triplets only from recipes that contain at least one candidate pair
    if not within_time_budget(t0, time_budget_seconds):
        summary["elapsed_seconds"] = round(elapsed_since(t0), 2)
        summary["time_exceeded"] = True
        return pd.DataFrame(columns=["ingA_id", "ingB_id", "ingC_id", "support_ABC", "PMI_ABC", "lift_ABC", "source", "sampled_recipes_n"]), summary

    recipes_with_pairs = []
    for rid, ings in recipe_sets.items():
        codes = [ing_to_code[s] for s in ings if s in ing_to_code]
        if len(codes) < 3:
            continue
        code_set = set(codes)
        for (a, b) in candidate_pairs:
            if a in code_set and b in code_set:
                recipes_with_pairs.append((rid, codes))
                break
    summary["stage2_recipes_with_pairs"] = len(recipes_with_pairs)

    triplet_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
    for rid, codes in recipes_with_pairs:
        if not within_time_budget(t0, time_budget_seconds):
            break
        codes_sorted = sorted(codes)
        for i in range(len(codes_sorted)):
            for j in range(i + 1, len(codes_sorted)):
                for k in range(j + 1, len(codes_sorted)):
                    triplet_counts[(codes_sorted[i], codes_sorted[j], codes_sorted[k])] += 1

    summary["stage2_triplets_raw"] = len(triplet_counts)
    triplets_list = [(t, c) for t, c in triplet_counts.items() if c >= min_support_triplets]
    triplets_list.sort(key=lambda x: -x[1])
    triplets_list = triplets_list[:max_triplets_scored]
    summary["stage2_triplets_after_support"] = len([x for x in triplet_counts.values() if x >= min_support_triplets])
    summary["stage2_triplets_scored"] = len(triplets_list)

    # Build output with PMI/lift (single-ingredient supports only for triplets)
    support_single_arr = support_single.reindex(all_ings, fill_value=0).astype(int).values
    rows = []
    for (i, j, k), sup_abc in triplets_list:
        sa = int(support_single_arr[i])
        sb = int(support_single_arr[j])
        sc = int(support_single_arr[k])
        if sa == 0 or sb == 0 or sc == 0:
            continue
        denom = (sa * sb * sc) / (N * N)
        if denom <= 0:
            continue
        lift_abc = sup_abc / denom
        pmi_abc = math.log((sup_abc * N * N) / (sa * sb * sc))
        a_id, b_id, c_id = all_ings[i], all_ings[j], all_ings[k]
        rows.append({
            "ingA_id": a_id, "ingB_id": b_id, "ingC_id": c_id,
            "support_ABC": sup_abc,
            "PMI_ABC": pmi_abc, "lift_ABC": lift_abc,
            "source": "bounded",
            "sampled_recipes_n": N,
        })
    summary["elapsed_seconds"] = round(elapsed_since(t0), 2)
    summary["time_exceeded"] = not within_time_budget(t0, time_budget_seconds)

    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ingA_id", "ingB_id", "ingC_id", "support_ABC", "PMI_ABC", "lift_ABC", "source", "sampled_recipes_n"])
    return out, summary


def compute_supports_and_pmi(
    recipe_ingredients: pd.DataFrame,
    min_support: int,
    drop_top_freq_pct: float = 0.0,
    max_triplets: int = 50000,
    smoke_n_recipes: Optional[int] = None,
) -> pd.DataFrame:
    """
    From recipe_ingredients (recipe_id, ingredient_id) compute:
    support_A, support_B, support_C, support_AB, support_AC, support_BC, support_ABC,
    PMI_ABC, lift_ABC, max_support_single.
    Filter: drop ingredients in top (drop_top_freq_pct*100)% by recipe count;
    require support_ABC >= min_support. Rank by PMI_ABC desc, then support_ABC desc.
    Cap at max_triplets. Returns DataFrame with columns ingA_id, ingB_id, ingC_id (sorted),
    support_ABC, support_A, support_B, support_C, support_AB, support_AC, support_BC,
    PMI_ABC, lift_ABC, max_support_single, N (total recipes).
    """
    if recipe_ingredients.empty or "recipe_id" not in recipe_ingredients.columns or "ingredient_id" not in recipe_ingredients.columns:
        return pd.DataFrame()
    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].copy()
    ri["ingredient_id"] = ri["ingredient_id"].astype(str).str.strip()
    ri["ingredient_id"] = ri["ingredient_id"].apply(lambda x: to_ingredient_id(x))
    if smoke_n_recipes and ri["recipe_id"].nunique() > smoke_n_recipes:
        recipe_sample = ri["recipe_id"].drop_duplicates().head(smoke_n_recipes)
        ri = ri[ri["recipe_id"].isin(recipe_sample)]
        logger.info("Triplet SMOKE: using %s recipes", len(recipe_sample))
    N = ri["recipe_id"].nunique()
    if N < 2:
        return pd.DataFrame()
    # Single-ingredient supports (recipe count per ingredient)
    support_single = ri.groupby("ingredient_id")["recipe_id"].nunique()
    if drop_top_freq_pct > 0 and drop_top_freq_pct < 100:
        threshold = support_single.quantile(1.0 - drop_top_freq_pct / 100.0)
        drop_ings = set(support_single[support_single >= threshold].index)
        ri = ri[~ri["ingredient_id"].isin(drop_ings)]
        logger.info("Dropped %s ingredients in top %.1f%% frequency", len(drop_ings), drop_top_freq_pct)
        support_single = ri.groupby("ingredient_id")["recipe_id"].nunique()
    # Pair supports: (ing_a, ing_b) -> recipe count (a < b)
    pairs_df = ri.merge(ri, on="recipe_id", suffixes=("_a", "_b"))
    pairs_df = pairs_df[pairs_df["ingredient_id_a"] < pairs_df["ingredient_id_b"]]
    support_pair_series = pairs_df.groupby(["ingredient_id_a", "ingredient_id_b"]).size()
    support_pair = dict(zip(support_pair_series.index.tolist(), support_pair_series.tolist()))
    # Triplet counts
    triplets: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for recipe_id, g in ri.groupby("recipe_id"):
        ings = sorted(g["ingredient_id"].unique().tolist())
        if len(ings) < 3:
            continue
        for i in range(len(ings)):
            for j in range(i + 1, len(ings)):
                for k in range(j + 1, len(ings)):
                    triplets[(ings[i], ings[j], ings[k])] += 1
    rows = []
    for (a, b, c), sup_abc in triplets.items():
        if sup_abc < min_support:
            continue
        sa = int(support_single.get(a, 0))
        sb = int(support_single.get(b, 0))
        sc = int(support_single.get(c, 0))
        if sa == 0 or sb == 0 or sc == 0:
            continue
        sab = int(support_pair.get((a, b) if a < b else (b, a), 0))
        sac = int(support_pair.get((a, c) if a < c else (c, a), 0))
        sbc = int(support_pair.get((b, c) if b < c else (c, b), 0))
        denom = (sa * sb * sc) / (N * N)
        if denom <= 0:
            continue
        lift_abc = sup_abc / denom
        pmi_abc = math.log((sup_abc * N * N) / (sa * sb * sc)) if (sa * sb * sc) > 0 else 0.0
        max_single = max(sa, sb, sc)
        rows.append({
            "ingA_id": a, "ingB_id": b, "ingC_id": c,
            "support_ABC": sup_abc,
            "support_A": sa, "support_B": sb, "support_C": sc,
            "support_AB": sab, "support_AC": sac, "support_BC": sbc,
            "PMI_ABC": pmi_abc, "lift_ABC": lift_abc,
            "max_support_single": max_single,
            "N": N,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["PMI_ABC", "support_ABC"], ascending=[False, False]).head(max_triplets)
    return df.reset_index(drop=True)


def triplet_category_score(
    ing_a: str,
    ing_b: str,
    ing_c: str,
    category: str,
    pair_mediation: pd.DataFrame,
    use_propagation: bool = True,
) -> Tuple[float, int, float]:
    """
    For triplet (A,B,C) and category, collect pair rows among {AB, AC, BC}.
    Returns (triplet_category_score, n_pairs_used, triplet_prop_boost).
    - If >=2 pairs: harmonic_mean(mechanistic_score) * 0.85 if only 2 pairs.
    - If 1 pair: mechanistic_score * 0.65.
    - If 0 pairs: 0.
    Optional propagation boost: final = sigmoid(pair_score) * (0.7 + 0.3 * prop_boost).
    """
    a = to_ingredient_id(ing_a)
    b = to_ingredient_id(ing_b)
    c = to_ingredient_id(ing_c)
    pairs_list = [_pair_key(a, b), _pair_key(a, c), _pair_key(b, c)]
    pm = pair_mediation
    if pm.empty or "category" not in pm.columns or "mechanistic_score" not in pm.columns:
        return 0.0, 0, 0.0
    cat_sub = pm[pm["category"].astype(str).str.strip() == str(category).strip()]
    if cat_sub.empty:
        return 0.0, 0, 0.0
    scores: List[float] = []
    prop_scores: List[float] = []
    for (pa, pb) in pairs_list:
        row = cat_sub[(cat_sub["ingA_id"] == pa) & (cat_sub["ingB_id"] == pb)]
        if row.empty:
            continue
        mech = float(row["mechanistic_score"].iloc[0]) if "mechanistic_score" in row.columns else 0.0
        scores.append(mech)
        if use_propagation and "propagated_pathway_score" in row.columns:
            pval = float(row["propagated_pathway_score"].iloc[0]) or 0.0
            if pval > 0:
                prop_scores.append(pval)
    n_pairs = len(scores)
    if n_pairs == 0:
        return 0.0, 0, 0.0
    if n_pairs == 1:
        triplet_pair_score = scores[0] * 0.65
    else:
        # Harmonic mean of mechanistic scores
        inv = sum(1.0 / (s + 1e-12) for s in scores)
        harmonic = n_pairs / inv if inv > 0 else 0.0
        triplet_pair_score = harmonic * 0.85 if n_pairs == 2 else harmonic
    # Optional propagation boost
    prop_boost = 0.0
    if use_propagation and prop_scores:
        prop_boost = float(np.mean(prop_scores))
        if prop_boost > 0:
            # Scale into [0,1] per category (simple: clip)
            prop_boost = min(1.0, prop_boost)
    # final = sigmoid(pair_score) * (0.7 + 0.3 * prop_boost)
    sig = 1.0 / (1.0 + math.exp(-np.clip(triplet_pair_score, -20, 20)))
    final_score = sig * (0.7 + 0.3 * prop_boost)
    return float(final_score), n_pairs, float(prop_boost)


def score_triplets_by_category(
    triplets_df: pd.DataFrame,
    pair_mediation: pd.DataFrame,
    categories: Optional[List[str]] = None,
    use_propagation: bool = True,
) -> pd.DataFrame:
    """
    For each row in triplets_df (ingA_id, ingB_id, ingC_id) and each category,
    compute triplet_category_score, n_pairs_used, triplet_prop_boost.
    Returns long DataFrame: ingA_id, ingB_id, ingC_id, category, triplet_category_score,
    n_pairs_used, triplet_prop_boost, support_ABC, PMI_ABC, lift_ABC (from triplets_df).
    """
    if triplets_df.empty or pair_mediation is None or pair_mediation.empty:
        return pd.DataFrame()
    cats = categories or list(pair_mediation["category"].dropna().unique().astype(str))
    if not cats:
        return pd.DataFrame()
    extra_cols = [c for c in ["support_ABC", "PMI_ABC", "lift_ABC"] if c in triplets_df.columns]
    rows = []
    for _, row in triplets_df.iterrows():
        a, b, c = row["ingA_id"], row["ingB_id"], row["ingC_id"]
        for cat in cats:
            score, n_pairs, prop_boost = triplet_category_score(a, b, c, cat, pair_mediation, use_propagation=use_propagation)
            r = {"ingA_id": a, "ingB_id": b, "ingC_id": c, "category": cat, "triplet_category_score": score, "n_pairs_used": n_pairs, "triplet_prop_boost": prop_boost}
            for col in extra_cols:
                r[col] = row[col]
            rows.append(r)
    return pd.DataFrame(rows)


def mine_and_score_triplets(
    recipe_ingredients: pd.DataFrame,
    pair_mediation: pd.DataFrame,
    min_support: Optional[int] = None,
    drop_top_freq_pct: Optional[float] = None,
    max_triplets: Optional[int] = None,
    smoke_n_recipes: Optional[int] = None,
    smoke: bool = False,
    # Bounded pipeline (production); overrides when provided
    use_bounded: bool = True,
    max_recipes: Optional[int] = None,
    max_unique_ingredients: Optional[int] = None,
    min_support_pairs: Optional[int] = None,
    min_support_triplets: Optional[int] = None,
    top_k_pairs: Optional[int] = None,
    max_triplets_scored: Optional[int] = None,
    time_budget_seconds: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Mine triplets then score by category. Uses bounded pipeline by default (production-safe).
    Returns (triplets_df, triplet_category_df, mining_summary or None).
    triplets_df columns: ingA_id, ingB_id, ingC_id, support_ABC, PMI_ABC, lift_ABC, source, sampled_recipes_n.
    triplet_category_df columns include: ingA_id, ingB_id, ingC_id, category, triplet_category_score, n_pairs_used.
    """
    mining_summary: Optional[Dict[str, Any]] = None
    if use_bounded:
        max_recipes = max_recipes if max_recipes is not None else getattr(config, "TRIPLET_MAX_RECIPES", 200_000)
        max_unique_ingredients = max_unique_ingredients if max_unique_ingredients is not None else getattr(config, "TRIPLET_MAX_UNIQUE_INGREDIENTS", 3000)
        min_support_pairs = min_support_pairs if min_support_pairs is not None else getattr(config, "TRIPLET_MIN_SUPPORT_PAIRS", 50)
        min_support_triplets = min_support_triplets if min_support_triplets is not None else getattr(config, "TRIPLET_MIN_SUPPORT_TRIPLETS", 25)
        top_k_pairs = top_k_pairs if top_k_pairs is not None else getattr(config, "TRIPLET_TOP_K_PAIRS", 20_000)
        max_triplets_scored = max_triplets_scored if max_triplets_scored is not None else getattr(config, "TRIPLET_MAX_TRIPLETS_SCORED", 50_000)
        time_budget_seconds = time_budget_seconds if time_budget_seconds is not None else getattr(config, "TRIPLET_TIME_BUDGET_SECONDS", 600.0)
        random_seed = random_seed if random_seed is not None else getattr(config, "TRIPLET_RANDOM_SEED", 42)
        if smoke and smoke_n_recipes:
            max_recipes = min(max_recipes, smoke_n_recipes)
        triplets_df, mining_summary = _mine_triplets_bounded(
            recipe_ingredients,
            max_recipes=max_recipes,
            max_unique_ingredients=max_unique_ingredients,
            min_support_pairs=min_support_pairs,
            min_support_triplets=min_support_triplets,
            top_k_pairs=top_k_pairs,
            max_triplets_scored=max_triplets_scored,
            time_budget_seconds=time_budget_seconds,
            random_seed=random_seed,
        )
    else:
        min_support = min_support or (config.TRIPLET_MIN_SUPPORT_SMOKE if smoke else config.TRIPLET_MIN_SUPPORT_FULL)
        drop_top_freq_pct = drop_top_freq_pct if drop_top_freq_pct is not None else config.TRIPLET_DROP_TOP_FREQ_PCT
        max_triplets = max_triplets or config.TRIPLET_MAX_TRIPLETS
        triplets_df = compute_supports_and_pmi(
            recipe_ingredients,
            min_support=min_support,
            drop_top_freq_pct=drop_top_freq_pct,
            max_triplets=max_triplets,
            smoke_n_recipes=smoke_n_recipes,
        )
        if "source" not in triplets_df.columns and not triplets_df.empty:
            triplets_df["source"] = "legacy"
            triplets_df["sampled_recipes_n"] = triplets_df.get("N", 0)

    if triplets_df.empty:
        return triplets_df, pd.DataFrame(), mining_summary
    categories = list(pair_mediation["category"].dropna().unique().astype(str)) if not pair_mediation.empty else None
    triplet_category_df = score_triplets_by_category(triplets_df, pair_mediation, categories=categories, use_propagation=True)
    return triplets_df, triplet_category_df, mining_summary


def mine_frequent_triplets(
    recipe_ingredients: pd.DataFrame,
    min_support: int = 50,
    max_count: int = 500,
    smoke_n_recipes: Optional[int] = None,
    top_k_ingredients: Optional[int] = None,
) -> List[Tuple[str, str, str, int]]:
    """
    Legacy API: mine triplets and return list of (ingA, ingB, ingC, count).
    Wraps compute_supports_and_pmi for backward compatibility.
    """
    drop = config.TRIPLET_DROP_TOP_FREQ_PCT if hasattr(config, "TRIPLET_DROP_TOP_FREQ_PCT") else 0.0
    df = compute_supports_and_pmi(
        recipe_ingredients,
        min_support=min_support,
        drop_top_freq_pct=drop,
        max_triplets=max_count,
        smoke_n_recipes=smoke_n_recipes,
    )
    if df.empty:
        return []
    return [(r["ingA_id"], r["ingB_id"], r["ingC_id"], int(r["support_ABC"])) for _, r in df.iterrows()]


def build_triplet_nodes_edges(
    triplets_df: pd.DataFrame,
    triplet_category_df: pd.DataFrame,
    run_id: str = "phase14",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build INT3 nodes and edges for Neo4j:
    - nodes: :ID = INT3_<A>_<B>_<C>, :LABEL = InteractionTriplet, name, source
    - edges_ing: INT3 -> ING (HAS_INGREDIENT)
    - edges_cat: INT3 -> CAT (AFFECTS) with support_ABC, PMI_ABC, lift_ABC, triplet_category_score, n_pairs_used, run_id
    Returns (nodes_df, edges_df) where edges_df = edges_ing + edges_cat.
    """
    if triplets_df.empty:
        return pd.DataFrame(columns=["node_id", "label", "name", "source"]), pd.DataFrame(columns=["source_id", "target_id", "edge_type"])
    node_rows = []
    seen = set()
    for _, r in triplets_df.iterrows():
        a, b, c = r["ingA_id"], r["ingB_id"], r["ingC_id"]
        nid = to_triplet_id(a, b, c)
        if nid in seen:
            continue
        seen.add(nid)
        node_rows.append({"node_id": nid, "label": "InteractionTriplet", "name": nid, "source": run_id})
    nodes_df = pd.DataFrame(node_rows)
    edge_rows = []
    for _, r in triplets_df.iterrows():
        a, b, c = r["ingA_id"], r["ingB_id"], r["ingC_id"]
        nid = to_triplet_id(a, b, c)
        for ing in (a, b, c):
            edge_rows.append({"source_id": nid, "target_id": ing, "edge_type": "HAS_INGREDIENT"})
    if not triplet_category_df.empty:
        for _, r in triplet_category_df.iterrows():
            if (r.get("triplet_category_score") or 0) <= 0:
                continue
            nid = to_triplet_id(r["ingA_id"], r["ingB_id"], r["ingC_id"])
            cat_id = to_category_id(str(r["category"]))
            edge_rows.append({
                "source_id": nid,
                "target_id": cat_id,
                "edge_type": "AFFECTS",
                "support_ABC": int(r.get("support_ABC", 0)),
                "PMI_ABC": float(r.get("PMI_ABC", 0)),
                "lift_ABC": float(r.get("lift_ABC", 0)),
                "triplet_category_score": float(r.get("triplet_category_score", 0)),
                "n_pairs_used": int(r.get("n_pairs_used", 0)),
                "run_id": run_id,
            })
    edges_df = pd.DataFrame(edge_rows)
    return nodes_df, edges_df
