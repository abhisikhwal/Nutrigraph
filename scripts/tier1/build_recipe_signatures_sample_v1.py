#!/usr/bin/env python3
"""
Distinctiveness-weighted recipe mechanistic signatures — validation sample only.

Aggregates ingredient category (Reactome top-level) and tissue v2 profiles into
recipe-level signatures using recipe-level species IDF to downweight staple backbone.

Usage (from repo root):
    python scripts/tier1/build_recipe_signatures_sample_v1.py

Outputs (new only):
    data/processed/tier1/recipe_signatures_sample_v1.parquet
    data/processed/tier1/recipe_signatures_sample_report_v1.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TIER1 = ROOT / "data/processed/tier1"
CANONICAL = ROOT / "data/processed/canonical"

RECIPES = CANONICAL / "recipes_expanded_v2.parquet"
RECIPE_ING = CANONICAL / "recipe_ingredients_expanded_v2.parquet"
STRING_MAP = CANONICAL / "ingredient_string_species_v2.parquet"
CATEGORY_PROFILES = TIER1 / "ingredient_category_profiles_v1.parquet"
TISSUE_PROFILES = TIER1 / "ingredient_tissue_profiles_v2.parquet"

OUT_PARQUET = TIER1 / "recipe_signatures_sample_v1.parquet"
OUT_REPORT = TIER1 / "recipe_signatures_sample_report_v1.json"

WEIGHTING_METHOD = "recipe_species_idf"
WEIGHTING_RATIONALE = (
    "Recipe-level species IDF (option a): w_i = log(N_recipes / df_i) where df_i is the "
    "number of distinct recipes containing species i. Staples (salt, onion, garlic, flour) "
    "appear in tens of thousands of recipes and receive near-zero weight; distinctive species "
    "retain influence. Ingredient category and tissue v2 profiles already encode enrichment "
    "distinctiveness at the species level (especially tissue v2); adding recipe IDF targets "
    "the recipe-level convergence/backbone problem directly without double-counting gene IDF."
)

# Diverse validation sample: 5 recognizable recipes per culinary type (40 total).
SAMPLE_TYPES: dict[str, list[str]] = {
    "curry": [
        "rnlg_1934086",  # Beef Korma Recipe
        "rnlg_916071",  # Lamb Shank Vindaloo
        "rnlg_65444",  # Tomato Chicken Curry
        "rnlg_1483274",  # Butter Chicken (Tikka Makhani)
        "rnlg_1145912",  # Pork Vindaloo With Raita
    ],
    "cake_bread": [
        "rnlg_4405",  # Mom's Apple Pie
        "rnlg_1020",  # Chocolate Cake
        "rnlg_6201",  # Coconut Pineapple Pie
        "rnlg_746",  # Pineapple Pie
        "rnlg_167",  # Banana Bread
    ],
    "salad": [
        "rnlg_593072",  # Greek Salad
        "rnlg_93792",  # Marinated Garden Salad
        "rnlg_26388",  # Spinach Salad
        "rnlg_34992",  # Spinach Salad And Dressing
        "rnlg_8410",  # Overnight Fruit Salad
    ],
    "stew": [
        "rnlg_223015",  # Jack's Lamb Stew
        "rnlg_25841",  # Ratatouille (Vegetable Stew)
        "rnlg_36525",  # Meat And Vegetable Stew
        "rnlg_434687",  # Delicious Lamb Stew
        "rnlg_52543",  # Lentil Soup II (legume-stew crossover)
    ],
    "fish": [
        "rnlg_226470",  # Baked Salmon Mousse
        "rnlg_234134",  # Gourmet Baked Salmon
        "rnlg_266308",  # Baked Salmon Molds
        "rnlg_552886",  # Cod Fish Cakes
        "rnlg_5067",  # Grilled Tuna
    ],
    "legume": [
        "rnlg_140095",  # Hummus Bi Tahini
        "rnlg_66672",  # Hummus Dip
        "rnlg_90101",  # Homemade Hummus
        "rnlg_14030",  # Cuban Black Bean Soup
        "rnlg_63314",  # Sasha's Favorite Lentil Soup
    ],
    "fruit_dessert": [
        "rnlg_546497",  # Raspberry Tart
        "rnlg_22888",  # Peach Cobbler
        "rnlg_8410",  # Overnight Fruit Salad
        "rnlg_622",  # Blueberry Muffins
        "rnlg_20",  # Banana Bread
    ],
    "fermented": [
        "rnlg_5464",  # Pickled Mushrooms
        "rnlg_899103",  # Kimchi Jigeh
        "rnlg_1753",  # Pickled Beets And Eggs
        "rnlg_2275",  # Pickled Cabbage
        "rnlg_480",  # Ma's Dill Pickles
    ],
}

# Fallback name search when a hard-coded id fails coverage gate.
TYPE_SEARCH: dict[str, str] = {
    "curry": "curry|korma|vindaloo|tikka masala|masala",
    "cake_bread": "banana bread|chocolate cake|apple pie|sourdough|muffin",
    "salad": "greek salad|spinach salad|garden salad|caesar salad|broccoli salad",
    "stew": "lamb stew|beef stew|vegetable stew|ratatouille|lentil soup",
    "fish": "baked salmon|grilled salmon|cod fish|tuna|fish fillet",
    "legume": "hummus|lentil soup|black bean soup|chickpea",
    "fruit_dessert": "apple pie|peach cobbler|raspberry tart|fruit salad",
    "fermented": "pickled|sauerkraut|kimchi|dill pickle",
}

STAPLE_BACKBONE_RECIPES = ["rnlg_1934086", "rnlg_4405", "rnlg_593072"]  # curry, pie, salad
SENSIBILITY_RECIPES = [
    "rnlg_1934086",
    "rnlg_4405",
    "rnlg_593072",
    "rnlg_226470",
    "rnlg_140095",
]
HERO_PREVIEW_RECIPES = ["rnlg_1934086", "rnlg_4405", "rnlg_140095"]

MIN_COVERAGE = 0.80
MIN_MAPPED = 5
PER_TYPE_TARGET = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shannon_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def load_live_string_map() -> pd.DataFrame:
    sm = pd.read_parquet(STRING_MAP)
    return sm[sm["mechanism_tier"] == "mechanistically_live"][
        ["ingredient_string", "species_node", "canonical_name"]
    ].copy()


def compute_species_idf(recipe_ing: pd.DataFrame, string_map: pd.DataFrame) -> dict[str, float]:
    ri = recipe_ing.copy()
    ri["ing_l"] = ri["ingredient_raw"].str.lower().str.strip()
    ri = ri.merge(string_map, left_on="ing_l", right_on="ingredient_string", how="inner")
    n_recipes = ri["recipe_id"].nunique()
    df = ri.groupby("species_node")["recipe_id"].nunique()
    return {str(sp): math.log(n_recipes / max(1, int(cnt))) for sp, cnt in df.items()}


def load_category_matrix() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    cat = pd.read_parquet(CATEGORY_PROFILES)
    sub = cat[
        (cat["enrichment_layer"] == "weighted_calibrated")
        & (cat["category_level"] == "top_level")
    ].copy()
    pivot_raw = sub.pivot_table(
        index="ingredient_id",
        columns="category_id",
        values="aggregated_enrichment",
        aggfunc="sum",
        fill_value=0.0,
    )
    row_sums = pivot_raw.sum(axis=1).replace(0, np.nan)
    pivot_norm = pivot_raw.div(row_sums, axis=0).fillna(0.0)
    id_to_name = (
        sub.drop_duplicates("category_id")
        .set_index("category_id")["category_name"]
        .astype(str)
        .to_dict()
    )
    return pivot_raw, pivot_norm, list(pivot_norm.columns.astype(str)), id_to_name


def load_tissue_matrix() -> tuple[pd.DataFrame, list[str]]:
    tis = pd.read_parquet(TISSUE_PROFILES)
    pivot = tis.pivot_table(
        index="ingredient_id",
        columns="tissue",
        values="normalized_score",
        aggfunc="sum",
        fill_value=0.0,
    )
    row_sums = pivot.sum(axis=1).replace(0, np.nan)
    pivot_norm = pivot.div(row_sums, axis=0).fillna(0.0)
    return pivot_norm, list(pivot_norm.columns.astype(str))


def recipe_species_list(
    recipe_id: str,
    recipe_ing: pd.DataFrame,
    string_map: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    sub = recipe_ing[recipe_ing["recipe_id"] == recipe_id].copy()
    sub["ing_l"] = sub["ingredient_raw"].str.lower().str.strip()
    mapped = sub.merge(string_map, left_on="ing_l", right_on="ingredient_string", how="left")
    species = mapped["species_node"].dropna().astype(str).unique().tolist()
    return species, mapped


def aggregate_signature(
    species_list: list[str],
    profile_mat: pd.DataFrame,
    feature_cols: list[str],
    weights: dict[str, float] | None,
) -> np.ndarray:
    vecs: list[np.ndarray] = []
    wts: list[float] = []
    for sp in species_list:
        if sp not in profile_mat.index:
            continue
        w = 1.0 if weights is None else weights.get(sp, 0.0)
        if w <= 0:
            continue
        vecs.append(profile_mat.loc[sp, feature_cols].to_numpy(dtype=float))
        wts.append(w)
    if not vecs:
        return np.zeros(len(feature_cols))
    w_arr = np.array(wts, dtype=float)
    mat = np.vstack(vecs)
    return (w_arr[:, None] * mat).sum(axis=0) / w_arr.sum()


def ingredient_load(species_list: list[str], profile_mat: pd.DataFrame) -> dict[str, float]:
    loads: dict[str, float] = {}
    for sp in species_list:
        if sp in profile_mat.index:
            loads[sp] = float(profile_mat.loc[sp].sum())
    return loads


def top_k_from_vector(
    vec: np.ndarray,
    labels: list[str],
    id_to_name: dict[str, str] | None,
    k: int = 5,
) -> list[dict[str, Any]]:
    idx = np.argsort(vec)[::-1][:k]
    out = []
    for i in idx:
        if vec[i] <= 0:
            continue
        label = labels[i]
        name = id_to_name.get(label, label) if id_to_name else label
        out.append({"id": label, "name": name, "score": round(float(vec[i]), 6)})
    return out


def hero_contributions(
    species_list: list[str],
    profile_mat_raw: pd.DataFrame,
    idf: dict[str, float],
    canonical: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for sp in species_list:
        if sp not in profile_mat_raw.index:
            continue
        w = idf.get(sp, 0.0)
        load = float(profile_mat_raw.loc[sp].sum())
        rows.append(
            {
                "species_node": sp,
                "canonical_name": canonical.get(sp, sp),
                "idf_weight": round(w, 4),
                "raw_enrichment_load": round(load, 4),
                "contribution": round(w * load, 4),
            }
        )
    rows.sort(key=lambda r: r["contribution"], reverse=True)
    return rows[:10]


def discover_sample_ids(
    recipes: pd.DataFrame,
    recipe_ing: pd.DataFrame,
    string_map: pd.DataFrame,
    seed: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Ensure PER_TYPE_TARGET recipes per type with coverage >= MIN_COVERAGE."""
    chosen: dict[str, list[str]] = {}
    used: set[str] = set()

    def coverage_for(rid: str) -> tuple[float, int, int]:
        sp, mapped = recipe_species_list(rid, recipe_ing, string_map)
        n = len(mapped)
        m = mapped["species_node"].notna().sum()
        return (m / n if n else 0.0, int(n), int(m))

    for rtype, ids in seed.items():
        picked: list[str] = []
        for rid in ids:
            if rid in used:
                continue
            cov, n, m = coverage_for(rid)
            if cov >= MIN_COVERAGE and m >= MIN_MAPPED:
                picked.append(rid)
                used.add(rid)
            if len(picked) >= PER_TYPE_TARGET:
                break
        if len(picked) < PER_TYPE_TARGET:
            pat = TYPE_SEARCH[rtype]
            cands = recipes[recipes["name"].str.contains(pat, case=False, na=False, regex=True)]
            for rid in cands["recipe_id"].astype(str):
                if rid in used:
                    continue
                cov, n, m = coverage_for(rid)
                if cov >= MIN_COVERAGE and m >= MIN_MAPPED:
                    picked.append(rid)
                    used.add(rid)
                if len(picked) >= PER_TYPE_TARGET:
                    break
        chosen[rtype] = picked
    return chosen


HUB_CATEGORIES = {"GO_NS:biological_process"}


def hub_excluded_similarity_summary(
    cat_vecs: dict[str, np.ndarray],
    cat_cols: list[str],
    type_by_id: dict[str, str],
) -> dict[str, Any]:
    hub_idx = {i for i, c in enumerate(cat_cols) if c in HUB_CATEGORIES}
    ids = list(cat_vecs.keys())
    cross, within = [], []
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            va = cat_vecs[a].copy()
            vb = cat_vecs[b].copy()
            for idx in hub_idx:
                va[idx] = 0.0
                vb[idx] = 0.0
            v = cosine_sim(va, vb)
            (within if type_by_id[a] == type_by_id[b] else cross).append(v)
    return {
        "excluded_categories": sorted(HUB_CATEGORIES),
        "mean_cross_type": round(float(np.mean(cross)), 4) if cross else None,
        "mean_within_type": round(float(np.mean(within)), 4) if within else None,
        "min_cross_type": round(float(np.min(cross)), 4) if cross else None,
    }


def build_report(
    sample_meta: list[dict[str, Any]],
    cat_vecs_weighted: dict[str, np.ndarray],
    cat_vecs_naive: dict[str, np.ndarray],
    cat_cols: list[str],
    id_to_name: dict[str, str],
    tissue_vecs: dict[str, np.ndarray],
    tissue_cols: list[str],
    idf: dict[str, float],
    canonical: dict[str, str],
    cat_mat: pd.DataFrame,
    recipe_mappings: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    ids = [m["recipe_id"] for m in sample_meta]
    name_by_id = {m["recipe_id"]: m["recipe_name"] for m in sample_meta}

    # Pairwise similarity (weighted category signatures).
    sim = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for i in ids:
        for j in ids:
            sim.loc[i, j] = cosine_sim(cat_vecs_weighted[i], cat_vecs_weighted[j])

    # Cross-type vs within-type summary.
    type_by_id = {m["recipe_id"]: m["recipe_type"] for m in sample_meta}
    cross, within = [], []
    for i in ids:
        for j in ids:
            if i >= j:
                continue
            v = float(sim.loc[i, j])
            if type_by_id[i] == type_by_id[j]:
                within.append(v)
            else:
                cross.append(v)

    # Staple backbone: naive vs weighted top categories.
    backbone = {}
    for rid in STAPLE_BACKBONE_RECIPES:
        if rid not in cat_vecs_naive:
            continue
        backbone[rid] = {
            "recipe_name": name_by_id[rid],
            "recipe_type": type_by_id.get(rid),
            "naive_top5_categories": top_k_from_vector(
                cat_vecs_naive[rid], cat_cols, id_to_name, 5
            ),
            "idf_weighted_top5_categories": top_k_from_vector(
                cat_vecs_weighted[rid], cat_cols, id_to_name, 5
            ),
        }

    # Sensibility eyeball.
    sensibility = {}
    for rid in SENSIBILITY_RECIPES:
        if rid not in cat_vecs_weighted:
            continue
        sensibility[rid] = {
            "recipe_name": name_by_id[rid],
            "recipe_type": type_by_id.get(rid),
            "top5_categories": top_k_from_vector(
                cat_vecs_weighted[rid], cat_cols, id_to_name, 5
            ),
            "top5_tissues": top_k_from_vector(
                tissue_vecs[rid], tissue_cols, None, 5
            ),
        }

    # Hero preview.
    hero = {}
    for rid in HERO_PREVIEW_RECIPES:
        sp_list = [x["species_node"] for x in recipe_mappings[rid]]
        hero[rid] = {
            "recipe_name": name_by_id[rid],
            "recipe_type": type_by_id.get(rid),
            "top_contributors": hero_contributions(sp_list, cat_mat, idf, canonical),
        }

    return {
        "pairwise_category_cosine": sim.round(4).to_dict(),
        "similarity_summary": {
            "mean_cross_type": round(float(np.mean(cross)), 4) if cross else None,
            "mean_within_type": round(float(np.mean(within)), 4) if within else None,
            "min_cross_type": round(float(np.min(cross)), 4) if cross else None,
            "max_cross_type": round(float(np.max(cross)), 4) if cross else None,
        },
        "hub_excluded_similarity": hub_excluded_similarity_summary(
            cat_vecs_weighted, cat_cols, type_by_id
        ),
        "staple_backbone_check": backbone,
        "sensibility_eyeball": sensibility,
        "hero_preview": hero,
    }


def main() -> int:
    print("=== Recipe signature sample build (validation) ===", flush=True)
    TIER1.mkdir(parents=True, exist_ok=True)

    recipes = pd.read_parquet(RECIPES)
    recipe_ing = pd.read_parquet(RECIPE_ING)
    string_map = load_live_string_map()
    canonical = (
        pd.read_parquet(STRING_MAP)
        .groupby("species_node")["canonical_name"]
        .first()
        .astype(str)
        .to_dict()
    )

    print("  Computing species recipe-level IDF from full corpus...", flush=True)
    idf = compute_species_idf(recipe_ing, string_map)
    print(f"  Species with IDF: {len(idf)}", flush=True)

    cat_mat_raw, cat_mat, cat_cols, id_to_name = load_category_matrix()
    tis_mat, tis_cols = load_tissue_matrix()
    print(f"  Category features: {len(cat_cols)}, tissue features: {len(tis_cols)}", flush=True)

    sample_by_type = discover_sample_ids(recipes, recipe_ing, string_map, SAMPLE_TYPES)
    flat: list[tuple[str, str, str]] = []
    for rtype, ids in sample_by_type.items():
        for rid in ids:
            flat.append((rid, rtype, recipes.loc[recipes["recipe_id"] == rid, "name"].iloc[0]))

    print(f"  Validation sample size: {len(flat)} recipes across {len(sample_by_type)} types", flush=True)

    rows_out: list[dict[str, Any]] = []
    cat_vecs_w: dict[str, np.ndarray] = {}
    cat_vecs_n: dict[str, np.ndarray] = {}
    tis_vecs: dict[str, np.ndarray] = {}
    recipe_mappings: dict[str, list[dict[str, str]]] = {}

    for rid, rtype, rname in flat:
        species, mapped = recipe_species_list(rid, recipe_ing, string_map)
        n_ing = len(mapped)
        n_mapped = int(mapped["species_node"].notna().sum())
        coverage = n_mapped / n_ing if n_ing else 0.0

        mapping_rows = []
        for _, r in mapped.iterrows():
            mapping_rows.append(
                {
                    "ingredient_raw": r["ingredient_raw"],
                    "species_node": r["species_node"] if pd.notna(r["species_node"]) else None,
                    "canonical_name": r["canonical_name"] if pd.notna(r.get("canonical_name")) else None,
                }
            )
        recipe_mappings[rid] = [m for m in mapping_rows if m["species_node"]]

        cat_w = aggregate_signature(species, cat_mat, cat_cols, idf)
        cat_n = aggregate_signature(species, cat_mat, cat_cols, None)
        tis_w = aggregate_signature(species, tis_mat, tis_cols, idf)
        cat_vecs_w[rid] = cat_w
        cat_vecs_n[rid] = cat_n
        tis_vecs[rid] = tis_w

        # Renormalize recipe vectors to simplex for interpretability.
        if cat_w.sum() > 0:
            cat_w = cat_w / cat_w.sum()
        if cat_n.sum() > 0:
            cat_n = cat_n / cat_n.sum()
        if tis_w.sum() > 0:
            tis_w = tis_w / tis_w.sum()

        loads = ingredient_load(species, cat_mat_raw)
        total_load = sum(loads.values())

        rows_out.append(
            {
                "recipe_id": rid,
                "recipe_name": rname,
                "recipe_type": rtype,
                "n_ingredients": n_ing,
                "n_mapped_species": n_mapped,
                "coverage": round(coverage, 4),
                "weighting_method": WEIGHTING_METHOD,
                "category_signature": json.dumps(
                    {id_to_name.get(c, c): float(v) for c, v in zip(cat_cols, cat_w) if v > 0}
                ),
                "tissue_signature": json.dumps(
                    {t: float(v) for t, v in zip(tis_cols, tis_w) if v > 0}
                ),
                "top5_categories": json.dumps(
                    top_k_from_vector(cat_w, cat_cols, id_to_name, 5)
                ),
                "top5_tissues": json.dumps(top_k_from_vector(tis_w, tis_cols, None, 5)),
                "category_shannon": round(shannon_entropy(cat_w), 4),
                "total_enrichment_load": round(total_load, 4),
                "ingredient_mappings": json.dumps(mapping_rows),
            }
        )

    out_df = pd.DataFrame(rows_out)
    out_df.to_parquet(OUT_PARQUET, index=False)

    validation = build_report(
        sample_meta=out_df[["recipe_id", "recipe_name", "recipe_type"]].to_dict("records"),
        cat_vecs_weighted=cat_vecs_w,
        cat_vecs_naive=cat_vecs_n,
        cat_cols=cat_cols,
        id_to_name=id_to_name,
        tissue_vecs=tis_vecs,
        tissue_cols=tis_cols,
        idf=idf,
        canonical=canonical,
        cat_mat=cat_mat_raw,
        recipe_mappings=recipe_mappings,
    )

    report = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "weighting_method": WEIGHTING_METHOD,
        "weighting_rationale": WEIGHTING_RATIONALE,
        "inputs": {
            "recipes": str(RECIPES),
            "recipe_ingredients": str(RECIPE_ING),
            "string_map": str(STRING_MAP),
            "category_profiles": str(CATEGORY_PROFILES),
            "tissue_profiles": str(TISSUE_PROFILES),
        },
        "input_hashes": {
            "recipes_expanded_v2": sha256_file(RECIPES),
            "recipe_ingredients_expanded_v2": sha256_file(RECIPE_ING),
            "ingredient_string_species_v2": sha256_file(STRING_MAP),
            "ingredient_category_profiles_v1": sha256_file(CATEGORY_PROFILES),
            "ingredient_tissue_profiles_v2": sha256_file(TISSUE_PROFILES),
        },
        "sample_size": len(flat),
        "sample_by_type": {k: v for k, v in sample_by_type.items()},
        "curated_recipes": [
            {
                "recipe_id": rid,
                "recipe_name": rname,
                "recipe_type": rtype,
                "ingredient_mappings": recipe_mappings[rid],
            }
            for rid, rtype, rname in flat
        ],
        "validation": validation,
        "outputs": {
            "parquet": str(OUT_PARQUET),
            "report": str(OUT_REPORT),
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  Wrote {OUT_PARQUET} ({len(out_df)} rows)", flush=True)
    print(f"  Wrote {OUT_REPORT}", flush=True)
    ss = validation["similarity_summary"]
    print(
        f"  Cross-type mean cosine={ss['mean_cross_type']} "
        f"within-type mean={ss['mean_within_type']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
