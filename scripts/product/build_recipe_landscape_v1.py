#!/usr/bin/env python3
"""
Recipe-message landscape discovery — run engine at scale, cluster, characterize.

Usage (from repo root):
    python scripts/product/build_recipe_landscape_v1.py

Outputs (new only):
    data/processed/product/landscape/
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, SCRIPT_DIR)

from recipe_message_engine import (  # noqa: E402
    REGION_IDS,
    UNIQUE_CONTRIBUTOR_CHANGE_THRESHOLD,
    RecipeMessageEngine,
)

OUT_DIR = ROOT / "data/processed/product/landscape"
RECIPES = ROOT / "data/processed/canonical/recipes_expanded_v2.parquet"
RECIPE_ING = ROOT / "data/processed/canonical/recipe_ingredients_expanded_v2.parquet"

TARGET_N = 400
MIN_COVERAGE = 0.70
MIN_RESOLVED = 3
RANDOM_SEED = 42

SIZE_BUCKETS = {
    "small": (3, 5),
    "medium": (6, 10),
    "large": (11, 999),
}

COMPOSITION_KEYWORDS: dict[str, set[str]] = {
    "spice_heavy": {
        "turmeric", "cumin", "coriander", "ginger", "chili", "chilli", "pepper",
        "cinnamon", "clove", "cloves", "cardamom", "paprika", "nutmeg", "allspice",
        "fenugreek", "mustard seed", "curry", "garam masala",
    },
    "meat_containing": {
        "chicken", "beef", "pork", "lamb", "bacon", "sausage", "turkey", "ham",
        "steak", "ground beef", "mince", "veal", "duck", "prosciutto",
    },
    "legume": {
        "lentil", "lentils", "bean", "beans", "chickpea", "chickpeas", "pea",
        "peas", "black bean", "kidney bean", "garbanzo", "dal", "tofu",
    },
    "produce_heavy": {
        "lettuce", "spinach", "kale", "cucumber", "tomato", "tomatoes", "carrot",
        "carrots", "broccoli", "celery", "zucchini", "squash", "pepper", "onion",
        "garlic", "apple", "banana", "orange", "lemon", "lime", "avocado",
        "mushroom", "mushrooms", "salad", "greens", "cabbage",
    },
    "baked_starch": {
        "flour", "sugar", "butter", "baking powder", "baking soda", "yeast",
        "bread", "pasta", "noodle", "noodles", "rice", "oat", "oats", "cornstarch",
        "cake", "cookie", "pastry", "dough", "breadcrumb", "breadcrumbs",
    },
    "dairy_heavy": {
        "milk", "cheese", "cream", "yogurt", "yoghurt", "butter", "parmesan",
        "mozzarella", "cheddar", "sour cream", "whipped cream", "ghee",
    },
    "seafood": {
        "salmon", "fish", "shrimp", "prawn", "tuna", "cod", "crab", "lobster",
        "scallop", "anchovy", "sardine", "trout", "seafood", "mussel", "clam",
    },
}

COMPOSITION_PRIORITY = [
    "seafood",
    "meat_containing",
    "spice_heavy",
    "legume",
    "dairy_heavy",
    "baked_starch",
    "produce_heavy",
    "mixed",
]


def normalize_theme_vector(theme_dict: dict[str, float], theme_ids: list[str]) -> dict[str, float]:
    vals = {tid: float(theme_dict.get(tid, 0.0)) for tid in theme_ids}
    total = sum(vals.values())
    if total <= 0:
        return vals
    return {k: v / total for k, v in vals.items()}


def classify_composition(ingredient_strings: list[str]) -> str:
    raw = [s.strip().lower() for s in ingredient_strings]
    scores: dict[str, int] = defaultdict(int)
    for comp, keywords in COMPOSITION_KEYWORDS.items():
        for ing in raw:
            for kw in keywords:
                if kw in ing or ing in kw:
                    scores[comp] += 1
                    break
    if not scores:
        return "mixed"
    best = max(scores.values())
    candidates = [c for c, s in scores.items() if s == best]
    for pref in COMPOSITION_PRIORITY:
        if pref in candidates:
            return pref
    return candidates[0]


def size_bucket(n_resolved: int) -> str:
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= n_resolved <= hi:
            return name
    return "large"


def build_resolvable_set(engine: RecipeMessageEngine) -> set[str]:
    keys: set[str] = set()
    keys.update(engine.string_map.keys())
    keys.update(engine.aliases.keys())
    return keys


def ingredient_resolves(raw: str, engine: RecipeMessageEngine, resolvable: set[str]) -> bool:
    key = raw.strip().lower()
    if key in resolvable:
        sid = engine.aliases.get(key) or engine.string_map.get(key)
        return bool(sid and sid in engine.profiles)
    for alias in resolvable:
        if key in alias or alias in key:
            sid = engine.aliases.get(alias) or engine.string_map.get(alias)
            if sid and sid in engine.profiles:
                return True
    return False


def load_recipe_candidates(engine: RecipeMessageEngine) -> pd.DataFrame:
    """Scan corpus (stratified by source) and score coverage + composition."""
    recipes = pd.read_parquet(RECIPES, columns=["recipe_id", "name", "source", "n_ingredients"])
    ing = pd.read_parquet(RECIPE_ING, columns=["recipe_id", "ingredient_raw"])
    ing["ingredient_raw"] = ing["ingredient_raw"].astype(str).str.strip()
    ing["raw_lower"] = ing["ingredient_raw"].str.lower()

    resolvable = build_resolvable_set(engine)

    # Deliberate source mix — avoid 97% RecipeNLG monoculture in sample
    source_caps = {
        "indian_food": 9999,
        "epicurious": 200,
        "foodcom": 300,
        "recipenlg": 800,
    }
    frames: list[pd.DataFrame] = []
    for source, cap in source_caps.items():
        rids = recipes[recipes["source"] == source]["recipe_id"].tolist()
        if len(rids) > cap:
            rng = np.random.default_rng(RANDOM_SEED)
            rids = rng.choice(rids, size=cap, replace=False).tolist()
        sub = ing[ing["recipe_id"].isin(rids)]
        frames.append(sub)
    pool = pd.concat(frames, ignore_index=True)

    rows: list[dict[str, Any]] = []
    grouped = pool.groupby("recipe_id")
    meta = recipes.set_index("recipe_id")

    for recipe_id, grp in grouped:
        if recipe_id not in meta.index:
            continue
        strings = grp["ingredient_raw"].tolist()
        n_input = len(strings)
        if n_input < MIN_RESOLVED:
            continue
        resolved_flags = [ingredient_resolves(s, engine, resolvable) for s in strings]
        n_resolved = sum(resolved_flags)
        coverage = n_resolved / n_input
        if coverage < MIN_COVERAGE or n_resolved < MIN_RESOLVED:
            continue
        resolved_strings = [s for s, ok in zip(strings, resolved_flags) if ok]
        comp = classify_composition(strings)
        sb = size_bucket(n_resolved)
        info = meta.loc[recipe_id]
        rows.append(
            {
                "recipe_id": recipe_id,
                "name": str(info["name"]),
                "source": str(info["source"]),
                "n_ingredients_corpus": int(info["n_ingredients"]),
                "n_input": n_input,
                "n_resolved": n_resolved,
                "coverage": round(coverage, 4),
                "size_bucket": sb,
                "composition_type": comp,
                "ingredient_strings": strings,
            }
        )

    return pd.DataFrame(rows)


def stratified_sample(candidates: pd.DataFrame, target: int = TARGET_N) -> pd.DataFrame:
    """Fill grid: size_bucket × composition_type, then top up to target."""
    shuffled = candidates.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in shuffled.iterrows():
        cells[(row["size_bucket"], row["composition_type"])].append(i)

    per_cell = max(10, target // max(1, len(cells)))
    picked_idx: list[int] = []
    used: set[int] = set()

    for key in sorted(cells.keys()):
        for i in cells[key][:per_cell]:
            if i not in used:
                picked_idx.append(i)
                used.add(i)

    for i in range(len(shuffled)):
        if len(picked_idx) >= target:
            break
        if i not in used:
            picked_idx.append(i)
            used.add(i)

    return shuffled.iloc[picked_idx[:target]].reset_index(drop=True)


def message_to_record(msg, theme_ids: list[str]) -> dict[str, Any]:
    theme_raw = {t["theme_id"]: t["strength"] for t in msg.effect_themes}
    theme_norm = normalize_theme_vector(
        {tid: theme_raw.get(tid, 0.0) for tid in theme_ids},
        theme_ids,
    )
    redundancy = msg.redundancy or []
    red_scores = [r["redundancy_score"] for r in redundancy]
    unique = [r["canonical_name"] for r in redundancy if r.get("is_unique_contributor")]

    top_regions = sorted(msg.body_regions.items(), key=lambda x: -x[1])[:5]
    top_themes = sorted(theme_norm.items(), key=lambda x: -x[1])[:5]

    return {
        "recipe_id": msg.recipe_id,
        "recipe_label": msg.recipe_label,
        "n_resolved": len(msg.resolved),
        "body_vector": {k: round(msg.body_regions.get(k, 0.0), 6) for k in REGION_IDS},
        "theme_vector_raw": {k: round(theme_raw.get(k, 0.0), 6) for k in theme_ids},
        "theme_vector_norm": {k: round(theme_norm.get(k, 0.0), 6) for k in theme_ids},
        "measured_fraction": msg.evidence_composition.get("measured_edge_fraction", 0.0),
        "redundancy_mean": round(float(np.mean(red_scores)), 4) if red_scores else None,
        "redundancy_max": round(float(max(red_scores)), 4) if red_scores else None,
        "redundancy_min": round(float(min(red_scores)), 4) if red_scores else None,
        "n_unique_contributors": len(unique),
        "unique_contributors": unique,
        "top_regions": [{"region": r, "intensity": round(v, 4)} for r, v in top_regions],
        "top_themes": [{"theme_id": t, "strength": round(v, 4)} for t, v in top_themes],
        "region_drivers": {
            r: drivers[:3]
            for r, drivers in msg.region_drivers.items()
            if drivers
        },
        "notable_pharmacology": msg.notable_pharmacology[:5],
    }


def cluster_analysis(
    matrix: np.ndarray,
    labels_names: list[str],
    recipe_ids: list[str],
    k_range: range = range(2, 9),
) -> dict[str, Any]:
    results: dict[str, Any] = {"silhouette_by_k": {}, "best_k": None, "best_silhouette": -1.0}
    n = matrix.shape[0]
    if n < 10:
        return {**results, "note": "too few samples for clustering"}

    best_labels = None
    for k in k_range:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        lbl = km.fit_predict(matrix)
        if len(set(lbl)) < 2:
            continue
        sil = silhouette_score(matrix, lbl)
        results["silhouette_by_k"][str(k)] = round(float(sil), 4)
        if sil > results["best_silhouette"]:
            results["best_silhouette"] = round(float(sil), 4)
            results["best_k"] = k
            best_labels = lbl

    if best_labels is None:
        results["verdict"] = "no_stable_clusters"
        return results

    results["verdict"] = (
        "weak_structure" if results["best_silhouette"] < 0.15
        else "moderate_structure" if results["best_silhouette"] < 0.35
        else "clear_clusters"
    )

    clusters: dict[int, dict[str, Any]] = {}
    for cid in sorted(set(best_labels)):
        mask = best_labels == cid
        sub = matrix[mask]
        centroid = sub.mean(axis=0)
        top_dims = np.argsort(-centroid)[:5]
        clusters[int(cid)] = {
            "n_recipes": int(mask.sum()),
            "fraction": round(float(mask.sum()) / n, 4),
            "defining_dimensions": [
                {"name": labels_names[i], "mean_intensity": round(float(centroid[i]), 4)}
                for i in top_dims
            ],
            "example_recipe_ids": [recipe_ids[i] for i in np.where(mask)[0][:8].tolist()],
        }
    results["clusters"] = clusters
    results["labels"] = best_labels.tolist()
    return results


def analyze_landscape(records: list[dict], sample_df: pd.DataFrame, theme_ids: list[str]) -> dict[str, Any]:
    n = len(records)
    body_mat = np.array([[r["body_vector"][rid] for rid in REGION_IDS] for r in records])
    theme_mat = np.array([[r["theme_vector_norm"][tid] for tid in theme_ids] for r in records])

    # Body region distribution
    body_stats: dict[str, Any] = {}
    for i, rid in enumerate(REGION_IDS):
        col = body_mat[:, i]
        body_stats[rid] = {
            "mean": round(float(col.mean()), 4),
            "std": round(float(col.std()), 4),
            "p50": round(float(np.median(col)), 4),
            "p90": round(float(np.percentile(col, 90)), 4),
            "max": round(float(col.max()), 4),
            "pct_recipes_above_0.10": round(float((col > 0.10).mean()), 4),
            "pct_recipes_above_0.25": round(float((col > 0.25).mean()), 4),
        }

    liver = body_mat[:, REGION_IDS.index("liver")]
    gut = body_mat[:, REGION_IDS.index("gut")]
    brain = body_mat[:, REGION_IDS.index("brain")]
    heart = body_mat[:, REGION_IDS.index("heart")]

    body_landscape = {
        "region_stats": body_stats,
        "liver_dominant_fraction": round(float((liver == body_mat.max(axis=1)).mean()), 4),
        "liver_gut_combined_mean": round(float((liver + gut).mean()), 4),
        "pct_liver_above_0.40": round(float((liver > 0.40).mean()), 4),
        "pct_brain_above_0.05": round(float((brain > 0.05).mean()), 4),
        "pct_heart_above_0.05": round(float((heart > 0.05).mean()), 4),
        "always_dark_regions": [
            rid for rid in REGION_IDS
            if body_stats[rid]["p90"] < 0.05
        ],
        "honest_summary": "",
    }
    body_landscape["honest_summary"] = (
        f"Liver is top region in {body_landscape['liver_dominant_fraction']:.0%} of recipes; "
        f"mean liver={body_stats['liver']['mean']:.2f}, gut={body_stats['gut']['mean']:.2f}. "
        f"Brain lights up (>5%) in {body_landscape['pct_brain_above_0.05']:.0%} of recipes."
    )

    recipe_ids = [r["recipe_id"] for r in records]
    body_cluster = cluster_analysis(body_mat, REGION_IDS, recipe_ids)
    theme_cluster = cluster_analysis(theme_mat, theme_ids, recipe_ids)

    # Cross-tab body cluster × composition (use best k labels)
    cross_tab: dict[str, Any] = {}
    if "labels" in body_cluster:
        comp_by_cluster: dict[int, Counter] = defaultdict(Counter)
        for i, lbl in enumerate(body_cluster["labels"]):
            comp = sample_df.iloc[i]["composition_type"]
            comp_by_cluster[lbl][comp] += 1
        cross_tab = {
            str(cid): dict(comp_by_cluster[cid].most_common())
            for cid in comp_by_cluster
        }

    # Redundancy analysis
    red_means = [r["redundancy_mean"] for r in records if r["redundancy_mean"] is not None]
    n_ing = sample_df["n_resolved"].values
    red_arr = np.array(red_means)
    n_arr = n_ing[: len(red_arr)]

    # Simple linear correlation
    if len(red_arr) > 2 and n_arr.std() > 0:
        corr = float(np.corrcoef(n_arr, red_arr)[0, 1])
    else:
        corr = 0.0

    unique_ing_counter: Counter = Counter()
    recipes_with_unique = 0
    for r in records:
        if r["n_unique_contributors"] > 0:
            recipes_with_unique += 1
        for ing in r.get("unique_contributors", []):
            unique_ing_counter[ing] += 1

    redundancy = {
        "mean_redundancy": round(float(np.mean(red_arr)), 4) if len(red_arr) else None,
        "p50_redundancy": round(float(np.median(red_arr)), 4) if len(red_arr) else None,
        "p90_redundancy": round(float(np.percentile(red_arr, 90)), 4) if len(red_arr) else None,
        "curry_benchmark_0.96_context": "Hand-picked spice curry avg ~0.96",
        "correlation_n_ingredients_vs_redundancy": round(corr, 4),
        "pct_recipes_with_any_unique_contributor": round(recipes_with_unique / n, 4),
        "top_unique_contributor_ingredients": unique_ing_counter.most_common(15),
    }

    # Outliers — distance from body centroid
    centroid = body_mat.mean(axis=0)
    dists = np.linalg.norm(body_mat - centroid, axis=1)
    outlier_idx = np.argsort(-dists)[:15]
    outliers = []
    for i in outlier_idx:
        rec = records[i]
        meta = sample_df.iloc[i]
        outliers.append(
            {
                "recipe_id": rec["recipe_id"],
                "name": meta["name"],
                "composition_type": meta["composition_type"],
                "source": meta["source"],
                "distance_from_centroid": round(float(dists[i]), 4),
                "body_vector": rec["body_vector"],
                "top_regions": rec["top_regions"],
            }
        )

    # Region discrimination by composition type
    comp_region_means: dict[str, dict[str, float]] = {}
    for comp in sample_df["composition_type"].unique():
        mask = sample_df["composition_type"] == comp
        idxs = np.where(mask.values[: len(records)])[0]
        if len(idxs) == 0:
            continue
        sub = body_mat[idxs]
        comp_region_means[comp] = {
            REGION_IDS[j]: round(float(sub[:, j].mean()), 4) for j in range(len(REGION_IDS))
        }

    surprising: list[str] = []
    if body_landscape["liver_dominant_fraction"] > 0.75:
        surprising.append(
            f"Flat liver dominance: liver is #1 region in {body_landscape['liver_dominant_fraction']:.0%} of recipes."
        )
    if body_cluster.get("best_silhouette", 0) < 0.15:
        surprising.append(
            f"Body vectors do not form strong clusters (best silhouette={body_cluster.get('best_silhouette')})."
        )
    if redundancy["mean_redundancy"] and redundancy["mean_redundancy"] > 0.85:
        surprising.append(
            f"High corpus-wide redundancy (mean={redundancy['mean_redundancy']}) — distributed signal is the norm."
        )
    dark = body_landscape["always_dark_regions"]
    if "brain" in dark or body_stats["brain"]["p90"] < 0.02:
        surprising.append(
            "Brain region essentially never lights up — likely tissue-truncation + no CNS delivery in model."
        )

    return {
        "n_recipes": n,
        "body_landscape": body_landscape,
        "body_clustering": body_cluster,
        "theme_clustering": theme_cluster,
        "composition_by_body_cluster": cross_tab,
        "composition_region_means": comp_region_means,
        "redundancy": redundancy,
        "outliers": outliers,
        "surprising_findings": surprising,
    }


def main() -> None:
    print("Loading message engine ...")
    engine = RecipeMessageEngine()
    theme_ids = engine.theme_ids

    print("Scanning recipe corpus for candidates (stratified sources) ...")
    candidates = load_recipe_candidates(engine)
    print(f"Candidates with >={MIN_COVERAGE:.0%} coverage: {len(candidates)}")

    sample = stratified_sample(candidates, TARGET_N)
    print(f"Sampled {len(sample)} recipes")

    sample_meta = sample[
        ["recipe_id", "name", "source", "size_bucket", "composition_type", "n_resolved", "coverage"]
    ].copy()

    print("Running message engine ...")
    records: list[dict[str, Any]] = []
    for i, row in sample.iterrows():
        msg = engine.compute_message(
            ingredient_strings=row["ingredient_strings"],
            recipe_id=str(row["recipe_id"]),
            recipe_label=str(row["name"]),
        )
        rec = message_to_record(msg, theme_ids)
        rec["source"] = row["source"]
        rec["composition_type"] = row["composition_type"]
        rec["size_bucket"] = row["size_bucket"]
        rec["coverage"] = row["coverage"]
        records.append(rec)
        if (len(records) % 50) == 0:
            print(f"  {len(records)}/{len(sample)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_meta["recipe_id"] = sample_meta["recipe_id"].astype(str)
    sample_path = OUT_DIR / "sample_composition_v1.json"
    sample_path.write_text(
        json.dumps(
            {
                "target_n": TARGET_N,
                "actual_n": len(sample),
                "min_coverage": MIN_COVERAGE,
                "stratification": {
                    "size_buckets": SIZE_BUCKETS,
                    "composition_types": COMPOSITION_PRIORITY,
                    "source_caps": {
                        "indian_food": "all",
                        "epicurious": 200,
                        "foodcom": 300,
                        "recipenlg": 800,
                    },
                },
                "by_size_bucket": sample_meta["size_bucket"].value_counts().to_dict(),
                "by_composition": sample_meta["composition_type"].value_counts().to_dict(),
                "by_source": sample_meta["source"].value_counts().to_dict(),
                "cross_tab": {
                    f"{a}|{b}": int(v)
                    for (a, b), v in sample_meta.groupby(["size_bucket", "composition_type"]).size().items()
                },
                "recipes": sample_meta.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    messages_path = OUT_DIR / "recipe_messages_v1.jsonl"
    with messages_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Analyzing landscape ...")
    analysis = analyze_landscape(records, sample_meta.reset_index(drop=True), theme_ids)
    analysis["generated_at"] = datetime.now(timezone.utc).isoformat()
    analysis["sample_composition"] = {
        "by_size_bucket": sample_meta["size_bucket"].value_counts().to_dict(),
        "by_composition": sample_meta["composition_type"].value_counts().to_dict(),
        "by_source": sample_meta["source"].value_counts().to_dict(),
    }

    report_path = OUT_DIR / "landscape_report_v1.json"
    report_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {sample_path.relative_to(ROOT)}")
    print(f"Wrote {messages_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")
    print("\n=== LANDSCAPE SUMMARY ===")
    print(analysis["body_landscape"]["honest_summary"])
    print(f"Body clustering: {analysis['body_clustering'].get('verdict')} "
          f"(k={analysis['body_clustering'].get('best_k')}, "
          f"sil={analysis['body_clustering'].get('best_silhouette')})")
    print(f"Theme clustering: {analysis['theme_clustering'].get('verdict')} "
          f"(k={analysis['theme_clustering'].get('best_k')}, "
          f"sil={analysis['theme_clustering'].get('best_silhouette')})")
    print(f"Mean redundancy: {analysis['redundancy']['mean_redundancy']}")
    print("Surprising:", analysis["surprising_findings"])


if __name__ == "__main__":
    main()
