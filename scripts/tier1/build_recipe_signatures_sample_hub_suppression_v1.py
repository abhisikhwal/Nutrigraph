#!/usr/bin/env python3
"""
Hub-category suppression re-validation on the same 40-recipe sample.

Applies category-level IDF at recipe aggregation (parallel to gene IDF / species IDF)
and compares downweighting vs hard exclusion of near-universal GO/Reactome hubs.

Usage (from repo root):
    python scripts/tier1/build_recipe_signatures_sample_hub_suppression_v1.py

Outputs (new only):
    data/processed/tier1/recipe_signatures_sample_hub_suppressed_v1.parquet
    data/processed/tier1/recipe_signatures_sample_hub_suppression_report_v1.json
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
TIER1 = ROOT / "data/processed/tier1"

SAMPLE_V1 = TIER1 / "recipe_signatures_sample_v1.parquet"
SAMPLE_V1_REPORT = TIER1 / "recipe_signatures_sample_report_v1.json"
OUT_PARQUET = TIER1 / "recipe_signatures_sample_hub_suppressed_v1.parquet"
OUT_REPORT = TIER1 / "recipe_signatures_sample_hub_suppression_report_v1.json"

# Import shared helpers from v1 sample builder (same inputs / sample).
_v1_spec = importlib.util.spec_from_file_location(
    "build_recipe_signatures_sample_v1", SCRIPT_DIR / "build_recipe_signatures_sample_v1.py"
)
_v1 = importlib.util.module_from_spec(_v1_spec)
assert _v1_spec.loader is not None
_v1_spec.loader.exec_module(_v1)

SENSIBILITY_RECIPES = _v1.SENSIBILITY_RECIPES
RECIPES = _v1.RECIPES
RECIPE_ING = _v1.RECIPE_ING
STRING_MAP = _v1.STRING_MAP
CATEGORY_PROFILES = _v1.CATEGORY_PROFILES

# Near-universal categories (>=60% of 445 profiled ingredients with any enrichment mass).
HUB_EXCLUSION_SET = {
    "GO_NS:biological_process",
    "GO:0006629",  # lipid metabolic process
    "R-HSA-162582",  # Signal Transduction
    "GO:0006355",  # regulation of DNA-templated transcription
    "R-HSA-1430728",  # Metabolism
    "GO:0006351",  # DNA-templated transcription
    "R-HSA-9748784",  # Drug ADME
}

CATEGORY_IDF_FORMULA = (
    "category_idf(c) = log(N_ingredients / df_c) where df_c is the count of species "
    "with aggregated_enrichment > 0 for top-level category c in "
    "ingredient_category_profiles_v1 (weighted_calibrated layer, N=445). "
    "Recipe aggregation (species IDF retained from v1): "
    "raw_recipe(c) = sum_i [ species_idf(i) * category_idf(c) * profile_i(c) ] / sum_i species_idf(i), "
    "where profile_i(c) is L1-normalized per-ingredient category mass. "
    "Final signature: raw_recipe / sum_c raw_recipe(c)."
)


def compute_category_idf(
    cat_cols: list[str],
    cat_mat_raw: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    """Ingredient-level category document frequency."""
    active = (cat_mat_raw > 0).astype(int)
    n_ing = len(cat_mat_raw)
    df = active.sum(axis=0)
    idf_vec = np.array(
        [math.log(n_ing / max(1, int(df[c]))) for c in cat_cols],
        dtype=float,
    )
    idf_map = {c: float(v) for c, v in zip(cat_cols, idf_vec)}
    stats = pd.DataFrame(
        {
            "category_id": cat_cols,
            "ingredient_df": df.values,
            "ingredient_df_pct": (df.values / n_ing).round(4),
            "category_idf": idf_vec.round(4),
        }
    )
    return idf_vec, idf_map, stats


def aggregate_recipe_category(
    species_list: list[str],
    cat_mat: pd.DataFrame,
    cat_cols: list[str],
    species_idf: dict[str, float],
    category_weights: np.ndarray,
) -> np.ndarray:
    """species_idf × category_weight(c) × profile_i(c), averaged over ingredients."""
    acc = np.zeros(len(cat_cols), dtype=float)
    w_sum = 0.0
    for sp in species_list:
        if sp not in cat_mat.index:
            continue
        wi = species_idf.get(sp, 0.0)
        if wi <= 0:
            continue
        acc += wi * cat_mat.loc[sp, cat_cols].to_numpy(dtype=float) * category_weights
        w_sum += wi
    if w_sum > 0:
        acc /= w_sum
    total = acc.sum()
    if total > 0:
        acc /= total
    return acc


def similarity_summary(
    vecs: dict[str, np.ndarray],
    type_by_id: dict[str, str],
) -> dict[str, Any]:
    ids = list(vecs.keys())
    cross, within = [], []
    sim: dict[str, dict[str, float]] = {}
    for i in ids:
        sim[i] = {}
        for j in ids:
            v = _v1.cosine_sim(vecs[i], vecs[j])
            sim[i][j] = round(v, 4)
            if i >= j:
                continue
            (within if type_by_id[i] == type_by_id[j] else cross).append(v)
    return {
        "pairwise_cosine": sim,
        "mean_cross_type": round(float(np.mean(cross)), 4) if cross else None,
        "mean_within_type": round(float(np.mean(within)), 4) if within else None,
        "within_minus_cross": round(float(np.mean(within) - np.mean(cross)), 4)
        if cross and within
        else None,
        "min_cross_type": round(float(np.min(cross)), 4) if cross else None,
        "max_cross_type": round(float(np.max(cross)), 4) if cross else None,
    }


def type_representatives(
    vecs: dict[str, np.ndarray],
    meta: pd.DataFrame,
    cat_cols: list[str],
    id_to_name: dict[str, str],
) -> dict[str, Any]:
    reps: dict[str, Any] = {}
    for rtype, grp in meta.groupby("recipe_type"):
        ids = grp["recipe_id"].tolist()
        stack = np.vstack([vecs[rid] for rid in ids])
        mean_vec = stack.mean(axis=0)
        if mean_vec.sum() > 0:
            mean_vec = mean_vec / mean_vec.sum()
        reps[rtype] = {
            "n_recipes": len(ids),
            "example_recipe": grp.iloc[0]["recipe_name"],
            "top5_categories": _v1.top_k_from_vector(mean_vec, cat_cols, id_to_name, 5),
        }
    return reps


def main() -> int:
    print("=== Hub-category suppression re-validation (40-recipe sample) ===", flush=True)
    TIER1.mkdir(parents=True, exist_ok=True)

    sample = pd.read_parquet(SAMPLE_V1)
    v1_report = json.loads(SAMPLE_V1_REPORT.read_text(encoding="utf-8"))
    before_summary = v1_report["validation"]["similarity_summary"]

    recipes = pd.read_parquet(RECIPES)
    recipe_ing = pd.read_parquet(RECIPE_ING)
    string_map = _v1.load_live_string_map()
    species_idf = _v1.compute_species_idf(recipe_ing, string_map)
    cat_mat_raw, cat_mat, cat_cols, id_to_name = _v1.load_category_matrix()

    cat_idf_vec, cat_idf_map, cat_idf_stats = compute_category_idf(cat_cols, cat_mat_raw)
    print(f"  Category IDF computed for {len(cat_cols)} top-level categories", flush=True)

    # Baseline (v1): species IDF only, no category suppression — recompute for parity.
    baseline_vecs: dict[str, np.ndarray] = {}
    idf_down_vecs: dict[str, np.ndarray] = {}
    exclude_vecs: dict[str, np.ndarray] = {}

    exclude_mask = np.array(
        [0.0 if c in HUB_EXCLUSION_SET else 1.0 for c in cat_cols],
        dtype=float,
    )
    idf_weights = cat_idf_vec.copy()
    exclude_weights = cat_idf_vec * exclude_mask

    rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        rid = row["recipe_id"]
        species, _ = _v1.recipe_species_list(rid, recipe_ing, string_map)

        base = _v1.aggregate_signature(species, cat_mat, cat_cols, species_idf)
        if base.sum() > 0:
            base = base / base.sum()
        baseline_vecs[rid] = base

        idf_vec = aggregate_recipe_category(
            species, cat_mat, cat_cols, species_idf, idf_weights
        )
        idf_down_vecs[rid] = idf_vec

        ex_vec = aggregate_recipe_category(
            species, cat_mat, cat_cols, species_idf, exclude_weights
        )
        exclude_vecs[rid] = ex_vec

        rows.append(
            {
                "recipe_id": rid,
                "recipe_name": row["recipe_name"],
                "recipe_type": row["recipe_type"],
                "category_signature_baseline": json.dumps(
                    {id_to_name.get(c, c): float(v) for c, v in zip(cat_cols, base) if v > 0}
                ),
                "category_signature_idf_suppressed": json.dumps(
                    {
                        id_to_name.get(c, c): float(v)
                        for c, v in zip(cat_cols, idf_vec)
                        if v > 0
                    }
                ),
                "category_signature_hub_excluded": json.dumps(
                    {
                        id_to_name.get(c, c): float(v)
                        for c, v in zip(cat_cols, ex_vec)
                        if v > 0
                    }
                ),
                "top5_categories_idf_suppressed": json.dumps(
                    _v1.top_k_from_vector(idf_vec, cat_cols, id_to_name, 5)
                ),
                "top5_categories_hub_excluded": json.dumps(
                    _v1.top_k_from_vector(ex_vec, cat_cols, id_to_name, 5)
                ),
                "category_shannon_idf_suppressed": round(_v1.shannon_entropy(idf_vec), 4),
                "category_shannon_hub_excluded": round(_v1.shannon_entropy(ex_vec), 4),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT_PARQUET, index=False)

    meta = sample[["recipe_id", "recipe_name", "recipe_type"]]
    type_by_id = dict(zip(meta["recipe_id"], meta["recipe_type"]))

    after_idf = similarity_summary(idf_down_vecs, type_by_id)
    after_exclude = similarity_summary(exclude_vecs, type_by_id)
    baseline = similarity_summary(baseline_vecs, type_by_id)

    sensibility: dict[str, Any] = {}
    for rid in SENSIBILITY_RECIPES:
        sensibility[rid] = {
            "recipe_name": sample.loc[sample["recipe_id"] == rid, "recipe_name"].iloc[0],
            "recipe_type": type_by_id[rid],
            "baseline_top5": _v1.top_k_from_vector(
                baseline_vecs[rid], cat_cols, id_to_name, 5
            ),
            "idf_suppressed_top5": _v1.top_k_from_vector(
                idf_down_vecs[rid], cat_cols, id_to_name, 5
            ),
            "hub_excluded_top5": _v1.top_k_from_vector(
                exclude_vecs[rid], cat_cols, id_to_name, 5
            ),
        }

    # Verdict thresholds (explicit for review).
    idf_gap = after_idf["within_minus_cross"] or 0.0
    ex_gap = after_exclude["within_minus_cross"] or 0.0
    if after_idf["mean_cross_type"] is not None and after_idf["mean_cross_type"] < 0.85 and idf_gap >= 0.05:
        verdict = "scale_ready_category_idf"
        verdict_note = (
            "Hub suppression via category-IDF de-converged sufficiently: cross-type similarity "
            "dropped below 0.85 and within-type exceeds cross-type by >=0.05."
        )
    elif after_exclude["mean_cross_type"] is not None and after_exclude["mean_cross_type"] < 0.85:
        verdict = "partial_exclusion_helps_option2_still_advised"
        verdict_note = (
            "Hard hub exclusion helps but downweighting alone may be insufficient; "
            "tail categories remain coarse — GO-slim remap (Option 2) recommended before scale."
        )
    else:
        verdict = "not_ready_option2_required"
        verdict_note = (
            "Hub suppression alone did not de-converge dishes enough for scaling. "
            "Cross-type similarity remains high; finer GO-slim category mapping (Option 2) required."
        )

    report = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_source": str(SAMPLE_V1),
        "n_recipes": len(sample),
        "category_idf_formula": CATEGORY_IDF_FORMULA,
        "hub_exclusion_set": sorted(HUB_EXCLUSION_SET),
        "hub_exclusion_rationale": (
            "Seven near-universal top-level categories present in >=63% of profiled ingredients, "
            "including GO_NS:biological_process (69.4%) — the generic GO namespace fallback bucket."
        ),
        "category_idf_stats_top15_lowest_idf": cat_idf_stats.nsmallest(15, "category_idf")[
            ["category_id", "ingredient_df_pct", "category_idf"]
        ].to_dict("records"),
        "before_after": {
            "baseline_v1_report": before_summary,
            "baseline_recomputed": {
                k: baseline[k]
                for k in (
                    "mean_cross_type",
                    "mean_within_type",
                    "within_minus_cross",
                    "min_cross_type",
                )
            },
            "after_category_idf_downweight": {
                k: after_idf[k]
                for k in (
                    "mean_cross_type",
                    "mean_within_type",
                    "within_minus_cross",
                    "min_cross_type",
                    "max_cross_type",
                )
            },
            "after_hub_hard_exclusion": {
                k: after_exclude[k]
                for k in (
                    "mean_cross_type",
                    "mean_within_type",
                    "within_minus_cross",
                    "min_cross_type",
                    "max_cross_type",
                )
            },
            "idf_vs_exclusion_cross_type_delta": round(
                (after_idf["mean_cross_type"] or 0) - (after_exclude["mean_cross_type"] or 0),
                4,
            ),
        },
        "similarity_matrices": {
            "baseline": baseline["pairwise_cosine"],
            "category_idf_suppressed": after_idf["pairwise_cosine"],
            "hub_excluded": after_exclude["pairwise_cosine"],
        },
        "type_representatives": {
            "baseline": type_representatives(baseline_vecs, meta, cat_cols, id_to_name),
            "category_idf_suppressed": type_representatives(
                idf_down_vecs, meta, cat_cols, id_to_name
            ),
            "hub_excluded": type_representatives(exclude_vecs, meta, cat_cols, id_to_name),
        },
        "sensibility_rerun": sensibility,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "outputs": {
            "parquet": str(OUT_PARQUET),
            "report": str(OUT_REPORT),
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  Wrote {OUT_PARQUET}", flush=True)
    print(f"  Wrote {OUT_REPORT}", flush=True)
    ba = report["before_after"]
    print(
        f"  Cross-type cosine: BEFORE={ba['baseline_recomputed']['mean_cross_type']} "
        f"IDF={ba['after_category_idf_downweight']['mean_cross_type']} "
        f"EXCLUDE={ba['after_hub_hard_exclusion']['mean_cross_type']}",
        flush=True,
    )
    print(
        f"  Within-cross gap: IDF={ba['after_category_idf_downweight']['within_minus_cross']} "
        f"EXCLUDE={ba['after_hub_hard_exclusion']['within_minus_cross']}",
        flush=True,
    )
    print(f"  Verdict: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
