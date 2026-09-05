#!/usr/bin/env python3
"""
40-recipe differentiation re-validation using finer category profiles (v2).

Uses ingredient_category_profiles_v2 (fine_recipe level) with species-IDF +
category-IDF downweighting from hub-suppression diagnostic.

Usage (from repo root):
    python scripts/tier1/build_recipe_signatures_sample_fine_v1.py

Outputs (new only):
    data/processed/tier1/recipe_signatures_sample_fine_v1.parquet
    data/processed/tier1/recipe_signatures_sample_fine_report_v1.json
"""
from __future__ import annotations

import importlib.util
import json
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
PROFILES_V2 = TIER1 / "ingredient_category_profiles_v2.parquet"
BUILD_REPORT_V2 = TIER1 / "reactome_category_build_report_v2.json"
HUB_REPORT = TIER1 / "recipe_signatures_sample_hub_suppression_report_v1.json"
OUT_PARQUET = TIER1 / "recipe_signatures_sample_fine_v1.parquet"
OUT_REPORT = TIER1 / "recipe_signatures_sample_fine_report_v1.json"

_v1_spec = importlib.util.spec_from_file_location(
    "build_recipe_signatures_sample_v1", SCRIPT_DIR / "build_recipe_signatures_sample_v1.py"
)
_v1 = importlib.util.module_from_spec(_v1_spec)
assert _v1_spec.loader is not None
_v1_spec.loader.exec_module(_v1)

_hub_spec = importlib.util.spec_from_file_location(
    "build_recipe_signatures_sample_hub_suppression_v1",
    SCRIPT_DIR / "build_recipe_signatures_sample_hub_suppression_v1.py",
)
_hub = importlib.util.module_from_spec(_hub_spec)
assert _hub_spec.loader is not None
_hub_spec.loader.exec_module(_hub)

SENSIBILITY_RECIPES = _v1.SENSIBILITY_RECIPES
FINE_LEVEL = "fine_recipe"


def load_fine_category_matrix() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    cat = pd.read_parquet(PROFILES_V2)
    sub = cat[
        (cat["enrichment_layer"] == "weighted_calibrated")
        & (cat["category_level"] == FINE_LEVEL)
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
    print("=== Fine-resolution 40-recipe re-validation ===", flush=True)
    TIER1.mkdir(parents=True, exist_ok=True)

    sample = pd.read_parquet(SAMPLE_V1)
    v1_report = json.loads((TIER1 / "recipe_signatures_sample_report_v1.json").read_text(encoding="utf-8"))
    hub_report = json.loads(HUB_REPORT.read_text(encoding="utf-8"))
    build_v2 = json.loads(BUILD_REPORT_V2.read_text(encoding="utf-8"))

    recipe_ing = pd.read_parquet(_v1.RECIPE_ING)
    string_map = _v1.load_live_string_map()
    species_idf = _v1.compute_species_idf(recipe_ing, string_map)

    cat_raw, cat_norm, cat_cols, id_to_name = load_fine_category_matrix()
    cat_idf_vec, cat_idf_map, _ = _hub.compute_category_idf(cat_cols, cat_raw)
    print(f"  Fine categories: {len(cat_cols)}", flush=True)

    fine_vecs: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for _, row in sample.iterrows():
        rid = row["recipe_id"]
        species, _ = _v1.recipe_species_list(rid, recipe_ing, string_map)
        vec = _hub.aggregate_recipe_category(
            species, cat_norm, cat_cols, species_idf, cat_idf_vec
        )
        fine_vecs[rid] = vec

        go_ns_mass = sum(
            vec[i] for i, c in enumerate(cat_cols) if str(c).startswith("GO_NS:")
        )
        rows.append(
            {
                "recipe_id": rid,
                "recipe_name": row["recipe_name"],
                "recipe_type": row["recipe_type"],
                "category_level": FINE_LEVEL,
                "weighting_method": "species_idf_plus_category_idf",
                "category_signature_fine": json.dumps(
                    {id_to_name.get(c, c): float(v) for c, v in zip(cat_cols, vec) if v > 0}
                ),
                "top5_categories_fine": json.dumps(
                    _v1.top_k_from_vector(vec, cat_cols, id_to_name, 5)
                ),
                "category_shannon_fine": round(_v1.shannon_entropy(vec), 4),
                "go_ns_share": round(float(go_ns_mass), 4),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT_PARQUET, index=False)

    meta = sample[["recipe_id", "recipe_name", "recipe_type"]]
    type_by_id = dict(zip(meta["recipe_id"], meta["recipe_type"]))
    after_fine = _hub.similarity_summary(fine_vecs, type_by_id)

    sensibility: dict[str, Any] = {}
    for rid in SENSIBILITY_RECIPES:
        sensibility[rid] = {
            "recipe_name": sample.loc[sample["recipe_id"] == rid, "recipe_name"].iloc[0],
            "recipe_type": type_by_id[rid],
            "v1_baseline_top5": v1_report["validation"]["sensibility_eyeball"][rid]["top5_categories"],
            "hub_suppressed_top5_v1": hub_report["sensibility_rerun"][rid]["idf_suppressed_top5"],
            "fine_v2_top5": _v1.top_k_from_vector(fine_vecs[rid], cat_cols, id_to_name, 5),
            "go_ns_share": round(
                float(
                    sum(
                        fine_vecs[rid][i]
                        for i, c in enumerate(cat_cols)
                        if str(c).startswith("GO_NS:")
                    )
                ),
                4,
            ),
        }

    gap = after_fine["within_minus_cross"] or 0.0
    cross = after_fine["mean_cross_type"] or 1.0
    if gap >= 0.08 and cross <= 0.85:
        verdict = "scale_ready"
        verdict_note = (
            f"Fine categories de-converged: within-cross gap={gap:.3f}, cross-type={cross:.3f}. "
            "Recipe types cluster — ready to scale with v2 profiles + IDF weighting."
        )
    elif gap >= 0.03 and cross < 0.92:
        verdict = "partial_improvement_review"
        verdict_note = (
            f"Meaningful improvement but clustering margin is modest (gap={gap:.3f}). "
            "Consider scaling with caution or further GO-slim refinement."
        )
    else:
        verdict = "not_ready_remaining_bottleneck"
        verdict_note = (
            f"Fine resolution did not achieve type clustering (gap={gap:.3f}, cross={cross:.3f}). "
            "Remaining bottleneck: shared tail categories across ingredients or mapping gaps."
        )

    mean_go_ns = float(out_df["go_ns_share"].mean())

    report = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_source": str(SAMPLE_V1),
        "category_profiles": str(PROFILES_V2),
        "category_level": FINE_LEVEL,
        "weighting": _hub.CATEGORY_IDF_FORMULA,
        "part_a_summary": {
            "go_recovered_from_597": build_v2["go_fix"]["recovered_from_v1_597_root"],
            "go_recovery_rate": build_v2["go_fix"]["recovery_rate_of_597"],
            "fine_categories_in_use": build_v2["category_counts_in_use_weighted_calibrated"]["fine_recipe"],
            "go_ns_share_mean_ingredient_fine": build_v2["go_ns_share_mean_across_ingredients_fine_level"],
            "go_ns_share_mean_recipe_sample": round(mean_go_ns, 4),
            "go_ns_share_v1_reference": 0.49,
        },
        "before_after_comparison": {
            "v1_baseline_cross_type": v1_report["validation"]["similarity_summary"]["mean_cross_type"],
            "v1_hub_suppressed_cross_type": hub_report["before_after"]["after_category_idf_downweight"]["mean_cross_type"],
            "fine_v2_cross_type": after_fine["mean_cross_type"],
            "v1_baseline_within_minus_cross": v1_report["validation"]["similarity_summary"].get(
                "within_minus_cross",
                round(
                    v1_report["validation"]["similarity_summary"]["mean_within_type"]
                    - v1_report["validation"]["similarity_summary"]["mean_cross_type"],
                    4,
                ),
            ),
            "v1_hub_suppressed_within_minus_cross": hub_report["before_after"]["after_category_idf_downweight"]["within_minus_cross"],
            "fine_v2_within_minus_cross": after_fine["within_minus_cross"],
            "fine_v2_within_type": after_fine["mean_within_type"],
            "fine_v2_min_cross_type": after_fine["min_cross_type"],
        },
        "similarity_matrix_fine_v2": after_fine["pairwise_cosine"],
        "type_representatives_fine_v2": type_representatives(
            fine_vecs, meta, cat_cols, id_to_name
        ),
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
    ba = report["before_after_comparison"]
    print(
        f"  Cross-type: v1={ba['v1_baseline_cross_type']} hub={ba['v1_hub_suppressed_cross_type']} "
        f"fine={ba['fine_v2_cross_type']}",
        flush=True,
    )
    print(
        f"  Within-cross gap: v1={ba['v1_baseline_within_minus_cross']} "
        f"hub={ba['v1_hub_suppressed_within_minus_cross']} fine={ba['fine_v2_within_minus_cross']}",
        flush=True,
    )
    print(f"  Verdict: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
