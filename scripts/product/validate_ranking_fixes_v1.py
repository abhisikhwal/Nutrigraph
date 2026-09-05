#!/usr/bin/env python3
"""Post-build validation for theme ranking fixes v1.2."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from retrieval_api import IngredientRetrieval  # noqa: E402

INDEX = ROOT / "data/processed/product/indexes/effect_themes.json"


def main() -> None:
    r = IngredientRetrieval()
    et = json.loads(INDEX.read_text(encoding="utf-8"))
    aliases = r.ingredient_lookup["aliases"]

    inf = et["themes"]["inflammation_immune"]
    print("=== FIX 1: inflammation theme membership ===")
    print(f"native pathways: {inf['n_pathways']}")
    print(f"retrieval pathways (incl eicosanoid): {inf['n_retrieval_pathways']}")
    print(f"sub-themes: {inf.get('include_sub_themes')}")
    print(f"theme genes: {inf.get('n_theme_relevant_genes')}")
    print("fragmentation audit:")
    for finding in et.get("theme_fragmentation_audit", [])[:8]:
        print(
            f"  {finding['axis']}: unthemed={finding['n_unthemed']} "
            f"themes={finding['themes_with_hits']}"
        )

    print("\n=== FIX 3: adaptive weights (selected themes) ===")
    for tid in [
        "inflammation_immune",
        "xenobiotic_detox",
        "neurotransmitter_brain",
        "eicosanoid_prostaglandin",
    ]:
        aw = et["theme_adaptive_weight_profiles"][tid]
        print(tid)
        for comp, diag in aw["component_diagnostics"].items():
            live = "LIVE" if diag["is_live"] else "DEAD"
            awt = diag.get("adapted_weight", aw["default_weights"][comp])
            print(f"  {comp}: {live} var={diag['variance']} adapted_w={awt}")

    print("\n=== INFLAMMATION TOP 10 ===")
    res = r.ingredients_by_effect("inflammation_immune", top_n=10)
    for i, ing in enumerate(res["ingredients"], 1):
        eb = ing.get("evidence_basis", {})
        moa = ing.get("theme_measured_moa_with_action", [])
        print(
            f"{i}. {ing['canonical_name']} score={ing['theme_relevance_score']} "
            f"moa_frac={ing.get('theme_measured_moa_fraction')} "
            f"pathways={ing['theme_pathways_engaged']} gate={ing['richness_gate']}"
        )
        print(f"   evidence: {eb}")
        if moa:
            print(f"   measured MoA: {moa}")

    print("\n=== SPICE RANKS (inflammation top-100) ===")
    all_rows = et["effect_to_ingredients"]["inflammation_immune"]["ingredients"]
    for name in ["turmeric", "ginger", "clove", "black pepper", "pepper"]:
        sid = aliases.get(name.lower())
        if not sid:
            print(name, "alias not found")
            continue
        entry = next((x for x in all_rows if x["species_id"] == sid), None)
        if entry:
            rank = all_rows.index(entry) + 1
            print(
                name,
                "rank",
                rank,
                "score",
                entry["theme_relevance_score"],
                "moa",
                entry.get("theme_measured_moa_with_action"),
            )
        else:
            print(name, "NOT in top-100")

    print("\n=== REGRESSION: xenobiotic_detox top 5 ===")
    for i, ing in enumerate(r.ingredients_by_effect("xenobiotic_detox", 5)["ingredients"], 1):
        print(i, ing["canonical_name"], ing["theme_relevance_score"], ing["richness_gate"])

    print("\n=== REGRESSION: neurotransmitter_brain top 5 ===")
    for i, ing in enumerate(
        r.ingredients_by_effect("neurotransmitter_brain", 5)["ingredients"], 1
    ):
        print(i, ing["canonical_name"], ing["theme_relevance_score"], ing["richness_gate"])

    print("\n=== THIN ARTIFACT CHECK (inflammation top 100) ===")
    for name in ["rapeseed", "soybean", "canola", "soy bean"]:
        sid = aliases.get(name.lower())
        hit = next((x for x in all_rows if x["species_id"] == sid), None) if sid else None
        print(name, "IN TOP100" if hit else "absent", hit["theme_relevance_score"] if hit else "")

    print("\n=== SCORE SPREAD (inflammation top 100) ===")
    scores = [x["theme_relevance_score"] for x in all_rows]
    print("min", min(scores), "max", max(scores), "range", round(max(scores) - min(scores), 4))
    print("std", round(statistics.pstdev(scores), 4))
    within1 = sum(1 for s in scores if scores[0] - s <= 1.0)
    print("within 1pt of #1:", within1)


if __name__ == "__main__":
    main()
