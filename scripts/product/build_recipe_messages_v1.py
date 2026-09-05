#!/usr/bin/env python3
"""
Build and validate recipe body-messages for diverse hand-picked recipes.

Usage (from repo root):
    python scripts/product/build_recipe_messages_v1.py

Outputs (new only):
    data/processed/product/recipe_messages/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from recipe_message_engine import RecipeMessageEngine  # noqa: E402

OUT_DIR = ROOT / "data/processed/product/recipe_messages"

VALIDATION_RECIPES: list[dict[str, object]] = [
    {
        "recipe_id": "spice_curry",
        "recipe_label": "Spice curry (Indian-style)",
        "ingredients": [
            "turmeric", "cumin", "coriander", "ginger", "garlic", "onion",
            "chili", "black pepper", "ghee", "tomato",
        ],
    },
    {
        "recipe_id": "plain_cake",
        "recipe_label": "Plain cake",
        "ingredients": [
            "flour", "sugar", "butter", "egg", "milk", "vanilla",
        ],
    },
    {
        "recipe_id": "green_salad",
        "recipe_label": "Green salad",
        "ingredients": [
            "lettuce", "cucumber", "tomato", "olive oil", "lemon",
        ],
    },
    {
        "recipe_id": "fish_dish",
        "recipe_label": "Pan-seared fish with herbs",
        "ingredients": [
            "salmon", "lemon", "garlic", "olive oil", "dill", "parsley",
        ],
    },
    {
        "recipe_id": "lentil_dal",
        "recipe_label": "Lentil dal",
        "ingredients": [
            "lentil", "turmeric", "cumin", "onion", "garlic", "ginger", "tomato",
        ],
    },
    {
        "recipe_id": "fruit_dessert",
        "recipe_label": "Baked fruit dessert",
        "ingredients": [
            "apple", "sugar", "cinnamon", "butter", "lemon",
        ],
    },
]


def top_regions(intensities: dict[str, float], labels: dict[str, str], n: int = 5) -> list[str]:
    ranked = sorted(intensities.items(), key=lambda x: -x[1])
    return [
        f"{labels.get(rid, rid)} ({val:.3f})"
        for rid, val in ranked[:n]
        if val > 0
    ]


def build_validation_report(messages: list[dict]) -> dict:
    """Synthesize decisive checks for body-map differentiation and redundancy thesis."""
    ids = [m["recipe_id"] for m in messages]
    region_profiles: dict[str, dict[str, float]] = {
        m["recipe_id"]: m["body_regions"]["intensities"] for m in messages
    }

    # Pairwise L1 between body-region vectors
    def body_l1(a: str, b: str) -> float:
        keys = list(region_profiles[a].keys())
        return sum(abs(region_profiles[a].get(k, 0) - region_profiles[b].get(k, 0)) for k in keys)

    pairs = [
        ("spice_curry", "plain_cake"),
        ("green_salad", "plain_cake"),
        ("fish_dish", "fruit_dessert"),
        ("lentil_dal", "spice_curry"),
    ]
    differentiation = [
        {
            "pair": [a, b],
            "body_region_l1": round(body_l1(a, b), 4),
        }
        for a, b in pairs
        if a in region_profiles and b in region_profiles
    ]

    curry = next(m for m in messages if m["recipe_id"] == "spice_curry")
    curry_redundancy = curry.get("redundancy", [])
    spice_rows = [r for r in curry_redundancy if r["canonical_name"] in {
        "Turmeric", "Cumin", "Coriander", "Ginger", "Black pepper", "Garden onion"
    }]
    avg_spice_redundancy = (
        sum(r["redundancy_score"] for r in spice_rows) / len(spice_rows) if spice_rows else 0.0
    )

    unique_contributors = [
        {
            "recipe_id": m["recipe_id"],
            "ingredients": [
                r["canonical_name"]
                for r in m.get("redundancy", [])
                if r.get("is_unique_contributor")
            ],
        }
        for m in messages
    ]

    return {
        "body_map_differentiation": {
            "pairwise_l1": differentiation,
            "verdict": (
                "Body-maps differentiate across dish types"
                if all(p["body_region_l1"] > 0.05 for p in differentiation)
                else "Partial differentiation — some dish types overlap in tissue landing"
            ),
        },
        "redundancy_thesis": {
            "spice_curry_avg_spice_redundancy": round(avg_spice_redundancy, 4),
            "spice_curry_spice_rows": spice_rows,
            "unique_contributors_by_recipe": unique_contributors,
            "verdict": (
                "Redundancy behaves as predicted — individual spices in curry are highly redundant"
                if avg_spice_redundancy >= 0.80
                else "Mixed — some spices more distinct than expected"
            ),
        },
    }


def main() -> None:
    engine = RecipeMessageEngine()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    messages: list[dict] = []
    for spec in VALIDATION_RECIPES:
        msg = engine.compute_message(
            ingredient_strings=list(spec["ingredients"]),  # type: ignore[arg-type]
            recipe_id=str(spec["recipe_id"]),
            recipe_label=str(spec["recipe_label"]),
        )
        payload = msg.to_dict()
        messages.append(payload)
        out_path = OUT_DIR / f"{spec['recipe_id']}_message.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path.relative_to(ROOT)}")

    report = build_validation_report(messages)
    summary = {
        "version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_recipes": len(messages),
        "recipes": [
            {
                "recipe_id": m["recipe_id"],
                "recipe_label": m["recipe_label"],
                "n_resolved": m["resolution"]["n_resolved"],
                "n_unresolved": m["resolution"]["n_unresolved"],
                "unresolved": m["resolution"]["unresolved"],
                "top_body_regions": top_regions(
                    m["body_regions"]["intensities"],
                    m["body_regions"]["labels"],
                ),
                "top_themes": [
                    f"{t['label']} ({t['normalized_strength']:.3f})"
                    for t in m["effect_themes"][:5]
                ],
                "evidence": m["evidence_composition"]["summary"],
                "notable_pharmacology_count": len(m["notable_pharmacology"]),
                "sample_pharmacology": [p["claim"] for p in m["notable_pharmacology"][:3]],
                "redundancy_highlights": [
                    {
                        "ingredient": r["canonical_name"],
                        "redundancy_score": r["redundancy_score"],
                        "interpretation": r["interpretation"],
                    }
                    for r in sorted(
                        m.get("redundancy", []),
                        key=lambda x: -x["redundancy_score"],
                    )[:4]
                ],
            }
            for m in messages
        ],
        "validation": report,
        "inputs": {
            "profiles": "data/processed/product/ingredient_profiles_v1_1.jsonl",
            "indexes": "data/processed/product/indexes/",
            "string_map": "data/processed/canonical/ingredient_string_species_v2.parquet",
        },
    }
    summary_path = OUT_DIR / "validation_report_v1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path.relative_to(ROOT)}")

    # Side-by-side body maps for review
    comparison_path = OUT_DIR / "body_region_comparison_v1.json"
    comparison = {
        "region_labels": messages[0]["body_regions"]["labels"] if messages else {},
        "recipes": {
            m["recipe_id"]: {
                "label": m["recipe_label"],
                "intensities": m["body_regions"]["intensities"],
                "top_drivers": {
                    rid: drivers[:3]
                    for rid, drivers in m["body_regions"]["drivers"].items()
                    if drivers
                },
            }
            for m in messages
        },
    }
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {comparison_path.relative_to(ROOT)}")

    print("\n=== VALIDATION SUMMARY ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
