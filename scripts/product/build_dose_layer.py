#!/usr/bin/env python3
"""
BUILD (recon-gated) — dose/quantity layer.

Part A: parse 5,000 ingredient lines → report parse rates.
Parts B–E: mass conversion, nutrient RDI, mechanism contribution, validation.

Existing layers read-only. New outputs under data/processed/product/dose/.
"""
from __future__ import annotations

import ast
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/product"))

from dose.quantity_parser import parse_ingredient_line
from dose.dose_engine import DoseEngine

OUT = ROOT / "data/processed/product/dose"
SAMPLES = OUT / "samples"
RAW = ROOT / "data/raw"
REPORT_PATH = OUT / "dose_layer_report.json"

RNG = random.Random(42)


def _collect_ingredient_lines(n: int = 5000) -> list[tuple[str, str]]:
    """Stratified sample: (source, line)."""
    target = {
        "recipenlg": 2500,
        "recipes4": 1500,
        "recipes5": 500,
        "Food_Recipe": 300,
        "recipes3": 200,
    }
    out: list[tuple[str, str]] = []

    # RecipeNLG full_dataset ingredients (quantity-rich)
    rnlg = RAW / "recipenlg/dataset/full_dataset.csv"
    if rnlg.exists() and target["recipenlg"]:
        df = pd.read_csv(rnlg, usecols=["ingredients"])
        lines: list[str] = []
        for v in df["ingredients"].dropna().sample(min(800, len(df)), random_state=42):
            try:
                arr = ast.literal_eval(str(v))
                lines.extend(str(x) for x in arr)
            except Exception:
                pass
        RNG.shuffle(lines)
        for ln in lines[: target["recipenlg"]]:
            out.append(("recipenlg", ln))

    # recipes4 ingredients_raw
    r4 = RAW / "new_recipes/recipes4.csv"
    if r4.exists():
        df = pd.read_csv(r4, usecols=["ingredients_raw"], nrows=3000)
        lines = []
        for v in df["ingredients_raw"].dropna():
            try:
                arr = ast.literal_eval(str(v))
                lines.extend(str(x) for x in arr)
            except Exception:
                pass
        RNG.shuffle(lines)
        need = target["recipes4"]
        for ln in lines[:need]:
            out.append(("recipes4", ln))

    # recipes5
    r5 = RAW / "new_recipes/recipes5.csv"
    if r5.exists():
        df = pd.read_csv(r5)
        lines = []
        for v in df.get("ingredients", pd.Series(dtype=str)).dropna():
            lines.extend(p.strip() for p in str(v).split(",") if p.strip())
        RNG.shuffle(lines)
        for ln in lines[: target["recipes5"]]:
            out.append(("recipes5", ln))

    # Food_Recipe (mostly names — tests partial/unparseable rate)
    fr = RAW / "new_recipes/Food_Recipe.csv"
    if fr.exists():
        df = pd.read_csv(fr, usecols=["ingredients_name"])
        lines = []
        for v in df["ingredients_name"].dropna():
            lines.extend(p.strip() for p in str(v).split(",") if p.strip())
        RNG.shuffle(lines)
        for ln in lines[: target["Food_Recipe"]]:
            out.append(("Food_Recipe", ln))

    # recipes3
    r3 = RAW / "new_recipes/recipes3.json"
    if r3.exists():
        data = json.loads(r3.read_text(encoding="utf-8"))
        lines = []
        for rec in RNG.sample(data, min(200, len(data))):
            lines.extend(str(x) for x in rec.get("ingredients", []))
        for ln in lines[: target["recipes3"]]:
            out.append(("recipes3", ln))

    return out[:n]


def part_a_recon(lines: list[tuple[str, str]]) -> dict:
    by_source: dict[str, Counter] = {}
    overall = Counter()
    examples: dict[str, list] = {k: [] for k in ("clean", "partial", "vague", "unparseable")}

    for source, line in lines:
        p = parse_ingredient_line(line)
        by_source.setdefault(source, Counter())[p.parse_class] += 1
        overall[p.parse_class] += 1
        if len(examples[p.parse_class]) < 5:
            examples[p.parse_class].append({"source": source, "raw": line, "parsed": p.to_dict()})

    n = len(lines)
    return {
        "n_sampled": n,
        "overall": {
            "clean_pct": round(100 * overall["clean"] / n, 2),
            "partial_pct": round(100 * overall["partial"] / n, 2),
            "vague_pct": round(100 * overall["vague"] / n, 2),
            "unparseable_pct": round(100 * overall["unparseable"] / n, 2),
            "counts": dict(overall),
        },
        "by_source": {
            src: {
                "n": sum(c.values()),
                "clean_pct": round(100 * c["clean"] / max(sum(c.values()), 1), 2),
                "partial_pct": round(100 * c["partial"] / max(sum(c.values()), 1), 2),
                "vague_pct": round(100 * c["vague"] / max(sum(c.values()), 1), 2),
                "unparseable_pct": round(100 * c["unparseable"] / max(sum(c.values()), 1), 2),
            }
            for src, c in by_source.items()
        },
        "examples": examples,
        "vague_quantity_policy": {
            "to_taste": "0.5g nominal trace; flagged vague, does not dominate totals",
            "pinch": "0.3g",
            "dash": "0.4g",
            "handful": "30g",
            "garnish": "2g",
            "note": "Trace amounts excluded from bulk-hero dominance checks; flagged in parse_class=vague",
        },
        "parser": "ingredient-parser-nlp (primary) + regex fallback",
    }


def part_b_mass_coverage(engine: DoseEngine, sample_lines: list[tuple[str, str]]) -> dict:
    """Simulate mass conversion on parsed sample."""
    conv = Counter()
    for _, line in sample_lines:
        p = parse_ingredient_line(line)
        iid, _ = engine.resolve_ingredient(p.ingredient_name)
        m = engine.mass_converter.convert(iid, p.ingredient_name, p.amount, p.unit)
        conv[m.conversion_method] += 1
    n = len(sample_lines)
    converted = n - conv.get("unconvertible", 0)
    return {
        "universe_coverage": engine.coverage_report()["fdc_map"],
        "typical_recipe_sample": {
            "n_lines": n,
            "converted_pct": round(100 * converted / max(n, 1), 2),
            "by_method": dict(conv),
        },
        "fallback_policy": (
            "1) FDC food_portion per-ingredient (modifier field for tsp/tbsp when measure_unit=undetermined); "
            "2) category defaults (spice/flour/legume/liquid); "
            "3) generic cup/tbsp/tsp; "
            "4) flag unconvertible"
        ),
    }


VALIDATION_RECIPES = {
    "spice_curry": {
        "label": "Spice curry (turmeric-forward)",
        "servings": 4,
        "lines": [
            "1 tbsp turmeric",
            "1 tsp cumin",
            "1 tsp coriander powder",
            "2 tbsp vegetable oil",
            "1 cup onion, chopped",
            "1 cup coconut milk",
            "salt to taste",
        ],
    },
    "lentil_dal": {
        "label": "Lentil dal",
        "servings": 4,
        "lines": [
            "1 cup red lentils",
            "1 tsp turmeric",
            "1 tsp cumin",
            "1 tbsp ghee",
            "1 cup onion, diced",
            "2 cups water",
            "salt to taste",
        ],
    },
    "plain_cake": {
        "label": "Plain cake",
        "servings": 8,
        "lines": [
            "2 cups all-purpose flour",
            "1 cup granulated sugar",
            "1/2 cup butter",
            "2 eggs",
            "1 cup milk",
            "2 tsp baking powder",
            "1 tsp vanilla extract",
        ],
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    print("PART A — sampling and parsing 5,000 ingredient lines...")
    lines = _collect_ingredient_lines(5000)
    part_a = part_a_recon(lines)
    print(json.dumps(part_a["overall"], indent=2))

    print("Loading dose engine...")
    engine = DoseEngine()

    print("PART B — mass conversion coverage...")
    part_b = part_b_mass_coverage(engine, lines)

    print("PART E — validating 3 recipes...")
    validations = {}
    for key, spec in VALIDATION_RECIPES.items():
        result = engine.analyze_ingredient_lines(
            spec["lines"],
            recipe_id=f"validation_{key}",
            recipe_label=spec["label"],
            source="validation",
            n_servings=spec["servings"],
        )
        out_path = SAMPLES / f"dose_validation_{key}.json"
        out_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        validations[key] = {
            "path": str(out_path.relative_to(ROOT)),
            "summary": _summarize_validation(result),
        }

    report = {
        "version": "dose_layer_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "part_a_quantity_parsing": part_a,
        "part_b_mass_conversion": part_b,
        "part_c_nutrient_rdi": {
            "source": "FDA Daily Values (2016 food label, 21 CFR 101.9)",
            "reference_nutrients": list(engine.nutrient_lookup.keys())[:5],
            "n_dv_nutrients": len(engine.nutrient_lookup),
            "basis_policy": "per_serving when servings known; else per_recipe with flag",
        },
        "part_d_mechanism": {
            "method": "relative contribution = mass_g × potency_proxy × theme_strength",
            "potency_proxy": "log(compound_count) × mean_pathway_enrichment; NOT an absolute dose",
            "note": "No compound RDI; within-dish ranking only",
        },
        "part_e_validation": validations,
        "module": {
            "entrypoint": "scripts/product/dose/dose_engine.py",
            "class": "DoseEngine.analyze_ingredient_lines",
            "live_input": "pass list[str] ingredient lines",
            "batch_input": "same code path via adapter",
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")
    print(json.dumps({k: validations[k]["summary"] for k in validations}, indent=2))


def _theme_hero(result, theme_id: str) -> dict | None:
    if not result.mechanism:
        return None
    for c in result.mechanism.contributions:
        if c.theme_id == theme_id and c.rank == 1:
            return {
                "canonical_name": c.canonical_name,
                "relative_contribution": c.relative_contribution,
                "mass_g": c.mass_g,
                "potency_proxy": c.potency_proxy,
            }
    return None


def _summarize_validation(result) -> dict:
    panel = result.nutrient_panel
    top_dv = sorted(panel.percent_dv.items(), key=lambda x: -x[1])[:6] if panel else []
    bulk = result.mechanism.bulk_hero if result.mechanism else None
    mech = result.mechanism.mechanistic_heroes if result.mechanism else {}
    return {
        "parsed_ingredients": [
            {
                "raw": i.raw,
                "amount": i.parsed.amount,
                "unit": i.parsed.unit,
                "name": i.parsed.ingredient_name,
                "mass_g": i.mass.mass_g,
                "conversion": i.mass.conversion_method,
            }
            for i in result.ingredients
        ],
        "top_percent_dv": [
            {"nutrient": k, "percent_dv": round(v, 1)} for k, v in top_dv
        ],
        "bulk_hero": bulk,
        "potency_hero": result.mechanism.potency_hero if result.mechanism else None,
        "inflammation_immune_hero": _theme_hero(result, "inflammation_immune"),
        "mechanistic_heroes": {
            tid: {
                "name": h["canonical_name"],
                "relative_contribution": h["relative_contribution"],
                "mass_g": h["mass_g"],
            }
            for tid, h in mech.items()
        },
        "warnings": result.warnings,
    }


if __name__ == "__main__":
    main()
