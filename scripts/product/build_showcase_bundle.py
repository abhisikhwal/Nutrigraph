#!/usr/bin/env python3
"""
BUILD — export showcase bundle for web portfolio + Neo4j load inventory.
Read-only on engine data. Writes to data/processed/product/showcase/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/product"))

from dose.dose_engine import DoseEngine
from dose.nutrient_rdi import FDA_DAILY_VALUES

OUT = ROOT / "data/processed/product/showcase"
PROFILES = ROOT / "data/processed/product/ingredient_profiles_v2.jsonl"
LOOKUP = ROOT / "data/processed/product/indexes/ingredient_lookup.json"
NODES = ROOT / "data/processed/product/universe_v2/ingredient_nodes_v2_locked.json"
LIVE = ROOT / "data/processed/product/universe_v2/universe_v2_live_report.json"
LANDSCAPE = ROOT / "data/processed/product/landscape/landscape_report_v1.json"
CALIB = ROOT / "data/processed/tier1/enrichment_v3_calibration_report.json"
KNN = ROOT / "data/processed/thread2/recon/knn_similarity_validation.json"
NUTRIENT_SUM = ROOT / "data/processed/product/nutrients/species_nutrient_summary_production.json"
MSG_DIR = ROOT / "data/processed/product/recipe_messages"

# Prefer exact canonical names; a few display aliases for matching
HERO_ALIASES = {
    "beef": "cattle (beef, veal)",
    "pork": "pig",
    "chili pepper": "chili",
    "black pepper": "pepper (spice)",
    "onion": "garden onion",
    "basil": "sweet basil",
    "thyme": "common thyme",
    "cocoa": "cocoa bean",
}
HERO_NAMES = [
    # rich spices
    "Turmeric", "Garlic", "Ginger", "Cumin", "Cinnamon", "Pepper (Spice)",
    "Pepper", "Chili pepper", "Cloves", "Cardamom", "Coriander", "Fenugreek",
    "Black pepper", "Mustard", "Saffron", "Nutmeg", "Star anise",
    # quiet-but-nutritious
    "Lettuce", "Spinach", "Kale", "Cucumber", "Celery stalks",
    # composites
    "soy sauce", "fish sauce", "Olive oil", "Coconut milk",
    # blends
    "garam masala", "biryani masala",
    # meat / legume / fruit / staples
    "Chicken", "Beef", "Pork", "Lentils", "Chickpea", "Common bean",
    "Sweet orange", "Lemon", "Banana", "Apple", "Mango",
    "Garden tomato", "Onion", "Garden onion", "Rice", "Wheat", "Flour",
    "Tea", "Coffee", "Cocoa bean", "Eggs", "Milk (Cow)",
    "Parsley", "Sweet basil", "Rosemary", "Common thyme",
    "Chili", "Black tea", "Sesame", "Walnut", "Almond",
]


def load_profiles() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with PROFILES.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            p = json.loads(line)
            iid = p["ingredient"].get("ingredient_id") or p["ingredient"]["species_id"]
            out[iid] = p
    return out


def pct_dv(nutrient_name: str, amount: float, unit: str) -> float | None:
    key = None
    for k, meta in FDA_DAILY_VALUES.items():
        if nutrient_name.lower() in k.lower() or meta["label"].lower() in nutrient_name.lower():
            key = k
            break
        if str(meta.get("nutrient_id")) and nutrient_name == k:
            key = k
            break
    # direct name match
    if nutrient_name in FDA_DAILY_VALUES:
        key = nutrient_name
    else:
        for k, meta in FDA_DAILY_VALUES.items():
            if meta["label"].lower() == nutrient_name.lower() or k.lower() == nutrient_name.lower():
                key = k
                break
            if nutrient_name.lower() in ("iron, fe", "iron") and k == "Iron, Fe":
                key = k
                break
            if nutrient_name.lower() in ("protein",) and k == "Protein":
                key = k
                break
            if "fiber" in nutrient_name.lower() and k == "Fiber, total dietary":
                key = k
                break
            if "carbohydrate" in nutrient_name.lower() and k.startswith("Carbohydrate"):
                key = k
                break
            if "vitamin c" in nutrient_name.lower() and "Vitamin C" in k:
                key = k
                break
            if "folate" in nutrient_name.lower() and "Folate" in k:
                key = k
                break
            if "potassium" in nutrient_name.lower() and "Potassium" in k:
                key = k
                break
            if "calcium" in nutrient_name.lower() and "Calcium" in k:
                key = k
                break
            if "sodium" in nutrient_name.lower() and "Sodium" in k:
                key = k
                break
            if nutrient_name.lower() in ("energy", "calories") and k == "Energy":
                key = k
                break
    if not key:
        return None
    dv = FDA_DAILY_VALUES[key]["dv_amount"]
    dv_unit = FDA_DAILY_VALUES[key]["unit"].upper()
    u = str(unit).upper()
    amt = float(amount)
    if u in {"G", "GRAM"} and dv_unit == "MG":
        amt *= 1000
    elif u == "MG" and dv_unit == "G":
        amt /= 1000
    elif u in {"UG", "MCG"} and dv_unit == "MG":
        amt /= 1000
    elif u == "MG" and dv_unit in {"UG", "MCG"}:
        amt *= 1000
    return round(100.0 * amt / dv, 1) if dv else None


def trim_profile(p: dict) -> dict[str, Any]:
    ing = p["ingredient"]
    prov = p.get("provenance", {})
    pathways = p.get("pathways", {})
    nutr = p.get("nutrition", {})

    compounds = []
    for c in (p.get("compounds", {}).get("top") or [])[:12]:
        row = {"name": c.get("name")}
        if c.get("is_distinctive"):
            row["distinctive"] = True
        compounds.append(row)

    targets = []
    for t in (p.get("targets", {}).get("top") or [])[:12]:
        row = {
            "gene": t.get("gene_symbol"),
            "evidence": t.get("evidence"),
            "confidence": t.get("confidence"),
        }
        if t.get("moa"):
            row["moa"] = t["moa"]
        targets.append(row)

    path_rows = []
    for pw in (pathways.get("top_ranked") or pathways.get("top_significant") or [])[:10]:
        path_rows.append({
            "name": pw.get("pathway_name") or pw.get("pathway"),
            "fold": round(float(pw["weighted_fold"]), 3) if pw.get("weighted_fold") is not None else None,
            "evidence": pw.get("evidence_split"),
            "drivers": (pw.get("driving_genes") or [])[:3],
        })

    tissues = [
        {"tissue": tis.get("tissue"), "score": round(float(tis["normalized_score"]), 4) if tis.get("normalized_score") is not None else None}
        for tis in (p.get("tissues", {}).get("top") or [])[:6]
    ]

    nutrition = []
    for n in (nutr.get("top_nutrients") or [])[:10]:
        name = n.get("nutrient_name")
        amount = n.get("amount")
        unit = n.get("unit", "")
        row = {"name": name, "amount": amount, "unit": unit}
        pdv = pct_dv(str(name or ""), float(amount or 0), str(unit))
        if pdv is not None:
            row["pct_dv"] = pdv
        nutrition.append(row)

    measured = float(prov.get("measured_fraction") or 0)
    out = {
        "id": ing.get("ingredient_id") or ing.get("species_id"),
        "name": ing.get("canonical_name"),
        "latin": ing.get("latin_name"),
        "node_type": ing.get("node_type", "species"),
        "data_status": ing.get("data_status", "full"),
        "measured_fraction": round(measured, 4),
        "coverage": ing.get("mechanism_coverage"),
        "n_compounds": p.get("compounds", {}).get("count", 0),
        "n_targets": p.get("targets", {}).get("count", 0),
        "broadly_active": bool(pathways.get("is_broadly_active", False)),
        "summary": prov.get("summary"),
        "compounds": compounds,
        "targets": targets,
        "pathways": path_rows,
        "tissues": tissues,
        "nutrition": nutrition,
    }
    if ing.get("backfill_source"):
        out["backfill_source"] = ing["backfill_source"]
    return out


def pick_heroes(profiles: dict[str, dict]) -> list[dict]:
    by_name = {p["ingredient"]["canonical_name"].lower(): p for p in profiles.values()}
    chosen: list[dict] = []
    seen: set[str] = set()
    for name in HERO_NAMES:
        key = HERO_ALIASES.get(name.lower(), name.lower())
        p = by_name.get(key)
        if not p and len(key) >= 5:
            # whole-word / startswith only for longer names
            for cn, pp in by_name.items():
                if cn == key or cn.startswith(key + " ") or cn.endswith(" " + key):
                    p = pp
                    break
        if not p:
            continue
        iid = p["ingredient"].get("ingredient_id") or p["ingredient"]["species_id"]
        if iid in seen:
            continue
        seen.add(iid)
        chosen.append(trim_profile(p))
        if len(chosen) >= 50:
            break
    return chosen


def build_search_index(profiles: dict[str, dict], lookup: dict) -> list[dict]:
    aliases_by_id: dict[str, list[str]] = {}
    for alias, iid in lookup.get("aliases", {}).items():
        aliases_by_id.setdefault(iid, []).append(alias)
    for iid, alist in lookup.get("alias_lists", {}).items():
        aliases_by_id.setdefault(iid, []).extend(alist)

    rows = []
    for iid, p in profiles.items():
        ing = p["ingredient"]
        name = ing["canonical_name"]
        aliases = sorted(set(a for a in aliases_by_id.get(iid, []) if a.lower() != name.lower()))[:8]
        status = ing.get("data_status", "full")
        n_comp = p.get("compounds", {}).get("count", 0)
        rich = status == "full" and n_comp > 0 and p.get("targets", {}).get("count", 0) > 0
        rows.append({
            "id": iid,
            "name": name,
            "aliases": aliases,
            "node_type": ing.get("node_type", "species"),
            "data_status": status,
            "has_rich_profile": bool(rich),
        })
    rows.sort(key=lambda r: r["name"].lower())
    return rows


def build_findings(profiles: dict[str, dict]) -> dict:
    live = json.loads(LIVE.read_text(encoding="utf-8"))
    landscape = json.loads(LANDSCAPE.read_text(encoding="utf-8"))
    calib = json.loads(CALIB.read_text(encoding="utf-8"))
    knn = json.loads(KNN.read_text(encoding="utf-8"))
    nutr = json.loads(NUTRIENT_SUM.read_text(encoding="utf-8"))

    # measured fraction from gene sets
    gs = pd.read_parquet(
        ROOT / "data/processed/integrated/ingredient_gene_sets_v3.parquet",
        columns=["evidence"],
    )
    measured_frac = float((gs["evidence"] == "measured").mean())
    predicted_frac = 1.0 - measured_frac

    # turmeric LOO from spice curry message
    curry_msg = json.loads((MSG_DIR / "spice_curry_message.json").read_text(encoding="utf-8"))
    turmeric_loo = None
    for row in curry_msg.get("redundancy", []):
        if str(row.get("canonical_name", "")).lower() == "turmeric" or row.get("input_string") == "turmeric":
            turmeric_loo = {
                "redundancy_score": row.get("redundancy_score"),
                "leave_one_out_change_pct": round(100 * (1 - float(row.get("redundancy_score", 1))), 2),
                "is_unique_contributor": row.get("is_unique_contributor"),
            }
            break

    # graph entity counts
    icc = pd.read_parquet(ROOT / "data/processed/canonical/ingredient_compound_canonical_v2.parquet")
    gene_sets = pd.read_parquet(ROOT / "data/processed/integrated/ingredient_gene_sets_v3.parquet", columns=["gene_symbol"])
    gpm = pd.read_parquet(ROOT / "data/interim/pathways/gene_pathway_mappings.parquet")
    tissues = pd.read_parquet(ROOT / "data/processed/tier1/ingredient_tissue_profiles_v2.parquet", columns=["tissue"])
    nutr_df = pd.read_parquet(ROOT / "data/processed/product/nutrients/species_nutrient_profiles_production.parquet", columns=["species_id", "nutrient_id"])

    scaffold_hit10 = knn["scaffold_split"]["overall_hit_at_10"]
    null_frac = calib["validation"]["null_calibration"]["frac_ingredients_with_any_sig"]

    findings = [
        {
            "id": "total_ingredients",
            "value": 695,
            "caption": "The live culinary universe: every ingredient the product can recognize.",
            "source": "universe_v2_live_report.json",
        },
        {
            "id": "ingredient_growth",
            "value": "463 → 695",
            "caption": "Expanded from the original 463 species nodes by adding 232 recipe-derived ingredients.",
            "source": "universe_v2_live_report.json",
        },
        {
            "id": "measured_vs_inferred",
            "value": {
                "measured_pct": round(100 * measured_frac, 1),
                "inferred_pct": round(100 * predicted_frac, 1),
            },
            "caption": "About one-fifth of ingredient→gene edges are lab-measured; the rest are structure-inferred.",
            "source": "ingredient_gene_sets_v3.parquet (evidence column)",
        },
        {
            "id": "inference_validation_hit_at_10",
            "value": round(scaffold_hit10, 4),
            "caption": "On a hard Murcko-scaffold split, nearest-neighbor inference recovers a true target in the top-10 ~86% of the time.",
            "source": "thread2/recon/knn_similarity_validation.json (scaffold_split)",
        },
        {
            "id": "fdr_null_calibration",
            "value": "1.3%",
            "caption": "Only 1.3% of random null ingredient gene-sets light up any pathway at q<0.1 — the enrichment test is conservative, not inflated.",
            "source": "enrichment_v3_calibration_report.json → validation.null_calibration",
        },
        {
            "id": "mean_recipe_redundancy",
            "value": landscape["redundancy"]["mean_redundancy"],
            "caption": "Across 400 recipes, leave-one-out redundancy averages ~0.90 — most ingredients are partially interchangeable.",
            "source": "landscape/landscape_report_v1.json",
        },
        {
            "id": "turmeric_in_curry_loo",
            "value": turmeric_loo,
            "caption": "In a spice curry, removing turmeric barely moves the body-message vector (~2% change) — spices are redundant with each other.",
            "source": "recipe_messages/spice_curry_message.json",
        },
        {
            "id": "compounds",
            "value": int(icc["compound_id"].nunique()),
            "caption": "Unique food compounds linked to ingredients in the measured chemistry layer.",
            "source": "ingredient_compound_canonical_v2.parquet",
        },
        {
            "id": "genes",
            "value": int(gene_sets["gene_symbol"].nunique()),
            "caption": "Human gene targets engaged by at least one ingredient (measured or predicted).",
            "source": "ingredient_gene_sets_v3.parquet",
        },
        {
            "id": "pathways",
            "value": int(gpm["pathway_id"].nunique()),
            "caption": "Biological pathways mapped from those genes.",
            "source": "gene_pathway_mappings.parquet",
        },
        {
            "id": "tissues",
            "value": int(tissues["tissue"].nunique()),
            "caption": "GTEx tissue compartments used for body-region attribution.",
            "source": "ingredient_tissue_profiles_v2.parquet",
        },
        {
            "id": "nutrient_coverage",
            "value": {
                "species_with_composition": nutr["coverage"]["total_with_composition"],
                "of_463": 463,
                "pct_of_463": round(100 * nutr["coverage"]["total_with_composition"] / 463, 1),
                "pct_of_695": round(100 * nutr["coverage"]["total_with_composition"] / 695, 1),
            },
            "caption": "94% of the original 463 species have USDA/FooDB composition; many new ING nodes are nutrition-only or name-only.",
            "source": "species_nutrient_summary_production.json (v2.1)",
        },
        {
            "id": "recipe_corpus_size",
            "value": live["part3_ingest"]["n_recipes"],
            "caption": "New recipe datasets ingested into the v2 store (plus the larger RecipeNLG corpus separately).",
            "source": "universe_v2_live_report.json → part3_ingest",
        },
        {
            "id": "mapping_coverage",
            "value": live["part3_ingest"]["mapping_coverage_pct"],
            "caption": "97.58% of ingredient mentions in the new recipe corpus resolve to the 695-node universe.",
            "source": "universe_v2_live_report.json → part3_ingest.mapping_coverage_pct",
        },
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "findings": findings,
    }


def slim_dose_demo(result, message_path: Path | None = None) -> dict:
    d = result.to_dict()
    # trim ingredients
    ings = []
    for i in d["ingredients"]:
        ings.append({
            "raw": i["raw"],
            "name": i.get("canonical_name") or i["parsed"]["ingredient_name"],
            "id": i.get("ingredient_id"),
            "amount": i["parsed"]["amount"],
            "unit": i["parsed"]["unit"],
            "mass_g": i.get("mass_g"),
        })
    panel = d.get("nutrient_panel") or {}
    nutrients = (panel.get("nutrients") or [])[:10]
    mech = d.get("mechanism") or {}
    heroes = {
        "bulk": mech.get("bulk_hero"),
        "potency": mech.get("potency_hero"),
        "theme": {
            tid: {
                "name": h.get("canonical_name"),
                "relative_contribution": round(h.get("relative_contribution", 0), 3),
                "mass_g": h.get("mass_g"),
                "theme": h.get("theme_label"),
            }
            for tid, h in list((mech.get("mechanistic_heroes") or {}).items())[:5]
        },
    }

    redundancy = []
    themes = []
    if message_path and message_path.exists():
        msg = json.loads(message_path.read_text(encoding="utf-8"))
        for r in msg.get("redundancy", [])[:12]:
            redundancy.append({
                "name": r.get("canonical_name") or r.get("input_string"),
                "redundancy_score": r.get("redundancy_score"),
                "leave_one_out_change": round(1 - float(r.get("redundancy_score", 1)), 4),
                "is_unique_contributor": r.get("is_unique_contributor"),
            })
        for t in msg.get("effect_themes", [])[:6]:
            themes.append({
                "theme": t.get("label") or t.get("theme_id"),
                "strength": t.get("strength") or t.get("score"),
                "drivers": [d.get("canonical_name") or d.get("name") for d in (t.get("drivers") or [])[:3]],
            })

    return {
        "recipe_id": d["recipe_id"],
        "label": d["recipe_label"],
        "n_servings": d.get("n_servings"),
        "ingredients": ings,
        "nutrient_panel": {
            "basis": panel.get("basis"),
            "source": panel.get("source"),
            "nutrients": nutrients,
        },
        "heroes": heroes,
        "redundancy": redundancy,
        "top_themes": themes,
        "warnings": d.get("warnings", []),
    }


def build_recipe_demos(engine: DoseEngine) -> dict:
    specs = {
        "spice_curry": {
            "label": "Spice curry",
            "servings": 4,
            "lines": [
                "1 tbsp turmeric", "1 tsp cumin", "1 tsp coriander powder",
                "2 tbsp vegetable oil", "1 cup onion, chopped", "1 cup coconut milk", "salt to taste",
            ],
            "message": MSG_DIR / "spice_curry_message.json",
        },
        "lentil_dal": {
            "label": "Lentil dal",
            "servings": 4,
            "lines": [
                "1 cup red lentils", "1 tsp turmeric", "1 tsp cumin",
                "1 tbsp ghee", "1 cup onion, diced", "2 cups water", "salt to taste",
            ],
            "message": MSG_DIR / "lentil_dal_message.json",
        },
        "plain_cake": {
            "label": "Plain cake",
            "servings": 8,
            "lines": [
                "2 cups all-purpose flour", "1 cup granulated sugar", "1/2 cup butter",
                "2 eggs", "1 cup milk", "2 tsp baking powder", "1 tsp vanilla extract",
            ],
            "message": MSG_DIR / "plain_cake_message.json",
        },
        "green_salad": {
            "label": "Green salad",
            "servings": 2,
            "lines": [
                "2 cups lettuce", "1 cup cucumber, sliced", "1 tomato",
                "2 tbsp olive oil", "1 tbsp lemon juice", "salt to taste",
            ],
            "message": MSG_DIR / "green_salad_message.json",
        },
    }
    demos = {}
    for key, spec in specs.items():
        result = engine.analyze_ingredient_lines(
            spec["lines"],
            recipe_id=f"showcase_{key}",
            recipe_label=spec["label"],
            source="showcase",
            n_servings=spec["servings"],
        )
        demos[key] = slim_dose_demo(result, spec["message"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "demos": demos,
        "note": "Dose from DoseEngine; redundancy/themes from recipe_message_engine outputs.",
    }


def build_methodology() -> dict:
    knn = json.loads(KNN.read_text(encoding="utf-8"))
    calib = json.loads(CALIB.read_text(encoding="utf-8"))
    live = json.loads(LIVE.read_text(encoding="utf-8"))
    nutr = json.loads(NUTRIENT_SUM.read_text(encoding="utf-8"))
    icc = pd.read_parquet(ROOT / "data/processed/canonical/ingredient_compound_canonical_v2.parquet")
    cg = pd.read_csv(ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv")

    stages = [
        {
            "id": "corpus",
            "title": "Chemistry corpus",
            "description": "Food compounds joined from FooDB contents to a unified compound master.",
            "metrics": {
                "ingredient_compound_edges": int(len(icc)),
                "unique_compounds_in_icc": int(icc["compound_id"].nunique()),
                "ingredients_with_compounds": int(icc["ingredient_id"].nunique()),
                "compound_gene_edges": int(len(cg)),
                "genes_in_compound_gene": int(cg["gene_symbol"].nunique()),
            },
        },
        {
            "id": "inference",
            "title": "Target inference",
            "description": "k-NN structural inference expands sparse measured compound→gene edges; confidence bands are scaffold-validated.",
            "metrics": {
                "method": "Murcko-scaffold-split k-NN",
                "scaffold_hit_at_10": knn["scaffold_split"]["overall_hit_at_10"],
                "random_split_hit_at_10": knn["scaffold_split"]["random_split_hit_at_10"],
                "test_n_evaluated": knn["scaffold_split"]["test_n_evaluated"],
            },
        },
        {
            "id": "enrichment",
            "title": "Pathway enrichment + calibration",
            "description": "Weighted pathway enrichment with permutation-null calibration so FDR is honest under confidence weights.",
            "metrics": {
                "chosen_method": calib["method_comparison"]["chosen_method"],
                "null_frac_any_sig": calib["validation"]["null_calibration"]["frac_ingredients_with_any_sig"],
                "null_interpretation": "1.3% null ingredients pass q<0.1 (conservative)",
            },
        },
        {
            "id": "universe",
            "title": "Universe expansion",
            "description": "463 species → 695 nodes by adding composites, blends, nutrition-only, and name-only ingredients from recipe corpora.",
            "metrics": {
                "total_nodes": 695,
                "existing_species": 463,
                "new_ing_nodes": 232,
                "new_by_status": live["final_universe"]["new_by_data_status"],
                "mapping_coverage_pct": live["part3_ingest"]["mapping_coverage_pct"],
            },
        },
        {
            "id": "nutrition",
            "title": "Nutrition join (USDA FDC)",
            "description": "Species mapped to FoodData Central foundation foods; composition locked at v2.1.",
            "metrics": {
                "with_composition": nutr["coverage"]["total_with_composition"],
                "fdc_mapped": nutr["coverage"]["fdc_mapped"],
                "foodb_fallback": nutr["coverage"]["foodb_fallback"],
                "version": nutr.get("version"),
                "status": nutr.get("status"),
            },
        },
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
    }


def build_neo4j_plan() -> dict:
    icc = pd.read_parquet(ROOT / "data/processed/canonical/ingredient_compound_canonical_v2.parquet")
    cg = pd.read_csv(ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv")
    gs = pd.read_parquet(ROOT / "data/processed/integrated/ingredient_gene_sets_v3.parquet")
    gpm = pd.read_parquet(ROOT / "data/interim/pathways/gene_pathway_mappings.parquet")
    tis = pd.read_parquet(ROOT / "data/processed/tier1/ingredient_tissue_profiles_v2.parquet")
    nutr = pd.read_parquet(ROOT / "data/processed/product/nutrients/species_nutrient_profiles_production.parquet")

    # measured/predicted on compound→gene via gene sets is ingredient-level;
    # compound_gene file uses source; gene sets have evidence
    n_ing = 695
    n_compounds_icc = int(icc["compound_id"].nunique())
    n_genes = int(gs["gene_symbol"].nunique())
    n_pathways = int(gpm["pathway_id"].nunique())
    n_tissues = int(tis["tissue"].nunique())
    n_nutrients = int(nutr["nutrient_id"].nunique())

    contains_edges = int(len(icc))  # ingredient→compound (445 ingredients in ICC)
    # For full 695, many ING lack ICC — edges stay at 2.02M for mechanism-live subset
    targets_edges = int(len(cg))  # compound→gene
    # Also ingredient→gene from gene sets (more natural for viz)
    ing_gene_edges = int(len(gs))
    pathway_edges = int(len(gpm))
    # gene→tissue: GTEx — estimate from genes in graph × tissues
    # Use tissue profile as ingredient→tissue (already aggregated)
    ing_tissue_edges = int(len(tis))
    nutrient_edges = int(len(nutr.drop_duplicates(["species_id", "nutrient_id"])))

    # GTEx gene×tissue estimate
    gtex_path = ROOT / "data/raw/gtex.gct"
    gtex_tissues = 54  # typical GTEx median file; refine if header available
    try:
        with gtex_path.open(encoding="utf-8", errors="ignore") as f:
            next(f)
            dims = next(f).strip().split("\t")
            if len(dims) >= 2:
                gtex_tissues = int(dims[1]) if dims[1].isdigit() else 54
    except Exception:
        pass
    gene_tissue_est = n_genes * gtex_tissues  # upper bound if all genes join GTEx

    # Size scenarios
    full_nodes = n_ing + n_compounds_icc + n_genes + n_pathways + n_tissues + n_nutrients
    full_edges = contains_edges + targets_edges + pathway_edges + gene_tissue_est + nutrient_edges

    # Visualizable trim: top-50 compounds per ingredient
    top_n = 50
    trimmed_contains = min(contains_edges, n_ing * top_n)  # upper; actual ~445*50
    trimmed_contains_actual = int(
        icc.groupby("ingredient_id").head(top_n).shape[0]
    ) if "ingredient_id" in icc.columns else 445 * top_n

    total_nodes_viz = n_ing + min(n_compounds_icc, trimmed_contains_actual) + n_genes + n_pathways + n_tissues + n_nutrients
    # better: unique compounds after trim
    trimmed_icc = icc.groupby("ingredient_id", group_keys=False).head(top_n)
    n_comp_trim = int(trimmed_icc["compound_id"].nunique())
    nodes_viz = n_ing + n_comp_trim + n_genes + n_pathways + n_tissues + n_nutrients
    edges_viz = (
        len(trimmed_icc) + targets_edges + pathway_edges + ing_tissue_edges + nutrient_edges
    )

    measured_n = int((gs["evidence"] == "measured").sum())
    predicted_n = int((gs["evidence"] == "predicted").sum())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CURRENT 695-node universe — inventory only, no load",
        "node_types": [
            {
                "label": "Ingredient",
                "count": n_ing,
                "source": "data/processed/product/universe_v2/ingredient_nodes_v2_locked.json + ingredient_profiles_v2.jsonl",
                "note": "463 SP_* with mechanism + 232 ING_* expansion nodes",
            },
            {
                "label": "Compound",
                "count": n_compounds_icc,
                "source": "data/processed/canonical/ingredient_compound_canonical_v2.parquet",
                "note": "Compounds appearing in ICC for 445 mechanism-live ingredients",
            },
            {
                "label": "Gene",
                "count": n_genes,
                "source": "data/processed/integrated/ingredient_gene_sets_v3.parquet",
            },
            {
                "label": "Pathway",
                "count": n_pathways,
                "source": "data/interim/pathways/gene_pathway_mappings.parquet",
            },
            {
                "label": "Tissue",
                "count": n_tissues,
                "source": "data/processed/tier1/ingredient_tissue_profiles_v2.parquet",
            },
            {
                "label": "Nutrient",
                "count": n_nutrients,
                "source": "data/processed/product/nutrients/species_nutrient_profiles_production.parquet",
            },
        ],
        "edge_types": [
            {
                "type": "CONTAINS",
                "from": "Ingredient",
                "to": "Compound",
                "count": contains_edges,
                "source": "data/processed/canonical/ingredient_compound_canonical_v2.parquet",
                "properties": ["foodb_id"],
                "heavy": True,
                "trim_recommendation": f"top-{top_n} compounds/ingredient → {len(trimmed_icc):,} edges, {n_comp_trim:,} compound nodes",
            },
            {
                "type": "TARGETS",
                "from": "Compound",
                "to": "Gene",
                "count": targets_edges,
                "source": "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv",
                "properties": ["source", "evidence_fields"],
                "note": "Prefer joining evidence via ingredient_gene_sets_v3 for measured/predicted at ingredient level",
            },
            {
                "type": "TARGETS_INGREDIENT",
                "from": "Ingredient",
                "to": "Gene",
                "count": ing_gene_edges,
                "source": "data/processed/integrated/ingredient_gene_sets_v3.parquet",
                "properties": ["evidence (measured|predicted)", "confidence", "n_supporting_compounds"],
                "honesty_layer": True,
                "measured_edges": measured_n,
                "predicted_edges": predicted_n,
            },
            {
                "type": "IN_PATHWAY",
                "from": "Gene",
                "to": "Pathway",
                "count": pathway_edges,
                "source": "data/interim/pathways/gene_pathway_mappings.parquet",
                "properties": ["database", "uniprot_accession"],
            },
            {
                "type": "EXPRESSED_IN",
                "from": "Gene",
                "to": "Tissue",
                "count_estimate": gene_tissue_est,
                "source": "data/raw/gtex.gct (build-time join; not a prebuilt edge table)",
                "note": f"Upper bound ≈ {n_genes} genes × ~{gtex_tissues} GTEx tissues if fully joined",
                "alternative": {
                    "type": "ATTRIBUTED_TO_TISSUE",
                    "from": "Ingredient",
                    "to": "Tissue",
                    "count": ing_tissue_edges,
                    "source": "data/processed/tier1/ingredient_tissue_profiles_v2.parquet",
                    "properties": ["normalized_score", "measured_fraction_of_score"],
                },
            },
            {
                "type": "CONTAINS_NUTRIENT",
                "from": "Ingredient",
                "to": "Nutrient",
                "count": nutrient_edges,
                "source": "data/processed/product/nutrients/species_nutrient_profiles_production.parquet",
                "properties": ["amount", "unit", "basis", "composition_source"],
            },
        ],
        "size_estimate": {
            "full_load": {
                "nodes_approx": full_nodes,
                "edges_approx": full_edges + ing_gene_edges,
                "verdict": "LARGE — ~50K nodes and ~2.4M+ edges; needs care (indexes, batch import, possibly Aura/enterprise or trimmed subgraph for viz)",
                "comparison_to_prior_recon": "Prior ~60K nodes / ~1M edges; current ICC alone is 2.02M ingredient→compound edges",
            },
            "visualizable_trim": {
                "nodes_approx": nodes_viz,
                "edges_approx": edges_viz,
                "trim": f"top-{top_n} compounds per ingredient",
                "verdict": "MODEST — fine for a small VPS / Neo4j Community (tens of thousands of nodes, low hundreds of thousands of edges)",
            },
        },
        "honesty_layer": {
            "supported": True,
            "how": "Store evidence='measured'|'predicted' and confidence on TARGETS / TARGETS_INGREDIENT edges; UI can filter or style by property",
            "source_of_truth": "data/processed/integrated/ingredient_gene_sets_v3.parquet",
            "measured_fraction_global": round(measured_n / max(measured_n + predicted_n, 1), 4),
        },
        "canonical_files_for_loader": [
            "data/processed/product/universe_v2/ingredient_nodes_v2_locked.json",
            "data/processed/product/ingredient_profiles_v2.jsonl",
            "data/processed/canonical/ingredient_compound_canonical_v2.parquet",
            "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv",
            "data/processed/integrated/ingredient_gene_sets_v3.parquet",
            "data/interim/pathways/gene_pathway_mappings.parquet",
            "data/processed/tier1/ingredient_tissue_profiles_v2.parquet",
            "data/processed/tier1/enrichment_weighted_v3_calibrated.parquet",
            "data/processed/product/nutrients/species_nutrient_profiles_production.parquet",
            "data/raw/gtex.gct",
        ],
        "heavy_flags": [
            "ingredient→compound CONTAINS is 2,022,289 edges — trim to top-N per ingredient for any browser-facing graph",
            "GTEx gene→tissue can explode to O(genes × tissues); prefer pre-aggregated ingredient→tissue for showcase viz",
            "Do not regenerate mechanism edges; load measured/predicted as properties only",
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading profiles...")
    profiles = load_profiles()
    lookup = json.loads(LOOKUP.read_text(encoding="utf-8"))

    print("Part 1 — ingredients...")
    heroes = pick_heroes(profiles)
    (OUT / "showcase_ingredients.json").write_text(
        json.dumps({"n": len(heroes), "ingredients": heroes}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print("Part 1 — search index...")
    index = build_search_index(profiles, lookup)
    (OUT / "showcase_search_index.json").write_text(
        json.dumps({"n": len(index), "nodes": index}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print("Part 1 — findings...")
    findings = build_findings(profiles)
    (OUT / "showcase_findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Part 1 — recipe demos (dose engine)...")
    engine = DoseEngine()
    demos = build_recipe_demos(engine)
    (OUT / "showcase_recipe_demos.json").write_text(
        json.dumps(demos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Part 1 — methodology...")
    method = build_methodology()
    (OUT / "showcase_methodology.json").write_text(
        json.dumps(method, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Part 2 — Neo4j load plan...")
    plan = build_neo4j_plan()
    (OUT / "neo4j_load_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # sizes
    sizes = {p.name: p.stat().st_size for p in OUT.glob("*.json")}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {k: {"bytes": v, "kb": round(v / 1024, 1)} for k, v in sizes.items()},
        "n_hero_ingredients": len(heroes),
        "n_search_nodes": len(index),
    }
    (OUT / "showcase_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
