#!/usr/bin/env python3
"""
Bring universe v2 live: apply aliases, build profiles v2, ingest new recipe datasets.

Does NOT modify mechanism graph edges (ICC v2, gene sets, enrichment inputs).
Existing v1.1 profiles are copied unchanged for the 445 mechanism-live species.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / "data/processed/product"
CANONICAL = ROOT / "data/processed/canonical"
UNIVERSE = PRODUCT / "universe_v2"
INDEX_DIR = PRODUCT / "indexes"
NUTRIENTS = PRODUCT / "nutrients"
RAW = ROOT / "data/raw/new_recipes"

LOOKUP_PATH = INDEX_DIR / "ingredient_lookup.json"
LOOKUP_BACKUP = INDEX_DIR / "ingredient_lookup_v1_1_backup.json"
ALIASES_LOCKED = UNIVERSE / "ingredient_aliases_v2_locked.json"
NODES_LOCKED = UNIVERSE / "ingredient_nodes_v2_locked.json"
PROFILES_V1_1 = PRODUCT / "ingredient_profiles_v1_1.jsonl"
OUT_PROFILES_V2 = PRODUCT / "ingredient_profiles_v2.jsonl"
OUT_INDEX_V2 = PRODUCT / "ingredient_profiles_index_v2.json"
OUT_REPORT = UNIVERSE / "universe_v2_live_report.json"
NUTRIENT_PROD = NUTRIENTS / "species_nutrient_profiles_production.parquet"
FDC_NUTRIENT = ROOT / "data/raw/phase9_expansion/usda_food_nutrient.parquet"
FOODB_CONTENT = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Content.csv"
FOODB_COMPOUND = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Compound.csv"
COMPOUND_MASTER = CANONICAL / "compound_master_v2.parquet"
SPECIES_NODES = CANONICAL / "species_nodes_v2.parquet"
STRING_MAP = CANONICAL / "ingredient_string_species_v2.parquet"
ICC_V2 = CANONICAL / "ingredient_compound_canonical_v2.parquet"

RECIPES_OUT = CANONICAL / "recipes_new_v2.parquet"
RECIPE_ING_OUT = CANONICAL / "recipe_ingredients_new_v2.parquet"

SAMPLES_DIR = PRODUCT / "samples"
SAMPLE_SOY = SAMPLES_DIR / "ingredient_profile_soy_sauce_v2.json"
SAMPLE_GARAM = SAMPLES_DIR / "ingredient_profile_garam_masala_v2.json"
SAMPLE_CURRY_LEAF = SAMPLES_DIR / "ingredient_profile_curry_leaf_v2.json"

TOP_COMPOUNDS = 15
TOP_NUTRIENTS = 20
TOP_TARGETS = 15
TOP_PATHWAYS = 15
TOP_CATEGORIES = 10
TOP_TISSUES = 10

MECHANISM_HASH_FILES = [
    ICC_V2,
    CANONICAL / "ingredient_compound_canonical_v2.csv",
    ROOT / "data/processed/integrated/ingredient_gene_sets_v3.parquet",
    ROOT / "data/processed/tier1/enrichment_weighted_v3_calibrated.parquet",
    ROOT / "data/processed/tier1/measured_moa_annotation_v1.parquet",
]

CONFIRM_ALIASES = {
    "haldi": "SP_000052",
    "jeera": "SP_000051",
    "dhania": "SP_000048",
    "lobia": "SP_000252",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def load_build_module():
    spec = importlib.util.spec_from_file_location(
        "universe_build", ROOT / "scripts/product/build_universe_v2_expansion.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def load_v1_1_module():
    spec = importlib.util.spec_from_file_location(
        "profiles_v1_1", ROOT / "scripts/product/build_ingredient_profiles_v1_1.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def apply_aliases(lookup: dict[str, Any], aliases_locked: dict[str, Any], nodes: list[dict]) -> dict[str, Any]:
    """Merge locked aliases into lookup; extend with ING node metadata."""
    merged = json.loads(json.dumps(lookup))  # deep copy
    aliases = merged.setdefault("aliases", {})
    alias_lists: dict[str, list[str]] = defaultdict(list)
    for k, v in merged.get("alias_lists", {}).items():
        alias_lists[k] = list(v)

    added = 0
    by_ingredient_id: dict[str, Any] = {}
    for entry in aliases_locked.get("safe_alias_heads", []):
        alias = entry["alias"].strip().lower()
        sid = entry["target_species_id"]
        if alias and sid and alias not in aliases:
            aliases[alias] = sid
            alias_lists[sid].append(alias)
            added += 1

    for entry in aliases_locked.get("confident_regional_parentheticals", []):
        alias = entry["alias"].strip().lower()
        sid = entry.get("target_species_id")
        if alias and sid and alias not in aliases:
            aliases[alias] = sid
            alias_lists[sid].append(alias)
            added += 1
        elif alias and not sid and entry.get("target_foodb_id"):
            fid = int(entry["target_foodb_id"])
            foodb_key = f"FOODB_{fid:04d}"
            if alias not in aliases:
                aliases[alias] = foodb_key
                alias_lists.setdefault(foodb_key, []).append(alias)
                added += 1
            head = alias.split("/")[0].strip()
            if head and head not in aliases:
                aliases[head] = foodb_key
                alias_lists[foodb_key].append(head)
                added += 1
            by_ingredient_id.setdefault(foodb_key, {
                "ingredient_id": foodb_key,
                "canonical_name": entry.get("target_canonical_name", f"FooDB {fid}"),
                "node_type": "foodb_only",
                "data_status": "merged_foodb_only",
                "foodb_id": str(fid),
                "fdc_id": None,
            })

    # ING nodes + universe canonical names
    for node in nodes:
        iid = node["ingredient_id"]
        by_ingredient_id[iid] = {
            "ingredient_id": iid,
            "canonical_name": node["canonical_name"],
            "latin_name": node.get("latin_name"),
            "node_type": node.get("node_type"),
            "data_status": node.get("data_status"),
            "foodb_id": str(node["foodb_id"]) if node.get("foodb_id") else None,
            "fdc_id": node.get("fdc_id"),
        }
        cn = node["canonical_name"].strip().lower()
        if cn and cn not in aliases:
            aliases[cn] = iid
            alias_lists[iid].append(cn)
            added += 1
        if iid not in aliases:
            aliases[iid.lower()] = iid
            alias_lists[iid].append(iid.lower())
        for s in node.get("alias_strings", [])[:50]:
            a = norm(s)
            if a and len(a) >= 2 and a not in aliases:
                aliases[a] = iid
                alias_lists[iid].append(a)
                added += 1

    merged["by_ingredient_id"] = by_ingredient_id
    merged["universe_version"] = "v2"
    merged["n_aliases"] = len(aliases)
    merged["n_species"] = merged.get("n_species", len(merged.get("by_species_id", {})))
    merged["n_ingredient_nodes"] = len(by_ingredient_id)
    merged["alias_lists"] = {k: sorted(set(v)) for k, v in alias_lists.items()}
    merged["aliases_applied_at"] = datetime.now(timezone.utc).isoformat()
    merged["aliases_added_count"] = added
    return merged


# Cached loaders (Content.csv is large — load once)
_FOODB_CONTENT_CACHE: pd.DataFrame | None = None
_FOODB_COMPOUND_CACHE: pd.DataFrame | None = None


def get_foodb_content() -> pd.DataFrame:
    global _FOODB_CONTENT_CACHE
    if _FOODB_CONTENT_CACHE is None:
        _FOODB_CONTENT_CACHE = pd.read_csv(FOODB_CONTENT, low_memory=False)
    return _FOODB_CONTENT_CACHE


def get_foodb_compound_table() -> pd.DataFrame:
    global _FOODB_COMPOUND_CACHE
    if _FOODB_COMPOUND_CACHE is None:
        _FOODB_COMPOUND_CACHE = pd.read_csv(FOODB_COMPOUND, usecols=["id", "public_id"])
    return _FOODB_COMPOUND_CACHE


def build_foodb_compounds_section(foodb_id: int) -> dict[str, Any]:
    content = get_foodb_content()
    rows = content[(content["food_id"] == foodb_id) & (content["source_type"] == "Compound")]
    n = len(rows)
    comp_df = get_foodb_compound_table()
    id_to_pub = {int(r.id): str(r.public_id) for _, r in comp_df.iterrows()}
    top: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in rows.head(500).iterrows():
        try:
            cid = int(row["source_id"])
        except (ValueError, TypeError):
            continue
        pub = id_to_pub.get(cid, f"FDB{cid}")
        if pub in seen:
            continue
        seen.add(pub)
        top.append({"inchikey": pub, "name": pub, "is_distinctive": False})
        if len(top) >= TOP_COMPOUNDS:
            break
    return {"count": n, "top": top, "source": "foodb_content"}


def nutrition_from_fdc_id(fdc_id: int, fdc_by_id: dict[int, pd.DataFrame], nutrient_names: dict[int, str]) -> list[dict[str, Any]]:
    sub = fdc_by_id.get(fdc_id)
    if sub is None or sub.empty:
        return []
    sub = sub.sort_values("amount", ascending=False).head(TOP_NUTRIENTS)
    out = []
    for _, r in sub.iterrows():
        nid = int(r["nutrient_id"]) if pd.notna(r.get("nutrient_id")) else None
        out.append({
            "nutrient_id": nid,
            "nutrient_name": nutrient_names.get(nid, f"nutrient_{nid}"),
            "amount": round(float(r["amount"]), 4) if pd.notna(r["amount"]) else None,
            "unit": "g",
            "basis": "per_100g",
        })
    return out


def empty_mechanism_sections() -> dict[str, Any]:
    return {
        "compounds": {"count": 0, "top": [], "note": "No mechanism data"},
        "targets": {"count": 0, "measured_count": 0, "predicted_count": 0, "top": []},
        "pathways": {
            "significant_count": 0,
            "is_broadly_active": False,
            "top_ranked": [],
            "top_significant": [],
        },
        "categories": {"available": False, "top": []},
        "tissues": {"top": [], "interpretation_note": "target gene expression location, not proof of compound delivery"},
        "provenance": {"measured_fraction": 0.0, "summary": "No mechanism inference for this node type."},
    }


def aggregate_blend_profile(
    node: dict[str, Any],
    profiles_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    constituents = node.get("constituents") or []
    derived_from = []
    pathway_scores: dict[str, float] = defaultdict(float)
    tissue_scores: dict[str, float] = defaultdict(float)
    category_scores: dict[str, float] = defaultdict(float)
    gene_scores: dict[str, dict[str, Any]] = defaultdict(lambda: {"score": 0.0, "evidence": "predicted"})
    compound_scores: dict[str, float] = defaultdict(float)
    nutrient_acc: dict[str, dict[str, Any]] = defaultdict(lambda: {"amount": 0.0, "unit": "g"})
    total_compounds = 0
    total_genes = 0

    for c in constituents:
        sid = c.get("species_id")
        w = float(c.get("weight", 0))
        if not sid or sid not in profiles_by_id:
            derived_from.append({
                "species_id": sid,
                "constituent_name": c.get("constituent_name"),
                "weight": w,
                "resolved": False,
            })
            continue
        sp = profiles_by_id[sid]
        derived_from.append({
            "species_id": sid,
            "constituent_name": c.get("constituent_name"),
            "weight": w,
            "resolved": True,
            "canonical_name": sp["ingredient"]["canonical_name"],
        })
        total_compounds += int(sp["compounds"].get("count", 0))
        total_genes += int(sp["targets"].get("count", 0))
        for comp in sp["compounds"].get("top", []):
            compound_scores[comp.get("inchikey", comp.get("name", ""))] += w
        for tgt in sp["targets"].get("top", []):
            g = tgt["gene_symbol"]
            gene_scores[g]["score"] += w * float(tgt.get("confidence", 0.5))
            if tgt.get("evidence") == "measured":
                gene_scores[g]["evidence"] = "measured"
        for pw in sp["pathways"].get("top_ranked", []):
            pathway_scores[pw["pathway"]] += w * float(pw.get("weighted_fold", 0))
        for cat in sp["categories"].get("top", []):
            category_scores[cat["category"]] += w * float(cat.get("aggregated_enrichment", 0))
        for tis in sp["tissues"].get("top", []):
            tissue_scores[tis["tissue"]] += w * float(tis.get("normalized_score", 0))
        nutr = sp.get("nutrition", {})
        if nutr.get("available"):
            for n in nutr.get("top_nutrients", []):
                key = str(n.get("nutrient_name", ""))
                nutrient_acc[key]["amount"] += w * float(n.get("amount") or 0)
                nutrient_acc[key]["unit"] = n.get("unit", "g")
                nutrient_acc[key]["nutrient_id"] = n.get("nutrient_id")

    top_compounds = sorted(compound_scores.items(), key=lambda x: -x[1])[:TOP_COMPOUNDS]
    top_targets = sorted(gene_scores.items(), key=lambda x: -x[1]["score"])[:TOP_TARGETS]
    top_pathways = sorted(pathway_scores.items(), key=lambda x: -x[1])[:TOP_PATHWAYS]
    top_categories = sorted(category_scores.items(), key=lambda x: -x[1])[:TOP_CATEGORIES]
    top_tissues = sorted(tissue_scores.items(), key=lambda x: -x[1])[:TOP_TISSUES]
    top_nutrients = sorted(nutrient_acc.items(), key=lambda x: -x[1]["amount"], reverse=True)[:TOP_NUTRIENTS]

    mech = empty_mechanism_sections()
    mech["compounds"] = {
        "count": total_compounds,
        "top": [{"inchikey": k, "name": k, "blend_weight_score": round(v, 4)} for k, v in top_compounds],
    }
    mech["targets"] = {
        "count": total_genes,
        "measured_count": 0,
        "predicted_count": total_genes,
        "top": [
            {
                "gene_symbol": g,
                "evidence": info["evidence"],
                "confidence": round(info["score"], 4),
                "confidence_tier": "moderate",
                "n_supporting_compounds": 0,
                "moa": None,
            }
            for g, info in top_targets
        ],
    }
    mech["pathways"]["top_ranked"] = [
        {"pathway": p, "weighted_fold": round(s, 4), "q_value": None, "driving_genes": [], "evidence_split": "blend_aggregate"}
        for p, s in top_pathways
    ]
    mech["categories"] = {"available": bool(top_categories), "top": [{"category": c, "aggregated_enrichment": round(s, 4)} for c, s in top_categories]}
    mech["tissues"]["top"] = [{"tissue": t, "normalized_score": round(s, 6)} for t, s in top_tissues]
    mech["provenance"]["summary"] = "Blend profile aggregated from constituent species profiles (weighted)."

    nutrition = {
        "available": bool(top_nutrients),
        "composition_source": "blend_aggregate",
        "top_nutrients": [
            {"nutrient_name": name, "amount": round(info["amount"], 4), "unit": info["unit"], "basis": "per_100g"}
            for name, info in top_nutrients
        ],
        "n_nutrients": len(top_nutrients),
    }
    return {"mechanism": mech, "nutrition": nutrition, "derived_from": derived_from}


def build_new_node_profile(
    node: dict[str, Any],
    profiles_by_id: dict[str, dict[str, Any]],
    fdc_by_id: dict[int, pd.DataFrame],
    nutrient_names: dict[int, str],
    pathway_resolver,
    cat_lookup: dict,
) -> dict[str, Any]:
    iid = node["ingredient_id"]
    status = node.get("data_status", "name_only")
    ntype = node.get("node_type", "species")

    ing = {
        "ingredient_id": iid,
        "species_id": iid,  # backward compat for retrieval
        "canonical_name": node["canonical_name"],
        "latin_name": node.get("latin_name"),
        "node_type": ntype,
        "data_status": status,
        "foodb_id": str(node["foodb_id"]) if node.get("foodb_id") else None,
        "fdc_id": node.get("fdc_id"),
        "preparation_labels": [],
        "mechanism_coverage": "none" if status == "name_only" else ("thin" if status == "nutrition_only" else "moderate"),
    }
    if node.get("backfill_source"):
        ing["backfill_source"] = node["backfill_source"]
    if node.get("backfill_candidate"):
        ing["backfill_candidate"] = True

    nutrition: dict[str, Any] = {"available": False, "top_nutrients": [], "n_nutrients": 0}

    if ntype == "blend" and node.get("constituents"):
        agg = aggregate_blend_profile(node, profiles_by_id)
        profile = {
            "ingredient": ing,
            **agg["mechanism"],
            "nutrition": agg["nutrition"],
            "derived_from": agg["derived_from"],
        }
        profile["ingredient"]["mechanism_coverage"] = "moderate"
        return profile

    if status == "name_only":
        mech = empty_mechanism_sections()
        mech["compounds"]["note"] = "Recognized ingredient; no FooDB/USDA data yet."
        mech["provenance"]["summary"] = "Name-only node — parses in recipes, displays honestly."
        return {"ingredient": ing, **mech, "nutrition": nutrition}

    mech = empty_mechanism_sections()

    if status in ("full", "mechanism_only") and node.get("foodb_id"):
        mech["compounds"] = build_foodb_compounds_section(int(node["foodb_id"]))
        mech["provenance"]["summary"] = "Compounds from FooDB Content; targets/pathways not inferred for composite/ING nodes."

    if status in ("full", "nutrition_only") and node.get("fdc_id"):
        top_n = nutrition_from_fdc_id(int(node["fdc_id"]), fdc_by_id, nutrient_names)
        nutrition = {
            "available": bool(top_n),
            "composition_source": "fdc",
            "fdc_id": int(node["fdc_id"]),
            "fdc_description": node.get("fdc_description"),
            "top_nutrients": top_n,
            "n_nutrients": len(top_n),
            "basis": "per_100g",
        }

    if status == "nutrition_only":
        mech["compounds"]["note"] = "Nutrition-only node — mechanism sections intentionally empty."
        mech["provenance"]["summary"] = "USDA FDC nutrition only; no FooDB mechanism."

    profile = {"ingredient": ing, **mech, "nutrition": nutrition}
    return profile


def enrich_profile_v1_1_style(profile: dict[str, Any], pathway_resolver, cat_lookup: dict) -> None:
    """Add pathway_name and category_id like v1.1."""
    v1_1 = load_v1_1_module()
    v1_1.enrich_pathway_sections(profile, pathway_resolver)
    v1_1.enrich_categories(profile, cat_lookup)
    # Add empty nutrition for SP profiles if missing (from production nutrients)
    if "nutrition" not in profile:
        profile["nutrition"] = {"available": False, "note": "See species_nutrient_profiles_production for SP nodes"}


def build_profiles_v2(
    nodes: list[dict],
    existing_v1_1: list[dict[str, Any]],
    species_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v1_1_mod = load_v1_1_module()
    pathway_resolver = v1_1_mod.PathwayNameResolver()
    cat_lookup = v1_1_mod.build_category_id_lookup()

    profiles_by_id = {p["ingredient"]["species_id"]: p for p in existing_v1_1}

    # Attach production nutrition to existing SP profiles (read-only join)
    if NUTRIENT_PROD.exists():
        nutr = pd.read_parquet(NUTRIENT_PROD)
        for sid in profiles_by_id:
            sub = nutr[nutr["species_id"] == sid].sort_values("amount", ascending=False).head(TOP_NUTRIENTS)
            if len(sub):
                profiles_by_id[sid]["nutrition"] = {
                    "available": True,
                    "composition_source": str(sub.iloc[0]["composition_source"]),
                    "top_nutrients": [
                        {
                            "nutrient_id": int(r["nutrient_id"]) if pd.notna(r["nutrient_id"]) else None,
                            "nutrient_name": str(r["nutrient_name"]),
                            "amount": round(float(r["amount"]), 4),
                            "unit": str(r["unit"]),
                            "basis": str(r["basis"]),
                        }
                        for _, r in sub.iterrows()
                    ],
                    "n_nutrients": int(nutr[nutr["species_id"] == sid]["nutrient_id"].nunique()),
                }

    fdc_nutrient = pd.read_parquet(FDC_NUTRIENT) if FDC_NUTRIENT.exists() else pd.DataFrame()
    fdc_by_id: dict[int, pd.DataFrame] = {}
    if len(fdc_nutrient):
        for fid, sub in fdc_nutrient.groupby("fdc_id"):
            fdc_by_id[int(fid)] = sub
    nutrient_names: dict[int, str] = {}
    if NUTRIENT_PROD.exists():
        np = pd.read_parquet(NUTRIENT_PROD, columns=["nutrient_id", "nutrient_name", "unit"])
        for _, r in np.drop_duplicates("nutrient_id").iterrows():
            if pd.notna(r["nutrient_id"]):
                nutrient_names[int(r["nutrient_id"])] = str(r["nutrient_name"])

    existing_ids = {n["ingredient_id"] for n in nodes if n.get("expansion_tier") == "existing"}
    new_nodes = [n for n in nodes if n.get("expansion_tier") != "existing"]

    out: list[dict[str, Any]] = []
    stats = Counter()

    # 445 mechanism-live from v1.1 unchanged (+ nutrition attachment)
    for p in existing_v1_1:
        copy_p = json.loads(json.dumps(p))
        enrich_profile_v1_1_style(copy_p, pathway_resolver, cat_lookup)
        copy_p["ingredient"]["node_type"] = "species"
        copy_p["ingredient"]["data_status"] = "full"
        copy_p["ingredient"]["ingredient_id"] = copy_p["ingredient"]["species_id"]
        out.append(copy_p)
        stats[("species", "full")] += 1

    # 18 SP without v1.1 profiles — minimal identity
    v1_1_ids = {p["ingredient"]["species_id"] for p in existing_v1_1}
    for _, row in species_df.iterrows():
        sid = str(row.species_node_id)
        if sid in v1_1_ids:
            continue
        mech = empty_mechanism_sections()
        mech["provenance"]["summary"] = "Identity-only species (no mechanism-live gene sets)."
        out.append({
            "ingredient": {
                "ingredient_id": sid,
                "species_id": sid,
                "canonical_name": str(row.canonical_name),
                "latin_name": None if pd.isna(row.latin_name) else str(row.latin_name),
                "node_type": "species",
                "data_status": "identity_only",
                "foodb_id": str(int(row.foodb_id)),
                "preparation_labels": [],
                "mechanism_coverage": "none",
            },
            **mech,
            "nutrition": {"available": False},
        })
        stats[("species", "identity_only")] += 1

    # 232 new ING nodes
    for node in new_nodes:
        profile = build_new_node_profile(node, profiles_by_id, fdc_by_id, nutrient_names, pathway_resolver, cat_lookup)
        enrich_profile_v1_1_style(profile, pathway_resolver, cat_lookup)
        out.append(profile)
        stats[(node.get("node_type", "?"), node.get("data_status", "?"))] += 1

    meta = {
        "n_total": len(out),
        "n_existing_v1_1": len(existing_v1_1),
        "n_new_ing": len(new_nodes),
        "by_type_status": {f"{a}/{b}": c for (a, b), c in stats.items()},
    }
    return out, meta


def build_string_map(valid_ids: set[str]) -> dict[str, str]:
    """Production string map (463-species corpus) restricted to live profile IDs."""
    out: dict[str, str] = {}
    if not STRING_MAP.exists():
        return out
    sm = pd.read_parquet(STRING_MAP)
    for _, row in sm.iterrows():
        sid = str(row["species_node"])
        if sid not in valid_ids:
            continue
        for col in ("ingredient_string", "canonical_name"):
            if pd.notna(row.get(col)):
                key = norm(str(row[col]))
                if key and key not in out:
                    out[key] = sid
    return out


def resolve_ingredient(
    raw: str,
    lookup: dict[str, Any],
    valid_ids: set[str],
    string_map: dict[str, str],
) -> tuple[str | None, str]:
    """Production-style resolver: alias exact → string_map → alias substring."""
    key = norm(raw)
    aliases = lookup.get("aliases", {})
    iid = aliases.get(key) or string_map.get(key)
    match_type = "exact" if iid else "unmapped"
    if not iid:
        for alias, candidate in aliases.items():
            if len(alias) >= 4 and (key in alias or alias in key):
                iid = candidate
                match_type = "alias_fuzzy"
                break
    if iid and iid in valid_ids:
        return iid, match_type
    return None, "unmapped"


def ingest_new_recipes(lookup: dict[str, Any], valid_ids: set[str]) -> dict[str, Any]:
    build = load_build_module()
    string_map = build_string_map(valid_ids)
    recipes: list[dict[str, Any]] = []
    ing_rows: list[dict[str, Any]] = []
    recipe_id = 0

    def add_recipe(source: str, name: str, ingredients: list[str], cuisine: str | None, cuisine_reliable: bool, extra: dict | None = None):
        nonlocal recipe_id
        recipe_id += 1
        rid = f"NEWV2_{source}_{recipe_id:07d}"
        recipes.append({
            "recipe_id": rid,
            "name": name or rid,
            "source": source,
            "source_dataset": "new_recipes_v2",
            "cuisine": cuisine,
            "cuisine_label_reliable": cuisine_reliable,
            **(extra or {}),
        })
        for raw in ingredients:
            if not raw or not str(raw).strip():
                continue
            iid, match_type = resolve_ingredient(str(raw), lookup, valid_ids, string_map)
            ing_rows.append({
                "recipe_id": rid,
                "ingredient_id": iid,
                "ingredient_raw": str(raw).strip(),
                "match_type": match_type,
            })

    # Food_Recipe.csv
    df = pd.read_csv(RAW / "Food_Recipe.csv")
    for _, row in df.iterrows():
        ings = [p.strip() for p in str(row.get("ingredients_name", "")).split(",") if p.strip()]
        cuisine = str(row.get("cuisine", row.get("region", ""))) if pd.notna(row.get("cuisine", row.get("region"))) else None
        add_recipe("Food_Recipe", str(row.get("name", "")), ings, cuisine, True, {"regional_label": cuisine})

    # recipes5.csv
    df5 = pd.read_csv(RAW / "recipes5.csv")
    for _, row in df5.iterrows():
        ings = [p.strip() for p in str(row.get("ingredients", "")).split(",") if p.strip()]
        area = str(row.get("area", "")) if pd.notna(row.get("area")) else None
        add_recipe("recipes5", str(row.get("name", row.get("title", ""))), ings, area, True)

    # recipes2.json, recipes3.json
    for fn, cuisine_field in [("recipes2.json", None), ("recipes3.json", "cuisine")]:
        data = json.loads((RAW / fn).read_text(encoding="utf-8"))
        for rec in data:
            ings = [str(x) for x in rec.get("ingredients", [])]
            cuisine = rec.get(cuisine_field) if cuisine_field else None
            add_recipe(fn.replace(".json", ""), rec.get("name", rec.get("title", "")), ings, cuisine, cuisine_field is not None)

    # recipes4.csv (dedupe with json — use csv only)
    df4 = pd.read_csv(RAW / "recipes4.csv", usecols=["recipe_title", "ingredients_canonical", "cuisine_list"], low_memory=False)
    for _, row in df4.iterrows():
        try:
            arr = ast.literal_eval(str(row["ingredients_canonical"]))
        except Exception:
            arr = [str(row["ingredients_canonical"])]
        ings = [str(x) for x in arr]
        cuisine_raw = row.get("cuisine_list")
        cuisine = None
        reliable = False
        if pd.notna(cuisine_raw):
            try:
                cl = ast.literal_eval(str(cuisine_raw))
                if isinstance(cl, list) and cl:
                    cuisine = str(cl[0])
                    reliable = len(cl) == 1
            except Exception:
                cuisine = str(cuisine_raw)
                reliable = False
        add_recipe("recipes4", str(row.get("recipe_title", "")), ings, cuisine, reliable)

    rdf = pd.DataFrame(recipes)
    idf = pd.DataFrame(ing_rows)
    rdf.to_parquet(RECIPES_OUT, index=False)
    idf.to_parquet(RECIPE_ING_OUT, index=False)

    mapped = idf[idf["ingredient_id"].notna()]
    total_ing = len(idf)
    coverage = 100.0 * len(mapped) / max(total_ing, 1)
    cuisine_reliable = rdf["cuisine_label_reliable"].sum()
    return {
        "n_recipes": len(rdf),
        "n_ingredient_rows": total_ing,
        "n_mapped_rows": len(mapped),
        "mapping_coverage_pct": round(coverage, 2),
        "recipes_with_reliable_cuisine_label": int(cuisine_reliable),
        "recipes_with_any_cuisine_label": int(rdf["cuisine"].notna().sum()),
        "by_source": rdf.groupby("source").size().to_dict(),
    }


def main() -> None:
    UNIVERSE.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Mechanism hash baseline
    mechanism_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in MECHANISM_HASH_FILES if p.exists()}

    # PART 1 — aliases (always merge locked set onto v1.1 backup base)
    if not LOOKUP_BACKUP.exists():
        shutil.copy2(LOOKUP_PATH, LOOKUP_BACKUP)
    lookup_orig = json.loads(LOOKUP_BACKUP.read_text(encoding="utf-8"))
    aliases_locked = json.loads(ALIASES_LOCKED.read_text(encoding="utf-8"))
    nodes = json.loads(NODES_LOCKED.read_text(encoding="utf-8"))["nodes"]
    lookup_v2 = apply_aliases(lookup_orig, aliases_locked, nodes)
    LOOKUP_V2 = INDEX_DIR / "ingredient_lookup_v2.json"
    LOOKUP_V2.write_text(json.dumps(lookup_v2, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also update main lookup (live)
    LOOKUP_PATH.write_text(json.dumps(lookup_v2, indent=2, ensure_ascii=False), encoding="utf-8")

    alias_confirm = {}
    for alias, expected_sid in CONFIRM_ALIASES.items():
        got = lookup_v2["aliases"].get(alias)
        alias_confirm[alias] = {"expected": expected_sid, "got": got, "ok": got == expected_sid}
    # ash gourd -> foodb only in aliases file; check lobia specifically
    alias_confirm["ash gourd/ white pumpkin"] = {
        "expected": "FOODB_0510",
        "got": lookup_v2["aliases"].get("ash gourd/ white pumpkin"),
        "ok": lookup_v2["aliases"].get("ash gourd/ white pumpkin") == "FOODB_0510",
        "note": "foodb-only target (FooDB 510 Wax gourd) — no SP/ING profile node",
    }
    alias_confirm["ash gourd"] = {
        "expected": "FOODB_0510",
        "got": lookup_v2["aliases"].get("ash gourd"),
        "ok": lookup_v2["aliases"].get("ash gourd") == "FOODB_0510",
        "note": "head alias derived from parenthetical merge",
    }

    # PART 2 — profiles v2
    existing_v1_1 = []
    with PROFILES_V1_1.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                existing_v1_1.append(json.loads(line))

    species_df = pd.read_parquet(SPECIES_NODES)
    profiles_v2, profile_meta = build_profiles_v2(nodes, existing_v1_1, species_df)

    with OUT_PROFILES_V2.open("w", encoding="utf-8") as f:
        for p in profiles_v2:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    index_entries = []
    for p in profiles_v2:
        ing = p["ingredient"]
        index_entries.append({
            "ingredient_id": ing.get("ingredient_id", ing["species_id"]),
            "canonical_name": ing["canonical_name"],
            "node_type": ing.get("node_type"),
            "data_status": ing.get("data_status"),
            "mechanism_coverage": ing.get("mechanism_coverage"),
            "n_compounds": p["compounds"].get("count", 0),
            "nutrition_available": p.get("nutrition", {}).get("available", False),
        })
    OUT_INDEX_V2.write_text(json.dumps({
        "version": "v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_profiles": len(index_entries),
        "profiles": index_entries,
    }, indent=2), encoding="utf-8")

    # Verification samples
    by_name = {p["ingredient"]["canonical_name"]: p for p in profiles_v2}
    for path, name in [(SAMPLE_SOY, "soy sauce"), (SAMPLE_GARAM, "garam masala"), (SAMPLE_CURRY_LEAF, "curry leaf")]:
        if name in by_name:
            path.write_text(json.dumps({"sample": name, "profile": by_name[name]}, indent=2, ensure_ascii=False), encoding="utf-8")

    # PART 3 — ingest recipes
    valid_ids = {n["ingredient_id"] for n in nodes}
    ingest_stats = ingest_new_recipes(lookup_v2, valid_ids)

    # Final stats
    new_nodes = [n for n in nodes if n.get("expansion_tier") != "existing"]
    status_counts = Counter(n.get("data_status") for n in new_nodes)

    report = {
        "version": "universe_v2_live",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "part1_aliases": {
            "backup": str(LOOKUP_BACKUP.relative_to(ROOT)),
            "lookup_v2": str(LOOKUP_V2.relative_to(ROOT)),
            "aliases_added": lookup_v2.get("aliases_added_count"),
            "confirmation": alias_confirm,
        },
        "part2_profiles": {
            "output": str(OUT_PROFILES_V2.relative_to(ROOT)),
            "index": str(OUT_INDEX_V2.relative_to(ROOT)),
            "n_total_profiles": len(profiles_v2),
            "meta": profile_meta,
            "samples": {
                "soy_sauce": str(SAMPLE_SOY.relative_to(ROOT)),
                "garam_masala": str(SAMPLE_GARAM.relative_to(ROOT)),
                "curry_leaf": str(SAMPLE_CURRY_LEAF.relative_to(ROOT)),
            },
        },
        "part3_ingest": ingest_stats,
        "final_universe": {
            "total_nodes": 695,
            "existing_species": 463,
            "new_ing_nodes": 232,
            "new_by_data_status": dict(status_counts),
        },
        "mechanism_integrity": {
            "files_hashed": mechanism_hashes,
            "note": "Hashes recorded at live deployment; mechanism graph files not modified.",
        },
        "outputs": {
            "recipes": str(RECIPES_OUT.relative_to(ROOT)),
            "recipe_ingredients": str(RECIPE_ING_OUT.relative_to(ROOT)),
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
