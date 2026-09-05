#!/usr/bin/env python3
"""
Ingredient profile assembler v1.1 — human-readable pathway (and category) names.

Reads v1 profiles + name lookup sources; writes v1.1 outputs only.
Core canonical files and v1 profiles are read-only.

Usage (from repo root):
    python scripts/product/build_ingredient_profiles_v1_1.py
    python scripts/product/build_ingredient_profiles_v1_1.py --rebuild-from-core

Outputs:
    data/processed/product/ingredient_profiles_v1_1.jsonl
    data/processed/product/ingredient_profiles_index_v1_1.json
    data/processed/product/ingredient_profiles_build_report_v1_1.json
    data/processed/product/pathway_display_names_v1_1.json
    data/processed/product/samples/ingredient_profile_turmeric_v1_1.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from pathway_display_names import (  # noqa: E402
    PathwayNameResolver,
    build_resolution_report,
)

PRODUCT = ROOT / "data/processed/product"
SAMPLES = PRODUCT / "samples"
TIER1 = ROOT / "data/processed/tier1"

V1_JSONL = PRODUCT / "ingredient_profiles_v1.jsonl"
ENRICHMENT = TIER1 / "enrichment_weighted_v3_calibrated.parquet"
CATEGORY_PROFILES = TIER1 / "ingredient_category_profiles_v2.parquet"

OUT_JSONL = PRODUCT / "ingredient_profiles_v1_1.jsonl"
OUT_INDEX = PRODUCT / "ingredient_profiles_index_v1_1.json"
OUT_REPORT = PRODUCT / "ingredient_profiles_build_report_v1_1.json"
OUT_PATHWAY_LOOKUP = PRODUCT / "pathway_display_names_v1_1.json"
OUT_TURMERIC = SAMPLES / "ingredient_profile_turmeric_v1_1.json"

TURMERIC_ID = "SP_000052"


def load_v1_module():
    spec = importlib.util.spec_from_file_location(
        "build_ingredient_profiles_v1",
        SCRIPT_DIR / "build_ingredient_profiles_v1.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def read_profiles_jsonl(path: Path) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))
    return profiles


def build_category_id_lookup() -> dict[tuple[str, str], str]:
    df = pd.read_parquet(CATEGORY_PROFILES)
    df = df[df["category_level"] == "fine_recipe"].copy()
    lookup: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        key = (str(row["ingredient_id"]), str(row["category_name"]))
        lookup[key] = str(row["category_id"])
    return lookup


def enrich_pathway_sections(profile: dict[str, Any], resolver: PathwayNameResolver) -> None:
    pathways = profile.get("pathways", {})
    for section in ("top_ranked", "top_significant"):
        entries = pathways.get(section, [])
        for entry in entries:
            raw = entry.get("pathway", "")
            resolved = resolver.resolve(raw)
            entry["pathway"] = resolved["pathway"]
            entry["pathway_name"] = resolved["pathway_name"]


def enrich_categories(profile: dict[str, Any], cat_lookup: dict[tuple[str, str], str]) -> None:
    species_id = profile["ingredient"]["species_id"]
    for entry in profile.get("categories", {}).get("top", []):
        cat_name = entry.get("category", "")
        cid = cat_lookup.get((species_id, cat_name))
        if cid:
            entry["category_id"] = cid
        # category field already holds the readable name; mirror into category_name
        if "category_name" not in entry:
            entry["category_name"] = cat_name


def audit_readable_fields(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    compound_missing_name = 0
    compound_total = 0
    pathway_missing_name = 0
    pathway_total = 0
    category_missing_id = 0
    category_total = 0

    for p in profiles:
        for c in p.get("compounds", {}).get("top", []):
            compound_total += 1
            name = c.get("name", "")
            if not name or name == c.get("inchikey"):
                compound_missing_name += 1
        for section in ("top_ranked", "top_significant"):
            for pw in p.get("pathways", {}).get(section, []):
                pathway_total += 1
                pname = pw.get("pathway_name", "")
                if not pname or pname == pw.get("pathway"):
                    if pw.get("pathway", "").startswith("R-HSA-"):
                        pathway_missing_name += 1
        for cat in p.get("categories", {}).get("top", []):
            category_total += 1
            if not cat.get("category_id"):
                category_missing_id += 1

    return {
        "compounds": {
            "top_entries": compound_total,
            "missing_readable_name": compound_missing_name,
            "all_named": compound_missing_name == 0,
        },
        "targets": {
            "note": "gene_symbol is human-readable; no ID resolution needed",
        },
        "tissues": {
            "note": "tissue labels are human-readable GTEx names",
        },
        "pathways_in_profiles": {
            "entries": pathway_total,
            "missing_pathway_name": pathway_missing_name,
        },
        "categories": {
            "top_entries": category_total,
            "missing_category_id": category_missing_id,
        },
    }


def write_index(profiles: list[dict[str, Any]], ranking_doc: dict[str, str]) -> None:
    index_entries = []
    for p in profiles:
        ing = p["ingredient"]
        index_entries.append(
            {
                "species_id": ing["species_id"],
                "canonical_name": ing["canonical_name"],
                "mechanism_coverage": ing["mechanism_coverage"],
                "n_compounds": p["compounds"]["count"],
                "n_genes": p["targets"]["count"],
                "measured_fraction": p["provenance"]["measured_fraction"],
                "categories_available": p["categories"]["available"],
                "significant_pathways": p["pathways"]["significant_count"],
                "is_broadly_active": p["pathways"]["is_broadly_active"],
            }
        )
    index = {
        "version": "v1.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_profiles": len(index_entries),
        "schema_additions": [
            "pathways.*.pathway_name (human-readable label alongside pathway ID)",
            "categories.top[].category_id and category_name (traceability)",
        ],
        "ranking_methods": ranking_doc,
        "profiles": index_entries,
    }
    OUT_INDEX.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ingredient profiles v1.1")
    parser.add_argument(
        "--rebuild-from-core",
        action="store_true",
        help="Re-run v1 assembler from core files instead of reading v1 jsonl",
    )
    args = parser.parse_args()

    print("Building pathway display-name lookup ...")
    enrichment = pd.read_parquet(ENRICHMENT, columns=["pathway_id"])
    pathway_ids = enrichment["pathway_id"].astype(str).unique().tolist()
    resolution_report = build_resolution_report(pathway_ids)
    resolver = PathwayNameResolver()

    # Persist full lookup table for retrieval layer
    lookup_table = {
        raw: resolver.resolve(raw)
        for raw in sorted(set(pathway_ids))
    }
    OUT_PATHWAY_LOOKUP.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATHWAY_LOOKUP.write_text(
        json.dumps(
            {
                "version": "v1.1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "resolution_summary": {
                    k: resolution_report[k]
                    for k in (
                        "n_pathways_total",
                        "n_resolved",
                        "n_unresolved",
                        "resolution_rate",
                        "reactome_count",
                        "go_count",
                        "unresolved",
                    )
                },
                "pathways": {
                    raw: {
                        "pathway": v["pathway"],
                        "pathway_name": v["pathway_name"],
                        "name_source": v["name_source"],
                    }
                    for raw, v in lookup_table.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    v1_mod = load_v1_module()

    if args.rebuild_from_core:
        print("Rebuilding base profiles from core (v1 assembler) ...")
        profiles, build_meta = v1_mod.build_profiles()
        source = "rebuilt_from_core_v1_assembler"
    else:
        if not V1_JSONL.exists():
            raise FileNotFoundError(
                f"{V1_JSONL} not found; run build_ingredient_profiles_v1.py first "
                "or pass --rebuild-from-core"
            )
        print(f"Reading base profiles from {V1_JSONL} ...")
        profiles = read_profiles_jsonl(V1_JSONL)
        build_meta = {"n_profiles": len(profiles), "source": "ingredient_profiles_v1.jsonl"}
        source = str(V1_JSONL.relative_to(ROOT))

    print("Enriching profiles with readable pathway and category names ...")
    cat_lookup = build_category_id_lookup()
    for profile in profiles:
        enrich_pathway_sections(profile, resolver)
        enrich_categories(profile, cat_lookup)

    readability_audit = audit_readable_fields(profiles)

    PRODUCT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for profile in profiles:
            fh.write(json.dumps(profile, ensure_ascii=False) + "\n")

    write_index(profiles, v1_mod.RANKING_DOC)

    turmeric = next(p for p in profiles if p["ingredient"]["species_id"] == TURMERIC_ID)
    turmeric_sample = {
        "sample_label": (
            "Turmeric v1.1 — saturated spice with readable pathway names "
            "(is_broadly_active, zero FDR-significant pathways)"
        ),
        "species_id": TURMERIC_ID,
        "profile": turmeric,
        "pathway_names_preview": [
            {"pathway": e["pathway"], "pathway_name": e["pathway_name"]}
            for e in turmeric["pathways"]["top_ranked"]
        ],
    }
    OUT_TURMERIC.write_text(json.dumps(turmeric_sample, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "base_profiles_source": source,
        "pathway_name_resolution": resolution_report,
        "readability_audit": readability_audit,
        "outputs": {
            "jsonl": str(OUT_JSONL.relative_to(ROOT)),
            "index": str(OUT_INDEX.relative_to(ROOT)),
            "pathway_lookup": str(OUT_PATHWAY_LOOKUP.relative_to(ROOT)),
            "turmeric_sample": str(OUT_TURMERIC.relative_to(ROOT)),
        },
        "schema_additions_v1_1": {
            "pathways.top_ranked[]": "pathway (stable ID) + pathway_name (human-readable)",
            "pathways.top_significant[]": "pathway + pathway_name",
            "categories.top[]": "category_id + category_name (category retained)",
            "compounds.top[]": "unchanged — inchikey + name from compound_master_v2",
        },
        "coverage": build_meta,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pathway resolution: {resolution_report['n_resolved']}/{resolution_report['n_pathways_total']} "
          f"({resolution_report['resolution_rate']:.1%})")
    print(f"Unresolved: {resolution_report['n_unresolved']}")
    print(f"Wrote {len(profiles)} profiles -> {OUT_JSONL}")
    print(f"Turmeric sample -> {OUT_TURMERIC}")


if __name__ == "__main__":
    main()
