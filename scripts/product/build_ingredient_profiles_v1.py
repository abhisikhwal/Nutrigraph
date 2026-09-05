#!/usr/bin/env python3
"""
Assemble one product-shaped ingredient profile per mechanism-live species (445).

Reads ONLY canonical core files (read-only). Writes new product outputs only.

Usage (from repo root):
    python scripts/product/build_ingredient_profiles_v1.py

Outputs:
    data/processed/product/ingredient_profiles_v1.jsonl
    data/processed/product/ingredient_profiles_index_v1.json
    data/processed/product/ingredient_profiles_build_report_v1.json
    data/processed/product/samples/ingredient_profile_{turmeric,black_tea,rapeseed_oil}_v1.json
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
CANONICAL = ROOT / "data/processed/canonical"
INTEGRATED = ROOT / "data/processed/integrated"
TIER1 = ROOT / "data/processed/tier1"
PRODUCT = ROOT / "data/processed/product"
SAMPLES = PRODUCT / "samples"

SPECIES_NODES = CANONICAL / "species_nodes_v2.parquet"
STRING_MAP = CANONICAL / "ingredient_string_species_v2.parquet"
ICC_V2 = CANONICAL / "ingredient_compound_canonical_v2.parquet"
COMPOUND_MASTER = CANONICAL / "compound_master_v2.parquet"
GENE_SETS_V3 = INTEGRATED / "ingredient_gene_sets_v3.parquet"
INTEGRATED_CG = INTEGRATED / "compound_gene_integrated_v1.parquet"
ENRICHMENT = TIER1 / "enrichment_weighted_v3_calibrated.parquet"
CATEGORY_PROFILES = TIER1 / "ingredient_category_profiles_v2.parquet"
TISSUE_PROFILES = TIER1 / "ingredient_tissue_profiles_v2.parquet"
MOA = TIER1 / "measured_moa_annotation_v1.parquet"

OUT_JSONL = PRODUCT / "ingredient_profiles_v1.jsonl"
OUT_INDEX = PRODUCT / "ingredient_profiles_index_v1.json"
OUT_REPORT = PRODUCT / "ingredient_profiles_build_report_v1.json"

TOP_COMPOUNDS = 15
TOP_TARGETS = 15
TOP_PATHWAYS_RANKED = 15
TOP_PATHWAYS_SIG = 15
TOP_CATEGORIES = 10
TOP_TISSUES = 10
Q_SIG = 0.10

TIER_ORDER = {"high": 3, "moderate": 2, "low": 1}
CG_TIER_MAP = {"predicted_high": "high", "predicted_moderate": "moderate"}

SAMPLE_SPECIES = {
    "turmeric": "SP_000052",
    "black_tea": "SP_000415",
    "rapeseed_oil": "SP_000418",
}

TISSUE_INTERPRETATION = (
    "target gene expression location, not proof of compound delivery"
)

RANKING_DOC = {
    "compounds": (
        "Top compounds by ingredient-specific distinctiveness: "
        "log(N_species / n_species_with_compound) computed from ICC v2 "
        "(inverse frequency across 445 mechanism-live species)."
    ),
    "targets": (
        "Top genes by evidence tier (measured before predicted), then confidence, "
        "then n_supporting_compounds."
    ),
    "pathways_top_ranked": (
        "Top pathways by weighted_fold_enrichment descending (all q-values); "
        "always populated so saturated spices never show an empty pathway view."
    ),
    "pathways_top_significant": (
        "FDR-passing subset (q < 0.10), ranked by weighted_fold_enrichment."
    ),
    "categories": (
        "Top fine_recipe categories by aggregated_enrichment descending."
    ),
    "tissues": (
        "Top tissues by normalized_score descending (enrichment-de-saturated v2)."
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if pd.isna(obj):
        return None
    return obj


def parse_preparation_labels(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, (list, np.ndarray)):
        return sorted({str(x) for x in raw if x is not None and str(x).strip()})
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return sorted({str(x) for x in parsed if str(x).strip()})
        except json.JSONDecodeError:
            pass
        return [s]
    return [str(raw)]


def mechanism_coverage(n_compounds: int, n_genes: int) -> str:
    if n_compounds < 50 or n_genes < 150:
        return "thin"
    if n_compounds >= 1000 and n_genes >= 800:
        return "rich"
    return "moderate"


def tier_from_gene_row(evidence: str, confidence: float, cg_tier: str | None) -> str:
    if evidence == "measured" or (cg_tier is None and confidence >= 0.95):
        return "high"
    if cg_tier:
        return CG_TIER_MAP.get(cg_tier, "moderate")
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.5:
        return "moderate"
    return "low"


def pathway_label(pathway_id: str) -> str:
    """Use pathway_id as display label (GO entries are human-readable; Reactome IDs as stable refs)."""
    return str(pathway_id)


def evidence_split(row: pd.Series) -> str:
    m = int(row.get("n_measured_overlap", 0) or 0)
    p = int(row.get("n_predicted_overlap", 0) or 0)
    if m and p:
        return f"{m} measured / {p} predicted"
    if m:
        return f"{m} measured"
    if p:
        return f"{p} predicted"
    return "unknown"


def driving_gene_symbols(driving_genes_json: str) -> list[str]:
    try:
        drivers = json.loads(driving_genes_json)
    except (json.JSONDecodeError, TypeError):
        return []
    return [d["gene_symbol"] for d in drivers if d.get("gene_symbol")]


def load_compound_names() -> dict[str, str]:
    cm = pd.read_parquet(COMPOUND_MASTER, columns=["compound_id", "name"])
    cm["compound_id"] = cm["compound_id"].astype(str).str.upper()
    cm["name"] = cm["name"].fillna("").astype(str).str.strip()
    names: dict[str, str] = {}
    for cid, sub in cm.groupby("compound_id"):
        nonempty = sub[sub["name"] != ""]["name"]
        names[cid] = nonempty.iloc[0] if len(nonempty) else cid
    return names


def build_compound_idf(icc: pd.DataFrame, species_ids: set[str]) -> dict[str, float]:
    icc = icc[icc["ingredient_id"].isin(species_ids)]
    freq = icc.groupby("compound_id")["ingredient_id"].nunique()
    n = max(1, len(species_ids))
    return {cid: math.log(n / max(1, int(cnt))) for cid, cnt in freq.items()}


def build_gene_tier_lookup(icc: pd.DataFrame, cg: pd.DataFrame) -> dict[tuple[str, str], str]:
    merged = icc.merge(cg, on="compound_id", how="inner")
    tier_rank = merged["confidence_tier"].map(lambda t: TIER_ORDER.get(CG_TIER_MAP.get(str(t), "low"), 0))
    merged = merged.assign(_tier_rank=tier_rank.fillna(0))
    best = (
        merged.sort_values("_tier_rank", ascending=False)
        .groupby(["ingredient_id", "gene_symbol"], as_index=False)
        .first()
    )
    lookup: dict[tuple[str, str], str] = {}
    for _, row in best.iterrows():
        tier_raw = row.get("confidence_tier")
        tier = CG_TIER_MAP.get(str(tier_raw), "moderate") if pd.notna(tier_raw) else "moderate"
        lookup[(str(row["ingredient_id"]), str(row["gene_symbol"]))] = tier
    return lookup


def build_moa_lookup(icc: pd.DataFrame, moa: pd.DataFrame) -> dict[tuple[str, str], str]:
    moa = moa.rename(columns={"compound": "compound_id"})
    moa["compound_id"] = moa["compound_id"].astype(str).str.upper()
    merged = icc.merge(moa, on="compound_id", how="inner")
    lookup: dict[tuple[str, str], str] = {}
    for (ing, gene), sub in merged.groupby(["ingredient_id", "gene_symbol"]):
        actions = sub["action_type"].dropna().astype(str).unique().tolist()
        if actions:
            lookup[(str(ing), str(gene))] = actions[0]
    return lookup


def build_profiles() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Canonical read set includes string map (recipe join layer); not merged into species profiles.
    _string_map = pd.read_parquet(STRING_MAP)

    species = pd.read_parquet(SPECIES_NODES)
    icc = pd.read_parquet(ICC_V2)
    icc["ingredient_id"] = icc["ingredient_id"].astype(str)
    icc["compound_id"] = icc["compound_id"].astype(str).str.upper()

    gene_sets = pd.read_parquet(GENE_SETS_V3)
    gene_sets["ingredient_id"] = gene_sets["ingredient_id"].astype(str)

    species_ids = set(gene_sets["ingredient_id"].unique())
    n_species = len(species_ids)

    compound_names = load_compound_names()
    compound_idf = build_compound_idf(icc, species_ids)

    cg = pd.read_parquet(INTEGRATED_CG)
    cg["compound_id"] = cg["compound_id"].astype(str).str.upper()
    gene_tier_lookup = build_gene_tier_lookup(icc, cg)

    moa_df = pd.read_parquet(MOA)
    moa_lookup = build_moa_lookup(icc, moa_df)

    enrichment = pd.read_parquet(ENRICHMENT)
    enrichment["ingredient_id"] = enrichment["ingredient_id"].astype(str)

    categories = pd.read_parquet(CATEGORY_PROFILES)
    categories = categories[categories["category_level"] == "fine_recipe"].copy()
    categories["ingredient_id"] = categories["ingredient_id"].astype(str)

    tissues = pd.read_parquet(TISSUE_PROFILES)
    tissues["ingredient_id"] = tissues["ingredient_id"].astype(str)

    species = species[species["species_node_id"].isin(species_ids)].copy()
    species = species.set_index("species_node_id")

    compound_counts = icc.groupby("ingredient_id").size().to_dict()
    gene_counts = gene_sets.groupby("ingredient_id").size().to_dict()
    measured_gene_counts = (
        gene_sets[gene_sets["evidence"] == "measured"]
        .groupby("ingredient_id")
        .size()
        .to_dict()
    )
    predicted_gene_counts = (
        gene_sets[gene_sets["evidence"] == "predicted"]
        .groupby("ingredient_id")
        .size()
        .to_dict()
    )

    icc_by_ing: dict[str, pd.DataFrame] = {
        ing: sub for ing, sub in icc.groupby("ingredient_id")
    }
    gs_by_ing: dict[str, pd.DataFrame] = {
        ing: sub for ing, sub in gene_sets.groupby("ingredient_id")
    }
    enr_by_ing: dict[str, pd.DataFrame] = {
        ing: sub for ing, sub in enrichment.groupby("ingredient_id")
    }
    cat_by_ing: dict[str, pd.DataFrame] = {
        ing: sub for ing, sub in categories.groupby("ingredient_id")
    }
    tissue_by_ing: dict[str, pd.DataFrame] = {
        ing: sub for ing, sub in tissues.groupby("ingredient_id")
    }

    profiles: list[dict[str, Any]] = []
    coverage_stats = defaultdict(int)

    for species_id in sorted(species_ids):
        node = species.loc[species_id]
        n_compounds = int(compound_counts.get(species_id, 0))
        n_genes = int(gene_counts.get(species_id, 0))
        cov = mechanism_coverage(n_compounds, n_genes)
        coverage_stats[cov] += 1

        # --- compounds ---
        icc_sub = icc_by_ing.get(species_id)
        top_compounds: list[dict[str, Any]] = []
        if icc_sub is not None and len(icc_sub):
            comp_df = icc_sub[["compound_id"]].drop_duplicates().copy()
            comp_df["distinctiveness"] = comp_df["compound_id"].map(
                lambda c: compound_idf.get(c, 0.0)
            )
            comp_df = comp_df.sort_values("distinctiveness", ascending=False).head(TOP_COMPOUNDS)
            for _, row in comp_df.iterrows():
                cid = row["compound_id"]
                top_compounds.append(
                    {
                        "inchikey": cid,
                        "name": compound_names.get(cid, cid),
                        "is_distinctive": bool(row["distinctiveness"] >= 1.0),
                    }
                )

        # --- targets ---
        gs_sub = gs_by_ing.get(species_id)
        measured_count = int(measured_gene_counts.get(species_id, 0))
        predicted_count = int(predicted_gene_counts.get(species_id, 0))
        top_targets: list[dict[str, Any]] = []

        if gs_sub is not None and len(gs_sub):
            gs_rank = gs_sub.copy()
            gs_rank["_evidence_rank"] = (gs_rank["evidence"] != "measured").astype(int)
            gs_rank = gs_rank.sort_values(
                ["_evidence_rank", "confidence", "n_supporting_compounds"],
                ascending=[True, False, False],
            ).head(TOP_TARGETS)

            for _, row in gs_rank.iterrows():
                gene = str(row["gene_symbol"])
                evidence = str(row["evidence"])
                conf = float(row["confidence"])
                n_sup = int(row["n_supporting_compounds"])
                cg_tier = gene_tier_lookup.get((species_id, gene))
                top_targets.append(
                    {
                        "gene_symbol": gene,
                        "evidence": evidence,
                        "confidence": round(conf, 4),
                        "confidence_tier": tier_from_gene_row(evidence, conf, cg_tier),
                        "n_supporting_compounds": n_sup,
                        "moa": moa_lookup.get((species_id, gene)),
                    }
                )

        # --- pathways ---
        enr_sub = enr_by_ing.get(species_id)
        sig_count = 0
        top_ranked: list[dict[str, Any]] = []
        top_significant: list[dict[str, Any]] = []
        is_broadly_active = False

        if enr_sub is not None and len(enr_sub):
            ranked = enr_sub.sort_values("weighted_fold_enrichment", ascending=False)
            sig_rows = enr_sub[enr_sub["q_value"] < Q_SIG].sort_values(
                "weighted_fold_enrichment", ascending=False
            )
            sig_count = len(sig_rows)

            for _, row in ranked.head(TOP_PATHWAYS_RANKED).iterrows():
                top_ranked.append(
                    {
                        "pathway": pathway_label(row["pathway_id"]),
                        "weighted_fold": round(float(row["weighted_fold_enrichment"]), 4),
                        "q_value": round(float(row["q_value"]), 6),
                        "driving_genes": driving_gene_symbols(row["driving_genes_json"]),
                        "evidence_split": evidence_split(row),
                    }
                )

            for _, row in sig_rows.head(TOP_PATHWAYS_SIG).iterrows():
                top_significant.append(
                    {
                        "pathway": pathway_label(row["pathway_id"]),
                        "weighted_fold": round(float(row["weighted_fold_enrichment"]), 4),
                        "q_value": round(float(row["q_value"]), 6),
                        "driving_genes": driving_gene_symbols(row["driving_genes_json"]),
                        "evidence_split": evidence_split(row),
                    }
                )

            is_broadly_active = sig_count == 0 and len(top_ranked) > 0 and n_genes >= 100

        # --- categories ---
        cat_sub = cat_by_ing.get(species_id)
        cat_available = cat_sub is not None and len(cat_sub) > 0
        top_categories: list[dict[str, Any]] = []
        if cat_available:
            cat_rank = cat_sub.sort_values("aggregated_enrichment", ascending=False).head(TOP_CATEGORIES)
            for _, row in cat_rank.iterrows():
                top_categories.append(
                    {
                        "category": str(row["category_name"]),
                        "aggregated_enrichment": round(float(row["aggregated_enrichment"]), 4),
                    }
                )

        # --- tissues ---
        tissue_sub = tissue_by_ing.get(species_id)
        top_tissues: list[dict[str, Any]] = []
        if tissue_sub is not None and len(tissue_sub):
            t_rank = tissue_sub.sort_values("normalized_score", ascending=False).head(TOP_TISSUES)
            for _, row in t_rank.iterrows():
                top_tissues.append(
                    {
                        "tissue": str(row["tissue"]),
                        "normalized_score": round(float(row["normalized_score"]), 6),
                    }
                )

        measured_fraction = float(measured_count / n_genes) if n_genes > 0 else 0.0
        predicted_pct = 1.0 - measured_fraction
        provenance_summary = (
            f"Mechanism is {measured_fraction:.0%} measured, {predicted_pct:.0%} "
            "structurally predicted (k-NN inference, validated hit@10 ~0.89)"
        )

        profile = {
            "ingredient": {
                "species_id": species_id,
                "canonical_name": str(node["canonical_name"]),
                "latin_name": None if pd.isna(node["latin_name"]) else str(node["latin_name"]),
                "preparation_labels": parse_preparation_labels(node["preparation_labels"]),
                "foodb_id": str(int(node["foodb_id"])) if pd.notna(node["foodb_id"]) else None,
                "mechanism_coverage": cov,
            },
            "compounds": {
                "count": n_compounds,
                "top": top_compounds,
            },
            "targets": {
                "count": n_genes,
                "measured_count": measured_count,
                "predicted_count": predicted_count,
                "top": top_targets,
            },
            "pathways": {
                "significant_count": sig_count,
                "is_broadly_active": is_broadly_active,
                "top_ranked": top_ranked,
                "top_significant": top_significant,
            },
            "categories": {
                "available": cat_available,
                "top": top_categories,
            },
            "tissues": {
                "top": top_tissues,
                "interpretation_note": TISSUE_INTERPRETATION,
            },
            "provenance": {
                "measured_fraction": round(measured_fraction, 4),
                "summary": provenance_summary,
            },
        }
        profiles.append(json_safe(profile))

    build_meta = {
        "n_profiles": len(profiles),
        "n_string_mappings_read": len(_string_map),
        "mechanism_coverage_distribution": dict(coverage_stats),
        "categories_available_count": sum(1 for p in profiles if p["categories"]["available"]),
        "categories_missing_count": sum(1 for p in profiles if not p["categories"]["available"]),
        "broadly_active_count": sum(1 for p in profiles if p["pathways"]["is_broadly_active"]),
        "zero_significant_pathway_count": sum(
            1 for p in profiles if p["pathways"]["significant_count"] == 0
        ),
    }
    return profiles, build_meta


def write_outputs(profiles: list[dict[str, Any]], build_meta: dict[str, Any]) -> None:
    PRODUCT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    profile_by_id = {p["ingredient"]["species_id"]: p for p in profiles}

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for profile in profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")

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
        "version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_profiles": len(index_entries),
        "ranking_methods": RANKING_DOC,
        "profiles": index_entries,
    }
    OUT_INDEX.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    sample_paths: dict[str, str] = {}
    sample_labels = {
        "turmeric": "Turmeric (saturated spice — broadly active, zero FDR-significant pathways)",
        "black_tea": "Black tea (measured-rich)",
        "rapeseed_oil": "Rapeseed oil (thin coverage — 3 compounds, 14 genes)",
    }
    for key, species_id in SAMPLE_SPECIES.items():
        profile = profile_by_id[species_id]
        out_path = SAMPLES / f"ingredient_profile_{key}_v1.json"
        payload = {
            "sample_label": sample_labels[key],
            "species_id": species_id,
            "profile": profile,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        sample_paths[key] = str(out_path.relative_to(ROOT))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "inputs": {
            "species_nodes_v2": {"path": str(SPECIES_NODES.relative_to(ROOT)), "sha256": sha256_file(SPECIES_NODES)},
            "ingredient_string_species_v2": {"path": str(STRING_MAP.relative_to(ROOT)), "sha256": sha256_file(STRING_MAP)},
            "ingredient_compound_canonical_v2": {"path": str(ICC_V2.relative_to(ROOT)), "sha256": sha256_file(ICC_V2)},
            "compound_master_v2": {"path": str(COMPOUND_MASTER.relative_to(ROOT)), "sha256": sha256_file(COMPOUND_MASTER)},
            "ingredient_gene_sets_v3": {"path": str(GENE_SETS_V3.relative_to(ROOT)), "sha256": sha256_file(GENE_SETS_V3)},
            "compound_gene_integrated_v1": {"path": str(INTEGRATED_CG.relative_to(ROOT)), "sha256": sha256_file(INTEGRATED_CG)},
            "enrichment_weighted_v3_calibrated": {"path": str(ENRICHMENT.relative_to(ROOT)), "sha256": sha256_file(ENRICHMENT)},
            "ingredient_category_profiles_v2": {"path": str(CATEGORY_PROFILES.relative_to(ROOT)), "sha256": sha256_file(CATEGORY_PROFILES)},
            "ingredient_tissue_profiles_v2": {"path": str(TISSUE_PROFILES.relative_to(ROOT)), "sha256": sha256_file(TISSUE_PROFILES)},
            "measured_moa_annotation_v1": {"path": str(MOA.relative_to(ROOT)), "sha256": sha256_file(MOA)},
        },
        "outputs": {
            "jsonl": str(OUT_JSONL.relative_to(ROOT)),
            "index": str(OUT_INDEX.relative_to(ROOT)),
            "samples": sample_paths,
        },
        "ranking_methods": RANKING_DOC,
        "schema_notes": {
            "pathway_label": (
                "pathway field uses pathway_id from enrichment table; GO terms are human-readable, "
                "Reactome entries are stable R-HSA-* IDs (full Reactome labels not in core read set)."
            ),
            "confidence_tier": (
                "measured targets → high; predicted targets → max tier from compound_gene_integrated_v1 "
                "via ingredient compounds, else confidence thresholds."
            ),
            "moa": (
                "ChEMBL action_type joined via measured compound→gene edges for ingredient compounds; "
                "null when no measured MoA annotation exists for that gene."
            ),
            "categories_available": (
                "true when fine_recipe category profiles exist for the species; "
                "all 445 mechanism-live species currently have profiles."
            ),
        },
        "coverage": build_meta,
        "field_gaps": [
            "Reactome pathway display names (R-HSA-* IDs only unless pathway_id is a GO term string)",
            "MoA null for most predicted-only targets (690 measured MoA edges total in corpus)",
            "Recipe/cuisine layer not included (parked)",
        ],
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    print("Building ingredient profiles v1 …")
    profiles, build_meta = build_profiles()
    write_outputs(profiles, build_meta)
    print(f"Wrote {len(profiles)} profiles -> {OUT_JSONL}")
    print(f"Index -> {OUT_INDEX}")
    print(f"Report -> {OUT_REPORT}")
    print("Coverage:", build_meta)


if __name__ == "__main__":
    main()
