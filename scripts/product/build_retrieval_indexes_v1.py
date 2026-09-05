#!/usr/bin/env python3
"""
Build inverted retrieval indexes over ingredient_profiles_v1_1.jsonl.

Usage (from repo root):
    python scripts/product/build_retrieval_indexes_v1.py

Outputs (new only):
    data/processed/product/indexes/
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from theme_definitions import (  # noqa: E402
    BODY_REGION_THEMES,
    CUISINE_SEED,
    THEME_RETRIEVAL_EXPANSIONS,
    audit_theme_fragmentation,
    compile_theme_patterns,
    expanded_theme_pathway_ids,
    match_effect_themes,
)

PRODUCT = ROOT / "data/processed/product"
INDEX_DIR = PRODUCT / "indexes"
CANONICAL = ROOT / "data/processed/canonical"
TIER1 = ROOT / "data/processed/tier1"
INTEGRATED = ROOT / "data/processed/integrated"

PROFILES_V1_1 = PRODUCT / "ingredient_profiles_v1_1.jsonl"
PATHWAY_NAMES = PRODUCT / "pathway_display_names_v1_1.json"
TISSUE_PROFILES = TIER1 / "ingredient_tissue_profiles_v2.parquet"
ENRICHMENT = TIER1 / "enrichment_weighted_v3_calibrated.parquet"
GENE_SETS = INTEGRATED / "ingredient_gene_sets_v3.parquet"
MOA = TIER1 / "measured_moa_annotation_v1.parquet"
SPECIES_NODES = CANONICAL / "species_nodes_v2.parquet"
STRING_MAP = CANONICAL / "ingredient_string_species_v2.parquet"
ICC = CANONICAL / "ingredient_compound_canonical_v2.parquet"
RECIPES = CANONICAL / "recipes_expanded_v2.parquet"
RECIPE_ING = CANONICAL / "recipe_ingredients_expanded_v2.parquet"
CATEGORY_PROFILES = TIER1 / "ingredient_category_profiles_v2.parquet"
PATHWAY_MAP = TIER1 / "pathway_category_map_v2.parquet"

Q_SIG = 0.10
TOP_N_TISSUE_INDEX = 100  # per tissue, ranked ingredients
TOP_N_PATHWAY_INDEX = 150  # per pathway
TOP_N_TARGET_INDEX = 100  # per gene

# Effect-theme ranking controls (v1.1 fix for thin-profile artifacts)
THIN_GENE_THRESHOLD = 150
LOW_RICHNESS_GENE_THRESHOLD = 800
THIN_GATE_MULTIPLIER = 0.05
LOW_RICHNESS_GATE_MULTIPLIER = 0.50
PATHWAY_STRENGTH_CAP = 6.0

# Default component weights (before per-theme adaptive redistribution).
DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "strength_specificity_norm": 0.25,
    "breadth_specificity_norm": 0.15,
    "measured_norm": 0.15,
    "significance_norm": 0.10,
    "measured_moa_norm": 0.35,
}
MOA_GENE_SCORE_WITH_ACTION = 1.0
MOA_GENE_SCORE_MEASURED = 0.55
MOA_NORM_CAP = 4.0
DEAD_VAR_THRESHOLD = 1e-5
DEAD_RANGE_THRESHOLD = 0.015
DEAD_NONZERO_FRACTION = 0.05
HIGH_CONFIDENCE = 1.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_profiles() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    profiles: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with PROFILES_V1_1.open(encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            profiles.append(p)
            by_id[p["ingredient"]["species_id"]] = p
    return profiles, by_id


def load_pathway_name_lookup() -> dict[str, str]:
    data = json.loads(PATHWAY_NAMES.read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    for _raw, info in data["pathways"].items():
        lookup[info["pathway"]] = info["pathway_name"]
    return lookup


def ingredient_summary(species_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    ing = profile["ingredient"]
    return {
        "species_id": species_id,
        "canonical_name": ing["canonical_name"],
        "mechanism_coverage": ing["mechanism_coverage"],
        "measured_fraction": profile["provenance"]["measured_fraction"],
        "n_genes": int(profile.get("targets", {}).get("count", 0)),
    }


def build_moa_lookup(
    gene_sets: pd.DataFrame,
    moa_df: pd.DataFrame,
    icc: pd.DataFrame,
) -> dict[tuple[str, str], str]:
    """Map (ingredient_id, gene_symbol) -> ChEMBL action_type when measured."""
    moa_df = moa_df.rename(columns={"compound": "compound_id"})
    moa_df["compound_id"] = moa_df["compound_id"].astype(str).str.upper()
    icc = icc.copy()
    icc["compound_id"] = icc["compound_id"].astype(str).str.upper()
    moa_merged = icc.merge(moa_df, on="compound_id", how="inner")
    moa_lookup: dict[tuple[str, str], str] = {}
    for (ing, gene), sub in moa_merged.groupby(["ingredient_id", "gene_symbol"]):
        acts = sub["action_type"].dropna().astype(str).unique()
        if len(acts):
            moa_lookup[(str(ing), str(gene))] = acts[0]
    return moa_lookup


def build_gene_evidence_lookup(gene_sets: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in gene_sets.iterrows():
        key = (str(row["ingredient_id"]), str(row["gene_symbol"]))
        lookup[key] = {
            "evidence": str(row["evidence"]),
            "confidence": float(row["confidence"]),
            "n_supporting_compounds": int(row["n_supporting_compounds"]),
        }
    return lookup


def parse_driving_genes(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        payload = raw
    genes: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("gene_symbol"):
                genes.append(str(item["gene_symbol"]))
            elif isinstance(item, str):
                genes.append(item)
    return genes


def derive_theme_relevant_genes(
    enr: pd.DataFrame,
    theme_pathway_ids: set[str],
    min_pathway_hits: int = 2,
) -> set[str]:
    """
    Data-derived theme gene set: genes appearing as drivers in >= min_pathway_hits
    theme pathways across the corpus.
    """
    sub = enr[enr["pathway_stable"].isin(theme_pathway_ids)]
    gene_pathway_counts: dict[str, int] = defaultdict(int)
    for _, row in sub.iterrows():
        genes = parse_driving_genes(row.get("driving_genes_json"))
        for gene in genes:
            gene_pathway_counts[gene] += 1
    return {g for g, cnt in gene_pathway_counts.items() if cnt >= min_pathway_hits}


def build_gene_theme_driver_counts(
    enr: pd.DataFrame,
    theme_pathway_ids: set[str],
) -> dict[str, int]:
    """How many ingredients have each gene as a driver in theme pathways."""
    sub = enr[enr["pathway_stable"].isin(theme_pathway_ids)]
    gene_ingredients: dict[str, set[str]] = defaultdict(set)
    for _, row in sub.iterrows():
        sid = str(row["ingredient_id"])
        for gene in parse_driving_genes(row.get("driving_genes_json")):
            gene_ingredients[gene].add(sid)
    return {gene: len(sids) for gene, sids in gene_ingredients.items()}


def gene_theme_specificity(gene: str, gene_driver_counts: dict[str, int], n_ingredients: int) -> float:
    n_ing = max(1, int(gene_driver_counts.get(gene, n_ingredients)))
    return max(0.0, min(1.0, math.log(n_ingredients / n_ing) / math.log(n_ingredients)))


def compute_measured_moa_norm(
    species_id: str,
    ingredient_driver_genes: set[str],
    gene_evidence: dict[tuple[str, str], dict[str, Any]],
    moa_lookup: dict[tuple[str, str], str],
    gene_driver_counts: dict[str, int],
    n_ingredients: int,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Gene-level measured MoA component in [0, 1].
    Scores only genes that drive this ingredient's theme pathway enrichments.
    measured + ChEMBL action_type highest; measured without MoA next; predicted zero.
    Gene theme-specificity downweights corpus-ubiquitous drivers.
    """
    hits: list[dict[str, Any]] = []
    raw_score = 0.0
    for gene in sorted(ingredient_driver_genes):
        info = gene_evidence.get((species_id, gene))
        if not info:
            continue
        evidence = info["evidence"]
        confidence = float(info["confidence"])
        moa = moa_lookup.get((species_id, gene))
        if evidence != "measured":
            continue
        spec = gene_theme_specificity(gene, gene_driver_counts, n_ingredients)
        if moa and str(moa).upper() in {"INHIBITOR", "AGONIST", "ANTAGONIST"}:
            # Pharmacological action is meaningful regardless of pathway ubiquity.
            gene_score = MOA_GENE_SCORE_WITH_ACTION * confidence
        else:
            gene_score = MOA_GENE_SCORE_MEASURED * confidence * (0.35 + 0.65 * spec)
        raw_score += gene_score
        hits.append(
            {
                "gene_symbol": gene,
                "evidence": evidence,
                "confidence": round(confidence, 4),
                "moa": moa,
                "gene_theme_specificity": round(spec, 4),
                "gene_moa_score": round(gene_score, 4),
            }
        )
    moa_norm = min(1.0, raw_score / MOA_NORM_CAP) if raw_score > 0 else 0.0
    hits.sort(key=lambda x: (-x["gene_moa_score"], x["gene_symbol"]))
    return moa_norm, hits


def compute_adaptive_weights(
    component_matrix: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """
    Per-theme weight adaptation from component variance across ingredients.
    Dead components (near-zero variance / range) have weight redistributed to live ones.
    """
    if not component_matrix:
        return dict(DEFAULT_COMPONENT_WEIGHTS), {}

    diagnostics: dict[str, dict[str, Any]] = {}
    live_flags: dict[str, bool] = {}
    variances: dict[str, float] = {}

    for comp, default_w in DEFAULT_COMPONENT_WEIGHTS.items():
        values = [float(row.get(comp, 0.0)) for row in component_matrix]
        if len(values) < 2:
            var = 0.0
        else:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        rng = max(values) - min(values) if values else 0.0
        nonzero_frac = sum(1 for v in values if v > 1e-6) / max(1, len(values))
        is_live = var > DEAD_VAR_THRESHOLD and rng > DEAD_RANGE_THRESHOLD
        if comp == "significance_norm":
            is_live = is_live and nonzero_frac >= DEAD_NONZERO_FRACTION
        if comp in {"strength_specificity_norm", "breadth_specificity_norm"}:
            # Specificity is embedded; treat as dead when range is negligible.
            if rng <= DEAD_RANGE_THRESHOLD:
                is_live = False
        live_flags[comp] = is_live
        variances[comp] = var
        diagnostics[comp] = {
            "default_weight": default_w,
            "variance": round(var, 8),
            "range": round(rng, 6),
            "nonzero_fraction": round(nonzero_frac, 4),
            "is_live": is_live,
        }

    if not any(live_flags.values()):
        # Fallback: keep default weights if everything looks dead.
        return dict(DEFAULT_COMPONENT_WEIGHTS), diagnostics

    dead_weight = sum(
        DEFAULT_COMPONENT_WEIGHTS[c] for c, live in live_flags.items() if not live
    )
    live_components = [c for c, live in live_flags.items() if live]
    live_var_sum = sum(max(variances[c], DEAD_VAR_THRESHOLD) for c in live_components)
    adapted = dict(DEFAULT_COMPONENT_WEIGHTS)
    for comp in DEFAULT_COMPONENT_WEIGHTS:
        if not live_flags[comp]:
            adapted[comp] = 0.0
    for comp in live_components:
        share = max(variances[comp], DEAD_VAR_THRESHOLD) / live_var_sum
        adapted[comp] = DEFAULT_COMPONENT_WEIGHTS[comp] + dead_weight * share
        diagnostics[comp]["adapted_weight"] = round(adapted[comp], 4)
    for comp, live in live_flags.items():
        if not live:
            diagnostics[comp]["adapted_weight"] = 0.0
            diagnostics[comp]["weight_redistributed"] = True
    return adapted, diagnostics


def parse_evidence_split(evidence_split: str, fallback_measured_fraction: float) -> tuple[int, int, float]:
    """
    Parse strings like:
      - '7 measured / 1 predicted'
      - '9 measured'
      - '6 predicted'
    Return (n_measured, n_predicted, measured_fraction).
    """
    txt = str(evidence_split or "").strip().lower()
    measured = 0
    predicted = 0
    if "measured /" in txt and "predicted" in txt:
        parts = txt.split("/")
        try:
            measured = int(parts[0].strip().split()[0])
            predicted = int(parts[1].strip().split()[0])
        except (ValueError, IndexError):
            pass
    elif txt.endswith("measured"):
        try:
            measured = int(txt.split()[0])
        except (ValueError, IndexError):
            pass
    elif txt.endswith("predicted"):
        try:
            predicted = int(txt.split()[0])
        except (ValueError, IndexError):
            pass
    total = measured + predicted
    if total > 0:
        return measured, predicted, measured / total
    return measured, predicted, float(fallback_measured_fraction)


def build_tissue_index(
    tissue_df: pd.DataFrame,
    profiles_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    index: dict[str, list[dict[str, Any]]] = {}
    for tissue, sub in tissue_df.groupby("tissue"):
        ranked = sub.sort_values("normalized_score", ascending=False).head(TOP_N_TISSUE_INDEX)
        entries: list[dict[str, Any]] = []
        for _, row in ranked.iterrows():
            sid = str(row["ingredient_id"])
            prof = profiles_by_id.get(sid)
            if not prof:
                continue
            entries.append(
                {
                    **ingredient_summary(sid, prof),
                    "normalized_score": round(float(row["normalized_score"]), 6),
                    "measured_fraction_of_tissue_score": round(
                        float(row.get("measured_fraction_of_score", 0) or 0), 4
                    ),
                }
            )
        index[str(tissue)] = entries
    return index


def build_pathway_index(
    enrichment: pd.DataFrame,
    pathway_names: dict[str, str],
    profiles_by_id: dict[str, dict[str, Any]],
    resolver,
) -> dict[str, Any]:
    # Normalize pathway ids in enrichment (may be bracket GO strings)
    enrichment = enrichment.copy()
    enrichment["pathway_stable"] = enrichment["pathway_id"].astype(str).map(
        lambda x: resolver.resolve(x)["pathway"]
    )
    enrichment["pathway_name"] = enrichment["pathway_stable"].map(
        lambda x: pathway_names.get(x, x)
    )

    index: dict[str, Any] = {}
    for pid, sub in enrichment.groupby("pathway_stable"):
        ranked = sub.sort_values("weighted_fold_enrichment", ascending=False).head(TOP_N_PATHWAY_INDEX)
        entries: list[dict[str, Any]] = []
        pname = pathway_names.get(pid, str(pid))
        for _, row in ranked.iterrows():
            sid = str(row["ingredient_id"])
            prof = profiles_by_id.get(sid)
            if not prof:
                continue
            q = float(row["q_value"])
            entries.append(
                {
                    **ingredient_summary(sid, prof),
                    "pathway_name": pname,
                    "weighted_fold": round(float(row["weighted_fold_enrichment"]), 4),
                    "q_value": round(q, 6),
                    "is_significant": q < Q_SIG,
                    "evidence_split": (
                        f"{int(row['n_measured_overlap'])} measured / {int(row['n_predicted_overlap'])} predicted"
                        if int(row["n_predicted_overlap"]) > 0
                        else f"{int(row['n_measured_overlap'])} measured"
                        if int(row["n_measured_overlap"]) > 0
                        else f"{int(row['n_predicted_overlap'])} predicted"
                    ),
                }
            )
        index[pid] = {
            "pathway_id": pid,
            "pathway_name": pname,
            "ingredients": entries,
        }
    return index


def build_target_index(
    gene_sets: pd.DataFrame,
    moa_df: pd.DataFrame,
    icc: pd.DataFrame,
    profiles_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    moa_lookup = build_moa_lookup(gene_sets, moa_df, icc)

    index: dict[str, Any] = {}
    for gene, sub in gene_sets.groupby("gene_symbol"):
        ranked = sub.sort_values(
            ["evidence", "confidence", "n_supporting_compounds"],
            ascending=[True, False, False],
        ).head(TOP_N_TARGET_INDEX)
        entries: list[dict[str, Any]] = []
        for _, row in ranked.iterrows():
            sid = str(row["ingredient_id"])
            prof = profiles_by_id.get(sid)
            if not prof:
                continue
            evidence = str(row["evidence"])
            entries.append(
                {
                    **ingredient_summary(sid, prof),
                    "gene_symbol": str(gene),
                    "evidence": evidence,
                    "confidence": round(float(row["confidence"]), 4),
                    "confidence_tier": "high" if evidence == "measured" else "moderate",
                    "n_supporting_compounds": int(row["n_supporting_compounds"]),
                    "n_measured_compounds": int(row["n_measured_compounds"]),
                    "n_predicted_compounds": int(row["n_predicted_compounds"]),
                    "moa": moa_lookup.get((sid, str(gene))),
                }
            )
        index[str(gene)] = {
            "gene_symbol": str(gene),
            "ingredients": entries,
            "n_ingredients_total": int(sub["ingredient_id"].nunique()),
        }
    return index


def build_ingredient_lookup(
    species: pd.DataFrame,
    string_map: pd.DataFrame,
    profiles_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_species: dict[str, Any] = {}
    for _, row in species.iterrows():
        sid = str(row["species_node_id"])
        if sid not in profiles_by_id:
            continue
        prep = row.get("preparation_labels")
        prep_list: list[str] = []
        if prep is not None and not (isinstance(prep, float) and math.isnan(prep)):
            if isinstance(prep, (list, tuple)):
                prep_list = [str(x) for x in prep if str(x).strip()]
        by_species[sid] = {
            "species_id": sid,
            "canonical_name": str(row["canonical_name"]),
            "latin_name": None if pd.isna(row.get("latin_name")) else str(row["latin_name"]),
            "preparation_labels": prep_list,
            "foodb_id": str(int(row["foodb_id"])) if pd.notna(row.get("foodb_id")) else None,
        }

    aliases: dict[str, str] = {}
    alias_meta: dict[str, list[str]] = defaultdict(list)

    def add_alias(alias: str, sid: str) -> None:
        key = alias.strip().lower()
        if not key or sid not in profiles_by_id:
            return
        if key not in aliases:
            aliases[key] = sid
        alias_meta[sid].append(key)

    for sid, info in by_species.items():
        add_alias(info["canonical_name"], sid)
        add_alias(sid, sid)
        for pl in info["preparation_labels"]:
            add_alias(pl, sid)

    for _, row in string_map.iterrows():
        sid = str(row["species_node"])
        if sid not in profiles_by_id:
            continue
        add_alias(str(row["ingredient_string"]), sid)
        if pd.notna(row.get("preparation_label")):
            add_alias(str(row["preparation_label"]), sid)
        if pd.notna(row.get("canonical_name")):
            add_alias(str(row["canonical_name"]), sid)

    return {
        "by_species_id": by_species,
        "aliases": aliases,
        "alias_lists": {k: sorted(set(v)) for k, v in alias_meta.items()},
        "n_aliases": len(aliases),
        "n_species": len(by_species),
    }


def build_effect_themes(
    pathway_index: dict[str, Any],
    enrichment: pd.DataFrame,
    resolver,
    profiles_by_id: dict[str, dict[str, Any]],
    category_profiles: pd.DataFrame,
    pathway_names: dict[str, str],
    compiled_rules: list[dict[str, Any]],
    gene_evidence: dict[tuple[str, str], dict[str, Any]],
    moa_lookup: dict[tuple[str, str], str],
) -> dict[str, Any]:
    theme_pathways: dict[str, set[str]] = defaultdict(set)
    theme_categories: dict[str, set[str]] = defaultdict(set)

    for pid, block in pathway_index.items():
        pname = block.get("pathway_name", pathway_names.get(pid, pid))
        for theme_id in match_effect_themes(pname, pathway_id=pid, compiled_rules=compiled_rules):
            theme_pathways[theme_id].add(pid)

    cat_fine = category_profiles[category_profiles["category_level"] == "fine_recipe"]
    for _, row in cat_fine.iterrows():
        cname = str(row["category_name"])
        cid = str(row["category_id"])
        for theme_id in match_effect_themes(cname, category_id=cid, compiled_rules=compiled_rules):
            theme_categories[theme_id].add(cid)

    # Merge pathway assignments from category keyword hits via pathway_map
    if PATHWAY_MAP.exists():
        pmap = pd.read_parquet(PATHWAY_MAP, columns=["pathway_id", "fine_category_id", "fine_category_name"])
        for _, row in pmap.iterrows():
            raw_pid = str(row["pathway_id"])
            # pathway_id in map may be bracket GO
            from pathway_display_names import PathwayNameResolver

            stable = PathwayNameResolver().resolve(raw_pid)["pathway"]
            cid = str(row["fine_category_id"])
            cname = str(row["fine_category_name"])
            for theme_id in match_effect_themes(cname, category_id=cid, compiled_rules=compiled_rules):
                theme_pathways[theme_id].add(stable)

    themes_out: dict[str, Any] = {}
    for rule in compiled_rules:
        tid = rule["theme_id"]
        native_pids = sorted(theme_pathways.get(tid, set()))
        retrieval_pids = sorted(expanded_theme_pathway_ids(tid, theme_pathways))
        cids = sorted(theme_categories.get(tid, set()))
        expansion = THEME_RETRIEVAL_EXPANSIONS.get(tid, {})
        themes_out[tid] = {
            "theme_id": tid,
            "label": rule["label"],
            "pathway_ids": native_pids,
            "pathway_names": [pathway_names.get(p, p) for p in native_pids[:20]],
            "n_pathways": len(native_pids),
            "retrieval_pathway_ids": retrieval_pids,
            "n_retrieval_pathways": len(retrieval_pids),
            "include_sub_themes": expansion.get("include_sub_themes", []),
            "retrieval_expansion_note": expansion.get("description"),
            "category_ids": cids,
            "n_categories": len(cids),
            "grouping_method": (
                "Keyword rules on pathway_name + category_name; "
                "explicit pathway_id/category_id lists in theme_definitions.py; "
                "Reactome fine categories via pathway_category_map_v2; "
                "optional retrieval sub-theme union via THEME_RETRIEVAL_EXPANSIONS"
            ),
        }

    theme_fragmentation_audit = audit_theme_fragmentation(
        theme_pathways, pathway_names, compiled_rules
    )

    # Full enrichment coverage for theme scoring (avoid truncation artifacts from top-N pathway index).
    enr = enrichment.copy()
    enr["pathway_stable"] = enr["pathway_id"].astype(str).map(lambda x: resolver.resolve(x)["pathway"])
    enr["pathway_name"] = enr["pathway_stable"].map(lambda x: pathway_names.get(x, x))
    n_ingredients_total = max(1, int(enr["ingredient_id"].nunique()))
    pathway_ingredient_counts = (
        enr.groupby("pathway_stable")["ingredient_id"].nunique().to_dict()
    )

    def pathway_specificity(pathway_id: str) -> float:
        """
        Specificity in [0,1]:
          0 => pathway appears in almost all ingredients (generic)
          1 => pathway appears in very few ingredients (specific)
        """
        n_ing = max(1, int(pathway_ingredient_counts.get(pathway_id, n_ingredients_total)))
        return max(0.0, min(1.0, math.log(n_ingredients_total / n_ing) / math.log(n_ingredients_total)))

    # effect_theme -> ingredient pathway hits (aggregate from full enrichment rows)
    theme_hits: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    theme_relevant_genes: dict[str, set[str]] = {}
    theme_gene_driver_counts: dict[str, dict[str, int]] = {}
    for tid, meta in themes_out.items():
        pids = set(meta["retrieval_pathway_ids"])
        if not pids:
            continue
        theme_relevant_genes[tid] = derive_theme_relevant_genes(enr, pids)
        theme_gene_driver_counts[tid] = build_gene_theme_driver_counts(enr, pids)
        themes_out[tid]["theme_relevant_genes"] = sorted(theme_relevant_genes[tid])[:50]
        themes_out[tid]["n_theme_relevant_genes"] = len(theme_relevant_genes[tid])
        sub = enr[enr["pathway_stable"].isin(pids)]
        for _, row in sub.iterrows():
            sid = str(row["ingredient_id"])
            prof = profiles_by_id.get(sid)
            if not prof:
                continue
            q = float(row["q_value"])
            n_measured = int(row.get("n_measured_overlap", 0) or 0)
            n_predicted = int(row.get("n_predicted_overlap", 0) or 0)
            if n_predicted > 0:
                evidence_split = f"{n_measured} measured / {n_predicted} predicted"
            elif n_measured > 0:
                evidence_split = f"{n_measured} measured"
            else:
                evidence_split = f"{n_predicted} predicted"
            theme_hits[tid][sid].append(
                {
                    **ingredient_summary(sid, prof),
                    "pathway_id": str(row["pathway_stable"]),
                    "pathway_name": str(row["pathway_name"]),
                    "weighted_fold": round(float(row["weighted_fold_enrichment"]), 4),
                    "q_value": round(q, 6),
                    "is_significant": q < Q_SIG,
                    "evidence_split": evidence_split,
                    "driving_genes": parse_driving_genes(row.get("driving_genes_json")),
                }
            )

    effect_index: dict[str, Any] = {}
    theme_adaptive_weights: dict[str, Any] = {}
    for tid, meta in themes_out.items():
        partial_rows: list[dict[str, Any]] = []
        n_theme_pathways = max(1, len(meta["retrieval_pathway_ids"]))
        breadth_norm_denom = math.log1p(n_theme_pathways)
        theme_genes = theme_relevant_genes.get(tid, set())
        gene_driver_counts = theme_gene_driver_counts.get(tid, {})

        for sid, hits in theme_hits[tid].items():
            if not hits:
                continue
            base = dict(hits[0])
            n_genes = int(base.get("n_genes", 0))
            mechanism_coverage = str(base.get("mechanism_coverage", ""))

            if mechanism_coverage == "thin" or n_genes < THIN_GENE_THRESHOLD:
                richness_gate = THIN_GATE_MULTIPLIER
                richness_gate_reason = "thin_or_low_gene_count"
            elif n_genes < LOW_RICHNESS_GENE_THRESHOLD:
                richness_gate = LOW_RICHNESS_GATE_MULTIPLIER
                richness_gate_reason = "moderate_gene_count"
            else:
                richness_gate = 1.0
                richness_gate_reason = "rich"

            breadth_count = len(hits)
            breadth_norm = math.log1p(breadth_count) / breadth_norm_denom if breadth_norm_denom > 0 else 0.0

            strength_vals: list[float] = []
            specificity_vals: list[float] = []
            measured_fracs: list[float] = []
            sig_count = 0
            measured_total = 0
            support_total = 0

            best_hit = max(hits, key=lambda x: float(x["weighted_fold"]))
            for h in hits:
                wf = float(h["weighted_fold"])
                strength_vals.append(min(wf, PATHWAY_STRENGTH_CAP))
                spec = pathway_specificity(str(h["pathway_id"]))
                specificity_vals.append(spec)
                if bool(h.get("is_significant", False)):
                    sig_count += 1
                m, p, frac = parse_evidence_split(
                    str(h.get("evidence_split", "")),
                    float(h.get("measured_fraction", 0.0)),
                )
                measured_total += m
                support_total += (m + p)
                measured_fracs.append(frac)

            strength_mean = sum(strength_vals) / max(1, len(strength_vals))
            strength_norm = min(1.0, strength_mean / PATHWAY_STRENGTH_CAP)
            specificity_mean = sum(specificity_vals) / max(1, len(specificity_vals))
            strength_specificity_norm = strength_norm * (0.4 + 0.6 * specificity_mean)
            breadth_specificity_norm = min(1.0, breadth_norm * (0.5 + 0.5 * specificity_mean))
            significance_norm = sig_count / max(1, breadth_count)
            if support_total > 0:
                measured_norm = measured_total / support_total
            else:
                measured_norm = sum(measured_fracs) / max(1, len(measured_fracs))

            measured_moa_norm, theme_moa_hits = compute_measured_moa_norm(
                sid,
                {g for h in hits for g in h.get("driving_genes", [])},
                gene_evidence,
                moa_lookup,
                gene_driver_counts,
                n_ingredients_total,
            )

            partial_rows.append(
                {
                    "sid": sid,
                    "base": base,
                    "hits": hits,
                    "best_hit": best_hit,
                    "richness_gate": richness_gate,
                    "richness_gate_reason": richness_gate_reason,
                    "breadth_count": breadth_count,
                    "sig_count": sig_count,
                    "components": {
                        "strength_norm": strength_norm,
                        "specificity_mean": specificity_mean,
                        "strength_specificity_norm": strength_specificity_norm,
                        "breadth_norm": breadth_norm,
                        "breadth_specificity_norm": breadth_specificity_norm,
                        "measured_norm": measured_norm,
                        "significance_norm": significance_norm,
                        "measured_moa_norm": measured_moa_norm,
                    },
                    "theme_moa_hits": theme_moa_hits,
                }
            )

        adapted_weights, weight_diagnostics = compute_adaptive_weights(
            [row["components"] for row in partial_rows]
        )
        theme_adaptive_weights[tid] = {
            "default_weights": DEFAULT_COMPONENT_WEIGHTS,
            "adapted_weights": {k: round(v, 4) for k, v in adapted_weights.items()},
            "component_diagnostics": weight_diagnostics,
        }

        ingredient_rows: list[dict[str, Any]] = []
        for row in partial_rows:
            comp = row["components"]
            base_score = sum(adapted_weights[k] * comp[k] for k in DEFAULT_COMPONENT_WEIGHTS)
            theme_relevance_score = 100.0 * row["richness_gate"] * base_score
            best_hit = row["best_hit"]
            moa_with_action = [
                h for h in row["theme_moa_hits"]
                if h.get("moa") in {"INHIBITOR", "AGONIST", "ANTAGONIST"}
            ]
            moa_with_action.sort(key=lambda x: (-x["gene_moa_score"], x["gene_symbol"]))
            ingredient_rows.append(
                {
                    **row["base"],
                    "best_pathway_id": best_hit["pathway_id"],
                    "best_pathway_name": best_hit["pathway_name"],
                    "best_weighted_fold": float(best_hit["weighted_fold"]),
                    "theme_relevance_score": round(theme_relevance_score, 4),
                    "theme_pathways_engaged": row["breadth_count"],
                    "theme_significant_pathways": row["sig_count"],
                    "theme_measured_evidence_fraction": round(comp["measured_norm"], 4),
                    "theme_measured_moa_fraction": round(comp["measured_moa_norm"], 4),
                    "theme_measured_moa_hits": row["theme_moa_hits"][:8],
                    "theme_measured_moa_with_action": moa_with_action[:5],
                    "n_theme_measured_moa_with_action": len(moa_with_action),
                    "evidence_basis": {
                        "pathway_breadth": row["breadth_count"],
                        "pathway_significant": row["sig_count"],
                        "best_pathway": best_hit["pathway_name"],
                        "measured_moa_genes": [
                            f"{h['gene_symbol']} ({h['moa']})" if h.get("moa") else h["gene_symbol"]
                            for h in moa_with_action[:5]
                        ],
                    },
                    "richness_gate": row["richness_gate"],
                    "richness_gate_reason": row["richness_gate_reason"],
                    "ranking_components": {
                        "strength_norm": round(comp["strength_norm"], 4),
                        "specificity_mean": round(comp["specificity_mean"], 4),
                        "strength_specificity_norm": round(comp["strength_specificity_norm"], 4),
                        "breadth_norm": round(comp["breadth_norm"], 4),
                        "breadth_specificity_norm": round(comp["breadth_specificity_norm"], 4),
                        "measured_norm": round(comp["measured_norm"], 4),
                        "significance_norm": round(comp["significance_norm"], 4),
                        "measured_moa_norm": round(comp["measured_moa_norm"], 4),
                    },
                }
            )

        ranked = sorted(
            ingredient_rows,
            key=lambda x: (
                -x["theme_relevance_score"],
                -x.get("n_theme_measured_moa_with_action", 0),
                -x["theme_measured_moa_fraction"],
                -x["theme_significant_pathways"],
                -x["theme_measured_evidence_fraction"],
                -x["best_weighted_fold"],
            ),
        )[:100]
        weight_formula = " + ".join(
            f"{adapted_weights[k]:.2f}*{k}" for k in DEFAULT_COMPONENT_WEIGHTS
        )
        effect_index[tid] = {
            **meta,
            "ranking_formula": (
                f"theme_relevance_score = 100 * richness_gate * ({weight_formula}); "
                f"measured_moa_norm = min(1, sum(gene_moa_scores)/{MOA_NORM_CAP}) "
                f"on ingredient theme driver genes; "
                f"gene_moa_score=(1.0 or 0.55)*conf*(0.35+0.65*gene_theme_specificity)"
            ),
            "ranking_thresholds": {
                "thin_gene_threshold": THIN_GENE_THRESHOLD,
                "low_richness_gene_threshold": LOW_RICHNESS_GENE_THRESHOLD,
                "thin_gate_multiplier": THIN_GATE_MULTIPLIER,
                "low_richness_gate_multiplier": LOW_RICHNESS_GATE_MULTIPLIER,
                "pathway_strength_cap": PATHWAY_STRENGTH_CAP,
                "moa_norm_cap": MOA_NORM_CAP,
                "default_component_weights": DEFAULT_COMPONENT_WEIGHTS,
            },
            "adaptive_weights": theme_adaptive_weights[tid],
            "ingredients": ranked,
        }

    return {
        "themes": themes_out,
        "effect_to_ingredients": effect_index,
        "n_themes": len(themes_out),
        "theme_fragmentation_audit": theme_fragmentation_audit,
        "theme_adaptive_weight_profiles": theme_adaptive_weights,
    }


def build_body_region_themes(
    tissue_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    regions: dict[str, Any] = {}
    region_ingredients: dict[str, Any] = {}

    for region_id, spec in BODY_REGION_THEMES.items():
        tissues = spec["tissues"]
        present = [t for t in tissues if t in tissue_index]
        agg: dict[str, dict[str, Any]] = {}
        for tissue in present:
            for entry in tissue_index[tissue]:
                sid = entry["species_id"]
                prev = agg.get(sid)
                score = entry["normalized_score"]
                if prev is None or score > prev["best_normalized_score"]:
                    agg[sid] = {
                        **entry,
                        "best_normalized_score": score,
                        "best_tissue": tissue,
                    }
        ranked = sorted(agg.values(), key=lambda x: -x["best_normalized_score"])[:100]
        regions[region_id] = {
            "region_id": region_id,
            "label": spec["label"],
            "description": spec["description"],
            "tissues": tissues,
            "tissues_indexed": present,
            "n_tissues": len(present),
        }
        region_ingredients[region_id] = {
            **regions[region_id],
            "ingredients": ranked,
        }

    return {
        "body_regions": regions,
        "region_to_ingredients": region_ingredients,
    }


def resolve_string_to_species(string_map: pd.DataFrame, query: str) -> str | None:
    q = query.strip().lower()
    hits = string_map[string_map["ingredient_string"].str.lower() == q]
    if len(hits):
        return str(hits.iloc[0]["species_node"])
    hits = string_map[string_map["canonical_name"].str.lower() == q]
    if len(hits):
        return str(hits.iloc[0]["species_node"])
    return None


def build_cuisine_layer(
    string_map: pd.DataFrame,
    profiles_by_id: dict[str, dict[str, Any]],
    effect_themes: dict[str, Any],
    recipes: pd.DataFrame,
    recipe_ing: pd.DataFrame,
) -> dict[str, Any]:
    effect_index = effect_themes["effect_to_ingredients"]
    theme_ids = list(effect_index.keys())

    def themes_for_species(sid: str) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for tid in theme_ids:
            for entry in effect_index[tid].get("ingredients", []):
                if entry["species_id"] == sid:
                    hits.append(
                        {
                            "theme_id": tid,
                            "label": effect_index[tid]["label"],
                            "theme_relevance_score": entry.get("theme_relevance_score", 0.0),
                            "best_weighted_fold": entry["best_weighted_fold"],
                            "is_significant": entry.get("is_significant", False),
                        }
                    )
                    break
        return sorted(
            hits,
            key=lambda x: (-x.get("theme_relevance_score", 0.0), -x["best_weighted_fold"]),
        )[:8]

    cuisines: dict[str, Any] = {}
    label_status = {
        "recipe_source_field": "dataset origin only (recipenlg, foodcom, epicurious, indian_food)",
        "cuisine_labels_sparse": True,
        "approach": "curated ingredient seed + indian_food recipe proxy (167 recipes)",
    }

    # Recipe-level species presence for IDF
    recipe_ing = recipe_ing.copy()
    recipe_ing["ingredient_raw"] = recipe_ing["ingredient_raw"].astype(str).str.strip().str.lower()
    sm = string_map.copy()
    sm["ingredient_string"] = sm["ingredient_string"].astype(str).str.lower()
    merged = recipe_ing.merge(
        sm[["ingredient_string", "species_node"]],
        left_on="ingredient_raw",
        right_on="ingredient_string",
        how="inner",
    )
    n_recipes_total = merged["recipe_id"].nunique()
    species_df = merged.groupby("species_node")["recipe_id"].nunique().rename("recipe_freq")

    for cuisine_id, seed in CUISINE_SEED.items():
        resolved: list[dict[str, Any]] = []
        unresolved_strings: list[str] = []
        for s in seed["ingredient_strings"]:
            sid = resolve_string_to_species(string_map, s)
            if sid and sid in profiles_by_id:
                prof = profiles_by_id[sid]
                freq = int(species_df.get(sid, 0))
                idf = math.log(n_recipes_total / max(1, freq)) if freq else 0.0
                resolved.append(
                    {
                        "ingredient_string": s,
                        "species_id": sid,
                        "canonical_name": prof["ingredient"]["canonical_name"],
                        "distinctiveness_score": round(idf, 4),
                        "recipe_frequency": freq,
                        "measured_fraction": prof["provenance"]["measured_fraction"],
                        "mechanistic_themes": themes_for_species(sid),
                    }
                )
            else:
                unresolved_strings.append(s)

        # Recipe-proxy distinctive ingredients for indian
        recipe_proxy_hits: list[dict[str, Any]] = []
        proxy_source = seed.get("recipe_source_proxy")
        if proxy_source:
            proxy_recipes = set(recipes[recipes["source"] == proxy_source]["recipe_id"])
            sub = merged[merged["recipe_id"].isin(proxy_recipes)]
            n_proxy = sub["recipe_id"].nunique()
            if n_proxy > 0:
                proxy_freq = sub.groupby("species_node")["recipe_id"].nunique()
                for sid, cnt in proxy_freq.items():
                    sid = str(sid)
                    if sid not in profiles_by_id:
                        continue
                    global_cnt = int(species_df.get(sid, 0))
                    if global_cnt == 0:
                        continue
                    ratio = (cnt / n_proxy) / (global_cnt / n_recipes_total)
                    if ratio < 1.5:
                        continue
                    prof = profiles_by_id[sid]
                    recipe_proxy_hits.append(
                        {
                            "species_id": sid,
                            "canonical_name": prof["ingredient"]["canonical_name"],
                            "proxy_enrichment_ratio": round(ratio, 3),
                            "proxy_recipe_frequency": int(cnt),
                            "measured_fraction": prof["provenance"]["measured_fraction"],
                            "mechanistic_themes": themes_for_species(sid),
                        }
                    )
                recipe_proxy_hits = sorted(
                    recipe_proxy_hits, key=lambda x: -x["proxy_enrichment_ratio"]
                )[:15]

        # Aggregate mechanistic themes across characteristic ingredients
        theme_weights: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"weight": 0.0, "contributing_ingredients": []}
        )
        for ing in resolved:
            for th in ing["mechanistic_themes"]:
                tid = th["theme_id"]
                theme_weights[tid]["weight"] += th["best_weighted_fold"]
                theme_weights[tid]["contributing_ingredients"].append(ing["canonical_name"])
                theme_weights[tid]["label"] = th["label"]

        collective_themes = sorted(
            [
                {
                    "theme_id": tid,
                    "label": meta["label"],
                    "aggregate_weight": round(meta["weight"], 3),
                    "contributing_ingredients": sorted(set(meta["contributing_ingredients"]))[:10],
                }
                for tid, meta in theme_weights.items()
            ],
            key=lambda x: -x["aggregate_weight"],
        )[:10]

        cuisines[cuisine_id] = {
            "cuisine_id": cuisine_id,
            "label": seed["label"],
            "label_source": (
                f"curated_seed"
                + (f"+recipe_proxy:{proxy_source}" if proxy_source else "")
            ),
            "characteristic_ingredients": sorted(
                resolved, key=lambda x: -x["distinctiveness_score"]
            ),
            "recipe_proxy_distinctive": recipe_proxy_hits,
            "collective_mechanistic_themes": collective_themes,
            "unresolved_seed_strings": unresolved_strings,
        }

    return {
        "label_status": label_status,
        "cuisines": cuisines,
        "n_cuisines": len(cuisines),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    print("Loading profiles v1.1 ...")
    profiles, profiles_by_id = load_profiles()
    pathway_names = load_pathway_name_lookup()

    from pathway_display_names import PathwayNameResolver

    resolver = PathwayNameResolver()

    print("Building tissue index ...")
    tissue_df = pd.read_parquet(TISSUE_PROFILES)
    tissue_index = build_tissue_index(tissue_df, profiles_by_id)

    print("Building pathway index ...")
    enrichment = pd.read_parquet(ENRICHMENT)
    pathway_index = build_pathway_index(enrichment, pathway_names, profiles_by_id, resolver)

    print("Building target index ...")
    gene_sets = pd.read_parquet(GENE_SETS)
    moa_df = pd.read_parquet(MOA)
    icc = pd.read_parquet(ICC, columns=["ingredient_id", "compound_id"])
    target_index = build_target_index(gene_sets, moa_df, icc, profiles_by_id)

    print("Building ingredient lookup ...")
    species = pd.read_parquet(SPECIES_NODES)
    string_map = pd.read_parquet(STRING_MAP)
    ingredient_lookup = build_ingredient_lookup(species, string_map, profiles_by_id)

    print("Building effect themes ...")
    compiled_rules = compile_theme_patterns()
    cat_profiles = pd.read_parquet(CATEGORY_PROFILES)
    gene_evidence = build_gene_evidence_lookup(gene_sets)
    moa_lookup = build_moa_lookup(gene_sets, moa_df, icc)
    effect_themes = build_effect_themes(
        pathway_index,
        enrichment,
        resolver,
        profiles_by_id,
        cat_profiles,
        pathway_names,
        compiled_rules,
        gene_evidence,
        moa_lookup,
    )

    print("Building body-region themes ...")
    body_regions = build_body_region_themes(tissue_index)

    print("Building cuisine layer ...")
    recipes = pd.read_parquet(RECIPES, columns=["recipe_id", "source"])
    recipe_ing = pd.read_parquet(RECIPE_ING, columns=["recipe_id", "ingredient_raw"])
    cuisine_layer = build_cuisine_layer(
        string_map, profiles_by_id, effect_themes, recipes, recipe_ing
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    write_json(INDEX_DIR / "tissue_to_ingredients.json", tissue_index)
    write_json(INDEX_DIR / "pathway_to_ingredients.json", pathway_index)
    write_json(INDEX_DIR / "target_to_ingredients.json", target_index)
    write_json(INDEX_DIR / "ingredient_lookup.json", ingredient_lookup)
    write_json(INDEX_DIR / "effect_themes.json", effect_themes)
    write_json(INDEX_DIR / "body_region_themes.json", body_regions)
    write_json(INDEX_DIR / "cuisine_distinctive_contributors.json", cuisine_layer)

    # Profile lookup reads ingredient_profiles_v1_1.jsonl directly (no duplicate store).

    # Run sample retrievals via retrieval_api
    spec = importlib.util.spec_from_file_location("retrieval_api", SCRIPT_DIR / "retrieval_api.py")
    api_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(api_mod)
    retriever = api_mod.IngredientRetrieval(index_dir=INDEX_DIR)

    samples = {
        "liver_tissue": retriever.ingredients_by_body_region("liver", top_n=10),
        "inflammation_effect": retriever.ingredients_by_effect("inflammation_immune", top_n=10),
        "cnr1_target": retriever.ingredients_by_target("CNR1", top_n=10),
    }

    manifest = {
        "version": "v1.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "indexes": {
            "tissue_to_ingredients": {
                "path": "tissue_to_ingredients.json",
                "n_tissues": len(tissue_index),
                "entries": sum(len(v) for v in tissue_index.values()),
            },
            "pathway_to_ingredients": {
                "path": "pathway_to_ingredients.json",
                "n_pathways": len(pathway_index),
            },
            "target_to_ingredients": {
                "path": "target_to_ingredients.json",
                "n_genes": len(target_index),
            },
            "ingredient_lookup": {
                "path": "ingredient_lookup.json",
                "n_species": ingredient_lookup["n_species"],
                "n_aliases": ingredient_lookup["n_aliases"],
            },
            "effect_themes": {
                "path": "effect_themes.json",
                "n_themes": effect_themes["n_themes"],
            },
            "body_region_themes": {
                "path": "body_region_themes.json",
                "n_regions": len(body_regions["body_regions"]),
            },
            "cuisine_distinctive_contributors": {
                "path": "cuisine_distinctive_contributors.json",
                "n_cuisines": cuisine_layer["n_cuisines"],
            },
            "profiles_source": str(PROFILES_V1_1.relative_to(ROOT)),
        },
        "pathway_name_resolution_in_index": {
            "n_pathways": len(pathway_index),
            "all_have_pathway_name": all("pathway_name" in v for v in pathway_index.values()),
        },
        "cuisine_label_status": cuisine_layer["label_status"],
        "sample_retrievals": samples,
        "inputs": {
            "ingredient_profiles_v1_1": str(PROFILES_V1_1.relative_to(ROOT)),
        },
    }
    write_json(INDEX_DIR / "retrieval_manifest.json", manifest)
    write_json(INDEX_DIR / "sample_retrievals.json", samples)

    # Access patterns doc
    doc = """# Retrieval Index Access Patterns (Option A)

Lightweight flat JSON indexes over `ingredient_profiles_v1_1.jsonl`.

## Python API (`scripts/product/retrieval_api.py`)

```python
from scripts.product.retrieval_api import IngredientRetrieval

r = IngredientRetrieval()

# Body region (aggregates GTEx tissues)
r.ingredients_by_body_region("liver", top_n=20)
r.ingredients_by_body_region("gut", top_n=20)

# Single tissue
r.ingredients_by_tissue("Liver", top_n=20)

# Effect / system theme (natural-language bridge)
r.ingredients_by_effect("inflammation_immune", top_n=20)
r.ingredients_by_effect("xenobiotic_detox", top_n=20)

# Receptor / target gene
r.ingredients_by_target("CNR1", top_n=20)
r.ingredients_by_target("PTGS2", top_n=20)

# Direct ingredient profile
r.profile("turmeric")
r.profile("SP_000052")

# Cuisine distinctive contributors
r.cuisine_themes("indian")
```

## Index files (`data/processed/product/indexes/`)

| File | Access pattern |
|------|----------------|
| `tissue_to_ingredients.json` | tissue → ranked ingredients |
| `pathway_to_ingredients.json` | pathway_id → ranked ingredients + significance |
| `target_to_ingredients.json` | gene_symbol → ranked ingredients + evidence |
| `ingredient_lookup.json` | name/alias → species_id |
| `effect_themes.json` | theme → pathway/category memberships + ranked ingredients |
| `body_region_themes.json` | body region → tissues + ranked ingredients |
| `cuisine_distinctive_contributors.json` | cuisine → characteristic ingredients + themes |
| `../ingredient_profiles_v1_1.jsonl` | species_id → full v1.1 profile (via API) |

All result entries preserve `measured_fraction`, `evidence`, `confidence`, `moa`, and `is_significant` where applicable.
"""
    (INDEX_DIR / "ACCESS_PATTERNS.md").write_text(doc, encoding="utf-8")

    print("Index build complete.")
    print(json.dumps(manifest["indexes"], indent=2))
    print("Sample retrievals written to sample_retrievals.json")


if __name__ == "__main__":
    main()
