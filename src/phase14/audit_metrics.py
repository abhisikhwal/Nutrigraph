"""
Phase14: Coverage and evidence audit — metrics and top_missing_evidence for quality-grade outputs.
Writes reports/phase14_coverage_audit.json and reports/top_missing_evidence.csv.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from .id_normalization import to_gene_id, to_ingredient_id

logger = logging.getLogger(__name__)


def _ts() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _ingredient_sets_from_tables_and_edges(
    ingredient_compound: pd.DataFrame,
    compound_gene: pd.DataFrame,
    mediation_edges: pd.DataFrame,
) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str], Set[str]]:
    """
    From ingredient_compound, compound_gene, and mediation_edges (IN_PATHWAY, MAPS_TO_CATEGORY),
    compute:
    - all_ings: unique ingredients in ingredient_compound
    - ings_with_compound: ingredients with >=1 compound
    - ings_with_gene: ingredients with >=1 compound that has a gene
    - ings_with_pathway: ingredients with compound->gene->pathway
    - cmp_ic: unique compounds in ingredient_compound
    - cmp_cg: unique compounds in compound_gene
    """
    all_ings: Set[str] = set()
    ing_cmp: Dict[str, Set[str]] = {}
    cmp_genes: Dict[str, Set[str]] = {}
    gene_paths: Set[str] = set()

    if not ingredient_compound.empty and "compound_id" in ingredient_compound.columns:
        ing_col = "ingredient_id" if "ingredient_id" in ingredient_compound.columns else ingredient_compound.columns[0]
        cmp_col = "compound_id"
        for _, r in ingredient_compound.iterrows():
            ing = to_ingredient_id(str(r.get(ing_col, "")))
            c = str(r.get(cmp_col, "")).strip()
            if ing and c:
                all_ings.add(ing)
                ing_cmp.setdefault(ing, set()).add(c)

    if not compound_gene.empty and "compound_id" in compound_gene.columns:
        cmp_col = "compound_id"
        gene_col = next((c for c in ("gene_id", "gene_symbol", "gene") if c in compound_gene.columns), None)
        if gene_col:
            for _, r in compound_gene.iterrows():
                c = str(r.get(cmp_col, "")).strip()
                g = to_gene_id(str(r.get(gene_col, "")).strip())
                if c and g and g != "GENE_unknown":
                    cmp_genes.setdefault(c, set()).add(g)

    if not mediation_edges.empty:
        gp = mediation_edges[mediation_edges["edge_type"] == "IN_PATHWAY"]
        for _, e in gp.iterrows():
            gene_paths.add(str(e["source_id"]))
            gene_paths.add(str(e["target_id"]))

    # Resolve gene -> pathway from edges (source_id=GENE, target_id=PATH)
    gene_to_path: Dict[str, Set[str]] = {}
    for _, e in mediation_edges.iterrows():
        if str(e.get("edge_type", "")) != "IN_PATHWAY":
            continue
        src, tgt = str(e["source_id"]), str(e["target_id"])
        gene_to_path.setdefault(src, set()).add(tgt)

    ings_with_compound = set(ing_cmp.keys())
    ings_with_gene: Set[str] = set()
    ings_with_pathway: Set[str] = set()
    for ing, cmps in ing_cmp.items():
        has_gene = False
        has_pathway = False
        for c in cmps:
            for g in cmp_genes.get(c, set()):
                has_gene = True
                if g in gene_to_path and gene_to_path[g]:
                    has_pathway = True
        if has_gene:
            ings_with_gene.add(ing)
        if has_pathway:
            ings_with_pathway.add(ing)

    cmp_ic = set()
    for cmps in ing_cmp.values():
        cmp_ic |= cmps
    cmp_cg = set(cmp_genes.keys()) if cmp_genes else set()

    return all_ings, ings_with_compound, ings_with_gene, ings_with_pathway, cmp_ic, cmp_cg


def compute_phase14_coverage_audit(
    ingredient_compound: pd.DataFrame,
    compound_gene: pd.DataFrame,
    mediation_edges: pd.DataFrame,
    atlas: pd.DataFrame,
    pair_mediation: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute coverage/evidence metrics for Phase14.
    Returns dict for reports/phase14_coverage_audit.json.
    """
    (
        all_ings,
        ings_with_compound,
        ings_with_gene,
        ings_with_pathway,
        cmp_ic,
        cmp_cg,
    ) = _ingredient_sets_from_tables_and_edges(ingredient_compound, compound_gene, mediation_edges)

    n_ing_total = len(all_ings) if all_ings else 0
    n_ing_gene = len(ings_with_gene)
    n_ing_pathway = len(ings_with_pathway)
    pct_ing_gene = round(100.0 * n_ing_gene / n_ing_total, 2) if n_ing_total else 0.0
    pct_ing_pathway = round(100.0 * n_ing_pathway / n_ing_total, 2) if n_ing_total else 0.0

    n_cmp_ic = len(cmp_ic)
    n_cmp_cg = len(cmp_cg)
    n_overlap = len(cmp_ic & cmp_cg)
    overlap_vs_cg = round(n_overlap / n_cmp_cg, 4) if n_cmp_cg else 0.0
    overlap_vs_ic = round(n_overlap / n_cmp_ic, 4) if n_cmp_ic else 0.0

    atlas_pair_cov_compound = 0.0
    atlas_pair_cov_gene = 0.0
    atlas_pair_cov_pathway = 0.0
    if not atlas.empty and "ingA_id" in atlas.columns and "ingB_id" in atlas.columns:
        a_set = atlas["ingA_id"].dropna().astype(str).apply(to_ingredient_id)
        b_set = atlas["ingB_id"].dropna().astype(str).apply(to_ingredient_id)
        n_rows = len(atlas)
        both_compound = (a_set.isin(ings_with_compound) & b_set.isin(ings_with_compound)).sum()
        both_gene = (a_set.isin(ings_with_gene) & b_set.isin(ings_with_gene)).sum()
        both_pathway = (a_set.isin(ings_with_pathway) & b_set.isin(ings_with_pathway)).sum()
        atlas_pair_cov_compound = round(100.0 * both_compound / n_rows, 2) if n_rows else 0.0
        atlas_pair_cov_gene = round(100.0 * both_gene / n_rows, 2) if n_rows else 0.0
        atlas_pair_cov_pathway = round(100.0 * both_pathway / n_rows, 2) if n_rows else 0.0

    frac_shared_genes_and_pathways = None
    if pair_mediation is not None and not pair_mediation.empty:
        sg = pair_mediation.get("shared_genes_count", pd.Series(dtype=float)).fillna(0)
        sp = pair_mediation.get("shared_pathways_count", pd.Series(dtype=float)).fillna(0)
        both = ((sg > 0) & (sp > 0)).sum()
        frac_shared_genes_and_pathways = round(both / len(pair_mediation), 4) if len(pair_mediation) else 0.0

    return {
        "n_unique_ingredients_total": n_ing_total,
        "n_unique_ingredients_with_gene": n_ing_gene,
        "pct_ingredients_with_gene": pct_ing_gene,
        "n_unique_ingredients_with_pathway": n_ing_pathway,
        "pct_ingredients_with_pathway": pct_ing_pathway,
        "n_unique_compounds_ic": n_cmp_ic,
        "n_unique_compounds_cg": n_cmp_cg,
        "n_overlap_compounds": n_overlap,
        "overlap_vs_cg": overlap_vs_cg,
        "overlap_vs_ic": overlap_vs_ic,
        "atlas_pair_cov_compound": atlas_pair_cov_compound,
        "atlas_pair_cov_gene": atlas_pair_cov_gene,
        "atlas_pair_cov_pathway": atlas_pair_cov_pathway,
        "frac_pair_mediation_with_shared_genes_and_pathways": frac_shared_genes_and_pathways,
    }


def build_top_missing_evidence(
    atlas: pd.DataFrame,
    pair_mediation: pd.DataFrame,
    ingredient_compound: pd.DataFrame,
    compound_gene: pd.DataFrame,
    mediation_edges: pd.DataFrame,
    top_n: int = 200,
) -> pd.DataFrame:
    """
    Top Phase13 rows (by q_global asc, stability_score desc) where mechanistic_score==0,
    with reason flags: no_compounds_A, no_compounds_B, no_genes_A, no_genes_B, no_pathways_A, no_pathways_B, no_shared_pathways.
    """
    if pair_mediation is None or pair_mediation.empty or atlas.empty:
        return pd.DataFrame()

    _, ings_with_compound, ings_with_gene, ings_with_pathway, _, _ = _ingredient_sets_from_tables_and_edges(
        ingredient_compound, compound_gene, mediation_edges
    )

    merge_cols = ["ingA_id", "ingB_id", "category"]
    pm = pair_mediation[merge_cols + ["mechanistic_score", "shared_compounds_count", "shared_genes_count", "shared_pathways_count"]].copy()
    pm["mechanistic_score"] = pm["mechanistic_score"].fillna(0)
    zero = pm[pm["mechanistic_score"] == 0].copy()
    if zero.empty:
        return pd.DataFrame(columns=list(atlas.columns) + ["no_compounds_A", "no_compounds_B", "no_genes_A", "no_genes_B", "no_pathways_A", "no_pathways_B", "no_shared_pathways"])

    merged = atlas.merge(
        zero[merge_cols + ["shared_compounds_count", "shared_genes_count", "shared_pathways_count"]],
        on=merge_cols,
        how="inner",
        suffixes=("", "_pm"),
    )
    if merged.empty:
        return pd.DataFrame()

    a_id = merged["ingA_id"].astype(str).apply(to_ingredient_id)
    b_id = merged["ingB_id"].astype(str).apply(to_ingredient_id)
    merged = merged.copy()
    merged["no_compounds_A"] = ~a_id.isin(ings_with_compound)
    merged["no_compounds_B"] = ~b_id.isin(ings_with_compound)
    merged["no_genes_A"] = ~a_id.isin(ings_with_gene)
    merged["no_genes_B"] = ~b_id.isin(ings_with_gene)
    merged["no_pathways_A"] = ~a_id.isin(ings_with_pathway)
    merged["no_pathways_B"] = ~b_id.isin(ings_with_pathway)
    merged["no_shared_pathways"] = (merged["shared_pathways_count"].fillna(0) <= 0)

    order = merged.sort_values(
        by=["q_global", "stability_score"],
        ascending=[True, False],
        na_position="last",
    )
    out = order.head(top_n)
    reason_cols = ["no_compounds_A", "no_compounds_B", "no_genes_A", "no_genes_B", "no_pathways_A", "no_pathways_B", "no_shared_pathways"]
    keep = [c for c in out.columns if c in list(atlas.columns) + ["shared_compounds_count", "shared_genes_count", "shared_pathways_count"] + reason_cols]
    return out[[c for c in keep if c in out.columns]]


def write_phase14_coverage_audit(output_dir: Path, audit: Dict[str, Any]) -> None:
    """Write reports/phase14_coverage_audit.json."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "phase14_coverage_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info("[%s] Wrote %s", _ts(), path)


def write_top_missing_evidence(output_dir: Path, df: pd.DataFrame) -> None:
    """Write reports/top_missing_evidence.csv."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "top_missing_evidence.csv"
    if df is not None and not df.empty:
        df.to_csv(path, index=False)
        logger.info("[%s] Wrote %s (%s rows)", _ts(), path, len(df))
    else:
        pd.DataFrame(columns=["ingA_id", "ingB_id", "category", "no_compounds_A", "no_compounds_B", "no_genes_A", "no_genes_B", "no_pathways_A", "no_pathways_B", "no_shared_pathways"]).to_csv(path, index=False)
        logger.info("[%s] Wrote %s (0 rows)", _ts(), path)
