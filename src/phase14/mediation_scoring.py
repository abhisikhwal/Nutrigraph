"""
Phase14: Mediation scoring — shared counts, mediated_path_score, mechanistic_score.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from . import phase14_config as config
from .id_normalization import to_ingredient_id, to_interaction_id

logger = logging.getLogger(__name__)


def _ts() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _build_ingredient_sets(
    mediation_edges: pd.DataFrame,
    mediation_nodes: pd.DataFrame,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    From mediation graph build:
    ing -> set of compound ids, ing -> set of gene ids (via compounds or direct), ing -> set of pathway ids.
    HAS_COMPOUND: source=ING, target=CMP. TARGETS: source=CMP, target=GENE. IN_PATHWAY: source=GENE, target=PATH.
    """
    ing_compounds: Dict[str, Set[str]] = {}
    ing_genes: Dict[str, Set[str]] = {}
    ing_pathways: Dict[str, Set[str]] = {}
    edges = mediation_edges
    if edges.empty:
        return ing_compounds, ing_genes, ing_pathways
    cmp_genes: Dict[str, Set[str]] = {}
    gene_pathways: Dict[str, Set[str]] = {}
    for _, e in edges.iterrows():
        src, tgt, etype = str(e["source_id"]), str(e["target_id"]), str(e.get("edge_type", ""))
        if etype == "HAS_COMPOUND":
            ing_compounds.setdefault(src, set()).add(tgt)
        elif etype == "TARGETS":
            cmp_genes.setdefault(src, set()).add(tgt)
        elif etype == "IN_PATHWAY":
            gene_pathways.setdefault(src, set()).add(tgt)
    for ing, cmps in ing_compounds.items():
        for c in cmps:
            for g in cmp_genes.get(c, set()):
                ing_genes.setdefault(ing, set()).add(g)
            for g in cmp_genes.get(c, set()):
                for path in gene_pathways.get(g, set()):
                    ing_pathways.setdefault(ing, set()).add(path)
    return ing_compounds, ing_genes, ing_pathways


def _category_to_pathways(mediation_edges: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build category -> set of pathway ids from MAPS_TO_CATEGORY edges (source=PATH, target=CAT)."""
    cat_paths: Dict[str, Set[str]] = {}
    for _, e in mediation_edges.iterrows():
        if str(e.get("edge_type", "")) != "MAPS_TO_CATEGORY":
            continue
        path_id = str(e["source_id"])
        cat_id = str(e["target_id"])
        cat_paths.setdefault(cat_id, set()).add(path_id)
    return cat_paths


def _build_compound_target_sets(compound_target_df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build compound_id -> set of target_name, and return (compound_targets, {}) for future use."""
    compound_targets: Dict[str, Set[str]] = {}
    if compound_target_df.empty or "compound_id" not in compound_target_df.columns:
        return compound_targets, {}
    target_col = next((c for c in compound_target_df.columns if "target" in c.lower() and "name" in c.lower()), None)
    if not target_col:
        return compound_targets, {}
    for _, row in compound_target_df.iterrows():
        cid = str(row.get("compound_id", "")).strip().upper()
        t = str(row.get(target_col, "")).strip()
        if cid and t:
            compound_targets.setdefault(cid, set()).add(t)
    return compound_targets, {}


def compute_mediation_features(
    atlas_confirmed: pd.DataFrame,
    mediation_edges: pd.DataFrame,
    mediation_nodes: pd.DataFrame,
    pathway_weights: Optional[Dict[str, float]] = None,
    compound_target_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    For each (ingA, ingB, category) row in atlas_confirmed compute:
    shared_compounds_count, shared_genes_count, shared_pathways_count,
    mediated_path_score, mediation_confidence.
    When compound_target_df is provided, also compute evidence_target_count and evidence_target_overlap.
    When ingredient->pathway is missing, use category->pathways as fallback so shared_pathways is non-zero.
    """
    ing_compounds, ing_genes, ing_pathways = _build_ingredient_sets(mediation_edges, mediation_nodes)
    _ct_df = compound_target_df if compound_target_df is not None else pd.DataFrame()
    compound_targets, _ = _build_compound_target_sets(_ct_df)
    cat_pathways = _category_to_pathways(mediation_edges)
    pathway_weights = pathway_weights or {}
    default_pw = 1.0
    rows = []
    for _, r in atlas_confirmed.iterrows():
        a = to_ingredient_id(str(r["ingA_id"]))
        b = to_ingredient_id(str(r["ingB_id"]))
        cat = str(r["category"]).strip()
        cat_id = f"CAT_{cat.replace(' ', '_').lower()}"
        cmp_a = ing_compounds.get(a, set())
        cmp_b = ing_compounds.get(b, set())
        gene_a = ing_genes.get(a, set())
        gene_b = ing_genes.get(b, set())
        path_a = ing_pathways.get(a, set())
        path_b = ing_pathways.get(b, set())
        if not path_a and not path_b and cat_id in cat_pathways:
            path_a = cat_pathways[cat_id].copy()
            path_b = cat_pathways[cat_id].copy()
        shared_compounds = len(cmp_a & cmp_b)
        shared_genes = len(gene_a & gene_b)
        shared_pathways = len(path_a & path_b)
        # Optional: compound_target evidence
        evidence_target_count = 0
        evidence_target_overlap = 0
        if compound_targets:
            shared_cmp = cmp_a & cmp_b
            targets_a: Set[str] = set()
            targets_b: Set[str] = set()
            for c in cmp_a:
                targets_a |= compound_targets.get(c, set())
            for c in cmp_b:
                targets_b |= compound_targets.get(c, set())
            evidence_target_overlap = len(targets_a & targets_b)
            for c in shared_cmp:
                evidence_target_count += len(compound_targets.get(c, set()))
        # Mediated path score: sum over pathways (pathway_weight * gene_support). Use shared_pathways as proxy for support.
        path_union = path_a | path_b
        mediated_path_score = 0.0
        for pid in path_union:
            w = pathway_weights.get(pid, default_pw)
            in_a = 1.0 if pid in path_a else 0.0
            in_b = 1.0 if pid in path_b else 0.0
            mediated_path_score += w * (in_a + in_b) / 2.0
        # Mediation confidence in [0,1]: simple function of shared counts
        raw_conf = (shared_genes / 10.0 + shared_compounds / 5.0 + shared_pathways / 5.0)
        mediation_confidence = min(1.0, raw_conf)
        row_out = {
            "ingA_id": a,
            "ingB_id": b,
            "category": cat,
            "shared_compounds_count": shared_compounds,
            "shared_genes_count": shared_genes,
            "shared_pathways_count": shared_pathways,
            "mediated_path_score": mediated_path_score,
            "mediation_confidence": mediation_confidence,
            "did": r.get("did"),
            "q_global": r.get("q_global"),
            "p_analytic": r.get("p_analytic"),
        }
        if compound_targets:
            row_out["evidence_target_count"] = evidence_target_count
            row_out["evidence_target_overlap"] = evidence_target_overlap
        rows.append(row_out)
    return pd.DataFrame(rows)


# Evidence caps for mechanistic_score (log1p normalization to avoid saturation)
EVIDENCE_COMPOUND_CAP = 100
EVIDENCE_GENE_CAP = 20
EVIDENCE_PATHWAY_CAP = 10
MEDIATION_WEIGHT_COMPOUNDS = 0.2
MEDIATION_WEIGHT_GENES = 0.4
MEDIATION_WEIGHT_PATHWAYS = 0.4


def add_mechanistic_score(
    features_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    a1: float = config.MECH_A1,
    a2: float = config.MECH_A2,
    a3: float = config.MECH_A3,
) -> pd.DataFrame:
    """
    Add mechanistic_score from evidence strength (log1p-capped) and stability/q weights.
    evidence_compounds = log1p(shared_compounds) / log1p(100), same for genes (cap 20), pathways (cap 10).
    mediation_strength = 0.2*ev_compounds + 0.4*ev_genes + 0.4*ev_paths.
    If shared_genes_count == 0 and shared_pathways_count == 0, mechanistic_score = 0.
    Otherwise mechanistic_score = clip(mediation_strength * stability_weight * q_weight, 0, 1).
    """
    import numpy as np
    df = features_df.copy()
    q = df.get("q_global")
    # q_weight: high significance (low q_global) -> weight near 1
    q_arr = np.asarray(q, dtype=float) if q is not None else np.zeros(len(df))
    q_clip = np.clip(q_arr, 1e-30, 1.0)
    q_weight = np.clip(1.0 - q_clip, 0.0, 1.0)
    if not stability_df.empty and "stability_score" in stability_df.columns:
        key_cols = [c for c in ["ingA_id", "ingB_id", "category"] if c in stability_df.columns]
        if key_cols:
            stab_merge = stability_df[key_cols + ["stability_score"]].drop_duplicates()
            df = df.merge(stab_merge, on=key_cols, how="left")
            stability_weight = df["stability_score"].fillna(0.5).astype(float)
        else:
            stability_weight = np.full(len(df), 0.5)
    else:
        stability_weight = np.full(len(df), 0.5)
    stability_weight = np.clip(stability_weight, 0.0, 1.0)
    sg = df["shared_genes_count"].fillna(0).astype(float)
    sc = df["shared_compounds_count"].fillna(0).astype(float)
    sp = df["shared_pathways_count"].fillna(0).astype(float)
    # Evidence components in [0,1] via log1p / log1p(cap)
    ev_compounds = np.log1p(np.minimum(sc, EVIDENCE_COMPOUND_CAP)) / math.log1p(EVIDENCE_COMPOUND_CAP)
    ev_genes = np.log1p(np.minimum(sg, EVIDENCE_GENE_CAP)) / math.log1p(EVIDENCE_GENE_CAP)
    ev_pathways = np.log1p(np.minimum(sp, EVIDENCE_PATHWAY_CAP)) / math.log1p(EVIDENCE_PATHWAY_CAP)
    mediation_strength = (
        MEDIATION_WEIGHT_COMPOUNDS * ev_compounds
        + MEDIATION_WEIGHT_GENES * ev_genes
        + MEDIATION_WEIGHT_PATHWAYS * ev_pathways
    )
    # Optional: boost from compound_target evidence (bounded)
    EVIDENCE_TARGET_WEIGHT = 0.15
    if "evidence_target_count" in df.columns:
        tc = np.asarray(df["evidence_target_count"].fillna(0).astype(float))
        ev_target = 1.0 / (1.0 + np.exp(-np.log1p(np.minimum(tc, 500))))
        mediation_strength = np.clip(mediation_strength + EVIDENCE_TARGET_WEIGHT * ev_target, 0.0, 1.0)
        df["evidence_target"] = ev_target
    if "evidence_target_overlap" in df.columns:
        tov = np.asarray(df["evidence_target_overlap"].fillna(0).astype(float))
        ev_overlap = 1.0 / (1.0 + np.exp(-np.log1p(np.minimum(tov, 100))))
        if "evidence_target" not in df.columns:
            df["evidence_target"] = np.zeros(len(df))
        df["evidence_target"] = np.maximum(df["evidence_target"], ev_overlap)
    # No genes and no pathways -> no mechanistic support, unless compound_target evidence exists
    no_mechanism = (sg == 0) & (sp == 0)
    has_target_evidence = np.zeros(len(df), dtype=bool)
    if "evidence_target_count" in df.columns:
        has_target_evidence |= (df["evidence_target_count"].fillna(0).astype(float) > 0)
    if "evidence_target_overlap" in df.columns:
        has_target_evidence |= (df["evidence_target_overlap"].fillna(0).astype(float) > 0)
    mechanistic = np.clip(mediation_strength * stability_weight * q_weight, 0.0, 1.0)
    df["mechanistic_score"] = np.where(no_mechanism & ~has_target_evidence, 0.0, mechanistic)
    # Expose components for audit and export (no "mystery scores")
    df["evidence_compounds"] = ev_compounds
    df["evidence_genes"] = ev_genes
    df["evidence_paths"] = ev_pathways
    df["q_weight"] = q_weight
    df["stability_weight"] = stability_weight
    df["mediation_strength"] = mediation_strength
    return df
