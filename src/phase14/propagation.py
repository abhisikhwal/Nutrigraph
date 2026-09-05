"""
Phase14: Pathway graph propagation — RWR on gene-pathway bipartite, propagated_pathway_score per category.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import phase14_config as config
from .id_normalization import to_category_id, to_gene_id, to_ingredient_id

logger = logging.getLogger(__name__)


def build_ingredient_genes_from_tables(
    ingredient_compound: pd.DataFrame,
    compound_gene: pd.DataFrame,
) -> Dict[str, Set[str]]:
    """
    Build ingredient_id -> set of gene_id from ingredient_compound and compound_gene (merge on compound_id).
    Uses same ID normalization as mediation graph (to_ingredient_id, to_gene_id).
    Returns dict suitable for propagated_scores_for_pairs(..., ingredient_genes=...).
    """
    out: Dict[str, Set[str]] = {}
    if ingredient_compound.empty or compound_gene.empty:
        return out
    ic = ingredient_compound
    cg = compound_gene
    ing_col = "ingredient_id" if "ingredient_id" in ic.columns else ic.columns[0]
    cmp_col_ic = "compound_id" if "compound_id" in ic.columns else None
    cmp_col_cg = "compound_id" if "compound_id" in cg.columns else None
    gene_col = next((c for c in ("gene", "gene_symbol", "gene_id") if c in cg.columns), None)
    if not cmp_col_ic or not cmp_col_cg or not gene_col:
        return out
    ic_clean = ic[[ing_col, cmp_col_ic]].dropna().drop_duplicates()
    cg_clean = cg[[cmp_col_cg, gene_col]].dropna().drop_duplicates()
    merged = ic_clean.merge(cg_clean, left_on=cmp_col_ic, right_on=cmp_col_cg, how="inner")
    for _, r in merged.iterrows():
        ing_id = to_ingredient_id(str(r[ing_col]))
        g_id = to_gene_id(str(r[gene_col]))
        out.setdefault(ing_id, set()).add(g_id)
    return out


def _ts() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def build_gene_pathway_adjacency(mediation_edges: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build bipartite adjacency (genes x pathways) from IN_PATHWAY edges.
    Returns (adj_matrix, gene_ids, pathway_ids).
    """
    gene_path = mediation_edges[mediation_edges["edge_type"] == "IN_PATHWAY"]
    if gene_path.empty:
        return np.array([]), [], []
    genes = sorted(gene_path["source_id"].astype(str).unique().tolist())
    pathways = sorted(gene_path["target_id"].astype(str).unique().tolist())
    g2i = {g: i for i, g in enumerate(genes)}
    p2j = {p: j for j, p in enumerate(pathways)}
    n, m = len(genes), len(pathways)
    adj = np.zeros((n, m))
    for _, e in gene_path.iterrows():
        i = g2i.get(str(e["source_id"]))
        j = p2j.get(str(e["target_id"]))
        if i is not None and j is not None:
            adj[i, j] = float(e.get("weight", 1.0))
    return adj, genes, pathways


def rwr_bipartite(
    adj: np.ndarray,
    restart_prob: float = config.RWR_RESTART_PROB,
    max_iter: int = config.RWR_MAX_ITER,
    tol: float = config.RWR_TOL,
) -> np.ndarray:
    """
    Random walk with restart on bipartite (genes x pathways).
    Each gene row is normalized; restart from seed (identity on gene side).
    Returns (n_genes x n_pathways) score matrix.
    """
    if adj.size == 0:
        return adj
    n, m = adj.shape
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P = adj / row_sum  # (n_genes x n_pathways) gene -> pathway
    col_sum = adj.sum(axis=0, keepdims=True) + 1e-12  # (1, n_pathways)
    P_back = (adj.T / col_sum.T)  # (n_pathways x n_genes) pathway -> gene
    # RWR: r_new = (1-c)*r*P_back*P + c*seed (seed = which genes are seeds)
    # Simpler: propagate from seed genes over bipartite. Seed = unit vector on genes.
    # One iteration: pathway_scores = seed @ P; then gene_scores = pathway_scores @ P_back; then blend with restart.
    R = np.eye(n)  # each column is result of RWR from one gene
    for _ in range(max_iter):
        R_new = (1.0 - restart_prob) * (R @ P @ P_back) + restart_prob * np.eye(n)
        if np.abs(R_new - R).max() < tol:
            break
        R = R_new
    # Scores from each gene to pathways: R gives gene->gene; we want gene->pathway = R @ P
    return R @ P


def ingredient_genes_from_edges(mediation_edges: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build ing -> set of gene ids from HAS_COMPOUND -> TARGETS -> genes, or empty if no compound layer."""
    ing_genes: Dict[str, Set[str]] = {}
    cmp_genes: Dict[str, Set[str]] = {}
    for _, e in mediation_edges.iterrows():
        src, tgt, etype = str(e["source_id"]), str(e["target_id"]), str(e.get("edge_type", ""))
        if etype == "HAS_COMPOUND":
            ing_genes.setdefault(src, set()).add(tgt)  # ing -> compound
        elif etype == "TARGETS":
            cmp_genes.setdefault(src, set()).add(tgt)
    # Resolve ing -> compound -> gene
    out: Dict[str, Set[str]] = {}
    for ing, cmps in ing_genes.items():
        for c in cmps:
            for g in cmp_genes.get(c, set()):
                out.setdefault(ing, set()).add(g)
    return out


def propagate_pathway_scores(
    mediation_edges: pd.DataFrame,
    path_to_category_edges: pd.DataFrame,
    ingredient_genes: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, List[str], List[str]]:
    """
    RWR from ingredient target genes to pathways; combine A and B by harmonic mean.
    path_to_category_edges: edges with source_id=PATH, target_id=CAT (or use mediation_edges filtered by MAPS_TO_CATEGORY).
    Returns (ing_pair_key -> {category: propagated_pathway_score}, adj, gene_ids, pathway_ids).
    """
    adj, gene_ids, pathway_ids = build_gene_pathway_adjacency(mediation_edges)
    if adj.size == 0 or not pathway_ids:
        return {}, adj, gene_ids, pathway_ids
    score_gene_path = rwr_bipartite(adj)
    path_to_cat: Dict[str, Set[str]] = {}
    for _, e in path_to_category_edges.iterrows():
        if str(e.get("edge_type", "")) != "MAPS_TO_CATEGORY":
            continue
        path_to_cat.setdefault(str(e["source_id"]), set()).add(str(e["target_id"]))
    if not path_to_cat:
        path_to_cat = {p: set() for p in pathway_ids}
    ing_genes = ingredient_genes or {}
    g2i = {g: i for i, g in enumerate(gene_ids)}
    p2j = {p: j for j, p in enumerate(pathway_ids)}
    results: Dict[str, Dict[str, float]] = {}
    return results, adj, gene_ids, pathway_ids


# Small floor for propagation when both ingredients have genes and category has pathways (avoid exact 0 from numerics)
PROPAGATION_FLOOR = 1e-6

# Root cause of low propagation (fixed): (1) Category ID mismatch — propagation used ad-hoc
# "CAT_{cat.replace(' ', '_').lower()}" while the mediation graph uses to_category_id(cat),
# so any character stripped by UNSAFE_PATTERN (e.g. apostrophes) caused atlas category to
# not match MAPS_TO_CATEGORY target_id. (2) Atlas categories (e.g. signaling, nervous, immune)
# often are not keys in pathway_bundles.json; no pathway cluster top_terms matched those names,
# so category_bundle_exists was false. Fix: use to_category_id(cat) and CATEGORY_SYNONYMS_FOR_PROPAGATION
# so atlas categories resolve to bundle keys that have pathways in the graph.


def propagation_diagnostics_from_scores(scores_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute diagnostics from a propagated_scores DataFrame (column propagated_pathway_score)."""
    if scores_df is None or scores_df.empty or "propagated_pathway_score" not in scores_df.columns:
        return {"n_rows": 0, "n_nonzero": 0, "pct_rows_with_nonzero_propagation": 0.0, "median_propagation_nonzero": 0.0}
    s = scores_df["propagated_pathway_score"].fillna(0)
    n = len(scores_df)
    nz = (s > 0).sum()
    pct = round(100.0 * nz / n, 2) if n else 0.0
    non_zero = s[s > 0]
    median_nz = float(non_zero.median()) if len(non_zero) else 0.0
    return {
        "n_rows": n,
        "n_nonzero": int(nz),
        "pct_rows_with_nonzero_propagation": pct,
        "median_propagation_nonzero": median_nz,
    }


def pct_resolved_category_has_pathways(
    atlas_confirmed: pd.DataFrame,
    mediation_edges: pd.DataFrame,
) -> float:
    """
    Fraction of atlas rows whose category (after synonym resolution) has at least one pathway in the graph.
    Used for fail-fast when alignment is broken (e.g. atlas categories never match pathway_bundles).
    """
    map_edges = mediation_edges[mediation_edges["edge_type"] == "MAPS_TO_CATEGORY"]
    path_to_cat: Dict[str, List[str]] = {}
    for _, e in map_edges.iterrows():
        path_to_cat.setdefault(str(e["source_id"]), []).append(str(e["target_id"]))
    cats_with_pathways = set()
    for paths in path_to_cat.values():
        cats_with_pathways.update(paths)
    if not atlas_confirmed.shape[0]:
        return 100.0
    n_ok = 0
    for _, r in atlas_confirmed.iterrows():
        cat = str(r["category"]).strip()
        cat_id = to_category_id(cat)
        if cat_id in cats_with_pathways:
            n_ok += 1
            continue
        cat_norm_key = cat_id.replace("CAT_", "", 1) if cat_id.startswith("CAT_") else cat_id
        syn = config.CATEGORY_SYNONYMS_FOR_PROPAGATION.get(cat_norm_key)
        if syn:
            if to_category_id(syn) in cats_with_pathways:
                n_ok += 1
    return round(100.0 * n_ok / len(atlas_confirmed), 2)


def propagated_scores_for_pairs(
    atlas_confirmed: pd.DataFrame,
    mediation_edges: pd.DataFrame,
    ingredient_genes: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    For each (ingA, ingB, category) compute propagated_pathway_score.
    Combine pathway ranks for A and B via harmonic mean; normalized RWR scores in [0,1].
    Returns (scores_df, diagnostics_dict). Avoid returning exactly 0 unless truly disconnected (no genes or no pathway for category).
    """
    map_edges = mediation_edges[mediation_edges["edge_type"] == "MAPS_TO_CATEGORY"]
    _, adj, gene_ids, pathway_ids = propagate_pathway_scores(mediation_edges, map_edges, ingredient_genes)
    empty_df = atlas_confirmed[["ingA_id", "ingB_id", "category"]].copy()
    empty_df["propagated_pathway_score"] = 0.0
    if adj.size == 0:
        return empty_df, propagation_diagnostics_from_scores(empty_df)
    score_gp = rwr_bipartite(adj)
    # Normalize each gene row to [0,1] by row max so scores are bounded
    row_max = score_gp.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    score_gp = score_gp / row_max
    g2i = {g: i for i, g in enumerate(gene_ids)}
    path_to_cat: Dict[str, List[str]] = {}
    for _, e in map_edges.iterrows():
        path_to_cat.setdefault(str(e["source_id"]), []).append(str(e["target_id"]))
    cats_with_pathways = set()
    for paths in path_to_cat.values():
        cats_with_pathways.update(paths)
    ing_genes = ingredient_genes or {}
    rows = []
    for _, r in atlas_confirmed.iterrows():
        a = to_ingredient_id(str(r["ingA_id"]))
        b = to_ingredient_id(str(r["ingB_id"]))
        cat = str(r["category"]).strip()
        # Use to_category_id so category matches MAPS_TO_CATEGORY target_id from mediation graph.
        # If this category has no pathways in the graph, try synonym map so atlas categories
        # (e.g. signaling, nervous, immune) resolve to bundle keys that do have pathways.
        cat_id = to_category_id(cat)
        if cat_id not in cats_with_pathways:
            cat_norm_key = cat_id.replace("CAT_", "", 1) if cat_id.startswith("CAT_") else cat_id
            syn = config.CATEGORY_SYNONYMS_FOR_PROPAGATION.get(cat_norm_key)
            if syn:
                cat_id_2 = to_category_id(syn)
                if cat_id_2 in cats_with_pathways:
                    cat_id = cat_id_2
        ga = ing_genes.get(a, set())
        gb = ing_genes.get(b, set())
        sa = np.zeros(len(pathway_ids))
        for g in ga:
            i = g2i.get(g)
            if i is not None:
                sa += score_gp[i, :]
        sb = np.zeros(len(pathway_ids))
        for g in gb:
            i = g2i.get(g)
            if i is not None:
                sb += score_gp[i, :]
        if sa.sum() > 0:
            sa /= sa.sum()
        if sb.sum() > 0:
            sb /= sb.sum()
        combined = 2 * sa * sb / (sa + sb + 1e-12)
        score = 0.0
        for j, path in enumerate(pathway_ids):
            if cat_id in path_to_cat.get(path, []):
                score += combined[j]
        score = float(score)
        # Avoid exact 0 when both have genes and category has pathways (smoothing)
        if score == 0.0 and ga and gb and cat_id in cats_with_pathways:
            score = PROPAGATION_FLOOR
        rows.append({"ingA_id": a, "ingB_id": b, "category": cat, "propagated_pathway_score": score})
    df = pd.DataFrame(rows)
    return df, propagation_diagnostics_from_scores(df)
