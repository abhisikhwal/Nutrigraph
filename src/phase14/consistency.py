"""
Phase14: Multi-layer consistency — pathway similarity (Jaccard), Laplacian smoothing, coherence_flag/score.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import phase14_config as config

logger = logging.getLogger(__name__)


def _ts() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def pathway_gene_sets(mediation_edges: pd.DataFrame) -> Dict[str, Set[str]]:
    """pathway_id -> set of gene ids from IN_PATHWAY (source=gene, target=pathway)."""
    out: Dict[str, Set[str]] = {}
    for _, e in mediation_edges[mediation_edges["edge_type"] == "IN_PATHWAY"].iterrows():
        g, p = str(e["source_id"]), str(e["target_id"])
        out.setdefault(p, set()).add(g)
    return out


def pathway_similarity_jaccard(pathway_genes: Dict[str, Set[str]]) -> np.ndarray:
    """Pairwise Jaccard similarity matrix (pathways x pathways)."""
    paths = sorted(pathway_genes.keys())
    n = len(paths)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pathway_genes[paths[i]], pathway_genes[paths[j]]
            inter = len(a & b)
            union = len(a | b)
            jacc = inter / union if union else 0.0
            sim[i, j] = sim[j, i] = jacc
    return sim


def laplacian_smooth(scores: np.ndarray, adj: np.ndarray, n_iter: int = config.LAPLACIAN_ITERATIONS) -> np.ndarray:
    """One or two iterations of Laplacian smoothing: score_new = score + alpha * (D - A) @ score."""
    if adj.size == 0 or n_iter < 1:
        return scores
    n = adj.shape[0]
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    alpha = 0.2
    out = scores.copy()
    for _ in range(n_iter):
        out = out - alpha * (L @ out)
    return out


def apply_pathway_smoothing(
    pathway_scores: Dict[str, float],
    pathway_genes: Dict[str, Set[str]],
    n_iter: int = config.LAPLACIAN_ITERATIONS,
) -> Dict[str, float]:
    """Smooth pathway scores using pathway-pathway Jaccard graph."""
    paths = sorted(pathway_genes.keys())
    if not paths:
        return pathway_scores
    idx = {p: i for i, p in enumerate(paths)}
    vec = np.array([pathway_scores.get(p, 0.0) for p in paths], dtype=float)
    sim = pathway_similarity_jaccard(pathway_genes)
    adj = sim  # use similarity as adjacency
    smoothed = laplacian_smooth(vec, adj, n_iter)
    return {p: float(smoothed[idx[p]]) for p in paths}


def coherence_flag_and_score(
    pair_mediation: pd.DataFrame,
    did_positive_threshold: float = 1e-6,
    incoherent_threshold: float = config.INCOHERENT_DID_SIGN_THRESHOLD,
) -> pd.DataFrame:
    """
    Add coherence_flag and coherence_score. Incoherent when DID positive but mediated score strongly negative,
    or DID negative but mediated score strongly positive. coherence_score in [0,1]; 0 = incoherent.
    """
    df = pair_mediation.copy()
    did = df.get("did", pd.Series(0.0))
    mech = df.get("mechanistic_score", pd.Series(0.5))
    if "mechanistic_score" not in df.columns:
        df["coherence_flag"] = True
        df["coherence_score"] = 1.0
        return df
    did_pos = (did > did_positive_threshold).astype(float)
    did_neg = (did < -did_positive_threshold).astype(float)
    mech_high = (mech > 0.5).astype(float)
    mech_low = (mech < 0.5).astype(float)
    incoherent = ((did_pos == 1) & (mech_low == 1)) | ((did_neg == 1) & (mech_high == 1))
    df["coherence_flag"] = ~incoherent
    df["coherence_score"] = np.where(incoherent, np.clip(1.0 - mech, 0, 1), np.clip(mech, 0, 1))
    return df
