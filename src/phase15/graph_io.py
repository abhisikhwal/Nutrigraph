"""
Phase15: Load graph from Phase14 snapshot + optional canonical CSVs.
Produces node/edge DataFrames and unified node id mapping.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_neo4j_export(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load nodes and edges from run_dir/neo4j/nodes.csv and edges.csv."""
    run_dir = Path(run_dir).resolve()
    nodes_path = run_dir / "neo4j" / "nodes.csv"
    edges_path = run_dir / "neo4j" / "edges.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing {edges_path}")
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path, low_memory=False)
    # Normalize column names
    id_col = ":ID" if ":ID" in nodes.columns else "node_id"
    if id_col not in nodes.columns and len(nodes.columns):
        id_col = nodes.columns[0]
    label_col = ":LABEL" if ":LABEL" in nodes.columns else "label"
    nodes = nodes.rename(columns={c: c.strip() for c in nodes.columns})
    edges = edges.rename(columns={c: c.strip() for c in edges.columns})
    logger.info("Loaded neo4j nodes=%d edges=%d", len(nodes), len(edges))
    return nodes, edges


def load_compound_gene(repo_root: Path) -> pd.DataFrame:
    """Load compound->gene edges. Prefer expanded, fallback canonical."""
    repo_root = Path(repo_root).resolve()
    for name in ["compound_gene_expanded_canonical.csv", "compound_gene_canonical.csv"]:
        p = repo_root / "data" / "processed" / "canonical" / name
        if p.exists():
            df = pd.read_csv(p)
            if "compound_id" in df.columns and ("gene_symbol" in df.columns or "gene" in df.columns):
                if "gene_symbol" not in df.columns:
                    df = df.rename(columns={"gene": "gene_symbol"})
                logger.info("Loaded compound_gene from %s: %d rows", p.name, len(df))
                return df[["compound_id", "gene_symbol"]].drop_duplicates().dropna()
    return pd.DataFrame(columns=["compound_id", "gene_symbol"])


def load_ingredient_compound(repo_root: Path) -> pd.DataFrame:
    """Load ingredient->compound edges."""
    repo_root = Path(repo_root).resolve()
    p = repo_root / "data" / "processed" / "canonical" / "ingredient_compound_canonical.csv"
    if not p.exists():
        return pd.DataFrame(columns=["ingredient_id", "compound_id"])
    df = pd.read_csv(p)
    if "ingredient_id" not in df.columns or "compound_id" not in df.columns:
        return pd.DataFrame(columns=["ingredient_id", "compound_id"])
    out = df[["ingredient_id", "compound_id"]].drop_duplicates().dropna()
    logger.info("Loaded ingredient_compound: %d rows", len(out))
    return out


def load_pair_category_mediation(run_dir: Path) -> pd.DataFrame:
    """Load pair_category_mediation.csv from run dir."""
    run_dir = Path(run_dir).resolve()
    p = run_dir / "pair_category_mediation.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    logger.info("Loaded pair_category_mediation: %d rows", len(df))
    return df


def build_ingredient_category_edges(pair_med: pd.DataFrame) -> pd.DataFrame:
    """From pair_category_mediation (ingA_id, ingB_id, category) build (ingredient_id, category)."""
    if pair_med.empty or "ingA_id" not in pair_med.columns or "category" not in pair_med.columns:
        return pd.DataFrame(columns=["ingredient_id", "category"])
    a = pair_med[["ingA_id", "category"]].rename(columns={"ingA_id": "ingredient_id"})
    b = pair_med[["ingB_id", "category"]].rename(columns={"ingB_id": "ingredient_id"})
    out = pd.concat([a, b], ignore_index=True).drop_duplicates().dropna()
    return out


def load_pathway_bundles(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Load pathway_bundles.json if present; may contain pathway->category mapping."""
    p = Path(repo_root).resolve() / "data" / "processed" / "features" / "pathway_bundles.json"
    if not p.exists():
        return None
    import json
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_repo_root() -> Path:
    """Repo root (parent of src)."""
    return Path(__file__).resolve().parents[2]
