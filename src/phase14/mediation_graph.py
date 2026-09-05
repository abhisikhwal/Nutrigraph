"""
Phase14: Build mediation KG — nodes and edges for ING, CMP, GENE, PATH, CAT, INT.
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from . import phase14_config as config
from .id_normalization import (
    to_category_id,
    to_compound_id,
    to_gene_id,
    to_ingredient_id,
    to_interaction_id,
    to_pathway_id,
)

logger = logging.getLogger(__name__)


def _ts() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _parse_list_column(ser: pd.Series) -> List[List[str]]:
    """Parse column that may contain '[\'a\', \'b\']' or list-like strings."""
    out: List[List[str]] = []
    for v in ser:
        if pd.isna(v):
            out.append([])
            continue
        if isinstance(v, list):
            out.append([str(x) for x in v])
            continue
        s = str(v).strip()
        if not s or s == "[]":
            out.append([])
            continue
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                out.append([str(x) for x in parsed])
            else:
                out.append([str(parsed)])
        except Exception:
            # Fallback: split by comma and strip quotes
            parts = re.findall(r"['\"]?([^,\[\]]+)['\"]?", s)
            out.append([p.strip().strip("'\"") for p in parts if p.strip()])
    return out


def _pathway_terms_to_categories(
    top_terms: str,
    pathway_bundles: Dict[str, List[str]],
) -> List[str]:
    """Map pathway top_terms string to category IDs (CAT_*) via keyword overlap with pathway_bundles."""
    if not top_terms or not pathway_bundles:
        return []
    try:
        terms = ast.literal_eval(top_terms) if isinstance(top_terms, str) and top_terms.startswith("[") else []
    except Exception:
        terms = [t.strip() for t in str(top_terms).lower().replace("'", "").split(",")]
    if not isinstance(terms, list):
        terms = [str(terms)]
    terms_str = " ".join(str(t).lower() for t in terms)
    matched: List[str] = []
    for cat, keywords in pathway_bundles.items():
        for kw in keywords:
            if kw.lower() in terms_str:
                cid = to_category_id(cat)
                if cid not in matched:
                    matched.append(cid)
                break
    return matched


def build_mediation_graph(
    atlas_confirmed: pd.DataFrame,
    kg_edges: pd.DataFrame,
    kg_nodes: pd.DataFrame,
    pathway_cluster_info: pd.DataFrame,
    target_functional_clusters: pd.DataFrame,
    pathway_bundles: Dict[str, List[str]],
    categories: Optional[List[str]] = None,
    ingredient_compound: Optional[pd.DataFrame] = None,
    compound_gene: Optional[pd.DataFrame] = None,
    ingredient_id_to_name: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build mediation_nodes and mediation_edges DataFrames.
    Partial graph when ingredient_compound or compound_gene are missing.
    When ingredient_id_to_name is provided, Ingredient node name is the human-readable name; otherwise name = id.
    """
    categories = categories or config.DEFAULT_CATEGORIES
    node_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    seen_nodes: Set[str] = set()
    id_to_name = ingredient_id_to_name or {}

    # ----- Ingredients & categories from Phase13 -----
    all_ings: Set[str] = set()
    for _, r in atlas_confirmed.iterrows():
        a = to_ingredient_id(str(r["ingA_id"]))
        b = to_ingredient_id(str(r["ingB_id"]))
        all_ings.add(a)
        all_ings.add(b)
    for ing in sorted(all_ings):
        nid = ing
        if nid in seen_nodes:
            continue
        seen_nodes.add(nid)
        display_name = id_to_name.get(nid, nid)
        node_rows.append({"node_id": nid, "label": "Ingredient", "name": display_name, "source": "phase13"})
    for cat in categories:
        cid = to_category_id(cat)
        if cid in seen_nodes:
            continue
        seen_nodes.add(cid)
        node_rows.append({"node_id": cid, "label": "Category", "name": cat, "source": "phase14"})

    # ----- Interaction nodes -----
    for _, r in atlas_confirmed.iterrows():
        iid = to_interaction_id(str(r["ingA_id"]), str(r["ingB_id"]))
        if iid in seen_nodes:
            continue
        seen_nodes.add(iid)
        node_rows.append({
            "node_id": iid,
            "label": "Interaction",
            "name": iid,
            "ingA_id": to_ingredient_id(str(r["ingA_id"])),
            "ingB_id": to_ingredient_id(str(r["ingB_id"])),
            "source": "phase13",
        })
        aid = to_ingredient_id(str(r["ingA_id"]))
        bid = to_ingredient_id(str(r["ingB_id"]))
        edge_rows.append({"source_id": iid, "target_id": aid, "edge_type": "HAS_INGREDIENT", "role": "A"})
        edge_rows.append({"source_id": iid, "target_id": bid, "edge_type": "HAS_INGREDIENT", "role": "B"})
        cat = str(r["category"]).strip()
        cat_id = to_category_id(cat)
        if cat_id not in seen_nodes:
            seen_nodes.add(cat_id)
            node_rows.append({"node_id": cat_id, "label": "Category", "name": cat, "source": "phase14"})
        edge_rows.append({
            "source_id": iid,
            "target_id": cat_id,
            "edge_type": "AFFECTS",
            "did": r.get("did"),
            "p_analytic": r.get("p_analytic"),
            "q_global": r.get("q_global"),
        })

    # ----- Pathway nodes + PATH -> CAT -----
    path_to_cats: Dict[str, List[str]] = {}
    if not pathway_cluster_info.empty and "cluster_id" in pathway_cluster_info.columns:
        top_terms_col = "top_terms" if "top_terms" in pathway_cluster_info.columns else pathway_cluster_info.columns[2]
        for _, row in pathway_cluster_info.iterrows():
            cid = row["cluster_id"]
            pid = to_pathway_id(f"cluster_{cid}", index=int(cid) if isinstance(cid, (int, float)) else None)
            if pid in seen_nodes:
                continue
            seen_nodes.add(pid)
            label = str(row.get("auto_label", pid))[:64]
            node_rows.append({"node_id": pid, "label": "Pathway", "name": label, "cluster_id": cid, "source": "pathway_cluster"})
            terms = row.get(top_terms_col, "")
            cats = _pathway_terms_to_categories(terms, pathway_bundles)
            path_to_cats[pid] = cats
            for c in cats:
                edge_rows.append({"source_id": pid, "target_id": c, "edge_type": "MAPS_TO_CATEGORY", "weight": 1.0})

    # ----- Gene nodes + GENE -> PATH (target clusters as pathway-like nodes) -----
    if not target_functional_clusters.empty and "cluster_id" in target_functional_clusters.columns:
        has_genes = "sample_genes" in target_functional_clusters.columns
        top_terms_col = "top_terms" if "top_terms" in target_functional_clusters.columns else None
        for _, row in target_functional_clusters.iterrows():
            tc_id = row["cluster_id"]
            path_id = to_pathway_id(f"tc_{tc_id}", index=int(tc_id) if isinstance(tc_id, (int, float)) else None)
            if path_id not in seen_nodes:
                seen_nodes.add(path_id)
                label = str(row.get("auto_label", path_id))[:64]
                node_rows.append({"node_id": path_id, "label": "Pathway", "name": label, "cluster_id": tc_id, "source": "target_cluster"})
            if top_terms_col:
                terms = row.get(top_terms_col, "")
                for c in _pathway_terms_to_categories(terms, pathway_bundles):
                    edge_rows.append({"source_id": path_id, "target_id": c, "edge_type": "MAPS_TO_CATEGORY", "weight": 1.0})
            if has_genes:
                raw = row["sample_genes"]
                genes = _parse_list_column(pd.Series([raw]))[0]
                for g in genes:
                    gid = to_gene_id(g)
                    if gid not in seen_nodes:
                        seen_nodes.add(gid)
                        node_rows.append({"node_id": gid, "label": "Gene", "name": g.upper(), "source": "target_cluster"})
                    edge_rows.append({"source_id": gid, "target_id": path_id, "edge_type": "IN_PATHWAY", "weight": 1.0})

    # ----- Compound layer (optional) -----
    cmp_ing: List[Tuple[str, str]] = []
    if ingredient_compound is not None and not ingredient_compound.empty:
        id_col = "ingredient_id" if "ingredient_id" in ingredient_compound.columns else ingredient_compound.columns[0]
        cmp_col = "compound_id" if "compound_id" in ingredient_compound.columns else next(
            (c for c in ingredient_compound.columns if "compound" in c.lower()), None
        )
        if cmp_col:
            for _, r in ingredient_compound.iterrows():
                ing = to_ingredient_id(str(r[id_col]))
                cmp_raw = r[cmp_col]
                cid = to_compound_id(str(cmp_raw))
                if cid not in seen_nodes:
                    seen_nodes.add(cid)
                    node_rows.append({"node_id": cid, "label": "Compound", "name": str(cmp_raw)[:64], "source": "metabolomics"})
                edge_rows.append({"source_id": ing, "target_id": cid, "edge_type": "HAS_COMPOUND", "weight": 1.0})
                cmp_ing.append((cid, ing))
    if compound_gene is not None and not compound_gene.empty:
        cmp_col = next((c for c in compound_gene.columns if "compound" in c.lower()), compound_gene.columns[0])
        gene_col = next((c for c in compound_gene.columns if "gene" in c.lower() or "symbol" in c.lower()), None)
        if gene_col:
            for _, r in compound_gene.iterrows():
                cid = to_compound_id(str(r[cmp_col]))
                gid = to_gene_id(str(r[gene_col]))
                if cid not in seen_nodes:
                    seen_nodes.add(cid)
                    node_rows.append({"node_id": cid, "label": "Compound", "name": str(r[cmp_col])[:64], "source": "compound_gene"})
                if gid not in seen_nodes:
                    seen_nodes.add(gid)
                    node_rows.append({"node_id": gid, "label": "Gene", "name": str(r[gene_col]).upper()[:32], "source": "compound_gene"})
                edge_rows.append({"source_id": cid, "target_id": gid, "edge_type": "TARGETS", "weight": 1.0})

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)
    logger.info("[%s] Mediation graph: %s nodes, %s edges", _ts(), len(nodes_df), len(edges_df))
    return nodes_df, edges_df
