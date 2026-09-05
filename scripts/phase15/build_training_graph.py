"""
Phase15: Build training graph from Phase14 snapshot + canonical CSVs.
Outputs train_graph_edges.parquet, train_graph_nodes.parquet, mapping tables, report JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.phase15.graph_io import (
    load_neo4j_export,
    load_compound_gene,
    load_ingredient_compound,
    load_pair_category_mediation,
    build_ingredient_category_edges,
    get_repo_root,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ensure_repo_path(path_str: str, run_dir: Path, repo_root: Path) -> Path:
    if Path(path_str).is_absolute():
        return Path(path_str)
    if (run_dir / path_str).exists():
        return run_dir / path_str
    return repo_root / path_str


def build_unified_nodes_edges(
    run_dir: Path,
    repo_root: Path,
    include_compound_gene: bool = True,
    include_ingredient_compound: bool = True,
    include_ingredient_category: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build node table (node_id_str, node_type) and edge table (head, relation, tail) with string IDs.
    Then caller will map to integer ids.
    """
    nodes_list: List[Dict[str, str]] = []
    edges_list: List[Dict[str, str]] = []

    # 1) Neo4j nodes: use :ID and :LABEL
    try:
        neo_n, neo_e = load_neo4j_export(run_dir)
        id_col = ":ID" if ":ID" in neo_n.columns else neo_n.columns[0]
        label_col = ":LABEL" if ":LABEL" in neo_n.columns else "label"
        for _, row in neo_n.iterrows():
            nid = row.get(id_col)
            if pd.isna(nid):
                continue
            nodes_list.append({"node_id_str": str(nid).strip(), "node_type": str(row.get(label_col, "Node")).strip()})
        # Neo4j edges: (START_ID, TYPE, END_ID)
        start_col = ":START_ID" if ":START_ID" in neo_e.columns else "source_id"
        end_col = ":END_ID" if ":END_ID" in neo_e.columns else "target_id"
        type_col = ":TYPE" if ":TYPE" in neo_e.columns else "edge_type"
        for _, row in neo_e.iterrows():
            h, t = row.get(start_col), row.get(end_col)
            r = row.get(type_col, "REL")
            if pd.isna(h) or pd.isna(t):
                continue
            edges_list.append({"head": str(h).strip(), "relation": str(r).strip(), "tail": str(t).strip()})
    except FileNotFoundError as e:
        logger.warning("Neo4j export not found: %s", e)

    # 2) Compound -> Gene
    if include_compound_gene:
        cg = load_compound_gene(repo_root)
        if not cg.empty:
            for _, row in cg.iterrows():
                c, g = row["compound_id"], row["gene_symbol"]
                if pd.isna(c) or pd.isna(g):
                    continue
                c, g = str(c).strip(), str(g).strip()
                nodes_list.append({"node_id_str": c, "node_type": "Compound"})
                nodes_list.append({"node_id_str": g, "node_type": "Gene"})
                edges_list.append({"head": c, "relation": "TARGETS_GENE", "tail": g})

    # 3) Ingredient -> Compound
    if include_ingredient_compound:
        ic = load_ingredient_compound(repo_root)
        if not ic.empty:
            for _, row in ic.iterrows():
                ing, c = row["ingredient_id"], row["compound_id"]
                if pd.isna(ing) or pd.isna(c):
                    continue
                ing, c = str(ing).strip(), str(c).strip()
                nodes_list.append({"node_id_str": ing, "node_type": "Ingredient"})
                nodes_list.append({"node_id_str": c, "node_type": "Compound"})
                edges_list.append({"head": ing, "relation": "HAS_COMPOUND", "tail": c})

    # 4) Ingredient -> Category (from pair_category_mediation)
    if include_ingredient_category:
        pair_med = load_pair_category_mediation(run_dir)
        ing_cat = build_ingredient_category_edges(pair_med)
        if not ing_cat.empty:
            for _, row in ing_cat.iterrows():
                ing, cat = row["ingredient_id"], row["category"]
                if pd.isna(ing) or pd.isna(cat):
                    continue
                ing, cat = str(ing).strip(), str(cat).strip()
                cat_id = f"CAT_{cat}" if not cat.startswith("CAT_") else cat
                nodes_list.append({"node_id_str": ing, "node_type": "Ingredient"})
                nodes_list.append({"node_id_str": cat_id, "node_type": "Category"})
                edges_list.append({"head": ing, "relation": "ASSOCIATED_CATEGORY", "tail": cat_id})

    nodes_df = pd.DataFrame(nodes_list).drop_duplicates(subset=["node_id_str"])
    edges_df = pd.DataFrame(edges_list).drop_duplicates()
    return nodes_df, edges_df, pd.DataFrame(nodes_list)


def assign_integer_ids(nodes_df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Assign deterministic integer id per node_id_str. Returns nodes with node_id_int, and mapping series."""
    nodes_df = nodes_df.drop_duplicates(subset=["node_id_str"]).sort_values("node_id_str").reset_index(drop=True)
    nodes_df["node_id_int"] = range(len(nodes_df))
    mapping = nodes_df.set_index("node_id_str")["node_id_int"]
    return nodes_df, mapping


def build_integer_edges(edges_df: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    """Map head/tail strings to integer ids."""
    edges = edges_df.copy()
    edges["head_int"] = edges["head"].map(mapping)
    edges["tail_int"] = edges["tail"].map(mapping)
    edges = edges.dropna(subset=["head_int", "tail_int"]).astype({"head_int": int, "tail_int": int})
    return edges


def compute_overlap_metrics(edges_df: pd.DataFrame, repo_root: Path) -> Dict[str, Any]:
    """overlap_vs_ic, overlap_vs_cg, n_overlap from compound_gene and ingredient_compound."""
    from src.phase15.graph_io import load_compound_gene, load_ingredient_compound
    ic = load_ingredient_compound(repo_root)
    cg = load_compound_gene(repo_root)
    if ic.empty or cg.empty:
        return {"n_overlap": 0, "overlap_vs_ic": 0.0, "overlap_vs_cg": 0.0}
    compounds_ic = set(ic["compound_id"].dropna().astype(str).str.strip())
    compounds_cg = set(cg["compound_id"].dropna().astype(str).str.strip())
    n_overlap = len(compounds_ic & compounds_cg)
    n_ic = len(compounds_ic)
    n_cg = len(compounds_cg)
    return {
        "n_overlap": n_overlap,
        "overlap_vs_ic": n_overlap / n_ic if n_ic else 0.0,
        "overlap_vs_cg": n_overlap / n_cg if n_cg else 0.0,
        "n_compounds_ic": n_ic,
        "n_compounds_cg": n_cg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase15 training graph")
    parser.add_argument("--run-dir", type=str, default=None, help="Phase14 run or snapshot dir; default snapshot")
    parser.add_argument("--out-dir", type=str, default=None, help="Output dir; default phase15_embeddings/<timestamp>")
    parser.add_argument("--repo-root", type=Path, default=get_repo_root())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()

    # Default run_dir = Phase14 snapshot
    if args.run_dir is None:
        snapshot = repo_root / "data" / "processed" / "milestones" / "phase14" / "v1_working_2026-02-19" / "phase14_20260219_204918"
        if snapshot.exists():
            run_dir = snapshot
        else:
            run_dir = repo_root / "data" / "processed" / "phase14_mediation" / "phase14_20260219_204918"
        if not run_dir.exists():
            logger.error("No default Phase14 run/snapshot found. Set --run-dir.")
            return 1
    else:
        run_dir = (repo_root / args.run_dir.replace("\\", "/")).resolve()
        if not run_dir.exists():
            logger.error("Run dir not found: %s", run_dir)
            return 1

    if args.out_dir is None:
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = repo_root / "data" / "processed" / "phase15_embeddings" / f"phase15_{stamp}"
    else:
        out_dir = (repo_root / args.out_dir.replace("\\", "/")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run dir: %s", run_dir)
    logger.info("Out dir: %s", out_dir)

    nodes_df, edges_df, _ = build_unified_nodes_edges(run_dir, repo_root)
    if nodes_df.empty:
        logger.error("No nodes produced")
        return 1

    nodes_df, mapping = assign_integer_ids(nodes_df, seed=args.seed)
    edges_int = build_integer_edges(edges_df, mapping)

    # Counts per relation
    relation_counts = edges_int.groupby("relation").size().to_dict()
    relation_counts = {str(k): int(v) for k, v in relation_counts.items()}

    # Top 20 nodes by degree
    degree = edges_int.groupby("head_int").size().add(edges_int.groupby("tail_int").size(), fill_value=0).astype(int)
    top_degree = degree.nlargest(20)
    node_id_str = nodes_df.set_index("node_id_int")["node_id_str"]
    top_20 = [{"node_id_int": int(i), "node_id_str": node_id_str.get(i, ""), "degree": int(d)} for i, d in top_degree.items()]

    overlap = compute_overlap_metrics(edges_df, repo_root)

    # Save
    nodes_df.to_parquet(tables_dir / "train_graph_nodes.parquet", index=False)
    edges_int.to_parquet(tables_dir / "train_graph_edges.parquet", index=False)
    mapping.reset_index().to_parquet(tables_dir / "node_id_mapping.parquet", index=False)
    edges_df.to_parquet(tables_dir / "train_graph_edges_str.parquet", index=False)

    report = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_nodes": int(len(nodes_df)),
        "n_edges": int(len(edges_int)),
        "relation_counts": relation_counts,
        "overlap_vs_ic": overlap["overlap_vs_ic"],
        "overlap_vs_cg": overlap["overlap_vs_cg"],
        "n_overlap": overlap["n_overlap"],
        "top_20_nodes_by_degree": top_20,
        "seed": args.seed,
    }
    with open(reports_dir / "build_training_graph_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_lines = [
        "# Build training graph report",
        f"- **Run dir**: {run_dir}",
        f"- **Nodes**: {report['n_nodes']}",
        f"- **Edges**: {report['n_edges']}",
        f"- **Overlap (vs IC)**: {report['overlap_vs_ic']:.4f}",
        f"- **Overlap (vs CG)**: {report['overlap_vs_cg']:.4f}",
        f"- **n_overlap**: {report['n_overlap']}",
        "",
        "## Relation counts",
    ]
    for r, c in relation_counts.items():
        summary_lines.append(f"- {r}: {c}")
    summary_lines.append("")
    summary_lines.append("## Top 20 nodes by degree")
    for x in top_20[:20]:
        summary_lines.append(f"- {x['node_id_str']} (int={x['node_id_int']}): degree {x['degree']}")
    with open(reports_dir / "build_training_graph_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    logger.info("Nodes %d edges %d relations %s", report["n_nodes"], report["n_edges"], relation_counts)
    print("Build training graph done. Report:", reports_dir / "build_training_graph_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
