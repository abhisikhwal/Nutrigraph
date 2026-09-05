"""
Phase15: Node2Vec baseline embeddings + link prediction eval (ROC-AUC, Hits@K).
Reads training graph from phase15 out_dir; writes embeddings + eval report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(42)


def _load_graph_tables(phase15_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = phase15_dir / "tables"
    nodes = pd.read_parquet(tables / "train_graph_nodes.parquet")
    edges = pd.read_parquet(tables / "train_graph_edges.parquet")
    mapping = pd.read_parquet(tables / "node_id_mapping.parquet")
    return nodes, edges, mapping


def build_nx_graph(edges: pd.DataFrame, weighted: bool = False):
    """Build undirected NetworkX graph from edges (head_int, tail_int)."""
    import networkx as nx
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(int(row["head_int"]), int(row["tail_int"]), weight=1.0)
    return G


def run_node2vec_embedding(
    G,
    dimensions: int = 128,
    walk_length: int = 80,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    workers: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, bool]:
    """Return (emb, used_placeholder). When placeholder=True do not report ROC-AUC/Hits@K."""
    try:
        from node2vec import Node2Vec
    except ImportError:
        logger.warning("node2vec not installed. pip install node2vec. Using degree-based placeholder.")
        n = max(G.nodes()) + 1 if G.nodes() else 0
        emb = np.zeros((n, dimensions))
        for i in range(n):
            if G.has_node(i):
                emb[i] = np.random.RandomState(seed + i).randn(dimensions) * 0.01
        deg = dict(G.degree())
        for i in range(n):
            if i in deg:
                emb[i] += deg[i] * 0.1
        return emb, True

    n2v = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=workers, seed=seed)
    model = n2v.fit()
    node_ids = sorted(G.nodes())
    emb = np.zeros((max(node_ids) + 1, dimensions))
    for i in node_ids:
        emb[i] = model.wv[str(i)]
    return emb, False


def compound_gene_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Filter edges to relation TARGETS_GENE (compound->gene)."""
    if "relation" not in edges.columns:
        return pd.DataFrame()
    return edges[edges["relation"] == "TARGETS_GENE"][["head_int", "tail_int"]].drop_duplicates()


def train_test_split_edges(
    cg_edges: pd.DataFrame,
    train_frac: float = 0.85,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split compound-gene edges into train and test."""
    if cg_edges.empty:
        return pd.DataFrame(), pd.DataFrame()
    idx = np.random.RandomState(seed).permutation(len(cg_edges))
    n = len(cg_edges)
    n_train = int(n * train_frac)
    train_df = cg_edges.iloc[idx[:n_train]]
    test_df = cg_edges.iloc[idx[n_train:]]
    return train_df, test_df


def evaluate_link_prediction(
    emb: np.ndarray,
    train_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    k_values: List[int] = [10, 50],
) -> Dict[str, Any]:
    """Dot-product scoring; ROC-AUC, AP, Hits@K. Filter train from candidates."""
    if test_edges.empty:
        return {"roc_auc": 0.0, "average_precision": 0.0, "hits_at_k": {}}
    train_set = set(zip(train_edges["head_int"].astype(int), train_edges["tail_int"].astype(int)))
    scores_pos = []
    for _, row in test_edges.iterrows():
        h, t = int(row["head_int"]), int(row["tail_int"])
        if h < emb.shape[0] and t < emb.shape[0]:
            scores_pos.append(float(np.dot(emb[h], emb[t])))
    scores_pos = np.array(scores_pos)
    # Negatives: random (h, t) not in train and not in test
    all_heads = set(train_edges["head_int"].astype(int)) | set(test_edges["head_int"].astype(int))
    all_tails = set(train_edges["tail_int"].astype(int)) | set(test_edges["tail_int"].astype(int))
    n_neg = min(len(scores_pos) * 5, 5000)
    rng = np.random.RandomState(43)
    scores_neg = []
    for _ in range(n_neg):
        h = rng.choice(list(all_heads)) if all_heads else 0
        t = rng.choice(list(all_tails)) if all_tails else 0
        if (h, t) in train_set or h >= emb.shape[0] or t >= emb.shape[0]:
            continue
        scores_neg.append(float(np.dot(emb[h], emb[t])))
    scores_neg = np.array(scores_neg) if scores_neg else np.array([0.0])
    y_true = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    y_score = np.concatenate([scores_pos, scores_neg])
    from sklearn.metrics import roc_auc_score, average_precision_score
    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    hits_at_k = {}
    for K in k_values:
        # For each test (h,t), rank t among negatives; count how often t in top-K
        hits = 0
        for _, row in test_edges.iterrows():
            h, t = int(row["head_int"]), int(row["tail_int"])
            if h >= emb.shape[0] or t >= emb.shape[0]:
                continue
            cand = list(all_tails - {t})[:500] + [t]
            scores = [np.dot(emb[h], emb[c]) for c in cand]
            rank = sorted(range(len(scores)), key=lambda i: -scores[i])
            if t in cand and cand.index(t) in [rank[i] for i in range(min(K, len(rank)))]:
                hits += 1
        hits_at_k[K] = hits / max(1, len(test_edges))
    return {"roc_auc": float(roc_auc), "average_precision": float(ap), "hits_at_k": hits_at_k}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase15 Node2Vec embeddings + eval")
    parser.add_argument("--phase15-dir", type=str, required=True, help="Phase15 run dir (contains tables/)")
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--walk-length", type=int, default=80)
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    phase15_dir = Path(args.phase15_dir).resolve()
    if not (phase15_dir / "tables" / "train_graph_edges.parquet").exists():
        logger.error("Phase15 dir must contain tables from build_training_graph. Run that first.")
        return 1

    nodes, edges, mapping = _load_graph_tables(phase15_dir)
    cg_edges = compound_gene_edges(edges)
    if cg_edges.empty:
        logger.warning("No TARGETS_GENE edges; link prediction eval will be N/A")
    train_cg, test_cg = train_test_split_edges(cg_edges, train_frac=0.85, seed=args.seed)

    G = build_nx_graph(edges)
    logger.info("Graph nodes=%d edges=%d", G.number_of_nodes(), G.number_of_edges())
    emb, used_placeholder = run_node2vec_embedding(
        G,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        seed=args.seed,
    )
    n_nodes = emb.shape[0]
    phase15_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = phase15_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = phase15_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = phase15_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings as parquet (node_id_int, dim_0, dim_1, ...)
    emb_df = pd.DataFrame(emb, columns=[f"dim_{i}" for i in range(emb.shape[1])])
    emb_df.insert(0, "node_id_int", range(len(emb_df)))
    emb_df.to_parquet(embeddings_dir / "node2vec_embeddings.parquet", index=False)
    emb_df.to_csv(embeddings_dir / "node2vec_embeddings.csv", index=False)

    if used_placeholder:
        eval_result = {"roc_auc": None, "average_precision": None, "hits_at_k": None}
        report = {
            "placeholder": True,
            "message": "PLACEHOLDER: node2vec not installed; metrics not reported. Install with: pip install node2vec",
            "n_nodes": n_nodes,
            "embedding_dim": args.dimensions,
            "roc_auc": None,
            "average_precision": None,
            "hits_at_k": None,
            "n_compound_gene_train": int(len(train_cg)),
            "n_compound_gene_test": int(len(test_cg)),
            "seed": args.seed,
        }
    else:
        eval_result = evaluate_link_prediction(emb, train_cg, test_cg, k_values=[10, 50])
        report = {
            "placeholder": False,
            "n_nodes": n_nodes,
            "embedding_dim": args.dimensions,
            "roc_auc": eval_result["roc_auc"],
            "average_precision": eval_result["average_precision"],
            "hits_at_k": eval_result["hits_at_k"],
            "n_compound_gene_train": int(len(train_cg)),
            "n_compound_gene_test": int(len(test_cg)),
            "seed": args.seed,
        }
    with open(reports_dir / "node2vec_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Neighbor examples: for a few nodes, top 5 by dot product
    if "node_id_int" in mapping.columns and "node_id_str" in mapping.columns:
        id_to_str = mapping.set_index("node_id_int")["node_id_str"]
    elif len(mapping.columns) >= 2:
        id_to_str = mapping.set_index(mapping.columns[1])[mapping.columns[0]]
    else:
        id_to_str = pd.Series(index=range(n_nodes), data=[f"n{i}" for i in range(n_nodes)])
    neighbor_rows = []
    for node_int in list(nodes["node_id_int"].dropna().astype(int).unique())[:10]:
        if node_int >= n_nodes:
            continue
        sim = emb @ emb[node_int]
        sim[node_int] = -1e9
        top5 = np.argsort(-sim)[:5]
        for j in top5:
            neighbor_rows.append({
                "node_id_int": node_int,
                "node_id_str": id_to_str.get(node_int, str(node_int)),
                "neighbor_id_int": int(j),
                "neighbor_id_str": id_to_str.get(j, str(j)),
                "dot_score": float(sim[j]),
            })
    neighbor_df = pd.DataFrame(neighbor_rows)
    neighbor_df.to_csv(reports_dir / "node2vec_neighbors_examples.csv", index=False)

    if used_placeholder:
        summary = [
            "# Node2Vec report (PLACEHOLDER)",
            "- node2vec not installed; embeddings are degree-based placeholder.",
            "- ROC-AUC / Hits@K are not reported. Install: pip install node2vec",
        ]
        logger.info("Node2Vec PLACEHOLDER (node2vec not installed). Install: pip install node2vec")
    else:
        summary = [
            "# Node2Vec report",
            f"- ROC-AUC: {report['roc_auc']:.4f}",
            f"- AP: {report['average_precision']:.4f}",
            f"- Hits@10: {report['hits_at_k'].get(10, 0):.4f}",
            f"- Hits@50: {report['hits_at_k'].get(50, 0):.4f}",
        ]
        logger.info("Node2Vec done. ROC-AUC=%.4f Hits@10=%.4f", report["roc_auc"], report["hits_at_k"].get(10, 0))
    with open(reports_dir / "node2vec_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
