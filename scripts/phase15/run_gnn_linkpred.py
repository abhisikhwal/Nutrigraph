"""
Phase15: GNN link prediction (PyTorch Geometric). Compound->gene task.
Stable training: reduced lr, weight_decay, gradient clipping; early stop on NaN; safe eval (no NaN in scores).
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


def _load_graph(phase15_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = phase15_dir / "tables"
    edges = pd.read_parquet(tables / "train_graph_edges.parquet")
    nodes = pd.read_parquet(tables / "train_graph_nodes.parquet")
    mapping = pd.read_parquet(tables / "node_id_mapping.parquet")
    return edges, nodes, mapping


def _try_torch_geometric():
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
        from torch_geometric.utils import negative_sampling
        return True, torch, Data, SAGEConv, negative_sampling
    except ImportError:
        return False, None, None, None, None


def build_pyg_data(edges: pd.DataFrame, n_nodes: int) -> Any:
    ok, torch, Data, SAGEConv, negative_sampling = _try_torch_geometric()
    if not ok:
        raise ImportError("PyTorch Geometric not installed. pip install torch_geometric")
    edge_index = torch.tensor(
        edges[["head_int", "tail_int"]].values.T.astype(np.int64),
        dtype=torch.long,
    )
    return Data(edge_index=edge_index, num_nodes=n_nodes)


def train_gnn(
    data,
    relation_edges: pd.DataFrame,
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[Any, np.ndarray]:
    import torch
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import negative_sampling
    torch.manual_seed(seed)
    np.random.seed(seed)

    class LinkPredictor(torch.nn.Module):
        def __init__(self, in_channels, hidden_dim, num_layers):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.embed_dim = hidden_dim

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index).relu()
            return x

    n_nodes = data.num_nodes
    in_channels = 64
    x = torch.randn(n_nodes, in_channels) * 0.1
    data.x = x
    model = LinkPredictor(in_channels, hidden_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)
    data = data.to(device)
    pos_edges = relation_edges[["head_int", "tail_int"]].values.astype(np.int64)
    pos_edges = torch.tensor(pos_edges, dtype=torch.long, device=device).t()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        neg_edges = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=n_nodes,
            num_neg_samples=pos_edges.size(1),
        )
        z = model(data.x, data.edge_index)
        pos_score = (z[pos_edges[0]] * z[pos_edges[1]]).sum(dim=1)
        neg_score = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean() - torch.log(1 - torch.sigmoid(neg_score) + 1e-8).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        loss_val = loss.item()
        if not np.isfinite(loss_val):
            logger.warning("Epoch %d loss is NaN/inf; stopping. Try lower lr (e.g. 5e-4).", epoch + 1)
            break
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d loss %.4f", epoch + 1, loss_val)

    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index).cpu().numpy()
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    return model, emb


def evaluate_gnn(
    emb: np.ndarray,
    train_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    existing_set: set,
    k_values: List[int] = [10, 50],
) -> Dict[str, Any]:
    """ROC-AUC, AP, Hits@K. If y_true has only one class, roc_auc is undefined."""
    if test_edges.empty:
        return {"roc_auc": None, "average_precision": None, "hits_at_k": {k: 0.0 for k in k_values}, "roc_auc_undefined": True}
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    pos_scores = []
    for _, row in test_edges.iterrows():
        h, t = int(row["head_int"]), int(row["tail_int"])
        if h < emb.shape[0] and t < emb.shape[0]:
            s = np.dot(emb[h], emb[t])
            pos_scores.append(float(s) if np.isfinite(s) else 0.0)
    pos_scores = np.array(pos_scores)
    pos_scores = np.nan_to_num(pos_scores, nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.RandomState(43)
    all_tails = set(train_edges["tail_int"].astype(int)) | set(test_edges["tail_int"].astype(int))
    neg_scores = []
    for _ in range(min(len(pos_scores) * 5, 5000)):
        h = rng.randint(0, emb.shape[0])
        t = rng.choice(list(all_tails)) if all_tails else 0
        if (h, t) in existing_set:
            continue
        if t < emb.shape[0]:
            s = np.dot(emb[h], emb[t])
            neg_scores.append(float(s) if np.isfinite(s) else 0.0)
    neg_scores = np.array(neg_scores) if neg_scores else np.array([0.0])
    neg_scores = np.nan_to_num(neg_scores, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    y_score = np.concatenate([pos_scores, neg_scores])
    roc_auc_undefined = False
    roc_auc = None
    ap = None
    if len(np.unique(y_true)) > 1:
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
            ap = float(average_precision_score(y_true, y_score))
        except Exception:
            roc_auc_undefined = True
    else:
        roc_auc_undefined = True
    hits_at_k = {}
    for K in k_values:
        hits = 0
        for _, row in test_edges.iterrows():
            h, t = int(row["head_int"]), int(row["tail_int"])
            if h >= emb.shape[0] or t >= emb.shape[0]:
                continue
            scores = np.nan_to_num(emb[h] @ emb.T, nan=0.0, posinf=0.0, neginf=0.0)
            rank = np.argsort(-scores)
            if t in rank[:K]:
                hits += 1
        hits_at_k[K] = hits / max(1, len(test_edges))
    return {"roc_auc": roc_auc, "average_precision": ap, "hits_at_k": hits_at_k, "roc_auc_undefined": roc_auc_undefined}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase15 GNN link prediction")
    parser.add_argument("--phase15-dir", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-debug-mode", action="store_true")
    args = parser.parse_args()
    phase15_dir = Path(args.phase15_dir).resolve()
    if not (phase15_dir / "tables" / "train_graph_edges.parquet").exists():
        logger.error("Run build_training_graph first.")
        return 1

    ok, torch, Data, SAGEConv, negative_sampling = _try_torch_geometric()
    if not ok:
        logger.error("PyTorch Geometric not installed. pip install torch torch_geometric")
        return 1

    edges, nodes, mapping = _load_graph(phase15_dir)
    cg = edges[edges["relation"] == "TARGETS_GENE"][["head_int", "tail_int"]].drop_duplicates()
    if cg.empty:
        logger.warning("No TARGETS_GENE edges.")
    n_nodes = int(nodes["node_id_int"].max()) + 1
    data = build_pyg_data(edges, n_nodes)
    idx = np.random.RandomState(args.seed).permutation(len(cg))
    n_train = int(len(cg) * 0.85)
    train_cg = cg.iloc[idx[:n_train]]
    test_cg = cg.iloc[idx[n_train:]]
    existing_set = set(zip(train_cg["head_int"].astype(int), train_cg["tail_int"].astype(int)))

    epochs = 2 if args.small_debug_mode else args.epochs
    logger.info("GNN training epochs=%d lr=%s", epochs, args.lr)
    model, emb = train_gnn(
        data,
        train_cg,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=epochs,
        lr=args.lr,
        weight_decay=5e-4,
        max_grad_norm=1.0,
        seed=args.seed,
    )
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    eval_metrics = evaluate_gnn(emb, train_cg, test_cg, existing_set, k_values=[10, 50])

    embeddings_dir = phase15_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = phase15_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = phase15_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    emb_df = pd.DataFrame(emb, columns=[f"dim_{i}" for i in range(emb.shape[1])])
    emb_df.insert(0, "node_id_int", range(len(emb_df)))
    emb_df.to_parquet(embeddings_dir / "gnn_embeddings.parquet", index=False)

    compounds = set(cg["head_int"].astype(int))
    genes = set(cg["tail_int"].astype(int))
    rows = []
    for h in list(compounds)[:300]:
        for t in list(genes)[:100]:
            if (h, t) in existing_set:
                continue
            if h < emb.shape[0] and t < emb.shape[0]:
                s = float(np.dot(emb[h], emb[t]))
                if np.isfinite(s):
                    rows.append({"head_int": h, "tail_int": t, "score": s})
    top_df = pd.DataFrame(rows).sort_values("score", ascending=False).head(500)
    if "node_id_str" in mapping.columns:
        str_map = mapping.set_index("node_id_int")["node_id_str"]
        top_df["head_str"] = top_df["head_int"].map(str_map)
        top_df["tail_str"] = top_df["tail_int"].map(str_map)
    elif len(mapping.columns) >= 2:
        str_map = mapping.set_index(mapping.columns[1])[mapping.columns[0]]
        top_df["head_str"] = top_df["head_int"].map(str_map)
        top_df["tail_str"] = top_df["tail_int"].map(str_map)
    top_df.to_csv(predictions_dir / "gnn_top_predictions_compound_gene.csv", index=False)

    report = {
        "roc_auc": eval_metrics["roc_auc"],
        "average_precision": eval_metrics["average_precision"],
        "hits_at_k": eval_metrics["hits_at_k"],
        "roc_auc_undefined": eval_metrics.get("roc_auc_undefined", False),
        "n_train_cg": int(len(train_cg)),
        "n_test_cg": int(len(test_cg)),
        "embedding_dim": args.hidden_dim,
        "epochs": epochs,
        "seed": args.seed,
    }
    with open(reports_dir / "gnn_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    summary_lines = ["# GNN report"]
    if report.get("roc_auc_undefined"):
        summary_lines.append("- ROC-AUC: undefined (single class in eval)")
    else:
        summary_lines.append(f"- ROC-AUC: {report['roc_auc']:.4f}")
    if report.get("average_precision") is not None:
        summary_lines.append(f"- AP: {report['average_precision']:.4f}")
    summary_lines.append(f"- Hits@10: {report['hits_at_k'].get(10, 0):.4f}")
    with open(reports_dir / "gnn_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    logger.info("GNN done. ROC-AUC=%s", report["roc_auc"] if report["roc_auc"] is not None else "undefined")
    return 0


if __name__ == "__main__":
    sys.exit(main())
