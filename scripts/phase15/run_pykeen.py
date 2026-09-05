"""
Phase15: PyKEEN KG embeddings (TransE). Train/valid/test split; pipeline with training, validation, testing.
Outputs pykeen_eval.json, pykeen_top_predictions_compound_gene.csv. Works offline on exported triples.
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


def _load_edges_and_mapping(phase15_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    tables = phase15_dir / "tables"
    edges = pd.read_parquet(tables / "train_graph_edges.parquet")
    mapping = pd.read_parquet(tables / "node_id_mapping.parquet")
    if "node_id_int" not in mapping.columns and len(mapping.columns) >= 2:
        mapping = mapping.rename(columns={mapping.columns[0]: "node_id_str", mapping.columns[1]: "node_id_int"})
    int_to_str = mapping.set_index("node_id_int")["node_id_str"]
    return edges, int_to_str


def _triples_to_strings(edges: pd.DataFrame, int_to_str: pd.Series) -> np.ndarray:
    """Return (n, 3) array of strings [head, relation, tail]."""
    rows = []
    for _, row in edges.iterrows():
        h, r, t = int(row["head_int"]), str(row["relation"]).strip(), int(row["tail_int"])
        hs = int_to_str.get(h, str(h))
        ts = int_to_str.get(t, str(t))
        rows.append([str(hs), str(r), str(ts)])
    return np.array(rows, dtype=object)


def _build_and_split_triples_factories(
    triples: np.ndarray,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[Any, Any, Any]:
    """Build one TriplesFactory from all triples, then split via PyKEEN split(); return (training, validation, testing)."""
    import torch
    from pykeen.triples import TriplesFactory
    from pykeen.triples.splitting import split
    all_factory = TriplesFactory.from_labeled_triples(triples=triples)
    mapped = all_factory.mapped_triples
    if not isinstance(mapped, torch.Tensor):
        mapped = torch.as_tensor(mapped)
    # split(mapped_triples, ratios=[0.8, 0.1, 0.1]) -> train, test, val (order per PyKEEN docs)
    parts = split(mapped, ratios=[ratios[0], ratios[1], ratios[2]], random_state=seed)
    train_t, test_t, val_t = parts[0], parts[1], parts[2]
    entity_to_id = all_factory.entity_to_id
    relation_to_id = all_factory.relation_to_id
    training = TriplesFactory(mapped_triples=train_t, entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    testing = TriplesFactory(mapped_triples=test_t, entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    validation = TriplesFactory(mapped_triples=val_t, entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    return training, validation, testing


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase15 PyKEEN KG embeddings")
    parser.add_argument("--phase15-dir", type=str, required=True, help="Phase15 run dir")
    parser.add_argument("--model", type=str, default="TransE", choices=["TransE", "RotatE"])
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    phase15_dir = Path(args.phase15_dir).resolve()
    if not (phase15_dir / "tables" / "train_graph_edges.parquet").exists():
        logger.error("Run build_training_graph first. Expected tables/train_graph_edges.parquet")
        return 1

    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        logger.error("PyKEEN not installed. pip install pykeen")
        return 1

    edges, int_to_str = _load_edges_and_mapping(phase15_dir)
    relations = edges["relation"].unique().tolist()
    if not relations:
        raise ValueError("No relations in edges. Check build_training_graph output.")
    if edges.empty:
        raise ValueError("Empty train graph edges. Run build_training_graph first.")

    triples = _triples_to_strings(edges, int_to_str)
    if len(triples) == 0:
        raise ValueError("No triples. Run build_training_graph first.")
    training, validation, testing = _build_and_split_triples_factories(triples, ratios=(0.8, 0.1, 0.1), seed=args.seed)
    if training.num_triples == 0:
        raise ValueError("Train set is empty after split. Need more triples.")

    phase15_dir.mkdir(parents=True, exist_ok=True)
    models_dir = phase15_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = models_dir / "pykeen_model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = phase15_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = phase15_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training %s epochs=%d (train=%d valid=%d test=%d)", args.model, args.epochs, training.num_triples, validation.num_triples, testing.num_triples)
    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=args.model,
        model_kwargs=dict(embedding_dim=args.embedding_dim),
        optimizer_kwargs=dict(lr=0.01),
        training_kwargs=dict(num_epochs=args.epochs, batch_size=args.batch_size),
        random_seed=args.seed,
        device="cpu",
    )
    result.save_to_directory(artifact_dir)

    # Evaluate on test
    try:
        from pykeen.evaluation import RankBasedEvaluator
        evaluator = RankBasedEvaluator()
        metrics = result.evaluate(triples=testing.mapped_triples, evaluator=evaluator)
        mrr = float(metrics.get("mean_reciprocal_rank", 0.0))
        hits_at_10 = float(metrics.get("hits_at_10", 0.0))
    except Exception as e:
        logger.warning("Evaluation failed: %s", e)
        mrr = 0.0
        hits_at_10 = 0.0

    # Top predictions for compound->gene
    import torch
    rel_to_id = training.relation_to_id
    rel_id_cg = rel_to_id.get("TARGETS_GENE", rel_to_id.get("TARGETS", None))
    existing_cg = set(
        zip(
            edges[edges["relation"] == "TARGETS_GENE"]["head_int"].astype(int),
            edges[edges["relation"] == "TARGETS_GENE"]["tail_int"].astype(int),
        )
    ) if "TARGETS_GENE" in edges["relation"].values else set()

    top_rows = []
    if rel_id_cg is not None:
        cg_edges = edges[edges["relation"].isin(["TARGETS_GENE", "TARGETS"])]
        compounds = cg_edges["head_int"].drop_duplicates().astype(int).tolist()[:500]
        genes = cg_edges["tail_int"].drop_duplicates().astype(int).tolist()[:300]
        model = result.model
        for h in compounds:
            for t in genes:
                if (h, t) in existing_cg:
                    continue
                try:
                    head_str = str(int_to_str.get(h, h))
                    tail_str = str(int_to_str.get(t, t))
                    hid = training.entity_to_id.get(head_str, None)
                    tid = training.entity_to_id.get(tail_str, None)
                    if hid is None or tid is None:
                        continue
                    if hasattr(model, "score_hrt"):
                        batch = torch.tensor([[hid, rel_id_cg, tid]], dtype=torch.long)
                        s = float(model.score_hrt(batch).item())
                        top_rows.append({"compound_id": head_str, "gene_symbol": tail_str, "score": s})
                except Exception:
                    continue
    top_df = pd.DataFrame(top_rows).sort_values("score", ascending=False).head(500)
    top_df.to_csv(predictions_dir / "pykeen_top_predictions_compound_gene.csv", index=False)

    report = {
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "num_entities": training.num_entities,
        "num_relations": training.num_relations,
        "mrr": mrr,
        "hits_at_10": hits_at_10,
        "n_train": training.num_triples,
        "n_valid": validation.num_triples,
        "n_test": testing.num_triples,
        "n_novel_predicted_compound_gene": len(top_df),
        "seed": args.seed,
    }
    with open(reports_dir / "pykeen_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(reports_dir / "pykeen_summary.md", "w", encoding="utf-8") as f:
        f.write(f"# PyKEEN report\n- MRR: {mrr:.4f}\n- Hits@10: {hits_at_10:.4f}\n- Top predictions: {len(top_df)}\n")
    logger.info("PyKEEN done. MRR=%.4f Hits@10=%.4f", mrr, hits_at_10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
