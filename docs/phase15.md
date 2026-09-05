# Phase15: Embeddings, Link Prediction, Causal Mediation, Demo UI

Phase15 builds on the Phase14 biological mediation layer with **research-grade embeddings**, **link prediction** (compound→gene), **causal mediation** analysis, and a **demo UI** for ingredient-pair exploration.

## What we did

1. **Graph extraction & training datasets**  
   We load the Phase14 snapshot (Neo4j export) and optionally canonical ingredient–compound and compound–gene CSVs to build a unified multi-relational graph. We produce integer node IDs, edge tables with relation types (e.g. `TARGETS_GENE`, `HAS_COMPOUND`, `ASSOCIATED_CATEGORY`), and report overlap metrics (`overlap_vs_ic`, `overlap_vs_cg`, `n_overlap`) and top nodes by degree.

2. **Node2Vec baseline**  
   We train Node2Vec on the (undirected) training graph and save node embeddings. We evaluate **compound→gene** link prediction on held-out edges using dot-product scoring and report **ROC-AUC**, **AP**, and **Hits@K** (K=10, 50). Outputs: `node2vec_embeddings.parquet`, `node2vec_eval.json`, `node2vec_neighbors_examples.csv`.

3. **PyKEEN KG embeddings**  
   We train TransE (and optionally RotatE) on the multi-relational triples and evaluate on held-out compound→gene triples. We report **MRR** and **Hits@10** and export **top predicted compound→gene** edges (with filtering to avoid leaking existing edges). Outputs: `pykeen_model_artifacts/`, `pykeen_eval.json`, `pykeen_top_predictions_compound_gene.csv`.

4. **GNN link prediction**  
   We use PyTorch Geometric (GraphSAGE) for link prediction with negative sampling, train/val/test split, and report **ROC-AUC**, **AP**, **Hits@K**. Runs on CPU with a deterministic seed; `--small-debug-mode` runs a few epochs for quick checks. Outputs: `gnn_embeddings.parquet`, `gnn_eval.json`, `gnn_top_predictions_compound_gene.csv`.

5. **Validation against external sources**  
   If ChEMBL or DrugBank target files exist locally (e.g. `data/external/chembl_targets.csv`, `drugbank_targets.csv`), we validate predicted compound→gene edges by InChIKey/gene match and write `validated_predictions.csv` and `unmatched_predictions.csv`. If files are missing, we report the exact missing paths and continue.

6. **Causal mediation (Phase15Causal)**  
   We replace the simple “propagated_pathway_score” usage with a **probabilistic mediation** setup: exposure (e.g. `shared_compounds_count` or `dose_proxy_AB`) → mediator (`propagated_pathway_score`) → outcome (`did`). We fit a simple SEM (mediator ~ exposure; outcome ~ exposure + mediator), run **bootstrap** for CI and p-values, and write per-category and global summaries. We include **sanity warnings** (e.g. non-independent samples) and state clearly that this is **hypothesis generation only**, not medical advice.

7. **Agent + Streamlit UI**  
   - **Agent** (`agent_explain_pair.py`): given `ingA_id`, `ingB_id`, we pull pair_category_mediation and evidence trails (shared compounds → shared genes → categories) and output a **structured JSON** and a **short plain-text report**. No API calls if offline.  
   - **Streamlit** (`streamlit run streamlit_app/app.py`): dropdowns for Ingredient A/B, display of mechanistic scores and causal mediation estimates, top shared/predicted genes, optional **pyvis** mini-graph, and an **export** button for the evidence JSON.

8. **Master runner**  
   `run_phase15.py` runs steps (1)–(6) in order, with flags to skip node2vec, pykeen, gnn, validate, or causal. All outputs go under `data/processed/phase15_embeddings/<timestamp>/` (reports, models, tables, embeddings, predictions, causal). It writes a **Phase15 Readiness** summary (`phase15_readiness.json`) with counts and overlap metrics at the end.

## What each model provides

| Component    | Purpose |
|-------------|---------|
| **Node2Vec** | Fast baseline embeddings; good for neighbor lookup and simple link scoring. |
| **PyKEEN**   | Multi-relational KG embeddings; captures relation types; good for ranking new compound–gene links. |
| **GNN**      | Graph structure–aware link prediction; uses neighborhood; good for compound→gene with local graph context. |
| **Causal**   | Mediation effect estimates and uncertainty; supports hypothesis generation for “pathway mediates ingredient–outcome.” |
| **Agent/UI** | Human-readable explanations and exploration without writing code. |

## How to cite / describe in a PhD pitch

- **Phase14** provides the biological mediation layer (ingredient→compound→gene→pathway→category) and propagation diagnostics.  
- **Phase15** adds: (1) **embedding and link prediction** for compound–gene (Node2Vec, PyKEEN, GNN) with standard metrics (ROC-AUC, MRR, Hits@K); (2) **causal mediation** (SEM + bootstrap) for hypothesis generation; (3) **validation** against ChEMBL/DrugBank when available; (4) **reproducible pipelines** with report JSON and summary markdown at every step; (5) **demo UI** for pair explanation and export.  
- Emphasise: deterministic audits, acceptance thresholds, no trial-and-error; all steps run locally (Windows/PowerShell, Jupyter); Phase14 snapshot as frozen input.

## Reproducibility

- **Config:** `configs/phase15_config.yaml` (paths, hyperparameters).  
- **Default input:** Phase14 snapshot `data/processed/milestones/phase14/v1_working_2026-02-19/phase14_20260219_204918`. Override with `--run-dir`.  
- **Output:** `data/processed/phase15_embeddings/<timestamp>/` with `reports/`, `models/`, `tables/`, `embeddings/`, `predictions/`, `causal/`.  
- **Run full pipeline:**  
  `python scripts/phase15/run_phase15.py`  
  Optional: `--skip-node2vec`, `--skip-pykeen`, `--skip-gnn`, `--skip-causal`, `--skip-validate`.

## Optional dependencies (Phase15)

Install as needed; existing env is unchanged.

```bash
pip install node2vec pykeen torch-geometric streamlit pyvis
```

See `docs/phase15.md` (this file) for scope; no change to `requirements.txt` required for core Phase14/earlier workflows.
