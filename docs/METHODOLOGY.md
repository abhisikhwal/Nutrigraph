# Methodology

Technical overview of the NutriGraph / global-food-genome pipeline. For a scannable product pitch, see the root [README](../README.md).

## Design principles

1. **Do not invent measured edges.** Lab assays (ChEMBL, BindingDB, etc.) stay measured; structural inference produces separately labeled predicted edges.
2. **Identity first.** Compounds resolve to InChIKey; proteins to UniProt then HGNC; genes to HGNC symbols; foods to stable ingredient IDs (and FDC where nutrition exists).
3. **Calibrate enrichment.** Weighted gene sets need a null that respects weights, or FDR is fiction.
4. **Ship a trimmed graph for viz.** The full ingredient→compound layer is millions of edges; the public Neo4j demo is intentionally capped.

## Stage 1 — Chemistry corpus

- **Food → compound:** FooDB contents (and related chemistry) produce `ingredient_compound_canonical_v2` (~2M edges, ~48k compounds among mechanism-live foods).
- **Compound identity:** FooDB + COCONUT (and supporting sources) unify into a compound master keyed by InChIKey.
- **Compound → gene (measured):** ChEMBL / BindingDB activities filtered to human single-protein targets, mapped UniProt → HGNC.

## Stage 2 — Structural inference

Dark food compounds lack assays. Fingerprint nearest-neighbor transfer from the measured corpus proposes targets:

- Representations: RDKit fingerprints; Murcko scaffolds for hard splits.
- Outputs: predicted compound→gene edges with confidence tiers / weights.
- Integration: `compound_gene_integrated_v1` carries `evidence ∈ {measured, predicted}` and `confidence_weight`.
- Validation: scaffold-split **hit@10 ≈ 0.858** (see methodology showcase JSON).

## Stage 3 — Ingredient gene sets

Predicted and measured compound→gene edges roll up per ingredient:

- `ingredient_gene_sets_v3` — per `(ingredient_id, gene_symbol)`: evidence, confidence, supporting-compound counts.
- Global honesty mix on this layer: **~20.4% measured / ~79.6% predicted**.

## Stage 4 — Pathways & enrichment

- Gene → pathway maps from GO + Reactome (`gene_pathway_mappings`).
- Weighted enrichment with **weight-permutation null** calibration (null rate of any significant pathway ~1.3% at q<0.1 in the chosen configuration).

## Stage 5 — Tissues

- GTEx median TPM → gene expressed-in tissue.
- Ingredient tissue profiles attribute mechanism weight to tissues via contributing genes (interpretation: expression location of targets, not proof of compound delivery).

## Stage 6 — Universe expansion

- Locked core: 463 species with mechanism coverage.
- Expansion: +232 recipe-derived nodes (composites, blends, nutrition-only, name-only) → **695** live ingredients.
- Recipe string mapping coverage ~**97.6%** on the expanded corpus.

## Stage 7 — Nutrition & dose

- Species ↔ USDA FoodData Central (with documented FooDB fallbacks).
- Nutrient profiles per species; optional dose helpers convert recipe quantities to relative mechanism contribution (not absolute compound RDI).

## Stage 8 — Product artifacts

| Artifact | Path | Role |
|----------|------|------|
| Ingredient profiles | `data/processed/product/ingredient_profiles_v2.jsonl` | Per-food card data |
| Showcase bundle | `data/processed/product/showcase/` | Static JSON for the marketing/demo site |
| Neo4j load | `data/processed/product/neo4j_load/` | Trimmed CSVs + Cypher + Neovis config |

### Trimmed Neo4j schema

```
(:Ingredient)-[:CONTAINS]->(:Compound)
(:Compound)-[:TARGETS {evidence, confidence}]->(:Gene)
(:Gene)-[:IN_PATHWAY]->(:Pathway)
(:Gene)-[:EXPRESSED_IN {score}]->(:Tissue)
(:Ingredient)-[:HAS_NUTRIENT {amount, unit}]->(:Nutrient)
```

Trim defaults (see `build_neo4j_trimmed_graph.py`): top-50 compounds/ingredient; all measured TARGETS + top-K predicted/compound; top-T tissues/gene.

**Public safety model:** a Neo4j **reader** user (no write/DDL) plus parameterized expand queries for any node — not a fixed allow-list of five canned queries. Optional thin API: timeout + result cap in front of Bolt.

## Reproducibility notes

- Canonical mechanism files should be treated as read-only inputs to product builds.
- Raw downloads are not in git (`data/raw/` gitignored). Rebuild from upstream sources and licenses.
- Exact script entrypoints evolve by phase under `scripts/phase*`, `scripts/thread2`, `scripts/tier1`, `scripts/product`.

## What this is not

Pharmacokinetics, clinical outcomes, personalized dosing, or medical advice. The graph answers: *which human molecular targets are associated with this food’s chemistry, and how much of that association is measured vs inferred?*
