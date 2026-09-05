# Phase14 Neo4j query cleanup — deliverable summary

## Concise diagnosis (why earlier queries felt off)

| Issue | Cause | Fix |
|-------|--------|-----|
| **Shared-compound too broad / double-count** | Pairs (A,B) and (B,A) both counted; no canonical order; self-pairs possible | Enforce `i1.id < i2.id`, exclude self-pairs; optional filter to compounds with ≥1 gene and/or hub filter |
| **Showcase returned giant ingredient lists** | Aggregation was by category only, so many interactions merged into one list per category | Group **per Interaction**: one row per (interaction_id, ingredient_pair, category, top_pathways, mechanistic_score) |
| **Unknown pathway dominated** | No filter on pathway display_name in demo queries | Demo/analysis queries exclude `display_name = "Unknown pathway"`; report in `unknown_pathway_report.json` |
| **mechanistic_score null on earlier queries** | Score was only on MEDIATED_BY; AFFECTS (INT→CAT) did not carry it | Export enriches AFFECTS with mechanistic_score from pair_mediation; regenerate Neo4j export |

---

## What was changed

### Export (no graph structure or scoring change)

- **`src/phase14/export.py`**
  - **`_enrich_affects_with_mechanistic_score(e_base, pair_mediation)`:** Sets `mechanistic_score` on every AFFECTS edge from `pair_category_mediation` (by int_id, cat_id).
  - **`write_neo4j_export`:** Calls the enrichment before writing edges; AFFECTS and MEDIATED_BY both carry mechanistic_score where applicable.
  - **`write_mechanistic_score_validation(e_df, output_dir)`:** Writes `reports/mechanistic_score_validation.json` (AFFECTS total, with score, sample rows).
  - **`write_unknown_pathway_report(n_df, e_df, output_dir)`:** Writes `reports/unknown_pathway_report.json` (MEDIATED_BY to Unknown pathway count, interactions only-Unknown, fraction).

### Query / analysis files

- **`scripts/neo4j/phase14_analysis_queries.md`** (new)
  - Diagnosis (same as above).
  - **A)** Shared-compound pack: canonical pair (`i1.id < i2.id`), no self-pairs, optional gene filter, hub filter, specificity = 1/ingredient_degree, weighted shared score, recommended query + variants.
  - **B)** Showcase per-interaction: one row per interaction with ingredient_pair, category, top_pathways, mechanistic_score; collect scoped per interaction.
  - **C)** Unknown pathway: filter in queries; report location.
  - **D)** mechanistic_score: on AFFECTS and MEDIATED_BY; validation report.
  - **E)** 10 analysis-grade Cypher queries (see list below).

- **`scripts/neo4j/phase14_showcase_queries.md`** (updated)
  - Query 1: use AFFECTS and `a.mechanistic_score`.
  - Query 2: per-interaction grouping with mechanistic_score.
  - Query 5: exclude Unknown pathway.
  - Reference to `phase14_analysis_queries.md` for full analysis pack.

---

## Reports (per run)

After regenerating the Neo4j export for a run:

- **`reports/mechanistic_score_validation.json`**  
  - `affects_total`, `affects_with_mechanistic_score`, `sample_rows`, note.
- **`reports/unknown_pathway_report.json`**  
  - `mediated_by_total`, `mediated_by_to_unknown_pathway`, `interactions_with_mediated_by`, `interactions_only_unknown_pathway`, `fraction_only_unknown`.

---

## Commands to run

**Regenerate Neo4j export (adds mechanistic_score to AFFECTS, writes validation + Unknown pathway report):**

```bash
python scripts/phase14/regenerate_neo4j_export.py --latest
```

For a specific run:

```bash
python scripts/phase14/regenerate_neo4j_export.py --run-dir data/processed/phase14_mediation/phase14_20260219_204918
```

---

## Cypher to use after the fix

Use the analysis pack for production-style queries:

**File:** `scripts/neo4j/phase14_analysis_queries.md`

**10 analysis queries:**

1. Top interactions by mechanistic_score (AFFECTS)
2. Interaction → ingredients → category (one row per interaction–category)
3. Interaction → top pathways (excluding Unknown)
4. Ingredient pair → shared compounds (deduped + weighted by specificity)
5. Ingredient → top genes
6. Gene → top pathways (excluding Unknown)
7. Category → top interactions
8. Pathways most frequently mediating interactions (excluding Unknown)
9. Compounds most reused across interactions
10. Interactions with low-quality explanation (only Unknown pathway)

**Shared-compound:** Use the “Recommended” or “Simpler” / “Exclude hub” variants in section A of the analysis pack.  
**Showcase:** Use the “per-interaction” and “mechanistic_score from AFFECTS” variants in section B and in `phase14_showcase_queries.md`.  
**Unknown pathway:** Add `WHERE p.display_name <> "Unknown pathway"` in pathway-based queries; check `reports/unknown_pathway_report.json` for counts.
