# Display name enrichment — summary

## What was done

1. **Display-name enrichment pass** (no graph structure or scoring changes):
   - **Compound:** If `name` is only CMP_/InChIKey-like, `display_name` is set from `compound_master` (name column) when available; otherwise `display_name` = `name`.
   - **Pathway:** `display_name` = clean version of `name`: underscores → spaces, repeated tokens collapsed, `"unknown pathway_unknown_pathway"` → `"Unknown pathway"`. Original `name` kept.
   - **Category:** `display_name` = human-readable (spaces for underscores, title-case).
   - **Ingredient:** `display_name` = `name` (already human from ingredient name fix).

2. **Neo4j export** now includes a `display_name` column in `neo4j/nodes.csv` (with `:ID`, `:LABEL`, `name`, `source`). Export is regenerated only (no Phase14/Phase15 rerun).

3. **Showcase Cypher pack:** 8 queries in `scripts/neo4j/phase14_showcase_queries.md`.

4. **Report:** After regeneration, see `reports/display_name_enrichment_report.json` in the run dir for counts and sample before/after.

## Node types enriched

| Type      | Enrichment |
|-----------|------------|
| Compound  | display_name from compound_master when name is id-like |
| Pathway   | clean_display_name (spaces, dedup, "Unknown pathway") |
| Category  | display_name title-case, underscores → spaces |
| Ingredient| display_name = name (already fixed) |
| Gene      | display_name = name (unchanged) |
| Interaction | display_name = name (id) |

## Report location

- **Per-run:** `data/processed/phase14_mediation/<run_id>/reports/display_name_enrichment_report.json`
  - `node_types_enriched`, `compound_improved`, `pathway_improved`, `category_improved`, `ingredient_with_name`
  - `sample_before_after`: compound, pathway, category

## Cypher query pack location

**`scripts/neo4j/phase14_showcase_queries.md`**

Queries:
1. Top interactions by mechanistic_score (MEDIATED_BY)
2. Interaction → ingredients → category
3. Ingredient → compound → gene
4. Ingredient → compound → gene → pathway → category
5. Interaction → mediated pathway → category
6. Top ingredients by compound count
7. Top compounds by gene count
8. Top genes by pathway count

## Exact command to regenerate Neo4j export

From repo root:

```bash
python scripts/phase14/regenerate_neo4j_export.py --latest
```

For a specific run:

```bash
python scripts/phase14/regenerate_neo4j_export.py --run-dir data/processed/phase14_mediation/phase14_20260219_204918
```

This will:
- Load mediation_nodes.csv and mediation_edges.csv
- Enrich Ingredient names from ingredients.parquet
- Run display-name enrichment (Compound, Pathway, Category, Ingredient)
- Write reports/display_name_enrichment_report.json
- Write neo4j/nodes.csv (with display_name) and neo4j/edges.csv
- Write reports/ingredient_name_export_validation.json
