# Phase14.5 / expansion layer — summary

## What changed

1. **Category bias diagnosis**  
   - Script: `scripts/phase14/diagnose_category_and_pathway.py`  
   - Writes `reports/category_bias_report.json`: counts per category (atlas, pair_mediation, Neo4j AFFECTS), score distribution per category.  
   - **Conclusion:** Bias is **upstream-only** (Phase13 atlas). Many top interactions are Apoptosis because the atlas has more confirmed pairs in that category. No artificial rebalancing; use category-balanced queries to explore others.

2. **Unknown pathway diagnosis**  
   - Same script writes `reports/unknown_pathway_diagnosis.json`: MEDIATED_BY to Unknown pathway, genes with/without IN_PATHWAY, cause (pathway_cluster_info cluster 0 has auto_label "unknown pathway_unknown_pathway").  
   - **Pathway label improvement:** For pathway nodes that would display as "Unknown pathway", the exporter now can replace with a better label from `pathway_cluster_info` (e.g. top_terms for cluster 0). See `display_names.pathway_unknown_to_better_label()` and its use in `regenerate_neo4j_export.py`.

3. **Expanded discovery graph**  
   - Script: `scripts/phase14/export_expanded_discovery.py`  
   - Builds a **second** graph with more Ingredient nodes (one-hop expansion: ingredients that share ≥1 compound with atlas ingredients and connect to existing graph compounds).  
   - Export goes to **`neo4j_expanded/`** only; **`neo4j/`** (confirmed) is never overwritten.  
   - Report: `reports/expanded_discovery_report.json` (before/after ingredient counts, new edges).

4. **Compound display names**  
   - Already improved in the earlier pass (compound_master name column).  
   - **Compound class:** No compound class column in `compound_master`; skipped. If a local source with class (e.g. flavonoid, alkaloid) is added later, a property or CompoundClass node can be added.

5. **Analysis pack v2**  
   - File: `scripts/neo4j/phase14_analysis_pack_v2.md`  
   - Eight analysis-grade Cypher queries: category-balanced top interactions, ingredient pairs by shared gene-linked compounds (weighted), by shared pathways (excluding Unknown), ingredients by gene targets, pathways mediating interactions, compounds reused, low-confidence interactions, category diversity.

6. **Export strategy and README**  
   - `data/processed/phase14_mediation/README_EXPORTS.md`: two export modes (confirmed vs expanded), commands, reports, intended use.

---

## What improved

- **Category bias:** Quantified and attributed to upstream atlas; category-balanced queries allow exploring beyond Apoptosis.  
- **Unknown pathway:** Cause identified (cluster 0 label); display name can be overridden from pathway_cluster_info; analysis queries can exclude Unknown pathway.  
- **Ingredient breadth:** Expanded discovery graph has more than 47 ingredients when ingredient_compound supports one-hop expansion; confirmed graph stays at 47.  
- **Pathway display:** Cluster 0 and similar “unknown” clusters get a better display label when data is available.  
- **Analysis:** Richer, category-aware Cypher pack and low-quality explanation report.

---

## What remains a limitation

- **Category distribution** is still driven by Phase13; to change it, Phase13 or the atlas construction would need to change.  
- **Unknown pathway** nodes are still in the graph; we only improve labels and filter in queries. Removing or merging them would require a graph rebuild.  
- **Compound class** is not added (no local source in compound_master).  
- **Expanded graph** does not add new AFFECTS or mechanistic_score for new ingredients; it only adds Ingredient nodes and HAS_COMPOUND edges to existing compounds.  
- **Pathway coverage** (genes with IN_PATHWAY) depends on target_functional_clusters and pathway_cluster_info; improving it further would need better gene→pathway mapping or more clusters.

---

## Commands to run

**Diagnostics (category bias + Unknown pathway):**
```bash
python scripts/phase14/diagnose_category_and_pathway.py --latest
```

**Regenerate confirmed graph only (with pathway label improvement):**
```bash
python scripts/phase14/regenerate_neo4j_export.py --latest
```

**Export expanded discovery graph:**
```bash
python scripts/phase14/export_expanded_discovery.py --latest
```

Optional: `--min-shared-compounds 2` for stricter expansion.
