# Phase14.5 deliverables — commands and locations

## 1. Category bias diagnosis

**Conclusion:** Bias is **upstream-only**. Atlas and pair_mediation have even counts per category (79 each); Neo4j AFFECTS in this run reflect that. Top mechanistic scores clustering in Apoptosis is due to Phase13 atlas content. No artificial fix; use category-balanced queries.

**Report:** `data/processed/phase14_mediation/<run_id>/reports/category_bias_report.json`

**Command:**
```bash
python scripts/phase14/diagnose_category_and_pathway.py
```
(Optional: `--run-dir <path>` and/or `--phase13-dir <path>`)

---

## 2. Unknown pathway diagnosis

**Cause:** pathway_cluster_info cluster_id=0 has auto_label `unknown pathway_unknown_pathway` (largest cluster). Display name was normalized to "Unknown pathway".

**Improvement:** Pathway label for cluster 0 (and similar) is now overridden at export using top_terms from pathway_cluster_info (e.g. "unknown pathway / wnt / fgfr3"). One pathway node (PATH_000000) gets a clearer label; MEDIATED_BY counts are unchanged.

**Report:** `reports/unknown_pathway_diagnosis.json` (genes with/without IN_PATHWAY, MEDIATED_BY to Unknown, cause, recommendation).

---

## 3. Expanded discovery graph (implemented)

**Strategy:** One-hop expansion. Add Ingredient nodes that (a) share at least `min_shared_compounds` with atlas ingredients and (b) connect to compounds already in the graph via ingredient_compound.

**Result (latest run):** 47 → 242 ingredients, 958,669 new HAS_COMPOUND edges. Confirmed graph unchanged.

**Export location:** `data/processed/phase14_mediation/<run_id>/neo4j_expanded/nodes.csv` and `edges.csv`

**Report:** `reports/expanded_discovery_report.json`

**Command:**
```bash
python scripts/phase14/export_expanded_discovery.py
```
Optional: `--run-dir <path>`, `--min-shared-compounds 2`

---

## 4. Pathway and compound enrichment

- **Pathway:** Unknown cluster display name improved from pathway_cluster_info top_terms (see above).
- **Compound:** Display names already from compound_master (earlier pass). No compound class column in compound_master; not added.

---

## 5. Commands to regenerate

**Confirmed graph only (ingredient names, display names, AFFECTS mechanistic_score, pathway label improvement):**
```bash
python scripts/phase14/regenerate_neo4j_export.py --latest
```
Output: `<run_id>/neo4j/nodes.csv`, `neo4j/edges.csv`, and reports under `reports/`.

**Expanded discovery graph:**
```bash
python scripts/phase14/export_expanded_discovery.py
```
Output: `<run_id>/neo4j_expanded/nodes.csv`, `neo4j_expanded/edges.csv`, and `reports/expanded_discovery_report.json`.

**Diagnostics (category + Unknown pathway):**
```bash
python scripts/phase14/diagnose_category_and_pathway.py
```

---

## 6. Summary (what changed, what improved, what remains)

| Item | What changed | What improved | Limitation |
|------|--------------|---------------|------------|
| Category bias | Diagnosed and reported | Category-balanced queries in analysis pack; no artificial reweighting | Distribution still from Phase13 |
| Unknown pathway | Cause identified; cluster 0 label improved from top_terms | Clearer display for PATH_000000; queries can exclude by id or name | Unknown node still present; gene→pathway coverage unchanged |
| Ingredient count | Second export adds one-hop ingredients | 47 → 242 in expanded graph | New ingredients have no AFFECTS/mechanistic_score |
| Pathway display | pathway_unknown_to_better_label() in export | One major “unknown” cluster has a descriptive label | Other clusters unchanged |
| Compound class | Checked compound_master | — | No local class column; not added |
| Analysis | New analysis pack v2 | 8 queries: category-balanced, shared compounds/pathways, low-confidence, diversity | — |

**Files added/updated:**
- `scripts/phase14/diagnose_category_and_pathway.py`
- `scripts/phase14/export_expanded_discovery.py`
- `src/phase14/display_names.py` (pathway_unknown_to_better_label)
- `scripts/phase14/regenerate_neo4j_export.py` (pathway label override)
- `scripts/neo4j/phase14_analysis_pack_v2.md`
- `data/processed/phase14_mediation/README_EXPORTS.md`
- `scripts/phase14/PHASE14_5_SUMMARY.md`
- `scripts/phase14/PHASE14_5_DELIVERABLES.md` (this file)
