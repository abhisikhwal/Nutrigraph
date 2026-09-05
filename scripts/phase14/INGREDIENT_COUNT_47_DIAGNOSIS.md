# Why the Phase14 Neo4j Export Has Only 47 Ingredient Nodes

## Concise diagnosis

**The 47 Ingredient nodes are intentional.** The drop from the full ingredient universe (1338 canonical / 818 in recipes) to 47 happens **at the Phase13 boundary**: the Phase14 run consumes `atlas_confirmed`, which contains only **statistically confirmed interaction pairs**. Those pairs involve exactly **47 unique ingredients**. Phase14 never adds Ingredient nodes from recipe_ingredients or the canonical table; it only adds ingredients that appear in the atlas. So the Neo4j export is a **mediation subgraph** (ingredients + compounds + genes + pathways that participate in or explain those confirmed pairs), not the full ingredient universe.

---

## Coverage table (run: phase14_20260219_204918)

| Stage | Unique ingredient count | Note |
|-------|-------------------------|------|
| Canonical ingredients (`data/processed/canonical/ingredients.parquet`) | 1,338 | Full ingredient universe (id normalized) |
| Recipe ingredients (`recipe_ingredients_expanded_v2.parquet`) | 818 | Ingredients that appear in recipes |
| **Phase13 atlas_confirmed** (unique ingA_id + ingB_id) | **47** | Only ingredients in at least one confirmed interaction pair |
| Atlas rows (pairs × categories) | 1,027 | Rows in atlas_confirmed |
| Mediation graph: Ingredient nodes | 47 | From `build_mediation_graph` (source: atlas only) |
| Mediation graph: total nodes | 60,571 | All node types (Ingredient, Compound, Gene, Pathway, etc.) |
| Neo4j `nodes.csv`: Ingredient nodes | 47 | Same as mediation; no further filtering on export |

**Where the drop happens:** The count goes from 818 (recipe ingredients) to **47** at **Phase13**. The atlas is produced by an earlier pipeline that tests ingredient pairs for significant biological effects and keeps only “confirmed” pairs; the union of all ingA_id and ingB_id in that file is 47. Phase14 does not add any Ingredient node that is not in the atlas.

---

## Where filtering happens (script / function)

| Layer | File | Function / logic | Effect |
|-------|------|-------------------|--------|
| **Phase13** (upstream) | Phase13 pipeline output | `atlas_confirmed.csv`: only rows that passed statistical confirmation | Only 47 unique ingredients appear in (ingA_id, ingB_id). |
| **Phase14 graph build** | `src/phase14/mediation_graph.py` | `build_mediation_graph()` lines 106–119 | Builds `all_ings` from **atlas_confirmed** only; appends one Ingredient node per unique `ingA_id` / `ingB_id`. No other source (recipe_ingredients, canonical ingredients) is used for Ingredient nodes. |
| **Phase14 export** | `src/phase14/export.py` | `write_neo4j_export()` → `neo4j_nodes_csv()` | Exports all nodes from `mediation_nodes`; no filter that drops Ingredient nodes. |

So the **design choice** is in **`build_mediation_graph()`**: Ingredient nodes are **only those participating in atlas-confirmed pairs**. The “filter” is “use only atlas_confirmed” as the source of Ingredient nodes.

---

## Intentional design

- **Yes.** The mediation graph is built to support **mechanistic explanation of confirmed interactions**: pathways, compounds, and genes that connect to those 47 ingredients. Including all 818 or 1338 ingredients would add many nodes with no edges to the current interaction set and would change the semantics of the graph (mediation subgraph vs full ingredient catalog).

---

## Recommendation

1. **Keep current design for the main Neo4j export**  
   The existing export is the right artifact for:
   - Demo/research on **mediated interactions** (why do these pairs have effects?).
   - Queries that start from confirmed pairs and traverse to compounds/pathways/genes.

2. **Optional: second “full ingredient universe” export**  
   If you need a Neo4j graph that includes **all** canonical or recipe ingredients (e.g. for browsing, search, or future pair discovery):
   - Add a **separate** export path (e.g. `write_neo4j_full_ingredients_export()` or a small script) that:
     - Reads `data/processed/canonical/ingredients.parquet` (and optionally recipe_ingredients).
     - Creates Ingredient nodes for all unique ingredient_ids with names from the same source used for the 47.
     - Writes to a different directory (e.g. `neo4j_full_ingredients/`) or with a different node label so it does not mix with the mediation subgraph.
   - Do **not** change scoring or Phase15; keep the 47-ingredient mediation graph as the source for pair-level analytics.

---

## Commands

**Regenerate coverage report for this run:**
```bash
python scripts/phase14/ingredient_coverage_report.py --phase14-run data/processed/phase14_mediation/phase14_20260219_204918
```

**Regenerate for latest Phase14 run and write report to run dir:**
```bash
python scripts/phase14/ingredient_coverage_report.py
```

**No code change required** for the 47-ingredient behavior; it follows from the current design. To add a full-ingredient Neo4j export, implement the optional second export above without changing Phase14 scoring or Phase15.
