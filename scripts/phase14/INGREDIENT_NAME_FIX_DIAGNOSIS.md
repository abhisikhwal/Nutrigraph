# Ingredient name fix – source-of-truth and export

## Source-of-truth ingredient mapping

- **Path:** `data/processed/canonical/ingredients.parquet`
- **Columns used:** `ingredient_id`, `canonical_name` (preferred), `scientific_name`, `name`
- **Usage:** Phase14 loads this via `load_ingredient_id_to_name(repo_root)` in `src/phase14/loaders.py` and passes the `ingredient_id -> name` dict into `build_mediation_graph` so Ingredient nodes get a human-readable `name`.

## Export script

- **Path:** `src/phase14/export.py`
- **Functions:** `write_neo4j_export()`, `neo4j_nodes_csv()`, `write_ingredient_name_export_validation()`
- **Output:** `neo4j/nodes.csv`, `neo4j/edges.csv`, `reports/ingredient_name_export_validation.json`

## Root cause

1. In **`src/phase14/mediation_graph.py`**, Ingredient nodes were created with `"name": nid` (i.e. `name` was set to the node id, e.g. `ING_000051`), so the mediation graph never had a human name for ingredients.
2. In **`src/phase14/export.py`**, `neo4j_nodes_csv()` does `if "name" not in df.columns: df["name"] = df[":ID"]`, so when `name` was missing or already equal to id, the Neo4j export wrote the same value for both `id` and `name`.

So the bug was at the **graph build**: the authoritative mapping (`ingredients.parquet`) existed but was never joined when creating Ingredient nodes.

## What was changed

1. **`src/phase14/loaders.py`**
   - Added `load_ingredient_id_to_name(repo_root)` to build `ingredient_id -> name` from `data/processed/canonical/ingredients.parquet` (keys normalized with `to_ingredient_id`).
   - Added `_is_id_like_name()` so we do not use id-like values as display names.

2. **`src/phase14/mediation_graph.py`**
   - `build_mediation_graph(..., ingredient_id_to_name=Optional[Dict[str, str]])`.
   - For each Ingredient node, `name = ingredient_id_to_name.get(nid, nid)` when the mapping is provided.

3. **`scripts/phase14/run_phase14.py`**
   - Loads `ingredient_id_to_name = load_ingredient_id_to_name(repo_root)` and passes it into `build_mediation_graph(..., ingredient_id_to_name=ingredient_id_to_name)`.

4. **`src/phase14/export.py`**
   - After writing Neo4j CSVs, calls `write_ingredient_name_export_validation(n_df, output_dir)` to write `reports/ingredient_name_export_validation.json` (total Ingredient nodes, count with `name != id`, sample of 20 rows).

5. **`scripts/phase14/regenerate_neo4j_export.py`** (new)
   - Regenerates only the Neo4j export for an existing Phase14 run: loads `mediation_nodes.csv` / `mediation_edges.csv`, enriches Ingredient names from `ingredients.parquet`, then calls `write_neo4j_export()` so you do not need to re-run all of Phase14.

## Backward compatibility

- Other node types (Category, Pathway, Gene, Compound, Interaction) are unchanged.
- If `ingredients.parquet` is missing, `load_ingredient_id_to_name` returns `{}` and Ingredient nodes keep `name = id` (previous behaviour).

## Commands

**Regenerate Neo4j export for the latest Phase14 run (no full rerun):**
```bash
python scripts/phase14/regenerate_neo4j_export.py --latest
```

**Regenerate for a specific run:**
```bash
python scripts/phase14/regenerate_neo4j_export.py --run-dir data/processed/phase14_mediation/phase14_YYYYMMDD_HHMMSS
```

**Full Phase14 run (future runs will have correct Ingredient names by default):**
```bash
python scripts/phase14/run_phase14.py --phase13-dir data/processed/phase13_interactions_v3_20260206_162122_b_gpu_stable
```
