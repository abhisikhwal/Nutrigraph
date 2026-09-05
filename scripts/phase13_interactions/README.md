# Phase 13: Ingredient Interaction Discovery (v3)

Robust, resumable pipeline for discovering ingredient–ingredient interactions on recipe functional signatures (Phase 17 v3 or Phase 8 v2).

## How to run

From the **repo root** (`global-food-genome`):

```bash
# Default: v3 signatures (Phase 17), config from config/phase13.yaml
python scripts/phase13_interactions/run_phase13.py

# Custom config
python scripts/phase13_interactions/run_phase13.py --config path/to/phase13.yaml

# Use v2 signatures (55 categories) instead of v3
python scripts/phase13_interactions/run_phase13.py --use-v2
```

Requirements: `pandas`, `numpy`, `pyyaml`, `statsmodels`, `pyarrow`. The pipeline **requires** `statsmodels` for OLS regression and will exit with a clear error if it is missing. Run from repo root so that `src.phase13` is importable (or add repo root to `PYTHONPATH`).

## Inputs (required)

- **Signatures (v3 preferred)**  
  - Tried in order:  
    - `data/processed/phase17_reaggregation/recipe_functional_signatures_v3.parquet`  
    - `data/processed/canonical/recipes_biological_effects_v3.parquet`  
  - With `--use-v2`: `data/processed/exports_v2/recipes_biological_effects_v2_FINAL.parquet`
- **Recipe–ingredient map**: `data/processed/canonical/recipe_ingredients_expanded_v2.parquet`

Optional: confounder table (e.g. cuisine) at path set in `config/phase13.yaml` under `confounders_path`. If missing, all recipes get `cuisine_unknown`.

## Expected outputs

All under `data/processed/phase13_interactions_v3/` (or `output_dir` in config):

| File | Description |
|------|-------------|
| `interactions_raw_v3.parquet` | All pair × category regression results (beta_int, se, t, p, n_both, etc.) |
| `interactions_adjusted_v3.parquet` | Same + q_category, q_global, significant_005, significant_001 |
| `ingredient_pairs_tested_v3.parquet` | Pairs actually tested (ingA_id, ingB_id, nA, nB, nBoth) |
| `pair_skip_reasons_v3.parquet` | Pairs skipped (e.g. nBoth &lt; min_joint) with reason, nA, nB, nBoth |
| `phase13_summary_v3.json` | Manifest (n_recipes, n_pairs, func_cols, confounders), null test summary, top 100 by q_value |
| `null_test_v3.json` | Permutation null: frac significant under permuted IA×IB, p-value distribution |
| `bootstrap_stability_v3.parquet` | For top interactions: beta_int_orig, mean, lower, upper over bootstrap samples |
| `shards/interactions_shard_*.parquet` | Per-shard results (used for resumability) |

## Resumability

- Pairs are split into shards (default 5000 pairs per shard).
- Each shard is written to `shards/interactions_shard_0001.parquet`, etc.
- On restart, already-written shards are skipped; only missing shards are run.
- Final step merges all shards into `interactions_raw_v3.parquet`.

## Configuration

Edit `config/phase13.yaml`:

- **Pair selection**: `min_ingredient_freq`, `top_k_ingredients`, `min_joint`, `seed`
- **Sharding**: `shard_size`, `shards_dir`
- **Null test**: `null_n_pairs`
- **Bootstrap**: `bootstrap_top_n`, `bootstrap_n_samples`
- **Paths**: `output_dir`, `recipe_ingredients_path`, `signature_paths_v3`, `signature_path_v2`, `confounders_path`

## Validation (hard fail)

The script exits with clear errors if:

- Signature or recipe_ingredients file is missing
- No `recipe_id` overlap between signatures and recipe_ingredients
- `recipe_id` or `ingredient_id` columns are missing
- No functional columns remain after dropping constant/near-constant ones
- No ingredients pass the frequency filter

## Scientific checks

- **Permutation null**: For a subset of pairs, the interaction term (IA×IB) is permuted across recipes; p-values should become approximately uniform and fraction significant at 0.05 ≈ 0.05. Results in `null_test_v3.json`.
- **Bootstrap stability**: Top interactions (by q-value) are re-estimated on bootstrap samples of recipes; CIs and sign stability are saved in `bootstrap_stability_v3.parquet`.

## Memory and runtime

Designed for ~32 GB RAM and 8 GB GPU (regression is CPU statsmodels). Large recipe × ingredient matrices are avoided; only selected ingredients and sharded pair lists are used. If OOM occurs, reduce `top_k_ingredients` or `shard_size` in config.
