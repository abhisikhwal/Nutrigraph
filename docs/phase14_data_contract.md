# Phase14 data contract

Phase14 (Biological Mediation Layer) expects canonical IDs and column names for Ingredient→Compound→Gene so the mediation graph and scoring are deterministic and auditable.

## Paths (relative to repo root)

- **Canonical ING→CMP (preferred):** `data/processed/canonical/ingredient_compound_links.csv` (required), optional `.parquet`.
- **Canonical CMP→GENE:** `data/processed/canonical/compound_gene_links.csv` or `.parquet`.
- **Build scripts:**  
  - `python scripts/phase14/build_canonical_ingredient_compounds.py` — produces `ingredient_compound_links.csv`.  
  - `python scripts/phase14/build_canonical_compound_gene.py` — produces `compound_gene_links.csv`.

## Ingredient (ING)

- **Canonical form:** `ING_` + zero-padded 6-digit numeric (e.g. `ING_000026`), or `ING_<normalized_string>` for non-numeric.
- **Source of truth for IDs:** `data/processed/canonical/recipe_ingredients_expanded_v2.parquet` (or CSV) — `ingredient_id` column. Phase13 atlas uses the same ING_ IDs (`ingA_id`, `ingB_id`).
- **Required in ING→CMP table:** `ingredient_id` — must match Phase13 / recipe_ingredients (same normalizer: `to_ingredient_id()` in `src/phase14/id_normalization.py`).

## Compound (CMP)

- **Canonical form:** Prefer **InChIKey** (27 chars, uppercase, with dashes) for join with compound_gene_links; otherwise a stable internal ID (e.g. `CMP_` prefix or source-specific ID).
- **Join key:** `compound_id` in both `ingredient_compound_links` and `compound_gene_links` must use the **same identifier space** (e.g. both InChIKey or both the same internal ID) so overlap % is meaningful.
- **Required in ING→CMP table:** `compound_id`.  
- **Required in CMP→GENE table:** `compound_id` (or `inchikey` column, then loader maps to `compound_id`).

## Gene

- **Canonical form:** Uppercase symbol or `GENE_`-prefixed (e.g. `TP53`).  
- **Required in CMP→GENE table:** `gene` or `gene_id` column (loader normalizes to uppercase).

## Required columns by file

| File | Required columns | Notes |
|------|------------------|--------|
| `ingredient_compound_links.csv` | `ingredient_id`, `compound_id` | Both must be non-empty; ingredient_id in ING_ form. |
| `compound_gene_links.csv` | `compound_id`, `gene` (or `gene_id`) | Optional: `inchikey`, `evidence`, `source`. |
| Phase13 atlas | `ingA_id`, `ingB_id`, `category` | Same ING_ IDs as recipe_ingredients. |

## Coverage gates

For the mediation layer to be biologically meaningful:

- **Overlap:** % of compounds in `ingredient_compound_links` that appear in `compound_gene_links` should be ≥ 1%.
- **Atlas coverage:** % of atlas rows with both ingA and ingB having ≥1 compound in ING→CMP should be ≥ 30%.

If these are not met, the build scripts exit with a nonzero code and write `data/processed/canonical/phase14_audit/audit_report.json` with example IDs from each side for diagnostics.

## Windows

- Prefer **CSV** for canonical files so loading is robust on Windows; parquet is optional.
- All paths in scripts and loaders are relative to repo root and use `pathlib.Path` for portability.
