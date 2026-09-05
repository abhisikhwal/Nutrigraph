# Phase14 scripts

## Compound identity pipeline

The **compound identity layer** resolves FooDB-style compound IDs (e.g. `FDB_22741`) to canonical InChIKeys so that **Ingredient → Compound → Gene** paths work end-to-end.

### How to run

From the **repo root**:

```bash
# Full run (fails with exit 1 if overlap < 20%)
python -m src.phase14.compound_identity --repo-root .

# With scan artifact and smoke mode (never exits non-zero; writes all reports)
python -m src.phase14.compound_identity --repo-root . --write-scan --smoke

# Optional: relax overlap gate (e.g. for testing)
python -m src.phase14.compound_identity --repo-root . --min-overlap-pct 5
```

Or via the script:

```bash
python scripts/phase14/build_compound_identity_layer.py [--repo-root .] [--write-scan]
```

### Outputs

| Output | Location | Description |
|--------|----------|-------------|
| compound_master | `data/processed/canonical/compound_master.csv` | Master table: inchikey, cid, fdb_id, fdb_id_raw, fdb_id_norm, name, source_file |
| ingredient_compound_canonical | `data/processed/canonical/ingredient_compound_canonical.csv` | Ingredient → compound (InChIKey) |
| compound_gene_canonical | `data/processed/canonical/compound_gene_canonical.csv` | Compound (InChIKey) → gene |
| Diagnostics | `data/processed/canonical/compound_identity_diagnostics.json` | Resolution %, overlap %, top InChIKeys, FDB pattern counts |
| Overlap report | `data/processed/canonical/compound_overlap_report.csv` | Per-InChIKey counts and appears_in_both |
| Unresolved FDB IDs | `reports/unresolved_fdb_ids.csv` | Long FDB IDs from compound_gene that could not be mapped; columns: fdb_id_norm, count_in_compound_gene, has_master_row, inchikey_present, cid_present, name_present, top_source_files_seen |
| Name fallback matches | `reports/name_fallback_matches.csv` | Name-based matches (score, name_src, name_matched, inchikey) |
| Compound scan | `data/processed/canonical/compound_scan.json` | Only when `--write-scan` is used |

### Interpreting the reports

- **compound_identity_diagnostics.json**  
  - `pct_ingredient_compounds_resolved`: share of ingredient-side compounds that got an InChIKey.  
  - `pct_compound_gene_rows_resolved`: share of compound_gene rows that got an InChIKey.  
  - `pct_final_overlap`: share of ingredient_compound InChIKeys that also appear in compound_gene.  
  - If overlap is low, check `top_50_inchikey_*` and `top_20_*_fdb_id_patterns` to see ID patterns on each side.

- **reports/unresolved_fdb_ids.csv**  
  Lists FDB IDs that appear in compound_gene but could not be resolved to InChIKey.  
  - `count_in_compound_gene`: how often each ID appears.  
  - `has_master_row`: whether it exists in compound_master.  
  - `inchikey_present` / `cid_present` / `name_present`: whether the master row has that field.  
  - Use this to add or fix sources (harvest, registry, name fallback) for long IDs.

- **reports/name_fallback_matches.csv**  
  Name-based matches used when no direct FDB→InChIKey or CID→InChIKey was available (exact and fuzzy ≥95).

### Phase14 loader requirement

For a **full run** (not smoke), the Phase14 loader requires:

- Canonical files: `ingredient_compound_canonical.csv` and `compound_gene_canonical.csv` (or legacy compound_gene_links).
- **Overlap ≥ 5%**: if overlap between ingredient_compound and compound_gene canonical InChIKeys is &lt; 5%, the loader raises an error and points to `reports/unresolved_fdb_ids.csv` and `compound_identity_diagnostics.json`.

Use `--smoke` when building or debugging the identity layer so that low overlap does not fail the pipeline; fix unresolved IDs and re-run without `--smoke` for production.
