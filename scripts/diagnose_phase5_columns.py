"""
Diagnose Phase 5 data schema issues.

This script:
1. Inspects all Phase 4 output files
2. Shows actual column names
3. Displays sample data
4. Generates corrected Section 1 code for Phase 5
"""

import pandas as pd
from pathlib import Path
import yaml

# Load paths
with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"
INTERIM_DIR = Path(paths["interim_data"]) / "chembl"

print("=" * 70)
print("PHASE 5 COLUMN DIAGNOSTIC")
print("=" * 70)

# File 1: compound_targets.parquet
print("\n1. COMPOUND-TARGET EDGES FILE")
print("-" * 70)

ct_file = CANONICAL_DIR / "compound_targets.parquet"
if ct_file.exists():
    ct = pd.read_parquet(ct_file)
    print(f"File exists: {ct_file}")
    print(f"   Rows: {len(ct):,}")
    print(f"\nColumns ({len(ct.columns)}):")
    for i, col in enumerate(ct.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\nSample data (first 3 rows):")
    print(ct.head(3).to_string())
    
    print(f"\nTarget ID column candidates:")
    target_cols = [col for col in ct.columns if 'target' in col.lower() or 'chembl' in col.lower()]
    for col in target_cols:
        unique_count = ct[col].nunique()
        print(f"   - {col}: {unique_count:,} unique values")
else:
    print(f"File not found: {ct_file}")

# File 2: targets.parquet (enriched)
print("\n\n2. ENRICHED TARGETS FILE")
print("-" * 70)

targets_file = CANONICAL_DIR / "targets.parquet"
if targets_file.exists():
    targets = pd.read_parquet(targets_file)
    print(f"File exists: {targets_file}")
    print(f"   Rows: {len(targets):,}")
    print(f"\nColumns ({len(targets.columns)}):")
    for i, col in enumerate(targets.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\nSample data (first 3 rows):")
    print(targets.head(3).to_string())
    
    print(f"\nImportant columns check:")
    id_cols = [col for col in targets.columns if 'chembl' in col.lower() or 'id' in col.lower()]
    name_cols = [col for col in targets.columns if 'name' in col.lower() or 'gene' in col.lower()]
    
    print(f"   ID columns: {id_cols}")
    print(f"   Name columns: {name_cols}")
    
else:
    print(f"File not found: {targets_file}")

# File 3: targets.parquet (interim - ChEMBL raw)
print("\n\n3. INTERIM TARGETS FILE (ChEMBL)")
print("-" * 70)

interim_targets_file = INTERIM_DIR / "targets.parquet"
if interim_targets_file.exists():
    interim_targets = pd.read_parquet(interim_targets_file)
    print(f"File exists: {interim_targets_file}")
    print(f"   Rows: {len(interim_targets):,}")
    print(f"\nColumns ({len(interim_targets.columns)}):")
    for i, col in enumerate(interim_targets.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\nSample data (first 3 rows):")
    print(interim_targets.head(3).to_string())
else:
    print(f"File not found: {interim_targets_file}")

# Generate corrected code
print("\n\n" + "=" * 70)
print("CORRECTED SECTION 1 CODE")
print("=" * 70)

if ct_file.exists() and interim_targets_file.exists():
    # Detect correct column names
    ct = pd.read_parquet(ct_file)
    targets = pd.read_parquet(interim_targets_file)
    
    # Find target ID column in compound_targets
    ct_target_col = None
    for col in ['target_chembl_id', 'chembl_id', 'target_id']:
        if col in ct.columns:
            ct_target_col = col
            break
    
    # Find target ID column in targets
    targets_id_col = None
    for col in ['chembl_id', 'target_chembl_id', 'target_id']:
        if col in targets.columns:
            targets_id_col = col
            break
    
    # Find gene name column
    gene_col = None
    for col in ['gene_name', 'gene_symbol', 'gene']:
        if col in targets.columns:
            gene_col = col
            break
    
    # Find target name column
    name_col = None
    for col in ['target_name', 'pref_name', 'name']:
        if col in targets.columns:
            name_col = col
            break
    
    print("\nDetected columns:")
    print(f"   Compound-targets target ID: '{ct_target_col}'")
    print(f"   Targets ID column: '{targets_id_col}'")
    print(f"   Gene name column: '{gene_col}'")
    print(f"   Target name column: '{name_col}'")
    
    print("\nUse this code for Section 1:")
    print("\n" + "=" * 70)
    print(f"""# Load compound-target edges
compound_targets_file = CANONICAL_DIR / "compound_targets.parquet"
if not compound_targets_file.exists():
    raise FileNotFoundError(f"Compound-targets file not found: {{compound_targets_file}}")

compound_targets = pd.read_parquet(compound_targets_file)
print(f"Loaded {{len(compound_targets):,}} compound-target edges")

# Get active targets
active_target_ids = compound_targets['{ct_target_col}'].unique()
print(f"Found {{len(active_target_ids):,}} unique targets with food compound interactions")

# Load all targets metadata from ChEMBL
targets_file = CANONICAL_DIR.parent.parent / "interim" / "chembl" / "targets.parquet"
if not targets_file.exists():
    raise FileNotFoundError(f"Targets file not found: {{targets_file}}")

all_targets = pd.read_parquet(targets_file)
print(f"Loaded {{len(all_targets):,}} total targets from ChEMBL")

# Filter to active targets
targets = all_targets[all_targets['{targets_id_col}'].isin(active_target_ids)].copy()
print(f"Filtered to {{len(targets):,}} targets with food compound interactions")

# Display target info
print(f"\\nTarget information:")
if 'target_type' in targets.columns:
    print(f"\\n   Target types:")
    for ttype, count in targets['target_type'].value_counts().items():
        print(f"     {{ttype}}: {{count}}")

print(f"\\n   Targets with gene names: {{targets['{gene_col}'].notna().sum():,}}")
print(f"   Targets with UniProt IDs: {{targets['uniprot_accession'].notna().sum():,}}")

# Create unified gene list for pathway queries
targets['gene_symbol'] = targets['{gene_col}'].fillna(targets['{name_col}'])

gene_list = targets['gene_symbol'].dropna().unique().tolist()
print(f"\\nGene list created: {{len(gene_list):,}} unique gene symbols")

# Sample genes
print(f"\\nSample genes: {{', '.join(gene_list[:15])}}...")

# Display sample targets
print(f"\\nSample targets:")
display(targets[['{targets_id_col}', '{name_col}', '{gene_col}', 'target_type', 'uniprot_accession']].head(10))

print_memory_usage()
""")
    print("=" * 70)
    
else:
    print("\nCannot generate code - files not found")
    print("\nCheck that you have:")
    print(f"  1. {ct_file}")
    print(f"  2. {interim_targets_file}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
