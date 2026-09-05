#!/usr/bin/env python3
"""
Build canonical targets table with enriched annotations.

Enriches ChEMBL targets with:
- Gene names and UniProt IDs
- Target classification
- Compound count
- Activity statistics

Output:
- data/processed/canonical/targets.parquet
"""

import pandas as pd
import logging
import yaml
import sys
import io
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

INTERIM_DIR = Path(paths["interim_data"]) / "chembl"
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"

def build_targets_table():
    """Build enriched targets table."""
    
    logger.info("="*60)
    logger.info("PHASE 4.4: Build Canonical Targets Table")
    logger.info("="*60)
    
    logger.info("\n⏳ Loading data...")
    
    # Load ChEMBL targets
    targets = pd.read_parquet(INTERIM_DIR / "targets.parquet")
    logger.info(f"Loaded {len(targets):,} ChEMBL targets")
    
    # Load compound-target edges
    compound_targets = pd.read_parquet(CANONICAL_DIR / "compound_targets.parquet")
    logger.info(f"Loaded {len(compound_targets):,} compound-target relationships")
    
    # Enrich targets with compound counts and activity stats
    logger.info("\n⏳ Computing target statistics...")
    
    target_stats = compound_targets.groupby('target_chembl_id').agg({
        'compound_id': 'nunique',  # Number of food compounds
        'pchembl_value': ['mean', 'median', 'max', 'min']
    }).reset_index()
    
    # Flatten column names
    target_stats.columns = [
        'target_chembl_id',
        'num_food_compounds',
        'avg_pchembl',
        'median_pchembl',
        'max_pchembl',
        'min_pchembl'
    ]
    
    # Merge with targets
    enriched_targets = targets.merge(
        target_stats,
        left_on='chembl_id',
        right_on='target_chembl_id',
        how='inner'
    )
    
    # Add target classification
    enriched_targets['target_class'] = enriched_targets['target_type'].map({
        'SINGLE PROTEIN': 'protein',
        'PROTEIN COMPLEX': 'complex',
        'PROTEIN FAMILY': 'family'
    }).fillna('other')
    
    # Select and rename columns
    final_targets = enriched_targets[[
        'tid',
        'chembl_id',
        'pref_name',
        'gene_name',
        'uniprot_accession',
        'target_type',
        'target_class',
        'organism',
        'num_food_compounds',
        'avg_pchembl',
        'median_pchembl',
        'max_pchembl',
        'min_pchembl'
    ]].rename(columns={
        'tid': 'target_id',
        'chembl_id': 'target_chembl_id',
        'pref_name': 'target_name'
    })
    
    # Sort by compound count
    final_targets = final_targets.sort_values('num_food_compounds', ascending=False)
    
    # Save
    output_path = CANONICAL_DIR / "targets.parquet"
    final_targets.to_parquet(output_path, index=False)
    logger.info(f"\n✅ Saved {len(final_targets):,} targets to {output_path}")
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("📊 Target Summary")
    logger.info("="*60)
    logger.info(f"Total targets: {len(final_targets):,}")
    logger.info(f"Targets with gene names: {final_targets['gene_name'].notna().sum():,}")
    logger.info(f"Targets with UniProt IDs: {final_targets['uniprot_accession'].notna().sum():,}")
    
    logger.info(f"\nTarget types:")
    for ttype, count in final_targets['target_type'].value_counts().items():
        logger.info(f"  {ttype}: {count:,}")
    
    logger.info(f"\nCompound distribution:")
    logger.info(f"  Mean compounds per target: {final_targets['num_food_compounds'].mean():.1f}")
    logger.info(f"  Median: {final_targets['num_food_compounds'].median():.0f}")
    logger.info(f"  Max: {final_targets['num_food_compounds'].max()}")
    
    # Top targets
    logger.info(f"\n🎯 Top 15 targets (by food compound count):")
    top_targets = final_targets[['target_name', 'gene_name', 'num_food_compounds', 'avg_pchembl']].head(15)
    for _, row in top_targets.iterrows():
        gene = row['gene_name'] if pd.notna(row['gene_name']) else '?'
        logger.info(f"  {row['target_name']} ({gene}): {row['num_food_compounds']} compounds (avg pChEMBL: {row['avg_pchembl']:.2f})")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Phase 4 complete!")
    logger.info("="*60)
    logger.info("\nOutputs:")
    logger.info(f"  1. Targets: {CANONICAL_DIR / 'targets.parquet'}")
    logger.info(f"  2. Compound-target edges: {CANONICAL_DIR / 'compound_targets.parquet'}")
    logger.info(f"  3. Network: data/processed/graph/compound_target_network.edgelist")
    
    logger.info("\nNext phase: Phase 5 - Pathway Mapping")

if __name__ == "__main__":
    build_targets_table()
