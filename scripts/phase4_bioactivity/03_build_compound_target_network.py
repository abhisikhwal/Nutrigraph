#!/usr/bin/env python3
"""
Build compound-target bioactivity network.

Links food compounds → ChEMBL compounds → human protein targets

Filters for:
- Active compounds only (pChEMBL >= 6, i.e., IC50/Ki <= 1 µM)
- Direct assays only (not predicted)

Output:
- data/processed/graph/compound_target_network.edgelist
- data/processed/canonical/compound_targets.parquet
"""

import pandas as pd
import logging
import yaml
import sys
import io
from pathlib import Path
from tqdm import tqdm

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

INTERIM_DIR = Path(paths["interim_data"]) / "chembl"
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"
GRAPH_DIR = Path(paths["processed_data"]) / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Activity threshold (pChEMBL >= 6 means IC50/Ki <= 1 µM)
PCHEMBL_THRESHOLD = 6.0

def build_network():
    """Build compound-target bioactivity network."""
    
    logger.info("="*60)
    logger.info("PHASE 4.3: Build Compound-Target Network")
    logger.info("="*60)
    
    logger.info("\n⏳ Loading data...")
    
    # Load matched food compounds
    food_matches = pd.read_parquet(INTERIM_DIR / "food_compound_matches.parquet")
    logger.info(f"Loaded {len(food_matches):,} food compound matches")
    
    # Load ChEMBL activities
    activities = pd.read_parquet(INTERIM_DIR / "activities.parquet")
    logger.info(f"Loaded {len(activities):,} bioactivity records")
    
    # Load targets
    targets = pd.read_parquet(INTERIM_DIR / "targets.parquet")
    logger.info(f"Loaded {len(targets):,} human targets")
    
    # Filter for ACTIVE compounds only (pChEMBL >= 6)
    logger.info(f"\n⏳ Filtering for active compounds (pChEMBL >= {PCHEMBL_THRESHOLD})...")
    logger.info(f"   (IC50/Ki/Kd <= 1 µM)")
    
    active = activities[activities['pchembl_value'] >= PCHEMBL_THRESHOLD].copy()
    logger.info(f"Active records: {len(active):,} / {len(activities):,} ({100*len(active)/len(activities):.1f}%)")
    
    # Link food compounds to activities
    logger.info("\n⏳ Linking food compounds to bioactivities...")
    
    food_activities = food_matches.merge(
        active,
        left_on='chembl_id',
        right_on='compound_chembl_id',
        how='inner'
    )
    
    logger.info(f"Food compound activities: {len(food_activities):,}")
    logger.info(f"Unique food compounds with activities: {food_activities['compound_id'].nunique():,}")
    
    # Create compound-target edges
    logger.info("\n⏳ Building compound-target network...")
    
    edges = food_activities.groupby(['compound_id', 'target_chembl_id']).agg({
        'pchembl_value': 'max',  # Take strongest activity
        'standard_type': 'first',
        'standard_value': 'min'   # Take most potent (lowest IC50/Ki)
    }).reset_index()
    
    logger.info(f"Network edges: {len(edges):,}")
    logger.info(f"  Unique compounds: {edges['compound_id'].nunique():,}")
    logger.info(f"  Unique targets: {edges['target_chembl_id'].nunique():,}")
    
    # Add target information
    edges = edges.merge(
        targets[['chembl_id', 'pref_name', 'gene_name', 'uniprot_accession']],
        left_on='target_chembl_id',
        right_on='chembl_id',
        how='left'
    )
    
    # Save network edgelist
    edgelist = edges[['compound_id', 'target_chembl_id', 'pchembl_value']].rename(columns={
        'pchembl_value': 'weight'
    })
    
    output_path = GRAPH_DIR / "compound_target_network.edgelist"
    edgelist.to_csv(output_path, index=False)
    logger.info(f"\n✅ Saved network to {output_path}")
    
    # Save compound-target table with details
    compound_targets = edges[[
        'compound_id',
        'target_chembl_id',
        'pref_name',
        'gene_name',
        'uniprot_accession',
        'pchembl_value',
        'standard_type',
        'standard_value'
    ]].copy()
    
    output_path = CANONICAL_DIR / "compound_targets.parquet"
    compound_targets.to_parquet(output_path, index=False)
    logger.info(f"Saved detailed table to {output_path}")
    
    # Network statistics
    logger.info("\n" + "="*60)
    logger.info("📊 Network Statistics")
    logger.info("="*60)
    logger.info(f"Total edges: {len(edges):,}")
    logger.info(f"Compounds (nodes): {edges['compound_id'].nunique():,}")
    logger.info(f"Targets (nodes): {edges['target_chembl_id'].nunique():,}")
    
    # Degree distribution
    compound_degrees = edges.groupby('compound_id').size()
    target_degrees = edges.groupby('target_chembl_id').size()
    
    logger.info(f"\nCompound degree (targets per compound):")
    logger.info(f"  Mean: {compound_degrees.mean():.1f}")
    logger.info(f"  Median: {compound_degrees.median():.0f}")
    logger.info(f"  Max: {compound_degrees.max()}")
    
    logger.info(f"\nTarget degree (compounds per target):")
    logger.info(f"  Mean: {target_degrees.mean():.1f}")
    logger.info(f"  Median: {target_degrees.median():.0f}")
    logger.info(f"  Max: {target_degrees.max()}")
    
    # Top targets
    logger.info(f"\n🎯 Top 10 targets (by compound count):")
    top_targets = edges.groupby(['target_chembl_id', 'pref_name']).size().sort_values(ascending=False).head(10)
    for (target_id, target_name), count in top_targets.items():
        logger.info(f"  {target_name}: {count} compounds")
    
    # Top compounds
    logger.info(f"\n🧪 Top 10 compounds (by target count):")
    top_compounds = edges.groupby('compound_id').size().sort_values(ascending=False).head(10)
    for compound_id, count in top_compounds.items():
        # Get compound name
        comp_name = food_matches[food_matches['compound_id'] == compound_id]['name'].iloc[0] if len(food_matches[food_matches['compound_id'] == compound_id]) > 0 else compound_id
        logger.info(f"  {comp_name}: {count} targets")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Network building complete!")
    logger.info("="*60)
    logger.info("\nNext step:")
    logger.info("  python scripts/phase4_bioactivity/04_build_targets_table.py")

if __name__ == "__main__":
    build_network()
