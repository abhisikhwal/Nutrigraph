#!/usr/bin/env python3
"""
Match food compounds to ChEMBL database.

Matching strategy:
1. Exact InChIKey match (connectivity layer - first 14 chars)
2. Report match statistics by compound class

Output:
- data/interim/chembl/food_compound_matches.parquet
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

CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"
INTERIM_DIR = Path(paths["interim_data"]) / "chembl"

def match_compounds():
    """Match food compounds to ChEMBL database."""
    
    logger.info("="*60)
    logger.info("PHASE 4.2: Match Food Compounds to ChEMBL")
    logger.info("="*60)
    
    logger.info("\n⏳ Loading data...")
    
    # Load food compounds (Phase 2)
    food_compounds = pd.read_parquet(CANONICAL_DIR / "compounds.parquet")
    logger.info(f"Loaded {len(food_compounds):,} food compounds")
    
    # Load ChEMBL compounds
    chembl_compounds = pd.read_parquet(INTERIM_DIR / "compounds.parquet")
    logger.info(f"Loaded {len(chembl_compounds):,} ChEMBL compounds")
    
    # Match by InChIKey connectivity layer (first 14 characters)
    logger.info("\n⏳ Matching by InChIKey connectivity layer...")
    logger.info("(Ignores stereochemistry - focuses on molecular skeleton)")
    
    # Extract connectivity layer
    food_compounds['inchikey_conn'] = food_compounds['inchikey'].str[:14]
    chembl_compounds['inchikey_conn'] = chembl_compounds['standard_inchi_key'].str[:14]
    
    # Remove nulls
    food_valid = food_compounds[food_compounds['inchikey_conn'].notna()].copy()
    chembl_valid = chembl_compounds[chembl_compounds['inchikey_conn'].notna()].copy()
    
    logger.info(f"Food compounds with valid InChIKey: {len(food_valid):,}")
    logger.info(f"ChEMBL compounds with valid InChIKey: {len(chembl_valid):,}")
    
    # Merge
    logger.info("\n⏳ Performing merge...")
    matches = food_valid.merge(
        chembl_valid[['chembl_id', 'inchikey_conn', 'molecular_weight', 'alogp']],
        on='inchikey_conn',
        how='inner',
        suffixes=('_food', '_chembl')
    )
    
    # Remove duplicates (keep first match)
    matches = matches.drop_duplicates(subset='compound_id', keep='first')
    
    match_rate = 100 * len(matches) / len(food_compounds)
    logger.info(f"\n✅ Matched {len(matches):,} / {len(food_compounds):,} compounds ({match_rate:.1f}%)")
    
    # Save matches
    output_path = INTERIM_DIR / "food_compound_matches.parquet"
    matches.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    # Summary by compound class
    if 'compound_class' in matches.columns:
        logger.info(f"\n📊 Matches by compound class:")
        class_matches = matches.groupby('compound_class').size().sort_values(ascending=False)
        for cls, count in class_matches.head(10).items():
            pct = 100 * count / len(matches)
            logger.info(f"  {cls}: {count:,} ({pct:.1f}%)")
    
    # Show example matches
    logger.info(f"\n📋 Example matches:")
    sample = matches[['compound_id', 'name', 'chembl_id', 'compound_class']].head(5)
    for _, row in sample.iterrows():
        logger.info(f"  {row['name']} ({row['compound_class']}) → {row['chembl_id']}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Compound matching complete!")
    logger.info("="*60)
    logger.info("\nNext step:")
    logger.info("  python scripts/phase4_bioactivity/03_build_compound_target_network.py")

if __name__ == "__main__":
    match_compounds()
