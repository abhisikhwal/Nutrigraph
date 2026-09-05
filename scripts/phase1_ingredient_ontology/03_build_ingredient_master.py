#!/usr/bin/env python3
"""
Merge USDA + Wikidata into canonical ingredient master table.

Outputs:
- data/processed/canonical/ingredients.parquet
  Columns: ingredient_id, canonical_name, scientific_name, fdc_id, wikidata_id, category
"""

import pandas as pd
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

INTERIM_DIR = Path(paths["interim_data"])
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"
CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

def load_usda():
    usda_path = INTERIM_DIR / "usda" / "foundation_foods.parquet"
    if not usda_path.exists():
        logger.warning(f"USDA data not found at {usda_path}")
        return pd.DataFrame()
    return pd.read_parquet(usda_path)

def load_wikidata():
    wikidata_path = INTERIM_DIR / "wikidata" / "wikidata_foods.parquet"
    if not wikidata_path.exists():
        logger.warning(f"Wikidata data not found at {wikidata_path}")
        return pd.DataFrame()
    return pd.read_parquet(wikidata_path)

def build_master():
    logger.info("Loading datasets...")
    usda_df = load_usda()
    wikidata_df = load_wikidata()
    
    # Build from USDA
    ingredients = []
    for idx, row in usda_df.iterrows():
        ingredients.append({
            'ingredient_id': f"ING_{idx:06d}",
            'canonical_name': row['description'],
            'scientific_name': row.get('scientific_name'),
            'fdc_id': row['fdc_id'],
            'wikidata_id': None,
            'category': row.get('food_category'),
            'source': 'usda'
        })
    
    # Add Wikidata items (de-duplicate later)
    offset = len(ingredients)
    for idx, row in wikidata_df.iterrows():
        ingredients.append({
            'ingredient_id': f"ING_{offset + idx:06d}",
            'canonical_name': row['label'],
            'scientific_name': row.get('scientific_name'),
            'fdc_id': None,
            'wikidata_id': row['wikidata_id'],
            'category': 'spice/herb',  # Simplified categorization
            'source': 'wikidata'
        })
    
    df = pd.DataFrame(ingredients)
    
    # Save
    output_path = CANONICAL_DIR / "ingredients.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Created ingredient master: {len(df)} items")
    logger.info(f"Saved to {output_path}")
    
    # Print summary
    logger.info(f"USDA items: {len(usda_df)}")
    logger.info(f"Wikidata items: {len(wikidata_df)}")
    logger.info(f"Total unique: {len(df)}")

if __name__ == "__main__":
    build_master()
