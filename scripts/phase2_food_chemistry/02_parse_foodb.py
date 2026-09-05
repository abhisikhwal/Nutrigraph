#!/usr/bin/env python3
"""
Parse FooDB CSV files to extract ingredient-compound mappings.

Key tables:
- Food.csv: Food items
- Compound.csv: Chemical compounds
- Content.csv: Food-compound relationships (with concentrations)

Output:
- data/interim/foodb/foods.parquet
- data/interim/foodb/compounds.parquet
- data/interim/foodb/food_compounds.parquet
"""

import pandas as pd
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = Path(paths["raw_data"]) / "foodb"
INTERIM_DIR = Path(paths["interim_data"]) / "foodb"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Check for CSV files in subdirectory (common after tar extraction)
CSV_DIR = RAW_DIR / "foodb_2020_04_07_csv"
if not CSV_DIR.exists():
    CSV_DIR = RAW_DIR

def check_foodb_files():
    """Check if FooDB CSV files exist."""
    required_files = ['Food.csv', 'Compound.csv', 'Content.csv']
    
    # Try both the main dir and subdirectory
    for search_dir in [CSV_DIR, RAW_DIR]:
        all_found = True
        for filename in required_files:
            filepath = search_dir / filename
            if not filepath.exists():
                all_found = False
                break
        
        if all_found:
            # Update CSV_DIR to the correct location
            globals()['CSV_DIR'] = search_dir
            logger.info(f"Found FooDB CSV files in: {search_dir}")
            return True
    
    logger.error(f"Missing required FooDB files")
    logger.info("Download FooDB from: https://foodb.ca/downloads")
    logger.info(f"Place extracted CSV files in: {RAW_DIR}")
    return False

def parse_foodb():
    """Parse FooDB CSV files."""
    
    if not check_foodb_files():
        logger.warning("FooDB files not available - skipping")
        logger.info("This is OK for research-only or commercial projects")
        logger.info("Alternative: Use PubChem + manual compound curation")
        return
    
    logger.info("Loading FooDB files...")
    
    # Load foods
    logger.info("Reading Food.csv...")
    foods = pd.read_csv(CSV_DIR / "Food.csv")
    logger.info(f"Loaded {len(foods)} foods")
    
    # Load compounds
    logger.info("Reading Compound.csv...")
    compounds = pd.read_csv(CSV_DIR / "Compound.csv")
    logger.info(f"Loaded {len(compounds)} compounds")
    
    # Load food-compound content
    logger.info("Reading Content.csv...")
    contents = pd.read_csv(CSV_DIR / "Content.csv")
    logger.info(f"Loaded {len(contents)} food-compound relationships")
    
    # Process foods
    foods_clean = foods[[
        'id', 'name', 'name_scientific', 'description',
        'food_group', 'food_subgroup'
    ]].rename(columns={
        'id': 'foodb_food_id',
        'name': 'food_name',
        'name_scientific': 'scientific_name'
    })
    
    # Process compounds
    compounds_clean = compounds[[
        'id', 'public_id', 'name', 'cas_number',
        'moldb_smiles', 'moldb_inchikey', 'kingdom', 'state'
    ]].rename(columns={
        'id': 'foodb_compound_id',
        'public_id': 'foodb_public_id',
        'name': 'compound_name',
        'moldb_smiles': 'smiles',
        'moldb_inchikey': 'inchikey'
    })
    
    # Process content (food-compound relationships)
    content_clean = contents[[
        'food_id', 'source_id', 'orig_content',
        'orig_min', 'orig_max', 'orig_unit'
    ]].rename(columns={
        'food_id': 'foodb_food_id',
        'source_id': 'foodb_compound_id',
        'orig_content': 'content_value',
        'orig_min': 'content_min',
        'orig_max': 'content_max',
        'orig_unit': 'content_unit'
    })
    
    # Save processed data
    foods_clean.to_parquet(INTERIM_DIR / "foods.parquet", index=False)
    logger.info(f"Saved foods to {INTERIM_DIR / 'foods.parquet'}")
    
    compounds_clean.to_parquet(INTERIM_DIR / "compounds.parquet", index=False)
    logger.info(f"Saved compounds to {INTERIM_DIR / 'compounds.parquet'}")
    
    content_clean.to_parquet(INTERIM_DIR / "food_compounds.parquet", index=False)
    logger.info(f"Saved food-compound mappings to {INTERIM_DIR / 'food_compounds.parquet'}")
    
    logger.info("✅ FooDB parsing complete")
    
    # Print summary
    logger.info(f"\nSummary:")
    logger.info(f"  Foods: {len(foods_clean)}")
    logger.info(f"  Compounds: {len(compounds_clean)}")
    logger.info(f"  Food-compound pairs: {len(content_clean)}")

if __name__ == "__main__":
    parse_foodb()
