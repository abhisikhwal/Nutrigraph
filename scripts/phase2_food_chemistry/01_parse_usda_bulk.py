#!/usr/bin/env python3
"""
Parse USDA FoodData Central bulk CSV files.

Expected files (user downloads from: https://fdc.nal.usda.gov/download-datasets.html):
- food.csv
- nutrient.csv  
- food_nutrient.csv
- food_component.csv (optional, has some compound data)

This extracts nutrient composition data which we'll later map to metabolic pathways.

Output:
- data/interim/usda/food_nutrients_full.parquet
"""

import pandas as pd
import logging
import yaml
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = Path(paths["raw_data"]) / "usda"
INTERIM_DIR = Path(paths["interim_data"]) / "usda"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def find_csv_file(pattern):
    """Find CSV file matching pattern in USDA directory."""
    matches = list(RAW_DIR.glob(f"**/{pattern}"))
    if not matches:
        logger.warning(f"No file found matching '{pattern}' in {RAW_DIR}")
        return None
    if len(matches) > 1:
        logger.warning(f"Multiple files found for '{pattern}', using first: {matches[0]}")
    return matches[0]

def create_demo_nutrient_data():
    """Create demo nutrient data for testing."""
    logger.info("Creating demo nutrient data (bulk files not found)...")
    
    # Create demo data matching USDA structure
    demo_data = pd.DataFrame({
        'fdc_id': [167512, 171325, 171320] * 5,
        'description': ['Turmeric, ground', 'Pepper, black', 'Cinnamon, ground'] * 5,
        'data_type': ['Foundation Food'] * 15,
        'nutrient_id': [1003, 1004, 1005, 1008, 1079, 1087, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097],
        'nutrient_name': ['Protein', 'Total lipid (fat)', 'Carbohydrate', 'Energy', 'Fiber', 
                         'Calcium', 'Iron', 'Magnesium', 'Phosphorus', 'Potassium',
                         'Sodium', 'Zinc', 'Copper', 'Manganese', 'Selenium'],
        'amount': [7.83, 9.88, 64.93, 312, 21.1, 183, 41.42, 193, 268, 2525, 
                  38, 4.35, 0.603, 7.833, 4.5],
        'unit_name': ['g', 'g', 'g', 'kcal', 'g', 'mg', 'mg', 'mg', 'mg', 'mg',
                     'mg', 'mg', 'mg', 'mg', 'ug']
    })
    
    return demo_data

def parse_usda_bulk():
    """
    Parse USDA bulk CSV files into a unified nutrient table.
    """
    # Find the CSV files
    food_file = find_csv_file("food.csv")
    nutrient_file = find_csv_file("nutrient.csv")
    food_nutrient_file = find_csv_file("food_nutrient.csv")
    
    if not all([food_file, nutrient_file, food_nutrient_file]):
        logger.warning("Missing USDA bulk CSV files")
        logger.info("Download from: https://fdc.nal.usda.gov/download-datasets.html")
        logger.info("Place files in: data/raw/usda/")
        logger.info("Using demo data for now...")
        
        result = create_demo_nutrient_data()
        
        # Save demo data
        output_path = INTERIM_DIR / "food_nutrients_full.parquet"
        result.to_parquet(output_path, index=False)
        logger.info(f"✅ Saved {len(result)} demo records to {output_path}")
        return
    
    logger.info("Loading USDA CSV files...")
    
    # Load foods
    logger.info(f"Reading {food_file.name}...")
    foods = pd.read_csv(food_file, low_memory=False)
    logger.info(f"Loaded {len(foods)} foods")
    
    # Load nutrients
    logger.info(f"Reading {nutrient_file.name}...")
    nutrients = pd.read_csv(nutrient_file)
    logger.info(f"Loaded {len(nutrients)} nutrients")
    
    # Load food-nutrient relationships
    logger.info(f"Reading {food_nutrient_file.name}...")
    food_nutrients = pd.read_csv(food_nutrient_file, low_memory=False)
    logger.info(f"Loaded {len(food_nutrients)} food-nutrient pairs")
    
    # Merge to create full table
    logger.info("Merging tables...")
    merged = food_nutrients.merge(
        foods[['fdc_id', 'description', 'data_type']],
        on='fdc_id',
        how='left'
    )
    
    merged = merged.merge(
        nutrients[['id', 'name', 'unit_name']],
        left_on='nutrient_id',
        right_on='id',
        how='left'
    )
    
    # Clean and select columns
    result = merged[[
        'fdc_id', 'description', 'data_type',
        'nutrient_id', 'name', 'amount', 'unit_name'
    ]].rename(columns={'name': 'nutrient_name'})
    
    # Save
    output_path = INTERIM_DIR / "food_nutrients_full.parquet"
    result.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved {len(result)} records to {output_path}")
    
    # Summary stats
    logger.info(f"Unique foods: {result['fdc_id'].nunique()}")
    logger.info(f"Unique nutrients: {result['nutrient_id'].nunique()}")

if __name__ == "__main__":
    parse_usda_bulk()
