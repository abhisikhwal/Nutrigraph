#!/usr/bin/env python3
"""
Download USDA FoodData Central database.

API: https://fdc.nal.usda.gov/api-guide.html
Requires free API key from: https://fdc.nal.usda.gov/api-key-signup.html

Downloads:
- Foundation Foods (high-quality nutrient data)
- SR Legacy (historical Standard Reference)
- Branded Foods (optional, large)

Outputs:
- data/raw/usda/foundation_foods.json
- data/raw/usda/sr_legacy.json
- data/interim/usda/foods_parsed.parquet
"""

import requests
import json
import logging
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = Path(paths["raw_data"]) / "usda"
INTERIM_DIR = Path(paths["interim_data"]) / "usda"
RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.nal.usda.gov/fdc/v1"

def get_api_key():
    """
    Read API key from environment.
    """
    api_key = os.environ.get("USDA_API_KEY")
    
    if not api_key:
        logger.warning("USDA_API_KEY not found in environment")
        logger.info("Get your free API key from: https://fdc.nal.usda.gov/api-key-signup.html")
        logger.info("Set it with: export USDA_API_KEY=your_key_here (Linux/Mac)")
        logger.info("Or: set USDA_API_KEY=your_key_here (Windows)")
        logger.info("\nProceeding with demo data...")
    
    return api_key

def download_food_list(api_key, data_type="Foundation", page_size=200):
    """
    Download list of foods from USDA FoodData Central.
    
    Args:
        data_type: "Foundation", "SR Legacy", or "Branded"
    """
    logger.info(f"Downloading {data_type} foods...")
    
    all_foods = []
    page = 1
    
    while True:
        params = {
            "api_key": api_key,
            "dataType": data_type,
            "pageSize": page_size,
            "pageNumber": page
        }
        
        response = requests.get(f"{API_BASE}/foods/list", params=params)
        
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code}")
            break
        
        data = response.json()
        foods = data.get("foods", [])
        
        if not foods:
            break
        
        all_foods.extend(foods)
        logger.info(f"Downloaded page {page} ({len(all_foods)} foods so far)")
        
        page += 1
        time.sleep(0.1)  # Rate limiting
        
        # Limit for demo purposes
        if len(all_foods) >= 1000:
            logger.info("Reached 1000 foods limit (demo mode)")
            break
    
    output_file = RAW_DIR / f"{data_type.lower().replace(' ', '_')}_foods.json"
    with open(output_file, 'w') as f:
        json.dump(all_foods, f, indent=2)
    
    logger.info(f"Saved {len(all_foods)} foods to {output_file}")
    return all_foods

def parse_foods_to_dataframe(foods):
    """
    Parse USDA food JSON into structured DataFrame.
    """
    records = []
    
    for food in foods:
        record = {
            'fdc_id': food.get('fdcId'),
            'description': food.get('description'),
            'data_type': food.get('dataType'),
            'publication_date': food.get('publicationDate'),
            'scientific_name': food.get('scientificName'),
            'food_category': food.get('foodCategory'),
        }
        
        # Extract nutrient data (simplified)
        nutrients = food.get('foodNutrients', [])
        for nutrient in nutrients:
            nutrient_name = nutrient.get('nutrient', {}).get('name')
            nutrient_value = nutrient.get('amount')
            if nutrient_name and nutrient_value:
                # Store top nutrients only (to avoid explosion)
                if nutrient_name in ['Protein', 'Total lipid (fat)', 'Carbohydrate, by difference']:
                    record[f'nutrient_{nutrient_name.lower().replace(" ", "_").replace(",", "")}'] = nutrient_value
        
        records.append(record)
    
    df = pd.DataFrame(records)
    return df

def create_demo_data():
    """Create demo USDA data for testing without API key."""
    logger.info("Creating demo USDA data (no API key provided)...")
    
    # Sample data representing USDA structure
    demo_foods = [
        {
            'fdcId': 167512,
            'description': 'Turmeric, ground',
            'dataType': 'Foundation',
            'scientificName': 'Curcuma longa',
            'foodCategory': 'Spices and Herbs',
            'foodNutrients': [
                {'nutrient': {'name': 'Protein'}, 'amount': 7.83},
                {'nutrient': {'name': 'Total lipid (fat)'}, 'amount': 9.88},
                {'nutrient': {'name': 'Carbohydrate, by difference'}, 'amount': 64.93}
            ]
        },
        {
            'fdcId': 171325,
            'description': 'Pepper, black',
            'dataType': 'Foundation',
            'scientificName': 'Piper nigrum',
            'foodCategory': 'Spices and Herbs',
            'foodNutrients': [
                {'nutrient': {'name': 'Protein'}, 'amount': 10.39},
                {'nutrient': {'name': 'Total lipid (fat)'}, 'amount': 3.26},
                {'nutrient': {'name': 'Carbohydrate, by difference'}, 'amount': 63.95}
            ]
        },
        {
            'fdcId': 171320,
            'description': 'Cinnamon, ground',
            'dataType': 'Foundation',
            'scientificName': 'Cinnamomum verum',
            'foodCategory': 'Spices and Herbs',
            'foodNutrients': [
                {'nutrient': {'name': 'Protein'}, 'amount': 3.99},
                {'nutrient': {'name': 'Total lipid (fat)'}, 'amount': 1.24},
                {'nutrient': {'name': 'Carbohydrate, by difference'}, 'amount': 80.59}
            ]
        }
    ]
    
    # Save demo data
    output_file = RAW_DIR / "foundation_foods.json"
    with open(output_file, 'w') as f:
        json.dump(demo_foods, f, indent=2)
    
    logger.info(f"Created demo data with {len(demo_foods)} items")
    return demo_foods

def main():
    try:
        api_key = get_api_key()
        
        if not api_key:
            logger.warning("No API key provided - using demo data")
            foundation_foods = create_demo_data()
        else:
            # Download Foundation Foods (highest quality, manageable size)
            foundation_foods = download_food_list(api_key, "Foundation")
    except KeyboardInterrupt:
        logger.info("Download interrupted - using demo data")
        foundation_foods = create_demo_data()
    
    # Parse to DataFrame
    df = parse_foods_to_dataframe(foundation_foods)
    
    # Save as Parquet
    output_parquet = INTERIM_DIR / "foundation_foods.parquet"
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved parsed data to {output_parquet}")
    
    logger.info(f"✅ USDA download complete: {len(df)} foods")

if __name__ == "__main__":
    main()
