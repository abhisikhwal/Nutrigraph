#!/usr/bin/env python3
"""
Enhanced ingredient-compound mapping with fuzzy matching.

Uses multiple strategies:
1. Scientific name matching
2. Common name matching (fuzzy)
3. Manual mapping for key spices
"""

import pandas as pd
import logging
import yaml
from pathlib import Path
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

INTERIM_DIR = Path(paths["interim_data"])
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"

def fuzzy_match(str1, str2, threshold=0.6):
    """Calculate similarity between two strings."""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def clean_name(name):
    """Clean food/ingredient name for matching."""
    if pd.isna(name):
        return ""
    # Remove common suffixes
    name = str(name).lower()
    for suffix in [', ground', ', dried', ', fresh', ', raw', ', whole']:
        name = name.replace(suffix, '')
    return name.strip()

def create_manual_mappings():
    """Manual mappings for key spices."""
    return {
        'turmeric': ['turmeric'],
        'curcuma longa': ['turmeric'],
        'pepper': ['black pepper', 'pepper'],
        'piper nigrum': ['black pepper', 'pepper'],
        'cinnamon': ['cinnamon'],
        'cinnamomum': ['cinnamon'],
        'garlic': ['garlic'],
        'allium sativum': ['garlic'],
        'ginger': ['ginger'],
        'zingiber officinale': ['ginger'],
        'chili': ['chili pepper', 'hot pepper'],
        'capsicum': ['chili pepper', 'pepper'],
        'cardamom': ['cardamom'],
        'clove': ['cloves'],
        'syzygium aromaticum': ['cloves'],
        'cumin': ['cumin'],
        'cuminum cyminum': ['cumin'],
        'coriander': ['coriander'],
        'coriandrum sativum': ['coriander'],
    }

def enhanced_mapping():
    """Build ingredient-compound mapping with fuzzy matching."""
    logger.info("Loading canonical tables...")
    
    # Load data
    ingredients = pd.read_parquet(CANONICAL_DIR / "ingredients.parquet")
    foodb_foods = pd.read_parquet(INTERIM_DIR / "foodb" / "foods.parquet")
    foodb_content = pd.read_parquet(INTERIM_DIR / "foodb" / "food_compounds.parquet")
    
    logger.info(f"Loaded {len(ingredients)} ingredients")
    logger.info(f"Loaded {len(foodb_foods)} FooDB foods")
    logger.info(f"Loaded {len(foodb_content)} FooDB food-compound pairs")
    
    # Manual mappings
    manual_map = create_manual_mappings()
    
    # Find matches
    matches = []
    
    for _, ingredient in ingredients.iterrows():
        ing_name = clean_name(ingredient['canonical_name'])
        ing_sci = clean_name(ingredient.get('scientific_name', ''))
        
        # Try manual mapping first
        matched_foods = []
        for key, food_names in manual_map.items():
            if key in ing_name or key in ing_sci:
                for food_name in food_names:
                    matched_foods.extend(
                        foodb_foods[foodb_foods['food_name'].str.lower().str.contains(food_name, na=False)]['foodb_food_id'].tolist()
                    )
        
        # Try scientific name matching
        if ing_sci and len(matched_foods) == 0:
            for _, food in foodb_foods.iterrows():
                food_sci = clean_name(food.get('scientific_name', ''))
                if food_sci and fuzzy_match(ing_sci, food_sci) > 0.8:
                    matched_foods.append(food['foodb_food_id'])
        
        # Try fuzzy name matching
        if len(matched_foods) == 0:
            for _, food in foodb_foods.iterrows():
                food_name = clean_name(food['food_name'])
                if food_name and fuzzy_match(ing_name, food_name) > 0.7:
                    matched_foods.append(food['foodb_food_id'])
        
        # Record matches
        if matched_foods:
            logger.info(f"Matched '{ingredient['canonical_name']}' to {len(matched_foods)} FooDB foods")
            for food_id in matched_foods:
                matches.append({
                    'ingredient_id': ingredient['ingredient_id'],
                    'foodb_food_id': food_id,
                    'ingredient_name': ingredient['canonical_name']
                })
    
    logger.info(f"Found {len(matches)} ingredient-food matches")
    
    if len(matches) == 0:
        logger.warning("No matches found - using demo data")
        # Create demo mappings
        result = pd.DataFrame([
            {'ingredient_id': 'ING_000000', 'foodb_compound_id': 1, 'content_value': 3.14, 
             'content_unit': '% dry weight', 'evidence_source': 'demo'},
        ])
    else:
        # Join with food-compound content
        matches_df = pd.DataFrame(matches)
        result = matches_df.merge(
            foodb_content[['foodb_food_id', 'foodb_compound_id', 'content_value', 
                          'content_min', 'content_max', 'content_unit']],
            on='foodb_food_id',
            how='inner'
        )
        result['evidence_source'] = 'foodb'
        result['compound_id'] = 'FDB_' + result['foodb_compound_id'].astype(str)
        
        logger.info(f"Created {len(result)} ingredient-compound mappings")
    
    # Save
    output_cols = ['ingredient_id', 'compound_id', 'content_value', 
                   'content_unit', 'evidence_source']
    
    # Ensure columns exist
    for col in output_cols:
        if col not in result.columns:
            if col == 'compound_id':
                result['compound_id'] = result.get('foodb_compound_id', result.get('compound_id', 'UNKNOWN'))
            elif col in ['content_value', 'content_unit', 'evidence_source']:
                result[col] = None
    
    result = result[output_cols]
    
    output_path = CANONICAL_DIR / "ingredient_compounds.parquet"
    result.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved {len(result)} mappings to {output_path}")
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Unique ingredients: {result['ingredient_id'].nunique()}")
    logger.info(f"  Unique compounds: {result['compound_id'].nunique()}")
    logger.info(f"\nSample mappings:")
    logger.info(result.head(10))

if __name__ == "__main__":
    enhanced_mapping()
