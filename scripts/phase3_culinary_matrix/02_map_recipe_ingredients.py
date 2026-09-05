#!/usr/bin/env python3
"""
Map noisy recipe ingredient strings to canonical ingredient IDs.

Uses fuzzy matching with thresholds + manual curation for common items.

Output:
- data/interim/recipenlg/recipe_ingredients_mapped.parquet
"""

import pandas as pd
import logging
import yaml
from pathlib import Path
from difflib import SequenceMatcher
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

INTERIM_DIR = Path(paths["interim_data"])
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"

# Manual mappings for common ingredients (expand as needed)
MANUAL_MAPPINGS = {
    'salt': 'salt',
    'pepper': 'pepper',
    'black pepper': 'pepper, black',
    'olive oil': 'olive oil',
    'vegetable oil': 'vegetable oil',
    'butter': 'butter',
    'garlic': 'garlic',
    'onion': 'onion',
    'onions': 'onion',
    'tomato': 'tomato',
    'tomatoes': 'tomato',
    'chicken': 'chicken',
    'beef': 'beef',
    'sugar': 'sugar',
    'flour': 'wheat flour',
    'egg': 'egg',
    'eggs': 'egg',
    'milk': 'milk',
    'water': 'water',
    # Spices (high priority)
    'turmeric': 'turmeric',
    'cumin': 'cumin',
    'coriander': 'coriander',
    'cinnamon': 'cinnamon',
    'ginger': 'ginger',
    'chili': 'chili pepper',
    'chili pepper': 'chili pepper',
    'paprika': 'paprika',
    'cardamom': 'cardamom',
    'nutmeg': 'nutmeg',
    'clove': 'clove',
    'cloves': 'clove',
    'saffron': 'saffron',
    'basil': 'basil',
    'oregano': 'oregano',
    'thyme': 'thyme',
    'rosemary': 'rosemary',
    'parsley': 'parsley',
    'cilantro': 'coriander',
    'bay leaf': 'bay leaf',
    'cayenne': 'cayenne pepper',
}

def fuzzy_match_ingredient(ingredient_raw, canonical_names, canonical_lookup, threshold=0.6):
    """
    Fuzzy match ingredient string to canonical name.
    """
    # Check manual mappings first
    ingredient_clean = ingredient_raw.lower().strip()
    
    if ingredient_clean in MANUAL_MAPPINGS:
        mapped_name = MANUAL_MAPPINGS[ingredient_clean]
        # Find in canonical
        for canonical in canonical_names:
            if mapped_name.lower() in canonical or canonical in mapped_name.lower():
                return canonical, 100
    
    # Fuzzy match
    best_match = None
    best_score = 0
    
    for canonical in canonical_names:
        score = SequenceMatcher(None, ingredient_clean, canonical).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = canonical
    
    return best_match, int(best_score * 100)

def map_ingredients():
    """
    Map recipe ingredients to canonical IDs.
    """
    logger.info("Loading data...")
    
    # Load canonical ingredients
    ingredients = pd.read_parquet(CANONICAL_DIR / "ingredients.parquet")
    canonical_names = ingredients['canonical_name'].str.lower().tolist()
    canonical_lookup = dict(zip(
        ingredients['canonical_name'].str.lower(),
        ingredients['ingredient_id']
    ))
    
    logger.info(f"Loaded {len(ingredients)} canonical ingredients")
    
    # Load recipe ingredients
    recipe_ingredients = pd.read_parquet(
        INTERIM_DIR / "recipenlg" / "recipe_ingredients_raw.parquet"
    )
    logger.info(f"Loaded {len(recipe_ingredients)} recipe-ingredient pairs")
    logger.info(f"Unique raw ingredients: {recipe_ingredients['ingredient_raw'].nunique()}")
    
    # Match ingredients
    logger.info("Matching ingredients (this may take a while)...")
    
    unique_ingredients = recipe_ingredients['ingredient_raw'].unique()
    matches = []
    
    for ingredient_raw in tqdm(unique_ingredients, desc="Matching"):
        matched_name, score = fuzzy_match_ingredient(
            ingredient_raw, canonical_names, canonical_lookup
        )
        
        matches.append({
            'ingredient_raw': ingredient_raw,
            'matched_name': matched_name,
            'match_score': score,
            'ingredient_id': canonical_lookup.get(matched_name.lower()) if matched_name else None
        })
    
    match_df = pd.DataFrame(matches)
    
    # Merge back
    result = recipe_ingredients.merge(
        match_df,
        on='ingredient_raw',
        how='left'
    )
    
    # Filter to successfully matched ingredients
    matched = result[result['ingredient_id'].notna()]
    
    logger.info(f"Successfully matched: {len(matched)} / {len(result)} ({100*len(matched)/len(result):.1f}%)")
    logger.info(f"Unique ingredients matched: {matched['ingredient_id'].nunique()}")
    
    # Save
    output_path = INTERIM_DIR / "recipenlg" / "recipe_ingredients_mapped.parquet"
    matched.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    # Save unmatched for manual curation
    unmatched = result[result['ingredient_id'].isna()]
    if len(unmatched) > 0:
        unmatched_counts = unmatched['ingredient_raw'].value_counts().head(100)
        
        unmatched_path = INTERIM_DIR / "recipenlg" / "unmatched_ingredients.csv"
        unmatched_counts.to_csv(unmatched_path)
        logger.info(f"Saved top unmatched ingredients to {unmatched_path}")
        logger.info("Review this file and add to MANUAL_MAPPINGS to improve coverage")
    
    logger.info("✅ Ingredient mapping complete")

if __name__ == "__main__":
    map_ingredients()
