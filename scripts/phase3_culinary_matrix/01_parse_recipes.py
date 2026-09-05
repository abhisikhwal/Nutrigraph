#!/usr/bin/env python3
"""
Parse RecipeNLG recipes into structured format.

RecipeNLG format:
- title: Recipe name
- ingredients: List of ingredient strings (often noisy)
- directions: Cooking instructions
- link: Source URL
- source: Website origin
- NER: Named entities (ingredients extracted)

Challenge: Ingredient strings are noisy (e.g., "2 cups chopped onion")
Solution: Extract ingredient names and map to canonical IDs

Output:
- data/interim/recipenlg/recipes_parsed.parquet
- data/interim/recipenlg/recipe_ingredients_raw.parquet
"""

import pandas as pd
import logging
import yaml
from pathlib import Path
import re
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = Path(paths["raw_data"]) / "recipenlg"
INTERIM_DIR = Path(paths["interim_data"]) / "recipenlg"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def find_recipe_file():
    """Find RecipeNLG CSV/JSON file."""
    # Check main dir and subdirectories
    csv_files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("**/*.csv"))
    json_files = list(RAW_DIR.glob("*.json")) + list(RAW_DIR.glob("**/*.json"))
    
    if csv_files:
        return csv_files[0], 'csv'
    elif json_files:
        return json_files[0], 'json'
    else:
        raise FileNotFoundError(f"No CSV or JSON files in {RAW_DIR}")

def clean_ingredient_string(ingredient_str):
    """
    Extract ingredient name from noisy string.
    
    Examples:
    "2 cups chopped onion" → "onion"
    "1/2 teaspoon ground black pepper" → "black pepper"
    "3 cloves garlic, minced" → "garlic"
    """
    # Remove quantities (numbers, fractions, units)
    ingredient_str = re.sub(r'\d+[\./]?\d*', '', ingredient_str)
    ingredient_str = re.sub(r'\b(cup|tablespoon|teaspoon|pound|ounce|lb|oz|tsp|tbsp|gram|kg|ml|liter)s?\b', '', ingredient_str, flags=re.IGNORECASE)
    
    # Remove cooking methods
    cooking_terms = ['chopped', 'diced', 'minced', 'sliced', 'ground', 'fresh', 'dried', 'crushed', 'grated', 'peeled', 'canned', 'frozen']
    for term in cooking_terms:
        ingredient_str = re.sub(rf'\b{term}\b', '', ingredient_str, flags=re.IGNORECASE)
    
    # Remove punctuation and extra spaces
    ingredient_str = re.sub(r'[,\(\)\-]', ' ', ingredient_str)
    ingredient_str = ' '.join(ingredient_str.split()).strip().lower()
    
    return ingredient_str

def parse_recipenlg():
    """
    Parse RecipeNLG dataset.
    """
    logger.info("Loading RecipeNLG dataset...")
    recipe_file, file_type = find_recipe_file()
    
    logger.info(f"Loading {recipe_file} ({file_type} format)")
    
    # Load data
    if file_type == 'csv':
        try:
            df = pd.read_csv(recipe_file, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(recipe_file, encoding='latin1', low_memory=False)
    else:  # json
        df = pd.read_json(recipe_file, lines=True)
    
    logger.info(f"Loaded {len(df)} recipes")
    
    # Check columns
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Limit to first 10,000 for demo (remove in production)
    if len(df) > 10000:
        logger.info("Limiting to 10,000 recipes for demo")
        df = df.head(10000)
    
    # Parse ingredient lists
    # RecipeNLG typically has 'NER' column with extracted ingredients as JSON list
    if 'NER' in df.columns:
        logger.info("Using NER column for ingredient extraction")
        df['ingredients_parsed'] = df['NER'].apply(
            lambda x: json.loads(x) if pd.notna(x) and x != '[]' and isinstance(x, str) else (x if isinstance(x, list) else [])
        )
    elif 'ingredients' in df.columns:
        logger.info("Using ingredients column")
        # Ingredients might be newline-separated, comma-separated, or JSON
        def parse_ingredients(x):
            if pd.isna(x):
                return []
            x = str(x)
            # Try JSON first
            if x.startswith('['):
                try:
                    return json.loads(x)
                except:
                    pass
            # Split by newline or comma
            items = re.split(r'[\n,]', x)
            return [clean_ingredient_string(i) for i in items if i.strip()]
        
        df['ingredients_parsed'] = df['ingredients'].apply(parse_ingredients)
    else:
        logger.error("No ingredient column found")
        return
    
    # Filter out recipes with no ingredients
    df = df[df['ingredients_parsed'].apply(len) > 0]
    logger.info(f"Recipes with ingredients: {len(df)}")
    
    # Create recipe ID
    df['recipe_id'] = 'RCP_' + df.index.astype(str).str.zfill(8)
    
    # Extract cuisine if available (often in 'source' or inferred from URL)
    # Simplified: map source domains to cuisines
    cuisine_map = {
        'allrecipes': 'american',
        'foodnetwork': 'american',
        'epicurious': 'western',
        'bonappetit': 'western',
        'cookpad': 'asian',
        'bbcgoodfood': 'british',
        'food.com': 'american',
        'tasty': 'american',
    }
    
    if 'source' in df.columns:
        df['cuisine'] = df['source'].apply(
            lambda x: next((v for k, v in cuisine_map.items() if k in str(x).lower()), 'unknown')
        )
    else:
        df['cuisine'] = 'unknown'
    
    # Save parsed recipes
    recipes = df[[
        'recipe_id', 'title', 'ingredients_parsed', 'cuisine', 'source'
    ]].copy()
    
    recipes.to_parquet(INTERIM_DIR / "recipes_parsed.parquet", index=False)
    logger.info(f"Saved to {INTERIM_DIR / 'recipes_parsed.parquet'}")
    
    # Create recipe-ingredient pairs (flattened)
    recipe_ingredients = []
    
    for _, row in tqdm(recipes.iterrows(), total=len(recipes), desc="Flattening"):
        for ingredient in row['ingredients_parsed']:
            if ingredient:  # Skip empty strings
                recipe_ingredients.append({
                    'recipe_id': row['recipe_id'],
                    'ingredient_raw': str(ingredient).strip().lower(),
                    'cuisine': row['cuisine']
                })
    
    ri_df = pd.DataFrame(recipe_ingredients)
    ri_df.to_parquet(INTERIM_DIR / "recipe_ingredients_raw.parquet", index=False)
    logger.info(f"Saved {len(ri_df)} recipe-ingredient pairs")
    logger.info(f"Unique raw ingredients: {ri_df['ingredient_raw'].nunique()}")
    
    logger.info("✅ Recipe parsing complete")

if __name__ == "__main__":
    parse_recipenlg()
