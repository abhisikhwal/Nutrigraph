#!/usr/bin/env python3
"""
Process FULL RecipeNLG dataset (2.2M recipes) with checkpointing and progress tracking.

This is the production version of phase3_full_recipe_processing.ipynb
Designed for batch/server execution with automatic resume capability.

Estimated time: 2-4 hours
Memory: 16GB+ RAM recommended

Features:
- Automatic checkpointing every 100K recipes
- Resume from interruption
- Memory-efficient batch processing
- Progress tracking with ETA
- Comprehensive logging
"""

import pandas as pd
import numpy as np
import json
import re
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from difflib import SequenceMatcher
import gc
import sys
import io
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase3_full_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load config
with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = Path(paths["raw_data"]) / "recipenlg"
INTERIM_DIR = Path(paths["interim_data"]) / "recipenlg"
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"
GRAPH_DIR = Path(paths["processed_data"]) / "graph"
FEATURES_DIR = Path(paths["processed_data"]) / "features"
CHECKPOINT_DIR = INTERIM_DIR / "checkpoints"

# Create directories
for directory in [INTERIM_DIR, GRAPH_DIR, FEATURES_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100_000

# Manual mappings for common ingredients
MANUAL_MAPPINGS = {
    'salt': 'salt', 'pepper': 'pepper', 'black pepper': 'pepper, black',
    'olive oil': 'olive oil', 'vegetable oil': 'vegetable oil',
    'butter': 'butter', 'garlic': 'garlic', 'onion': 'onion',
    'onions': 'onion', 'tomato': 'tomato', 'tomatoes': 'tomato',
    'chicken': 'chicken', 'beef': 'beef', 'sugar': 'sugar',
    'flour': 'wheat flour', 'egg': 'egg', 'eggs': 'egg',
    'milk': 'milk', 'water': 'water',
    # Spices
    'turmeric': 'turmeric', 'cumin': 'cumin', 'coriander': 'coriander',
    'cinnamon': 'cinnamon', 'ginger': 'ginger', 'chili': 'chili pepper',
    'paprika': 'paprika', 'cardamom': 'cardamom', 'nutmeg': 'nutmeg',
    'clove': 'clove', 'cloves': 'clove', 'saffron': 'saffron',
    'basil': 'basil', 'oregano': 'oregano', 'thyme': 'thyme',
    'rosemary': 'rosemary', 'parsley': 'parsley',
}

def clean_ingredient_string(ingredient_str):
    """Extract ingredient name from noisy string."""
    ingredient_str = re.sub(r'\\d+[\\.\/]?\\d*', '', ingredient_str)
    ingredient_str = re.sub(
        r'\\b(cup|tablespoon|teaspoon|pound|ounce|lb|oz|tsp|tbsp|gram|kg|ml|liter)s?\\b',
        '', ingredient_str, flags=re.IGNORECASE
    )
    
    cooking_terms = ['chopped', 'diced', 'minced', 'sliced', 'ground', 'fresh',
                     'dried', 'crushed', 'grated', 'peeled', 'canned', 'frozen']
    for term in cooking_terms:
        ingredient_str = re.sub(rf'\\b{term}\\b', '', ingredient_str, flags=re.IGNORECASE)
    
    ingredient_str = re.sub(r'[,\\(\\)\\-]', ' ', ingredient_str)
    ingredient_str = ' '.join(ingredient_str.split()).strip().lower()
    
    return ingredient_str

def parse_ingredients(x):
    """Parse ingredients from column."""
    if pd.isna(x):
        return []
    
    if isinstance(x, str) and x.startswith('['):
        try:
            return json.loads(x)
        except:
            pass
    
    if isinstance(x, list):
        return x
    
    items = re.split(r'[\\n,]', str(x))
    return [clean_ingredient_string(i) for i in items if i.strip()]

def fuzzy_match_ingredient(ingredient_raw, canonical_names, canonical_lookup, threshold=0.6):
    """Fuzzy match ingredient to canonical ID."""
    ingredient_clean = ingredient_raw.lower().strip()
    
    if ingredient_clean in MANUAL_MAPPINGS:
        mapped_name = MANUAL_MAPPINGS[ingredient_clean]
        for canonical in canonical_names:
            if mapped_name.lower() in canonical or canonical in mapped_name.lower():
                return canonical, 100
    
    best_match = None
    best_score = 0
    
    for canonical in canonical_names:
        score = SequenceMatcher(None, ingredient_clean, canonical).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = canonical
    
    return best_match, int(best_score * 100)

def step1_parse_recipes(df):
    """Parse all recipes with checkpointing."""
    logger.info(f\"STEP 1: Parsing {len(df):,} recipes...\")\n    
    checkpoint_file = CHECKPOINT_DIR / \"parsing_checkpoint.parquet\"
    progress_file = CHECKPOINT_DIR / \"parsing_progress.json\"
    
    # Check for checkpoint
    if checkpoint_file.exists() and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        start_idx = progress['last_processed']
        parsed_recipes = pd.read_parquet(checkpoint_file)
        logger.info(f\"Resuming from recipe {start_idx:,}\")\n",
    else:
        start_idx = 0
        parsed_recipes = []
    
    recipes_list = []
    ing_col = 'NER' if 'NER' in df.columns else 'ingredients'
    
    for i in tqdm(range(start_idx, len(df)), desc=\"Parsing\"):
        row = df.iloc[i]
        ingredients = parse_ingredients(row[ing_col])
        
        if not ingredients:
            continue
        
        recipes_list.append({
            'recipe_id': f\"RCP_{i:08d}\",
            'title': row.get('title', ''),
            'ingredients_parsed': ingredients,
            'source': row.get('source', ''),
            'cuisine': 'unknown'
        })
        
        # Checkpoint
        if (i + 1) % BATCH_SIZE == 0:
            if isinstance(parsed_recipes, list):
                all_recipes = pd.DataFrame(recipes_list)
            else:
                all_recipes = pd.concat([parsed_recipes, pd.DataFrame(recipes_list)], ignore_index=True)
            
            all_recipes.to_parquet(checkpoint_file, index=False)
            with open(progress_file, 'w') as f:
                json.dump({'last_processed': i + 1}, f)
            
            logger.info(f\"Checkpoint saved: {i + 1:,} recipes\")
            recipes_list = []
            gc.collect()
    
    # Final save
    if recipes_list:
        if isinstance(parsed_recipes, list):
            all_recipes = pd.DataFrame(recipes_list)
        elif checkpoint_file.exists():
            parsed_recipes = pd.read_parquet(checkpoint_file)
            all_recipes = pd.concat([parsed_recipes, pd.DataFrame(recipes_list)], ignore_index=True)
        else:
            all_recipes = pd.DataFrame(recipes_list)
    else:
        all_recipes = pd.read_parquet(checkpoint_file)
    
    output_file = INTERIM_DIR / \"recipes_parsed_full.parquet\"
    all_recipes.to_parquet(output_file, index=False)
    
    # Clean up checkpoints
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    if progress_file.exists():
        progress_file.unlink()
    
    logger.info(f\"Step 1 complete: {len(all_recipes):,} recipes parsed\")\n    return all_recipes

def step2_flatten_ingredients(recipes_df):
    """Flatten to recipe-ingredient pairs."""
    logger.info(\"STEP 2: Flattening to recipe-ingredient pairs...\")\n    
    recipe_ingredients = []
    for _, row in tqdm(recipes_df.iterrows(), total=len(recipes_df), desc=\"Flattening\"):
        for ingredient in row['ingredients_parsed']:
            if ingredient:
                recipe_ingredients.append({
                    'recipe_id': row['recipe_id'],
                    'ingredient_raw': str(ingredient).strip().lower(),
                    'cuisine': row['cuisine']
                })
    
    ri_df = pd.DataFrame(recipe_ingredients)
    output_file = INTERIM_DIR / \"recipe_ingredients_raw_full.parquet\"
    ri_df.to_parquet(output_file, index=False)
    
    logger.info(f\"Step 2 complete: {len(ri_df):,} pairs ({ri_df['ingredient_raw'].nunique():,} unique)\")\n    return ri_df

def step3_map_ingredients(ri_df):
    """Map ingredients to canonical IDs."""
    logger.info(\"STEP 3: Mapping ingredients to canonical IDs...\")\n    
    # Load canonical
    canonical_df = pd.read_parquet(CANONICAL_DIR / \"ingredients.parquet\")
    canonical_names = canonical_df['canonical_name'].str.lower().tolist()
    canonical_lookup = dict(zip(
        canonical_df['canonical_name'].str.lower(),
        canonical_df['ingredient_id']
    ))
    
    logger.info(f\"Loaded {len(canonical_df):,} canonical ingredients\")\n    
    # Match unique ingredients
    unique_ingredients = ri_df['ingredient_raw'].unique()
    logger.info(f\"Matching {len(unique_ingredients):,} unique ingredients...\")\n    
    matches = []
    for ingredient_raw in tqdm(unique_ingredients, desc=\"Matching\"):
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
    result = ri_df.merge(match_df, on='ingredient_raw', how='left')
    matched = result[result['ingredient_id'].notna()]
    
    match_rate = len(matched) / len(result) * 100
    logger.info(f\"Matched: {len(matched):,} / {len(result):,} ({match_rate:.1f}%)\")\n    logger.info(f\"Unique ingredients: {matched['ingredient_id'].nunique():,}\")\n    
    output_file = INTERIM_DIR / \"recipe_ingredients_mapped_full.parquet\"
    matched.to_parquet(output_file, index=False)
    
    logger.info(f\"Step 3 complete: {output_file}\")\n    return matched

def step4_build_network(matched_df):
    \"\"\"Build co-occurrence network.\"\"\"
    logger.info(\"STEP 4: Building co-occurrence network...\")\n    
    edges = []
    for recipe_id, group in tqdm(matched_df.groupby('recipe_id'), desc=\"Co-occurrences\"):
        ingredients = group['ingredient_id'].unique()
        if len(ingredients) >= 2:
            for ing1, ing2 in combinations(sorted(ingredients), 2):
                edges.append((ing1, ing2))
    
    logger.info(f\"Edge instances: {len(edges):,}\")\n    
    edge_weights = Counter(edges)
    edgelist = pd.DataFrame([
        {'ingredient_1': edge[0], 'ingredient_2': edge[1], 'weight': weight}
        for edge, weight in edge_weights.items()
    ])
    
    output_file = GRAPH_DIR / \"ingredient_cooccurrence_full.edgelist\"
    edgelist.to_csv(output_file, index=False)
    
    logger.info(f\"Step 4 complete: {len(edgelist):,} edges saved to {output_file}\")\n    return edgelist

def step5_molecular_fingerprints(matched_df):
    \"\"\"Compute molecular fingerprints for recipes.\"\"\"
    logger.info(\"STEP 5: Computing molecular fingerprints...\")\n    
    # Load compound mappings
    ing_compounds = pd.read_parquet(CANONICAL_DIR / \"ingredient_compounds.parquet\")
    logger.info(f\"Loaded {len(ing_compounds):,} ingredient-compound mappings\")\n    
    # Merge
    recipe_compounds = matched_df.merge(
        ing_compounds[['ingredient_id', 'compound_id']].drop_duplicates(),
        on='ingredient_id',
        how='inner'
    )
    
    logger.info(f\"Recipe-compound pairs: {len(recipe_compounds):,}\")\n    logger.info(f\"Recipes with compounds: {recipe_compounds['recipe_id'].nunique():,}\")\n    
    # Compute vectors
    recipe_vectors = []
    for recipe_id, group in tqdm(recipe_compounds.groupby('recipe_id'), desc=\"Vectors\"):
        compounds = set(group['compound_id'].unique())
        num_ingredients = matched_df[matched_df['recipe_id'] == recipe_id]['ingredient_id'].nunique()
        cuisine = matched_df[matched_df['recipe_id'] == recipe_id]['cuisine'].iloc[0]
        
        recipe_vectors.append({
            'recipe_id': recipe_id,
            'cuisine': cuisine,
            'num_ingredients': num_ingredients,
            'num_compounds': len(compounds),
            'compound_list': list(compounds)
        })
    
    rv_df = pd.DataFrame(recipe_vectors)
    output_file = FEATURES_DIR / \"recipe_compound_vectors_full.parquet\"
    rv_df.to_parquet(output_file, index=False)
    
    logger.info(f\"Step 5 complete: {len(rv_df):,} fingerprints saved\")\n    return rv_df

def step6_synergy_scores(rv_df):
    \"\"\"Compute molecular synergy scores.\"\"\"
    logger.info(\"STEP 6: Computing molecular synergy scores...\")\n    
    rv_df['compound_diversity'] = rv_df['num_compounds'] / rv_df['num_ingredients']
    
    synergy_df = rv_df[['recipe_id', 'cuisine', 'compound_diversity',
                         'num_ingredients', 'num_compounds']]
    
    output_file = FEATURES_DIR / \"molecular_synergy_scores_full.parquet\"
    synergy_df.to_parquet(output_file, index=False)
    
    logger.info(f\"Step 6 complete: {output_file}\")\n    logger.info(f\"Avg compound diversity: {synergy_df['compound_diversity'].mean():.1f}\")\n    
    return synergy_df

def main():
    \"\"\"Main execution pipeline.\"\"\"
    start_time = datetime.now()
    logger.info(\"=\"*60)
    logger.info(\"PHASE 3 FULL DATASET PROCESSING\")\n    logger.info(\"=\"*60)
    
    try:
        # Load RecipeNLG
        logger.info(\"Loading RecipeNLG dataset...\")\n        recipe_files = list(RAW_DIR.glob(\"**/*.csv\"))
        if not recipe_files:
            raise FileNotFoundError(f\"No CSV files in {RAW_DIR}\")
        
        recipe_file = recipe_files[0]
        logger.info(f\"File: {recipe_file}\")\n        logger.info(f\"Size: {recipe_file.stat().st_size / 1024 / 1024:.1f} MB\")\n        
        try:
            df = pd.read_csv(recipe_file, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(recipe_file, encoding='latin1', low_memory=False)
        
        logger.info(f\"Loaded {len(df):,} recipes\")\n        
        # Step 1: Parse recipes
        recipes_df = step1_parse_recipes(df)
        del df
        gc.collect()
        
        # Step 2: Flatten
        ri_df = step2_flatten_ingredients(recipes_df)
        del recipes_df
        gc.collect()
        
        # Step 3: Map ingredients
        matched_df = step3_map_ingredients(ri_df)
        del ri_df
        gc.collect()
        
        # Step 4: Build network
        edge_df = step4_build_network(matched_df)
        
        # Step 5: Molecular fingerprints
        rv_df = step5_molecular_fingerprints(matched_df)
        
        # Step 6: Synergy scores
        synergy_df = step6_synergy_scores(rv_df)
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(\"\\n\" + \"=\"*60)
        logger.info(\"PROCESSING COMPLETE!\")\n        logger.info(\"=\"*60)
        logger.info(f\"Total time: {elapsed/60:.1f} minutes\")\n        logger.info(f\"Recipes processed: {len(synergy_df):,}\")\n        logger.info(f\"Co-occurrence edges: {len(edge_df):,}\")\n        logger.info(f\"Avg compound diversity: {synergy_df['compound_diversity'].mean():.1f}\")\n        logger.info(\"=\"*60)
        
        return True
        
    except KeyboardInterrupt:
        logger.warning(\"\\n⚠️  Processing interrupted!\")\n        logger.info(\"Checkpoints saved - you can resume by running this script again.\")\n        return False
    except Exception as e:
        logger.error(f\"\\n❌ Error: {e}\")\n        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == \"__main__\":
    success = main()
    sys.exit(0 if success else 1)
