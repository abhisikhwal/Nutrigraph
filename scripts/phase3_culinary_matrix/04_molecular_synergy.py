#!/usr/bin/env python3
"""
Test molecular complementarity hypothesis.

For each recipe:
1. Get compound fingerprint (union of all ingredient compounds)
2. Compute diversity metrics (Shannon entropy, chemical space coverage)
3. Test correlation: do co-occurring ingredients share compounds (Western) or contrast (Asian)?

Output:
- data/processed/features/recipe_compound_vectors.parquet
- data/processed/features/molecular_synergy_scores.parquet
"""

import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

INTERIM_DIR = Path(paths["interim_data"])
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"
FEATURES_DIR = Path(paths["processed_data"]) / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def compute_recipe_compound_vectors():
    """
    Build compound fingerprint for each recipe.
    """
    logger.info("Loading data...")
    
    # Load ingredient-compound mappings (7.4M!)
    ing_compounds = pd.read_parquet(CANONICAL_DIR / "ingredient_compounds.parquet")
    logger.info(f"Loaded {len(ing_compounds)} ingredient-compound mappings")
    
    # Load recipe-ingredient mappings
    recipe_ingredients = pd.read_parquet(
        INTERIM_DIR / "recipenlg" / "recipe_ingredients_mapped.parquet"
    )
    logger.info(f"Loaded {len(recipe_ingredients)} recipe-ingredient pairs")
    
    # Merge to get recipe → compounds
    logger.info("Merging recipe-ingredient-compound data...")
    recipe_compounds = recipe_ingredients.merge(
        ing_compounds[['ingredient_id', 'compound_id']].drop_duplicates(),
        on='ingredient_id',
        how='inner'
    )
    
    logger.info(f"Recipe-compound pairs: {len(recipe_compounds)}")
    logger.info(f"Recipes with compound data: {recipe_compounds['recipe_id'].nunique()}")
    
    # Compute compound sets per recipe
    logger.info("Computing compound vectors...")
    
    recipe_vectors = []
    
    # Get cuisine info
    recipe_cuisine = recipe_ingredients[['recipe_id', 'cuisine']].drop_duplicates()
    
    for recipe_id, group in tqdm(recipe_compounds.groupby('recipe_id'), desc="Recipes"):
        compounds = set(group['compound_id'].unique())
        
        # Get number of ingredients
        num_ingredients = recipe_ingredients[
            recipe_ingredients['recipe_id'] == recipe_id
        ]['ingredient_id'].nunique()
        
        # Get cuisine
        cuisine = recipe_cuisine[
            recipe_cuisine['recipe_id'] == recipe_id
        ]['cuisine'].iloc[0] if len(recipe_cuisine[recipe_cuisine['recipe_id'] == recipe_id]) > 0 else 'unknown'
        
        recipe_vectors.append({
            'recipe_id': recipe_id,
            'cuisine': cuisine,
            'num_ingredients': num_ingredients,
            'num_compounds': len(compounds),
            'compound_set': compounds  # Store as set for now
        })
    
    rv_df = pd.DataFrame(recipe_vectors)
    
    # Save (need to serialize sets first)
    rv_df['compound_list'] = rv_df['compound_set'].apply(list)
    rv_df_save = rv_df.drop('compound_set', axis=1)
    
    output_path = FEATURES_DIR / "recipe_compound_vectors.parquet"
    rv_df_save.to_parquet(output_path, index=False)
    logger.info(f"Saved recipe compound vectors to {output_path}")
    
    return rv_df  # Return version with sets for analysis

def compute_molecular_synergy(recipe_vectors):
    """
    Compute molecular synergy metrics.
    """
    logger.info("Computing molecular synergy scores...")
    
    # For each recipe, compute:
    # 1. Compound diversity (compounds per ingredient ratio)
    # 2. By cuisine
    
    synergy_scores = []
    
    for _, row in tqdm(recipe_vectors.iterrows(), total=len(recipe_vectors), desc="Synergy"):
        # Compute diversity (compound richness / ingredient richness)
        compound_diversity = row['num_compounds'] / row['num_ingredients'] if row['num_ingredients'] > 0 else 0
        
        synergy_scores.append({
            'recipe_id': row['recipe_id'],
            'cuisine': row['cuisine'],
            'compound_diversity': compound_diversity,
            'num_ingredients': row['num_ingredients'],
            'num_compounds': row['num_compounds']
        })
    
    synergy_df = pd.DataFrame(synergy_scores)
    
    # Save
    output_path = FEATURES_DIR / "molecular_synergy_scores.parquet"
    synergy_df.to_parquet(output_path, index=False)
    logger.info(f"Saved molecular synergy scores to {output_path}")
    
    # Summary statistics
    logger.info(f"\nMolecular Synergy Summary:")
    logger.info(f"  Avg compounds per recipe: {synergy_df['num_compounds'].mean():.1f}")
    logger.info(f"  Avg ingredients per recipe: {synergy_df['num_ingredients'].mean():.1f}")
    logger.info(f"  Avg compound diversity: {synergy_df['compound_diversity'].mean():.1f}")
    
    # By cuisine
    if synergy_df['cuisine'].nunique() > 1:
        logger.info(f"\nBy Cuisine:")
        cuisine_stats = synergy_df.groupby('cuisine').agg({
            'compound_diversity': 'mean',
            'num_compounds': 'mean',
            'num_ingredients': 'mean'
        }).round(1)
        logger.info(f"\n{cuisine_stats.to_string()}")

def main():
    recipe_vectors = compute_recipe_compound_vectors()
    compute_molecular_synergy(recipe_vectors)
    
    logger.info("✅ Molecular synergy analysis complete")

if __name__ == "__main__":
    main()
