#!/usr/bin/env python3
"""
Create ingredient → compound mappings.

Links:
- Ingredient master (Phase 1) → Compound master (Phase 2)
- Via FooDB food-compound content table
- Via known spice-compound associations

Output:
- data/processed/canonical/ingredient_compounds.parquet

Schema:
- ingredient_id
- compound_id
- content_value (concentration if available)
- content_unit
- evidence_source (foodb, literature, etc.)
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

def create_demo_mappings():
    """Create demo ingredient-compound mappings based on known associations."""
    logger.info("Creating demo ingredient-compound mappings (known spice compounds)...")
    
    # Known spice-compound associations
    mappings = [
        # Turmeric (ING_000000) - Curcuminoids
        {'ingredient_id': 'ING_000000', 'ingredient_name': 'Turmeric, ground', 
         'compound_id': 969516, 'compound_name': 'Curcumin', 
         'content_value': 3.14, 'content_unit': '% dry weight', 'evidence_source': 'literature'},
        
        # Black pepper (ING_000001) - Piperine
        {'ingredient_id': 'ING_000001', 'ingredient_name': 'Pepper, black',
         'compound_id': 638024, 'compound_name': 'Piperine',
         'content_value': 5.0, 'content_unit': '% dry weight', 'evidence_source': 'literature'},
        
        # Cinnamon (ING_000002) - Cinnamaldehyde
        {'ingredient_id': 'ING_000002', 'ingredient_name': 'Cinnamon, ground',
         'compound_id': 637511, 'compound_name': 'Cinnamaldehyde',
         'content_value': 0.5, 'content_unit': '% dry weight', 'evidence_source': 'literature'},
        
        # Additional polyphenols in spices
        {'ingredient_id': 'ING_000000', 'ingredient_name': 'Turmeric, ground',
         'compound_id': 5280343, 'compound_name': 'Quercetin',
         'content_value': 0.02, 'content_unit': '% dry weight', 'evidence_source': 'literature'},
        
        {'ingredient_id': 'ING_000002', 'ingredient_name': 'Cinnamon, ground',
         'compound_id': 3314, 'compound_name': 'Eugenol',
         'content_value': 1.0, 'content_unit': '% dry weight', 'evidence_source': 'literature'},
    ]
    
    return pd.DataFrame(mappings)

def map_ingredients_compounds():
    """Build ingredient-compound mapping table."""
    logger.info("Loading canonical tables...")
    
    # Load ingredients from Phase 1
    ingredients = pd.read_parquet(CANONICAL_DIR / "ingredients.parquet")
    logger.info(f"Loaded {len(ingredients)} ingredients")
    
    # Load compounds from current phase
    compounds = pd.read_parquet(CANONICAL_DIR / "compounds.parquet")
    logger.info(f"Loaded {len(compounds)} compounds")
    
    # Load FooDB mappings if available
    foodb_content_path = INTERIM_DIR / "foodb" / "food_compounds.parquet"
    foodb_foods_path = INTERIM_DIR / "foodb" / "foods.parquet"
    
    if foodb_content_path.exists() and foodb_foods_path.exists():
        logger.info("FooDB data available - attempting to use it")
        foodb_content = pd.read_parquet(foodb_content_path)
        foodb_foods = pd.read_parquet(foodb_foods_path)
        
        logger.info(f"Loaded {len(foodb_content)} FooDB food-compound pairs")
        
        # Merge food names
        mapping = foodb_content.merge(
            foodb_foods[['foodb_food_id', 'food_name']],
            on='foodb_food_id'
        )
        
        # Try to link to ingredient master via name matching
        mapping = mapping.merge(
            ingredients[['ingredient_id', 'canonical_name']],
            left_on='food_name',
            right_on='canonical_name',
            how='inner'
        )
        
        result = mapping[[
            'ingredient_id',
            'foodb_compound_id',
            'content_value',
            'content_min',
            'content_max',
            'content_unit'
        ]]
        
        result['evidence_source'] = 'foodb'
        result['compound_id'] = 'FDB_' + result['foodb_compound_id'].astype(str)
        
        logger.info(f"Matched {len(result)} FooDB ingredient-compound pairs")
    else:
        logger.warning("FooDB data not available - using demo mappings")
        result = create_demo_mappings()
        
        # Keep only relevant columns
        result = result[[
            'ingredient_id',
            'compound_id',
            'content_value',
            'content_unit',
            'evidence_source'
        ]]
    
    # Save
    output_path = CANONICAL_DIR / "ingredient_compounds.parquet"
    result.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved {len(result)} ingredient-compound mappings to {output_path}")
    
    logger.info(f"Unique ingredients: {result['ingredient_id'].nunique()}")
    logger.info(f"Unique compounds: {result['compound_id'].nunique()}")
    
    # Show sample mappings
    logger.info("\nSample mappings:")
    logger.info(result.head(10))

if __name__ == "__main__":
    map_ingredients_compounds()
