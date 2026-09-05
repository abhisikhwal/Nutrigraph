"""
Build ingredient co-occurrence networks from recipe data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from itertools import combinations
from collections import Counter

logger = logging.getLogger(__name__)


class CooccurrenceNetwork:
    """
    Build and analyze ingredient co-occurrence networks.
    """
    
    def __init__(self):
        logger.info("Initialized CooccurrenceNetwork")
    
    def build_cooccurrence_matrix(
        self,
        recipe_ingredient_map: pd.DataFrame,
        recipe_col: str = 'recipe_id',
        ingredient_col: str = 'ingredient_id',
        min_recipe_count: int = 5
    ) -> pd.DataFrame:
        """
        Build co-occurrence matrix from recipes.
        
        Args:
            recipe_ingredient_map: Long-format recipe-ingredient mapping
            recipe_col: Column with recipe IDs
            ingredient_col: Column with ingredient IDs
            min_recipe_count: Minimum times ingredient appears
            
        Returns:
            Co-occurrence matrix (ingredient x ingredient)
        """
        logger.info("Building co-occurrence matrix...")
        
        # Filter rare ingredients
        ingredient_counts = recipe_ingredient_map[ingredient_col].value_counts()
        common_ingredients = ingredient_counts[
            ingredient_counts >= min_recipe_count
        ].index
        
        filtered = recipe_ingredient_map[
            recipe_ingredient_map[ingredient_col].isin(common_ingredients)
        ]
        
        logger.info(
            f"Using {len(common_ingredients)} ingredients "
            f"(filtered from {len(ingredient_counts)})"
        )
        
        # Group ingredients by recipe
        recipe_groups = filtered.groupby(recipe_col)[ingredient_col].apply(list)
        
        # Count co-occurrences
        cooccurrence = Counter()
        for ingredients in recipe_groups:
            # All pairs in this recipe
            for pair in combinations(sorted(ingredients), 2):
                cooccurrence[pair] += 1
        
        # Convert to DataFrame (edgelist format)
        edges = []
        for (ing1, ing2), count in cooccurrence.items():
            edges.append({
                'ingredient_1': ing1,
                'ingredient_2': ing2,
                'cooccurrence_count': count
            })
        
        edgelist = pd.DataFrame(edges)
        
        logger.info(f"Built network with {len(edgelist)} edges")
        
        return edgelist
    
    def calculate_pmi(
        self,
        edgelist: pd.DataFrame,
        recipe_ingredient_map: pd.DataFrame,
        count_col: str = 'cooccurrence_count'
    ) -> pd.DataFrame:
        """
        Calculate Pointwise Mutual Information (PMI) scores.
        
        PMI measures how much more often two ingredients co-occur
        than would be expected by chance.
        
        Args:
            edgelist: Co-occurrence edgelist
            recipe_ingredient_map: Full recipe-ingredient mapping
            count_col: Column with co-occurrence counts
            
        Returns:
            Edgelist with PMI scores added
        """
        logger.info("Calculating PMI scores...")
        
        n_recipes = recipe_ingredient_map['recipe_id'].nunique()
        ingredient_counts = recipe_ingredient_map['ingredient_id'].value_counts()
        
        edgelist = edgelist.copy()
        
        # Calculate PMI for each pair
        pmi_scores = []
        for _, row in edgelist.iterrows():
            ing1 = row['ingredient_1']
            ing2 = row['ingredient_2']
            count_pair = row[count_col]
            
            # P(ing1, ing2)
            p_pair = count_pair / n_recipes
            
            # P(ing1), P(ing2)
            p_ing1 = ingredient_counts.get(ing1, 0) / n_recipes
            p_ing2 = ingredient_counts.get(ing2, 0) / n_recipes
            
            # PMI = log2(P(x,y) / (P(x) * P(y)))
            if p_ing1 > 0 and p_ing2 > 0:
                pmi = np.log2(p_pair / (p_ing1 * p_ing2))
            else:
                pmi = 0.0
            
            pmi_scores.append(pmi)
        
        edgelist['pmi'] = pmi_scores
        
        logger.info(f"PMI range: [{edgelist['pmi'].min():.2f}, {edgelist['pmi'].max():.2f}]")
        
        return edgelist
    
    def find_strong_pairs(
        self,
        edgelist: pd.DataFrame,
        min_cooccurrence: int = 10,
        min_pmi: float = 0.0,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Find strongly co-occurring ingredient pairs.
        
        Args:
            edgelist: Co-occurrence edgelist with PMI scores
            min_cooccurrence: Minimum co-occurrence count
            min_pmi: Minimum PMI score
            top_k: Return only top K pairs (by PMI)
            
        Returns:
            Filtered edgelist
        """
        filtered = edgelist[
            (edgelist['cooccurrence_count'] >= min_cooccurrence) &
            (edgelist['pmi'] >= min_pmi)
        ].copy()
        
        filtered = filtered.sort_values('pmi', ascending=False)
        
        if top_k is not None:
            filtered = filtered.head(top_k)
        
        logger.info(f"Found {len(filtered)} strong pairs")
        
        return filtered
