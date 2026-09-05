#!/usr/bin/env python3
"""
Build ingredient co-occurrence network from recipes.

Network structure:
- Nodes: Ingredients (canonical IDs)
- Edges: Co-occurrence in same recipe
- Edge weights: Number of recipes where pair appears together

Also creates per-cuisine subnetworks.

Output:
- data/processed/graph/ingredient_cooccurrence.edgelist
- data/processed/graph/cuisine_networks/american.edgelist
- data/processed/graph/cuisine_networks/asian.edgelist
- etc.
"""

import pandas as pd
import networkx as nx
import logging
import yaml
from pathlib import Path
from itertools import combinations
from collections import Counter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

INTERIM_DIR = Path(paths["interim_data"])
GRAPH_DIR = Path(paths["processed_data"]) / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

def build_cooccurrence_network():
    """
    Build ingredient co-occurrence network.
    """
    logger.info("Loading recipe-ingredient mappings...")
    
    recipe_ingredients = pd.read_parquet(
        INTERIM_DIR / "recipenlg" / "recipe_ingredients_mapped.parquet"
    )
    
    logger.info(f"Loaded {len(recipe_ingredients)} recipe-ingredient pairs")
    logger.info(f"Unique recipes: {recipe_ingredients['recipe_id'].nunique()}")
    logger.info(f"Unique ingredients: {recipe_ingredients['ingredient_id'].nunique()}")
    
    # Build co-occurrence edges
    logger.info("Computing co-occurrences...")
    
    edges = []
    
    # Group by recipe
    for recipe_id, group in tqdm(recipe_ingredients.groupby('recipe_id'), desc="Recipes"):
        ingredients = group['ingredient_id'].unique()
        
        # Create edges for all pairs
        if len(ingredients) >= 2:
            for ing1, ing2 in combinations(sorted(ingredients), 2):
                edges.append((ing1, ing2))
    
    logger.info(f"Total edge instances: {len(edges)}")
    
    # Count edge weights
    edge_weights = Counter(edges)
    
    # Create edgelist
    edgelist = [
        {'ingredient_1': edge[0], 'ingredient_2': edge[1], 'weight': weight}
        for edge, weight in edge_weights.items()
    ]
    
    edge_df = pd.DataFrame(edgelist)
    
    # Save global network
    output_path = GRAPH_DIR / "ingredient_cooccurrence.edgelist"
    edge_df.to_csv(output_path, index=False)
    logger.info(f"Saved global co-occurrence network: {len(edge_df)} edges")
    
    # Build per-cuisine networks
    cuisine_dir = GRAPH_DIR / "cuisine_networks"
    cuisine_dir.mkdir(exist_ok=True)
    
    for cuisine in recipe_ingredients['cuisine'].unique():
        if cuisine == 'unknown':
            continue
        
        logger.info(f"Building network for {cuisine} cuisine...")
        
        cuisine_recipes = recipe_ingredients[recipe_ingredients['cuisine'] == cuisine]
        cuisine_edges = []
        
        for recipe_id, group in cuisine_recipes.groupby('recipe_id'):
            ingredients = group['ingredient_id'].unique()
            if len(ingredients) >= 2:
                for ing1, ing2 in combinations(sorted(ingredients), 2):
                    cuisine_edges.append((ing1, ing2))
        
        if cuisine_edges:
            cuisine_weights = Counter(cuisine_edges)
            cuisine_edgelist = [
                {'ingredient_1': edge[0], 'ingredient_2': edge[1], 'weight': weight}
                for edge, weight in cuisine_weights.items()
            ]
            
            cuisine_df = pd.DataFrame(cuisine_edgelist)
            cuisine_output = cuisine_dir / f"{cuisine}.edgelist"
            cuisine_df.to_csv(cuisine_output, index=False)
            logger.info(f"  {cuisine}: {len(cuisine_df)} edges")
    
    logger.info("✅ Co-occurrence network building complete")
    
    # Network statistics
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['ingredient_1'], row['ingredient_2'], weight=row['weight'])
    
    logger.info(f"\nNetwork Statistics:")
    logger.info(f"  Nodes: {G.number_of_nodes()}")
    logger.info(f"  Edges: {G.number_of_edges()}")
    logger.info(f"  Density: {nx.density(G):.4f}")
    
    if G.number_of_nodes() > 0:
        # Get largest connected component for clustering
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc)
        logger.info(f"  Largest component: {len(largest_cc)} nodes")
        logger.info(f"  Avg clustering (largest component): {nx.average_clustering(G_cc):.4f}")

if __name__ == "__main__":
    build_cooccurrence_network()
