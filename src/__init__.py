"""
NutriGraph
==============

A knowledge graph and pipeline linking foods → chemistry → human molecular targets,
with measured vs inferred honesty on every edge.

Package layout (under src/):
- data: Dataset downloaders, parsers, and validators
- features: Feature engineering (fingerprints and related helpers)
- models: Machine learning models for prediction and integration
- graph: Network analysis and co-occurrence matrices
- utils: Shared utilities (logging, I/O, ontology matching)
"""

__version__ = "0.1.0"
__author__ = "Abhinav Sikhwal"

# Make key functions accessible at package level
# Example: from food_genome import load_config
from .utils.io import load_config, save_parquet, load_parquet

__all__ = [
    "load_config",
    "save_parquet",
    "load_parquet",
]
