"""
Phase14: Dose-response proxy — ingredient frequency, compound count, dose_proxy_A/B/AB.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .id_normalization import to_ingredient_id

logger = logging.getLogger(__name__)


def _ts() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def ingredient_frequency(recipe_ingredients: pd.DataFrame) -> Dict[str, float]:
    """Global prevalence: count of recipes containing ingredient / total recipes."""
    if recipe_ingredients.empty or "recipe_id" not in recipe_ingredients.columns or "ingredient_id" not in recipe_ingredients.columns:
        return {}
    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
    n_recipes = ri["recipe_id"].nunique()
    cnt = ri.groupby("ingredient_id").size()
    return (cnt / n_recipes).to_dict()


def ingredient_compound_count(mediation_edges: pd.DataFrame) -> Dict[str, int]:
    """Number of compounds per ingredient from HAS_COMPOUND edges."""
    out: Dict[str, int] = {}
    for _, e in mediation_edges[mediation_edges["edge_type"] == "HAS_COMPOUND"].iterrows():
        ing = str(e["source_id"])
        out[ing] = out.get(ing, 0) + 1
    return out


def add_dose_proxy(
    pair_mediation: pd.DataFrame,
    recipe_ingredients: pd.DataFrame,
    mediation_edges: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add dose_proxy_A, dose_proxy_B, dose_proxy_AB (geometric mean or min of A,B).
    Proxy based on: ingredient frequency in recipes, within-recipe count if available, compound count.
    """
    df = pair_mediation.copy()
    freq = ingredient_frequency(recipe_ingredients)
    cmp_count = ingredient_compound_count(mediation_edges)
    def proxy(ing: str) -> float:
        f = freq.get(to_ingredient_id(ing), 0.0)
        c = cmp_count.get(to_ingredient_id(ing), 0)
        return max(1e-6, f * (1.0 + 0.1 * c))
    pa = df["ingA_id"].map(proxy)
    pb = df["ingB_id"].map(proxy)
    df["dose_proxy_A"] = pa
    df["dose_proxy_B"] = pb
    df["dose_proxy_AB"] = np.sqrt(pa * pb)
    return df
