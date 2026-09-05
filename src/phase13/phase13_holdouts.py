"""
Phase 13 v3 — Holdout tests: ingredient holdout and cuisine holdout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .phase13_config import Phase13Config
from .phase13_utils import _ts


def run_ingredient_holdout(
    config: Phase13Config,
    top_interactions: pd.DataFrame,
    all_ingredients: List[str],
    random_seed: int,
) -> Dict[str, Any]:
    """
    Randomly hold out holdout_ingredient_frac of ingredients.
    Return overlap stats: how many top-K interactions (by pair) still have both ingredients in training set.
    """
    rng = np.random.default_rng(random_seed)
    n_hold = max(1, int(len(all_ingredients) * config.holdout_ingredient_frac))
    holdout_ings = set(rng.choice(all_ingredients, size=n_hold, replace=False).tolist())
    train_ings = set(all_ingredients) - holdout_ings

    # Top pairs (dedupe by pair)
    pairs = top_interactions[["ingA_id", "ingB_id"]].drop_duplicates()
    in_train = 0
    both_held = 0
    for _, row in pairs.iterrows():
        a, b = row["ingA_id"], row["ingB_id"]
        if a in train_ings and b in train_ings:
            in_train += 1
        elif a in holdout_ings or b in holdout_ings:
            both_held += 1
    return {
        "holdout_type": "ingredient",
        "n_ingredients_total": len(all_ingredients),
        "n_ingredients_held_out": n_hold,
        "n_top_pairs": len(pairs),
        "n_pairs_both_in_train": in_train,
        "n_pairs_at_least_one_held": both_held,
        "overlap_frac": in_train / len(pairs) if len(pairs) > 0 else 0,
        "timestamp": _ts(),
    }


def run_cuisine_holdout(
    config: Phase13Config,
    top_interactions: pd.DataFrame,
    cuisine_or_cluster_values: List[str],
    random_seed: int,
) -> Dict[str, Any]:
    """
    Hold out holdout_cuisine_frac of cuisines/clusters.
    Return stats for comparison: we do not recompute full pipeline; we report what fraction of top
    interactions involve cuisines that would be in training set.
    """
    rng = np.random.default_rng(random_seed)
    cuisines = list(set(cuisine_or_cluster_values))
    n_hold = max(1, int(len(cuisines) * config.holdout_cuisine_frac))
    holdout_cuisines = set(rng.choice(cuisines, size=n_hold, replace=False).tolist())
    return {
        "holdout_type": "cuisine",
        "n_cuisines_total": len(cuisines),
        "n_cuisines_held_out": n_hold,
        "held_out_cuisines": list(holdout_cuisines),
        "note": "Full recompute on remaining cuisines not implemented in this stub; use for reporting.",
        "timestamp": _ts(),
    }


def write_holdout_report(
    output_dir: Path,
    ingredient_holdout: Dict[str, Any],
    cuisine_holdout: Dict[str, Any],
) -> None:
    with open(output_dir / "holdout_report.json", "w") as f:
        json.dump({"ingredient": ingredient_holdout, "cuisine": cuisine_holdout}, f, indent=2)


def write_holdout_overlap_csv(
    output_dir: Path,
    top_interactions: pd.DataFrame,
    overlap_metrics: List[Dict[str, Any]],
) -> None:
    """Write holdout_overlap.csv with overlap metrics per run if multiple."""
    if not overlap_metrics:
        pd.DataFrame(columns=["holdout", "overlap_frac", "n_pairs"]).to_csv(output_dir / "holdout_overlap.csv", index=False)
        return
    df = pd.DataFrame(overlap_metrics)
    df.to_csv(output_dir / "holdout_overlap.csv", index=False)
