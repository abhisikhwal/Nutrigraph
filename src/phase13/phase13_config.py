"""
Phase 13 v3 — Configuration dataclass for quality-first ingredient interaction discovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class Phase13Config:
    """Configuration for Phase 13 stratified DiD pipeline."""

    signatures_path: str
    recipe_ingredients_path: str
    ingredient_names_path: Optional[str] = None
    output_dir: str = "data/processed/phase13_interactions_v3"

    # Core thresholds (auto-adjust supported)
    min_joint: int = 50
    min_A_only: int = 200
    min_B_only: int = 200
    min_none: int = 200

    # Ingredient frequency for candidate set
    min_ingredient_freq: int = 500
    top_k_ingredients: int = 500

    # Stratification
    bins_num_ingredients: int = 8
    bins_num_compounds: int = 8
    weighting: str = "size"  # "size" or "inv_var"
    min_group_per_stratum: int = 10

    # Compute
    shard_size_pairs: int = 5000
    max_pairs: Optional[int] = None
    random_seed: int = 42

    # Multiple testing
    fdr_alpha: float = 0.05

    # Null tests
    n_permutation: int = 500
    null_n_pairs: int = 500

    # Bootstrap
    bootstrap_top_k: int = 500
    n_bootstrap: int = 200
    bootstrap_recipe_sample_frac: float = 0.8

    # Regression audit
    audit_top_k: int = 300

    # Holdouts
    holdout_ingredient_frac: float = 0.1
    holdout_cuisine_frac: float = 0.2

    # Sensitivity analysis
    sensitivity_bins: List[Tuple[int, int]] = field(
        default_factory=lambda: [(6, 6), (8, 8), (10, 10)]
    )
    sensitivity_n_pairs: int = 10_000

    # Auto-relax when eligible == 0
    min_pairs_after_filter: int = 2000
    relaxed_min_joint: int = 20
    relaxed_min_A_only: int = 100
    relaxed_min_B_only: int = 100
    relaxed_min_none: int = 100

    # Optional paths
    confounders_path: Optional[str] = None
    priority_ingredients_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], tuple):
                v = [list(x) for x in v]
            d[k] = v
        return d

    @classmethod
    def from_yaml(
        cls,
        paths_yaml: Optional[Path] = None,
        phase13_yaml: Optional[Path] = None,
        repo_root: Optional[Path] = None,
        output_dir_override: Optional[str] = None,
    ) -> "Phase13Config":
        """Load from config/paths.yaml and config/phase13.yaml (or env)."""
        root = repo_root or Path(".").resolve()
        if (root / "config").exists():
            root = root
        else:
            root = root.parent

        paths: Dict[str, Any] = {}
        if paths_yaml is None:
            p = root / "config" / "paths.yaml"
            if p.exists():
                with open(p) as f:
                    paths = yaml.safe_load(f) or {}
        else:
            with open(paths_yaml) as f:
                paths = yaml.safe_load(f) or {}

        cfg: Dict[str, Any] = {}
        if phase13_yaml is None:
            p = root / "config" / "phase13.yaml"
            if p.exists():
                with open(p) as f:
                    cfg = yaml.safe_load(f) or {}
        else:
            with open(phase13_yaml) as f:
                cfg = yaml.safe_load(f) or {}

        # Resolve signature path (v3 preferred)
        processed = paths.get("processed_data", "data/processed")
        sig_paths = cfg.get("signature_paths_v3") or [
            "data/processed/phase17_reaggregation/recipe_functional_signatures_v3.parquet",
            "data/processed/canonical/recipes_biological_effects_v3.parquet",
        ]
        signatures_path = ""
        for rel in sig_paths:
            full = root / rel if not rel.startswith("/") else Path(rel)
            if full.exists():
                signatures_path = str(full)
                break
        if not signatures_path:
            signatures_path = str(root / sig_paths[0])

        recipe_ingredients_path = cfg.get("recipe_ingredients_path") or str(
            root / "data/processed/canonical/recipe_ingredients_expanded_v2.parquet"
        )
        if not (root / recipe_ingredients_path).exists() and not Path(recipe_ingredients_path).is_absolute():
            recipe_ingredients_path = str(root / recipe_ingredients_path)

        # Ingredient names: canonical or ingredients.parquet
        ing_names = None
        for candidate in [
            "data/processed/canonical/ingredients.parquet",
            paths.get("ingredient_master", ""),
        ]:
            if candidate and (root / candidate).exists():
                ing_names = str(root / candidate) if not Path(candidate).is_absolute() else candidate
                break

        out_dir = output_dir_override or cfg.get("output_dir") or "data/processed/phase13_interactions_v3"
        if not Path(out_dir).is_absolute():
            out_dir = str(root / out_dir)

        return cls(
            signatures_path=signatures_path,
            recipe_ingredients_path=recipe_ingredients_path,
            ingredient_names_path=ing_names,
            output_dir=out_dir,
            min_joint=cfg.get("min_joint", 50),
            min_A_only=cfg.get("min_a_only", 200),
            min_B_only=cfg.get("min_b_only", 200),
            min_none=cfg.get("min_none", 200),
            min_ingredient_freq=cfg.get("min_ingredient_freq", 500),
            top_k_ingredients=cfg.get("top_k_ingredients", 500),
            bins_num_ingredients=cfg.get("bins_num_ingredients", 8),
            bins_num_compounds=cfg.get("bins_num_compounds", 8),
            weighting=cfg.get("weighting", "size"),
            shard_size_pairs=cfg.get("shard_size", 5000),
            max_pairs=cfg.get("max_pairs"),
            random_seed=cfg.get("seed", 42),
            fdr_alpha=cfg.get("fdr_alpha", 0.05),
            n_permutation=cfg.get("null_n_permutations", 500),
            null_n_pairs=cfg.get("null_n_pairs", 500),
            bootstrap_top_k=cfg.get("bootstrap_top_n", 500),
            n_bootstrap=cfg.get("bootstrap_n_samples", 200),
            bootstrap_recipe_sample_frac=cfg.get("bootstrap_recipe_sample_frac", 0.8),
            audit_top_k=cfg.get("audit_top_k", 300),
            holdout_ingredient_frac=cfg.get("holdout_ingredient_frac", 0.1),
            holdout_cuisine_frac=cfg.get("holdout_cuisine_frac", 0.2),
            min_pairs_after_filter=cfg.get("min_pairs_after_filter", 2000),
            relaxed_min_joint=cfg.get("relaxed_min_joint", 20),
            relaxed_min_A_only=cfg.get("relaxed_min_a_only", 100),
            relaxed_min_B_only=cfg.get("relaxed_min_b_only", 100),
            relaxed_min_none=cfg.get("relaxed_min_none", 100),
            confounders_path=cfg.get("confounders_path"),
            sensitivity_bins=[tuple(x) for x in cfg.get("sensitivity_bins", [[6, 6], [8, 8], [10, 10]])],
        )
