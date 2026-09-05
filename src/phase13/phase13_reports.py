"""
Phase 13 v3 — Final output pack: summary JSON, interaction atlas, plots.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .phase13_utils import _ts, get_git_commit


def generate_phase13_summary(
    output_dir: Path,
    config: Any,
    run_id: str,
    dataset_counts: Dict[str, int],
    thresholds_used: Dict[str, Any],
    n_pairs_candidate: int,
    n_pairs_tested: int,
    n_pairs_eligible: int,
    n_significant_q005: int,
    per_category_significant: Dict[str, int],
    null_test_stats: Optional[Dict[str, Any]] = None,
    bootstrap_stats: Optional[Dict[str, Any]] = None,
    regression_audit_stats: Optional[Dict[str, Any]] = None,
    holdout_stats: Optional[Dict[str, Any]] = None,
    sensitivity_stats: Optional[Dict[str, Any]] = None,
    file_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    summary = {
        "run_id": run_id,
        "timestamp": _ts(),
        "git_commit": get_git_commit(output_dir),
        "dataset_counts": dataset_counts,
        "thresholds_used": thresholds_used,
        "n_pairs_candidate": n_pairs_candidate,
        "n_pairs_tested": n_pairs_tested,
        "n_pairs_eligible": n_pairs_eligible,
        "n_significant_q005": n_significant_q005,
        "per_category_significant": per_category_significant,
        "null_test": null_test_stats or {},
        "bootstrap": bootstrap_stats or {},
        "regression_audit": regression_audit_stats or {},
        "holdout": holdout_stats or {},
        "sensitivity": sensitivity_stats or {},
        "file_list": file_list or [],
    }
    with open(output_dir / "phase13_summary_v3.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def generate_interaction_atlas(
    adjusted: pd.DataFrame,
    output_dir: Path,
    top_n: int = 1000,
    ingredient_names: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Write interaction_atlas_top1000.csv with columns:
    ingA_name, ingB_name, category, delta, se, q, nBoth, cuisines_where_present, notes.
    """
    top = adjusted.nsmallest(top_n, "q_global") if "q_global" in adjusted.columns else adjusted.head(top_n)
    if top.empty:
        cols = ["ingA_name", "ingB_name", "category", "delta", "se", "q", "nBoth", "cuisines_where_present", "notes"]
        pd.DataFrame(columns=cols).to_csv(output_dir / "interaction_atlas_top1000.csv", index=False)
        return pd.DataFrame()

    atlas = top[["ingA_id", "ingB_id", "category", "delta", "se", "nBoth"]].copy()
    if "q_global" in top.columns:
        atlas["q"] = top["q_global"].values
    else:
        atlas["q"] = np.nan
    atlas["ingA_name"] = atlas["ingA_id"]
    atlas["ingB_name"] = atlas["ingB_id"]
    if ingredient_names is not None and "ingredient_id" in ingredient_names.columns and "canonical_name" in ingredient_names.columns:
        name_map = ingredient_names.set_index("ingredient_id")["canonical_name"].to_dict()
        atlas["ingA_name"] = atlas["ingA_id"].map(lambda x: name_map.get(x, x))
        atlas["ingB_name"] = atlas["ingB_id"].map(lambda x: name_map.get(x, x))
    atlas["cuisines_where_present"] = ""
    atlas["notes"] = ""
    atlas = atlas[["ingA_name", "ingB_name", "category", "delta", "se", "q", "nBoth", "cuisines_where_present", "notes"]]
    atlas.to_csv(output_dir / "interaction_atlas_top1000.csv", index=False)
    return atlas


def generate_volcano_plot(
    adjusted: pd.DataFrame,
    output_dir: Path,
    category: Optional[str] = None,
) -> None:
    """Volcano plot: delta vs -log10(q) per category or overall."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    df = adjusted.copy()
    if "q_global" not in df.columns:
        return
    df["neg_log10_q"] = -np.log10(np.clip(df["q_global"], 1e-300, 1))
    if category:
        df = df[df["category"] == category]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["delta"], df["neg_log10_q"], alpha=0.5, s=10)
    ax.set_xlabel("delta")
    ax.set_ylabel("-log10(q)")
    ax.set_title(f"Volcano plot" + (f" — {category}" if category else ""))
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    name = f"volcano_{category or 'all'}.png".replace("/", "_")
    plt.savefig(output_dir / name, dpi=150)
    plt.close()


def generate_q_histogram(adjusted: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if "q_global" not in adjusted.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(adjusted["q_global"].dropna(), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("q_global")
    ax.set_ylabel("Count")
    ax.set_title("Q-value distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "q_value_hist.png", dpi=150)
    plt.close()


def deliverables_checklist(output_dir: Path) -> Dict[str, bool]:
    required = [
        "interactions_raw_v3.parquet",
        "interactions_adjusted_v3.parquet",
        "ingredient_pairs_tested_v3.parquet",
        "pair_skip_reasons_v3.parquet",
        "recipes_with_confounders.parquet",
        "null_test_v3.json",
        "bootstrap_stability_v3.parquet",
        "holdout_report.json",
        "regression_audit_summary.json",
        "sensitivity_overlap.json",
        "phase13_summary_v3.json",
        "interaction_atlas_top1000.csv",
    ]
    result = {}
    for f in required:
        result[f] = (output_dir / f).exists()
    return result
