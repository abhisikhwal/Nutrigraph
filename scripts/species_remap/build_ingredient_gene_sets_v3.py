#!/usr/bin/env python3
"""
Rebuild ingredient gene sets on expanded v2 species layer (463 nodes).

Uses unchanged compound_gene_integrated_v1 + ingredient_compound_canonical_v2
with independence-guarded noisy-OR (same logic as v2 on 223 set).

STOP after saturation + collapse checkpoint — no enrichment.

Usage (from repo root):
    python scripts/species_remap/build_ingredient_gene_sets_v3.py

Outputs (data/processed/integrated/ only; v1/v2 untouched):
    ingredient_gene_sets_v3.parquet
    ingredient_gene_sets_v3_saturation_report.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "scripts" / "thread2"))

from apply_independence_guard import (  # noqa: E402
    build_guarded_gene_sets,
    build_nn_scaffold_map,
    distribution_summary,
)
from integrate_weighted_edges import collapse_metrics, composition_stats  # noqa: E402

INTEGRATED_DIR = ROOT / "data/processed/integrated"
CANONICAL = ROOT / "data/processed/canonical"

ICC_V2 = CANONICAL / "ingredient_compound_canonical_v2.parquet"
SPECIES_NODES = CANONICAL / "species_nodes_v2.parquet"
STRING_MAP = CANONICAL / "ingredient_string_species_v2.parquet"
INTEGRATED_CG = INTEGRATED_DIR / "compound_gene_integrated_v1.parquet"
MEASURED_CG = CANONICAL / "compound_gene_expanded_canonical_normalized.csv"

INGREDIENT_GENES_V2 = INTEGRATED_DIR / "ingredient_gene_sets_v2.parquet"
INGREDIENT_GENES_V3 = INTEGRATED_DIR / "ingredient_gene_sets_v3.parquet"
REPORT_OUT = INTEGRATED_DIR / "ingredient_gene_sets_v3_saturation_report.json"

# 18 identity-only species (zero compound edges in v2 ICC)
THIN_SPECIES_FOODB = {965, 966, 967, 968, 969, 970, 973, 974, 975, 977, 978, 979, 981, 982, 983, 984, 988, 993}

SAMPLE_SPECIES = [
    ("SP_000032", "paprika", "Pepper (Capsicum) — second-pass recovery"),
    ("SP_000259", "bacon", "Domestic pig — second-pass recovery"),
    ("SP_000235", "salmon", "Salmonidae — family-level coarse"),
    ("SP_000383", "cornstarch", "Corn — starch parent species"),
    ("SP_000052", "turmeric", "Turmeric — spice with rich phytochemistry"),
]

SATURATION_BALLOON_THRESHOLD = 1500


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def histogram_buckets(counts: np.ndarray) -> dict[str, int]:
    if len(counts) == 0:
        return {}
    bins = [0, 1, 10, 50, 100, 500, 1000, 1500, 2000, 5000, 100000]
    labels = [
        "0",
        "1-9",
        "10-49",
        "50-99",
        "100-499",
        "500-999",
        "1000-1499",
        "1500-1999",
        "2000-4999",
        "5000+",
    ]
    hist: dict[str, int] = {}
    for i in range(len(labels)):
        lo = bins[i]
        hi = bins[i + 1] if i + 1 < len(bins) else None
        if hi is None:
            n = int((counts >= lo).sum())
        else:
            n = int(((counts >= lo) & (counts < hi)).sum())
        hist[labels[i]] = n
    return hist


def per_ingredient_gene_counts(
    gene_df: pd.DataFrame,
    all_ingredient_ids: list[str],
) -> pd.Series:
    """Include zero-gene species not present in gene_df rows."""
    counts = gene_df.groupby("ingredient_id").size()
    for ing in all_ingredient_ids:
        if ing not in counts.index:
            counts[ing] = 0
    return counts


def sample_gene_set(
    ingredient_id: str,
    gene_df: pd.DataFrame,
    string_map: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    sub = gene_df[gene_df["ingredient_id"] == ingredient_id].copy()
    strings = string_map.loc[string_map["species_node"] == ingredient_id, "ingredient_string"].tolist()
    n_meas = int((sub["evidence"] == "measured").sum()) if len(sub) else 0
    n_pred = int((sub["evidence"] == "predicted").sum()) if len(sub) else 0
    top = sub.sort_values("confidence", ascending=False).head(15)
    return {
        "species_node": ingredient_id,
        "label": label,
        "example_strings": strings[:5],
        "n_genes": int(len(sub)),
        "n_measured_evidence": n_meas,
        "n_predicted_evidence": n_pred,
        "frac_predicted": round(n_pred / len(sub), 3) if len(sub) else 0.0,
        "top_genes_sample": top[
            ["gene_symbol", "confidence", "evidence", "n_supporting_compounds", "redundancy_flag"]
        ].to_dict(orient="records"),
    }


def saturation_verdict(v3_median: float, v2_median: float) -> dict[str, Any]:
    delta = v3_median - v2_median
    pct_change = round(100 * delta / v2_median, 1) if v2_median else 0.0
    if v3_median > SATURATION_BALLOON_THRESHOLD:
        case = "BALLOONED_STOP"
        recommendation = (
            f"Median {v3_median:.0f} exceeds {SATURATION_BALLOON_THRESHOLD} threshold — "
            "apply membership/confidence floor before enrichment."
        )
    elif v3_median <= v2_median * 1.05:
        case = "SIMILAR_OR_LOWER_PROCEED"
        recommendation = (
            f"Median {v3_median:.0f} vs v2 {v2_median:.0f} — similar or lower; safe to proceed to weighted enrichment."
        )
    elif v3_median <= SATURATION_BALLOON_THRESHOLD:
        case = "MODERATE_INCREASE_PROCEED_WITH_CAUTION"
        recommendation = (
            f"Median rose to {v3_median:.0f} (+{pct_change}%) but below {SATURATION_BALLOON_THRESHOLD}; "
            "enrichment viable with monitoring."
        )
    else:
        case = "BALLOONED_STOP"
        recommendation = "Median exceeded threshold."
    return {
        "case": case,
        "v2_median_genes_per_ingredient": v2_median,
        "v3_median_genes_per_ingredient": v3_median,
        "delta_median": delta,
        "pct_change_vs_v2": pct_change,
        "balloon_threshold": SATURATION_BALLOON_THRESHOLD,
        "recommendation": recommendation,
    }


def main() -> int:
    print("=== Ingredient gene sets v3 (463-species layer) ===", flush=True)
    INTEGRATED_DIR.mkdir(parents=True, exist_ok=True)

    for path in (ICC_V2, INTEGRATED_CG, INGREDIENT_GENES_V2, SPECIES_NODES):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1

    pre_integrated_hash = sha256_file(INTEGRATED_CG)
    pre_measured_hash = sha256_file(MEASURED_CG)

    print("Loading inputs...", flush=True)
    ingredient_compound = pd.read_parquet(ICC_V2, columns=["ingredient_id", "compound_id"])
    integrated = pd.read_parquet(INTEGRATED_CG)
    species_nodes = pd.read_parquet(SPECIES_NODES)
    string_map = pd.read_parquet(STRING_MAP)
    v2_genes = pd.read_parquet(INGREDIENT_GENES_V2)

    all_species_ids = species_nodes["species_node_id"].astype(str).tolist()
    n_species_total = len(all_species_ids)
    n_with_icc = int(ingredient_compound["ingredient_id"].nunique())

    print(f"  ICC v2: {len(ingredient_compound):,} edges, {n_with_icc} species with compounds", flush=True)
    print(f"  Species nodes: {n_species_total}", flush=True)

    print("Building Murcko scaffold map (cached neighbors)...", flush=True)
    nn_to_scaffold, scaffold_meta = build_nn_scaffold_map(integrated)

    print("Building independence-guarded gene sets...", flush=True)
    v3_full = build_guarded_gene_sets(ingredient_compound, integrated, nn_to_scaffold)
    v3_out = v3_full.drop(columns=["confidence_v1_naive_noisy_or"], errors="ignore")
    v3_out.to_parquet(INGREDIENT_GENES_V3, index=False)
    print(f"  Wrote {len(v3_out):,} gene rows for {v3_out['ingredient_id'].nunique()} species with genes", flush=True)

    # --- Saturation metrics ---
    v2_counts = v2_genes.groupby("ingredient_id").size()
    v3_counts_all = per_ingredient_gene_counts(v3_out, all_species_ids)
    v3_counts_with_compounds = v3_out.groupby("ingredient_id").size()

    thin_ids = set(
        species_nodes.loc[species_nodes["foodb_id"].astype(int).isin(THIN_SPECIES_FOODB), "species_node_id"].astype(str)
    )

    v3_non_thin = v3_counts_all.drop(index=list(thin_ids), errors="ignore")
    v3_non_thin_with_genes = v3_counts_with_compounds

    v2_gpi = v2_counts.to_numpy(dtype=float)
    v3_all = v3_counts_all.to_numpy(dtype=float)
    v3_nc = v3_non_thin_with_genes.to_numpy(dtype=float)

    comp_v3 = composition_stats(v3_out)
    comp_v2 = composition_stats(v2_genes)
    v2_pred_frac = float((v2_genes["evidence"] == "predicted").mean() * 100)
    v3_pred_frac = float((v3_out["evidence"] == "predicted").mean() * 100)

    # --- Collapse ---
    v3_sets_all = {str(ing): frozenset() for ing in all_species_ids}
    for ing, grp in v3_out.groupby("ingredient_id"):
        v3_sets_all[str(ing)] = frozenset(grp["gene_symbol"].astype(str))

    v3_sets_collapse = {k: v for k, v in v3_sets_all.items() if k not in thin_ids}
    v2_sets = {
        str(ing): frozenset(g["gene_symbol"].astype(str))
        for ing, g in v2_genes.groupby("ingredient_id")
    }

    collapse_v2 = collapse_metrics(v2_sets)
    collapse_v3_all = collapse_metrics(v3_sets_all)
    collapse_v3_excl_thin = collapse_metrics(v3_sets_collapse)

    # v2 baseline from independence guard report
    v2_guarded_collapse = {
        "n_ingredients_in_collapse_groups": 117,
        "n_unique_gene_sets": 135,
        "largest_collapse_group_size": 41,
        "n_ingredients": 223,
    }

    samples = [
        sample_gene_set(sp, v3_out, string_map, label)
        for sp, _name, label in SAMPLE_SPECIES
        if sp in all_species_ids
    ]

    verdict = saturation_verdict(float(np.median(v3_non_thin_with_genes)), float(v2_counts.median()))

    report: dict[str, Any] = {
        "phase": "SATURATION_GATE_STOP_BEFORE_ENRICHMENT",
        "inputs": {
            "ingredient_compound_v2": str(ICC_V2.relative_to(ROOT)),
            "compound_gene_integrated_v1": str(INTEGRATED_CG.relative_to(ROOT)),
            "species_nodes_v2": str(SPECIES_NODES.relative_to(ROOT)),
            "guard_logic": "independence-guarded noisy-OR (nn max per group, then noisy-OR across groups)",
        },
        "mechanism_layers_unchanged": {
            "compound_gene_integrated_v1_sha256": pre_integrated_hash,
            "post_build_integrated_sha256": sha256_file(INTEGRATED_CG),
            "integrated_unchanged": pre_integrated_hash == sha256_file(INTEGRATED_CG),
            "measured_cg_sha256": pre_measured_hash,
            "post_build_measured_sha256": sha256_file(MEASURED_CG),
            "measured_unchanged": pre_measured_hash == sha256_file(MEASURED_CG),
            "ingredient_gene_sets_v2_unchanged": True,
        },
        "saturation_gate": {
            "genes_per_ingredient_v2_223set": {
                **distribution_summary(v2_gpi),
                "note": "223-ingredient v2 guarded set (baseline)",
            },
            "genes_per_ingredient_v3_all_463": {
                **distribution_summary(v3_all),
                "includes_18_empty_thin_species": True,
            },
            "genes_per_ingredient_v3_with_compounds_445": distribution_summary(v3_nc),
            "genes_per_ingredient_v3_excluding_18_thin": distribution_summary(v3_non_thin.to_numpy(dtype=float)),
            "histogram_v3_with_compounds": histogram_buckets(v3_nc),
            "histogram_v2_223": histogram_buckets(v2_gpi),
            "spread_assessment": {
                "v3_iqr": float(np.percentile(v3_nc, 75) - np.percentile(v3_nc, 25)) if len(v3_nc) else 0,
                "v2_iqr": float(np.percentile(v2_gpi, 75) - np.percentile(v2_gpi, 25)),
                "v3_cv": float(np.std(v3_nc) / np.mean(v3_nc)) if len(v3_nc) and np.mean(v3_nc) else 0,
                "uniform_high": bool(np.percentile(v3_nc, 75) - np.percentile(v3_nc, 25) < 200 and np.median(v3_nc) > 1000)
                if len(v3_nc)
                else False,
                "note": "Low IQR + high median suggests uniform saturation; spread IQR indicates varied breadth",
            },
            "predicted_vs_measured_composition": {
                "v2_pct_gene_rows_predicted": round(v2_pred_frac, 2),
                "v3_pct_gene_rows_predicted": round(v3_pred_frac, 2),
                "v2_composition_stats": comp_v2,
                "v3_composition_stats": comp_v3,
            },
            "verdict": verdict,
        },
        "collapse_remeasurement": {
            "v2_223_guarded_baseline": v2_guarded_collapse,
            "v2_recomputed_from_parquet": {
                k: collapse_v2[k]
                for k in (
                    "n_ingredients_in_collapse_groups",
                    "n_unique_gene_sets",
                    "largest_collapse_group_size",
                    "n_empty_gene_sets",
                )
            },
            "v3_all_463_including_empty_thin": {
                k: collapse_v3_all[k]
                for k in (
                    "n_ingredients_in_collapse_groups",
                    "n_unique_gene_sets",
                    "largest_collapse_group_size",
                    "n_empty_gene_sets",
                )
            },
            "v3_445_excluding_18_thin_identity_nodes": {
                k: collapse_v3_excl_thin[k]
                for k in (
                    "n_ingredients_in_collapse_groups",
                    "n_unique_gene_sets",
                    "largest_collapse_group_size",
                    "n_empty_gene_sets",
                )
            },
            "thin_species_handling": {
                "n_thin_identity_only": len(thin_ids),
                "excluded_from_primary_collapse_metric": True,
                "empty_gene_set_count_all_463": int(collapse_v3_all["n_empty_gene_sets"]),
            },
            "delta_v3_excl_thin_vs_v2": {
                "ingredients_in_collapse_groups": (
                    collapse_v3_excl_thin["n_ingredients_in_collapse_groups"]
                    - v2_guarded_collapse["n_ingredients_in_collapse_groups"]
                ),
                "unique_gene_sets": (
                    collapse_v3_excl_thin["n_unique_gene_sets"] - v2_guarded_collapse["n_unique_gene_sets"]
                ),
                "largest_collapse_group": (
                    collapse_v3_excl_thin["largest_collapse_group_size"]
                    - v2_guarded_collapse["largest_collapse_group_size"]
                ),
                "interpretation": (
                    "Expanded universe adds distinct species; collapse dilutes (more unique sets) "
                    "unless compound mass drives convergence. Compare unique_gene_sets / n_ingredients ratio."
                ),
            },
        },
        "new_species_samples": samples,
        "outputs": {
            "ingredient_gene_sets_v3": str(INGREDIENT_GENES_V3.relative_to(ROOT)),
            "ingredient_gene_sets_v2_preserved": str(INGREDIENT_GENES_V2.relative_to(ROOT)),
        },
        "stop_note": "Gene sets v3 built. Enrichment NOT run — review saturation verdict before proceeding.",
    }

    REPORT_OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n--- Saturation gate ---", flush=True)
    print(f"  v2 median genes/ingredient: {verdict['v2_median_genes_per_ingredient']:.0f}", flush=True)
    print(f"  v3 median (445 w/ compounds): {verdict['v3_median_genes_per_ingredient']:.0f}", flush=True)
    print(f"  v3 median (all 463): {float(np.median(v3_all)):.0f}", flush=True)
    print(f"  Verdict: {verdict['case']}", flush=True)
    print(f"\n--- Collapse (445 excl thin) ---", flush=True)
    c = report["collapse_remeasurement"]["v3_445_excluding_18_thin_identity_nodes"]
    print(
        f"  {c['n_ingredients_in_collapse_groups']} in collapse groups, "
        f"{c['n_unique_gene_sets']} unique sets, largest={c['largest_collapse_group_size']}",
        flush=True,
    )
    print(f"\nWrote {INGREDIENT_GENES_V3}", flush=True)
    print(f"Wrote {REPORT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
