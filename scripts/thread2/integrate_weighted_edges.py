#!/usr/bin/env python3
"""
Integration Parts 1+2: gap-fill compound→gene layer + ingredient gene sets (noisy-OR).

STOP after ingredient-collapse re-measurement — no enrichment / category hop.

Usage (from repo root):
    python scripts/thread2/integrate_weighted_edges.py

Inputs (hard-pinned):
    data/processed/canonical/compound_gene_expanded_canonical_normalized.csv
    data/processed/thread2/inference/predicted_compound_gene_weighted_v2.parquet
    data/processed/thread2/inference/confidence_weight_spec_v2.json
    data/processed/canonical/ingredient_compound_canonical.csv

Outputs (data/processed/integrated/ only):
    compound_gene_integrated_v1.parquet
    predicted_compound_gene_excluded_low_v1.parquet
    ingredient_gene_sets_v1.parquet
    integration_report_v1.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTEGRATED_DIR = ROOT / "data/processed/integrated"

MEASURED_CG = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"
PREDICTED_WEIGHTED = ROOT / "data/processed/thread2/inference/predicted_compound_gene_weighted_v2.parquet"
WEIGHT_SPEC = ROOT / "data/processed/thread2/inference/confidence_weight_spec_v2.json"
INGREDIENT_COMPOUND = ROOT / "data/processed/canonical/ingredient_compound_canonical.csv"

INTEGRATED_CG_OUT = INTEGRATED_DIR / "compound_gene_integrated_v1.parquet"
EXCLUDED_LOW_OUT = INTEGRATED_DIR / "predicted_compound_gene_excluded_low_v1.parquet"
INGREDIENT_GENES_OUT = INTEGRATED_DIR / "ingredient_gene_sets_v1.parquet"
REPORT_OUT = INTEGRATED_DIR / "integration_report_v1.json"

TIERS_INTEGRATED = {"predicted_high", "predicted_moderate"}
SIMILARITY_REDUNDANCY_THRESHOLD = 0.95  # compounds sharing NN at this tanimoto → non-independent


def noisy_or(weights: list[float]) -> float:
    if not weights:
        return 0.0
    w = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - w))


def collapse_metrics(ing_gene_sets: dict[str, frozenset[str]]) -> dict[str, Any]:
    sig_to_ings: dict[frozenset[str], list[str]] = defaultdict(list)
    for ing, gs in ing_gene_sets.items():
        sig_to_ings[gs].append(ing)

    groups = sorted(sig_to_ings.values(), key=len, reverse=True)
    n_ings = len(ing_gene_sets)
    n_unique = len(sig_to_ings)
    collapsed_ings = sum(len(g) for g in groups if len(g) > 1)
    n_collapse_groups = sum(1 for g in groups if len(g) > 1)
    sizes = [len(g) for g in groups if len(g) > 1]

    return {
        "n_ingredients": n_ings,
        "n_unique_gene_sets": n_unique,
        "n_ingredients_in_collapse_groups": collapsed_ings,
        "n_collapse_groups": n_collapse_groups,
        "largest_collapse_group_size": max(sizes) if sizes else 1,
        "collapse_group_size_distribution": dict(Counter(sizes)),
        "n_singleton_gene_sets": sum(1 for g in groups if len(g) == 1),
        "genes_per_ingredient_mean": float(np.mean([len(g) for g in ing_gene_sets.values()])),
        "genes_per_ingredient_median": float(np.median([len(g) for g in ing_gene_sets.values()])),
        "n_empty_gene_sets": sum(1 for g in ing_gene_sets.values() if len(g) == 0),
        "signature_to_ingredients": {str(i): ings for i, ings in enumerate(groups[:20])},
    }


def build_measured_layer(measured: pd.DataFrame) -> pd.DataFrame:
    df = measured[["compound_id", "gene_symbol"]].drop_duplicates()
    df = df.assign(
        source="measured",
        evidence="measured",
        confidence_weight=1.0,
        confidence_tier=pd.NA,
        nearest_neighbor_inchikey=pd.NA,
        nearest_neighbor_tanimoto=np.nan,
        prediction_score=np.nan,
    )
    return df


def build_predicted_layers(predicted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = predicted.rename(
        columns={
            "dark_compound_inchikey": "compound_id",
            "predicted_gene_symbol": "gene_symbol",
        }
    )
    integrated_mask = pred["confidence_tier"].isin(TIERS_INTEGRATED)
    integrated = pred.loc[integrated_mask].copy()
    excluded = pred.loc[~integrated_mask].copy()
    return integrated, excluded


def integrate_compound_gene(
    measured_df: pd.DataFrame,
    predicted_integrated: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    measured = build_measured_layer(measured_df)

    pred_rows = predicted_integrated.copy()
    pred_rows["source"] = "predicted"
    pred_rows["evidence"] = "predicted"

    meas_keys = measured[["compound_id", "gene_symbol"]].drop_duplicates()
    meas_keys["_is_measured"] = True
    merged = pred_rows.merge(meas_keys, on=["compound_id", "gene_symbol"], how="left")
    n_dup = int(merged["_is_measured"].fillna(False).sum())
    pred_added = merged.loc[merged["_is_measured"].isna()].drop(columns=["_is_measured"])

    cols = [
        "compound_id",
        "gene_symbol",
        "source",
        "evidence",
        "confidence_weight",
        "confidence_tier",
        "nearest_neighbor_inchikey",
        "nearest_neighbor_tanimoto",
        "prediction_score",
    ]
    integrated = pd.concat([measured[cols], pred_added[cols]], ignore_index=True)
    integrated = integrated.drop_duplicates(subset=["compound_id", "gene_symbol"], keep="first")

    stats = {
        "total_edges": int(len(integrated)),
        "measured_edges": int(len(measured)),
        "predicted_candidates_high_moderate": int(len(predicted_integrated)),
        "predicted_added": int(len(pred_added)),
        "predicted_dropped_as_duplicate": n_dup,
        "unique_genes_integrated": int(integrated["gene_symbol"].nunique()),
        "unique_genes_measured_only": int(measured["gene_symbol"].nunique()),
        "unique_compounds_integrated": int(integrated["compound_id"].nunique()),
    }
    return integrated, stats


def compound_gene_lookup(integrated: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """compound_id -> list of edge records."""
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in integrated.itertuples(index=False):
        lookup[row.compound_id].append(
            {
                "gene_symbol": row.gene_symbol,
                "source": row.source,
                "confidence_weight": float(row.confidence_weight),
                "confidence_tier": row.confidence_tier,
                "nearest_neighbor_inchikey": row.nearest_neighbor_inchikey,
                "nearest_neighbor_tanimoto": row.nearest_neighbor_tanimoto,
            }
        )
    return lookup


def assess_redundancy(supporting: list[dict[str, Any]]) -> tuple[bool, str]:
    """
    Flag predicted-only support driven by compounds sharing the same structural NN
    (non-independent evidence for noisy-OR).
    """
    pred = [s for s in supporting if s["source"] == "predicted"]
    if len(pred) < 2:
        return False, ""

    nn_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in pred:
        nn = s.get("nearest_neighbor_inchikey")
        if pd.notna(nn) and str(nn).strip():
            nn_groups[str(nn)].append(s)

    if not nn_groups:
        return False, ""

    largest_nn, largest_cluster = max(nn_groups.items(), key=lambda kv: len(kv[1]))
    if len(largest_cluster) < 2:
        return False, ""

    cluster_weights = [s["confidence_weight"] for s in largest_cluster]
    all_weights = [s["confidence_weight"] for s in pred]
    cluster_noisy = noisy_or(cluster_weights)
    full_noisy = noisy_or(all_weights)
    cluster_share = cluster_noisy / full_noisy if full_noisy > 0 else 0.0

    if cluster_share >= 0.5:
        return True, (
            f"{len(largest_cluster)} compounds share NN {largest_nn}; "
            f"cluster contributes ~{cluster_share:.0%} of noisy-OR mass"
        )
    return False, ""


def build_ingredient_gene_sets(
    ingredient_compound: pd.DataFrame,
    integrated: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cg_lookup = compound_gene_lookup(integrated)

    ing_to_compounds: dict[str, set[str]] = defaultdict(set)
    for row in ingredient_compound.itertuples(index=False):
        ing_to_compounds[row.ingredient_id].add(row.compound_id)

    rows: list[dict[str, Any]] = []
    redundancy_flags = 0
    ing_stats: list[dict[str, Any]] = []

    for ing, compounds in sorted(ing_to_compounds.items()):
        gene_support: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cmp in compounds:
            for edge in cg_lookup.get(cmp, []):
                gene_support[edge["gene_symbol"]].append({**edge, "compound_id": cmp})

        n_measured_genes = 0
        n_predicted_only_genes = 0
        n_predicted_genes_added = 0

        for gene, supporting in gene_support.items():
            has_measured = any(s["source"] == "measured" for s in supporting)
            n_meas = sum(1 for s in supporting if s["source"] == "measured")
            n_pred = sum(1 for s in supporting if s["source"] == "predicted")
            n_compounds = len({s["compound_id"] for s in supporting})

            if has_measured:
                confidence = 1.0
                evidence = "measured"
                n_measured_genes += 1
            else:
                weights = [s["confidence_weight"] for s in supporting if s["source"] == "predicted"]
                confidence = noisy_or(weights)
                evidence = "predicted"
                n_predicted_only_genes += 1
                n_predicted_genes_added += 1

            red_flag, red_note = assess_redundancy(supporting) if not has_measured else (False, "")
            if red_flag:
                redundancy_flags += 1

            rows.append(
                {
                    "ingredient_id": ing,
                    "gene_symbol": gene,
                    "confidence": confidence,
                    "evidence": evidence,
                    "n_supporting_compounds": n_compounds,
                    "n_measured_compounds": n_meas,
                    "n_predicted_compounds": n_pred,
                    "redundancy_flag": red_flag,
                    "redundancy_note": red_note if red_note else pd.NA,
                }
            )

        ing_stats.append(
            {
                "ingredient_id": ing,
                "n_genes": len(gene_support),
                "n_measured_evidence_genes": n_measured_genes,
                "n_predicted_only_genes": n_predicted_only_genes,
            }
        )

    df = pd.DataFrame(rows)
    meta = {
        "n_ingredient_gene_rows": int(len(df)),
        "n_ingredients": int(len(ing_to_compounds)),
        "genes_with_redundancy_flag": redundancy_flags,
        "pct_predicted_only_genes_redundant": (
            round(100 * redundancy_flags / max(1, int((df["evidence"] == "predicted").sum())), 2)
        ),
        "redundancy_heuristic": (
            "predicted-only gene flagged when >=2 supporting compounds share the same "
            "nearest_neighbor_inchikey AND that cluster contributes >=50% of noisy-OR mass"
        ),
        "independence_caveat": (
            "noisy-OR treats compound-level predictions as independent; compounds sharing "
            "the same structural nearest neighbor may not be independent evidence"
        ),
    }
    return df, meta


def ingredient_gene_set_dict(
    ingredient_genes: pd.DataFrame,
) -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for ing, grp in ingredient_genes.groupby("ingredient_id"):
        out[str(ing)] = frozenset(grp["gene_symbol"].astype(str))
    return out


def measured_only_ingredient_genes(
    measured: pd.DataFrame,
    ingredient_compound: pd.DataFrame,
) -> dict[str, frozenset[str]]:
    cg = measured.groupby("compound_id")["gene_symbol"].apply(set).to_dict()
    ing_genes: dict[str, set[str]] = defaultdict(set)
    for row in ingredient_compound.itertuples(index=False):
        ing_genes[row.ingredient_id] |= cg.get(row.compound_id, set())
    return {k: frozenset(v) for k, v in ing_genes.items()}


def composition_stats(ingredient_genes: pd.DataFrame) -> dict[str, Any]:
    per_ing = []
    for ing, grp in ingredient_genes.groupby("ingredient_id"):
        n = len(grp)
        n_meas = int((grp["evidence"] == "measured").sum())
        n_pred = int((grp["evidence"] == "predicted").sum())
        per_ing.append(
            {
                "n_genes": n,
                "n_measured": n_meas,
                "n_predicted": n_pred,
                "frac_predicted": n_pred / n if n else 0.0,
            }
        )
    df = pd.DataFrame(per_ing)
    gained = df[df["n_predicted"] > 0]
    return {
        "n_ingredients_with_any_predicted_genes": int(len(gained)),
        "mean_frac_predicted_among_genes": float(df["frac_predicted"].mean()),
        "median_frac_predicted_among_genes": float(df["frac_predicted"].median()),
        "mean_n_predicted_genes_per_ingredient": float(df["n_predicted"].mean()),
        "median_n_predicted_genes_per_ingredient": float(df["n_predicted"].median()),
    }


def sample_deconverged_pairs(
    before: dict[str, frozenset[str]],
    after: dict[str, frozenset[str]],
    ingredient_genes: pd.DataFrame,
    n_pairs: int = 5,
) -> list[dict[str, Any]]:
    """Find ingredient pairs identical before but distinct after."""
    sig_before: dict[frozenset[str], list[str]] = defaultdict(list)
    for ing, gs in before.items():
        sig_before[gs].append(ing)

    candidates: list[tuple[int, str, str]] = []
    for ings in sig_before.values():
        if len(ings) < 2:
            continue
        for i in range(len(ings)):
            for j in range(i + 1, len(ings)):
                a, b = ings[i], ings[j]
                if after.get(a) != after.get(b):
                    diff_size = len(after.get(a, frozenset()) ^ after.get(b, frozenset()))
                    candidates.append((diff_size, a, b))

    candidates.sort(reverse=True)

    samples: list[dict[str, Any]] = []
    gene_lookup = ingredient_genes.set_index(["ingredient_id", "gene_symbol"])

    for _, a, b in candidates[:n_pairs]:
        ga = after.get(a, frozenset())
        gb = after.get(b, frozenset())
        only_a = sorted(ga - gb)
        only_b = sorted(gb - ga)
        shared = sorted(ga & gb)

        def _gene_detail(ing: str, genes: list[str]) -> list[dict]:
            detail = []
            for g in genes[:15]:
                try:
                    row = gene_lookup.loc[(ing, g)]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    detail.append(
                        {
                            "gene": g,
                            "evidence": str(row["evidence"]),
                            "confidence": float(row["confidence"]),
                        }
                    )
                except KeyError:
                    detail.append({"gene": g})
            return detail

        samples.append(
            {
                "ingredient_a": a,
                "ingredient_b": b,
                "shared_genes_count": len(shared),
                "genes_only_in_a_count": len(only_a),
                "genes_only_in_b_count": len(only_b),
                "differentiating_genes_a_sample": _gene_detail(a, only_a),
                "differentiating_genes_b_sample": _gene_detail(b, only_b),
            }
        )
    return samples


def main() -> int:
    print("=== Integration Parts 1+2 (gap-fill + noisy-OR) ===", flush=True)
    INTEGRATED_DIR.mkdir(parents=True, exist_ok=True)

    for path in (MEASURED_CG, PREDICTED_WEIGHTED, WEIGHT_SPEC, INGREDIENT_COMPOUND):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1

    measured_raw = pd.read_csv(MEASURED_CG)
    predicted = pd.read_parquet(PREDICTED_WEIGHTED)
    ingredient_compound = pd.read_csv(INGREDIENT_COMPOUND)
    weight_spec = json.loads(WEIGHT_SPEC.read_text(encoding="utf-8"))

    print(f"Loaded measured edges: {len(measured_raw):,}", flush=True)
    print(f"Loaded predicted weighted: {len(predicted):,}", flush=True)

    # --- Part 1 ---
    pred_integrated, pred_excluded = build_predicted_layers(predicted)
    integrated_cg, part1_stats = integrate_compound_gene(measured_raw, pred_integrated)

    integrated_cg.to_parquet(INTEGRATED_CG_OUT, index=False)
    pred_excluded.to_parquet(EXCLUDED_LOW_OUT, index=False)

    print("\n--- Part 1: compound->gene integrated layer ---", flush=True)
    for k, v in part1_stats.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}", flush=True)
    print(f"  tiers_integrated: {sorted(TIERS_INTEGRATED)}", flush=True)
    print(f"  tiers_excluded_file: predicted_low, withheld (not in gene sets)", flush=True)

    # --- Part 2 ---
    ingredient_genes, part2_meta = build_ingredient_gene_sets(ingredient_compound, integrated_cg)
    ingredient_genes.to_parquet(INGREDIENT_GENES_OUT, index=False)

    print("\n--- Part 2: ingredient gene sets (noisy-OR) ---", flush=True)
    for k, v in part2_meta.items():
        print(f"  {k}: {v}", flush=True)

    # --- Checkpoint: collapse ---
    before_sets = measured_only_ingredient_genes(measured_raw, ingredient_compound)
    after_sets = ingredient_gene_set_dict(ingredient_genes)

    before_metrics = collapse_metrics(before_sets)
    after_metrics = collapse_metrics(after_sets)

    collapse_delta = {
        "ingredients_in_collapse_groups_before": before_metrics["n_ingredients_in_collapse_groups"],
        "ingredients_in_collapse_groups_after": after_metrics["n_ingredients_in_collapse_groups"],
        "delta_ingredients_in_collapse": (
            after_metrics["n_ingredients_in_collapse_groups"]
            - before_metrics["n_ingredients_in_collapse_groups"]
        ),
        "unique_gene_sets_before": before_metrics["n_unique_gene_sets"],
        "unique_gene_sets_after": after_metrics["n_unique_gene_sets"],
        "delta_unique_gene_sets": (
            after_metrics["n_unique_gene_sets"] - before_metrics["n_unique_gene_sets"]
        ),
        "largest_collapse_group_before": before_metrics["largest_collapse_group_size"],
        "largest_collapse_group_after": after_metrics["largest_collapse_group_size"],
        "collapse_verdict": (
            "de-converged"
            if after_metrics["n_ingredients_in_collapse_groups"]
            < before_metrics["n_ingredients_in_collapse_groups"]
            else (
                "unchanged_or_worse"
                if after_metrics["n_ingredients_in_collapse_groups"]
                >= before_metrics["n_ingredients_in_collapse_groups"]
                else "partial"
            )
        ),
    }

    comp = composition_stats(ingredient_genes)
    samples = sample_deconverged_pairs(before_sets, after_sets, ingredient_genes, n_pairs=5)

    print("\n--- CHECKPOINT: ingredient-level collapse ---", flush=True)
    print(
        f"  Before (measured only): {before_metrics['n_ingredients_in_collapse_groups']}/223 "
        f"ingredients in collapse groups, {before_metrics['n_unique_gene_sets']} unique gene sets, "
        f"largest group {before_metrics['largest_collapse_group_size']}",
        flush=True,
    )
    print(
        f"  After (integrated):     {after_metrics['n_ingredients_in_collapse_groups']}/223 "
        f"ingredients in collapse groups, {after_metrics['n_unique_gene_sets']} unique gene sets, "
        f"largest group {after_metrics['largest_collapse_group_size']}",
        flush=True,
    )
    print(
        f"  Delta: {collapse_delta['delta_ingredients_in_collapse']:+d} ingredients in collapse, "
        f"{collapse_delta['delta_unique_gene_sets']:+d} unique gene sets",
        flush=True,
    )
    print(
        f"  Genes/ingredient — before mean={before_metrics['genes_per_ingredient_mean']:.1f} "
        f"median={before_metrics['genes_per_ingredient_median']:.0f}; "
        f"after mean={after_metrics['genes_per_ingredient_mean']:.1f} "
        f"median={after_metrics['genes_per_ingredient_median']:.0f}",
        flush=True,
    )

    report: dict[str, Any] = {
        "inputs": {
            "measured_compound_gene": str(MEASURED_CG),
            "predicted_weighted": str(PREDICTED_WEIGHTED),
            "weight_spec": str(WEIGHT_SPEC),
            "ingredient_compound": str(INGREDIENT_COMPOUND),
        },
        "weight_spec_version": weight_spec.get("version", "v2"),
        "integration_policy": {
            "gap_fill_only": True,
            "measured_weight": 1.0,
            "measured_overrides_predicted_duplicate_pairs": True,
            "predicted_tiers_included": sorted(TIERS_INTEGRATED),
            "predicted_tiers_excluded_from_gene_sets": ["predicted_low", "withheld"],
            "gene_confidence_combination": "noisy_or for predicted-only; measured → 1.0",
        },
        "part1_compound_gene_layer": part1_stats,
        "part2_ingredient_gene_sets": part2_meta,
        "composition": comp,
        "checkpoint_collapse": {
            "before_measured_only": before_metrics,
            "after_integrated": after_metrics,
            "delta": collapse_delta,
        },
        "deconvergence_samples": samples,
        "outputs": {
            "compound_gene_integrated": str(INTEGRATED_CG_OUT),
            "predicted_excluded_low": str(EXCLUDED_LOW_OUT),
            "ingredient_gene_sets": str(INGREDIENT_GENES_OUT),
        },
        "stop_note": "Parts 1+2 complete. Enrichment and category hop NOT built — awaiting review.",
    }

    # Trim large signature map for JSON readability
    report["checkpoint_collapse"]["before_measured_only"].pop("signature_to_ingredients", None)
    report["checkpoint_collapse"]["after_integrated"].pop("signature_to_ingredients", None)

    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {INTEGRATED_CG_OUT}", flush=True)
    print(f"Wrote {EXCLUDED_LOW_OUT}", flush=True)
    print(f"Wrote {INGREDIENT_GENES_OUT}", flush=True)
    print(f"Wrote {REPORT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
