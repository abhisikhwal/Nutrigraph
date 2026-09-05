#!/usr/bin/env python3
"""
Independence guard for ingredient gene-set noisy-OR confidence (v2).

Within each nearest_neighbor_inchikey group: max(weight).
Across groups: noisy-OR on group representatives.

Also computes scaffold-grouped sensitivity (Murcko scaffold of neighbor).

Usage (from repo root):
    python scripts/thread2/apply_independence_guard.py

Inputs:
    data/processed/integrated/ingredient_gene_sets_v1.parquet
    data/processed/integrated/compound_gene_integrated_v1.parquet
    data/processed/canonical/ingredient_compound_canonical.csv

Outputs (v1 untouched):
    data/processed/integrated/ingredient_gene_sets_v2.parquet
    data/processed/integrated/independence_guard_report_v2.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from integrate_weighted_edges import (  # noqa: E402
    collapse_metrics,
    compound_gene_lookup,
    noisy_or,
)

ROOT = Path(__file__).resolve().parents[2]
INTEGRATED_DIR = ROOT / "data/processed/integrated"
CORPUS_PATH = ROOT / "data/processed/corpus/compound_target_corpus_v1.parquet"

INGREDIENT_GENES_V1 = INTEGRATED_DIR / "ingredient_gene_sets_v1.parquet"
INTEGRATED_CG = INTEGRATED_DIR / "compound_gene_integrated_v1.parquet"
INGREDIENT_COMPOUND = ROOT / "data/processed/canonical/ingredient_compound_canonical.csv"
MEASURED_CG = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"

INGREDIENT_GENES_V2 = INTEGRATED_DIR / "ingredient_gene_sets_v2.parquet"
REPORT_OUT = INTEGRATED_DIR / "independence_guard_report_v2.json"

CONFIDENCE_FLOORS_FOR_SENSITIVITY = [0.3, 0.5, 0.7]


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"n": 0}
    qs = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(qs[1]),
        "p25": float(qs[2]),
        "p75": float(qs[4]),
        "p90": float(qs[5]),
    }


def neighbor_group_key(nn: Any, compound_id: str) -> str:
    if pd.isna(nn) or not str(nn).strip():
        return f"__missing_{compound_id}"
    return str(nn)


def guarded_confidence_inchikey(
    pred_support: list[dict[str, Any]],
) -> tuple[float, int, int]:
    """
    Group by nearest_neighbor_inchikey; max within group; noisy-OR across groups.
    Returns (confidence, n_neighbor_groups, n_compounds).
    """
    nn_groups: dict[str, list[float]] = defaultdict(list)
    for s in pred_support:
        key = neighbor_group_key(s.get("nearest_neighbor_inchikey"), s["compound_id"])
        nn_groups[key].append(float(s["confidence_weight"]))
    group_weights = [max(ws) for ws in nn_groups.values()]
    return noisy_or(group_weights), len(nn_groups), len(pred_support)


def guarded_confidence_scaffold(
    pred_support: list[dict[str, Any]],
    nn_to_scaffold: dict[str, str | None],
) -> tuple[float, int]:
    """Sensitivity: group by Murcko scaffold of nearest neighbor."""
    scaf_groups: dict[str, list[float]] = defaultdict(list)
    for s in pred_support:
        nn = neighbor_group_key(s.get("nearest_neighbor_inchikey"), s["compound_id"])
        scaf = nn_to_scaffold.get(nn)
        key = scaf if scaf else f"__noscaf_{nn}"
        scaf_groups[key].append(float(s["confidence_weight"]))
    group_weights = [max(ws) for ws in scaf_groups.values()]
    return noisy_or(group_weights), len(scaf_groups)


def build_nn_scaffold_map(
    integrated: pd.DataFrame,
) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Murcko scaffold for each unique nearest_neighbor_inchikey in predicted edges."""
    pred = integrated.loc[integrated["source"] == "predicted"]
    nn_keys = sorted(pred["nearest_neighbor_inchikey"].dropna().astype(str).unique())

    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        import feasibility_recon as fr

        fr.configure_rdkit_logging()
    except ImportError:
        return {}, {"available": False, "error": "RDKit not available"}

    corpus = pd.read_parquet(
        CORPUS_PATH,
        columns=["compound_inchikey", "compound_smiles", "compound_inchi"],
    )
    corpus = corpus.drop_duplicates("compound_inchikey")
    lookup = corpus.set_index("compound_inchikey")

    nn_to_scaffold: dict[str, str | None] = {}
    ok = failed = missing = 0
    for nn in nn_keys:
        if nn not in lookup.index:
            nn_to_scaffold[nn] = None
            missing += 1
            continue
        row = lookup.loc[nn]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        mol = fr.mol_from_smiles_or_inchi(row.get("compound_smiles"), row.get("compound_inchi"))
        if mol is None:
            nn_to_scaffold[nn] = None
            failed += 1
            continue
        try:
            nn_to_scaffold[nn] = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
            ok += 1
        except Exception:
            nn_to_scaffold[nn] = None
            failed += 1

    meta = {
        "available": True,
        "n_unique_neighbors": len(nn_keys),
        "scaffolds_ok": ok,
        "scaffolds_failed": failed,
        "neighbors_missing_from_corpus": missing,
    }
    return nn_to_scaffold, meta


def build_guarded_gene_sets(
    ingredient_compound: pd.DataFrame,
    integrated: pd.DataFrame,
    nn_to_scaffold: dict[str, str | None],
) -> pd.DataFrame:
    cg_lookup = compound_gene_lookup(integrated)
    ing_to_compounds: dict[str, set[str]] = defaultdict(set)
    for row in ingredient_compound.itertuples(index=False):
        ing_to_compounds[row.ingredient_id].add(row.compound_id)

    rows: list[dict[str, Any]] = []
    for ing, compounds in sorted(ing_to_compounds.items()):
        gene_support: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cmp in compounds:
            for edge in cg_lookup.get(cmp, []):
                gene_support[edge["gene_symbol"]].append({**edge, "compound_id": cmp})

        for gene, supporting in gene_support.items():
            has_measured = any(s["source"] == "measured" for s in supporting)
            pred = [s for s in supporting if s["source"] == "predicted"]
            n_meas = sum(1 for s in supporting if s["source"] == "measured")
            n_pred = len(pred)
            n_compounds = len({s["compound_id"] for s in supporting})

            if has_measured:
                conf_v2 = 1.0
                conf_scaffold = 1.0
                n_nn_groups = 0
                n_scaffold_groups = 0
                evidence = "measured"
                redundant = False
            else:
                conf_v1 = noisy_or([s["confidence_weight"] for s in pred])
                conf_v2, n_nn_groups, _ = guarded_confidence_inchikey(pred)
                conf_scaffold, n_scaffold_groups = guarded_confidence_scaffold(
                    pred, nn_to_scaffold
                )
                evidence = "predicted"
                redundant = n_pred >= 2 and n_nn_groups < n_pred

            rows.append(
                {
                    "ingredient_id": ing,
                    "gene_symbol": gene,
                    "confidence": conf_v2,
                    "confidence_v1_naive_noisy_or": conf_v1 if not has_measured else 1.0,
                    "confidence_scaffold_grouped": conf_scaffold,
                    "evidence": evidence,
                    "n_supporting_compounds": n_compounds,
                    "n_measured_compounds": n_meas,
                    "n_predicted_compounds": n_pred,
                    "n_independent_neighbor_groups": n_nn_groups,
                    "n_scaffold_neighbor_groups": n_scaffold_groups,
                    "redundancy_flag": redundant,
                    "confidence_method": (
                        "measured_authoritative"
                        if has_measured
                        else "nn_max_per_group_then_noisy_or"
                    ),
                }
            )
    return pd.DataFrame(rows)


def deflation_analysis(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> dict[str, Any]:
    merged = v1.merge(
        v2[
            [
                "ingredient_id",
                "gene_symbol",
                "confidence",
                "confidence_v1_naive_noisy_or",
                "confidence_scaffold_grouped",
                "n_independent_neighbor_groups",
            ]
        ].rename(columns={"confidence": "confidence_v2_guarded"}),
        on=["ingredient_id", "gene_symbol"],
    )
    merged["confidence_v1"] = merged["confidence"]
    merged["delta_v2"] = merged["confidence_v2_guarded"] - merged["confidence_v1"]
    merged["delta_scaffold"] = merged["confidence_scaffold_grouped"] - merged["confidence_v1"]

    pred_only = merged["evidence"] == "predicted"
    redundant = merged["redundancy_flag"] == True  # noqa: E712

    redundant_rows = merged.loc[pred_only & redundant]
    all_pred = merged.loc[pred_only]

    return {
        "previously_flagged_redundant_rows": {
            "n": int(len(redundant_rows)),
            "confidence_v1": distribution_summary(redundant_rows["confidence_v1"].to_numpy()),
            "confidence_v2_guarded": distribution_summary(redundant_rows["confidence_v2_guarded"].to_numpy()),
            "delta_v2_minus_v1": distribution_summary(redundant_rows["delta_v2"].to_numpy()),
            "confidence_scaffold_grouped": distribution_summary(
                redundant_rows["confidence_scaffold_grouped"].to_numpy()
            ),
            "delta_scaffold_minus_v1": distribution_summary(
                redundant_rows["delta_scaffold"].to_numpy()
            ),
        },
        "all_predicted_only_rows": {
            "n": int(len(all_pred)),
            "confidence_v1": distribution_summary(all_pred["confidence_v1"].to_numpy()),
            "confidence_v2_guarded": distribution_summary(all_pred["confidence_v2_guarded"].to_numpy()),
            "delta_v2_minus_v1": distribution_summary(all_pred["delta_v2"].to_numpy()),
            "confidence_scaffold_grouped": distribution_summary(
                all_pred["confidence_scaffold_grouped"].to_numpy()
            ),
            "delta_scaffold_minus_v1": distribution_summary(all_pred["delta_scaffold"].to_numpy()),
        },
        "pct_redundant_rows_deflated": float(
            (redundant_rows["delta_v2"] < -1e-9).mean() * 100 if len(redundant_rows) else 0.0
        ),
        "mean_abs_deflation_redundant": float(
            redundant_rows["delta_v2"].abs().mean() if len(redundant_rows) else 0.0
        ),
    }


def gene_set_size_analysis(v1: pd.DataFrame, v2: pd.DataFrame) -> dict[str, Any]:
    v1_per_ing = v1.groupby("ingredient_id").size()
    v2_per_ing = v2.groupby("ingredient_id").size()

    out: dict[str, Any] = {
        "confidence_floor_applied_for_membership": False,
        "note": (
            "v2 retains all genes present in v1; only confidence values change. "
            "Gene counts unchanged unless a confidence floor is applied (none by default)."
        ),
        "genes_per_ingredient_v1_median": float(v1_per_ing.median()),
        "genes_per_ingredient_v2_median": float(v2_per_ing.median()),
        "genes_per_ingredient_v1_mean": float(v1_per_ing.mean()),
        "genes_per_ingredient_v2_mean": float(v2_per_ing.mean()),
        "total_gene_rows_v1": int(len(v1)),
        "total_gene_rows_v2": int(len(v2)),
    }

    floor_sensitivity: dict[str, Any] = {}
    for floor in CONFIDENCE_FLOORS_FOR_SENSITIVITY:
        v1_keep = v1.groupby("ingredient_id").apply(
            lambda g: (g["confidence"] >= floor).sum(), include_groups=False
        )
        v2_keep = v2.groupby("ingredient_id").apply(
            lambda g: (g["confidence"] >= floor).sum(), include_groups=False
        )
        floor_sensitivity[str(floor)] = {
            "median_genes_per_ingredient_v1": float(v1_keep.median()),
            "median_genes_per_ingredient_v2": float(v2_keep.median()),
            "genes_dropped_v1_vs_unfiltered": int(len(v1) - v1_keep.sum()),
            "genes_dropped_v2_vs_unfiltered": int(len(v2) - v2_keep.sum()),
            "additional_genes_dropped_v2_vs_v1_at_floor": int(v1_keep.sum() - v2_keep.sum()),
        }
    out["hypothetical_confidence_floor_sensitivity"] = floor_sensitivity
    return out


def collapse_at_floors(df: pd.DataFrame, floors: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for floor in floors:
        filtered: dict[str, frozenset[str]] = {}
        for ing, grp in df.groupby("ingredient_id"):
            kept = grp.loc[grp["confidence"] >= floor, "gene_symbol"].astype(str)
            filtered[str(ing)] = frozenset(kept)
        result[str(floor)] = collapse_metrics(filtered)
    return result


def measured_only_collapse(
    measured: pd.DataFrame,
    ingredient_compound: pd.DataFrame,
) -> dict[str, Any]:
    cg = measured.groupby("compound_id")["gene_symbol"].apply(set).to_dict()
    ing_genes: dict[str, set[str]] = defaultdict(set)
    for row in ingredient_compound.itertuples(index=False):
        ing_genes[row.ingredient_id] |= cg.get(row.compound_id, set())
    return collapse_metrics({k: frozenset(v) for k, v in ing_genes.items()})


def main() -> int:
    print("=== Independence guard: ingredient gene sets v2 ===", flush=True)
    INTEGRATED_DIR.mkdir(parents=True, exist_ok=True)

    for path in (INGREDIENT_GENES_V1, INTEGRATED_CG, INGREDIENT_COMPOUND):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1

    v1 = pd.read_parquet(INGREDIENT_GENES_V1)
    integrated = pd.read_parquet(INTEGRATED_CG)
    ingredient_compound = pd.read_csv(INGREDIENT_COMPOUND)
    measured = pd.read_csv(MEASURED_CG)

    print("Building Murcko scaffold map for nearest neighbors...", flush=True)
    nn_to_scaffold, scaffold_meta = build_nn_scaffold_map(integrated)
    print(f"  scaffold meta: {scaffold_meta}", flush=True)

    print("Rebuilding gene sets with independence guard...", flush=True)
    v2_full = build_guarded_gene_sets(ingredient_compound, integrated, nn_to_scaffold)

    # v2 output columns (drop internal recompute column for v1 naive in main file? keep for audit)
    v2_out = v2_full.drop(columns=["confidence_v1_naive_noisy_or"])
    v2_out.to_parquet(INGREDIENT_GENES_V2, index=False)

    deflation = deflation_analysis(v1, v2_full)
    size_analysis = gene_set_size_analysis(v1, v2_out)

    # Raw collapse: all genes (no floor)
    v1_sets = {
        str(ing): frozenset(g["gene_symbol"].astype(str))
        for ing, g in v1.groupby("ingredient_id")
    }
    v2_sets = {
        str(ing): frozenset(g["gene_symbol"].astype(str))
        for ing, g in v2_out.groupby("ingredient_id")
    }
    collapse_v1 = collapse_metrics(v1_sets)
    collapse_v2_raw = collapse_metrics(v2_sets)
    collapse_measured = measured_only_collapse(measured, ingredient_compound)
    collapse_floors_v1 = collapse_at_floors(v1, CONFIDENCE_FLOORS_FOR_SENSITIVITY)
    collapse_floors_v2 = collapse_at_floors(v2_out, CONFIDENCE_FLOORS_FOR_SENSITIVITY)

    report: dict[str, Any] = {
        "guard_method": {
            "default": (
                "Within each nearest_neighbor_inchikey group: representative weight = max(w_i). "
                "N compounds borrowing from the same neighbor = 1 independent evidence unit. "
                "Across distinct neighbor groups: confidence = 1 - prod(1 - w_group)."
            ),
            "scaffold_sensitivity": (
                "Same max-then-noisy-OR but groups formed by Murcko scaffold of neighbor compound "
                "(not default; reported for correlation sensitivity bound)."
            ),
        },
        "scaffold_map_meta": scaffold_meta,
        "deflation": deflation,
        "gene_set_sizes": size_analysis,
        "collapse_remeasurement": {
            "measured_only_baseline": collapse_measured,
            "v1_integrated_all_genes": collapse_v1,
            "v2_guarded_all_genes_raw": collapse_v2_raw,
            "raw_collapse_delta_v2_vs_v1": {
                "ingredients_in_collapse_groups": (
                    collapse_v2_raw["n_ingredients_in_collapse_groups"]
                    - collapse_v1["n_ingredients_in_collapse_groups"]
                ),
                "unique_gene_sets": (
                    collapse_v2_raw["n_unique_gene_sets"] - collapse_v1["n_unique_gene_sets"]
                ),
                "largest_group": (
                    collapse_v2_raw["largest_collapse_group_size"]
                    - collapse_v1["largest_collapse_group_size"]
                ),
                "note": (
                    "Raw collapse uses gene-set identity (frozenset of gene symbols). "
                    "With no confidence floor, membership is unchanged from v1 -> collapse identical."
                ),
            },
            "at_confidence_floors_v1": collapse_floors_v1,
            "at_confidence_floors_v2_guarded": collapse_floors_v2,
        },
        "outputs": {
            "ingredient_gene_sets_v2": str(INGREDIENT_GENES_V2),
            "ingredient_gene_sets_v1_unchanged": str(INGREDIENT_GENES_V1),
        },
        "stop_note": "Guard applied. Enrichment NOT rebuilt — review deflation and collapse first.",
    }

    for key in ("measured_only_baseline", "v1_integrated_all_genes", "v2_guarded_all_genes_raw"):
        report["collapse_remeasurement"][key].pop("signature_to_ingredients", None)
    for floor_dict in (
        report["collapse_remeasurement"]["at_confidence_floors_v1"],
        report["collapse_remeasurement"]["at_confidence_floors_v2_guarded"],
    ):
        for v in floor_dict.values():
            v.pop("signature_to_ingredients", None)

    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Deflation (previously flagged redundant rows) ---", flush=True)
    rd = deflation["previously_flagged_redundant_rows"]
    print(f"  n={rd['n']:,}", flush=True)
    print(
        f"  confidence v1 mean={rd['confidence_v1']['mean']:.4f} "
        f"-> v2 guarded mean={rd['confidence_v2_guarded']['mean']:.4f} "
        f"(delta mean={rd['delta_v2_minus_v1']['mean']:.4f})",
        flush=True,
    )
    print(
        f"  scaffold-grouped mean={rd['confidence_scaffold_grouped']['mean']:.4f} "
        f"(delta vs v1 mean={rd['delta_scaffold_minus_v1']['mean']:.4f})",
        flush=True,
    )

    print("\n--- Gene set sizes (no membership floor) ---", flush=True)
    print(
        f"  median genes/ingredient v1={size_analysis['genes_per_ingredient_v1_median']:.0f} "
        f"v2={size_analysis['genes_per_ingredient_v2_median']:.0f}",
        flush=True,
    )

    print("\n--- Raw collapse (all genes, no floor) ---", flush=True)
    print(
        f"  v1: {collapse_v1['n_ingredients_in_collapse_groups']}/223 in collapse, "
        f"{collapse_v1['n_unique_gene_sets']} unique sets, largest={collapse_v1['largest_collapse_group_size']}",
        flush=True,
    )
    print(
        f"  v2: {collapse_v2_raw['n_ingredients_in_collapse_groups']}/223 in collapse, "
        f"{collapse_v2_raw['n_unique_gene_sets']} unique sets, largest={collapse_v2_raw['largest_collapse_group_size']}",
        flush=True,
    )

    print(f"\nWrote {INGREDIENT_GENES_V2}", flush=True)
    print(f"Wrote {REPORT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
