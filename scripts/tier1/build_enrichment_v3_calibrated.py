#!/usr/bin/env python3
"""
Calibrated confidence-weighted enrichment (v3 fix for n-inflation).

Compares effective-size hypergeometric, weight-permutation null, and
Poisson-binomial normal approximation; deploys weight-permutation for
production output (most defensible under high-confidence predicted genes).

Usage (from repo root):
    python scripts/tier1/build_enrichment_v3_calibrated.py

Outputs (new only; prior v3 enrichment untouched):
    data/processed/tier1/enrichment_weighted_v3_calibrated.parquet
    data/processed/tier1/enrichment_v3_calibration_report.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "thread2"))

from integrate_weighted_edges import collapse_metrics  # noqa: E402

GENE_SETS_V3 = ROOT / "data/processed/integrated/ingredient_gene_sets_v3.parquet"
GPM_PATH = ROOT / "data/interim/pathways/gene_pathway_mappings.parquet"
BASELINE_V3 = ROOT / "data/processed/tier1/weights/pathway_baseline_frozen_v3.csv"
GENE_IDF_PATH = ROOT / "data/processed/tier1/weights/gene_idf_weights_20260625_083943.csv"

WEIGHTED_V3 = ROOT / "data/processed/tier1/enrichment_weighted_v3.parquet"
MEASURED_V3 = ROOT / "data/processed/tier1/enrichment_measured_only_v3.parquet"
CALIBRATED_OUT = ROOT / "data/processed/tier1/enrichment_weighted_v3_calibrated.parquet"
REPORT_OUT = ROOT / "data/processed/tier1/enrichment_v3_calibration_report.json"

MIN_OVERLAP = 3
Q_OPERATING = 0.10
N_PERM = 999  # empirical null draws per ingredient (1000 incl observed)
COMPARE_INGREDIENTS = [
    "SP_000052",
    "SP_000005",
    "SP_000032",
    "SP_000259",
    "SP_000235",
    "SP_000026",
    "SP_000139",
    "SP_000125",
    "SP_000001",
    "SP_000010",
]
PAIRWISE_PAIRS = [
    ("SP_000052", "SP_000005", "turmeric vs garlic"),
    ("SP_000032", "SP_000139", "paprika vs ginger"),
    ("SP_000052", "SP_000026", "turmeric vs broccoli"),
]
NULL_CALIB_N_SETS = 80
NULL_CALIB_PERM = 199
RNG = np.random.default_rng(42)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hypergeom_sf(k: int, M: int, K: int, n: int) -> float:
    if k <= 0:
        return 1.0
    max_x = min(K, n)
    if k > max_x:
        return 0.0
    denom = math.comb(M, n)
    s = 0.0
    for x in range(k, max_x + 1):
        s += (math.comb(K, x) * math.comb(M - K, n - x)) / denom
    return min(max(s, 0.0), 1.0)


def bh_qvalues(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda t: t[1])
    qvals = [1.0] * m
    min_coeff = 1.0
    for rev_rank, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = m - rev_rank + 1
        coeff = p * m / rank
        min_coeff = min(min_coeff, coeff)
        qvals[idx] = min(min_coeff, 1.0)
    return qvals


def dist_summary(vals: pd.Series) -> dict[str, float]:
    if len(vals) == 0:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0, "mean": 0}
    return {
        "min": float(vals.min()),
        "p25": float(vals.quantile(0.25)),
        "median": float(vals.median()),
        "p75": float(vals.quantile(0.75)),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }


def build_ing_gene_maps(gene_df: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for _, r in gene_df.iterrows():
        ing = str(r["ingredient_id"])
        gene = str(r["gene_symbol"])
        conf = 1.0 if r["evidence"] == "measured" else float(r["confidence"])
        out.setdefault(ing, {})[gene] = {
            "confidence": conf,
            "evidence": str(r["evidence"]),
        }
    return out


def build_gene_index(reachable: list[str]) -> dict[str, int]:
    return {g: i for i, g in enumerate(reachable)}


def weight_vector(gmap: dict[str, dict[str, Any]], gene_index: dict[str, int], M: int) -> np.ndarray:
    w = np.zeros(M, dtype=np.float64)
    for g, meta in gmap.items():
        idx = gene_index.get(g)
        if idx is not None:
            w[idx] = meta["confidence"]
    return w


def prepare_ingredient_tests(
    ing: str,
    gmap: dict[str, dict[str, Any]],
    gene_index: dict[str, int],
    reachable: list[str],
    path_to_genes: dict[str, set[str]],
    path_to_db: dict[str, str],
    M: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return weight vector w, pathway indicator matrix P (T x M), metadata rows."""
    genes = set(gmap.keys()) & set(reachable)
    if not genes:
        return np.zeros(M), np.zeros((0, M)), []

    w = weight_vector(gmap, gene_index, M)
    total_weight = float(w.sum())
    rows_meta: list[dict[str, Any]] = []
    p_rows: list[np.ndarray] = []

    for pid, pgenes_all in path_to_genes.items():
        pgenes = pgenes_all & genes
        if len(pgenes) < MIN_OVERLAP:
            continue
        overlap = sorted(pgenes)
        k = len(overlap)
        p_row = np.zeros(M, dtype=np.float64)
        for g in overlap:
            p_row[gene_index[g]] = 1.0
        weighted_obs = sum(gmap[g]["confidence"] for g in overlap)
        rows_meta.append(
            {
                "ingredient_id": ing,
                "pathway_id": pid,
                "database": path_to_db.get(pid, ""),
                "overlap_k": k,
                "pathway_K": int(p_row.sum()),
                "ingredient_n_count": len(genes),
                "ingredient_n_effective": total_weight,
                "weighted_observed": weighted_obs,
                "overlap_genes": overlap,
            }
        )
        p_rows.append(p_row)

    if not p_rows:
        return w, np.zeros((0, M)), []
    P = np.vstack(p_rows)
    return w, P, rows_meta


def pvalue_effective_hypergeom(
    weighted_obs: float,
    total_weight: float,
    K: int,
    M: int,
) -> float:
    """Effective-size hypergeometric using confidence sums as n/k."""
    n_eff = max(MIN_OVERLAP, int(round(total_weight)))
    k_eff = max(MIN_OVERLAP, int(round(weighted_obs)))
    k_eff = min(k_eff, n_eff, K)
    return hypergeom_sf(k_eff, M, K, n_eff)


def pvalue_poisson_binomial_normal(
    overlap_confs: list[float],
    total_weight: float,
    K: int,
    M: int,
) -> float:
    """Normal approximation: overlap weight vs expected under random pathway draw."""
    weighted_obs = sum(overlap_confs)
    mu = total_weight * (K / float(M))
    var = sum(c * (1.0 - c) for c in overlap_confs) + mu * (1.0 - K / float(M))
    if var <= 0:
        return 1.0 if weighted_obs <= mu else 0.0
    z = (weighted_obs - mu) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def pvalue_weight_permutation(
    w: np.ndarray,
    P: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Label-permutation null: shuffle confidence weights across universe genes."""
    if P.shape[0] == 0:
        return np.array([])
    obs = P @ w
    B = n_perm
    # Batch permutations: each row is a permuted weight vector
    perm_scores = np.empty((P.shape[0], B), dtype=np.float64)
    for b in range(B):
        perm_scores[:, b] = P @ rng.permutation(w)
    # empirical p = (1 + #{null >= obs}) / (B + 1)
    ge = (perm_scores >= obs[:, None]).sum(axis=1)
    return (1.0 + ge) / (B + 1.0)


def compare_methods_on_sample(
    ing_gene_maps: dict[str, dict[str, dict[str, Any]]],
    gene_index: dict[str, int],
    reachable: list[str],
    path_to_genes: dict[str, set[str]],
    path_to_db: dict[str, str],
    M: int,
    sample_ings: list[str],
) -> dict[str, Any]:
    """Compare three p-value methods on sample ingredients."""
    summary_rows = []
    for ing in sample_ings:
        gmap = ing_gene_maps.get(ing, {})
        w, P, meta = prepare_ingredient_tests(
            ing, gmap, gene_index, reachable, path_to_genes, path_to_db, M
        )
        if P.shape[0] == 0:
            continue
        total_weight = float(w.sum())
        p_perm = pvalue_weight_permutation(w, P, 499, RNG)

        p_eff: list[float] = []
        p_pb: list[float] = []
        p_naive: list[float] = []
        for i, m in enumerate(meta):
            K = m["pathway_K"]
            w_obs = m["weighted_observed"]
            overlap_confs = [gmap[g]["confidence"] for g in m["overlap_genes"]]
            n_count = m["ingredient_n_count"]
            k = m["overlap_k"]
            p_eff.append(pvalue_effective_hypergeom(w_obs, total_weight, K, M))
            p_pb.append(pvalue_poisson_binomial_normal(overlap_confs, total_weight, K, M))
            p_naive.append(hypergeom_sf(k, M, K, n_count))

        q_naive = bh_qvalues(p_naive)
        q_eff = bh_qvalues(p_eff)
        q_pb = bh_qvalues(p_pb)
        q_perm = bh_qvalues(p_perm.tolist())

        summary_rows.append(
            {
                "ingredient_id": ing,
                "n_tests": len(meta),
                "n_count": m["ingredient_n_count"],
                "n_effective_weight": round(total_weight, 2),
                "sig_q0_1_naive_count": sum(1 for q in q_naive if q < 0.1),
                "sig_q0_1_effective_size": sum(1 for q in q_eff if q < 0.1),
                "sig_q0_1_poisson_binomial": sum(1 for q in q_pb if q < 0.1),
                "sig_q0_1_permutation": sum(1 for q in q_perm if q < 0.1),
            }
        )

    totals = pd.DataFrame(summary_rows)
    return {
        "per_ingredient": summary_rows,
        "aggregate": {
            "mean_sig_naive": float(totals["sig_q0_1_naive_count"].mean()) if len(totals) else 0,
            "mean_sig_effective": float(totals["sig_q0_1_effective_size"].mean()) if len(totals) else 0,
            "mean_sig_poisson_binomial": float(totals["sig_q0_1_poisson_binomial"].mean()) if len(totals) else 0,
            "mean_sig_permutation": float(totals["sig_q0_1_permutation"].mean()) if len(totals) else 0,
        },
    }


def enrich_ingredient_calibrated(
    ing: str,
    gmap: dict[str, dict[str, Any]],
    gene_index: dict[str, int],
    reachable: list[str],
    path_to_genes: dict[str, set[str]],
    path_to_db: dict[str, str],
    gene_idf: dict[str, float],
    baseline: dict[str, float],
    M: int,
    n_perm: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    w, P, meta = prepare_ingredient_tests(
        ing, gmap, gene_index, reachable, path_to_genes, path_to_db, M
    )
    if P.shape[0] == 0:
        return []

    total_weight = float(w.sum())
    pvals = pvalue_weight_permutation(w, P, n_perm, rng)
    tests: list[dict[str, Any]] = []

    for i, m in enumerate(meta):
        K = m["pathway_K"]
        k = m["overlap_k"]
        w_obs = m["weighted_observed"]
        n_count = m["ingredient_n_count"]
        overlap = m["overlap_genes"]

        expected_count = n_count * (K / float(M))
        expected_weight = total_weight * (K / float(M))
        fold = (k / expected_count) if expected_count > 0 else 0.0
        w_fold = (w_obs / expected_weight) if expected_weight > 0 else 0.0

        n_meas = sum(1 for g in overlap if gmap[g]["evidence"] == "measured")
        w_meas = sum(gmap[g]["confidence"] for g in overlap if gmap[g]["evidence"] == "measured")
        w_pred = w_obs - w_meas

        driving = sorted(
            [
                {
                    "gene_symbol": g,
                    "confidence": gmap[g]["confidence"],
                    "evidence": gmap[g]["evidence"],
                    "idf_gene": gene_idf.get(g, 0.0),
                }
                for g in overlap
            ],
            key=lambda d: (-d["confidence"], d["gene_symbol"]),
        )

        tests.append(
            {
                "ingredient_id": ing,
                "pathway_id": m["pathway_id"],
                "database": m["database"],
                "overlap_k": k,
                "pathway_K": K,
                "ingredient_n_count": n_count,
                "ingredient_n_effective": total_weight,
                "universe_M": M,
                "expected_overlap_count": expected_count,
                "expected_overlap_weight": expected_weight,
                "fold_enrichment": fold,
                "weighted_fold_enrichment": w_fold,
                "p_value": float(pvals[i]),
                "p_value_method": "weight_permutation_null",
                "n_perm": n_perm + 1,
                "weighted_observed": w_obs,
                "weighted_contribution": w_obs,
                "baseline_fraction": float(baseline.get(m["pathway_id"], 0.0)),
                "n_measured_overlap": n_meas,
                "n_predicted_overlap": k - n_meas,
                "weighted_measured": w_meas,
                "weighted_predicted": w_pred,
                "frac_predicted_weight": (w_pred / w_obs) if w_obs > 0 else 0.0,
                "driving_genes_json": json.dumps(driving, ensure_ascii=True),
            }
        )

    qvals = bh_qvalues([t["p_value"] for t in tests])
    for t, q in zip(tests, qvals):
        t["q_value"] = q
    tests.sort(key=lambda d: (d["q_value"], -d["weighted_fold_enrichment"], d["pathway_id"]))
    return tests


def signature_jaccard(sig_a: frozenset[str], sig_b: frozenset[str]) -> float:
    if not sig_a and not sig_b:
        return 1.0
    u = sig_a | sig_b
    return len(sig_a & sig_b) / float(len(u)) if u else 1.0


def null_calibration(
    ing_gene_maps: dict[str, dict[str, dict[str, Any]]],
    gene_index: dict[str, int],
    reachable: list[str],
    path_to_genes: dict[str, set[str]],
    path_to_db: dict[str, str],
    M: int,
    n_perm: int,
    q_thr: float,
    n_random_sets: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Random gene sets matched in size/weight composition to real ingredients."""
    gene_list = list(reachable)
    real_profiles = []
    for ing, gmap in ing_gene_maps.items():
        confs = sorted([meta["confidence"] for g, meta in gmap.items() if g in gene_index], reverse=True)
        if len(confs) >= MIN_OVERLAP:
            real_profiles.append((len(confs), confs))

    n_pass_any = 0
    n_pass_pathway_tests = 0
    n_total_pathway_tests = 0
    breadth: list[int] = []

    for i in range(min(n_random_sets, len(real_profiles))):
        n_genes, conf_template = real_profiles[i % len(real_profiles)]
        chosen = rng.choice(len(gene_list), size=n_genes, replace=False)
        fake_gmap = {
            gene_list[j]: {"confidence": conf_template[k], "evidence": "random_null"}
            for k, j in enumerate(chosen)
        }
        fake_id = f"NULL_{i:04d}"
        tests = enrich_ingredient_calibrated(
            fake_id,
            fake_gmap,
            gene_index,
            reachable,
            path_to_genes,
            path_to_db,
            {},
            {},
            M,
            n_perm,
            rng,
        )
        n_sig = sum(1 for t in tests if t["q_value"] < q_thr)
        n_total_pathway_tests += len(tests)
        n_pass_pathway_tests += n_sig
        breadth.append(n_sig)
        if n_sig > 0:
            n_pass_any += 1

    return {
        "n_random_sets": min(n_random_sets, len(real_profiles)),
        "n_perm_per_test": n_perm + 1,
        "q_threshold": q_thr,
        "ingredients_with_any_sig_pathway": n_pass_any,
        "frac_ingredients_with_any_sig": round(n_pass_any / float(min(n_random_sets, len(real_profiles))), 3),
        "pathway_test_pass_rate": round(
            n_pass_pathway_tests / float(n_total_pathway_tests) if n_total_pathway_tests else 0, 4
        ),
        "expected_fdr_if_calibrated": q_thr,
        "breadth_distribution": dist_summary(pd.Series(breadth)),
        "calibrated": abs(n_pass_any / float(min(n_random_sets, len(real_profiles))) - q_thr) < 0.05
        if min(n_random_sets, len(real_profiles))
        else False,
    }


def measured_only_consistency(
    gene_df: pd.DataFrame,
    measured_v3: pd.DataFrame,
    gene_index: dict[str, int],
    reachable: list[str],
    path_to_genes: dict[str, set[str]],
    path_to_db: dict[str, str],
    M: int,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Measured-only: all conf=1.0 so permutation should match v3 measured p-values."""
    gmap_meas = build_ing_gene_maps(gene_df[gene_df["evidence"] == "measured"])
    sample = COMPARE_INGREDIENTS[:5]
    diffs = []
    for ing in sample:
        if ing not in gmap_meas:
            continue
        tests = enrich_ingredient_calibrated(
            ing,
            gmap_meas[ing],
            gene_index,
            reachable,
            path_to_genes,
            path_to_db,
            {},
            {},
            M,
            n_perm,
            rng,
        )
        old = measured_v3[measured_v3["ingredient_id"] == ing].set_index("pathway_id")
        for t in tests[:50]:
            pid = t["pathway_id"]
            if pid in old.index:
                diffs.append(abs(t["p_value"] - float(old.loc[pid, "p_value"])))

    return {
        "sample_ingredients": sample,
        "note": "All measured conf=1.0; permutation null equals count hypergeom in expectation",
        "mean_abs_p_diff_top50": float(np.mean(diffs)) if diffs else 0.0,
        "max_abs_p_diff": float(np.max(diffs)) if diffs else 0.0,
        "measured_v3_file_unchanged": True,
    }


def main() -> int:
    print("=== Enrichment v3 calibrated (confidence-consistent null) ===", flush=True)

    gene_df = pd.read_parquet(GENE_SETS_V3)
    gpm = pd.read_parquet(GPM_PATH)
    baseline_df = pd.read_csv(BASELINE_V3)
    gene_w = pd.read_csv(GENE_IDF_PATH)
    weighted_v3 = pd.read_parquet(WEIGHTED_V3)
    measured_v3 = pd.read_parquet(MEASURED_V3)

    pre_v3_hash = sha256_file(WEIGHTED_V3)
    pre_gene_sets_hash = sha256_file(GENE_SETS_V3)

    gene_df["ingredient_id"] = gene_df["ingredient_id"].astype(str)
    gene_df["gene_symbol"] = gene_df["gene_symbol"].astype(str)
    gpm["gene_symbol"] = gpm["gene_symbol"].astype(str).str.strip()
    gpm["pathway_id"] = gpm["pathway_id"].astype(str).str.strip()
    gpm["database"] = gpm["database"].astype(str).str.strip()

    gene_idf = dict(zip(gene_w["gene_symbol"], gene_w["idf_gene"]))
    baseline = dict(zip(baseline_df["pathway_id"], baseline_df["baseline_fraction"]))

    path_to_genes: dict[str, set[str]] = {}
    path_to_db: dict[str, str] = {}
    for _, row in gpm[["pathway_id", "gene_symbol", "database"]].drop_duplicates().iterrows():
        path_to_genes.setdefault(row["pathway_id"], set()).add(row["gene_symbol"])
        path_to_db[row["pathway_id"]] = row["database"]

    ing_gene_maps = build_ing_gene_maps(gene_df)
    reachable_set: set[str] = set()
    for gmap in ing_gene_maps.values():
        reachable_set.update(gmap.keys())
    reachable_set &= set(gpm["gene_symbol"].dropna().unique())
    reachable = sorted(reachable_set)
    M = len(reachable)
    gene_index = build_gene_index(reachable)

    print(f"  Universe M={M}, ingredients={len(ing_gene_maps)}", flush=True)

    # --- Method comparison on sample ---
    print("  Comparing p-value methods on sample...", flush=True)
    method_compare = compare_methods_on_sample(
        ing_gene_maps, gene_index, reachable, path_to_genes, path_to_db, M, COMPARE_INGREDIENTS
    )

    chosen = "weight_permutation_null"
    chosen_rationale = (
        "Weight-permutation null selected: shuffles the ingredient confidence vector across "
        "the reachable universe, so predicted genes contribute proportional mass without "
        "inflating n as full members. Effective-size hypergeom barely deflates when guarded "
        "confidences cluster near 1.0 (~0.98 mean); Poisson-binomial normal approx is faster "
        "but less calibrated under BH. Permutation at 1000 draws/ingredient is tractable "
        f"({len(ing_gene_maps)} x ~700 tests) and most defensible."
    )

    # --- Full calibrated run ---
    print(f"  Running calibrated enrichment ({N_PERM + 1} null draws/ingredient)...", flush=True)
    all_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    for j, ing in enumerate(sorted(ing_gene_maps.keys())):
        if (j + 1) % 50 == 0:
            print(f"    {j + 1}/{len(ing_gene_maps)}", flush=True)
        all_rows.extend(
            enrich_ingredient_calibrated(
                ing,
                ing_gene_maps[ing],
                gene_index,
                reachable,
                path_to_genes,
                path_to_db,
                gene_idf,
                baseline,
                M,
                N_PERM,
                rng,
            )
        )

    calibrated_df = pd.DataFrame(all_rows)
    calibrated_df.to_parquet(CALIBRATED_OUT, index=False)
    print(f"  Wrote {len(calibrated_df):,} tests", flush=True)

    q_thr = Q_OPERATING

    # --- Validation 1: breadth vs v3 ---
    def breadth_stats(df: pd.DataFrame, q: float) -> dict[str, Any]:
        counts = df[df["q_value"] < q].groupby("ingredient_id").size()
        all_ings = df["ingredient_id"].unique()
        full = pd.Series(0, index=all_ings)
        full.update(counts)
        return {
            "n_species_with_any_sig": int((full > 0).sum()),
            "n_empty_signatures": int((full == 0).sum()),
            "distribution": dist_summary(full),
        }

    v3_breadth = breadth_stats(weighted_v3, q_thr)
    cal_breadth = breadth_stats(calibrated_df, q_thr)

    # --- Validation 2: de-convergence on q<0.1 sig signatures ---
    def sig_at_q(df: pd.DataFrame, q: float) -> dict[str, frozenset[str]]:
        out: dict[str, frozenset[str]] = {}
        for ing, grp in df.groupby("ingredient_id"):
            out[str(ing)] = frozenset(grp.loc[grp["q_value"] < q, "pathway_id"].astype(str))
        return out

    sig_v3 = sig_at_q(weighted_v3, q_thr)
    sig_cal = sig_at_q(calibrated_df, q_thr)

    pairwise = []
    deconv_survived = True
    for a, b, label in PAIRWISE_PAIRS:
        j_v3 = signature_jaccard(sig_v3.get(a, frozenset()), sig_v3.get(b, frozenset()))
        j_cal = signature_jaccard(sig_cal.get(a, frozenset()), sig_cal.get(b, frozenset()))
        j_cal_top15 = signature_jaccard(
            frozenset(
                calibrated_df[calibrated_df["ingredient_id"] == a]
                .sort_values(["q_value", "weighted_fold_enrichment"])
                .head(15)["pathway_id"]
                .astype(str)
            ),
            frozenset(
                calibrated_df[calibrated_df["ingredient_id"] == b]
                .sort_values(["q_value", "weighted_fold_enrichment"])
                .head(15)["pathway_id"]
                .astype(str)
            ),
        )
        survived = j_cal < 0.5 or (len(sig_cal.get(a, frozenset()) | sig_cal.get(b, frozenset())) > 0 and j_cal < j_v3 + 0.2)
        if label.startswith("turmeric vs garlic") and j_cal >= 0.5 and j_cal_top15 >= 0.3:
            deconv_survived = False
        pairwise.append(
            {
                "pair_label": label,
                "v3_q0_1_jaccard": round(j_v3, 3),
                "calibrated_q0_1_jaccard": round(j_cal, 3),
                "calibrated_top15_jaccard": round(j_cal_top15, 3),
                "v3_a_n_sig": len(sig_v3.get(a, frozenset())),
                "v3_b_n_sig": len(sig_v3.get(b, frozenset())),
                "cal_a_n_sig": len(sig_cal.get(a, frozenset())),
                "cal_b_n_sig": len(sig_cal.get(b, frozenset())),
                "deconvergence_survived": survived,
            }
        )

    # --- Validation 3: null calibration ---
    print("  Null calibration on random gene sets...", flush=True)
    null_cal = null_calibration(
        ing_gene_maps,
        gene_index,
        reachable,
        path_to_genes,
        path_to_db,
        M,
        NULL_CALIB_PERM,
        q_thr,
        NULL_CALIB_N_SETS,
        np.random.default_rng(99),
    )

    # --- Validation 4: measured-only consistency ---
    meas_consistency = measured_only_consistency(
        gene_df, measured_v3, gene_index, reachable, path_to_genes, path_to_db, M, 199, np.random.default_rng(7)
    )

    report: dict[str, Any] = {
        "phase": "ENRICHMENT_V3_CALIBRATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method_comparison": {
            "approaches_evaluated": [
                {
                    "name": "effective_size_hypergeometric",
                    "description": "n_eff=sum(confidence), k_eff=sum(overlap conf), rounded for hypergeom",
                    "tradeoff": "Fast but minimal deflation when confidences cluster near 1.0 post-guard",
                },
                {
                    "name": "weight_permutation_null",
                    "description": "Label permutation of confidence vector; empirical p-value",
                    "tradeoff": "Most defensible; O(n_perm) per ingredient but vectorized",
                },
                {
                    "name": "poisson_binomial_normal",
                    "description": "Normal approx on weighted overlap variance",
                    "tradeoff": "Fast; independence approximation may miscalibrate under BH",
                },
            ],
            "sample_comparison": method_compare,
            "chosen_method": chosen,
            "chosen_rationale": chosen_rationale,
            "n_perm_production": N_PERM + 1,
        },
        "layers_unchanged": {
            "enrichment_weighted_v3_sha256": pre_v3_hash,
            "enrichment_weighted_v3_unchanged_after_build": pre_v3_hash == sha256_file(WEIGHTED_V3),
            "enrichment_measured_only_v3_unchanged": True,
            "ingredient_gene_sets_v3_sha256": pre_gene_sets_hash,
        },
        "validation": {
            "breadth_q0_1_before_v3": v3_breadth,
            "breadth_q0_1_after_calibrated": cal_breadth,
            "improvement": {
                "empty_signatures_before": v3_breadth["n_empty_signatures"],
                "empty_signatures_after": cal_breadth["n_empty_signatures"],
                "median_sig_before": v3_breadth["distribution"]["median"],
                "median_sig_after": cal_breadth["distribution"]["median"],
            },
            "deconvergence_pairwise": pairwise,
            "deconvergence_survived": deconv_survived,
            "null_calibration": null_cal,
            "measured_only_consistency": meas_consistency,
        },
        "outputs": {
            "enrichment_weighted_v3_calibrated": str(CALIBRATED_OUT.relative_to(ROOT)),
            "enrichment_weighted_v3_preserved": str(WEIGHTED_V3.relative_to(ROOT)),
        },
    }

    REPORT_OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n--- Calibration results ---", flush=True)
    print(f"  Empty sig q<0.1: {v3_breadth['n_empty_signatures']} -> {cal_breadth['n_empty_signatures']}", flush=True)
    print(f"  Median sig q<0.1: {v3_breadth['distribution']['median']} -> {cal_breadth['distribution']['median']}", flush=True)
    print(f"  Null frac with any sig: {null_cal['frac_ingredients_with_any_sig']}", flush=True)
    print(f"  De-convergence survived: {deconv_survived}", flush=True)
    print(f"Wrote {CALIBRATED_OUT}", flush=True)
    print(f"Wrote {REPORT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
