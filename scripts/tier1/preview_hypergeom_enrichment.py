from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IC_PATH = ROOT / "data/processed/canonical/ingredient_compound_canonical.csv"
CG_PATH = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"
GPM_PATH = ROOT / "data/interim/pathways/gene_pathway_mappings.parquet"

WEIGHTS_DIR = ROOT / "data/processed/tier1/weights"
PREV_FORMULA_REPORT = WEIGHTS_DIR / "_enrichment_preview_report.json"
OUT_PREVIEW = WEIGHTS_DIR / "_hypergeom_preview.json"

PREVIEW_INGS = ["ING_000951", "ING_000390", "ING_000912"]


def latest_weight_csv(prefix: str) -> Path:
    files = sorted(WEIGHTS_DIR.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No weight table found for prefix: {prefix}")
    return files[-1]


def hypergeom_sf(k: int, M: int, K: int, n: int) -> float:
    """P[X >= k] for Hypergeom(M, K, n)."""
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
    for rank_rev, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = m - rank_rev + 1
        coeff = p * m / rank
        if coeff < min_coeff:
            min_coeff = coeff
        qvals[idx] = min(min_coeff, 1.0)
    return qvals


def main() -> int:
    ic = pd.read_csv(IC_PATH)
    cg = pd.read_csv(CG_PATH)
    gpm = pd.read_parquet(GPM_PATH)

    # Read existing weights (do not recompute).
    cmp_w_path = latest_weight_csv("compound_idf_weights")
    gene_w_path = latest_weight_csv("gene_idf_weights")
    baseline_path = latest_weight_csv("pathway_baseline_frozen")
    cmp_w = pd.read_csv(cmp_w_path)
    gene_w = pd.read_csv(gene_w_path)
    baseline = pd.read_csv(baseline_path)

    ic["ingredient_id"] = ic["ingredient_id"].astype(str).str.strip()
    ic["compound_id"] = ic["compound_id"].astype(str).str.strip().str.upper()
    cg["compound_id"] = cg["compound_id"].astype(str).str.strip().str.upper()
    cg["gene_symbol"] = cg["gene_symbol"].astype(str).str.strip()
    gpm["gene_symbol"] = gpm["gene_symbol"].astype(str).str.strip()
    gpm["pathway_id"] = gpm["pathway_id"].astype(str).str.strip()
    gpm["database"] = gpm["database"].astype(str).str.strip()

    gene_idf = dict(zip(gene_w["gene_symbol"], gene_w["idf_gene"]))

    # Universe choice: full gpm gene set (586 genes).
    universe_genes = set(gpm["gene_symbol"].dropna().unique().tolist())
    M = len(universe_genes)

    # Maps.
    ing_to_compounds = (
        ic.groupby("ingredient_id")["compound_id"]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )
    cmp_to_genes = (
        cg.groupby("compound_id")["gene_symbol"]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )

    path_to_genes = {}
    path_to_db = {}
    for _, row in gpm[["pathway_id", "gene_symbol", "database"]].drop_duplicates().iterrows():
        path_to_genes.setdefault(row["pathway_id"], set()).add(row["gene_symbol"])
        path_to_db[row["pathway_id"]] = row["database"]

    baseline_map = dict(zip(baseline["pathway_id"], baseline["baseline_fraction"]))

    prev_formula_counts = {}
    if PREV_FORMULA_REPORT.exists():
        prev = json.loads(PREV_FORMULA_REPORT.read_text(encoding="utf-8"))
        vals = prev.get("preview_concentration_n_pathways_enrichment_gt_0", {}).get("values", [])
        pings = prev.get("preview_selection", {}).get("ingredient_ids", [])
        for i, ing in enumerate(pings):
            if i < len(vals):
                prev_formula_counts[ing] = vals[i]

    preview_rows = []
    significant_counts = []
    ranking_change = []

    for ing in PREVIEW_INGS:
        compounds = ing_to_compounds.get(ing, set())
        ing_genes = set()
        for c in compounds:
            ing_genes |= cmp_to_genes.get(c, set())
        ing_genes &= universe_genes
        n = len(ing_genes)

        tests = []
        for pid, p_genes in path_to_genes.items():
            p_genes_u = p_genes & universe_genes
            K = len(p_genes_u)
            if K == 0 or n == 0:
                continue
            overlap = sorted(ing_genes & p_genes_u)
            k = len(overlap)
            expected = n * (K / float(M))
            pval = hypergeom_sf(k, M, K, n)
            fold = (k / expected) if expected > 0 else 0.0

            # IDF-weighted refinement (not replacing test):
            obs_w = sum(gene_idf.get(g, 0.0) for g in overlap)
            path_w_sum = sum(gene_idf.get(g, 0.0) for g in p_genes_u)
            exp_w = (n / float(M)) * path_w_sum
            fold_w = (obs_w / exp_w) if exp_w > 0 else 0.0

            tests.append(
                {
                    "pathway_id": pid,
                    "database": path_to_db.get(pid, ""),
                    "overlap_count": k,
                    "expected_overlap": expected,
                    "fold_enrichment": fold,
                    "p_value": pval,
                    "overlap_genes": overlap,
                    "baseline_fraction": float(baseline_map.get(pid, 0.0)),
                    "weighted_observed_sum_idf_gene": obs_w,
                    "weighted_expected_sum_idf_gene": exp_w,
                    "weighted_fold_enrichment": fold_w,
                }
            )

        pvals = [t["p_value"] for t in tests]
        qvals = bh_qvalues(pvals)
        for t, q in zip(tests, qvals):
            t["q_value"] = q

        sig = [t for t in tests if t["q_value"] < 0.05]
        sig_sorted = sorted(sig, key=lambda d: d["fold_enrichment"], reverse=True)
        top20 = sig_sorted[:20]

        # Rank-change signal from weighting (on significant set).
        sig_by_fold = [x["pathway_id"] for x in sorted(sig, key=lambda d: d["fold_enrichment"], reverse=True)]
        sig_by_wfold = [
            x["pathway_id"] for x in sorted(sig, key=lambda d: d["weighted_fold_enrichment"], reverse=True)
        ]
        top10_unweighted = sig_by_fold[:10]
        top10_weighted = sig_by_wfold[:10]
        changed = len(set(top10_unweighted) ^ set(top10_weighted))

        significant_counts.append(len(sig))
        ranking_change.append(
            {
                "ingredient_id": ing,
                "top10_unweighted": top10_unweighted,
                "top10_weighted": top10_weighted,
                "symmetric_diff_count": changed,
            }
        )

        preview_rows.append(
            {
                "ingredient_id": ing,
                "ingredient_name": None,
                "n_compounds": len(compounds),
                "n_genes": n,
                "n_raw_pathways_reached": len(tests),
                "n_significant_q_lt_0_05": len(sig),
                "n_prev_formula_positive": int(prev_formula_counts.get(ing, -1)),
                "top20_significant_by_fold_enrichment": top20,
                "top_contributing_genes_for_top_pathways": [
                    {
                        "pathway_id": t["pathway_id"],
                        "genes": t["overlap_genes"][:10],
                    }
                    for t in top20[:10]
                ],
            }
        )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_opened": [
            str(IC_PATH),
            str(CG_PATH),
            str(GPM_PATH),
            str(cmp_w_path),
            str(gene_w_path),
            str(baseline_path),
        ],
        "implementation": {
            "background_universe": {
                "choice": "full gpm gene set",
                "n_genes": M,
                "reason": "stable, canonical pathway annotation universe independent of ingredient selection",
            },
            "test": "Hypergeometric right-tail P(X>=k), BH-FDR per ingredient across tested pathways",
            "effect_size": "fold_enrichment = observed_overlap / expected_overlap",
            "weighted_refinement": "weighted_fold_enrichment using gene IDF sums on overlap vs expected IDF sum",
            "note": "IDF-weighted refinement is reported alongside, not replacing p/q significance test",
        },
        "preview_ingredients": preview_rows,
        "comparison_to_prev_formula": {
            "prev_formula_positive_counts": prev_formula_counts,
            "hypergeom_significant_counts": {
                r["ingredient_id"]: r["n_significant_q_lt_0_05"] for r in preview_rows
            },
        },
        "concentration_summary_significant_q_lt_0_05": {
            "values": significant_counts,
            "min": int(min(significant_counts)) if significant_counts else 0,
            "median": float(pd.Series(significant_counts).median()) if significant_counts else 0.0,
            "max": int(max(significant_counts)) if significant_counts else 0,
        },
        "idf_weighting_rank_change_signal": ranking_change,
        "failure_modes": {
            "ingredients_with_zero_significant": [
                r["ingredient_id"] for r in preview_rows if r["n_significant_q_lt_0_05"] == 0
            ],
            "ingredients_with_all_pathways_significant": [
                r["ingredient_id"]
                for r in preview_rows
                if r["n_significant_q_lt_0_05"] == r["n_raw_pathways_reached"]
            ],
        },
    }
    OUT_PREVIEW.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["concentration_summary_significant_q_lt_0_05"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
