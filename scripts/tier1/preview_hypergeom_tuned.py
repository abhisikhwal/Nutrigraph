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
OUT_PATH = ROOT / "data/processed/tier1/weights/_hypergeom_tuned_preview.json"

PREVIEW_INGS = ["ING_000951", "ING_000390", "ING_000912"]
Q_THRESHOLDS = [0.05, 0.10, 0.25]
MIN_OVERLAP = 3


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


def run_for_universe(
    universe_name: str,
    universe_genes: set[str],
    ing_to_genes: dict[str, set[str]],
    path_to_genes: dict[str, set[str]],
    path_to_db: dict[str, str],
) -> dict:
    M = len(universe_genes)
    out = {"universe_name": universe_name, "M": M, "ingredients": {}}

    for ing in PREVIEW_INGS:
        gset = ing_to_genes.get(ing, set()) & universe_genes
        n = len(gset)

        tests_all = []
        for pid, pgenes in path_to_genes.items():
            p_u = pgenes & universe_genes
            K = len(p_u)
            if K == 0 or n == 0:
                continue
            overlap = sorted(gset & p_u)
            k = len(overlap)
            if k == 0:
                continue
            expected = n * (K / float(M))
            p = hypergeom_sf(k, M, K, n)
            fold = (k / expected) if expected > 0 else 0.0
            tests_all.append(
                {
                    "pathway_id": pid,
                    "database": path_to_db.get(pid, ""),
                    "k": k,
                    "K": K,
                    "n": n,
                    "M": M,
                    "expected": expected,
                    "fold_enrichment": fold,
                    "p_value": p,
                    "genes": overlap,
                }
            )

        # before overlap filter
        pvals_all = [t["p_value"] for t in tests_all]
        qvals_all = bh_qvalues(pvals_all)
        for t, q in zip(tests_all, qvals_all):
            t["q_value"] = q

        # after overlap filter k >= MIN_OVERLAP
        tests_filtered = [t for t in tests_all if t["k"] >= MIN_OVERLAP]
        pvals_f = [t["p_value"] for t in tests_filtered]
        qvals_f = bh_qvalues(pvals_f)
        for t, q in zip(tests_filtered, qvals_f):
            t["q_value"] = q

        curve = {}
        for qthr in Q_THRESHOLDS:
            sig = [t for t in tests_filtered if t["q_value"] < qthr]
            curve[f"q_lt_{qthr:.2f}"] = len(sig)
            curve[f"q_lt_{qthr:.2f}_and_fold_ge_2"] = sum(
                1 for t in sig if t["fold_enrichment"] >= 2.0
            )

        out["ingredients"][ing] = {
            "n_genes": n,
            "n_tested_before_filter_k_ge_1": len(tests_all),
            "n_tested_after_filter_k_ge_3": len(tests_filtered),
            "breadth_curve_counts": curve,
            "tests_filtered": tests_filtered,  # keep for later selection
        }
    return out


def choose_operating_threshold(frame: dict) -> dict:
    # Candidate thresholds: q only and q+fold>=2
    candidates = []
    for qthr in Q_THRESHOLDS:
        candidates.append({"q": qthr, "require_fold_ge_2": False})
        candidates.append({"q": qthr, "require_fold_ge_2": True})

    best = None
    for c in candidates:
        counts = []
        for ing in PREVIEW_INGS:
            tests = frame["ingredients"][ing]["tests_filtered"]
            sig = [t for t in tests if t["q_value"] < c["q"]]
            if c["require_fold_ge_2"]:
                sig = [t for t in sig if t["fold_enrichment"] >= 2.0]
            counts.append(len(sig))
        med = float(pd.Series(counts).median())
        in_range = all(10 <= x <= 100 for x in counts)
        # score: prefer all-in-range, then median close to 55
        score = (0 if in_range else 1, abs(med - 55.0))
        cand = {**c, "counts": counts, "median": med, "all_in_target_range": in_range, "score": score}
        if best is None or cand["score"] < best["score"]:
            best = cand
    return best


def main() -> int:
    ic = pd.read_csv(IC_PATH)
    cg = pd.read_csv(CG_PATH)
    gpm = pd.read_parquet(GPM_PATH)

    ic["ingredient_id"] = ic["ingredient_id"].astype(str).str.strip()
    ic["compound_id"] = ic["compound_id"].astype(str).str.strip().str.upper()
    cg["compound_id"] = cg["compound_id"].astype(str).str.strip().str.upper()
    cg["gene_symbol"] = cg["gene_symbol"].astype(str).str.strip()
    gpm["gene_symbol"] = gpm["gene_symbol"].astype(str).str.strip()
    gpm["pathway_id"] = gpm["pathway_id"].astype(str).str.strip()
    gpm["database"] = gpm["database"].astype(str).str.strip()

    # maps
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
    ing_to_genes = {}
    for ing, cmps in ing_to_compounds.items():
        gs = set()
        for c in cmps:
            gs |= cmp_to_genes.get(c, set())
        ing_to_genes[ing] = gs

    path_to_genes = {}
    path_to_db = {}
    for _, row in gpm[["pathway_id", "gene_symbol", "database"]].drop_duplicates().iterrows():
        path_to_genes.setdefault(row["pathway_id"], set()).add(row["gene_symbol"])
        path_to_db[row["pathway_id"]] = row["database"]

    universe_full = set(gpm["gene_symbol"].dropna().unique())
    universe_reachable = set().union(*[ing_to_genes.get(i, set()) for i in sorted(ing_to_genes)])
    # keep only genes that are in gpm universe
    universe_reachable &= universe_full

    frame_full = run_for_universe("full_gpm_genes", universe_full, ing_to_genes, path_to_genes, path_to_db)
    frame_reachable = run_for_universe(
        "reachable_genes_across_ingredients", universe_reachable, ing_to_genes, path_to_genes, path_to_db
    )

    # Preferred tuned frame for decision: reachable universe + k>=3 filter.
    operating = choose_operating_threshold(frame_reachable)

    # Build top25 at chosen threshold.
    top25 = {}
    for ing in PREVIEW_INGS:
        tests = frame_reachable["ingredients"][ing]["tests_filtered"]
        sig = [t for t in tests if t["q_value"] < operating["q"]]
        if operating["require_fold_ge_2"]:
            sig = [t for t in sig if t["fold_enrichment"] >= 2.0]
        sig_sorted = sorted(sig, key=lambda d: d["fold_enrichment"], reverse=True)[:25]
        top25[ing] = sig_sorted

    # remove heavy embedded tests from report payload (keep counts + top25 only)
    for fr in (frame_full, frame_reachable):
        for ing in PREVIEW_INGS:
            fr["ingredients"][ing].pop("tests_filtered", None)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_opened": [str(IC_PATH), str(CG_PATH), str(GPM_PATH)],
        "implementation": {
            "core_method": "Hypergeometric right-tail ORA + BH-FDR per ingredient",
            "minimum_overlap_filter": f"k >= {MIN_OVERLAP}",
            "universes_compared": {
                "full_gpm_genes_M": len(universe_full),
                "reachable_genes_M": len(universe_reachable),
            },
        },
        "before_after_test_counts": {
            "full_universe": frame_full,
            "reachable_universe": frame_reachable,
        },
        "operating_threshold_choice": operating,
        "top25_at_operating_threshold": top25,
        "failure_mode_check": {
            "counts_at_operating_threshold": {
                PREVIEW_INGS[i]: operating["counts"][i] for i in range(len(PREVIEW_INGS))
            },
            "all_zero": all(c == 0 for c in operating["counts"]),
            "all_saturated_gt_100": all(c > 100 for c in operating["counts"]),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["operating_threshold_choice"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
