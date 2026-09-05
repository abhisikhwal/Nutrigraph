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
TUNED_PREVIEW_PATH = WEIGHTS_DIR / "_hypergeom_tuned_preview.json"

OUT_DIR = ROOT / "data/processed/tier1"
OUT_PARQUET = OUT_DIR / "ingredient_enrichment_profiles_v1.parquet"
OUT_JSONL = OUT_DIR / "ingredient_enrichment_profiles_v1.jsonl"
OUT_REPORT = OUT_DIR / "tier1_enrichment_build_report.json"

MIN_OVERLAP = 3
Q_THRESHOLDS = [0.05, 0.10, 0.25]
PREVIEW_INGS = ["ING_000951", "ING_000390", "ING_000912"]


def latest_weight_csv(prefix: str) -> Path:
    files = sorted(WEIGHTS_DIR.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No weight table found for prefix: {prefix}")
    return files[-1]


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


def main() -> int:
    ic = pd.read_csv(IC_PATH)
    cg = pd.read_csv(CG_PATH)
    gpm = pd.read_parquet(GPM_PATH)

    cmp_w_path = latest_weight_csv("compound_idf_weights")
    gene_w_path = latest_weight_csv("gene_idf_weights")
    baseline_path = latest_weight_csv("pathway_baseline_frozen")
    _ = pd.read_csv(cmp_w_path)  # read-only per requirement
    gene_w = pd.read_csv(gene_w_path)
    _ = pd.read_csv(baseline_path)  # read-only per requirement
    gene_idf = dict(zip(gene_w["gene_symbol"], gene_w["idf_gene"]))

    ic["ingredient_id"] = ic["ingredient_id"].astype(str).str.strip()
    ic["compound_id"] = ic["compound_id"].astype(str).str.strip().str.upper()
    cg["compound_id"] = cg["compound_id"].astype(str).str.strip().str.upper()
    cg["gene_symbol"] = cg["gene_symbol"].astype(str).str.strip()
    cg["raw_gene_name"] = cg["raw_gene_name"].astype(str).str.strip()
    cg["source"] = cg["source"].astype(str).str.strip()
    gpm["gene_symbol"] = gpm["gene_symbol"].astype(str).str.strip()
    gpm["pathway_id"] = gpm["pathway_id"].astype(str).str.strip()
    gpm["database"] = gpm["database"].astype(str).str.strip()

    # Maps
    ing_to_compounds = (
        ic.groupby("ingredient_id")["compound_id"].apply(lambda s: sorted(set(s.dropna().astype(str)))).to_dict()
    )

    # Keep per-row provenance for compound->gene
    cmp_gene_rows = {}
    for _, r in cg.iterrows():
        cmp_gene_rows.setdefault(r["compound_id"], []).append(
            {
                "gene_symbol": r["gene_symbol"],
                "raw_gene_name": r["raw_gene_name"],
                "source": r["source"],
            }
        )

    path_to_genes = {}
    path_to_db = {}
    for _, row in gpm[["pathway_id", "gene_symbol", "database"]].drop_duplicates().iterrows():
        path_to_genes.setdefault(row["pathway_id"], set()).add(row["gene_symbol"])
        path_to_db[row["pathway_id"]] = row["database"]

    universe_full = set(gpm["gene_symbol"].dropna().unique())

    # Build ingredient->gene provenance map and identify compounds-but-no-gene ingredients.
    ing_gene_prov = {}
    compounds_but_no_gene = []
    for ing, compounds in ing_to_compounds.items():
        gene_map = {}
        for c in compounds:
            for rec in cmp_gene_rows.get(c, []):
                g = rec["gene_symbol"]
                entry = gene_map.setdefault(
                    g,
                    {"raw_gene_names": set(), "sources": set(), "compound_ids": set()},
                )
                entry["raw_gene_names"].add(rec["raw_gene_name"])
                entry["sources"].add(rec["source"])
                entry["compound_ids"].add(c)
        if not gene_map:
            compounds_but_no_gene.append(ing)
        ing_gene_prov[ing] = gene_map

    # Reachable universe per approved tuned method.
    reachable = set()
    for gene_map in ing_gene_prov.values():
        reachable.update(gene_map.keys())
    reachable &= universe_full
    M = len(reachable)  # expected 366

    # Build profiles only for ingredients that reach >=1 gene (222 expected).
    profiles = []
    rows = []
    for ing in sorted(ing_to_compounds.keys()):
        gene_map = ing_gene_prov.get(ing, {})
        if not gene_map:
            continue
        ing_genes = set(gene_map.keys()) & reachable
        n = len(ing_genes)

        # Genes with no pathway in canonical gpm (honest termini).
        no_pathway_genes = sorted(g for g in gene_map.keys() if g not in universe_full)
        no_pathway_gene_branches = []
        for g in no_pathway_genes:
            gp = gene_map[g]
            no_pathway_gene_branches.append(
                {
                    "gene_symbol": g,
                    "raw_gene_name": sorted(gp["raw_gene_names"]),
                    "source": sorted(gp["sources"]),
                    "compound_ids": sorted(gp["compound_ids"]),
                    "no_annotated_pathway": True,
                }
            )

        tests = []
        for pid, pgenes_all in path_to_genes.items():
            pgenes = pgenes_all & reachable
            K = len(pgenes)
            if K == 0 or n == 0:
                continue
            overlap = sorted(ing_genes & pgenes)
            k = len(overlap)
            if k < MIN_OVERLAP:
                continue
            expected = n * (K / float(M))
            pval = hypergeom_sf(k, M, K, n)
            fold = (k / expected) if expected > 0 else 0.0

            obs_w = sum(gene_idf.get(g, 0.0) for g in overlap)
            path_w_sum = sum(gene_idf.get(g, 0.0) for g in pgenes)
            exp_w = (n / float(M)) * path_w_sum
            weighted_fold = (obs_w / exp_w) if exp_w > 0 else 0.0

            # Contributing genes with provenance.
            contrib_genes = []
            for g in overlap:
                gp = gene_map[g]
                contrib_genes.append(
                    {
                        "gene_symbol": g,
                        "raw_gene_name": sorted(gp["raw_gene_names"]),
                        "source": sorted(gp["sources"]),
                        "compound_ids": sorted(gp["compound_ids"]),
                    }
                )

            tests.append(
                {
                    "pathway_id": pid,
                    "database": path_to_db.get(pid, ""),
                    "overlap_k": k,
                    "expected_overlap": expected,
                    "fold_enrichment": fold,
                    "p_value": pval,
                    "weighted_fold_idf_gene": weighted_fold,
                    "contributing_genes": contrib_genes,
                }
            )

        qvals = bh_qvalues([t["p_value"] for t in tests])
        for t, q in zip(tests, qvals):
            t["q_value"] = q
        tests_sorted = sorted(tests, key=lambda d: (d["q_value"], -d["fold_enrichment"], d["pathway_id"]))

        profile = {
            "ingredient_id": ing,
            "ingredient_name": None,
            "n_compounds": len(ing_to_compounds[ing]),
            "n_genes": len(gene_map),
            "n_genes_with_pathway_universe": n,
            "n_genes_no_pathway": len(no_pathway_genes),
            "no_pathway_gene_branches": no_pathway_gene_branches,
            "pathway_tests": tests_sorted,  # full ranked result (k>=3), no q cutoff
            "method": {
                "universe": "reachable_genes_across_ingredients",
                "M": M,
                "min_overlap_k": MIN_OVERLAP,
                "q_correction": "BH per ingredient",
            },
        }
        profiles.append(profile)
        rows.append(
            {
                "ingredient_id": ing,
                "ingredient_name": None,
                "n_compounds": len(ing_to_compounds[ing]),
                "n_genes": len(gene_map),
                "n_genes_with_pathway_universe": n,
                "n_genes_no_pathway": len(no_pathway_genes),
                "n_pathway_tests_k_ge_3": len(tests_sorted),
                "pathway_tests_json": json.dumps(tests_sorted, ensure_ascii=True),
                "no_pathway_gene_branches_json": json.dumps(no_pathway_gene_branches, ensure_ascii=True),
            }
        )

    # Funnel checks
    n_with_compound = len(ing_to_compounds)
    n_with_gene = len([ing for ing, gm in ing_gene_prov.items() if len(gm) > 0])
    n_profiles = len(profiles)
    if not (n_with_compound == 223 and n_with_gene == 222 and n_profiles == 222):
        raise RuntimeError(
            f"Funnel mismatch: compounds={n_with_compound}, with_gene={n_with_gene}, profiles={n_profiles}"
        )
    if sorted(compounds_but_no_gene) != ["ING_000413"]:
        raise RuntimeError(f"Unexpected compounds-but-no-gene ingredients: {sorted(compounds_but_no_gene)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("ingredient_id").to_parquet(OUT_PARQUET, index=False)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for p in sorted(profiles, key=lambda x: x["ingredient_id"]):
            fh.write(json.dumps(p, ensure_ascii=True) + "\n")

    # Breadth distributions at q thresholds
    breadth = {}
    for q in Q_THRESHOLDS:
        counts = []
        for p in profiles:
            counts.append(sum(1 for t in p["pathway_tests"] if t["q_value"] < q))
        s = pd.Series(counts)
        breadth[f"q_lt_{q:.2f}"] = {
            "min": int(s.min()),
            "median": float(s.median()),
            "max": int(s.max()),
            "values": counts,
        }

    zero_q25 = [
        p["ingredient_id"] for p in profiles if sum(1 for t in p["pathway_tests"] if t["q_value"] < 0.25) == 0
    ]

    # Preview consistency check against tuned preview signatures.
    preview_sig = {}
    if TUNED_PREVIEW_PATH.exists():
        tuned = json.loads(TUNED_PREVIEW_PATH.read_text(encoding="utf-8"))
        for ing in PREVIEW_INGS:
            prev_list = tuned.get("top25_at_operating_threshold", {}).get(ing, [])
            preview_sig[ing] = [x["pathway_id"] for x in prev_list]

    scaled_preview_sets = {}
    for ing in PREVIEW_INGS:
        p = next((x for x in profiles if x["ingredient_id"] == ing), None)
        if p is None:
            scaled_preview_sets[ing] = []
        else:
            scaled_preview_sets[ing] = [t["pathway_id"] for t in p["pathway_tests"] if t["q_value"] < 0.25]

    preview_consistency = {}
    for ing in PREVIEW_INGS:
        prev_set = set(preview_sig.get(ing, []))
        scaled_set = set(scaled_preview_sets.get(ing, []))
        if prev_set:
            j = len(prev_set & scaled_set) / float(len(prev_set | scaled_set)) if (prev_set | scaled_set) else 1.0
        else:
            j = 1.0 if not scaled_set else 0.0
        preview_consistency[ing] = {
            "preview_top25_count": len(prev_set),
            "scaled_q_lt_0.25_count": len(scaled_set),
            "intersection_count": len(prev_set & scaled_set),
            "jaccard": j,
            "contains_carbonic_signature": any(
                pid in scaled_set for pid in ["R-HSA-1475029", "R-HSA-1237044", "R-HSA-1247673"]
            ),
            "contains_transporter_signature": any(
                pid in scaled_set
                for pid in ["organic anion transport [GO:0015711]", "sodium-independent organic anion transport [GO:0043252]"]
            ),
        }

    # Three NEW example profiles (not preview), full pathway_tests included.
    non_preview = [p for p in profiles if p["ingredient_id"] not in PREVIEW_INGS]
    # pick distinct by q<0.25 counts: low, median-ish, high
    non_preview_sorted = sorted(
        non_preview,
        key=lambda p: sum(1 for t in p["pathway_tests"] if t["q_value"] < 0.25),
    )
    ex_low = non_preview_sorted[0]
    ex_med = non_preview_sorted[len(non_preview_sorted) // 2]
    ex_high = non_preview_sorted[-1]
    examples = [ex_low, ex_med, ex_high]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "opened_input_files": [
            str(IC_PATH),
            str(CG_PATH),
            str(GPM_PATH),
            str(cmp_w_path),
            str(gene_w_path),
            str(baseline_path),
        ],
        "forbidden_inputs_opened": [],
        "method": {
            "universe": "reachable_genes_across_ingredients",
            "M": M,
            "min_overlap_k": MIN_OVERLAP,
            "test": "hypergeometric right-tail",
            "q_correction": "BH per ingredient",
            "store_full_ranked_result": True,
            "no_top_n_padding": True,
        },
        "funnel": {
            "ingredients_with_compound": n_with_compound,
            "ingredients_reach_gene": n_with_gene,
            "profiles_built": n_profiles,
            "compounds_but_no_gene_ingredients": sorted(compounds_but_no_gene),
        },
        "outputs": {"parquet": str(OUT_PARQUET), "jsonl": str(OUT_JSONL)},
        "breadth_distribution": breadth,
        "zero_enriched_at_q_lt_0.25": {
            "count": len(zero_q25),
            "ingredients": zero_q25,
        },
        "preview_consistency": preview_consistency,
        "new_example_profiles_full": examples,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "profiles_built": n_profiles,
                "M": M,
                "breadth_distribution": {
                    k: {
                        "min": v["min"],
                        "median": v["median"],
                        "max": v["max"],
                    }
                    for k, v in breadth.items()
                },
                "zero_enriched_q0.25_count": len(zero_q25),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
