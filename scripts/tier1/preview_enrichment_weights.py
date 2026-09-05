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
TIER1_PATH = ROOT / "data/processed/tier1/ingredient_mechanism_profiles.parquet"

OUT_DIR = ROOT / "data/processed/tier1/weights"
PREVIEW_JSON = OUT_DIR / "_enrichment_preview.json"
REPORT_JSON = OUT_DIR / "_enrichment_preview_report.json"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _dist(vals: pd.Series) -> dict:
    return {
        "min": float(vals.min()),
        "p25": float(vals.quantile(0.25)),
        "median": float(vals.median()),
        "p75": float(vals.quantile(0.75)),
        "p90": float(vals.quantile(0.9)),
        "p95": float(vals.quantile(0.95)),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _ts()

    ic = pd.read_csv(IC_PATH)
    cg = pd.read_csv(CG_PATH)
    gpm = pd.read_parquet(GPM_PATH)
    tier1 = pd.read_parquet(TIER1_PATH)

    ic["ingredient_id"] = ic["ingredient_id"].astype(str).str.strip()
    ic["compound_id"] = ic["compound_id"].astype(str).str.strip().str.upper()
    cg["compound_id"] = cg["compound_id"].astype(str).str.strip().str.upper()
    cg["gene_symbol"] = cg["gene_symbol"].astype(str).str.strip()
    gpm["gene_symbol"] = gpm["gene_symbol"].astype(str).str.strip()
    gpm["pathway_id"] = gpm["pathway_id"].astype(str).str.strip()
    gpm["database"] = gpm["database"].astype(str).str.strip()

    # Step 1a: compound IDF
    n_ingredients = ic["ingredient_id"].nunique()  # expected 223
    cmp_ing = (
        ic.groupby("compound_id")["ingredient_id"]
        .nunique()
        .rename("n_ingredients_containing")
        .reset_index()
    )
    cmp_ing["idf_compound"] = cmp_ing["n_ingredients_containing"].apply(
        lambda n: math.log(n_ingredients / float(n))
    )
    cmp_weights_path = OUT_DIR / f"compound_idf_weights_{stamp}.csv"
    cmp_ing.sort_values(["idf_compound", "compound_id"], ascending=[False, True]).to_csv(
        cmp_weights_path, index=False
    )

    # Step 1b: gene IDF
    n_pathways_total = gpm["pathway_id"].nunique()  # expected 4543
    gene_pw = (
        gpm.groupby("gene_symbol")["pathway_id"]
        .nunique()
        .rename("n_pathways_for_gene")
        .reset_index()
    )
    gene_pw["idf_gene"] = gene_pw["n_pathways_for_gene"].apply(
        lambda n: math.log(n_pathways_total / float(n))
    )
    gene_weights_path = OUT_DIR / f"gene_idf_weights_{stamp}.csv"
    gene_pw.sort_values(["idf_gene", "gene_symbol"], ascending=[False, True]).to_csv(
        gene_weights_path, index=False
    )

    # Step 1c: frozen baseline from existing 222 profiles
    # Avoid DataFrame explode on huge JSON strings; use streaming counts.
    n_profiles = len(tier1)
    path_counts: dict[str, int] = {}
    for s in tier1["pathways_json"].tolist():
        pset = {p["pathway_id"] for p in json.loads(s)}
        for pid in pset:
            path_counts[pid] = path_counts.get(pid, 0) + 1
    path_ing = pd.DataFrame(
        {
            "pathway_id": list(path_counts.keys()),
            "n_ingredients_reaching_pathway": list(path_counts.values()),
        }
    )
    path_ing["baseline_fraction"] = path_ing["n_ingredients_reaching_pathway"] / float(n_profiles)
    path_ing["baseline_version_utc"] = datetime.now(timezone.utc).isoformat()
    path_baseline_path = OUT_DIR / f"pathway_baseline_frozen_{stamp}.csv"
    path_ing.sort_values(["baseline_fraction", "pathway_id"], ascending=[False, True]).to_csv(
        path_baseline_path, index=False
    )

    # maps for scoring
    cmp_idf = dict(zip(cmp_ing["compound_id"], cmp_ing["idf_compound"]))
    gene_idf = dict(zip(gene_pw["gene_symbol"], gene_pw["idf_gene"]))
    baseline = dict(zip(path_ing["pathway_id"], path_ing["baseline_fraction"]))

    ing_to_compounds = (
        ic.groupby("ingredient_id")["compound_id"]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )
    cmp_to_genes = (
        cg.groupby("compound_id")["gene_symbol"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .to_dict()
    )
    gene_to_paths = {}
    for _, row in gpm[["gene_symbol", "pathway_id", "database"]].drop_duplicates().iterrows():
        gene_to_paths.setdefault(row["gene_symbol"], []).append(
            {"pathway_id": row["pathway_id"], "database": row["database"]}
        )

    # pick 3 distinct ingredients by compound set distance
    all_ings = sorted(ing_to_compounds.keys())
    first = max(all_ings, key=lambda i: len(ing_to_compounds.get(i, set())))

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / float(len(a | b))

    second = max(
        [i for i in all_ings if i != first],
        key=lambda i: 1.0 - jaccard(ing_to_compounds[first], ing_to_compounds[i]),
    )
    third = max(
        [i for i in all_ings if i not in {first, second}],
        key=lambda i: min(
            1.0 - jaccard(ing_to_compounds[first], ing_to_compounds[i]),
            1.0 - jaccard(ing_to_compounds[second], ing_to_compounds[i]),
        ),
    )
    preview_ingredients = [first, second, third]

    saturated_order = (
        path_ing.sort_values("n_ingredients_reaching_pathway", ascending=False)["pathway_id"].tolist()
    )
    saturated_rank = {p: idx + 1 for idx, p in enumerate(saturated_order)}

    EPS = 1e-12
    previews = []
    concentration = []
    for ing in preview_ingredients:
        compounds = sorted(ing_to_compounds.get(ing, set()))
        pathway_scores: dict[str, float] = {}
        pathway_db: dict[str, str] = {}
        pathway_gene_contrib: dict[str, dict[str, float]] = {}
        pathway_cmp_contrib: dict[str, dict[str, float]] = {}

        for cmp_id in compounds:
            c_w = cmp_idf.get(cmp_id, 0.0)
            for gene in cmp_to_genes.get(cmp_id, []):
                g_w = gene_idf.get(gene, 0.0)
                contrib = c_w * g_w
                for p in gene_to_paths.get(gene, []):
                    pid = p["pathway_id"]
                    pathway_db[pid] = p["database"]
                    pathway_scores[pid] = pathway_scores.get(pid, 0.0) + contrib
                    pathway_gene_contrib.setdefault(pid, {})
                    pathway_gene_contrib[pid][gene] = pathway_gene_contrib[pid].get(gene, 0.0) + contrib
                    pathway_cmp_contrib.setdefault(pid, {})
                    pathway_cmp_contrib[pid][cmp_id] = pathway_cmp_contrib[pid].get(cmp_id, 0.0) + contrib

        total_raw = sum(pathway_scores.values())
        scored = []
        for pid, raw in pathway_scores.items():
            frac = raw / total_raw if total_raw > 0 else 0.0
            b = baseline.get(pid, 0.0)
            enr = math.log((frac + EPS) / (b + EPS))
            scored.append(
                {
                    "pathway_id": pid,
                    "database": pathway_db.get(pid, ""),
                    "raw_strength": raw,
                    "normalized_strength": frac,
                    "baseline_fraction": b,
                    "enrichment_score": enr,
                    "top_genes": [
                        {"gene_symbol": g, "contribution": v}
                        for g, v in sorted(
                            pathway_gene_contrib.get(pid, {}).items(), key=lambda kv: kv[1], reverse=True
                        )[:5]
                    ],
                    "top_compounds": [
                        {"compound_id": c, "contribution": v}
                        for c, v in sorted(
                            pathway_cmp_contrib.get(pid, {}).items(), key=lambda kv: kv[1], reverse=True
                        )[:5]
                    ],
                    "saturated_rank_proxy": saturated_rank.get(pid, 999999),
                }
            )
        scored.sort(key=lambda d: d["enrichment_score"], reverse=True)
        top20 = scored[:20]
        dropped = sorted(
            [x for x in scored if x["baseline_fraction"] >= 0.9],
            key=lambda d: d["enrichment_score"],
        )[:5]

        n_meaningful = sum(1 for x in scored if x["enrichment_score"] > 0)
        concentration.append(n_meaningful)
        previews.append(
            {
                "ingredient_id": ing,
                "ingredient_name": None,
                "n_compounds": len(compounds),
                "n_pathways_raw": len(scored),
                "n_pathways_meaningful_enrichment_score_gt_0": n_meaningful,
                "top20_pathways_by_enrichment": top20,
                "five_high_baseline_pathways_that_drop_under_enrichment": dropped,
            }
        )

    preview_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_opened": [str(IC_PATH), str(CG_PATH), str(GPM_PATH), str(TIER1_PATH)],
        "formulas": {
            "idf_compound": "log(N_ingredients / n_ingredients_containing_compound), N_ingredients=223",
            "idf_gene": "log(N_pathways / n_pathways_for_gene), N_pathways=4543",
            "raw_connection_strength": "sum over all ingredient compound->gene->pathway paths of (idf_compound * idf_gene)",
            "normalized_strength": "raw_connection_strength(pathway) / sum_raw_connection_strength(all pathways for ingredient)",
            "enrichment_score": "log((normalized_strength + 1e-12) / (baseline_fraction + 1e-12))",
        },
        "preview_ingredients": previews,
    }
    PREVIEW_JSON.write_text(json.dumps(preview_payload, indent=2), encoding="utf-8")

    top10_promiscuous = (
        gene_pw.sort_values("n_pathways_for_gene", ascending=False).head(10)[
            ["gene_symbol", "n_pathways_for_gene", "idf_gene"]
        ]
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "opened_input_files": [str(IC_PATH), str(CG_PATH), str(GPM_PATH), str(TIER1_PATH)],
        "outputs": {
            "compound_weights_csv": str(cmp_weights_path),
            "gene_weights_csv": str(gene_weights_path),
            "pathway_baseline_csv": str(path_baseline_path),
            "preview_json": str(PREVIEW_JSON),
        },
        "weight_distributions": {
            "idf_compound": _dist(cmp_ing["idf_compound"]),
            "idf_gene": _dist(gene_pw["idf_gene"]),
            "baseline_fraction": _dist(path_ing["baseline_fraction"]),
        },
        "checks": {
            "n_ingredients": int(n_ingredients),
            "ubiquitous_compounds_ge_90pct": int((cmp_ing["n_ingredients_containing"] >= 0.9 * n_ingredients).sum()),
            "ubiquitous_compounds_idf_min_median_max": {
                "min": float(cmp_ing.loc[cmp_ing["n_ingredients_containing"] >= 0.9 * n_ingredients, "idf_compound"].min()),
                "median": float(cmp_ing.loc[cmp_ing["n_ingredients_containing"] >= 0.9 * n_ingredients, "idf_compound"].median()),
                "max": float(cmp_ing.loc[cmp_ing["n_ingredients_containing"] >= 0.9 * n_ingredients, "idf_compound"].max()),
            },
            "top10_promiscuous_genes_with_idf": top10_promiscuous.to_dict(orient="records"),
            "baseline_pathways_ge_90pct": int((path_ing["baseline_fraction"] >= 0.9).sum()),
            "baseline_pathways_total": int(len(path_ing)),
        },
        "preview_selection": {
            "ingredient_ids": preview_ingredients,
            "pairwise_jaccard": {
                f"{preview_ingredients[0]}__{preview_ingredients[1]}": jaccard(
                    ing_to_compounds[preview_ingredients[0]], ing_to_compounds[preview_ingredients[1]]
                ),
                f"{preview_ingredients[0]}__{preview_ingredients[2]}": jaccard(
                    ing_to_compounds[preview_ingredients[0]], ing_to_compounds[preview_ingredients[2]]
                ),
                f"{preview_ingredients[1]}__{preview_ingredients[2]}": jaccard(
                    ing_to_compounds[preview_ingredients[1]], ing_to_compounds[preview_ingredients[2]]
                ),
            },
        },
        "preview_concentration_n_pathways_enrichment_gt_0": {
            "values": concentration,
            "min": int(min(concentration)),
            "median": float(pd.Series(concentration).median()),
            "max": int(max(concentration)),
        },
        "judgment_calls": {
            "gene_weight_formula": "Used log(N_pathways / n_pathways_for_gene) for symmetry with compound IDF and interpretability.",
            "enrichment_formula": "Used log-ratio of ingredient-normalized weighted strength vs frozen baseline fraction with epsilon 1e-12.",
            "meaningful_threshold": "enrichment_score > 0",
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
