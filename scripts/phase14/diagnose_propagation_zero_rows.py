"""
Diagnose why Phase14 propagation is zero for most atlas rows.
Loads same inputs as Phase14; recomputes ingredient_genes and propagated_scores;
for each (ingA_id, ingB_id, category) computes explicit boolean reasons and zero_reason.
Outputs: propagation_zero_row_diagnosis.csv, prints top zero_reason counts and examples.
No new datasets. Repo: global-food-genome (Windows).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

# Reason priority (first match wins)
ZERO_REASONS = [
    "missing_bundle",
    "no_genes_A",
    "no_genes_B",
    "no_pathways_A",
    "no_pathways_B",
    "no_bundle_overlap",
    "other",
]


def _safe_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def main() -> int:
    from src.phase14.loaders import (
        resolve_phase14_inputs,
        resolved_to_paths,
        load_phase13_csvs,
        load_ingredient_compound,
        load_compound_gene,
        load_pathway_bundles,
        load_pathway_cluster_info,
        load_target_functional_clusters,
    )
    from src.phase14.mediation_graph import build_mediation_graph
    from src.phase14.propagation import build_ingredient_genes_from_tables, propagated_scores_for_pairs
    from src.phase14.id_normalization import to_ingredient_id, to_category_id
    from src.phase14 import phase14_config as config

    # Resolve paths (same as Phase14)
    phase13_dir = Path("data/processed/phase13_interactions_v3_20260206_162122_b_gpu_stable")
    for d in REPO_ROOT.glob("data/processed/phase13_*"):
        if (d / "atlas_confirmed.csv").exists():
            phase13_dir = d.relative_to(REPO_ROOT)
            break
    resolved = resolve_phase14_inputs(REPO_ROOT, phase13_dir)
    paths = resolved_to_paths(REPO_ROOT, resolved)

    # Load inputs
    phase13_dir_path = paths.get("phase13_dir") or (REPO_ROOT / phase13_dir)
    if not phase13_dir_path or not phase13_dir_path.exists():
        phase13_dir_path = REPO_ROOT / "data" / "processed" / "phase13_interactions_v3_20260206_162122_b_gpu_stable"
    if not (phase13_dir_path / "atlas_confirmed.csv").exists():
        print("ERROR: atlas_confirmed.csv not found. Check phase13 dir.")
        return 1

    atlas, _, kg_edges, kg_nodes = load_phase13_csvs(phase13_dir_path)
    ingredient_compound = load_ingredient_compound(paths)
    compound_gene = load_compound_gene(paths, chosen=None, full_run_gate=False)
    pathway_bundles = load_pathway_bundles(paths)
    pathway_cluster_info = load_pathway_cluster_info(paths)
    target_clusters = load_target_functional_clusters(paths)

    if pathway_cluster_info.empty:
        pathway_cluster_info = pd.DataFrame(columns=["cluster_id", "top_terms"])
    if target_clusters.empty:
        target_clusters = pd.DataFrame(columns=["cluster_id", "sample_genes", "top_terms"])

    # Build mediation graph and edges
    mediation_nodes, mediation_edges = build_mediation_graph(
        atlas, kg_edges, kg_nodes,
        pathway_cluster_info, target_clusters, pathway_bundles,
        ingredient_compound=ingredient_compound, compound_gene=compound_gene,
    )
    ingredient_genes = build_ingredient_genes_from_tables(ingredient_compound, compound_gene)
    prop_scores, prop_diag = propagated_scores_for_pairs(atlas, mediation_edges, ingredient_genes=ingredient_genes)

    # MAPS_TO_CATEGORY and IN_PATHWAY
    map_edges = mediation_edges[mediation_edges["edge_type"] == "MAPS_TO_CATEGORY"]
    in_pathway = mediation_edges[mediation_edges["edge_type"] == "IN_PATHWAY"]
    path_to_cats: dict = {}
    for _, e in map_edges.iterrows():
        path_to_cats.setdefault(str(e["source_id"]), set()).add(str(e["target_id"]))
    gene_to_pathways: dict = {}
    for _, e in in_pathway.iterrows():
        gene_to_pathways.setdefault(str(e["source_id"]), set()).add(str(e["target_id"]))
    pathway_ids = sorted(in_pathway["target_id"].astype(str).unique().tolist())

    # Bundle keys (categories that have at least one pathway)
    cats_with_pathways = set()
    for paths_list in path_to_cats.values():
        cats_with_pathways.update(paths_list)

    rows = []
    for _, r in atlas.iterrows():
        a = to_ingredient_id(str(r["ingA_id"]))
        b = to_ingredient_id(str(r["ingB_id"]))
        cat_raw = _safe_str(r["category"])
        cat_id = to_category_id(cat_raw) if cat_raw else ""

        ga = ingredient_genes.get(a, set())
        gb = ingredient_genes.get(b, set())
        genes_A_count = len(ga)
        genes_B_count = len(gb)
        has_genes_A = genes_A_count > 0
        has_genes_B = genes_B_count > 0
        has_any_gene_overlap = len(ga & gb) > 0

        pathways_A = set()
        for g in ga:
            pathways_A.update(gene_to_pathways.get(g, set()))
        pathways_B = set()
        for g in gb:
            pathways_B.update(gene_to_pathways.get(g, set()))
        has_any_pathway_edges_for_A = len(pathways_A) > 0
        has_any_pathway_edges_for_B = len(pathways_B) > 0

        # Resolved cat_id: same as propagation (synonym if direct not in cats_with_pathways)
        resolved_cat_id = cat_id
        if cat_id not in cats_with_pathways:
            cat_norm_key = cat_id.replace("CAT_", "", 1) if cat_id.startswith("CAT_") else cat_id
            syn = getattr(config, "CATEGORY_SYNONYMS_FOR_PROPAGATION", {}).get(cat_norm_key)
            if syn:
                resolved_cat_id = to_category_id(syn)
                if resolved_cat_id in cats_with_pathways:
                    pass  # use resolved_cat_id
                else:
                    resolved_cat_id = cat_id
        resolved_category_has_pathways = resolved_cat_id in cats_with_pathways

        bundle_pathways = {p for p, cats in path_to_cats.items() if resolved_cat_id in cats}
        category_bundle_exists = cat_id in cats_with_pathways
        bundle_pathway_id_count = len(bundle_pathways)

        n_A_bundle_paths = len(pathways_A & bundle_pathways)
        n_B_bundle_paths = len(pathways_B & bundle_pathways)
        n_shared_bundle_paths = len((pathways_A & pathways_B) & bundle_pathways)

        score_row = prop_scores[(prop_scores["ingA_id"] == a) & (prop_scores["ingB_id"] == b) & (prop_scores["category"] == cat_raw)]
        final_propagated_score = float(score_row["propagated_pathway_score"].iloc[0]) if len(score_row) else 0.0

        zero_reason = "other"
        if final_propagated_score > 0:
            zero_reason = "nonzero"
        elif not category_bundle_exists or bundle_pathway_id_count == 0:
            zero_reason = "missing_bundle"
        elif not has_genes_A:
            zero_reason = "no_genes_A"
        elif not has_genes_B:
            zero_reason = "no_genes_B"
        elif not has_any_pathway_edges_for_A:
            zero_reason = "no_pathways_A"
        elif not has_any_pathway_edges_for_B:
            zero_reason = "no_pathways_B"
        elif n_shared_bundle_paths == 0 and (n_A_bundle_paths == 0 or n_B_bundle_paths == 0):
            zero_reason = "no_bundle_overlap"

        rows.append({
            "ingA_id": a,
            "ingB_id": b,
            "category": cat_raw,
            "cat_id": cat_id,
            "resolved_category_has_pathways": resolved_category_has_pathways,
            "has_genes_A": has_genes_A,
            "has_genes_B": has_genes_B,
            "genes_A_count": genes_A_count,
            "genes_B_count": genes_B_count,
            "has_any_gene_overlap": has_any_gene_overlap,
            "has_any_pathway_edges_for_A": has_any_pathway_edges_for_A,
            "has_any_pathway_edges_for_B": has_any_pathway_edges_for_B,
            "category_bundle_exists": category_bundle_exists,
            "bundle_pathway_id_count": bundle_pathway_id_count,
            "n_A_bundle_paths": n_A_bundle_paths,
            "n_B_bundle_paths": n_B_bundle_paths,
            "n_shared_bundle_paths": n_shared_bundle_paths,
            "final_propagated_score": final_propagated_score,
            "zero_reason": zero_reason,
        })

    df = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "data" / "processed" / "canonical" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "propagation_zero_row_diagnosis.csv"
    df.to_csv(out_path, index=False)
    print("Wrote", out_path)

    # Top counts of zero_reason
    reason_counts = Counter(df["zero_reason"])
    print("\n--- Top zero_reason counts ---")
    for reason, count in reason_counts.most_common(20):
        print(f"  {reason}: {count}")

    # Top categories with most zeros (where zero_reason != 'nonzero')
    zero_df = df[df["zero_reason"] != "nonzero"]
    if not zero_df.empty:
        cat_zeros = zero_df.groupby("category").size().sort_values(ascending=False)
        print("\n--- Top categories with most zero rows ---")
        for cat, cnt in cat_zeros.head(15).items():
            print(f"  {cat}: {cnt}")

    # Examples (5) for each top zero_reason
    print("\n--- Examples (5) per top zero_reason ---")
    for reason in [r for r, _ in reason_counts.most_common(10) if r != "nonzero"]:
        sub = df[df["zero_reason"] == reason].head(5)
        print(f"\n  [{reason}]")
        for _, ex in sub.iterrows():
            print(f"    ingA={ex['ingA_id']} ingB={ex['ingB_id']} category={ex['category']} cat_id={ex['cat_id']} bundle_exists={ex['category_bundle_exists']} gA={ex['genes_A_count']} gB={ex['genes_B_count']} n_shared_bundle={ex['n_shared_bundle_paths']}")

    # Summary stats
    pct_nonzero = 100.0 * (df["zero_reason"] == "nonzero").sum() / len(df) if len(df) else 0
    pct_bundle_ok = 100.0 * df["category_bundle_exists"].sum() / len(df) if len(df) else 0
    pct_resolved_bundle = 100.0 * df["resolved_category_has_pathways"].sum() / len(df) if "resolved_category_has_pathways" in df.columns and len(df) else 0
    print("\n--- Summary ---")
    print(f"  pct_rows_with_nonzero_propagation: {pct_nonzero:.2f}%")
    print(f"  pct_rows_category_bundle_exists: {pct_bundle_ok:.2f}%")
    print(f"  pct_rows_resolved_category_has_pathways: {pct_resolved_bundle:.2f}%")
    print(f"  n_rows: {len(df)}")

    # Persist summary for gate
    summary_path = out_dir / "propagation_zero_diagnosis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "zero_reason_counts": dict(reason_counts),
            "pct_rows_with_nonzero_propagation": round(pct_nonzero, 2),
            "pct_rows_category_bundle_exists": round(pct_bundle_ok, 2),
            "pct_rows_resolved_category_has_pathways": round(pct_resolved_bundle, 2),
            "n_rows": len(df),
        }, f, indent=2)
    print("Wrote", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
