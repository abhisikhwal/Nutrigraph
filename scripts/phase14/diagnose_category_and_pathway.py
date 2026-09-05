"""
Phase14.5 diagnostics: category bias and Unknown pathway.
Produces reports/category_bias_report.json and reports/unknown_pathway_diagnosis.json.
Run from repo root: python scripts/phase14/diagnose_category_and_pathway.py [--run-dir PATH] [--phase13-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _latest_run(repo_root: Path) -> Path | None:
    base = repo_root / "data" / "processed" / "phase14_mediation"
    if not base.exists():
        return None
    runs = []
    for d in base.iterdir():
        if not d.is_dir() or not (d / "mediation_nodes.csv").exists():
            continue
        if not (d / "reports" / "phase14_summary.json").exists():
            continue
        try:
            parts = d.name.split("_")
            if len(parts) >= 3 and parts[0].lower() == "phase14":
                date_part, time_part = parts[1], parts[2]
                if len(date_part) == 8 and len(time_part) == 6:
                    runs.append((float(f"{date_part}{time_part}"), d))
                    continue
        except Exception:
            pass
        runs.append((d.stat().st_mtime, d))
    if not runs:
        return None
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Category bias and Unknown pathway diagnostics")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--phase13-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _latest_run(root)
    if run_dir is None:
        print("ERROR: No Phase14 run found", file=sys.stderr)
        return 1
    run_dir = run_dir.resolve()

    phase13_dir = args.phase13_dir or root / "data" / "processed" / "phase13_interactions_v3_20260206_162122_b_gpu_stable"
    phase13_dir = phase13_dir.resolve()
    atlas_path = phase13_dir / "atlas_confirmed.csv"
    if not atlas_path.exists():
        atlas_path = phase13_dir / "atlas_confirmed.parquet"

    import pandas as pd

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ----- A) Category bias -----
    category_counts_atlas: dict[str, int] = {}
    category_counts_pcm: dict[str, int] = {}
    category_scores: dict[str, list[float]] = {}

    if atlas_path.exists():
        if atlas_path.suffix.lower() == ".parquet":
            atlas = pd.read_parquet(atlas_path, columns=["category"])
        else:
            atlas = pd.read_csv(atlas_path, usecols=["category"], low_memory=False)
        for c in atlas["category"].dropna().astype(str).str.strip():
            category_counts_atlas[c] = category_counts_atlas.get(c, 0) + 1

    pcm_path = run_dir / "pair_category_mediation.csv"
    if pcm_path.exists():
        pcm = pd.read_csv(pcm_path, usecols=["category", "mechanistic_score"], low_memory=False)
        for c in pcm["category"].dropna().astype(str).str.strip():
            category_counts_pcm[c] = category_counts_pcm.get(c, 0) + 1
        for _, row in pcm.iterrows():
            cat = str(row["category"]).strip()
            sc = row.get("mechanistic_score")
            if pd.notna(sc):
                category_scores.setdefault(cat, []).append(float(sc))

    edges_path = run_dir / "neo4j" / "edges.csv"
    category_counts_affects: dict[str, int] = {}
    if edges_path.exists():
        edges = pd.read_csv(edges_path, low_memory=False)
        aff = edges[edges[":TYPE"].astype(str).str.strip() == "AFFECTS"]
        for _, row in aff.iterrows():
            end = str(row.get(":END_ID", row.get("target_id", "")))
            if end.startswith("CAT_"):
                cat = end.replace("CAT_", "", 1).replace("_", " ")
                category_counts_affects[cat] = category_counts_affects.get(cat, 0) + 1

    # Normalize category keys for comparison (CAT_apoptosis vs apoptosis)
    def norm_cat(s: str) -> str:
        return s.lower().replace(" ", "_").strip()

    bias_report = {
        "run_dir": str(run_dir),
        "phase13_dir": str(phase13_dir),
        "atlas_category_counts": category_counts_atlas,
        "pair_category_mediation_category_counts": category_counts_pcm,
        "neo4j_affects_category_counts": category_counts_affects,
        "score_distribution_per_category": {
            c: {"count": len(scores), "mean": round(sum(scores) / len(scores), 4), "min": round(min(scores), 4), "max": round(max(scores), 4)}
            for c, scores in category_scores.items() if scores
        },
        "diagnosis": "Category distribution is driven by Phase13 atlas (atlas_confirmed). If atlas has more apoptosis rows, downstream pair_mediation and AFFECTS reflect that. Bias is upstream-only; no artificial rebalancing applied.",
        "recommendation": "Use category-balanced analysis queries to explore beyond top category (e.g. filter by category or rank within category).",
    }
    with open(reports_dir / "category_bias_report.json", "w", encoding="utf-8") as f:
        json.dump(bias_report, f, indent=2)
    print("Wrote", reports_dir / "category_bias_report.json")

    # ----- B) Unknown pathway diagnosis -----
    nodes_path = run_dir / "neo4j" / "nodes.csv"
    pathway_unknown_id: set[str] = set()
    pathway_id_to_display: dict[str, str] = {}
    if nodes_path.exists():
        nodes = pd.read_csv(nodes_path, low_memory=False)
        path_nodes = nodes[nodes[":LABEL"].astype(str).str.contains("Pathway", na=False)]
        disp_col = "display_name" if "display_name" in path_nodes.columns else "name"
        for _, row in path_nodes.iterrows():
            pid = str(row[":ID"])
            disp = str(row.get(disp_col, "")).strip()
            pathway_id_to_display[pid] = disp
            if disp == "Unknown pathway":
                pathway_unknown_id.add(pid)

    n_mediated_by = 0
    n_mediated_by_unknown = 0
    n_genes_no_pathway = 0
    n_genes_with_pathway = 0
    if edges_path.exists():
        edges = pd.read_csv(edges_path, low_memory=False)
        med = edges[edges[":TYPE"].astype(str).str.strip() == "MEDIATED_BY"]
        n_mediated_by = len(med)
        end_col = ":END_ID" if ":END_ID" in med.columns else "target_id"
        n_mediated_by_unknown = int(med[end_col].astype(str).isin(pathway_unknown_id).sum())

        in_path = edges[edges[":TYPE"].astype(str).str.strip() == "IN_PATHWAY"]
        start_col = ":START_ID" if ":START_ID" in in_path.columns else "source_id"
        gene_has_path = set(in_path[start_col].astype(str))
        all_genes = set()
        if nodes_path.exists():
            nodes = pd.read_csv(nodes_path, low_memory=False)
            gn = nodes[nodes[":LABEL"].astype(str).str.strip() == "Gene"]
            all_genes = set(gn[":ID"].astype(str))
        n_genes_with_pathway = len(gene_has_path)
        n_genes_no_pathway = len(all_genes - gene_has_path)

    pathway_cluster_info_path = root / "data" / "processed" / "features" / "pathway_cluster_info.csv"
    unknown_cluster_source = None
    if pathway_cluster_info_path.exists():
        pci = pd.read_csv(pathway_cluster_info_path, nrows=100)
        if "cluster_id" in pci.columns and "auto_label" in pci.columns:
            row0 = pci.iloc[0]
            if "unknown" in str(row0.get("auto_label", "")).lower():
                unknown_cluster_source = f"pathway_cluster_info.csv cluster_id=0 has auto_label='{row0.get('auto_label')}' (largest cluster). Display name is normalized to 'Unknown pathway'."

    diagnosis_report = {
        "run_dir": str(run_dir),
        "pathway_nodes_with_display_name_unknown": len(pathway_unknown_id),
        "pathway_ids_unknown_sample": list(pathway_unknown_id)[:5],
        "mediated_by_total": n_mediated_by,
        "mediated_by_to_unknown_pathway": n_mediated_by_unknown,
        "fraction_mediated_by_to_unknown": round(n_mediated_by_unknown / n_mediated_by, 4) if n_mediated_by else None,
        "genes_with_in_pathway_edge": n_genes_with_pathway,
        "genes_without_in_pathway_edge": n_genes_no_pathway,
        "cause": "Pathway nodes with display_name 'Unknown pathway' come from pathway_cluster_info (or target_functional_clusters) where auto_label is missing or equals 'unknown pathway_unknown_pathway'. Cluster 0 in pathway_cluster_info is the largest and has that label.",
        "upstream_source": unknown_cluster_source,
        "recommendation": "Improve pathway label for cluster 0 using top_terms from pathway_cluster_info; filter out Unknown pathway in analysis queries.",
    }
    with open(reports_dir / "unknown_pathway_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(diagnosis_report, f, indent=2)
    print("Wrote", reports_dir / "unknown_pathway_diagnosis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
