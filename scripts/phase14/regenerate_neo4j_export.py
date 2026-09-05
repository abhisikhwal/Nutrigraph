"""
Regenerate only Neo4j export (neo4j/nodes.csv, neo4j/edges.csv) for an existing Phase14 run.
Enriches Ingredient names from ingredients.parquet and display_name for Compound/Pathway/Category/Ingredient.
Run from repo root: python scripts/phase14/regenerate_neo4j_export.py [--run-dir PATH | --latest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _latest_phase14_run(repo_root: Path) -> Path | None:
    base = repo_root / "data" / "processed" / "phase14_mediation"
    if not base.exists():
        return None
    runs: list[tuple[float, Path]] = []
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
                    ts = float(f"{date_part}{time_part}")
                    runs.append((ts, d))
                    continue
        except Exception:
            pass
        runs.append((d.stat().st_mtime, d))
    if not runs:
        return None
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate Neo4j export for a Phase14 run with ingredient names")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root")
    parser.add_argument("--run-dir", type=Path, default=None, help="Phase14 run directory (default: latest)")
    parser.add_argument("--latest", action="store_true", help="Use latest Phase14 run (default if no --run-dir)")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    run_dir = args.run_dir
    if run_dir is not None:
        run_dir = run_dir.resolve()
        if not run_dir.exists():
            print(f"ERROR: Run dir does not exist: {run_dir}", file=sys.stderr)
            return 1
        if not (run_dir / "mediation_nodes.csv").exists():
            print(f"ERROR: mediation_nodes.csv not found in {run_dir}", file=sys.stderr)
            return 1
    else:
        run_dir = _latest_phase14_run(repo_root)
        if run_dir is None:
            print("ERROR: No Phase14 run found under data/processed/phase14_mediation (need mediation_nodes.csv and reports/phase14_summary.json)", file=sys.stderr)
            return 1
    print(f"Run dir: {run_dir}")

    import pandas as pd
    from src.phase14.loaders import load_ingredient_id_to_name
    from src.phase14.export import write_neo4j_export

    mediation_nodes = pd.read_csv(run_dir / "mediation_nodes.csv")
    mediation_edges = pd.read_csv(run_dir / "mediation_edges.csv", low_memory=False)
    pair_path = run_dir / "pair_category_mediation.csv"
    pair_mediation = pd.read_csv(pair_path) if pair_path.exists() else None

    id_to_name = load_ingredient_id_to_name(repo_root=repo_root)
    if id_to_name:
        if "label" not in mediation_nodes.columns or "node_id" not in mediation_nodes.columns:
            print("ERROR: mediation_nodes must have node_id and label columns", file=sys.stderr)
            return 1
        mask = mediation_nodes["label"].astype(str).str.strip().str.lower() == "ingredient"
        if "name" not in mediation_nodes.columns:
            mediation_nodes["name"] = mediation_nodes["node_id"]
        def _resolve_name(nid):
            if pd.isna(nid):
                return nid
            k = str(nid).strip()
            return id_to_name.get(k, nid)
        mediation_nodes.loc[mask, "name"] = mediation_nodes.loc[mask, "node_id"].map(_resolve_name)
        print(f"Enriched {mask.sum()} Ingredient nodes with names from ingredients.parquet ({len(id_to_name)} mappings)")
    else:
        print("WARNING: No ingredient_id->name mapping loaded; Ingredient names will remain as ids")

    # Display-name enrichment for Compound, Pathway, Category, Ingredient
    from src.phase14.display_names import enrich_display_names, pathway_unknown_to_better_label
    mediation_nodes, display_stats = enrich_display_names(mediation_nodes, repo_root)
    # Override "Unknown pathway" with better label from pathway_cluster_info top_terms where available
    pathway_better = pathway_unknown_to_better_label(repo_root)
    if pathway_better:
        path_mask = (mediation_nodes["label"].astype(str).str.strip().str.lower() == "pathway") & (mediation_nodes.get("display_name", mediation_nodes.get("name", "")).astype(str).str.strip() == "Unknown pathway")
        for idx in mediation_nodes.index[path_mask]:
            nid = mediation_nodes.at[idx, "node_id"]
            if nid in pathway_better:
                mediation_nodes.at[idx, "display_name"] = pathway_better[nid]
                if "name" in mediation_nodes.columns:
                    mediation_nodes.at[idx, "name"] = pathway_better[nid]
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    display_report = {
        "node_types_enriched": ["Compound", "Pathway", "Category", "Ingredient"],
        "compound_improved": display_stats.get("compound_improved", 0),
        "pathway_improved": display_stats.get("pathway_improved", 0),
        "category_improved": display_stats.get("category_improved", 0),
        "ingredient_with_name": display_stats.get("ingredient_improved", 0),
        "sample_before_after": {
            "compound": display_stats.get("compound_before_after", []),
            "pathway": display_stats.get("pathway_before_after", []),
            "category": display_stats.get("category_before_after", []),
        },
    }
    with open(reports_dir / "display_name_enrichment_report.json", "w", encoding="utf-8") as f:
        json.dump(display_report, f, indent=2)
    print(f"Display names: Compound {display_report['compound_improved']}, Pathway {display_report['pathway_improved']}, Category {display_report['category_improved']} improved")

    run_id = run_dir.name
    write_neo4j_export(run_dir, mediation_nodes, mediation_edges, pair_mediation=pair_mediation, run_id=run_id)
    print(f"Neo4j export written to {run_dir / 'neo4j'} and validation to {run_dir / 'reports' / 'ingredient_name_export_validation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
