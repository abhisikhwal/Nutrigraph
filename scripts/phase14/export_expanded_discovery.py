"""
Phase14.5: Export expanded discovery graph (more ingredients) without replacing confirmed graph.
Adds ingredients that share compounds with atlas ingredients (one-hop expansion).
Output: run_dir/neo4j_expanded/ and reports/expanded_discovery_report.json.
Run from repo root: python scripts/phase14/export_expanded_discovery.py [--run-dir PATH] [--min-shared-compounds 1]
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
    parser = argparse.ArgumentParser(description="Export expanded discovery graph (more ingredients)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--min-shared-compounds", type=int, default=1, help="Min shared compounds with atlas to add ingredient")
    args = parser.parse_args()
    root = args.repo_root.resolve()

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _latest_run(root)
    if run_dir is None:
        print("ERROR: No Phase14 run found", file=sys.stderr)
        return 1
    run_dir = run_dir.resolve()

    import pandas as pd
    from src.phase14.loaders import load_ingredient_id_to_name, load_ingredient_compound
    from src.phase14.id_normalization import to_ingredient_id, to_compound_id

    from src.phase14.loaders import discover_all
    paths = discover_all(root)

    # Load confirmed graph
    nodes_df = pd.read_csv(run_dir / "mediation_nodes.csv", low_memory=False)
    edges_df = pd.read_csv(run_dir / "mediation_edges.csv", low_memory=False)
    atlas_ingredients = set(
        nodes_df[nodes_df["label"].astype(str).str.strip().str.lower() == "ingredient"]["node_id"].astype(str).str.strip()
    )
    graph_compounds = set(
        edges_df[edges_df["edge_type"] == "HAS_COMPOUND"]["target_id"].astype(str).str.strip()
    )

    n_confirmed_ingredients = len(atlas_ingredients)
    n_confirmed_compounds = len(graph_compounds)

    # Load ingredient_compound
    ic_df = load_ingredient_compound(paths)
    if ic_df is None or ic_df.empty:
        print("WARNING: No ingredient_compound; expanded graph will have same ingredients as confirmed.", file=sys.stderr)
        expanded_ingredients = set(atlas_ingredients)
        new_edges = []
    else:
        id_col = "ingredient_id" if "ingredient_id" in ic_df.columns else ic_df.columns[0]
        cmp_col = "compound_id" if "compound_id" in ic_df.columns else next((c for c in ic_df.columns if "compound" in c.lower()), ic_df.columns[1])
        ic_df = ic_df.dropna(subset=[id_col, cmp_col])
        ic_df["ingredient_id"] = ic_df[id_col].astype(str).str.strip().apply(lambda x: to_ingredient_id(x))
        ic_df["compound_id"] = ic_df[cmp_col].astype(str).str.strip().apply(lambda x: to_compound_id(x))

        # Compounds that are in the graph
        ic_in_graph = ic_df[ic_df["compound_id"].isin(graph_compounds)]
        # For each candidate ingredient (not in atlas), count shared compounds with graph
        candidate_shared = ic_in_graph[~ic_in_graph["ingredient_id"].isin(atlas_ingredients)].groupby("ingredient_id")["compound_id"].nunique()
        to_add = candidate_shared[candidate_shared >= args.min_shared_compounds].index.tolist()
        expanded_ingredients = set(atlas_ingredients) | set(to_add)
        new_ingredient_set = set(to_add)
        new_edges = []
        for ing_id in new_ingredient_set:
            comps = ic_in_graph[ic_in_graph["ingredient_id"] == ing_id]["compound_id"].unique()
            for cid in comps:
                if cid in graph_compounds:
                    new_edges.append({"source_id": ing_id, "target_id": cid, "edge_type": "HAS_COMPOUND", "weight": 1.0})

    id_to_name = load_ingredient_id_to_name(repo_root=root)
    new_node_rows = []
    for ing_id in expanded_ingredients - atlas_ingredients:
        name = id_to_name.get(ing_id, ing_id)
        new_node_rows.append({"node_id": ing_id, "label": "Ingredient", "name": name, "source": "expanded_discovery"})

    if not new_node_rows:
        extended_nodes = nodes_df
        extended_edges = edges_df
    else:
        new_nodes = pd.DataFrame(new_node_rows)
        extended_nodes = pd.concat([nodes_df, new_nodes], ignore_index=True)
        if new_edges:
            new_edges_df = pd.DataFrame(new_edges)
            extended_edges = pd.concat([edges_df, new_edges_df], ignore_index=True)
        else:
            extended_edges = edges_df

    n_expanded_ingredients = len(extended_nodes[extended_nodes["label"].astype(str).str.strip().str.lower() == "ingredient"])
    n_new_edges = len(extended_edges) - len(edges_df)

    # Export to neo4j_expanded/ (do NOT overwrite neo4j/)
    out_dir = run_dir / "neo4j_expanded"
    out_dir.mkdir(parents=True, exist_ok=True)
    from src.phase14.export import neo4j_nodes_csv, neo4j_edges_csv, NEO4J_NODE_HEADERS, NEO4J_EDGE_HEADERS
    from src.phase14.display_names import enrich_display_names

    extended_nodes = enrich_display_names(extended_nodes, root)[0]
    n_df = neo4j_nodes_csv(extended_nodes)
    e_df = neo4j_edges_csv(extended_edges, include_properties=True)
    n_df.to_csv(out_dir / "nodes.csv", index=False)
    e_df.to_csv(out_dir / "edges.csv", index=False)

    report = {
        "run_dir": str(run_dir),
        "confirmed_ingredient_count": n_confirmed_ingredients,
        "expanded_ingredient_count": n_expanded_ingredients,
        "added_ingredient_count": n_expanded_ingredients - n_confirmed_ingredients,
        "confirmed_compound_count": n_confirmed_compounds,
        "new_edges_added": n_new_edges,
        "expansion_strategy": "One-hop: ingredients that share at least min_shared_compounds with atlas ingredients (via ingredient_compound) and connect to existing graph compounds.",
        "min_shared_compounds": args.min_shared_compounds,
        "output_dir": str(out_dir),
    }
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "expanded_discovery_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Expanded discovery export:", out_dir)
    print("Ingredients: confirmed", n_confirmed_ingredients, "-> expanded", n_expanded_ingredients, "(+%d)" % (n_expanded_ingredients - n_confirmed_ingredients))
    print("Report:", reports_dir / "expanded_discovery_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
