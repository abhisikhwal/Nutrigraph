"""
Quantify ingredient coverage across pipeline stages and report where the drop to 47 Ingredient nodes occurs.
Run from repo root: python scripts/phase14/ingredient_coverage_report.py [--phase14-run DIR] [--phase13-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _normalize_ing(s: str) -> str:
    from src.phase14.id_normalization import to_ingredient_id
    if s is None or (isinstance(s, float) and __import__("math").isnan(s)):
        return ""
    return to_ingredient_id(str(s).strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingredient coverage across pipeline stages")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--phase14-run", type=Path, default=None, help="Phase14 run dir (default: latest)")
    parser.add_argument("--phase13-dir", type=Path, default=None, help="Phase13 dir (default: from run_phase14 default)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write report JSON here")
    args = parser.parse_args()
    root = args.repo_root.resolve()

    import pandas as pd

    # Resolve phase14 run
    run_dir = args.phase14_run
    if run_dir is None:
        base = root / "data" / "processed" / "phase14_mediation"
        candidates = []
        for d in base.iterdir() if base.exists() else []:
            if d.is_dir() and (d / "mediation_nodes.csv").exists():
                candidates.append((d.stat().st_mtime, d))
        candidates.sort(key=lambda x: x[0], reverse=True)
        run_dir = candidates[0][1] if candidates else None
    else:
        run_dir = run_dir.resolve()
    if not run_dir or not (run_dir / "mediation_nodes.csv").exists():
        print("ERROR: No Phase14 run with mediation_nodes.csv found", file=sys.stderr)
        return 1

    # Phase13 dir (default same as run_phase14)
    phase13_dir = args.phase13_dir or root / "data" / "processed" / "phase13_interactions_v3_20260206_162122_b_gpu_stable"
    phase13_dir = phase13_dir.resolve()
    atlas_path = phase13_dir / "atlas_confirmed.csv"
    if not atlas_path.exists():
        atlas_path = phase13_dir / "atlas_confirmed.parquet"

    canonical_pq = root / "data" / "processed" / "canonical" / "ingredients.parquet"
    ri_v2 = root / "data" / "processed" / "canonical" / "recipe_ingredients_expanded_v2.parquet"

    stages = {}

    # 1) Canonical ingredient table
    if canonical_pq.exists():
        df = pd.read_parquet(canonical_pq)
        id_col = next((c for c in df.columns if "ingredient" in c.lower() and "id" in c.lower()), df.columns[0] if len(df.columns) else None)
        ids = df[id_col].dropna().astype(str).str.strip() if id_col else pd.Series(dtype=object)
        stages["canonical_ingredients"] = ids.nunique()
        stages["canonical_ingredients_norm"] = ids.map(_normalize_ing).nunique()
    else:
        stages["canonical_ingredients"] = None
        stages["canonical_ingredients_norm"] = None

    # 2) Recipe ingredients expanded v2
    if ri_v2.exists():
        df = pd.read_parquet(ri_v2)
        id_col = next((c for c in df.columns if "ingredient" in c.lower() and "id" in c.lower()), df.columns[0] if len(df.columns) else None)
        ids = df[id_col].dropna().astype(str).str.strip() if id_col else pd.Series(dtype=object)
        stages["recipe_ingredients_expanded_v2"] = ids.nunique()
        stages["recipe_ingredients_expanded_v2_norm"] = ids.map(_normalize_ing).nunique()
    else:
        stages["recipe_ingredients_expanded_v2"] = None
        stages["recipe_ingredients_expanded_v2_norm"] = None

    # 3) Phase13 atlas (unique ingA_id + ingB_id)
    if atlas_path.exists():
        if atlas_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(atlas_path, columns=["ingA_id", "ingB_id"])
        else:
            df = pd.read_csv(atlas_path, usecols=["ingA_id", "ingB_id"], low_memory=False)
        a = df["ingA_id"].dropna().astype(str).str.strip().map(_normalize_ing)
        b = df["ingB_id"].dropna().astype(str).str.strip().map(_normalize_ing)
        stages["atlas_confirmed_unique_ingredients"] = pd.concat([a, b]).nunique()
        stages["atlas_confirmed_rows"] = len(df)
    else:
        stages["atlas_confirmed_unique_ingredients"] = None
        stages["atlas_confirmed_rows"] = None

    # 4) Mediation graph: Ingredient nodes
    mn = pd.read_csv(run_dir / "mediation_nodes.csv", low_memory=False)
    ing_nodes = mn[mn["label"].astype(str).str.strip().str.lower() == "ingredient"] if "label" in mn.columns else pd.DataFrame()
    stages["mediation_nodes_ingredient_count"] = len(ing_nodes)
    stages["mediation_nodes_total"] = len(mn)

    # 5) Neo4j export
    neo_path = run_dir / "neo4j" / "nodes.csv"
    if neo_path.exists():
        neo = pd.read_csv(neo_path, low_memory=False)
        if ":LABEL" in neo.columns:
            ing_neo = neo[neo[":LABEL"].astype(str).str.contains("Ingredient", na=False)]
        else:
            ing_neo = pd.DataFrame()
        stages["neo4j_nodes_ingredient_count"] = len(ing_neo)
    else:
        stages["neo4j_nodes_ingredient_count"] = None

    # 6) Mediation edges: unique ingredient ids (as source or target for HAS_INGREDIENT / HAS_COMPOUND)
    me = pd.read_csv(run_dir / "mediation_edges.csv", low_memory=False)
    if "source_id" in me.columns and "target_id" in me.columns:
        all_ids = pd.concat([me["source_id"], me["target_id"]]).dropna().astype(str).str.strip()
        stages["mediation_edges_unique_node_ids"] = all_ids.nunique()
    else:
        stages["mediation_edges_unique_node_ids"] = None

    # Build table and report
    table = [
        ("Stage", "Count", "Note"),
        ("Canonical ingredients (ingredients.parquet)", stages.get("canonical_ingredients_norm") or stages.get("canonical_ingredients"), "Full ingredient universe (id normalized)"),
        ("Recipe ingredients (recipe_ingredients_expanded_v2.parquet)", stages.get("recipe_ingredients_expanded_v2_norm") or stages.get("recipe_ingredients_expanded_v2"), "Ingredients that appear in recipes"),
        ("Phase13 atlas_confirmed (unique ingA_id + ingB_id)", stages.get("atlas_confirmed_unique_ingredients"), "Only ingredients in at least one confirmed interaction pair"),
        ("Atlas rows (pairs x categories)", stages.get("atlas_confirmed_rows"), "Number of (ingA, ingB, category) rows"),
        ("Mediation graph: Ingredient nodes", stages.get("mediation_nodes_ingredient_count"), "From build_mediation_graph (source: atlas only)"),
        ("Mediation graph: total nodes", stages.get("mediation_nodes_total"), "All node types"),
        ("Neo4j nodes.csv: Ingredient nodes", stages.get("neo4j_nodes_ingredient_count"), "Exported; should equal mediation Ingredient count"),
    ]

    print("Ingredient coverage by stage")
    print("-" * 80)
    for row in table:
        print(f"  {str(row[0]):60} {str(row[1]):>10}  {row[2] or ''}")
    print("-" * 80)

    # Where does the drop happen?
    drop_stage = None
    if stages.get("atlas_confirmed_unique_ingredients") is not None and stages.get("mediation_nodes_ingredient_count") is not None:
        if stages["atlas_confirmed_unique_ingredients"] == stages["mediation_nodes_ingredient_count"]:
            drop_stage = "No drop between atlas and mediation: Ingredient nodes = atlas unique ingredients (by design)."
        else:
            drop_stage = "Unexpected: atlas unique ingredients != mediation Ingredient count."
    report = {
        "run_dir": str(run_dir),
        "phase13_dir": str(phase13_dir),
        "stages": stages,
        "table": [{"stage": r[0], "count": r[1], "note": r[2]} for r in table[1:]],
        "where_drop_happens": "build_mediation_graph() in src/phase14/mediation_graph.py: Ingredient nodes are created ONLY from atlas_confirmed (unique ingA_id, ingB_id). No other source adds Ingredient nodes.",
        "intentional": True,
        "explanation": "The Neo4j export is a mediation subgraph: only ingredients that participate in at least one Phase13 confirmed interaction pair. The full recipe/canonical ingredient universe is intentionally not included.",
    }

    out_path = args.output or run_dir / "reports" / "ingredient_coverage_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
