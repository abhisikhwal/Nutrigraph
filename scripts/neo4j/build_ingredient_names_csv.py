"""
Discover ING_ID -> human name sources, build ingredient_id,name CSV for Neo4j LOAD CSV.
Validates against Phase14 Neo4j nodes.csv; writes to run dir and snapshot neo4j/ folder.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Name-like column headers we look for
NAME_COL_CANDIDATES = [
    "ingredient_name", "ingredient_name_clean", "display_name", "name", "raw_name",
    "canonical_name", "ing_name", "ingredient_raw", "scientific_name",
]
# ID-like column headers for ingredients
ID_COL_CANDIDATES = ["ingredient_id", "ing_id", ":ID", "id", "ingredient"]


def run_ripgrep(pattern: str, glob: str, path: Path) -> List[Path]:
    """Return list of file paths containing pattern (ripgrep -l). Fallback: glob under data/scripts/configs."""
    try:
        r = subprocess.run(
            ["rg", "-l", "--glob", glob, pattern, str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(path),
        )
        if r.returncode == 0 and r.stdout:
            return [(path / p.strip()).resolve() if not Path(p.strip()).is_absolute() else Path(p.strip()).resolve() for p in r.stdout.strip().split("\n") if p.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Fallback: glob under data/ and scripts/ only, read first 8k
    out: List[Path] = []
    suffix = glob.replace("*", "") if glob.startswith("*") else "." + glob
    for base in [path / "data", path / "scripts", path / "configs"]:
        if not base.exists():
            continue
        for f in base.rglob("*" + suffix):
            if not f.is_file():
                continue
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                    if pattern in fp.read(8192):
                        out.append(f.resolve())
            except Exception:
                pass
    return out


def discovery_report(repo_root: Path) -> Tuple[List[Dict], List[Path]]:
    """Search for files with ING_ and/or name-like headers. Return shortlist + candidate paths."""
    report: List[Dict] = []
    candidates: List[Path] = []

    # Known high-value paths (no full repo scan)
    known = [
        repo_root / "data" / "processed" / "canonical" / "ingredients.parquet",
        repo_root / "data" / "processed" / "canonical" / "ingredients_fixed.parquet",
        repo_root / "data" / "processed" / "canonical" / "ingredients_expanded.parquet",
        repo_root / "data" / "processed" / "canonical" / "recipe_ingredients_expanded.parquet",
        repo_root / "data" / "processed" / "canonical" / "ingredient_compound_canonical.csv",
        repo_root / "data" / "processed" / "recovered" / "ingredients_RECOVERED.csv",
        repo_root / "data" / "processed" / "milestones" / "phase14" / "v1_working_2026-02-19" / "phase14_20260219_204918" / "neo4j" / "nodes.csv",
        repo_root / "data" / "processed" / "phase14_mediation" / "phase14_20260219_204918" / "neo4j" / "nodes.csv",
    ]
    for p in known:
        if p.exists():
            candidates.append(p.resolve())

    # Build shortlist with 20-line preview (for CSV/readable; parquet we just note path)
    for path in sorted(set(candidates), key=lambda x: str(x))[:40]:
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
        entry = {"path": str(path), "relative": str(rel)}
        try:
            if path.suffix.lower() == ".csv":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()[:21]
                entry["preview"] = "".join(lines)
                entry["preview_line_count"] = len(lines)
            else:
                entry["preview"] = "(binary or non-CSV; use pandas to inspect)"
        except Exception as e:
            entry["preview"] = f"(read error: {e})"
        report.append(entry)

    return report, candidates


def load_mapping_from_parquet(repo_root: Path) -> Dict[str, str]:
    """Build id -> name from data/processed/canonical/ingredients.parquet (canonical_name, scientific_name)."""
    mapping: Dict[str, str] = {}
    p = repo_root / "data" / "processed" / "canonical" / "ingredients.parquet"
    if not p.exists():
        return mapping
    try:
        import pandas as pd
        df = pd.read_parquet(p)
        if "ingredient_id" not in df.columns:
            return mapping
        id_col = "ingredient_id"
        # Prefer canonical_name; fallback scientific_name
        for _, row in df.iterrows():
            ing_id = row.get(id_col)
            if pd.isna(ing_id) or not str(ing_id).strip():
                continue
            ing_id = str(ing_id).strip()
            name = None
            for col in ["canonical_name", "scientific_name", "name"]:
                if col in df.columns:
                    v = row.get(col)
                    if not pd.isna(v) and str(v).strip() and not _is_id_like(str(v)):
                        name = str(v).strip()
                        break
            if name and ing_id:
                mapping[ing_id] = name
    except Exception as e:
        raise RuntimeError(f"Failed to load {p}: {e}") from e
    return mapping


def load_mapping_from_recipe_ingredients(repo_root: Path) -> Dict[str, List[str]]:
    """Return ingredient_id -> list of ingredient_raw (for fallback / frequency)."""
    from collections import defaultdict
    out: Dict[str, List[str]] = defaultdict(list)
    for name in ["recipe_ingredients_expanded.parquet", "recipe_ingredients_expanded.csv"]:
        p = repo_root / "data" / "processed" / "canonical" / name
        if not p.exists():
            continue
        try:
            import pandas as pd
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p, columns=["ingredient_id", "ingredient_raw"])
            else:
                df = pd.read_csv(p, nrows=300000, usecols=["ingredient_id", "ingredient_raw"], low_memory=False)
            if "ingredient_id" not in df.columns or "ingredient_raw" not in df.columns:
                continue
            for _, row in df.iterrows():
                iid = row.get("ingredient_id")
                raw = row.get("ingredient_raw")
                if pd.isna(iid) or pd.isna(raw):
                    continue
                iid = str(iid).strip()
                raw = str(raw).strip()
                if raw and not _is_id_like(raw) and len(raw) > 1:
                    out[iid].append(raw)
            break
        except Exception:
            continue
    return dict(out)


def _is_id_like(s: str) -> bool:
    if not s or not s.strip():
        return True
    s = s.strip()
    if re.match(r"^ING_\d+$", s):
        return True
    if re.match(r"^[A-Z]{2,}_[A-Z0-9_]+$", s) and len(s) < 25:
        return True
    return False


def build_mapping(repo_root: Path, use_recipe_ingredients: bool = False) -> Dict[str, str]:
    """
    Build best id -> human name mapping.
    (A) ingredients.parquet canonical_name/scientific_name (primary)
    (B) Optional: recipe_ingredients ingredient_raw for missing ids
    """
    mapping = load_mapping_from_parquet(repo_root)
    if use_recipe_ingredients:
        recipe_raw = load_mapping_from_recipe_ingredients(repo_root)
        from collections import Counter
        for ing_id, raw_list in recipe_raw.items():
            if ing_id in mapping or not raw_list:
                continue
            counts = Counter(raw_list)
            best = max(counts.keys(), key=lambda x: (counts[x], len(x)))
            if not _is_id_like(best):
                mapping[ing_id] = best
    return mapping


def validate_against_nodes(nodes_csv: Path, mapping: Dict[str, str]) -> Dict[str, Any]:
    """Filter nodes to Ingredient label, report coverage."""
    import csv as csv_mod
    ingredient_ids: List[str] = []
    with open(nodes_csv, "r", encoding="utf-8", newline="") as f:
        r = csv_mod.DictReader(f)
        for row in r:
            label = (row.get(":LABEL") or row.get("label") or "").strip()
            if label != "Ingredient":
                continue
            nid = row.get(":ID") or row.get("id") or row.get("node_id")
            if nid:
                ingredient_ids.append(str(nid).strip())
    total = len(ingredient_ids)
    mapped = sum(1 for iid in ingredient_ids if iid in mapping)
    unmapped = [iid for iid in ingredient_ids if iid not in mapping][:20]
    return {
        "total_ingredient_nodes": total,
        "mapped_count": mapped,
        "coverage_pct": (100.0 * mapped / total) if total else 0.0,
        "sample_unmapped_ids": unmapped,
    }


def write_ingredient_names_csv(mapping: Dict[str, str], out_path: Path) -> None:
    """Write CSV with header ingredient_id,name. Only rows that have a valid name."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(iid, name) for iid, name in sorted(mapping.items()) if name and not _is_id_like(name)]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ingredient_id", "name"])
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ingredient_id,name CSV for Neo4j")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--nodes-csv", type=Path, default=None, help="Neo4j nodes.csv; default snapshot path")
    parser.add_argument("--out-dir", type=Path, default=None, help="Also write here; default run dir + snapshot neo4j/")
    parser.add_argument("--discovery-only", action="store_true", help="Only print discovery report and exit")
    parser.add_argument("--use-recipe-ingredients", action="store_true", help="Also use recipe_ingredients for missing names (slower)")
    args = parser.parse_args()
    repo = args.repo_root.resolve()

    # Default nodes path (snapshot)
    snapshot_nodes = repo / "data" / "processed" / "milestones" / "phase14" / "v1_working_2026-02-19" / "phase14_20260219_204918" / "neo4j" / "nodes.csv"
    run_dir_neo4j = repo / "data" / "processed" / "phase14_mediation" / "phase14_20260219_204918" / "neo4j"
    nodes_csv = args.nodes_csv or snapshot_nodes
    if not nodes_csv.is_absolute():
        nodes_csv = (repo / nodes_csv).resolve()

    # ---- 1) Discovery ----
    print("=== 1) Repo-wide discovery ===\n")
    report, candidates = discovery_report(repo)
    print(f"Found {len(candidates)} candidate files (ING_ or name-like headers).\n")
    for i, entry in enumerate(report[:25]):
        print(f"--- Candidate {i+1}: {entry['relative']} ---")
        prev = entry.get("preview", "")
        if len(prev) > 1500:
            prev = prev[:1500] + "\n..."
        print(prev)
        print()

    if args.discovery_only:
        return 0

    # ---- 2) Build mapping ----
    print("=== 2) Build ING_ID -> name mapping ===\n")
    mapping = build_mapping(repo, use_recipe_ingredients=args.use_recipe_ingredients)
    print(f"Mapping size: {len(mapping)} ingredient_id -> name")
    if mapping:
        sample = list(mapping.items())[:10]
        for k, v in sample:
            print(f"  {k} -> {v}")

    # ---- 3) Validate against Neo4j nodes ----
    if nodes_csv.exists():
        print("\n=== 3) Validation vs Neo4j nodes.csv ===\n")
        stats = validate_against_nodes(nodes_csv, mapping)
        print(f"Total Ingredient nodes: {stats['total_ingredient_nodes']}")
        print(f"Mapped (have human name): {stats['mapped_count']}")
        print(f"Coverage: {stats['coverage_pct']:.1f}%")
        print(f"Sample unmapped ids: {stats['sample_unmapped_ids']}")
    else:
        print(f"\nNodes CSV not found: {nodes_csv}; skipping validation.")
        stats = {}

    # ---- 4) Output artifacts ----
    print("\n=== 4) Output artifacts ===\n")
    out_paths = []
    # Run dir
    run_dir_neo4j.mkdir(parents=True, exist_ok=True)
    out1 = run_dir_neo4j / "ingredient_names.csv"
    write_ingredient_names_csv(mapping, out1)
    out_paths.append(out1)
    print(f"Wrote: {out1}")
    # Snapshot dir
    snapshot_neo4j = snapshot_nodes.parent
    if snapshot_neo4j != run_dir_neo4j:
        out2 = snapshot_neo4j / "ingredient_names.csv"
        write_ingredient_names_csv(mapping, out2)
        out_paths.append(out2)
        print(f"Wrote: {out2}")
    if args.out_dir:
        out3 = Path(args.out_dir).resolve() / "ingredient_names.csv"
        write_ingredient_names_csv(mapping, out3)
        out_paths.append(out3)
        print(f"Wrote: {out3}")

    # ---- 5) Neo4j Cypher ----
    print("\n=== 5) Neo4j Cypher (LOAD CSV) ===\n")
    print("Place ingredient_names.csv in Neo4j import dir (e.g. import/ingredient_names.csv).")
    print("Then run:\n")
    cypher = (
        "LOAD CSV WITH HEADERS FROM 'file:///ingredient_names.csv' AS row\n"
        "MATCH (i:Ingredient {id: row.ingredient_id})\n"
        "SET i.name = row.name;"
    )
    print(cypher)
    print("\n(If using Neo4j Desktop or different import path, adjust the file URL.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
