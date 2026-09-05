"""
Phase14 input validation and path resolution (shared logic).
All paths in returned dicts are relative to repo_root.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Phase13 required (relative to phase13_dir)
PHASE13_REQUIRED = {
    "atlas_confirmed.csv": ["ingA_id", "ingB_id", "category", "did", "p_analytic", "q_global"],
    "bootstrap_stability.csv": ["ingA_id", "ingB_id", "category", "p_boot", "stability_score"],
}

PHASE11_PREFERRED = [
    "phase11_metabolomics/compound_metabolite_links.parquet",
    "phase11_metabolomics/hmdb_metabolites.parquet",
]
PHASE11_ALTERNATIVE_PATTERNS = ["**/phase11_metabolomics/*.parquet", "**/ingredient*compound*.parquet", "**/compound*ingredient*.parquet"]

PHASE12_PREFERRED = ["phase12_genetics/food_compound_gene_links.parquet"]
PHASE12_ALTERNATIVE_PATTERNS = ["**/phase12_genetics/*.parquet", "**/food_compound_gene*.parquet"]

PHASE16_PREFERRED = [
    "phase16_bindingdb/compound_target_edges_bindingdb.parquet",
    "phase16_bindingdb/bindingdb_matched.parquet",
]
PHASE16_AT_LEAST_ONE_PATTERNS = ["**/phase16_bindingdb/*.parquet", "**/phase16_bindingdb/*.csv"]

FEATURES_REQUIRED_OPTIONAL = [
    ("features/pathway_cluster_info.csv", "pathway_cluster_info"),
    ("features/pathway_bundles.json", "pathway_bundles"),
    ("features/target_functional_clusters.csv", "target_clusters"),
]

RI_PREFERRED = ["canonical/recipe_ingredients_expanded_v2.parquet", "canonical/recipe_ingredients_expanded.parquet"]
RI_PATTERNS = ["**/recipe_ingredients*expanded*.parquet", "**/recipe_ingredients*.parquet"]
SIG_PREFERRED = ["phase17_reaggregation/recipe_functional_signatures_v3.parquet", "exports_v2/recipes_biological_effects_v2_FINAL.parquet"]
SIG_PATTERNS = ["**/recipe*functional*signature*.parquet", "**/recipes_biological*.parquet"]


def _rel_to_repo(repo_root: Path, absolute_path: Path) -> str:
    """Return path as relative to repo_root if under it."""
    try:
        return str(absolute_path.relative_to(repo_root))
    except ValueError:
        return str(absolute_path)


def _resolve(repo_root: Path, rel: str) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        return (repo_root / p).resolve()
    return p


def probe_csv(path: Path, max_rows: int = 50) -> Tuple[Optional[List[str]], Optional[str]]:
    if not path.exists():
        return None, "File not found"
    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=max_rows)
        return list(df.columns), None
    except Exception as e:
        return None, str(e)


def probe_parquet(path: Path) -> Tuple[Optional[List[str]], Optional[str]]:
    if not path.exists():
        return None, "File not found"
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        return [c for c in t.column_names], None
    except ImportError:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return list(df.columns), None
        except Exception as e:
            return None, str(e)
    except Exception as e:
        return None, str(e)


def file_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "size_bytes": st.st_size,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def find_best_candidate(repo_root: Path, processed: Path, preferred: List[str], patterns: List[str]) -> Optional[Path]:
    for rel in preferred:
        p = _resolve(repo_root, "data/processed/" + rel)
        if p.exists():
            return p
    candidates: List[Path] = []
    for pat in patterns:
        for p in processed.glob(pat):
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def validate_phase13(repo_root: Path, phase13_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    report: Dict[str, Any] = {"phase13_dir": _rel_to_repo(repo_root, phase13_dir), "files": {}, "schema_ok": True}
    errors: List[str] = []
    if not phase13_dir.is_absolute():
        phase13_dir = (repo_root / phase13_dir).resolve()
    if not phase13_dir.exists():
        errors.append(f"Phase13 dir not found: {phase13_dir}")
        report["phase13_dir_exists"] = False
        return report, errors
    report["phase13_dir_exists"] = True

    for fname, required_cols in PHASE13_REQUIRED.items():
        p = phase13_dir / fname
        info = file_info(p)
        report["files"][fname] = info
        if not info["exists"]:
            errors.append(f"Required file missing: {fname}")
            continue
        report["files"][fname]["path_rel"] = _rel_to_repo(repo_root, p)
        cols, err = probe_csv(p)
        report["files"][fname]["schema_probe"] = {"columns": cols, "error": err}
        if err:
            errors.append(f"{fname}: schema probe failed: {err}")
            report["schema_ok"] = False
        elif cols:
            missing = [c for c in required_cols if c not in cols]
            if missing:
                errors.append(f"{fname}: missing columns: {missing}")
                report["schema_ok"] = False
            report["files"][fname]["required_cols_ok"] = len(missing) == 0

    kg_edges_path = phase13_dir / "kg" / "kg_edges.csv"
    if not kg_edges_path.exists():
        kg_edges_path = phase13_dir / "kg_edges.csv"
    report["files"]["kg_edges.csv"] = file_info(kg_edges_path)
    if not report["files"]["kg_edges.csv"]["exists"]:
        report["files"]["kg_edges.csv"]["optional"] = True
        report["files"]["kg_edges.csv"]["note"] = "Optional; ground truth from atlas_confirmed+bootstrap_stability"
    else:
        report["files"]["kg_edges.csv"]["path_rel"] = _rel_to_repo(repo_root, kg_edges_path)
        cols, err = probe_csv(kg_edges_path)
        report["files"]["kg_edges.csv"]["schema_probe"] = {"columns": cols, "error": err}
    return report, errors


def validate_optional_dir(
    repo_root: Path,
    processed: Path,
    preferred: List[str],
    patterns: List[str],
    label: str,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    report: Dict[str, Any] = {"label": label, "preferred": [], "found": [], "selected": None}
    for rel in preferred:
        p = _resolve(repo_root, "data/processed/" + rel)
        info = file_info(p)
        rec = {"rel": rel, **info}
        if info.get("exists"):
            rec["path_rel"] = _rel_to_repo(repo_root, p)
            if p.suffix.lower() == ".parquet":
                cols, err = probe_parquet(p)
                if err and p.with_suffix(".csv").exists():
                    cols, err = probe_csv(p.with_suffix(".csv"))
            else:
                cols, err = probe_csv(p)
            rec["schema_probe"] = {"columns": cols, "error": err}
            report["found"].append(rec)
        report["preferred"].append(rec)
    selected = find_best_candidate(repo_root, processed, preferred, patterns)
    if selected is not None:
        report["selected"] = _rel_to_repo(repo_root, selected)
    return report, selected


def run_validation(repo_root: Path, phase13_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Optional[str]], List[str]]:
    """
    Run full validation. Returns (report_dict, selected_paths_rel, errors).
    selected_paths_rel: keys phase13_dir, atlas_confirmed, kg_edges, kg_nodes, bootstrap_stability,
    metabolomics, genetics, binding, pathway_cluster_info, pathway_bundles, target_clusters,
    recipe_ingredients, signatures. Values are paths relative to repo_root (str) or None.
    """
    processed = repo_root / "data" / "processed"
    repo_root = Path(repo_root).resolve()
    phase13_dir = Path(phase13_dir)
    phase13_abs = (repo_root / phase13_dir).resolve()
    report: Dict[str, Any] = {"repo_root": str(repo_root), "phase13_dir_arg": str(phase13_dir)}
    all_errors: List[str] = []
    selected: Dict[str, Optional[str]] = {}

    p13_report, p13_errors = validate_phase13(repo_root, phase13_abs)
    report["phase13"] = p13_report
    all_errors.extend(p13_errors)

    if p13_report.get("phase13_dir_exists") or phase13_abs.exists():
        selected["phase13_dir"] = _rel_to_repo(repo_root, phase13_abs)
        selected["atlas_confirmed"] = _rel_to_repo(repo_root, phase13_abs / "atlas_confirmed.csv")
        selected["bootstrap_stability"] = _rel_to_repo(repo_root, phase13_abs / "bootstrap_stability.csv")
        kg_e = phase13_abs / "kg" / "kg_edges.csv"
        if not kg_e.exists():
            kg_e = phase13_abs / "kg_edges.csv"
        selected["kg_edges"] = _rel_to_repo(repo_root, kg_e) if kg_e.exists() else None
        kg_n = phase13_abs / "kg" / "kg_nodes.csv"
        if not kg_n.exists():
            kg_n = phase13_abs / "kg_nodes.csv"
        selected["kg_nodes"] = _rel_to_repo(repo_root, kg_n) if kg_n.exists() else None
    else:
        selected["phase13_dir"] = None
        selected["atlas_confirmed"] = None
        selected["bootstrap_stability"] = None
        selected["kg_edges"] = None
        selected["kg_nodes"] = None

    r11, sel11 = validate_optional_dir(repo_root, processed, PHASE11_PREFERRED, PHASE11_ALTERNATIVE_PATTERNS, "phase11_metabolomics")
    report["phase11"] = r11
    selected["metabolomics"] = _rel_to_repo(repo_root, sel11) if sel11 else None

    r12, sel12 = validate_optional_dir(repo_root, processed, PHASE12_PREFERRED, PHASE12_ALTERNATIVE_PATTERNS, "phase12_genetics")
    report["phase12"] = r12
    selected["genetics"] = _rel_to_repo(repo_root, sel12) if sel12 else None

    r16, sel16 = validate_optional_dir(repo_root, processed, PHASE16_PREFERRED, PHASE16_AT_LEAST_ONE_PATTERNS, "phase16_bindingdb")
    report["phase16"] = r16
    selected["binding"] = _rel_to_repo(repo_root, sel16) if sel16 else None

    for rel, key in FEATURES_REQUIRED_OPTIONAL:
        p = processed / rel
        selected[key] = _rel_to_repo(repo_root, p) if p.exists() else None
    report["features"] = {k: selected.get(k) for k in ["pathway_cluster_info", "pathway_bundles", "target_clusters"]}

    ri_sel = find_best_candidate(repo_root, processed, RI_PREFERRED, RI_PATTERNS)
    selected["recipe_ingredients"] = _rel_to_repo(repo_root, ri_sel) if ri_sel else None
    sig_sel = find_best_candidate(repo_root, processed, SIG_PREFERRED, SIG_PATTERNS)
    selected["signatures"] = _rel_to_repo(repo_root, sig_sel) if sig_sel else None

    report["selected_paths"] = selected
    report["errors"] = all_errors
    return report, selected, all_errors


def print_validation_table(report: Dict[str, Any], selected: Dict[str, Optional[str]]) -> None:
    rows: List[Tuple[str, str, str, str]] = []
    p13 = report.get("phase13", {})
    if p13.get("phase13_dir_exists"):
        for fname, info in p13.get("files", {}).items():
            if info.get("exists"):
                size = info.get("size_bytes", 0)
                mtime = (info.get("mtime_iso") or "")[:19]
                rows.append(("Phase13 (required)", fname, "FOUND", f"{size} bytes  {mtime}"))
            else:
                rows.append(("Phase13 (required)", fname, "MISSING", ""))
    else:
        rows.append(("Phase13 (required)", "phase13_dir", "MISSING", str(p13.get("phase13_dir", ""))))

    for label, key in [("Phase11 metabolomics", "metabolomics"), ("Phase12 genetics", "genetics"), ("Phase16 binding", "binding")]:
        path_rel = selected.get(key)
        if path_rel:
            p = Path(report["repo_root"]) / path_rel
            if p.exists():
                st = p.stat()
                mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()[:19]
                rows.append((label, key, "FOUND", f"{st.st_size} bytes  {mtime_iso}"))
            else:
                rows.append((label, key, "FOUND", path_rel))
        else:
            rows.append((label, key, "MISSING", ""))

    for key in ["pathway_cluster_info", "pathway_bundles", "target_clusters"]:
        path_rel = selected.get(key)
        if path_rel:
            p = Path(report["repo_root"]) / path_rel
            if p.exists():
                st = p.stat()
                mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()[:19]
                rows.append(("Features", key, "FOUND", f"{st.st_size} bytes  {mtime_iso}"))
            else:
                rows.append(("Features", key, "FOUND", path_rel))
        else:
            rows.append(("Features", key, "MISSING", ""))

    for label, key in [("Optional", "recipe_ingredients"), ("Optional", "signatures")]:
        path_rel = selected.get(key)
        if path_rel:
            p = Path(report["repo_root"]) / path_rel
            if p.exists():
                st = p.stat()
                mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()[:19]
                rows.append((label, key, "FOUND", f"{st.st_size} bytes  {mtime_iso}"))
            else:
                rows.append((label, key, "FOUND", path_rel))
        else:
            rows.append((label, key, "MISSING", ""))

    col_w = (28, 32, 10, 42)
    print("  ".join([f"{'Category':<{col_w[0]}}", f"{'Input':<{col_w[1]}}", f"{'Status':<{col_w[2]}}", f"{'Detail':<{col_w[3]}}"]))
    print("  ".join("-" * w for w in col_w))
    for cat, inp, status, detail in rows:
        print("  ".join([f"{cat[:col_w[0]]:<{col_w[0]}}", f"{inp[:col_w[1]]:<{col_w[1]}}", f"{status:<{col_w[2]}}", f"{str(detail)[:col_w[3]]:<{col_w[3]}}"]))
