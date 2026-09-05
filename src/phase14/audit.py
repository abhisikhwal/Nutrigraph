"""
Phase14: Data audit and discovery — scan data/processed for capability-tagged files,
pick best sources for Ingredient→Compound and Compound→Gene, write audit reports.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# CSV columns often mixed-type; read as str to avoid DtypeWarning (audit scan)
AUDIT_CSV_DTYPE_STR_COLUMNS = [
    "fdb_id", "fdb_id_raw", "fdb_id_norm", "bindingdb_id", "name", "role",
]

# Flexible column detection for compound–gene / compound–target
COMPOUND_KEY_COLUMNS = [
    "compound_id", "inchikey", "inchi_key", "InChIKey", "pubchem_cid", "chembl_id",
]
GENE_TARGET_COLUMNS = [
    "gene", "gene_symbol", "gene_id", "target", "target_name", "uniprot", "uniprot_id",
]

# Capability tags: which columns imply which use case
CAPABILITY_RULES = [
    (
        "INGREDIENT_COMPOUND",
        lambda cols: _has_ingredient_id(cols) and _has_compound_key(cols),
    ),
    (
        "COMPOUND_GENE",
        lambda cols: _has_compound_key(cols) and _has_gene_or_target(cols),
    ),
    (
        "COMPOUND_TARGET",
        lambda cols: _has_compound_key(cols) and _has_gene_or_target(cols),
    ),
    (
        "COMPOUND_METABOLITE",
        lambda cols: _has_any(cols, ["hmdb", "metabolite", "metabolite_id", "metabolite_name", "metabolite_hmdb_id"]),
    ),
    (
        "FOOD_COMPOUND",
        lambda cols: _has_any(cols, ["food", "food_name", "ingredient_name", "food_id"]) and _has_compound_key(cols),
    ),
    (
        "INGREDIENT_INCHIKEY",
        lambda cols: _has_ingredient_id(cols) and _has_any(cols, ["inchikey", "inchi_key", "InChIKey"]),
    ),
    (
        "INCHIKEY_COMPOUND",
        lambda cols: _has_any(cols, ["inchikey", "inchi_key", "InChIKey"]) and _has_any(cols, ["compound_id", "pubchem_cid", "chembl_id"]),
    ),
]


def _has_ingredient_id(cols: Set[str]) -> bool:
    c = {x.lower() for x in cols}
    return "ingredient_id" in c or "ing_id" in c


def _has_compound_key(cols: Set[str]) -> bool:
    return _has_any(cols, COMPOUND_KEY_COLUMNS)


def _has_gene_or_target(cols: Set[str]) -> bool:
    return _has_any(cols, GENE_TARGET_COLUMNS)


def _has_any(cols: Set[str], candidates: List[str]) -> bool:
    c = {x.lower() for x in cols}
    return any(k.lower() in c for k in candidates)


def _sample_load(path: Path, n: int = 50) -> Tuple[pd.DataFrame, Optional[int]]:
    """Load up to n rows; return (sample_df, total_rows or None if expensive). Uses dtype=str, low_memory=False for CSV to avoid DtypeWarnings (mixed columns e.g. fdb_id, name, role)."""
    if not path.exists():
        return pd.DataFrame(), None
    suf = path.suffix.lower()
    try:
        if suf == ".csv":
            df = pd.read_csv(path, nrows=n, dtype=str, low_memory=False)
            try:
                full = pd.read_csv(path, dtype=str, low_memory=False)
                total = len(full)
            except Exception:
                total = None
            return df, total
        if suf in (".parquet", ".pq"):
            df = pd.read_parquet(path)
            total = len(df)
            if total > n:
                df = df.head(n)
            return df, total
    except Exception as e:
        logger.debug("Sample load failed %s: %s", path, e)
        return pd.DataFrame(), None
    return pd.DataFrame(), None


def _tag_capabilities(columns: List[str]) -> List[str]:
    cols_set: Set[str] = set(columns)
    tags: List[str] = []
    for tag, pred in CAPABILITY_RULES:
        if pred(cols_set):
            tags.append(tag)
    return tags


def scan_processed_data(root: Path) -> pd.DataFrame:
    """
    Recursively scan data/processed/** for .parquet, .csv, .json.
    For parquet/csv: sample-load (head 50 rows), record path, rows (if cheap),
    columns, dtypes (approx), and capability tags.
    """
    root = Path(root).resolve()
    processed = root / "data" / "processed"
    if not processed.exists():
        logger.warning("Processed dir not found: %s", processed)
        return pd.DataFrame(columns=["path", "path_rel", "rows", "columns", "dtypes_str", "capabilities", "file_type"])

    rows_out: List[Dict[str, Any]] = []
    for ext in ("*.parquet", "*.pq", "*.csv"):
        for path in processed.rglob(ext):
            if not path.is_file():
                continue
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = str(path)
            file_type = path.suffix.lower().replace(".", "")
            sample, total = _sample_load(path, 50)
            if sample.empty and path.suffix.lower() in (".parquet", ".pq", ".csv"):
                rows_out.append({
                    "path": str(path),
                    "path_rel": rel,
                    "rows": None,
                    "columns": [],
                    "dtypes_str": "",
                    "capabilities": [],
                    "file_type": file_type,
                })
                continue
            columns = list(sample.columns)
            dtypes_str = ",".join(f"{c}:{str(sample.dtypes.get(c, ''))[:4]}" for c in columns[:15])
            if len(columns) > 15:
                dtypes_str += "..."
            capabilities = _tag_capabilities(columns)
            rows_out.append({
                "path": str(path),
                "path_rel": rel,
                "rows": total,
                "columns": columns,
                "dtypes_str": dtypes_str,
                "capabilities": capabilities,
                "file_type": file_type,
            })
    for path in processed.rglob("*.json"):
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        rows_out.append({
            "path": str(path),
            "path_rel": rel,
            "rows": None,
            "columns": [],
            "dtypes_str": "",
            "capabilities": [],
            "file_type": "json",
        })

    df = pd.DataFrame(rows_out)
    if not df.empty:
        logger.info("Scan found %s files under data/processed", len(df))
    return df


def _canonical_compound_gene_required_cols(df: pd.DataFrame) -> bool:
    """True if df has compound_id and at least one of gene, gene_symbol, gene_id."""
    cols = set(c.lower() for c in df.columns)
    if "compound_id" not in cols:
        return False
    return "gene" in cols or "gene_symbol" in cols or "gene_id" in cols


def _canonical_ingredient_compound_required_cols(df: pd.DataFrame) -> bool:
    """True if df has ingredient_id and compound_id."""
    cols = set(c.lower() for c in df.columns)
    return "ingredient_id" in cols and "compound_id" in cols


def pick_best_sources(scan_df: pd.DataFrame, repo_root: Path) -> Dict[str, Any]:
    """
    Select best available source files in priority order.
    Canonical files (CSV first) are top priority: compound_gene_canonical.csv, ingredient_compound_canonical.csv.
    Then: direct Ingredient→Compound from scan, Compound→Gene from scan, etc.
    Returns dict with chosen paths and reasoning.
    """
    repo_root = Path(repo_root).resolve()
    result: Dict[str, Any] = {
        "ingredient_compound": None,
        "ingredient_compound_reason": "",
        "ingredient_inchikey": None,
        "inchikey_compound": None,
        "compound_gene": None,
        "compound_gene_reason": "",
        "compound_metabolite": None,
        "food_compound": None,
        "ingredient_food_bridge": None,
        "reasoning": [],
    }

    canonical_dir = repo_root / "data" / "processed" / "canonical"

    # 0) Top priority: canonical compound_gene (CSV first, then parquet)
    for name in ("compound_gene_canonical.csv", "compound_gene_canonical.parquet"):
        path = canonical_dir / name
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, nrows=1, dtype=str, low_memory=False)
            else:
                df = pd.read_parquet(path, columns=None)
                if not df.empty:
                    df = df.head(1)
            if not df.empty and _canonical_compound_gene_required_cols(df):
                result["compound_gene"] = str(path.resolve())
                result["compound_gene_reason"] = "canonical"
                result["reasoning"].append(f"Compound-gene: canonical {path.name}")
                break
        except Exception as e:
            logger.debug("Canonical compound_gene check %s: %s", path, e)

    # 0b) Top priority: canonical ingredient_compound (CSV first)
    for name in ("ingredient_compound_canonical.csv", "ingredient_compound_links.csv", "ingredient_compound_links.parquet"):
        path = canonical_dir / name
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, nrows=1, dtype=str, low_memory=False)
            else:
                df = pd.read_parquet(path)
                if not df.empty:
                    df = df.head(1)
            if not df.empty and _canonical_ingredient_compound_required_cols(df):
                result["ingredient_compound"] = str(path.resolve())
                result["ingredient_compound_reason"] = "canonical"
                result["reasoning"].append(f"Ingredient-compound: canonical {path.name}")
                break
        except Exception as e:
            logger.debug("Canonical ingredient_compound check %s: %s", path, e)

    if scan_df.empty:
        if not result["reasoning"]:
            result["reasoning"].append("Scan empty; canonical preferred when present.")
        return result

    # Flatten capabilities: one row per (path, cap)
    has_cap = []
    for _, r in scan_df.iterrows():
        for cap in r.get("capabilities") or []:
            has_cap.append({"path": r["path"], "path_rel": r.get("path_rel", r["path"]), "capability": cap})
    cap_df = pd.DataFrame(has_cap)
    if cap_df.empty:
        if not result["reasoning"]:
            result["reasoning"].append("No capability tags on any file.")
        return result

    # 1) Direct INGREDIENT_COMPOUND (only if not already set from canonical)
    if result["ingredient_compound"] is None:
        direct = cap_df[cap_df["capability"] == "INGREDIENT_COMPOUND"]
        if not direct.empty:
            p = direct.iloc[0]["path"]
            result["ingredient_compound"] = p
            result["ingredient_compound_reason"] = "direct_ingredient_compound"
            result["reasoning"].append(f"Using direct INGREDIENT_COMPOUND: {direct.iloc[0]['path_rel']}")

    # 2) Two-step InChIKey: INGREDIENT_INCHIKEY + INCHIKEY_COMPOUND (only if ingredient_compound not yet set)
    if result["ingredient_compound"] is None:
        ing_ik = cap_df[cap_df["capability"] == "INGREDIENT_INCHIKEY"]
        ik_cmp = cap_df[cap_df["capability"] == "INCHIKEY_COMPOUND"]
        if not ing_ik.empty and not ik_cmp.empty:
            result["ingredient_inchikey"] = ing_ik.iloc[0]["path"]
            result["inchikey_compound"] = ik_cmp.iloc[0]["path"]
            result["ingredient_compound_reason"] = "inchikey_two_step"
            result["reasoning"].append(
                f"Using InChIKey bridge: ingredient_inchikey={ing_ik.iloc[0]['path_rel']}, inchikey_compound={ik_cmp.iloc[0]['path_rel']}"
            )
        else:
            if ing_ik.empty:
                result["reasoning"].append("No INGREDIENT_INCHIKEY file found.")
            if ik_cmp.empty:
                result["reasoning"].append("No INCHIKEY_COMPOUND file found.")

    # 3) Food→Compound + explicit bridge (only if we have a bridge file)
    food_cmp = cap_df[cap_df["capability"] == "FOOD_COMPOUND"]
    if not food_cmp.empty:
        result["food_compound"] = food_cmp.iloc[0]["path"]
        result["reasoning"].append(f"FOOD_COMPOUND available: {food_cmp.iloc[0]['path_rel']} (will use only if explicit ingredient↔food bridge exists)")

    # 4) Compound–gene: choose from scan only if not already set from canonical
    if result["compound_gene"] is None:
        cg = cap_df[cap_df["capability"].isin(["COMPOUND_GENE", "COMPOUND_TARGET"])]
        if not cg.empty:
            def _compound_gene_priority(path_rel: str) -> int:
                p = path_rel.replace("\\", "/").lower()
                if "canonical" in p and "compound_gene" in p:
                    return -1
                if "food_compound_gene_links" in p and "phase12" in p:
                    return 0
                if "compound_target_edges_bindingdb" in p and "phase16" in p:
                    return 1
                if "gene_chemical_links" in p:
                    return 2
                return 3
            cg = cg.copy()
            cg["_prio"] = cg["path_rel"].astype(str).apply(_compound_gene_priority)
            cg = cg.sort_values("_prio")
            result["compound_gene"] = cg.iloc[0]["path"]
            result["compound_gene_reason"] = cg.iloc[0]["capability"]
            result["reasoning"].append(f"Compound–gene source: {cg.iloc[0]['path_rel']}")

    # Compound–metabolite (for reference only; not used as ingredient→compound)
    cm = cap_df[cap_df["capability"] == "COMPOUND_METABOLITE"]
    if not cm.empty:
        result["compound_metabolite"] = cm.iloc[0]["path"]
        result["reasoning"].append(f"COMPOUND_METABOLITE (compound→metabolite only): {cm.iloc[0]['path_rel']}")

    return result


def write_audit(
    scan_df: pd.DataFrame,
    chosen: Dict[str, Any],
    output_dir: Path,
    run_id: str = "phase14",
) -> None:
    """Write data_audit.json and data_audit.csv to output_dir/reports/."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Chosen sources as flat rows for CSV
    audit_rows = []
    for key in ["ingredient_compound", "ingredient_inchikey", "inchikey_compound", "compound_gene", "compound_metabolite", "food_compound", "ingredient_food_bridge"]:
        val = chosen.get(key)
        if val:
            reason = chosen.get(f"{key}_reason") or (chosen.get("reasoning") or [])
            if isinstance(reason, list):
                reason = "; ".join(str(r) for r in reason[:3])
            audit_rows.append({"role": key, "path": val, "reason": reason})

    audit_report = {
        "run_id": run_id,
        "scan_file_count": 0 if scan_df.empty else len(scan_df),
        "chosen_sources": {k: v for k, v in chosen.items() if k != "reasoning" and v is not None and not k.endswith("_reason")},
        "reasoning": chosen.get("reasoning", []),
    }
    with open(report_dir / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    audit_csv = pd.DataFrame(audit_rows)
    if not audit_csv.empty:
        audit_csv.to_csv(report_dir / "data_audit.csv", index=False)
    else:
        pd.DataFrame(columns=["role", "path", "reason"]).to_csv(report_dir / "data_audit.csv", index=False)

    if not scan_df.empty:
        scan_export = scan_df.copy()
        if "columns" in scan_export.columns:
            scan_export["columns"] = scan_export["columns"].apply(lambda x: "|".join(x) if isinstance(x, list) else str(x))
        if "capabilities" in scan_export.columns:
            scan_export["capabilities"] = scan_export["capabilities"].apply(lambda x: "|".join(x) if isinstance(x, list) else str(x))
        scan_export.to_csv(report_dir / "data_scan.csv", index=False)

    logger.info("Wrote audit to %s", report_dir)
