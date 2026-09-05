"""
Phase16: Canonical compound master v2 — single table with max coverage.
Canonical compound_id = InChIKey when available. Crosswalk: coconut_id, fdb_id_norm, pharmgkb_id, chembl_id, bindingdb, pubchem_cid, smiles, name, normalized_name.
Sources: compound_master.csv seed, ingredient_compounds, coconut JSONs, pharmgkb_chemicals, bindingdb edges, chembl (local only).
Deterministic ID normalization; outputs parquet, csv, diagnostics JSON.
"""
from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "compound_id", "coconut_id", "coconut_base", "fdb_id_norm", "pharmgkb_id", "chembl_id",
    "bindingdb_id", "pubchem_cid", "smiles", "inchi", "name", "normalized_name", "source_files",
]

_FDB_PATTERN = re.compile(r"^(?:FDB[_\-]?|fdb[_\-]?|FOODB\s*:\s*)(\d+)$", re.IGNORECASE)
_COCONUT_PATTERN = re.compile(r"^(COCONUT_CNP\d+)(?:\.\d+)?$", re.IGNORECASE)
INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def normalize_inchikey(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper()
    if not s or s == "NAN":
        return None
    s_clean = s.replace("-", "")
    if len(s_clean) == 25:
        return f"{s_clean[:14]}-{s_clean[14:24]}-{s_clean[24]}"
    if len(s_clean) >= 27:
        return f"{s_clean[:14]}-{s_clean[14:24]}-{s_clean[24]}"
    return None


def looks_like_inchikey(s: str) -> bool:
    if not s or len(s) < 25:
        return False
    u = str(s).strip().upper().replace("-", "")
    return len(u) >= 25 and u[:14].isalpha() and u[14:24].isalnum() and u[24:].isalnum()


def normalize_fdb_id(x: Optional[str]) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    m = _FDB_PATTERN.match(s)
    if not m:
        return None
    num = m.group(1).lstrip("0") or "0"
    return f"FDB_{num}"


def normalize_coconut_id(x: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (full_id, base_without_version). COCONUT_CNP12345.1 -> (COCONUT_CNP12345.1, COCONUT_CNP12345)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None, None
    s = str(x).strip()
    if not s:
        return None, None
    m = _COCONUT_PATTERN.match(s)
    if m:
        base = m.group(1)
        return s, base
    if s.upper().startswith("COCONUT_"):
        return s, s.split(".")[0] if "." in s else s
    return None, None


def normalize_chembl_id(x: Optional[str]) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().upper()
    return s if s and s.startswith("CHEMBL") else None


def normalized_name(s: Optional[Any]) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    s = s.casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def _load_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False, dtype=str)
    except Exception as e:
        logger.warning("Load failed %s: %s", path, e)
        return None


def _col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _ingest_seed_master(canonical_dir: Path) -> pd.DataFrame:
    for name in ("compound_master.csv", "compound_master.parquet"):
        path = canonical_dir / name
        df = _load_df(path)
        if df is not None and not df.empty:
            rows = []
            ik_col = _col(df, ["inchikey", "inchi_key"])
            cid_col = _col(df, ["cid", "pubchem_cid"])
            fdb_col = _col(df, ["fdb_id_norm", "fdb_id"])
            chembl_col = _col(df, ["chembl_id"])
            bdb_col = _col(df, ["bindingdb_id"])
            name_col = _col(df, ["name", "compound_name"])
            smiles_col = _col(df, ["smiles", "canonical_smiles"])
            for _, r in df.iterrows():
                ik = normalize_inchikey(r.get(ik_col)) if ik_col else None
                cid = _safe_str(r.get(cid_col))
                if cid and cid.replace(".", "").isdigit():
                    cid = str(int(float(cid)))
                fdb = normalize_fdb_id(r.get(fdb_col)) if fdb_col else None
                chembl = normalize_chembl_id(r.get(chembl_col)) if chembl_col else None
                name = _safe_str(r.get(name_col)) if name_col else ""
                compound_id = ik if ik else ""
                if not compound_id and not fdb and not cid and not name:
                    continue
                rows.append({
                    "compound_id": compound_id or "",
                    "coconut_id": "",
                    "coconut_base": "",
                    "fdb_id_norm": fdb or "",
                    "pharmgkb_id": "",
                    "chembl_id": chembl or "",
                    "bindingdb_id": _safe_str(r.get(bdb_col)) if bdb_col else "",
                    "pubchem_cid": cid,
                    "smiles": _safe_str(r.get(smiles_col)) if smiles_col else "",
                    "inchi": "",
                    "name": name,
                    "normalized_name": normalized_name(name),
                    "source_files": name or "compound_master",
                })
            out = pd.DataFrame(rows)
            logger.info("Seed compound_master: %s rows from %s", len(out), name)
            return out
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _ingest_ingredient_compounds(processed: Path) -> pd.DataFrame:
    for pattern in ["**/ingredient_compound*.parquet", "**/ingredient_compound*.csv"]:
        for path in processed.glob(pattern):
            if "canonical" not in path.as_posix() and "compound_master" not in path.name:
                continue
            df = _load_df(path)
            if df is None or df.empty:
                continue
            cmp_col = _col(df, ["compound_id", "inchikey"])
            if not cmp_col:
                continue
            rows = []
            for v in df[cmp_col].dropna().astype(str).str.strip().unique():
                if not v:
                    continue
                ik = normalize_inchikey(v) if looks_like_inchikey(v) else None
                fdb = normalize_fdb_id(v) if _FDB_PATTERN.match(str(v).strip()) else None
                coco_full, coco_base = normalize_coconut_id(v)
                compound_id = ik or ""
                rows.append({
                    "compound_id": compound_id,
                    "coconut_id": coco_full or "",
                    "coconut_base": coco_base or "",
                    "fdb_id_norm": fdb or "",
                    "pharmgkb_id": "",
                    "chembl_id": "",
                    "bindingdb_id": "",
                    "pubchem_cid": "",
                    "smiles": "",
                    "inchi": "",
                    "name": "",
                    "normalized_name": "",
                    "source_files": "ingredient_compounds",
                })
            if rows:
                logger.info("Ingredient_compounds: %s unique from %s", len(rows), path.name)
                return pd.DataFrame(rows)
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _ingest_coconut_jsons(processed: Path) -> pd.DataFrame:
    for path in processed.rglob("inchikey_to_compound_id.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        rows = []
        for ik_raw, ids in data.items():
            ik = normalize_inchikey(ik_raw)
            if not ik:
                continue
            id_list = ids if isinstance(ids, list) else [ids]
            for item in id_list:
                if isinstance(item, dict):
                    cid = item.get("compound_id") or item.get("coconut_id") or ""
                else:
                    cid = str(item)
                coco_full, coco_base = normalize_coconut_id(cid)
                rows.append({
                    "compound_id": ik,
                    "coconut_id": coco_full or cid,
                    "coconut_base": coco_base or cid.split(".")[0] if "." in str(cid) else str(cid),
                    "fdb_id_norm": "",
                    "pharmgkb_id": "",
                    "chembl_id": "",
                    "bindingdb_id": "",
                    "pubchem_cid": "",
                    "smiles": "",
                    "inchi": "",
                    "name": "",
                    "normalized_name": "",
                    "source_files": "coconut_mapping",
                })
        if rows:
            logger.info("Coconut mapping: %s rows from %s", len(rows), path.name)
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _ingest_pharmgkb(repo_root: Path) -> pd.DataFrame:
    for path in Path(repo_root).rglob("pharmgkb_chemicals.parquet"):
        df = _load_df(path)
        if df is None or df.empty:
            continue
        cl = {c.lower().replace(" ", "_"): c for c in df.columns}
        pharmgkb_col = cl.get("pharmgkb_id") or cl.get("chemical_id") or cl.get("id")
        name_col = cl.get("name") or cl.get("compound_name") or cl.get("chemical_name")
        cid_col = cl.get("cid") or cl.get("pubchem_cid") or cl.get("pubchem_id")
        ik_col = cl.get("inchikey") or cl.get("inchi_key")
        smiles_col = cl.get("smiles") or cl.get("canonical_smiles")
        inchi_col = cl.get("inchi")
        rows = []
        for _, r in df.iterrows():
            ik = normalize_inchikey(r.get(ik_col)) if ik_col else None
            cid = _safe_str(r.get(cid_col))
            if cid and cid.replace(".", "").isdigit():
                cid = str(int(float(cid)))
            name = _safe_str(r.get(name_col)) if name_col else ""
            rows.append({
                "compound_id": ik or "",
                "coconut_id": "",
                "coconut_base": "",
                "fdb_id_norm": "",
                "pharmgkb_id": _safe_str(r.get(pharmgkb_col)) if pharmgkb_col else "",
                "chembl_id": "",
                "bindingdb_id": "",
                "pubchem_cid": cid,
                "smiles": _safe_str(r.get(smiles_col)) if smiles_col else "",
                "inchi": _safe_str(r.get(inchi_col)) if inchi_col else "",
                "name": name,
                "normalized_name": normalized_name(name),
                "source_files": "pharmgkb_chemicals",
            })
        if rows:
            logger.info("PharmGKB chemicals: %s rows", len(rows))
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _ingest_bindingdb(processed: Path) -> pd.DataFrame:
    path = processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if not path.exists():
        for p in processed.rglob("compound_target_edges_bindingdb.parquet"):
            path = p
            break
        else:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = _load_df(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    cmp_col = _col(df, ["compound_id"])
    if not cmp_col:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    rows = []
    for v in df[cmp_col].dropna().astype(str).str.strip().unique():
        if not v:
            continue
        ik = normalize_inchikey(v) if looks_like_inchikey(v) else None
        coco_full, coco_base = normalize_coconut_id(v)
        fdb = normalize_fdb_id(v)
        rows.append({
            "compound_id": ik or "",
            "coconut_id": coco_full or "",
            "coconut_base": coco_base or "",
            "fdb_id_norm": fdb or "",
            "pharmgkb_id": "",
            "chembl_id": "",
            "bindingdb_id": v if "BDB" in str(v).upper() or v.startswith("BDB") else "",
            "pubchem_cid": "",
            "smiles": "",
            "inchi": "",
            "name": "",
            "normalized_name": "",
            "source_files": "bindingdb_edges",
        })
    logger.info("BindingDB edges: %s unique compound_ids", len(rows))
    return pd.DataFrame(rows)


def _ingest_chembl(processed: Path) -> pd.DataFrame:
    for path in processed.rglob("chembl*.parquet"):
        if "target" in path.name.lower() and "compound" not in path.name.lower():
            continue
        df = _load_df(path)
        if df is None or df.empty:
            continue
        ik_col = _col(df, ["inchikey", "inchi_key"])
        chembl_col = _col(df, ["chembl_id", "compound_chembl_id"])
        cid_col = _col(df, ["cid", "pubchem_cid"])
        name_col = _col(df, ["name", "pref_name"])
        smiles_col = _col(df, ["smiles"])
        if not (ik_col or chembl_col):
            continue
        rows = []
        for _, r in df.iterrows():
            ik = normalize_inchikey(r.get(ik_col)) if ik_col else None
            chembl = normalize_chembl_id(r.get(chembl_col)) if chembl_col else None
            if not (ik or chembl):
                continue
            cid = _safe_str(r.get(cid_col))
            if cid and cid.replace(".", "").isdigit():
                cid = str(int(float(cid)))
            rows.append({
                "compound_id": ik or "",
                "coconut_id": "",
                "coconut_base": "",
                "fdb_id_norm": "",
                "pharmgkb_id": "",
                "chembl_id": chembl or "",
                "bindingdb_id": "",
                "pubchem_cid": cid,
                "smiles": _safe_str(r.get(smiles_col)) if smiles_col else "",
                "inchi": "",
                "name": _safe_str(r.get(name_col)) if name_col else "",
                "normalized_name": normalized_name(r.get(name_col)) if name_col else "",
                "source_files": "chembl",
            })
        if rows:
            logger.info("ChEMBL: %s rows from %s", len(rows), path.name)
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def merge_master_rows(sources: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge all source rows; deduplicate by compound_id (InChIKey); fill crosswalk from all sources."""
    all_rows: List[Dict[str, Any]] = []
    for df in sources:
        if df.empty:
            continue
        for _, r in df.iterrows():
            all_rows.append(r.to_dict())
    if not all_rows:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    by_ik: Dict[str, Dict[str, Any]] = {}
    for row in all_rows:
        ik = (row.get("compound_id") or "").strip()
        if ik and looks_like_inchikey(ik):
            ik = normalize_inchikey(ik) or ik
            if ik not in by_ik:
                by_ik[ik] = {k: (row.get(k) or "") for k in CANONICAL_COLUMNS}
                by_ik[ik]["compound_id"] = ik
            else:
                for k in CANONICAL_COLUMNS:
                    if k == "compound_id":
                        continue
                    val = row.get(k)
                    if val and not (str(by_ik[ik].get(k) or "").strip()):
                        by_ik[ik][k] = val
                by_ik[ik]["source_files"] = (by_ik[ik].get("source_files") or "") + ";" + (row.get("source_files") or "")
    # Build reverse maps from by_ik
    coconut_to_ik: Dict[str, str] = {}
    fdb_to_ik: Dict[str, str] = {}
    cid_to_ik: Dict[str, str] = {}
    chembl_to_ik: Dict[str, str] = {}
    pharmgkb_to_ik: Dict[str, str] = {}
    for ik, r in by_ik.items():
        for c in (r.get("coconut_base") or "").strip(), (r.get("coconut_id") or "").strip():
            if c:
                coconut_to_ik[c] = ik
        f = (r.get("fdb_id_norm") or "").strip()
        if f:
            fdb_to_ik[f] = ik
        c = (r.get("pubchem_cid") or "").strip()
        if c:
            cid_to_ik[c] = ik
        ch = (r.get("chembl_id") or "").strip()
        if ch:
            chembl_to_ik[ch] = ik
        p = (r.get("pharmgkb_id") or "").strip()
        if p:
            pharmgkb_to_ik[p] = ik
    # Resolve rows without inchikey and merge into by_ik
    for row in all_rows:
        ik = (row.get("compound_id") or "").strip()
        if ik and looks_like_inchikey(ik):
            continue
        ik = None
        coco = (row.get("coconut_base") or row.get("coconut_id") or "").strip()
        if coco:
            ik = coconut_to_ik.get(coco)
        if not ik and row.get("fdb_id_norm"):
            ik = fdb_to_ik.get((row.get("fdb_id_norm") or "").strip())
        if not ik and row.get("pubchem_cid"):
            ik = cid_to_ik.get(str(row.get("pubchem_cid")).strip())
        if not ik and row.get("chembl_id"):
            ik = chembl_to_ik.get((row.get("chembl_id") or "").strip())
        if not ik and row.get("pharmgkb_id"):
            ik = pharmgkb_to_ik.get((row.get("pharmgkb_id") or "").strip())
        if ik and ik in by_ik:
            for k in CANONICAL_COLUMNS:
                if k == "compound_id":
                    continue
                val = row.get(k)
                if val and not (str(by_ik[ik].get(k) or "").strip()):
                    by_ik[ik][k] = val
            by_ik[ik]["source_files"] = (by_ik[ik].get("source_files") or "") + ";" + (row.get("source_files") or "")
    out_rows = list(by_ik.values())
    # Add placeholder rows for compounds that never got an InChIKey (keep for resolver later)
    seen_other: Set[str] = set()
    for row in all_rows:
        ik = (row.get("compound_id") or "").strip()
        if ik and looks_like_inchikey(ik):
            continue
        key = (row.get("coconut_id") or row.get("coconut_base") or row.get("fdb_id_norm") or row.get("pubchem_cid") or row.get("chembl_id") or row.get("pharmgkb_id") or "").strip()
        if not key or key in seen_other:
            continue
        seen_other.add(key)
        out_rows.append({k: (row.get(k) or "") for k in CANONICAL_COLUMNS})
    return pd.DataFrame(out_rows)


def build_compound_master_v2(repo_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build master v2 from all local sources; return (df, diagnostics)."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"
    sources = [
        _ingest_seed_master(canonical_dir),
        _ingest_ingredient_compounds(processed),
        _ingest_coconut_jsons(processed),
        _ingest_pharmgkb(repo_root),
        _ingest_bindingdb(processed),
        _ingest_chembl(processed),
    ]
    merged = merge_master_rows(sources)
    diagnostics = {
        "n_rows": len(merged),
        "n_with_inchikey": int((merged["compound_id"].astype(str).str.strip() != "").sum()) if "compound_id" in merged.columns else 0,
        "n_with_pubchem_cid": int((merged["pubchem_cid"].astype(str).str.strip() != "").sum()) if "pubchem_cid" in merged.columns else 0,
        "n_with_fdb_id_norm": int((merged["fdb_id_norm"].astype(str).str.strip() != "").sum()) if "fdb_id_norm" in merged.columns else 0,
        "n_with_coconut_id": int((merged["coconut_id"].astype(str).str.strip() != "").sum()) if "coconut_id" in merged.columns else 0,
        "n_with_chembl_id": int((merged["chembl_id"].astype(str).str.strip() != "").sum()) if "chembl_id" in merged.columns else 0,
        "n_with_pharmgkb_id": int((merged["pharmgkb_id"].astype(str).str.strip() != "").sum()) if "pharmgkb_id" in merged.columns else 0,
        "sources_ingested": [s for s in ["seed", "ingredient_compounds", "coconut", "pharmgkb", "bindingdb", "chembl"]],
    }
    return merged, diagnostics


def write_compound_master_v2(repo_root: Path, output_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """Build, write parquet/csv/diagnostics; return (parquet_path, csv_path, diagnostics_path)."""
    df, diagnostics = build_compound_master_v2(repo_root)
    out_dir = output_dir or (Path(repo_root) / "data" / "processed" / "canonical")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_path = out_dir / "compound_master_v2.parquet"
    csv_path = out_dir / "compound_master_v2.csv"
    diag_path = out_dir / "compound_master_v2_diagnostics.json"
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False)
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Wrote compound_master_v2: %s rows, %s with inchikey", len(df), diagnostics.get("n_with_inchikey"))
    return pq_path, csv_path, diag_path
