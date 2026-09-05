"""
Phase14: Compound identity resolution layer.

- Scan data/processed for files containing compound-related columns.
- Build compound_master (inchikey, cid, fdb_id, chembl_id, bindingdb_id, coconut_id, smiles, name, source_file).
- Harmonized mapping: FoodDB_ID / Coconut ID -> canonical InChIKey (direct, CID, SMILES, name fallbacks).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Column names (case-insensitive) that indicate compound-related data
COMPOUND_SCAN_KEYS = [
    "inchikey", "inchi_key", "cid", "pubchem", "pubchem_cid",
    "smiles", "canonical_smiles", "chembl_id", "bindingdb_id",
    "compound_name", "compound_id", "synonym", "name", "fdb_id",
]

# InChIKey: 27 chars with two dashes
INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def _safe_str(val: Any) -> str:
    """Coerce value to non-NaN string for CSV/DataFrame values."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s == "nan" else s


def canonicalize_coconut_id(val: Any) -> Optional[str]:
    """Canonicalize Coconut ID: strip whitespace, keep exact 'COCONUT_CNP...' form."""
    s = _safe_str(val)
    if not s:
        return None
    s = s.strip()
    if "COCONUT_" in s.upper():
        return s
    return None


def _inchikey_from_smiles_or_inchi(smiles: Optional[str], inchi: Optional[str]) -> Optional[str]:
    """If RDKit is available, compute InChIKey from SMILES or InChI. Otherwise return None."""
    try:
        from rdkit import Chem
        mol = None
        inchi_s = _safe_str(inchi)
        smiles_s = _safe_str(smiles)
        if inchi_s:
            try:
                mol = Chem.MolFromInchi(inchi_s)
            except Exception:
                pass
        if mol is None and smiles_s:
            try:
                mol = Chem.MolFromSmiles(smiles_s)
            except Exception:
                pass
        if mol is not None:
            return Chem.inchi.MolToInchiKey(mol) or None
    except ImportError:
        pass
    return None


def normalize_inchikey(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper()
    if not s:
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


# FooDB ID patterns: FDB022741, FDB_22741, FDB-22741, fdb022741, FOODB:22741
_FDB_PATTERN = re.compile(
    r"^(?:FDB[_\-]?|fdb[_\-]?|FOODB\s*:\s*)(\d+)$",
    re.IGNORECASE,
)


def normalize_fdb_id(x: Optional[str]) -> Optional[str]:
    """
    Normalize FooDB-style IDs to canonical form FDB_<integer without leading zeros>.

    Accepts: FDB022741, FDB_22741, FDB-22741, fdb022741, FOODB:22741
    Returns: FDB_22741, FDB_4 (for FDB000004), or None for non-FDB IDs.
    """
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


def looks_like_fdb_id(s: str) -> bool:
    return normalize_fdb_id(s) is not None


def fdb_id_to_numeric(fdb: str) -> Optional[str]:
    """FDB_123 or FDB123 -> '123'. Uses normalize_fdb_id then strips prefix."""
    norm = normalize_fdb_id(fdb)
    if norm is None:
        return None
    return norm.replace("FDB_", "")


def normalize_name(s: Optional[Any]) -> str:
    """Casefold, strip, collapse punctuation to space, collapse multiple spaces for reliable matching."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    s = s.casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def scan_processed_for_compound_columns(repo_root: Path) -> List[Dict[str, Any]]:
    """
    Scan data/processed for CSV/Parquet/JSON files that contain any compound-related column.
    Returns list of { path, path_str, columns_found, sample_columns }.
    """
    processed = Path(repo_root) / "data" / "processed"
    if not processed.exists():
        return []

    keys_lower = {k.lower() for k in COMPOUND_SCAN_KEYS}
    results = []

    for ext in ("*.csv", "*.parquet", "*.pq"):
        for path in processed.rglob(ext):
            if not path.is_file():
                continue
            try:
                if ext == "*.csv":
                    df = pd.read_csv(path, nrows=0)
                    cols = list(df.columns)
                else:
                    try:
                        import pyarrow.parquet as pq
                        tbl = pq.read_table(path, columns=[])
                        cols = tbl.schema.names
                    except Exception:
                        df = pd.read_parquet(path)
                        cols = list(df.columns) if hasattr(df, "columns") else []
                found = [c for c in cols if c.lower() in keys_lower or any(k in c.lower() for k in ("inchikey", "cid", "smiles", "chembl", "bindingdb", "compound", "fdb", "synonym", "pubchem"))]
                if found:
                    results.append({
                        "path": str(path),
                        "path_str": str(path.relative_to(processed) if processed in path.parents else path),
                        "columns_found": found,
                        "sample_columns": cols[:15],
                    })
            except Exception:
                continue

    # JSON files that are known compound lookups
    for path in processed.rglob("*.json"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(x in name for x in ("compound", "inchikey", "registry", "coconut", "lookup")):
            results.append({
                "path": str(path),
                "path_str": str(path.relative_to(processed) if processed in path.parents else path),
                "columns_found": ["json_key_value"],
                "sample_columns": [],
            })

    return results


def load_csv_or_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    suf = path.suffix.lower()
    try:
        if suf == ".csv":
            return pd.read_csv(path, low_memory=False)
        if suf in (".parquet", ".pq"):
            return pd.read_parquet(path)
    except Exception:
        pass
    return None


HARVEST_FDB_COLUMNS = ["fdb_id", "foodb_id", "compound_id"]
HARVEST_VALUE_COLUMNS = [
    "inchikey", "inchi_key", "InChIKey",
    "cid", "pubchem_cid", "PubChemCID",
    "smiles", "canonical_smiles",
    "compound_name", "name",
]


def harvest_fdb_inchikey_pairs(repo_root: Path) -> pd.DataFrame:
    """
    Scan data/processed for CSV/Parquet that contain FDB-like column AND (inchikey/cid/smiles/name).
    Return DataFrame with: fdb_id_raw, fdb_id_norm, inchikey, cid, smiles, name, source_file.
    Drops rows where fdb_id_norm is null and inchikey/cid/name are all null.
    """
    processed = Path(repo_root) / "data" / "processed"
    if not processed.exists():
        return pd.DataFrame(columns=["fdb_id_raw", "fdb_id_norm", "inchikey", "cid", "smiles", "name", "source_file"])

    fdb_cols_lower = {c.lower() for c in HARVEST_FDB_COLUMNS}
    value_cols_lower = {c.lower().replace(" ", "_"): c for c in HARVEST_VALUE_COLUMNS}
    # also allow partial match: column containing "inchikey", "cid", "pubchem", "smiles", "name", "compound_name"
    value_keys = ["inchikey", "inchi_key", "cid", "pubchem", "smiles", "canonical_smiles", "compound_name", "name"]

    out_rows: List[Dict[str, Any]] = []
    # Limit to dirs that may contain FDB+InChIKey/CID/name (skip e.g. phase13 large outputs)
    allowed_dirs = ("phase12_genetics", "phase14_chemical_identity", "phase14_mediation", "phase15_coconut", "phase16_bindingdb", "canonical", "phase17_reaggregation")
    for ext in ("*.csv", "*.parquet", "*.pq"):
        for path in processed.rglob(ext):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(processed)
                if not any(d in rel.parts for d in allowed_dirs):
                    continue
            except ValueError:
                continue
            if path.name.startswith("ingredient_compound_links") or path.name.startswith("compound_gene_links"):
                continue
            try:
                df = load_csv_or_parquet(path)
                if df is None or df.empty:
                    continue
                cols = list(df.columns)
                cl = {c.lower().replace(" ", "_"): c for c in cols}
                # Need at least one FDB-like column
                fdb_col = None
                for k in ("fdb_id", "foodb_id", "compound_id"):
                    if k in cl:
                        fdb_col = cl[k]
                        break
                if not fdb_col:
                    continue
                # And at least one value column
                ik_col = cl.get("inchikey") or cl.get("inchi_key")
                cid_col = cl.get("cid") or cl.get("pubchem_cid") or cl.get("pubchemcid")
                smiles_col = cl.get("smiles") or cl.get("canonical_smiles")
                name_col = cl.get("compound_name") or cl.get("name")
                if not any([ik_col, cid_col, smiles_col, name_col]):
                    continue
                source_name = path.name
                for _, r in df.iterrows():
                    raw = r.get(fdb_col)
                    if pd.isna(raw) or not str(raw).strip():
                        continue
                    raw = str(raw).strip()
                    norm = normalize_fdb_id(raw)
                    if not norm and not looks_like_fdb_id(raw):
                        continue
                    if not norm:
                        norm = ""
                    ik = None
                    if ik_col:
                        v = r.get(ik_col)
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            ik = normalize_inchikey(str(v).strip())
                    cid_val = None
                    if cid_col:
                        v = r.get(cid_col)
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            cid_val = str(int(v)) if isinstance(v, (int, float)) else str(v).strip()
                    sm = None
                    if smiles_col:
                        v = r.get(smiles_col)
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            sm = str(v).strip()[:500]
                    nm = None
                    if name_col:
                        v = r.get(name_col)
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            nm = str(v).strip()[:300]
                    if not norm and not ik and not cid_val and not nm:
                        continue
                    out_rows.append({
                        "fdb_id_raw": raw,
                        "fdb_id_norm": norm or normalize_fdb_id(raw),
                        "inchikey": ik or "",
                        "cid": cid_val or "",
                        "smiles": sm or "",
                        "name": nm or "",
                        "source_file": source_name,
                    })
            except Exception:
                continue

    if not out_rows:
        return pd.DataFrame(columns=["fdb_id_raw", "fdb_id_norm", "inchikey", "cid", "smiles", "name", "source_file"])
    h = pd.DataFrame(out_rows)
    h["fdb_id_norm"] = h["fdb_id_raw"].apply(lambda x: normalize_fdb_id(x) or "")
    h = h[h["fdb_id_norm"].notna() & (h["fdb_id_norm"] != "")]
    h = h.drop_duplicates(subset=["fdb_id_norm", "inchikey", "cid", "name"], keep="first")
    return h


def build_compound_master_from_sources(
    repo_root: Path,
    scan_results: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Build compound_master with columns:
    compound_master_id, inchikey, cid, fdb_id, chembl_id, bindingdb_id, smiles, canonical_smiles, name, source_file
    """
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"

    rows: List[Dict[str, Any]] = []
    seen_inchikey: Set[str] = set()
    master_id = 0

    def _add(inchikey: Optional[str], cid: Optional[str], fdb_id: Optional[str],
             chembl_id: Optional[str], bindingdb_id: Optional[str],
             coconut_id: Optional[str],
             smiles: Optional[str], canonical_smiles: Optional[str],
             name: Optional[str], source_file: str) -> None:
        nonlocal master_id
        fdb_raw = _safe_str(fdb_id)
        fdb_norm = normalize_fdb_id(fdb_raw) if fdb_raw else ""
        coco = canonicalize_coconut_id(coconut_id) if coconut_id else ""
        if not any([inchikey, cid, fdb_raw, fdb_norm, chembl_id, bindingdb_id, coco]):
            return
        master_id += 1
        row = {
            "compound_master_id": f"CMP_{master_id:08d}",
            "inchikey": (inchikey or "").strip() if inchikey else "",
            "cid": _safe_str(cid),
            "fdb_id": fdb_norm or fdb_raw,
            "fdb_id_raw": fdb_raw,
            "fdb_id_norm": fdb_norm,
            "chembl_id": _safe_str(chembl_id),
            "bindingdb_id": _safe_str(bindingdb_id),
            "coconut_id": coco or "",
            "smiles": (_safe_str(smiles))[:500],
            "canonical_smiles": (_safe_str(canonical_smiles))[:500],
            "name": (_safe_str(name))[:300],
            "source_file": source_file,
        }
        rows.append(row)
        if inchikey:
            seen_inchikey.add(inchikey)

    # 1) compound_registry_lookup.json (InChIKey -> { compound_id, name })
    registry_path = processed / "phase14_chemical_identity" / "compound_registry_lookup.json"
    if registry_path.exists():
        try:
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
            for ik, data in registry.items():
                ik_norm = normalize_inchikey(ik)
                if not ik_norm:
                    continue
                cid_val = data.get("compound_id") if isinstance(data, dict) else None
                # Canonical FDB_<int without leading zeros>
                fdb_val = normalize_fdb_id(f"FDB_{cid_val}") if cid_val is not None else ""
                name_val = data.get("name") if isinstance(data, dict) else None
                _add(ik_norm, None, fdb_val or None, None, None, None, None, None, name_val, "compound_registry_lookup.json")
        except Exception:
            pass

    # 2) inchikey_to_compound_id.json (InChIKey -> numeric or COCONUT_CNP...); add rows and build coconut_id -> inchikey for later fill
    coconut_ik_path = processed / "phase15_coconut" / "inchikey_to_compound_id.json"
    coconut_to_ik: Dict[str, str] = {}
    if coconut_ik_path.exists():
        try:
            with open(coconut_ik_path, encoding="utf-8") as f:
                ik2id = json.load(f)
            for ik, raw_id in ik2id.items():
                ik_norm = normalize_inchikey(ik)
                if not ik_norm:
                    continue
                if raw_id is None:
                    continue
                s = _safe_str(raw_id)
                if not s:
                    continue
                coco = canonicalize_coconut_id(s)
                if coco:
                    coconut_to_ik[coco] = ik_norm
                    coconut_to_ik[coco.strip()] = ik_norm
                    base = coco.split(".")[0] if "." in coco else coco
                    if base and base not in coconut_to_ik:
                        coconut_to_ik[base] = ik_norm
                    _add(ik_norm, None, None, None, None, coco, None, None, None, "inchikey_to_compound_id.json")
                else:
                    fdb_val = normalize_fdb_id(f"FDB_{raw_id}") if raw_id is not None else None
                    _add(ik_norm, None, fdb_val, None, None, None, None, None, None, "inchikey_to_compound_id.json")
        except Exception:
            pass

    # 3) ingredient_compound_links: do not add every unique to master (mapping resolved via registry/coconut)

    # 4) food_compound_gene_links.parquet (compound_id FDB022741-style, compound_name); normalize to fdb_id_norm
    fcg = processed / "phase12_genetics" / "food_compound_gene_links.parquet"
    if fcg.exists():
        df = load_csv_or_parquet(fcg)
        if df is not None and not df.empty:
            cmp_col = next((c for c in df.columns if "compound_id" in c.lower() or c.lower() == "compound_id"), None)
            name_col = next((c for c in df.columns if "compound_name" in c.lower() or "name" in c.lower()), None)
            if cmp_col:
                # Normalize compound_id when FDB-style for consistent join (fdb_id_raw = v, fdb_id_norm = normalize_fdb_id(v))
                for _, r in df[[cmp_col] + ([name_col] if name_col else [])].drop_duplicates().iterrows():
                    v = r.get(cmp_col)
                    if pd.isna(v) or not str(v).strip():
                        continue
                    v = str(v).strip()
                    nm = str(r.get(name_col, "") or "").strip() if name_col else ""
                    if looks_like_fdb_id(v):
                        _add(None, None, v, None, None, None, None, None, nm or None, "food_compound_gene_links.parquet")
                    elif looks_like_inchikey(v):
                        _add(normalize_inchikey(v), None, None, None, None, None, None, None, nm or None, "food_compound_gene_links.parquet")

    # 5) compound_target_edges_bindingdb.parquet (compound_id often COCONUT_*)
    bdb = processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if bdb.exists():
        df = load_csv_or_parquet(bdb)
        if df is not None and not df.empty and "compound_id" in df.columns:
            for v in df["compound_id"].dropna().astype(str).str.strip().unique():
                if not v:
                    continue
                if v.startswith("COCONUT_"):
                    _add(None, None, None, None, v, v, None, None, None, "compound_target_edges_bindingdb.parquet")
                elif looks_like_inchikey(v):
                    _add(normalize_inchikey(v), None, None, None, v, None, None, None, None, "compound_target_edges_bindingdb.parquet")

    # 6) compound_metabolite_links if present
    for candidate in [
        processed / "compound_metabolite_links.parquet",
        processed / "canonical" / "compound_metabolite_links.parquet",
    ]:
        if candidate.exists():
            df = load_csv_or_parquet(candidate)
            if df is not None and not df.empty:
                cl = {c.lower(): c for c in df.columns}
                for _, r in df.iterrows():
                    ik = None
                    for k in ("inchikey", "inchi_key"):
                        if k in cl and r.get(cl[k]) is not None:
                            ik = normalize_inchikey(r[cl[k]])
                            break
                    cid = cl.get("cid") or cl.get("pubchem_cid")
                    cid_val = str(r[cid]).strip() if cid and r.get(cl[cid]) is not None else None
                    fdb = cl.get("fdb_id") or (cl.get("compound_id") if "compound_id" in cl else None)
                    fdb_val = str(r[cl[fdb]]).strip() if fdb and r.get(cl[fdb]) is not None else None
                    if ik or cid_val or fdb_val:
                        _add(ik, cid_val, fdb_val if fdb_val and looks_like_fdb_id(fdb_val) else None,
                             None, None, None, None, None, None, candidate.name)
            break

    # 7) PharmGKB chemicals: Name, PubChem Compound Identifiers, SMILES, InChI; derive InChIKey (explicit else RDKit else null)
    _rdkit_warned: List[bool] = []  # one-element list to allow closure to mutate
    phase12 = processed / "phase12_genetics"
    pharmgkb_path = phase12 / "pharmgkb_chemicals.parquet"
    for path in ([pharmgkb_path] if pharmgkb_path.exists() else []) + list(phase12.glob("*pharmgkb*.parquet")) + list(phase12.glob("*pharmgkb*.csv")):
        if not path.is_file():
            continue
        df = load_csv_or_parquet(path)
        if df is None or df.empty:
            continue
        cl = {c.lower().replace(" ", "_"): c for c in df.columns}
        name_col = cl.get("name") or cl.get("compound_name") or cl.get("chemical_name") or cl.get("drug_name")
        cid_col = cl.get("pubchem_compound_identifiers") or cl.get("cid") or cl.get("pubchem_cid") or cl.get("pubchem_id")
        smiles_col = cl.get("smiles")
        inchi_col = cl.get("inchi")
        ik_col = cl.get("inchikey") or cl.get("inchi_key")
        source_label = "phase12_genetics/pharmgkb_chemicals.parquet" if "pharmgkb_chemicals" in path.name else path.name
        if not name_col:
            continue
        for _, r in df.iterrows():
            nm = r.get(name_col)
            if pd.isna(nm) or not _safe_str(nm):
                continue
            name_norm = normalize_name(nm)
            if not name_norm:
                continue
            cid_val = None
            if cid_col:
                v = r.get(cid_col)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    try:
                        if isinstance(v, (int, float)):
                            cid_val = str(int(v))
                        else:
                            s = _safe_str(v)
                            if s.isdigit():
                                cid_val = s
                            else:
                                cid_val = s
                    except (ValueError, TypeError):
                        cid_val = _safe_str(v) or None
            ik_val = None
            if ik_col:
                v = r.get(ik_col)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    ik_val = normalize_inchikey(_safe_str(v))
            if not ik_val and (smiles_col or inchi_col):
                sm = r.get(smiles_col) if smiles_col else None
                inch = r.get(inchi_col) if inchi_col else None
                ik_val = _inchikey_from_smiles_or_inchi(sm, inch)
                if not ik_val and not _rdkit_warned:
                    try:
                        from rdkit import Chem  # noqa: F401
                    except ImportError:
                        logger.warning("RDKit not installed; InChIKey from SMILES/InChI skipped. Pipeline remains functional.")
                        _rdkit_warned.append(True)
            _add(ik_val, cid_val, None, None, None, None, None, None, name_norm, source_label)

    # 8) Harvest FDB+InChIKey/CID/smiles/name from all processed files (long-ID bridge)
    harvested = harvest_fdb_inchikey_pairs(repo_root)
    if not harvested.empty:
        for _, r in harvested.iterrows():
            fdb_raw = (r.get("fdb_id_raw") or "").strip()
            fdb_norm = (r.get("fdb_id_norm") or "").strip()
            if not fdb_norm:
                fdb_norm = normalize_fdb_id(fdb_raw) or ""
            if not fdb_norm and not fdb_raw:
                continue
            ik = (r.get("inchikey") or "").strip() or None
            if ik and len(ik) < 25:
                ik = None
            cid_val = (r.get("cid") or "").strip() or None
            sm = (r.get("smiles") or "").strip() or None
            nm = (r.get("name") or "").strip() or None
            _add(ik, cid_val, fdb_raw or fdb_norm, None, None, None, sm, sm, nm, r.get("source_file", "harvested"))

    if not rows:
        return pd.DataFrame(columns=[
            "compound_master_id", "inchikey", "cid", "fdb_id", "fdb_id_raw", "fdb_id_norm",
            "chembl_id", "bindingdb_id", "coconut_id", "smiles", "canonical_smiles", "name", "source_file",
        ])

    master = pd.DataFrame(rows)

    # Ensure coconut_id column exists
    if "coconut_id" not in master.columns:
        master["coconut_id"] = ""

    # Fill inchikey from coconut_to_ik for rows that have coconut_id but no long inchikey
    for idx, r in master.iterrows():
        ik = _safe_str(r.get("inchikey"))
        if ik and len(ik) >= 25:
            continue
        coco = _safe_str(r.get("coconut_id"))
        if not coco:
            continue
        resolved = coconut_to_ik.get(coco) or coconut_to_ik.get(coco.strip())
        if not resolved and "." in coco:
            resolved = coconut_to_ik.get(coco.split(".")[0])
        if resolved:
            master.at[idx, "inchikey"] = resolved
            coconut_to_ik[coco] = resolved
            if "." in coco and coco.split(".")[0] not in coconut_to_ik:
                coconut_to_ik[coco.split(".")[0]] = resolved

    # Ensure fdb_id_norm is set from fdb_id when missing
    if "fdb_id_norm" not in master.columns:
        master["fdb_id_norm"] = master.get("fdb_id", pd.Series(dtype=object)).apply(
            lambda x: normalize_fdb_id(x) if pd.notna(x) and str(x).strip() else ""
        )
    if "fdb_id_raw" not in master.columns:
        master["fdb_id_raw"] = master.get("fdb_id", pd.Series(dtype=object)).fillna("").astype(str).str.strip()

    # Merge rows that share fdb_id_norm or inchikey to fill in inchikey from registry (use normalized for joins)
    fdb_to_ik: Dict[str, str] = {}
    for _, r in master.iterrows():
        ik = (r.get("inchikey") or "").strip()
        fdb_norm = (r.get("fdb_id_norm") or "").strip() or (r.get("fdb_id") or "").strip()
        fdb_raw = (r.get("fdb_id_raw") or "").strip()
        if ik and fdb_norm:
            fdb_to_ik[fdb_norm] = ik
        if ik and fdb_raw and fdb_raw != fdb_norm:
            fdb_to_ik[fdb_raw] = ik
        num = fdb_id_to_numeric(fdb_norm or fdb_raw)
        if num and ik:
            fdb_to_ik[f"FDB_{num}"] = ik
            fdb_to_ik[f"FDB{num}"] = ik

    # Fill inchikey in master from registry/coconut/harvested where we have fdb_id
    def _fill_ik(r):
        ik = (r.get("inchikey") or "").strip()
        if ik and len(ik) >= 25:
            return ik
        fdb_norm = (r.get("fdb_id_norm") or "").strip() or (r.get("fdb_id") or "").strip()
        fdb_raw = (r.get("fdb_id_raw") or "").strip()
        return fdb_to_ik.get(fdb_norm) or fdb_to_ik.get(fdb_raw) or (
            fdb_to_ik.get(f"FDB_{fdb_id_to_numeric(fdb_norm or fdb_raw)}") if fdb_id_to_numeric(fdb_norm or fdb_raw) else ""
        )
    master["inchikey"] = master.apply(_fill_ik, axis=1)

    # B) CID->InChIKey fallback from repo only: build from rows that have both, then fill missing
    cid_to_ik: Dict[str, str] = {}
    for _, r in master.iterrows():
        ik = (r.get("inchikey") or "").strip()
        cid = (r.get("cid") or "").strip()
        if ik and len(ik) >= 25 and cid:
            cid_to_ik[cid] = ik
    for idx, r in master.iterrows():
        if (r.get("inchikey") or "").strip() and len((r.get("inchikey") or "").strip()) >= 25:
            continue
        cid = (r.get("cid") or "").strip()
        if cid and cid in cid_to_ik:
            master.at[idx, "inchikey"] = cid_to_ik[cid]

    return master


def build_fdb_to_inchikey_with_fallbacks(
    repo_root: Path,
    compound_master: pd.DataFrame,
    report_dir: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, str], pd.DataFrame]:
    """
    Full FDB -> InChIKey with direct, registry, CID, name exact/fuzzy (fuzzy threshold >= 95).
    If report_dir is set, writes name_fallback_matches.csv (score, name_src, name_matched, inchikey).
    Returns (fdb_to_ik, resolution_method, name_fallback_matches_df).
    """
    processed = Path(repo_root) / "data" / "processed"
    name_fallback_rows: List[Dict[str, Any]] = []
    registry_path = processed / "phase14_chemical_identity" / "compound_registry_lookup.json"
    coconut_ik_path = processed / "phase15_coconut" / "inchikey_to_compound_id.json"

    fdb_to_ik: Dict[str, str] = {}
    resolution_method: Dict[str, str] = {}

    # 1) From compound_master: use fdb_id_norm and coconut_id for joins; register raw variants too
    for _, r in compound_master.iterrows():
        ik = _safe_str(r.get("inchikey"))
        fdb_norm = _safe_str(r.get("fdb_id_norm")) or _safe_str(r.get("fdb_id"))
        fdb_raw = _safe_str(r.get("fdb_id_raw"))
        coco = _safe_str(r.get("coconut_id"))
        if ik and len(ik) >= 25:
            for key in (fdb_norm, fdb_raw, f"FDB_{fdb_id_to_numeric(fdb_norm or fdb_raw)}", f"FDB{fdb_id_to_numeric(fdb_norm or fdb_raw)}"):
                if key and key not in fdb_to_ik:
                    fdb_to_ik[key] = ik
                    resolution_method[key] = "direct"
            if coco:
                for key in (coco, coco.strip(), coco.split(".")[0] if "." in coco else coco):
                    if key and key not in fdb_to_ik:
                        fdb_to_ik[key] = ik
                        resolution_method[key] = "direct"

    # 2) Registry: InChIKey -> compound_id => canonical FDB_<id> -> InChIKey (and FDB<id> variant)
    if registry_path.exists():
        try:
            with open(registry_path, encoding="utf-8") as f:
                reg = json.load(f)
            for ik, data in reg.items():
                ik_norm = normalize_inchikey(ik)
                if not ik_norm:
                    continue
                cid_val = data.get("compound_id") if isinstance(data, dict) else None
                if cid_val is not None:
                    fdb_canon = normalize_fdb_id(f"FDB_{cid_val}")
                    for fdb_key in (fdb_canon, f"FDB_{cid_val}", f"FDB{cid_val}"):
                        if fdb_key and fdb_key not in fdb_to_ik:
                            fdb_to_ik[fdb_key] = ik_norm
                            resolution_method[fdb_key] = "direct"
        except Exception:
            pass

    # 3) inchikey_to_compound_id.json: InChIKey -> numeric => FDB_<id> -> InChIKey
    if coconut_ik_path.exists():
        try:
            with open(coconut_ik_path, encoding="utf-8") as f:
                ik2id = json.load(f)
            for ik, num_id in ik2id.items():
                ik_norm = normalize_inchikey(ik)
                if not ik_norm:
                    continue
                fdb_canon = normalize_fdb_id(f"FDB_{num_id}")
                for fdb_key in (fdb_canon, f"FDB_{num_id}", f"FDB{num_id}"):
                    if fdb_key and fdb_key not in fdb_to_ik:
                        fdb_to_ik[fdb_key] = ik_norm
                        resolution_method[fdb_key] = "direct"
        except Exception:
            pass

    # 4) CID fallback: compound_master has cid; use cid <-> inchikey; join on fdb_id_norm
    cid_to_ik = {}
    for _, r in compound_master.iterrows():
        ik = _safe_str(r.get("inchikey"))
        cid = _safe_str(r.get("cid"))
        if ik and len(ik) >= 25 and cid:
            cid_to_ik[cid] = ik
    for _, r in compound_master.iterrows():
        fdb = _safe_str(r.get("fdb_id_norm")) or _safe_str(r.get("fdb_id"))
        cid = _safe_str(r.get("cid"))
        if fdb and fdb not in fdb_to_ik and cid and cid in cid_to_ik:
            fdb_to_ik[fdb] = cid_to_ik[cid]
            resolution_method[fdb] = "cid"

    # 5) Name fuzzy fallback: only for FDB still unresolved; match name -> inchikey from master, then fdb -> name
    name_to_ik: Dict[str, str] = {}
    for _, r in compound_master.iterrows():
        ik = _safe_str(r.get("inchikey"))
        name = _safe_str(r.get("name"))
        if ik and len(ik) >= 25 and name:
            name_norm = " ".join(name.lower().split())
            if name_norm and name_norm not in name_to_ik:
                name_to_ik[name_norm] = ik
    fdb_to_name: Dict[str, str] = {}
    for _, r in compound_master.iterrows():
        fdb = _safe_str(r.get("fdb_id_norm")) or _safe_str(r.get("fdb_id"))
        name = _safe_str(r.get("name"))
        if fdb and name:
            fdb_to_name[fdb] = " ".join(name.lower().split())
    for fdb, name_norm in fdb_to_name.items():
        if fdb in fdb_to_ik:
            continue
        if name_norm in name_to_ik:
            ik = name_to_ik[name_norm]
            fdb_to_ik[fdb] = ik
            resolution_method[fdb] = "name_exact"
            name_fallback_rows.append({"score": 100, "name_src": name_norm, "name_matched": name_norm, "inchikey": ik})
            continue
        # Fuzzy: threshold >= 95 (internal only)
        try:
            from rapidfuzz import fuzz
            best_ik = None
            best_score = 0
            best_name = None
            for other_name, ik in name_to_ik.items():
                sc = fuzz.ratio(name_norm, other_name)
                if sc >= 95 and sc > best_score:
                    best_score = sc
                    best_ik = ik
                    best_name = other_name
            if best_ik:
                fdb_to_ik[fdb] = best_ik
                resolution_method[fdb] = "name_fuzzy"
                name_fallback_rows.append({"score": best_score, "name_src": name_norm, "name_matched": best_name or "", "inchikey": best_ik})
        except ImportError:
            break

    name_fallback_df = pd.DataFrame(name_fallback_rows)
    if report_dir and not name_fallback_df.empty:
        report_dir.mkdir(parents=True, exist_ok=True)
        name_fallback_df.to_csv(report_dir / "name_fallback_matches.csv", index=False)
    return fdb_to_ik, resolution_method, name_fallback_df


def write_unresolved_fdb_report(
    cg_raw_compound_series: pd.Series,
    compound_master: pd.DataFrame,
    fdb_to_ik: Dict[str, str],
    report_path: Path,
) -> pd.DataFrame:
    """
    Write reports/unresolved_fdb_ids.csv: fdb_id_norm, count_in_compound_gene, has_master_row,
    inchikey_present, cid_present, name_present, top_source_files_seen.
    Highlights long IDs from compound_gene that remain unmapped.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    raw = cg_raw_compound_series.dropna().astype(str).str.strip()
    fdb_norms = raw.apply(normalize_fdb_id)
    fdb_counts = fdb_norms[fdb_norms.notna() & (fdb_norms != "")].value_counts()
    key_col = compound_master["fdb_id_norm"].fillna(compound_master.get("fdb_id", ""))
    key_col = key_col.astype(str).str.strip()
    master_fdb = set(key_col[key_col != ""])
    agg = compound_master.copy()
    agg["_fdb_key"] = key_col
    agg = agg[agg["_fdb_key"] != ""]
    if agg.empty:
        master_by_fdb = {}
    else:
        master_by_fdb = agg.groupby("_fdb_key").agg(
            inchikey_present=("inchikey", lambda s: (s.astype(str).str.strip().str.len() >= 25).any()),
            cid_present=("cid", lambda s: (s.astype(str).str.strip() != "").any()),
            name_present=("name", lambda s: (s.astype(str).str.strip() != "").any()),
            source_files=("source_file", lambda s: "|".join(s.dropna().astype(str).unique()[:5])),
        ).to_dict("index")
    rows = []
    for fdb_norm, count in fdb_counts.items():
        if not fdb_norm:
            continue
        info = master_by_fdb.get(fdb_norm, {})
        rows.append({
            "fdb_id_norm": fdb_norm,
            "count_in_compound_gene": int(count),
            "has_master_row": fdb_norm in master_fdb,
            "inchikey_present": info.get("inchikey_present", False) or (fdb_norm in fdb_to_ik),
            "cid_present": info.get("cid_present", False),
            "name_present": info.get("name_present", False),
            "top_source_files_seen": (info.get("source_files") or "")[:500],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("count_in_compound_gene", ascending=False)
    df.to_csv(report_path, index=False)
    return df


def resolve_compound_to_inchikey(
    compound_id: str,
    fdb_to_ik: Dict[str, str],
) -> Optional[str]:
    """Resolve a compound_id (FDB*, COCONUT_*, InChIKey, etc.) to canonical InChIKey using fdb_to_ik and normalization."""
    if not compound_id or (isinstance(compound_id, float) and pd.isna(compound_id)):
        return None
    s = str(compound_id).strip()
    if looks_like_inchikey(s):
        return normalize_inchikey(s)
    norm_fdb = normalize_fdb_id(s)
    if norm_fdb and norm_fdb in fdb_to_ik:
        return fdb_to_ik[norm_fdb]
    if s in fdb_to_ik:
        return fdb_to_ik[s]
    if s.startswith("COCONUT_") and "." in s:
        base = s.split(".")[0]
        if base in fdb_to_ik:
            return fdb_to_ik[base]
    return fdb_to_ik.get(s)


def canonicalize_compound_gene_with_provenance(
    cg_df: pd.DataFrame,
    compound_master: pd.DataFrame,
    fdb_to_ik: Dict[str, str],
    *,
    compound_id_col: str = "compound_id",
    gene_col: str = "gene",
    compound_name_col: Optional[str] = None,
    source_col: Optional[str] = None,
    source_file_default: str = "food_compound_gene_links.parquet",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Resolve compound_gene rows to InChIKey with provenance.
    Order: (a) FDB normalize -> compound_master/fdb_to_ik, (b) compound_name exact + fuzzy -> name_to_ik, (c) CID fallback.
    Returns (canonical_df, stats) where canonical_df has: compound_id (InChIKey), gene, source_file, match_type, match_score, original_compound_id, original_name.
    """
    stats: Dict[str, int] = {"fdb_join": 0, "name_exact": 0, "name_fuzzy": 0, "cid_fallback": 0, "unresolved": 0}
    if cg_df.empty:
        return pd.DataFrame(columns=[
            "compound_id", "gene", "source_file", "match_type", "match_score", "original_compound_id", "original_name",
        ]), stats

    # Build name -> InChIKey and cid -> InChIKey from compound_master (normalized name)
    name_to_ik: Dict[str, str] = {}
    cid_to_ik: Dict[str, str] = {}
    for _, r in compound_master.iterrows():
        ik = _safe_str(r.get("inchikey"))
        if not ik or len(ik) < 25:
            continue
        nm = _safe_str(r.get("name"))
        if nm:
            key = normalize_name(nm)
            if key and key not in name_to_ik:
                name_to_ik[key] = ik
        cid = _safe_str(r.get("cid"))
        if cid:
            cid_to_ik[cid] = ik

    # CID fallback: prebuild name_norm -> ik and fdb_norm -> ik (only for rows with cid in cid_to_ik)
    name_norm_to_ik_cid: Dict[str, str] = {}
    fdb_to_ik_cid: Dict[str, str] = {}
    for _, r in compound_master.iterrows():
        cid = _safe_str(r.get("cid"))
        if not cid or cid not in cid_to_ik:
            continue
        ik = cid_to_ik[cid]
        nm = _safe_str(r.get("name"))
        if nm:
            key = normalize_name(nm)
            if key and key not in name_norm_to_ik_cid:
                name_norm_to_ik_cid[key] = ik
        fdb = _safe_str(r.get("fdb_id_norm")) or _safe_str(r.get("fdb_id"))
        if fdb and fdb not in fdb_to_ik_cid:
            fdb_to_ik_cid[fdb] = ik

    cmp_col = compound_id_col if compound_id_col in cg_df.columns else cg_df.columns[0]
    gene_c = gene_col if gene_col in cg_df.columns else next((c for c in cg_df.columns if "gene" in c.lower()), cg_df.columns[1])
    name_c = compound_name_col
    if name_c is None:
        name_c = next((c for c in cg_df.columns if "name" in c.lower() and "compound" in c.lower()), None) or next((c for c in cg_df.columns if c.lower() in ("compound_name", "name")), None)
    src_c = source_col if source_col and source_col in cg_df.columns else None

    out_rows: List[Dict[str, Any]] = []
    for _, r in cg_df.iterrows():
        orig_id = r.get(cmp_col)
        if pd.isna(orig_id) or not str(orig_id).strip():
            continue
        orig_id = str(orig_id).strip()
        gene_val = r.get(gene_c)
        if pd.isna(gene_val) or not str(gene_val).strip():
            continue
        gene_val = str(gene_val).strip().upper()
        orig_name = (r.get(name_c) or "") if name_c else ""
        if pd.isna(orig_name):
            orig_name = ""
        orig_name = str(orig_name).strip()
        src_file = str(r.get(src_c) or source_file_default).strip() if src_c else source_file_default

        inchikey: Optional[str] = None
        match_type = "unresolved"
        match_score: Optional[int] = None

        norm_fdb = normalize_fdb_id(orig_id)

        # (a) FDB normalize -> lookup; or already InChIKey
        if looks_like_inchikey(orig_id):
            inchikey = normalize_inchikey(orig_id)
            if inchikey:
                match_type = "fdb_join"
                match_score = 100
        if not inchikey and norm_fdb and norm_fdb in fdb_to_ik:
            inchikey = fdb_to_ik[norm_fdb]
            match_type = "fdb_join"
            match_score = 100

        # (b) compound_name exact then fuzzy
        if not inchikey and orig_name:
            name_norm = normalize_name(orig_name)
            if name_norm in name_to_ik:
                inchikey = name_to_ik[name_norm]
                match_type = "name_exact"
                match_score = 100
            if not inchikey and name_norm:
                try:
                    from rapidfuzz import fuzz
                    best_ik = None
                    best_score = 0
                    for other_name, ik in name_to_ik.items():
                        sc = fuzz.ratio(name_norm, other_name)
                        if sc >= 95 and sc > best_score:
                            best_score = sc
                            best_ik = ik
                    if best_ik:
                        inchikey = best_ik
                        match_type = "name_fuzzy"
                        match_score = best_score
                except ImportError:
                    pass

        # (c) CID fallback: lookup by name_norm or fdb (prebuilt maps)
        if not inchikey:
            if orig_name:
                name_norm = normalize_name(orig_name)
                if name_norm in name_norm_to_ik_cid:
                    inchikey = name_norm_to_ik_cid[name_norm]
                    match_type = "cid_fallback"
                    match_score = 90
            if not inchikey and norm_fdb and norm_fdb in fdb_to_ik_cid:
                inchikey = fdb_to_ik_cid[norm_fdb]
                match_type = "cid_fallback"
                match_score = 85

        if inchikey and len(inchikey) >= 25:
            stats[match_type] = stats.get(match_type, 0) + 1
            out_rows.append({
                "compound_id": inchikey,
                "gene": gene_val,
                "source_file": src_file,
                "match_type": match_type,
                "match_score": match_score if match_score is not None else 0,
                "original_compound_id": orig_id,
                "original_name": orig_name,
            })
        else:
            stats["unresolved"] = stats.get("unresolved", 0) + 1

    out = pd.DataFrame(out_rows)
    return out, stats


def write_compound_gene_diagnostics(
    reports_dir: Path,
    stats: Dict[str, int],
    cg_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    ing_canonical_df: pd.DataFrame,
    *,
    compound_id_col: str = "compound_id",
    compound_name_col: Optional[str] = None,
    gene_col: str = "gene",
) -> None:
    """
    Always write: compound_gene_resolution_stats.json, unresolved_compound_gene_top.csv, overlap_after_resolution.json.
    Uses Path(reports_dir) for Windows-safe paths.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "compound_gene_resolution_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Unresolved: rows in cg_df that are not in canonical_df (by original compound_id + gene)
    if not canonical_df.empty and "original_compound_id" in canonical_df.columns:
        resolved_keys = set(zip(canonical_df["original_compound_id"].astype(str), canonical_df["gene"].astype(str)))
    else:
        resolved_keys = set()
    cmp_c = compound_id_col if compound_id_col in cg_df.columns else cg_df.columns[0]
    gene_c = gene_col if gene_col in cg_df.columns else cg_df.columns[1]
    name_c = compound_name_col or next((c for c in cg_df.columns if "name" in c.lower()), None)
    unresolved = cg_df[~cg_df.apply(lambda r: (str(r.get(cmp_c, "")), str(r.get(gene_c, ""))) in resolved_keys, axis=1)]
    if not unresolved.empty:
        if name_c and name_c in unresolved.columns:
            agg = unresolved.groupby([cmp_c, name_c]).size().reset_index(name="count")
            agg.columns = ["compound_id", "compound_name", "count"]
        else:
            agg = unresolved.groupby(cmp_c).size().reset_index(name="count")
            agg["compound_name"] = ""
            agg = agg[["compound_id", "compound_name", "count"]]
        agg = agg.sort_values("count", ascending=False).head(500)
        agg.to_csv(reports_dir / "unresolved_compound_gene_top.csv", index=False)
    else:
        pd.DataFrame(columns=["compound_id", "compound_name", "count"]).to_csv(reports_dir / "unresolved_compound_gene_top.csv", index=False)

    set_ing = set(ing_canonical_df["compound_id"].dropna().astype(str).str.strip()) if "compound_id" in ing_canonical_df.columns else set()
    set_cg = set(canonical_df["compound_id"].dropna().astype(str).str.strip()) if not canonical_df.empty else set()
    overlap = set_ing & set_cg
    overlap_pct = (100.0 * len(overlap) / len(set_ing)) if set_ing else 0.0
    with open(reports_dir / "overlap_after_resolution.json", "w", encoding="utf-8") as f:
        json.dump({
            "pct_overlap": round(overlap_pct, 2),
            "n_ingredient_compound_compounds": len(set_ing),
            "n_compound_gene_compounds": len(set_cg),
            "n_overlap": len(overlap),
        }, f, indent=2)


def score_compound_gene_source_overlap(
    ingredient_compound_inchikey_set: Set[str],
    candidate_compound_gene_df: pd.DataFrame,
    fdb_to_inchikey: Dict[str, str],
    compound_id_column: str = "compound_id",
) -> float:
    """
    Score overlap between ingredient_compound canonical InChIKeys and a candidate compound->gene table.
    Resolves candidate compound_id to InChIKey via fdb_to_inchikey; returns fraction of
    ingredient_compound_inchikey_set that appear in the resolved candidate compound set.
    Returns 0.0 if ingredient set is empty.
    """
    if not ingredient_compound_inchikey_set:
        return 0.0
    if candidate_compound_gene_df.empty or compound_id_column not in candidate_compound_gene_df.columns:
        return 0.0
    resolved: Set[str] = set()
    for v in candidate_compound_gene_df[compound_id_column].dropna().astype(str).str.strip().unique():
        ik = resolve_compound_to_inchikey(v, fdb_to_inchikey)
        if ik and len(ik) >= 25:
            resolved.add(ik)
    overlap = ingredient_compound_inchikey_set & resolved
    return len(overlap) / len(ingredient_compound_inchikey_set)


def build_overlap_diagnostics(
    ing_cmp_canonical: pd.DataFrame,
    cg_canonical: pd.DataFrame,
    ing_cmp_raw_compound_series: Optional[pd.Series] = None,
    cg_raw_compound_series: Optional[pd.Series] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Build extended diagnostics and compound_overlap_report.csv.
    Returns (diagnostics_dict, compound_overlap_report_df).
    """
    cmp_col_ing = "compound_id" if "compound_id" in ing_cmp_canonical.columns else ing_cmp_canonical.columns[-1]
    cmp_col_cg = "compound_id" if "compound_id" in cg_canonical.columns else cg_canonical.columns[0]
    top50_ing = (
        ing_cmp_canonical[cmp_col_ing]
        .value_counts()
        .head(50)
        .index.tolist()
    )
    top50_cg = (
        cg_canonical[cmp_col_cg]
        .value_counts()
        .head(50)
        .index.tolist()
    )

    # FDB pattern counts (regex pattern -> count) for raw compound_id
    def _pattern_counts(series: Optional[pd.Series]) -> List[Tuple[str, int]]:
        if series is None or series.empty:
            return []
        patterns = [
            (r"^FDB\d+$", "FDB<digits>"),
            (r"^FDB_\d+$", "FDB_<digits>"),
            (r"^FDB-\d+$", "FDB-<digits>"),
            (r"^FOODB:\d+$", "FOODB:<digits>"),
            (r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", "InChIKey"),
            (r"^COCONUT_", "COCONUT_*"),
        ]
        out: List[Tuple[str, int]] = []
        s = series.astype(str).str.strip()
        for pat, name in patterns:
            count = s.str.match(pat, na=False).sum()
            if count > 0:
                out.append((name, int(count)))
        return sorted(out, key=lambda x: -x[1])[:20]

    ing_patterns = _pattern_counts(ing_cmp_raw_compound_series)
    cg_patterns = _pattern_counts(cg_raw_compound_series)

    set_ing = set(ing_cmp_canonical[cmp_col_ing].dropna().astype(str).str.strip())
    set_cg = set(cg_canonical[cmp_col_cg].dropna().astype(str).str.strip())
    overlap_ik = set_ing & set_cg

    # compound_overlap_report: canonical_inchikey, count_in_ingredient_compound, count_in_compound_gene, appears_in_both
    all_ik = set_ing | set_cg
    rows = []
    ing_counts = ing_cmp_canonical[cmp_col_ing].value_counts()
    cg_counts = cg_canonical[cmp_col_cg].value_counts()
    for ik in all_ik:
        rows.append({
            "canonical_inchikey": ik,
            "count_in_ingredient_compound": int(ing_counts.get(ik, 0)),
            "count_in_compound_gene": int(cg_counts.get(ik, 0)),
            "appears_in_both": ik in overlap_ik,
        })
    report_df = pd.DataFrame(rows)
    report_df = report_df.sort_values(
        ["appears_in_both", "count_in_ingredient_compound", "count_in_compound_gene"],
        ascending=[False, False, False],
    )

    diagnostics = {
        "top_50_inchikey_ingredient_compound": top50_ing,
        "top_50_inchikey_compound_gene": top50_cg,
        "top_20_compound_gene_fdb_id_patterns": [{"pattern": p, "count": c} for p, c in cg_patterns],
        "top_20_ingredient_compound_fdb_id_patterns": [{"pattern": p, "count": c} for p, c in ing_patterns],
    }
    return diagnostics, report_df


def run_compound_identity_pipeline(
    repo_root: Path,
    *,
    write_scan: bool = False,
    min_overlap_pct: float = 20.0,
    smoke: bool = False,
) -> int:
    """
    Full pipeline: scan, compound_master, FDB->InChIKey, canonical CSVs, overlap diagnostics, overlap report.
    When smoke=True, do not exit non-zero; write diagnostics and unresolved report.
    Returns 0 on success or smoke run, 1 on failure (e.g. overlap < min_overlap_pct).
    """
    import json
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Scan
    scan_results = scan_processed_for_compound_columns(repo_root)
    if write_scan:
        with open(canonical_dir / "compound_scan.json", "w", encoding="utf-8") as f:
            json.dump(scan_results, f, indent=2)
    # 2) Compound master
    compound_master = build_compound_master_from_sources(repo_root, scan_results)
    if compound_master.empty:
        return 1
    subset = [c for c in ["inchikey", "fdb_id_norm"] if c in compound_master.columns]
    if subset:
        compound_master = compound_master.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    compound_master["compound_master_id"] = [f"CMP_{i+1:08d}" for i in range(len(compound_master))]
    compound_master.to_csv(canonical_dir / "compound_master.csv", index=False)

    # 3) FDB -> InChIKey
    fdb_to_ik, resolution_method, _ = build_fdb_to_inchikey_with_fallbacks(repo_root, compound_master, report_dir=reports_dir)

    # 4) Ingredient-compound canonical
    ing_cmp_path = canonical_dir / "ingredient_compound_links.csv"
    if not ing_cmp_path.exists():
        ing_cmp_path = canonical_dir / "ingredient_compound_links.parquet"
    if not ing_cmp_path.exists():
        ing_cmp_path = processed / "phase14_mediation" / "derived" / "ingredient_compound_links.csv"
    if not ing_cmp_path.exists():
        ing_cmp_path = processed / "phase14_mediation" / "derived" / "ingredient_compound_links.parquet"
    if not ing_cmp_path.exists():
        return 1
    ing_cmp = load_csv_or_parquet(ing_cmp_path)
    if ing_cmp is None or ing_cmp.empty:
        return 1
    id_col = "ingredient_id" if "ingredient_id" in ing_cmp.columns else ing_cmp.columns[0]
    cmp_col = "compound_id" if "compound_id" in ing_cmp.columns else ing_cmp.columns[1]
    ing_cmp = ing_cmp[[id_col, cmp_col]].dropna(how="all")
    ing_cmp[id_col] = ing_cmp[id_col].astype(str).str.strip()
    ing_cmp[cmp_col] = ing_cmp[cmp_col].astype(str).str.strip()
    ing_cmp_raw_series = ing_cmp[cmp_col].copy()

    def _to_ik(cval: str) -> Optional[str]:
        if not cval:
            return None
        if looks_like_inchikey(cval):
            return normalize_inchikey(cval)
        norm = normalize_fdb_id(cval)
        if norm and norm in fdb_to_ik:
            return fdb_to_ik[norm]
        if cval in fdb_to_ik:
            return fdb_to_ik[cval]
        if cval.startswith("COCONUT_") and "." in cval:
            base = cval.split(".")[0]
            if base in fdb_to_ik:
                return fdb_to_ik[base]
        return fdb_to_ik.get(cval)

    uniques = ing_cmp[cmp_col].dropna().astype(str).str.strip().unique()
    resolve_map = {u: _to_ik(u) for u in uniques}
    ing_cmp["compound_id_canonical"] = ing_cmp[cmp_col].astype(str).str.strip().map(resolve_map)
    ing_cmp_canonical = ing_cmp[ing_cmp["compound_id_canonical"].notna()][[id_col, "compound_id_canonical"]].rename(
        columns={"compound_id_canonical": "compound_id"}
    ).drop_duplicates()
    ing_cmp_canonical.to_csv(canonical_dir / "ingredient_compound_canonical.csv", index=False)

    # 5) Compound-gene canonical with provenance (FDB -> name exact/fuzzy -> CID fallback)
    cg_path = canonical_dir / "compound_gene_links.csv"
    if not cg_path.exists():
        cg_path = canonical_dir / "compound_gene_links.parquet"
    if not cg_path.exists():
        cg_path = processed / "phase12_genetics" / "food_compound_gene_links.parquet"
    if not cg_path.exists():
        return 1
    cg = load_csv_or_parquet(cg_path)
    if cg is None or cg.empty:
        return 1
    cmp_col_cg = next((c for c in cg.columns if "compound" in c.lower() and "id" in c.lower()), None) or next((c for c in cg.columns if "inchikey" in c.lower()), cg.columns[0])
    gene_col_cg = next((c for c in cg.columns if c.lower() in ("gene", "gene_symbol", "symbol")), None) or next((c for c in cg.columns if "gene" in c.lower() or "target" in c.lower()), None)
    if not gene_col_cg:
        gene_col_cg = [c for c in cg.columns if c != cmp_col_cg][0] if len(cg.columns) > 1 else None
    if not gene_col_cg:
        return 1
    name_col_cg = next((c for c in cg.columns if "compound_name" in c.lower() or (c.lower() == "name")), None)
    src_col_cg = next((c for c in cg.columns if "source" in c.lower()), None)
    cg_work = cg[[cmp_col_cg, gene_col_cg] + ([name_col_cg] if name_col_cg else []) + ([src_col_cg] if src_col_cg else [])].copy()
    cg_work = cg_work.dropna(subset=[cmp_col_cg, gene_col_cg], how="all")
    cg_work[cmp_col_cg] = cg_work[cmp_col_cg].astype(str).str.strip()
    cg_work[gene_col_cg] = cg_work[gene_col_cg].astype(str).str.strip().str.upper()
    cg_work = cg_work[cg_work[cmp_col_cg].str.len() > 0]
    cg_work = cg_work[cg_work[gene_col_cg].str.len() > 0]
    cg_canonical, cg_stats = canonicalize_compound_gene_with_provenance(
        cg_work, compound_master, fdb_to_ik,
        compound_id_col=cmp_col_cg, gene_col=gene_col_cg,
        compound_name_col=name_col_cg, source_col=src_col_cg,
        source_file_default=cg_path.name,
    )
    cg_raw_series = cg_work[cmp_col_cg].copy()
    cg_canonical.to_csv(canonical_dir / "compound_gene_canonical.csv", index=False)
    write_compound_gene_diagnostics(
        reports_dir, cg_stats, cg_work, cg_canonical, ing_cmp_canonical,
        compound_id_col=cmp_col_cg, compound_name_col=name_col_cg, gene_col=gene_col_cg,
    )

    # 6) Overlap and diagnostics
    set_ing = set(ing_cmp_canonical["compound_id"].unique())
    set_cg = set(cg_canonical["compound_id"].unique())
    overlap = set_ing & set_cg
    overlap_pct = (100.0 * len(overlap) / len(set_ing)) if set_ing else 0.0
    ext_diag, overlap_report_df = build_overlap_diagnostics(
        ing_cmp_canonical, cg_canonical,
        ing_cmp_raw_compound_series=ing_cmp_raw_series,
        cg_raw_compound_series=cg_raw_series,
    )
    overlap_report_df.to_csv(canonical_dir / "compound_overlap_report.csv", index=False)

    # Unresolved FDB report (long IDs from compound_gene that remain unmapped)
    unresolved_df = write_unresolved_fdb_report(
        cg_raw_series, compound_master, fdb_to_ik, reports_dir / "unresolved_fdb_ids.csv"
    )

    n_ing_unique = ing_cmp[cmp_col].nunique()
    n_ing_resolved = ing_cmp_canonical["compound_id"].nunique()
    diagnostics = {
        "pct_ingredient_compounds_resolved": round(100.0 * n_ing_resolved / n_ing_unique, 2) if n_ing_unique else 0,
        "pct_compound_gene_rows_resolved": round(100.0 * len(cg_canonical) / len(cg), 2) if len(cg) else 0,
        "pct_final_overlap": round(overlap_pct, 2),
        "min_overlap_required": min_overlap_pct,
        "n_overlap": len(overlap),
        "n_ing_cmp_compounds": len(set_ing),
        "n_cg_compounds": len(set_cg),
        "fdb_to_inchikey_count": len(fdb_to_ik),
        **ext_diag,
    }
    with open(canonical_dir / "compound_identity_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    if overlap_pct < min_overlap_pct and not smoke:
        return 1
    return 0


if __name__ == "__main__":
    import argparse
    import json
    import sys
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    parser = argparse.ArgumentParser(description="Phase14 compound identity pipeline")
    parser.add_argument("--repo-root", type=Path, default=_root)
    parser.add_argument("--write-scan", action="store_true", help="Write compound_scan.json")
    parser.add_argument("--min-overlap-pct", type=float, default=20.0)
    parser.add_argument("--smoke", action="store_true", help="Do not exit non-zero; write diagnostics and unresolved report")
    args = parser.parse_args()
    code = run_compound_identity_pipeline(
        args.repo_root, write_scan=args.write_scan, min_overlap_pct=args.min_overlap_pct, smoke=args.smoke
    )
    _canonical = args.repo_root / "data" / "processed" / "canonical"
    _reports = args.repo_root / "reports"
    if (_canonical / "compound_identity_diagnostics.json").exists():
        with open(_canonical / "compound_identity_diagnostics.json", encoding="utf-8") as _f:
            _d = json.load(_f)
        print("pct_ingredient_compounds_resolved: %.2f%%" % _d.get("pct_ingredient_compounds_resolved", 0))
        print("pct_compound_gene_rows_resolved: %.2f%%" % _d.get("pct_compound_gene_rows_resolved", 0))
        print("pct_final_overlap: %.2f%% (n_overlap=%s)" % (_d.get("pct_final_overlap", 0), _d.get("n_overlap", 0)))
        if code != 0 and not args.smoke:
            print("FAIL: overlap < %.0f%%" % args.min_overlap_pct)
        else:
            print("OK: overlap >= %.0f%% (or smoke run)" % args.min_overlap_pct)
    if _reports.exists() and (_reports / "unresolved_fdb_ids.csv").exists():
        _unres = pd.read_csv(_reports / "unresolved_fdb_ids.csv")
        if not _unres.empty and "fdb_id_norm" in _unres.columns and "count_in_compound_gene" in _unres.columns:
            _top = _unres.head(10)
            print("top 10 unresolved fdb IDs by count:")
            for _, _r in _top.iterrows():
                print("  %s  count=%s" % (_r.get("fdb_id_norm", ""), _r.get("count_in_compound_gene", 0)))
    sys.exit(code)
