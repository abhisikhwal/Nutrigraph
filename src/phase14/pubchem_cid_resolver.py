"""
Resolve PubChem CID (numeric or CID:xxx / pubchem:xxx) to InChIKey using ONLY local repo assets.
Sources (priority): compound_master.csv, pharmgkb_chemicals.parquet, then scan of data/processed.
Returns resolver_used: cid_master / cid_pharmgkb / cid_scan / unresolved.
NaN-safe throughout. No web APIs.
"""
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

RESOLVER_CID_MASTER = "cid_master"
RESOLVER_CID_PHARMGKB = "cid_pharmgkb"
RESOLVER_CID_SCAN = "cid_scan"
RESOLVER_UNRESOLVED = "unresolved"


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return s if s else ""


def normalize_cid(compound_id: Any) -> Optional[str]:
    """
    Normalize CID: accept "1014", "CID:1014", "pubchem:1014" -> "1014".
    Returns None if not a valid numeric CID. NaN-safe.
    """
    raw = _safe_str(compound_id)
    if not raw:
        return None
    raw_upper = raw.upper()
    # Strip prefixes
    if raw_upper.startswith("CID:") or raw_upper.startswith("CID "):
        raw = raw[4:].lstrip()
    elif "PUBCHEM" in raw_upper and ("COMPOUND" in raw_upper or ":" in raw):
        m = re.search(r"(\d+)", raw)
        if m:
            raw = m.group(1)
        else:
            return None
    if not raw.isdigit():
        return None
    return raw


def _load_df(path: Path, dtype_str: bool = False) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False, dtype=str if dtype_str else None)
    except Exception as e:
        logger.warning("pubchem_cid_resolver load %s: %s", path, e)
        return None


def _col(df: pd.DataFrame, names: list) -> Optional[str]:
    if df is None or df.empty:
        return None
    low = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "_")
        if k in low:
            return low[k]
    return None


def build_cid_master_map(repo_root: Path) -> Dict[str, str]:
    """Build cid -> inchikey from compound_master.csv (rows with both cid and inchikey)."""
    repo_root = Path(repo_root).resolve()
    canonical = repo_root / "data" / "processed" / "canonical"
    out: Dict[str, str] = {}
    for p in [canonical / "compound_master.csv", canonical / "compound_master.parquet"]:
        if not p.exists():
            continue
        df = _load_df(p, dtype_str=True)
        if df is None or df.empty:
            continue
        cid_col = _col(df, ["cid", "pubchem_cid", "pubchem_compound_id"])
        ik_col = _col(df, ["inchikey", "inchi_key", "compound_id"])
        if not cid_col or not ik_col:
            continue
        for _, r in df.iterrows():
            cid = _safe_str(r.get(cid_col))
            if not cid:
                continue
            if not cid.isdigit():
                cid = normalize_cid(cid)
                if cid is None:
                    continue
            ik = _safe_str(r.get(ik_col))
            if not ik or len(ik) < 25:
                continue
            out[cid] = ik
        break
    logger.info("build_cid_master_map: %s entries", len(out))
    return out


def build_cid_pharmgkb_map(repo_root: Path) -> Dict[str, str]:
    """
    Build cid -> inchikey from pharmgkb_chemicals.parquet.
    Parse PubChem Compound Identifiers (float-safe); if InChI exists derive InChIKey via RDKit;
    if SMILES and RDKit, derive InChIKey. Otherwise store by CID only when InChIKey column present.
    """
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    phase12 = processed / "phase12_genetics"
    out: Dict[str, str] = {}
    for path in list(phase12.glob("pharmgkb_chemicals.parquet")):
        if not path.exists():
            continue
        df = _load_df(path)
        if df is None or df.empty:
            continue
        cl = {c.lower().replace(" ", "_"): c for c in df.columns}
        cid_col = cl.get("pubchem_compound_identifiers") or cl.get("pubchem_cid") or cl.get("cid")
        ik_col = cl.get("inchikey") or cl.get("inchi_key")
        inchi_col = cl.get("inchi")
        smiles_col = cl.get("smiles")
        cross_col = cl.get("cross_references") or cl.get("cross-references")
        for _, r in df.iterrows():
            cids = set()
            if cid_col:
                v = r.get(cid_col)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    try:
                        cids.add(str(int(float(v))))
                    except (ValueError, TypeError):
                        cids.add(_safe_str(v))
            if cross_col:
                raw = _safe_str(r.get(cross_col))
                for part in raw.replace(";", ",").split(","):
                    m = re.search(r"(\d+)", part)
                    if m and ("pubchem" in part.lower() or "compound" in part.lower()):
                        cids.add(m.group(1))
            if not cids:
                continue
            ik = None
            if ik_col:
                ik = _safe_str(r.get(ik_col))
                if len(ik) < 25:
                    ik = None
            if not ik and inchi_col:
                try:
                    from rdkit import Chem
                    inchi_val = _safe_str(r.get(inchi_col))
                    if inchi_val:
                        mol = Chem.MolFromInchi(inchi_val)
                        if mol:
                            ik = Chem.inchi.MolToInchiKey(mol)
                except Exception:
                    pass
            if not ik and smiles_col:
                try:
                    from rdkit import Chem
                    smi = _safe_str(r.get(smiles_col))
                    if smi:
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            ik = Chem.inchi.MolToInchiKey(mol)
                except Exception:
                    pass
            if not ik:
                continue
            for cid in cids:
                if cid and cid.isdigit():
                    out[cid] = ik
                    break
        break
    logger.info("build_cid_pharmgkb_map: %s entries", len(out))
    return out


def build_cid_scan_map(repo_root: Path) -> Dict[str, str]:
    """Scan data/processed for files containing both cid and inchikey/inchi_key; build cid -> inchikey."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    out: Dict[str, str] = {}
    seen: set = set()
    for ext in ["*.parquet", "*.csv"]:
        for p in processed.rglob(ext):
            if p.name.startswith(".") or "phase16_bindingdb" in str(p) and "compound_target" in p.name:
                continue
            key = (p.parent, p.name)
            if key in seen:
                continue
            seen.add(key)
            df = _load_df(p, dtype_str=(p.suffix.lower() == ".csv"))
            if df is None or df.empty or len(df) > 500_000:
                continue
            cid_col = _col(df, ["cid", "pubchem_cid", "pubchem_compound_id", "compound_id"])
            ik_col = _col(df, ["inchikey", "inchi_key"])
            if not cid_col or not ik_col:
                continue
            for _, r in df.iterrows():
                cid = _safe_str(r.get(cid_col))
                if not cid:
                    continue
                if not cid.isdigit():
                    cid = normalize_cid(cid)
                    if cid is None:
                        continue
                ik = _safe_str(r.get(ik_col))
                if not ik or len(ik) < 25:
                    continue
                out.setdefault(cid, ik)
    logger.info("build_cid_scan_map: %s entries", len(out))
    return out


def build_all_cid_maps(repo_root: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Build (cid_master, cid_pharmgkb, cid_scan) in priority order. Fast prebuild."""
    cid_master = build_cid_master_map(repo_root)
    cid_pharmgkb = build_cid_pharmgkb_map(repo_root)
    cid_scan = build_cid_scan_map(repo_root)
    return cid_master, cid_pharmgkb, cid_scan


def resolve_cid(
    compound_id: Any,
    cid_master: Optional[Dict[str, str]] = None,
    cid_pharmgkb: Optional[Dict[str, str]] = None,
    cid_scan: Optional[Dict[str, str]] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[Optional[str], str]:
    """
    Resolve compound_id (numeric or CID:xxx / pubchem:xxx) to InChIKey.
    Returns (inchikey, resolver_used) with resolver_used in cid_master / cid_pharmgkb / cid_scan / unresolved.
    """
    cid = normalize_cid(compound_id)
    if cid is None:
        return None, RESOLVER_UNRESOLVED
    if repo_root is not None and (cid_master is None or cid_pharmgkb is None or cid_scan is None):
        m, p, s = build_all_cid_maps(Path(repo_root).resolve())
        if cid_master is None:
            cid_master = m
        if cid_pharmgkb is None:
            cid_pharmgkb = p
        if cid_scan is None:
            cid_scan = s
    cid_master = cid_master or {}
    cid_pharmgkb = cid_pharmgkb or {}
    cid_scan = cid_scan or {}
    if cid in cid_master:
        return cid_master[cid], RESOLVER_CID_MASTER
    if cid in cid_pharmgkb:
        return cid_pharmgkb[cid], RESOLVER_CID_PHARMGKB
    if cid in cid_scan:
        return cid_scan[cid], RESOLVER_CID_SCAN
    return None, RESOLVER_UNRESOLVED
