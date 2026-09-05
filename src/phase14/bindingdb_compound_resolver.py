"""
Option 1: Resolve BindingDB compound_id to canonical InChIKey.
Handlers: COCONUT_*, CHEMBL*, numeric CID (CID:xxxx or plain), SMILES (optional).
Maps from compound_master (chembl_id, bindingdb_id, cid, name_norm) + pharmgkb_chemicals.
Output: compound_id_raw, compound_id_type, inchikey, cid, name, resolver_used_compound.
No external API calls.
"""
from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .compound_identity import normalize_inchikey, looks_like_inchikey, _safe_str
from .compound_identity_bridge import load_csv_or_parquet
from .pubchem_cid_resolver import (
    normalize_cid,
    resolve_cid,
    build_all_cid_maps,
)

logger = logging.getLogger(__name__)

RESOLVER_COCONUT_MAP = "coconut_map"
RESOLVER_CHEMBL_MAP = "chembl_map"
RESOLVER_CID_MAP = "cid_map"
RESOLVER_NAME_MAP = "name_map"
RESOLVER_SMILES_RDKIT = "smiles_rdkit"
RESOLVER_DIRECT_INCHIKEY = "direct_inchikey"


def _load_df(path: Path, dtype_str: bool = False) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if dtype_str:
            return pd.read_csv(path, low_memory=False, dtype=str)
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        logger.warning("Load failed %s: %s", path, e)
        return None


def _normalize_name(s: str) -> str:
    if not s:
        return ""
    return " ".join(str(s).lower().strip().split())


def build_coconut_to_inchikey_map(repo_root: Path) -> Dict[str, str]:
    """Build COCONUT_* -> InChIKey from inchikey_to_compound_id.json (invert) and compound_identity_bridge."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    canonical = processed / "canonical"
    coconut_to_ik: Dict[str, str] = {}

    json_path = processed / "phase15_coconut" / "inchikey_to_compound_id.json"
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                ik2id = json.load(f)
            for ik, raw_id in ik2id.items():
                if raw_id is None:
                    continue
                s = _safe_str(raw_id)
                if not s or "COCONUT_" not in s.upper():
                    continue
                ik_norm = normalize_inchikey(ik)
                if not ik_norm or len(ik_norm) < 25:
                    continue
                coconut_to_ik[s] = ik_norm
                if "." in s:
                    base = s.split(".")[0]
                    if base:
                        coconut_to_ik[base] = ik_norm
        except Exception as e:
            logger.warning("build_coconut_to_inchikey_map: JSON %s", e)

    bridge_path = canonical / "compound_identity_bridge.csv"
    if bridge_path.exists():
        df = load_csv_or_parquet(bridge_path)
        if df is not None and not df.empty:
            ns_col = next((c for c in df.columns if "namespace" in c.lower()), None)
            sid_col = next((c for c in df.columns if "source_id" in c.lower() or "source_id" == c), None)
            ik_col = next((c for c in df.columns if "inchikey" in c.lower()), None)
            if sid_col and ik_col:
                for _, r in df.iterrows():
                    if ns_col and _safe_str(r.get(ns_col)).upper() != "COCONUT":
                        continue
                    sid = _safe_str(r.get(sid_col))
                    if not sid or "COCONUT_" not in sid.upper():
                        continue
                    ik = _safe_str(r.get(ik_col))
                    if len(ik) >= 25:
                        coconut_to_ik[sid] = ik
                        if "." in sid:
                            coconut_to_ik[sid.split(".")[0]] = ik

    for p in [canonical / "compound_master.csv", canonical / "compound_master.parquet"]:
        if not p.exists():
            continue
        df = _load_df(p, dtype_str=True)
        if df is None or df.empty:
            continue
        coco_col = next((c for c in df.columns if "coconut" in c.lower()), None)
        ik_col = next((c for c in df.columns if "inchikey" in c.lower()), None) or next((c for c in df.columns if c == "compound_id"), None)
        if not coco_col or not ik_col:
            continue
        for _, r in df.iterrows():
            coco = _safe_str(r.get(coco_col))
            if not coco or "COCONUT_" not in coco.upper():
                continue
            ik = _safe_str(r.get(ik_col))
            if ik and looks_like_inchikey(ik):
                coconut_to_ik[coco] = normalize_inchikey(ik)
                if "." in coco:
                    coconut_to_ik[coco.split(".")[0]] = normalize_inchikey(ik)
        break

    logger.info("build_coconut_to_inchikey_map: %s entries", len(coconut_to_ik))
    return coconut_to_ik


def build_compound_master_maps(repo_root: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load compound_master.csv with dtype=str, low_memory=False; build chembl_id->ik, bindingdb_id->ik, cid->ik, name_norm->ik."""
    repo_root = Path(repo_root).resolve()
    canonical = repo_root / "data" / "processed" / "canonical"
    chembl_to_ik: Dict[str, str] = {}
    bindingdb_to_ik: Dict[str, str] = {}
    cid_to_ik: Dict[str, str] = {}
    name_to_ik: Dict[str, str] = {}

    for p in [canonical / "compound_master.csv", canonical / "compound_master.parquet"]:
        if not p.exists():
            continue
        df = _load_df(p, dtype_str=True)
        if df is None or df.empty:
            continue
        ik_col = next((c for c in df.columns if "inchikey" in c.lower()), None) or next((c for c in df.columns if c == "compound_id"), None)
        if not ik_col:
            continue
        chembl_col = next((c for c in df.columns if "chembl" in c.lower()), None)
        bdb_col = next((c for c in df.columns if "bindingdb" in c.lower()), None)
        cid_col = next((c for c in df.columns if "cid" in c.lower() and "pubchem" not in c.lower()), None) or next((c for c in df.columns if "pubchem" in c.lower()), None)
        name_col = next((c for c in df.columns if c.lower() == "name"), None)
        for _, r in df.iterrows():
            ik = _safe_str(r.get(ik_col))
            if not ik or not looks_like_inchikey(ik):
                continue
            ik = normalize_inchikey(ik)
            if chembl_col:
                c = _safe_str(r.get(chembl_col)).upper()
                if c and c.startswith("CHEMBL"):
                    chembl_to_ik[c] = ik
            if bdb_col:
                b = _safe_str(r.get(bdb_col))
                if b:
                    bindingdb_to_ik[b] = ik
                    bindingdb_to_ik[b.upper()] = ik
            if cid_col:
                c = _safe_str(r.get(cid_col))
                if c and c.isdigit():
                    cid_to_ik[c] = ik
            if name_col:
                nm = _normalize_name(r.get(name_col))
                if nm:
                    name_to_ik[nm] = ik
        break

    logger.info("compound_master maps: chembl=%s bindingdb=%s cid=%s name=%s", len(chembl_to_ik), len(bindingdb_to_ik), len(cid_to_ik), len(name_to_ik))
    return chembl_to_ik, bindingdb_to_ik, cid_to_ik, name_to_ik


def build_pharmgkb_cid_to_inchikey(repo_root: Path, write_csv: bool = True) -> Dict[str, str]:
    """From pharmgkb_chemicals: PubChem Compound Identifiers, Cross-references (PubChem Compound:<CID>), InChI/InChIKey. Optional write pharmgkb_cid_to_inchikey.csv."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    phase12 = processed / "phase12_genetics"
    cid_to_ik: Dict[str, str] = {}
    for path in list(phase12.glob("pharmgkb_chemicals.parquet")) + []:
        if not path.exists():
            continue
        df = _load_df(path)
        if df is None or df.empty:
            continue
        cl = {c.lower().replace(" ", "_"): c for c in df.columns}
        cid_col = cl.get("pubchem_compound_identifiers") or cl.get("pubchem_cid") or cl.get("cid")
        ik_col = cl.get("inchikey") or cl.get("inchi_key")
        inchi_col = cl.get("inchi")
        cross_col = cl.get("cross_references") or cl.get("cross-references")
        for _, r in df.iterrows():
            ik = None
            if ik_col:
                ik = normalize_inchikey(r.get(ik_col))
            if not ik and inchi_col:
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromInchi(_safe_str(r.get(inchi_col)))
                    if mol:
                        ik = Chem.inchi.MolToInchiKey(mol)
                except Exception:
                    pass
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
                    part = part.strip()
                    if "pubchem" in part.lower() and "compound" in part.lower():
                        m = re.search(r"(\d+)", part)
                        if m:
                            cids.add(m.group(1))
            for cid in cids:
                if cid and cid.isdigit():
                    if ik:
                        cid_to_ik[cid] = ik
                    break
        break

    if write_csv and cid_to_ik:
        out_path = repo_root / "data" / "processed" / "canonical" / "pharmgkb_cid_to_inchikey.csv"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"cid": c, "inchikey": ik} for c, ik in cid_to_ik.items()]).to_csv(out_path, index=False)
            logger.info("Wrote %s", out_path.name)
        except Exception as e:
            logger.warning("Could not write pharmgkb_cid_to_inchikey.csv: %s", e)
    return cid_to_ik


def _inchikey_from_smiles(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.inchi.MolToInchiKey(mol)
    except Exception:
        pass
    return None


def resolve_bindingdb_compound(
    compound_id_raw: str,
    repo_root: Path,
    coconut_to_ik: Optional[Dict[str, str]] = None,
    cid_to_ik: Optional[Dict[str, str]] = None,
    chembl_to_ik: Optional[Dict[str, str]] = None,
    bindingdb_to_ik: Optional[Dict[str, str]] = None,
    name_to_ik: Optional[Dict[str, str]] = None,
    cid_master_map: Optional[Dict[str, str]] = None,
    cid_pharmgkb_map: Optional[Dict[str, str]] = None,
    cid_scan_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str, str]:
    """
    Resolve BindingDB compound_id to canonical InChIKey.
    Returns (inchikey, cid, smiles, name, source_file, resolver_used).
    Order: (a) direct InChIKey (b) COCONUT (c) CHEMBL (d) CID (e) name exact/fuzzy95.
    """
    raw = _safe_str(compound_id_raw)
    if not raw:
        return None, None, None, None, "", "unresolved_empty"

    repo_root = Path(repo_root).resolve()

    if looks_like_inchikey(raw):
        return normalize_inchikey(raw), None, None, None, "", RESOLVER_DIRECT_INCHIKEY

    if coconut_to_ik is None:
        coconut_to_ik = build_coconut_to_inchikey_map(repo_root)
    if cid_to_ik is None:
        cid_to_ik = _build_cid_to_inchikey(repo_root)
        pharm = build_pharmgkb_cid_to_inchikey(repo_root, write_csv=False)
        for c, ik in pharm.items():
            cid_to_ik.setdefault(c, ik)
    if chembl_to_ik is None or bindingdb_to_ik is None or name_to_ik is None:
        cm_chembl, cm_bdb, cm_cid, cm_name = build_compound_master_maps(repo_root)
        if chembl_to_ik is None:
            chembl_to_ik = cm_chembl
        if bindingdb_to_ik is None:
            bindingdb_to_ik = cm_bdb
        if name_to_ik is None:
            name_to_ik = cm_name
        for c, ik in cm_cid.items():
            cid_to_ik.setdefault(c, ik)

    if raw.upper().startswith("COCONUT_"):
        ik = coconut_to_ik.get(raw) or coconut_to_ik.get(raw.split(".")[0] if "." in raw else "")
        if ik:
            return ik, None, None, None, "inchikey_to_compound_id.json", RESOLVER_COCONUT_MAP
        return None, None, None, None, "", "unresolved_coconut"

    if raw.upper().startswith("CHEMBL"):
        ik = chembl_to_ik.get(raw.upper())
        if ik:
            return ik, None, None, None, "compound_master", RESOLVER_CHEMBL_MAP

    # Numeric or CID:xxx / pubchem:xxx => use pubchem_cid_resolver for cid_master / cid_pharmgkb / cid_scan tags
    cid_norm = normalize_cid(raw)
    if cid_norm is not None:
        ik, cid_resolver = resolve_cid(
            raw,
            cid_master=cid_master_map,
            cid_pharmgkb=cid_pharmgkb_map,
            cid_scan=cid_scan_map,
            repo_root=repo_root,
        )
        if ik:
            return ik, cid_norm, None, None, "", cid_resolver
        # Fallback to merged cid_to_ik
        if cid_to_ik is None:
            cid_to_ik = _build_cid_to_inchikey(repo_root)
            pharm = build_pharmgkb_cid_to_inchikey(repo_root, write_csv=False)
            for c, ik_val in pharm.items():
                cid_to_ik.setdefault(c, ik_val)
        if cid_norm in cid_to_ik:
            return cid_to_ik[cid_norm], cid_norm, None, None, "pharmgkb_chemicals/compound_master", RESOLVER_CID_MAP

    if raw in bindingdb_to_ik or raw.upper() in bindingdb_to_ik:
        return bindingdb_to_ik.get(raw) or bindingdb_to_ik.get(raw.upper()), None, None, None, "compound_master", "bindingdb_id"

    nm = _normalize_name(raw)
    if nm and name_to_ik and nm in name_to_ik:
        return name_to_ik[nm], None, None, None, "compound_master", RESOLVER_NAME_MAP

    if name_to_ik:
        try:
            from rapidfuzz import fuzz
            best_score = 0
            best_ik = None
            for key, ik in name_to_ik.items():
                score = fuzz.ratio(nm, key)
                if score >= 95 and score > best_score:
                    best_score = score
                    best_ik = ik
            if best_ik:
                return best_ik, None, None, None, "compound_master", "name_fuzzy95"
        except ImportError:
            pass

    return None, None, None, None, "", "unresolved"


def _build_cid_to_inchikey(repo_root: Path) -> Dict[str, str]:
    """From pharmgkb_chemicals and compound_master build cid -> inchikey (local only)."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    cid_to_ik: Dict[str, str] = {}
    for p in list((processed / "phase12_genetics").glob("pharmgkb_chemicals.parquet")) + []:
        if not p.exists():
            continue
        df = _load_df(p)
        if df is None or df.empty:
            continue
        cl = {c.lower().replace(" ", "_"): c for c in df.columns}
        cid_col = cl.get("pubchem_compound_identifiers") or cl.get("pubchem_cid") or cl.get("cid")
        ik_col = cl.get("inchikey") or cl.get("inchi_key")
        if not cid_col:
            continue
        for _, r in df.iterrows():
            v = r.get(cid_col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            try:
                cid = str(int(float(v)))
            except (ValueError, TypeError):
                cid = _safe_str(v)
            if not cid:
                continue
            ik = None
            if ik_col:
                ik = normalize_inchikey(r.get(ik_col))
            if ik:
                cid_to_ik[cid] = ik
    for p in [processed / "canonical" / "compound_master.csv", processed / "canonical" / "compound_master.parquet"]:
        if not p.exists():
            continue
        df = _load_df(p, dtype_str=True)
        if df is None or df.empty:
            continue
        cid_col = next((c for c in df.columns if "cid" in c.lower()), None)
        ik_col = next((c for c in df.columns if "inchikey" in c.lower()), None) or next((c for c in df.columns if c == "compound_id"), None)
        if not cid_col or not ik_col:
            continue
        for _, r in df.iterrows():
            cid = _safe_str(r.get(cid_col))
            if not cid or not cid.isdigit():
                continue
            ik = normalize_inchikey(r.get(ik_col))
            if ik:
                cid_to_ik[cid] = ik
        break
    return cid_to_ik


def resolve_bindingdb_compounds_batch(
    compound_ids: List[str],
    repo_root: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resolve many compound_id_raw; return DataFrame with compound_id_raw, compound_id_type, inchikey, cid, name, source_file, resolver_used and report."""
    repo_root = Path(repo_root).resolve()
    coconut_to_ik = build_coconut_to_inchikey_map(repo_root)
    cid_to_ik = _build_cid_to_inchikey(repo_root)
    pharm = build_pharmgkb_cid_to_inchikey(repo_root, write_csv=True)
    for c, ik in pharm.items():
        cid_to_ik.setdefault(c, ik)
    chembl_to_ik, bindingdb_to_ik, cm_cid, name_to_ik = build_compound_master_maps(repo_root)
    for c, ik in cm_cid.items():
        cid_to_ik.setdefault(c, ik)
    cid_master_map, cid_pharmgkb_map, cid_scan_map = build_all_cid_maps(repo_root)

    rows: List[Dict[str, Any]] = []
    n_ok = 0
    resolver_counts: Dict[str, int] = {}
    for raw in compound_ids:
        raw = _safe_str(raw)
        if not raw:
            continue
        cid_type = "COCONUT" if raw.upper().startswith("COCONUT_") else ("CHEMBL" if raw.upper().startswith("CHEMBL") else ("CID" if (raw.isdigit() or raw.upper().startswith("CID") or (normalize_cid(raw) is not None)) else ("INCHIKEY" if looks_like_inchikey(raw) else "OTHER")))
        ik, cid, smiles, name, source_file, resolver = resolve_bindingdb_compound(
            raw, repo_root, coconut_to_ik, cid_to_ik, chembl_to_ik, bindingdb_to_ik, name_to_ik,
            cid_master_map=cid_master_map, cid_pharmgkb_map=cid_pharmgkb_map, cid_scan_map=cid_scan_map,
        )
        rows.append({
            "compound_id_raw": raw,
            "compound_id_type": cid_type,
            "inchikey": ik or "",
            "cid": cid or "",
            "smiles": smiles or "",
            "name": name or "",
            "source_file": source_file or "",
            "resolver_used": resolver,
        })
        if ik:
            n_ok += 1
        resolver_counts[resolver] = resolver_counts.get(resolver, 0) + 1
    report = {"n_total": len(rows), "n_resolved": n_ok, "pct_resolved": round(100.0 * n_ok / len(rows), 2) if rows else 0, "resolver_counts": resolver_counts}
    return pd.DataFrame(rows), report
