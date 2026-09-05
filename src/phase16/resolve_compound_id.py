"""
Canonical resolver: resolve_to_inchikey(compound_id, name=None, cid=None, extra_ids=dict, master_df) with ordered fallbacks.
(1) direct inchikey (2) compound_id matches inchikey pattern (3) coconut_id/base (4) chembl_id (5) fdb_id_norm
(6) pharmgkb_id (7) pubchem cid (8) normalized_name exact (9) normalized_name fuzzy (rapidfuzz >= 95), log match.
Returns (inchikey|None, resolution_method).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

from .compound_master_v2 import (
    normalize_inchikey,
    looks_like_inchikey,
    normalize_fdb_id,
    normalize_coconut_id,
    normalize_chembl_id,
    normalized_name,
)


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _col(df: pd.DataFrame, names: list) -> Optional[str]:
    if df is None or df.empty:
        return None
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def resolve_to_inchikey(
    compound_id: str,
    name: Optional[str] = None,
    cid: Optional[str] = None,
    extra_ids: Optional[Dict[str, str]] = None,
    master_df: Optional[pd.DataFrame] = None,
    fuzzy_threshold: int = 95,
) -> Tuple[Optional[str], str]:
    """
    Resolve to canonical InChIKey using ordered fallbacks.
    Returns (inchikey or None, resolution_method).
    """
    compound_id = _safe_str(compound_id)
    name = _safe_str(name) if name is not None else ""
    cid = _safe_str(cid) if cid is not None else ""
    extra_ids = extra_ids or {}
    if master_df is None or master_df.empty:
        if compound_id and looks_like_inchikey(compound_id):
            return normalize_inchikey(compound_id), "inchikey_direct"
        return None, "no_master"

    # (1) direct inchikey in input
    if compound_id and looks_like_inchikey(compound_id):
        ik = normalize_inchikey(compound_id)
        if ik:
            return ik, "inchikey_direct"
    # (2) compound_id matches inchikey pattern (already covered)

    # Build lookup series from master
    ik_col = _col(master_df, ["compound_id", "inchikey"])
    if not ik_col:
        return None, "no_inchikey_column"
    master_ik = master_df[ik_col].dropna().astype(str).str.strip()
    master_ik = master_ik[master_ik.str.len() >= 25]
    coconut_col = _col(master_df, ["coconut_id", "coconut_base"])
    chembl_col = _col(master_df, ["chembl_id"])
    fdb_col = _col(master_df, ["fdb_id_norm"])
    pharmgkb_col = _col(master_df, ["pharmgkb_id"])
    cid_col = _col(master_df, ["pubchem_cid"])
    name_col = _col(master_df, ["normalized_name", "name"])

    # (3) coconut_id / coconut_base
    coco_full, coco_base = normalize_coconut_id(compound_id)
    if coco_full or coco_base:
        for c in (coco_base, coco_full):
            if not c:
                continue
            if coconut_col:
                match = master_df[master_df[coconut_col].astype(str).str.strip() == c]
                if not match.empty and ik_col:
                    ik = _safe_str(match.iloc[0][ik_col])
                    if looks_like_inchikey(ik):
                        return normalize_inchikey(ik), "coconut_id"
            # also check compound_id column for COCONUT string
            match = master_df[master_df[ik_col].astype(str).str.strip() == c]
            if not match.empty:
                # compound_id in master might be the coconut id when no inchikey
                pass
    if compound_id.upper().startswith("COCONUT_"):
        if coconut_col:
            match = master_df[master_df[coconut_col].astype(str).str.strip().str.upper() == compound_id.upper()]
            if not match.empty:
                row = match.iloc[0]
                ik = _safe_str(row.get(ik_col))
                if ik and looks_like_inchikey(ik):
                    return normalize_inchikey(ik), "coconut_id"
                # row may have compound_id = inchikey in another column
                for col in master_df.columns:
                    if "inchikey" in col.lower() or col == "compound_id":
                        v = row.get(col)
                        if v and looks_like_inchikey(str(v)):
                            return normalize_inchikey(v), "coconut_id"

    # (4) chembl_id
    chembl = normalize_chembl_id(compound_id) or extra_ids.get("chembl_id")
    if chembl and chembl_col:
        match = master_df[master_df[chembl_col].astype(str).str.strip().str.upper() == chembl.upper()]
        if not match.empty and ik_col:
            ik = _safe_str(match.iloc[0][ik_col])
            if ik and looks_like_inchikey(ik):
                return normalize_inchikey(ik), "chembl_id"

    # (5) fdb_id_norm
    fdb = normalize_fdb_id(compound_id) or extra_ids.get("fdb_id_norm")
    if fdb and fdb_col:
        match = master_df[master_df[fdb_col].astype(str).str.strip() == fdb]
        if not match.empty and ik_col:
            ik = _safe_str(match.iloc[0][ik_col])
            if ik and looks_like_inchikey(ik):
                return normalize_inchikey(ik), "fdb_id_norm"

    # (6) pharmgkb_id
    pa = extra_ids.get("pharmgkb_id") or (compound_id if compound_id.upper().startswith("PA") else "")
    if pa and pharmgkb_col:
        match = master_df[master_df[pharmgkb_col].astype(str).str.strip() == _safe_str(pa)]
        if not match.empty and ik_col:
            ik = _safe_str(match.iloc[0][ik_col])
            if ik and looks_like_inchikey(ik):
                return normalize_inchikey(ik), "pharmgkb_id"

    # (7) pubchem cid
    cid_val = cid or extra_ids.get("pubchem_cid") or (compound_id if compound_id.isdigit() else "")
    if cid_val and cid_col:
        cid_clean = str(int(float(cid_val))) if str(cid_val).replace(".", "").isdigit() else cid_val
        match = master_df[master_df[cid_col].astype(str).str.strip() == cid_clean]
        if not match.empty and ik_col:
            ik = _safe_str(match.iloc[0][ik_col])
            if ik and looks_like_inchikey(ik):
                return normalize_inchikey(ik), "pubchem_cid"

    # (8) normalized_name exact
    nm = normalized_name(name or compound_id)
    if nm and name_col:
        master_norm = master_df[name_col].dropna().astype(str).apply(lambda x: normalized_name(x))
        match = master_df[master_norm == nm]
        if not match.empty and ik_col:
            ik = _safe_str(match.iloc[0][ik_col])
            if ik and looks_like_inchikey(ik):
                return normalize_inchikey(ik), "normalized_name_exact"

    # (9) normalized_name fuzzy (rapidfuzz >= 95)
    if _HAS_RAPIDFUZZ and nm and name_col:
        best_score = 0
        best_ik = None
        for _, r in master_df.iterrows():
            m_norm = normalized_name(r.get(name_col))
            if not m_norm:
                continue
            score = fuzz.ratio(nm, m_norm)
            if score >= fuzzy_threshold and score > best_score:
                ik = _safe_str(r.get(ik_col))
                if ik and looks_like_inchikey(ik):
                    best_score = score
                    best_ik = normalize_inchikey(ik)
        if best_ik:
            logger.info("resolve_to_inchikey fuzzy name match score=%s -> %s", best_score, best_ik[:20])
            return best_ik, "normalized_name_fuzzy"

    return None, "unresolved"
