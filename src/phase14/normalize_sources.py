"""
Phase14: Canonicalize compound->gene/target sources to strict output schema.
Deterministic normalization; no heuristics-only logic.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Strict output schema for compound_gene_links (compound_id = join key: inchikey when present, else cid/chembl/original)
CANONICAL_COMPOUND_GENE_COLUMNS = [
    "inchikey", "compound_id", "gene", "evidence", "source",
    "uniprot", "cid", "chembl_id", "gene_kind",
]

COMPOUND_INPUT_COLS = [
    "compound_id", "chemical_id", "inchikey", "inchi_key", "InChIKey", "pubchem_cid", "cid", "chembl_id",
]
GENE_INPUT_COLS = [
    "gene", "gene_symbol", "symbol", "target", "target_name", "uniprot", "uniprot_id",
]

# InChIKey: 27 chars with two dashes
INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def _normalize_inchikey_val(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper()
    if not s:
        return None
    s = s.replace("-", "")
    if len(s) == 25:
        return f"{s[:14]}-{s[14:24]}-{s[24]}"
    if len(s) == 27 and "-" in str(val):
        return s
    if len(s) == 27:
        return f"{s[:14]}-{s[14:24]}-{s[24]}"
    return None if len(s) < 25 else s[:27]


def _looks_like_inchikey(s: str) -> bool:
    if not s or len(s) < 25:
        return False
    u = s.strip().upper().replace("-", "")
    return len(u) >= 25 and u[:14].isalpha() and u[14:24].isalnum() and u[24:].isalnum()


def _pick_compound_col(df: pd.DataFrame) -> Optional[str]:
    cl = {c.lower(): c for c in df.columns}
    for k in COMPOUND_INPUT_COLS:
        if k.lower() in cl:
            return cl[k.lower()]
    return None


def _pick_gene_col(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], str]:
    """Return (gene_col, uniprot_col, gene_kind). gene_kind is 'gene' or 'uniprot'."""
    cl = {c.lower(): c for c in df.columns}
    for k in ["gene_symbol", "gene", "symbol", "gene_id"]:
        if k in cl:
            return cl[k], None, "gene"
    for k in ["target", "target_name", "uniprot", "uniprot_id"]:
        if k in cl:
            return cl[k], cl[k] if "uniprot" in k or "target" in k else None, "uniprot"
    return None, None, "gene"


def _find_uniprot_col(df: pd.DataFrame) -> Optional[str]:
    cl = {c.lower(): c for c in df.columns}
    for k in ["uniprot", "uniprot_id", "target", "target_name"]:
        if k in cl:
            return cl[k]
    return None


def canonicalize_compound_gene(df: pd.DataFrame, source_path: str) -> pd.DataFrame:
    """
    Normalize a compound->gene/target dataframe to strict canonical schema.
    Output columns: inchikey (uppercase), gene (uppercase), evidence, source,
    optional uniprot, optional cid, optional chembl_id, gene_kind.

    Accepts compound identifiers: inchikey, inchi_key, InChIKey, pubchem_cid, cid, chembl_id, compound_id.
    Accepts gene/target: gene, gene_symbol, symbol, target, target_name, uniprot, uniprot_id.
    If only uniprot/target exist, store in uniprot and set gene=uniprot, gene_kind="uniprot".
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=CANONICAL_COMPOUND_GENE_COLUMNS)

    source_name = Path(source_path).name if source_path else "unknown"
    cmp_col = _pick_compound_col(df)
    gene_col, uniprot_col, gene_kind_default = _pick_gene_col(df)
    uniprot_col = uniprot_col or _find_uniprot_col(df)

    if not cmp_col:
        logger.warning("canonicalize_compound_gene: no compound column in %s (columns: %s)", source_name, list(df.columns))
        return pd.DataFrame(columns=CANONICAL_COMPOUND_GENE_COLUMNS)
    if not gene_col and not uniprot_col:
        logger.warning("canonicalize_compound_gene: no gene/target column in %s", source_name)
        return pd.DataFrame(columns=CANONICAL_COMPOUND_GENE_COLUMNS)

    # Optional inchikey-specific column
    ik_col = None
    for c in df.columns:
        if c.lower() in ("inchikey", "inchi_key"):
            ik_col = c
            break

    # Optional cid/chembl
    cid_col = None
    chembl_col = None
    for c in df.columns:
        if c.lower() in ("pubchem_cid", "cid"):
            cid_col = cid_col or c
        if c.lower() == "chembl_id":
            chembl_col = c

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        cmp_val = r.get(cmp_col)
        if cmp_val is None or (isinstance(cmp_val, float) and pd.isna(cmp_val)):
            continue
        cmp_str = str(cmp_val).strip()
        if not cmp_str:
            continue

        inchikey = _normalize_inchikey_val(r.get(ik_col) if ik_col else None)
        if not inchikey and _looks_like_inchikey(cmp_str):
            inchikey = _normalize_inchikey_val(cmp_str)
        if not inchikey:
            inchikey = _normalize_inchikey_val(cmp_str)
        cid_val = None
        if cid_col:
            v = r.get(cid_col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                cid_val = str(int(v)) if isinstance(v, (int, float)) else str(v).strip()
        chembl_val = None
        if chembl_col:
            v = r.get(chembl_col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                chembl_val = str(v).strip()

        gene_val = None
        uniprot_val = None
        gene_kind = gene_kind_default
        if gene_col:
            v = r.get(gene_col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                gene_val = str(v).strip().upper()
        if uniprot_col and uniprot_col != gene_col:
            v = r.get(uniprot_col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                uniprot_val = str(v).strip().upper()
        if not gene_val and uniprot_val:
            gene_val = uniprot_val
            gene_kind = "uniprot"
        if not gene_val:
            continue

        if not inchikey and (cid_val or chembl_val or cmp_str):
            inchikey = ""
        join_key = inchikey or cid_val or chembl_val or cmp_str
        rows.append({
            "inchikey": inchikey or "",
            "compound_id": str(join_key).strip().upper() if join_key else "",
            "gene": gene_val,
            "evidence": "canonicalized",
            "source": source_name,
            "uniprot": uniprot_val or "",
            "cid": cid_val or "",
            "chembl_id": chembl_val or "",
            "gene_kind": gene_kind,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=CANONICAL_COMPOUND_GENE_COLUMNS)
    out = out.drop_duplicates(subset=["compound_id", "gene"])
    out = out[out["gene"].str.len() > 0]
    out = out[out["compound_id"].str.len() > 0]
    for c in CANONICAL_COMPOUND_GENE_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out[[c for c in CANONICAL_COMPOUND_GENE_COLUMNS if c in out.columns]]
