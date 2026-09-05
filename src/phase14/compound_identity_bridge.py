"""
Phase14: Compound Identity Bridge — unified mapping from local repo only (no external APIs).
Builds: source_namespace, source_id, inchikey, cid, smiles, name, provenance_file.
Used by all compound_gene expansions to resolve compound IDs to InChIKey for overlap with ING->CMP.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .compound_identity import (
    load_csv_or_parquet,
    normalize_fdb_id,
    normalize_inchikey,
    looks_like_inchikey,
    looks_like_fdb_id,
    _safe_str,
)

logger = logging.getLogger(__name__)

BRIDGE_COLUMNS = [
    "source_namespace",
    "source_id",
    "inchikey",
    "cid",
    "smiles",
    "name",
    "provenance_file",
]

NAMESPACE_FOODB = "FOODB"
NAMESPACE_COCONUT = "COCONUT"
NAMESPACE_CHEMBL = "CHEMBL"
NAMESPACE_PUBCHEM = "PUBCHEM"
NAMESPACE_NAME = "NAME"


def _add_row(
    rows: List[Dict[str, Any]],
    namespace: str,
    source_id: str,
    inchikey: Optional[str] = None,
    cid: Optional[str] = None,
    smiles: Optional[str] = None,
    name: Optional[str] = None,
    provenance: str = "",
) -> None:
    if not source_id or not source_id.strip():
        return
    source_id = source_id.strip()
    inchikey = (inchikey or "").strip() if inchikey else ""
    if inchikey and len(inchikey) < 25:
        inchikey = ""
    cid = _safe_str(cid)
    smiles = (_safe_str(smiles))[:500] if smiles else ""
    name = (_safe_str(name))[:300] if name else ""
    rows.append({
        "source_namespace": namespace,
        "source_id": source_id,
        "inchikey": inchikey,
        "cid": cid,
        "smiles": smiles,
        "name": name,
        "provenance_file": provenance,
    })


def _ingest_foodb_from_sources(repo_root: Path, processed: Path) -> List[Dict[str, Any]]:
    """Ingest FooDB IDs and resolve to InChIKey from registry/compound_master."""
    rows: List[Dict[str, Any]] = []
    # compound_registry_lookup.json: InChIKey -> compound_id (numeric) => FDB_<id> -> InChIKey
    registry_path = processed / "phase14_chemical_identity" / "compound_registry_lookup.json"
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
                    fdb = normalize_fdb_id(f"FDB_{cid_val}")
                    if fdb:
                        _add_row(rows, NAMESPACE_FOODB, fdb, inchikey=ik_norm, provenance=registry_path.name)
        except Exception as e:
            logger.warning("Bridge: could not load registry %s: %s", registry_path.name, e)
    # compound_master: fdb_id_norm -> inchikey
    for p in [processed / "canonical" / "compound_master.csv", processed / "canonical" / "compound_master.parquet"]:
        if not p.exists():
            continue
        try:
            df = load_csv_or_parquet(p)
            if df is None or df.empty:
                continue
            fdb_col = next((c for c in df.columns if "fdb" in c.lower() and "norm" in c.lower()), None) or next((c for c in df.columns if "fdb" in c.lower()), None)
            ik_col = next((c for c in df.columns if "inchikey" in c.lower()), None)
            if not fdb_col or not ik_col:
                continue
            for _, r in df.iterrows():
                fdb = normalize_fdb_id(r.get(fdb_col))
                ik = _safe_str(r.get(ik_col))
                if fdb and ik and len(ik) >= 25:
                    _add_row(rows, NAMESPACE_FOODB, fdb, inchikey=ik, provenance=p.name)
            break
        except Exception as e:
            logger.warning("Bridge: could not load compound_master %s: %s", p.name, e)
    # food_compound_gene_links: compound_id FDB-style
    fcg = processed / "phase12_genetics" / "food_compound_gene_links.parquet"
    if fcg.exists():
        df = load_csv_or_parquet(fcg)
        if df is not None and not df.empty:
            cmp_col = next((c for c in df.columns if "compound" in c.lower() and "id" in c.lower()), None)
            if cmp_col:
                for v in df[cmp_col].dropna().astype(str).str.strip().unique():
                    if not v:
                        continue
                    fdb = normalize_fdb_id(v) if looks_like_fdb_id(v) else v
                    if fdb:
                        _add_row(rows, NAMESPACE_FOODB, fdb, provenance=fcg.name)
    return rows


def _ingest_coconut(repo_root: Path, processed: Path) -> List[Dict[str, Any]]:
    """Coconut: inchikey_to_compound_id.json and compound_master coconut_id."""
    rows: List[Dict[str, Any]] = []
    json_path = processed / "phase15_coconut" / "inchikey_to_compound_id.json"
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
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
                if "COCONUT_" in s.upper():
                    _add_row(rows, NAMESPACE_COCONUT, s, inchikey=ik_norm, provenance=json_path.name)
        except Exception as e:
            logger.warning("Bridge: could not load inchikey_to_compound_id.json: %s", e)
    for p in [processed / "canonical" / "compound_master.csv", processed / "canonical" / "compound_master.parquet"]:
        if not p.exists():
            continue
        try:
            df = load_csv_or_parquet(p)
            if df is None or df.empty or "coconut_id" not in df.columns:
                continue
            ik_col = next((c for c in df.columns if "inchikey" in c.lower()), None)
            for _, r in df.iterrows():
                coco = _safe_str(r.get("coconut_id"))
                if not coco or "COCONUT_" not in coco.upper():
                    continue
                ik = _safe_str(r.get(ik_col)) if ik_col else ""
                if len(ik) >= 25:
                    _add_row(rows, NAMESPACE_COCONUT, coco, inchikey=ik, provenance=p.name)
            break
        except Exception as e:
            logger.warning("Bridge: could not load compound_master for coconut: %s", e)
    return rows


def _ingest_pharmgkb(processed: Path) -> List[Dict[str, Any]]:
    """PharmGKB chemicals: PubChem Compound Identifiers, Name; link CID->InChIKey from local if present."""
    rows: List[Dict[str, Any]] = []
    phase12 = processed / "phase12_genetics"
    for path in list(phase12.glob("pharmgkb_chemicals.parquet")) + list(phase12.glob("*pharmgkb*.parquet")):
        if not path.is_file():
            continue
        df = load_csv_or_parquet(path)
        if df is None or df.empty:
            continue
        cl = {c.lower().replace(" ", "_"): c for c in df.columns}
        name_col = cl.get("name") or cl.get("compound_name")
        cid_col = cl.get("pubchem_compound_identifiers") or cl.get("cid") or cl.get("pubchem_cid")
        ik_col = cl.get("inchikey") or cl.get("inchi_key")
        for _, r in df.iterrows():
            name = _safe_str(r.get(name_col)) if name_col else ""
            cid = None
            if cid_col:
                v = r.get(cid_col)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    try:
                        cid = str(int(float(v)))
                    except (ValueError, TypeError):
                        cid = _safe_str(v)
            ik = None
            if ik_col:
                v = r.get(ik_col)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    ik = normalize_inchikey(str(v))
            if cid:
                _add_row(rows, NAMESPACE_PUBCHEM, cid, inchikey=ik, cid=cid, name=name or None, provenance=path.name)
            if name:
                _add_row(rows, NAMESPACE_NAME, name, inchikey=ik, cid=cid or "", name=name, provenance=path.name)
    return rows


def _ingest_bindingdb_ids(processed: Path) -> List[Dict[str, Any]]:
    """BindingDB edges: collect compound_id (COCONUT_*); resolution to InChIKey done via bridge merge."""
    rows: List[Dict[str, Any]] = []
    bdb = processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if not bdb.exists():
        return rows
    df = load_csv_or_parquet(bdb)
    if df is None or df.empty or "compound_id" not in df.columns:
        return rows
    for v in df["compound_id"].dropna().astype(str).str.strip().unique():
        if not v:
            continue
        if str(v).startswith("COCONUT_"):
            _add_row(rows, NAMESPACE_COCONUT, str(v), provenance=bdb.name)
        elif looks_like_inchikey(v):
            _add_row(rows, NAMESPACE_COCONUT, str(v), inchikey=normalize_inchikey(v), provenance=bdb.name)
    return rows


def build_bridge(repo_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build unified compound identity bridge from local repo only.
    Returns (bridge_df, report_dict).
    """
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    if not processed.exists():
        return pd.DataFrame(columns=BRIDGE_COLUMNS), {"error": "data/processed not found", "sources": {}}

    all_rows: List[Dict[str, Any]] = []
    counts_per_source: Dict[str, int] = {}
    resolved_per_source: Dict[str, int] = {}

    # 1) FooDB
    foodb_rows = _ingest_foodb_from_sources(repo_root, processed)
    all_rows.extend(foodb_rows)
    counts_per_source["FOODB"] = len(foodb_rows)
    resolved_per_source["FOODB"] = sum(1 for r in foodb_rows if r.get("inchikey") and len(r.get("inchikey", "")) >= 25)

    # 2) Coconut
    coconut_rows = _ingest_coconut(repo_root, processed)
    all_rows.extend(coconut_rows)
    counts_per_source["COCONUT"] = len(coconut_rows)
    resolved_per_source["COCONUT"] = sum(1 for r in coconut_rows if r.get("inchikey") and len(r.get("inchikey", "")) >= 25)

    # 3) PharmGKB
    pharm_rows = _ingest_pharmgkb(processed)
    all_rows.extend(pharm_rows)
    counts_per_source["PharmGKB"] = len(pharm_rows)
    resolved_per_source["PharmGKB"] = sum(1 for r in pharm_rows if r.get("inchikey") and len(r.get("inchikey", "")) >= 25)

    # 4) BindingDB (ids only; resolution via COCONUT in bridge)
    bdb_rows = _ingest_bindingdb_ids(processed)
    all_rows.extend(bdb_rows)
    counts_per_source["BindingDB_ids"] = len(bdb_rows)

    if not all_rows:
        return pd.DataFrame(columns=BRIDGE_COLUMNS), {"sources": counts_per_source, "resolved": resolved_per_source, "n_total": 0, "n_with_inchikey": 0, "pct_resolved": 0.0}

    bridge_df = pd.DataFrame(all_rows)
    # Dedupe by (source_namespace, source_id), keeping first (so InChIKey from best source)
    bridge_df = bridge_df.drop_duplicates(subset=["source_namespace", "source_id"], keep="first").reset_index(drop=True)

    # Fill inchikey for COCONUT/BindingDB by merging on source_id from rows that have inchikey
    source_id_to_ik: Dict[str, str] = {}
    for _, r in bridge_df.iterrows():
        ik = _safe_str(r.get("inchikey"))
        sid = _safe_str(r.get("source_id"))
        if ik and len(ik) >= 25 and sid:
            source_id_to_ik[sid] = ik
            if sid.startswith("COCONUT_") and "." in sid:
                base = sid.split(".")[0]
                if base and base not in source_id_to_ik:
                    source_id_to_ik[base] = ik
    def fill_ik(r):
        ik = _safe_str(r.get("inchikey"))
        if ik and len(ik) >= 25:
            return ik
        sid = _safe_str(r.get("source_id"))
        return source_id_to_ik.get(sid) or source_id_to_ik.get(sid.split(".")[0] if "." in sid else sid) or ""
    bridge_df["inchikey"] = bridge_df.apply(fill_ik, axis=1)

    n_total = len(bridge_df)
    n_with_inchikey = int((bridge_df["inchikey"].astype(str).str.len() >= 25).sum())
    pct_resolved = round(100.0 * n_with_inchikey / n_total, 2) if n_total else 0.0
    resolved_per_source["BindingDB_ids"] = n_with_inchikey  # approximate; bridge now has filled IK for some

    report = {
        "n_total": n_total,
        "n_with_inchikey": n_with_inchikey,
        "pct_resolved": pct_resolved,
        "counts_per_source": counts_per_source,
        "resolved_per_source": resolved_per_source,
    }
    return bridge_df, report


def write_bridge(repo_root: Path) -> Tuple[Path, Path]:
    """Build bridge, write CSV and report JSON. Returns (bridge_path, report_path)."""
    repo_root = Path(repo_root).resolve()
    canonical_dir = repo_root / "data" / "processed" / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    bridge_df, report = build_bridge(repo_root)
    bridge_path = canonical_dir / "compound_identity_bridge.csv"
    report_path = canonical_dir / "compound_identity_bridge_report.json"
    if not bridge_df.empty:
        bridge_df.to_csv(bridge_path, index=False)
        logger.info("Wrote %s: rows=%s, with InChIKey=%s (%.1f%%)", bridge_path.name, len(bridge_df), report["n_with_inchikey"], report["pct_resolved"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", report_path.name)
    return bridge_path, report_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build compound identity bridge from local repo")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()
    write_bridge(args.repo_root)
