"""
Phase16: Compound→gene expansion v2 with target→gene bridging and evidence scoring.
Uses compound_master_v2 + resolve_to_inchikey; builds uniprot_id->gene_symbol from local files;
outputs compound_gene_expanded_v2_raw.csv, compound_gene_expanded_v2_canonical.csv, report JSON.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

from .compound_master_v2 import _load_df, _col, normalize_inchikey, looks_like_inchikey
from .resolve_compound_id import resolve_to_inchikey

CANONICAL_COLS = [
    "inchikey", "gene_symbol", "evidence_source", "evidence_strength", "resolution_method",
    "original_compound_id", "original_name", "pubchem_cid", "chembl_id", "coconut_id", "fdb_id_norm",
    "uniprot_id", "target_name",
]


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def build_uniprot_to_gene(repo_root: Path) -> Dict[str, str]:
    """Scan processed data for files with uniprot + gene_symbol/gene_name; build uniprot_id -> gene_symbol."""
    processed = Path(repo_root) / "data" / "processed"
    if not processed.exists():
        return {}
    uniprot_to_gene: Dict[str, str] = {}
    for path in list(processed.rglob("*.parquet")) + list(processed.rglob("*.csv")):
        if path.stat().st_size > 50_000_000:
            continue
        try:
            df = _load_df(path)
            if df is None or df.empty or len(df) > 500_000:
                continue
        except Exception:
            continue
        uniprot_col = _col(df, ["uniprot_id", "uniprot", "uniprot_accession", "uniprot_accession_x", "uniprot_accession_y"])
        gene_col = _col(df, ["gene_symbol", "gene_name", "gene"])
        if not uniprot_col or not gene_col:
            continue
        for _, r in df.iterrows():
            u = _safe_str(r.get(uniprot_col))
            g = _safe_str(r.get(gene_col))
            if u and g and len(u) <= 20 and len(g) <= 30:
                u = u.upper()
                g = g.upper()
                if u not in uniprot_to_gene or len(g) < len(uniprot_to_gene.get(u, "")):
                    uniprot_to_gene[u] = g
    logger.info("build_uniprot_to_gene: %s mappings from repo", len(uniprot_to_gene))
    return uniprot_to_gene


def _evidence_strength_genetics() -> float:
    return 1.0


def _evidence_strength_bindingdb(affinity_nM: Optional[float], organism: Optional[str]) -> float:
    base = 0.5
    if organism and "homo sapiens" not in str(organism).lower() and str(organism).strip():
        return 0.0
    if affinity_nM is not None and not (isinstance(affinity_nM, float) and pd.isna(affinity_nM)):
        try:
            anm = float(affinity_nM)
            if anm <= 100:
                return base + 0.3
            if anm <= 1000:
                return base + 0.1
        except (TypeError, ValueError):
            pass
    return base


def _evidence_strength_chembl(pchembl_or_potency: Optional[float]) -> float:
    base = 0.5
    if pchembl_or_potency is not None and not (isinstance(pchembl_or_potency, float) and pd.isna(pchembl_or_potency)):
        try:
            p = float(pchembl_or_potency)
            if p >= 7:
                return base + 0.3
            if p >= 6:
                return base + 0.1
        except (TypeError, ValueError):
            pass
    return base


def load_genetics_edges(repo_root: Path, master_df: pd.DataFrame) -> pd.DataFrame:
    """Load food_compound_gene_links; resolve to inchikey; return rows with CANONICAL_COLS."""
    path = Path(repo_root) / "data" / "processed" / "phase12_genetics" / "food_compound_gene_links.parquet"
    if not path.exists():
        for p in Path(repo_root).rglob("food_compound_gene_links.parquet"):
            path = p
            break
        else:
            return pd.DataFrame(columns=CANONICAL_COLS)
    df = _load_df(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=CANONICAL_COLS)
    cmp_col = _col(df, ["compound_id", "inchikey"])
    name_col = _col(df, ["compound_name", "name"])
    gene_col = _col(df, ["gene_symbol", "gene_id", "gene"])
    if not cmp_col or not gene_col:
        return pd.DataFrame(columns=CANONICAL_COLS)
    rows = []
    for _, r in df.iterrows():
        cid = _safe_str(r.get(cmp_col))
        g = _safe_str(r.get(gene_col))
        name = _safe_str(r.get(name_col)) if name_col else ""
        if not cid or not g:
            continue
        ik, method = resolve_to_inchikey(cid, name=name, master_df=master_df)
        if not ik:
            ik = normalize_inchikey(cid) if looks_like_inchikey(cid) else ""
        if not ik:
            continue
        rows.append({
            "inchikey": ik,
            "gene_symbol": g.upper(),
            "evidence_source": "food_compound_gene_links",
            "evidence_strength": _evidence_strength_genetics(),
            "resolution_method": method,
            "original_compound_id": cid,
            "original_name": name,
            "pubchem_cid": "",
            "chembl_id": "",
            "coconut_id": "",
            "fdb_id_norm": "",
            "uniprot_id": "",
            "target_name": "",
        })
    return pd.DataFrame(rows)


def load_bindingdb_edges(
    repo_root: Path,
    master_df: pd.DataFrame,
    uniprot_to_gene: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load BindingDB edges; resolve compound to inchikey; map target to gene_symbol; keep human or blank organism. Return (df, diagnostics)."""
    path = Path(repo_root) / "data" / "processed" / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if not path.exists():
        for p in Path(repo_root).rglob("compound_target_edges_bindingdb.parquet"):
            path = p
            break
        else:
            return pd.DataFrame(columns=CANONICAL_COLS), {"n_total": 0, "n_dropped_organism": 0, "n_dropped_no_gene": 0, "n_resolved": 0}
    df = _load_df(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=CANONICAL_COLS), {"n_total": 0}
    cmp_col = _col(df, ["compound_id"])
    target_col = _col(df, ["target_name"])
    uniprot_col = _col(df, ["uniprot_id", "uniprot"])
    organism_col = _col(df, ["organism"])
    affinity_col = _col(df, ["affinity_nM"])
    if not cmp_col:
        return pd.DataFrame(columns=CANONICAL_COLS), {"n_total": len(df)}
    n_total = len(df)
    n_drop_org = 0
    n_drop_gene = 0
    n_resolved = 0
    rows = []
    for _, r in df.iterrows():
        organism = _safe_str(r.get(organism_col)) if organism_col else ""
        if organism and "homo sapiens" not in organism.lower() and organism.upper() != "HUMAN":
            n_drop_org += 1
            continue
        cid = _safe_str(r.get(cmp_col))
        if not cid:
            continue
        uniprot = _safe_str(r.get(uniprot_col)) if uniprot_col else ""
        target_name = _safe_str(r.get(target_col)) if target_col else ""
        gene = uniprot_to_gene.get(uniprot.upper()) if uniprot else ""
        if not gene and target_name and len(target_name) <= 15 and target_name.upper().isalpha():
            gene = target_name.upper()
        if not gene:
            gene = uniprot.upper() if uniprot else ""
        if not gene:
            n_drop_gene += 1
            continue
        try:
            aff = r.get(affinity_col)
            aff_float = float(aff) if aff is not None and not (isinstance(aff, float) and pd.isna(aff)) else None
        except (TypeError, ValueError):
            aff_float = None
        ik, method = resolve_to_inchikey(cid, master_df=master_df)
        if not ik:
            ik = normalize_inchikey(cid) if looks_like_inchikey(cid) else ""
        if not ik:
            continue
        n_resolved += 1
        strength = _evidence_strength_bindingdb(aff_float, organism or None)
        if strength == 0:
            continue
        rows.append({
            "inchikey": ik,
            "gene_symbol": gene,
            "evidence_source": "bindingdb",
            "evidence_strength": strength,
            "resolution_method": method,
            "original_compound_id": cid,
            "original_name": "",
            "pubchem_cid": "",
            "chembl_id": "",
            "coconut_id": "",
            "fdb_id_norm": "",
            "uniprot_id": uniprot,
            "target_name": target_name,
        })
    diag = {"n_total": n_total, "n_dropped_organism": n_drop_org, "n_dropped_no_gene": n_drop_gene, "n_resolved": n_resolved}
    return pd.DataFrame(rows), diag


def load_chembl_edges_if_present(
    repo_root: Path,
    master_df: pd.DataFrame,
    uniprot_to_gene: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """If ChEMBL compound-target/activity file exists, load and resolve to (inchikey, gene_symbol)."""
    processed = Path(repo_root) / "data" / "processed"
    canonical = processed / "canonical"
    # Prefer compound_targets.parquet or compound_target_edges with gene/uniprot
    for name in ("compound_targets.parquet", "compound_target_edges.parquet", "targets.parquet"):
        path = canonical / name
        if not path.exists():
            continue
        df = _load_df(path)
        if df is None or df.empty:
            continue
        cmp_col = _col(df, ["compound_id", "inchikey"])
        gene_col = _col(df, ["gene_name", "gene_symbol", "gene"])
        uniprot_col = _col(df, ["uniprot_accession", "uniprot_id"])
        potency_col = _col(df, ["pchembl_value", "standard_value"])
        if not cmp_col:
            continue
        rows = []
        for _, r in df.iterrows():
            cid = _safe_str(r.get(cmp_col))
            if not cid:
                continue
            g = _safe_str(r.get(gene_col)) if gene_col else ""
            u = _safe_str(r.get(uniprot_col)) if uniprot_col else ""
            if not g and u:
                g = uniprot_to_gene.get(u.upper(), "")
            if not g:
                g = u.upper() if u else ""
            if not g:
                continue
            ik, method = resolve_to_inchikey(cid, master_df=master_df)
            if not ik:
                ik = normalize_inchikey(cid) if looks_like_inchikey(cid) else ""
            if not ik:
                continue
            pot = r.get(potency_col) if potency_col else None
            try:
                pot_f = float(pot) if pot is not None and not (isinstance(pot, float) and pd.isna(pot)) else None
            except (TypeError, ValueError):
                pot_f = None
            strength = _evidence_strength_chembl(pot_f)
            rows.append({
                "inchikey": ik,
                "gene_symbol": g.upper(),
                "evidence_source": "chembl",
                "evidence_strength": strength,
                "resolution_method": method,
                "original_compound_id": cid,
                "original_name": "",
                "pubchem_cid": "",
                "chembl_id": "",
                "coconut_id": "",
                "fdb_id_norm": "",
                "uniprot_id": u,
                "target_name": "",
            })
        if rows:
            logger.info("ChEMBL edges: %s from %s", len(rows), name)
            return pd.DataFrame(rows), {"n_loaded": len(rows), "source": name}
    return pd.DataFrame(columns=CANONICAL_COLS), {}


def build_expanded_v2(
    repo_root: Path,
    master_df: Optional[pd.DataFrame] = None,
    master_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build raw and canonical compound→gene tables v2; return (raw_df, canonical_df, report)."""
    repo_root = Path(repo_root).resolve()
    if master_df is None:
        path = master_path or (repo_root / "data" / "processed" / "canonical" / "compound_master_v2.parquet")
        if not path.exists():
            path = repo_root / "data" / "processed" / "canonical" / "compound_master_v2.csv"
        master_df = _load_df(path) if path.exists() else pd.DataFrame()
    uniprot_to_gene = build_uniprot_to_gene(repo_root)
    report = {"n_edges_raw": 0, "n_edges_canonical": 0, "n_unique_compounds_raw": 0, "n_unique_compounds_canonical": 0, "n_unique_genes": 0, "resolution_rates": {}, "n_dropped_organism": 0, "n_dropped_no_gene": 0, "sources": {}}
    all_raw: List[Dict[str, Any]] = []
    genetics_df = load_genetics_edges(repo_root, master_df)
    if not genetics_df.empty:
        report["sources"]["food_compound_gene_links"] = len(genetics_df)
        for _, r in genetics_df.iterrows():
            all_raw.append(r.to_dict())
    bdb_df, bdb_diag = load_bindingdb_edges(repo_root, master_df, uniprot_to_gene)
    if not bdb_df.empty:
        report["sources"]["bindingdb"] = len(bdb_df)
        report["n_dropped_organism"] = bdb_diag.get("n_dropped_organism", 0)
        report["n_dropped_no_gene"] = bdb_diag.get("n_dropped_no_gene", 0)
        report["resolution_rates"]["bindingdb"] = bdb_diag
        for _, r in bdb_df.iterrows():
            all_raw.append(r.to_dict())
    chembl_df, chembl_diag = load_chembl_edges_if_present(repo_root, master_df, uniprot_to_gene)
    if not chembl_df.empty:
        report["sources"]["chembl"] = len(chembl_df)
        report["resolution_rates"]["chembl"] = chembl_diag
        for _, r in chembl_df.iterrows():
            all_raw.append(r.to_dict())
    if not all_raw:
        raw_df = pd.DataFrame(columns=CANONICAL_COLS)
        canonical_df = raw_df.copy()
        return raw_df, canonical_df, report
    raw_df = pd.DataFrame(all_raw)
    report["n_edges_raw"] = len(raw_df)
    report["n_unique_compounds_raw"] = int(raw_df["inchikey"].nunique()) if "inchikey" in raw_df.columns else 0
    # Dedupe by (inchikey, gene_symbol), keep max evidence_strength
    canonical_df = raw_df.sort_values("evidence_strength", ascending=False).drop_duplicates(subset=["inchikey", "gene_symbol"], keep="first")
    report["n_edges_canonical"] = len(canonical_df)
    report["n_unique_compounds_canonical"] = int(canonical_df["inchikey"].nunique())
    report["n_unique_genes"] = int(canonical_df["gene_symbol"].nunique())
    return raw_df, canonical_df, report


def write_expanded_v2(
    repo_root: Path,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    """Build and write raw CSV, canonical CSV, report JSON. Returns (raw_path, canonical_path, report_path, report). Exit nonzero only if no inchikey resolved for any source."""
    repo_root = Path(repo_root).resolve()
    raw_df, canonical_df, report = build_expanded_v2(repo_root)
    out_dir = output_dir or (repo_root / "data" / "processed" / "canonical")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "compound_gene_expanded_v2_raw.csv"
    canon_path = out_dir / "compound_gene_expanded_v2_canonical.csv"
    report_path = out_dir / "compound_gene_expansion_v2_report.json"
    raw_df.to_csv(raw_path, index=False)
    canonical_df.to_csv(canon_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote compound_gene_expanded_v2: raw=%s canonical=%s compounds=%s", len(raw_df), len(canonical_df), report.get("n_unique_compounds_canonical"))
    return raw_path, canon_path, report_path, report
