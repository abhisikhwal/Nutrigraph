"""
Phase16: Compound→gene expansion using only repo data.

- Load PharmGKB chemicals; resolve to InChIKey via CID map and name fuzzy match.
- Expand compound→gene edges from food_compound_gene_links + BindingDB; resolve BindingDB compounds to InChIKey.
- Output: compound_gene_expanded_raw.csv, compound_gene_expanded_canonical.csv, compound_gene_expansion_report.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Optional fuzzy matching
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# Reuse Phase14 helpers when available
def _normalize_inchikey(val: Any) -> Optional[str]:
    try:
        from src.phase14.compound_identity import normalize_inchikey as _ik
        return _ik(val)
    except Exception:
        pass
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


def _load_csv_or_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False, dtype=str)
    except Exception as e:
        logger.warning("Load failed %s: %s", path, e)
        return None


def load_pharmgkb_chemicals(repo_root: Path) -> pd.DataFrame:
    """
    Load PharmGKB chemicals from data/processed/phase12_genetics/pharmgkb_chemicals.parquet.
    Returns df with columns: pharmgkb_id, name, cid (PubChem), smiles, inchikey (if derivable), source="pharmgkb_chemicals".
    """
    path = Path(repo_root) / "data" / "processed" / "phase12_genetics" / "pharmgkb_chemicals.parquet"
    if not path.exists():
        for p in Path(repo_root).glob("**/pharmgkb_chemicals.parquet"):
            path = p
            break
        else:
            logger.info("PharmGKB chemicals not found; returning empty DataFrame")
            return pd.DataFrame(columns=["pharmgkb_id", "name", "cid", "smiles", "inchikey", "source"])
    df = _load_csv_or_parquet(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["pharmgkb_id", "name", "cid", "smiles", "inchikey", "source"])
    cl = {c.lower().replace(" ", "_"): c for c in df.columns}
    pharmgkb_id_col = cl.get("pharmgkb_id") or cl.get("chemical_id") or cl.get("id")
    name_col = cl.get("name") or cl.get("compound_name") or cl.get("chemical_name") or cl.get("drug_name")
    cid_col = cl.get("cid") or cl.get("pubchem_cid") or cl.get("pubchem_id")
    smiles_col = cl.get("smiles") or cl.get("canonical_smiles")
    ik_col = cl.get("inchikey") or cl.get("inchi_key")
    out = []
    for _, r in df.iterrows():
        row = {
            "pharmgkb_id": _safe_str(r.get(pharmgkb_id_col)) if pharmgkb_id_col else "",
            "name": _safe_str(r.get(name_col)) if name_col else "",
            "cid": _safe_str(r.get(cid_col)) if cid_col else "",
            "smiles": _safe_str(r.get(smiles_col)) if smiles_col else "",
            "inchikey": "",
            "source": "pharmgkb_chemicals",
        }
        if ik_col and r.get(ik_col):
            ik = _normalize_inchikey(r.get(ik_col))
            if ik:
                row["inchikey"] = ik
        if row["cid"] and row["cid"].isdigit():
            row["cid"] = str(int(float(row["cid"])))
        out.append(row)
    out_df = pd.DataFrame(out)
    logger.info("load_pharmgkb_chemicals: %s rows from %s", len(out_df), path.name)
    return out_df


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def build_cid_to_inchikey(compound_master: pd.DataFrame) -> Dict[str, str]:
    """Build map cid -> inchikey from compound_master (repo-derived only)."""
    cid_col = _col(compound_master, ["cid", "pubchem_cid", "pubchem_id"])
    ik_col = _col(compound_master, ["inchikey", "inchi_key"])
    if not cid_col or not ik_col:
        return {}
    out = {}
    for _, r in compound_master.iterrows():
        cid = r.get(cid_col)
        ik = r.get(ik_col)
        if cid is None or pd.isna(cid) or ik is None or pd.isna(ik):
            continue
        try:
            cid_s = str(int(float(cid)))
        except (ValueError, TypeError):
            cid_s = str(cid).strip()
        if not cid_s:
            continue
        ik_norm = _normalize_inchikey(ik)
        if ik_norm:
            out[cid_s] = ik_norm
    logger.info("build_cid_to_inchikey: %s entries from compound_master", len(out))
    return out


def _col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def resolve_pharmgkb_to_inchikey(
    pharmgkb_df: pd.DataFrame,
    cid_to_ik: Dict[str, str],
    compound_master: pd.DataFrame,
    fuzzy_threshold: int = 95,
) -> pd.DataFrame:
    """
    Add inchikey to pharmgkb_df using:
    A) CID -> InChIKey map
    B) name exact / fuzzy match against compound_master.name (rapidfuzz >= fuzzy_threshold)
    C) keep unresolved with reason
    """
    out = pharmgkb_df.copy()
    if "inchikey" not in out.columns:
        out["inchikey"] = ""
    if "resolution_reason" not in out.columns:
        out["resolution_reason"] = ""
    name_col_master = _col(compound_master, ["name", "compound_name"])
    ik_col_master = _col(compound_master, ["inchikey", "inchi_key"])
    if not name_col_master or not ik_col_master:
        for i in range(len(out)):
            if out.loc[out.index[i], "inchikey"]:
                continue
            cid = out.loc[out.index[i], "cid"] if "cid" in out.columns else ""
            if cid and str(cid).strip() in cid_to_ik:
                out.loc[out.index[i], "inchikey"] = cid_to_ik[str(cid).strip()]
                out.loc[out.index[i], "resolution_reason"] = "cid_map"
        return out
    master_names = compound_master[name_col_master].dropna().astype(str).str.strip()
    master_names = master_names[master_names.str.len() > 0].unique().tolist()
    master_ik = {}
    for _, r in compound_master.iterrows():
        nm = _safe_str(r.get(name_col_master))
        ik = _normalize_inchikey(r.get(ik_col_master))
        if nm and ik:
            master_ik[nm.lower()] = ik
    resolved_cid = 0
    resolved_exact = 0
    resolved_fuzzy = 0
    for i in range(len(out)):
        row = out.iloc[i]
        if row.get("inchikey") and _normalize_inchikey(row["inchikey"]):
            continue
        cid = _safe_str(row.get("cid", ""))
        if cid and cid in cid_to_ik:
            out.iloc[i, out.columns.get_loc("inchikey")] = cid_to_ik[cid]
            out.iloc[i, out.columns.get_loc("resolution_reason")] = "cid_map"
            resolved_cid += 1
            continue
        name = _safe_str(row.get("name", ""))
        if not name:
            out.iloc[i, out.columns.get_loc("resolution_reason")] = "no_name_no_cid"
            continue
        name_lower = name.lower()
        if name_lower in master_ik:
            out.iloc[i, out.columns.get_loc("inchikey")] = master_ik[name_lower]
            out.iloc[i, out.columns.get_loc("resolution_reason")] = "name_exact"
            resolved_exact += 1
            continue
        if _HAS_RAPIDFUZZ and master_names:
            best_score = 0
            best_ik = None
            for mn in master_names:
                score = fuzz.ratio(name_lower, mn.lower())
                if score >= fuzzy_threshold and score > best_score:
                    best_score = score
                    best_ik = master_ik.get(mn.lower())
            if best_ik:
                out.iloc[i, out.columns.get_loc("inchikey")] = best_ik
                out.iloc[i, out.columns.get_loc("resolution_reason")] = f"name_fuzzy_{best_score}"
                resolved_fuzzy += 1
                continue
        out.iloc[i, out.columns.get_loc("resolution_reason")] = "unresolved"
    logger.info(
        "resolve_pharmgkb_to_inchikey: cid=%s exact=%s fuzzy=%s unresolved=%s",
        resolved_cid, resolved_exact, resolved_fuzzy,
        (out["resolution_reason"] == "unresolved").sum() + (out["resolution_reason"] == "no_name_no_cid").sum(),
    )
    return out


def _resolve_bindingdb_compound_to_inchikey(
    compound_id: str,
    cid_to_ik: Dict[str, str],
    chembl_to_ik: Dict[str, str],
    bindingdb_to_ik: Dict[str, str],
    name_to_ik: Dict[str, str],
    fuzzy_threshold: int = 95,
) -> Tuple[Optional[str], str]:
    """
    Resolve one BindingDB compound_id (from file) to InChIKey using master-derived maps.
    Preference: inchikey direct > cid->inchikey > chembl_id->inchikey > bindingdb_id->inchikey > name->inchikey (fuzzy).
    Returns (inchikey or None, reason).
    """
    compound_id = _safe_str(compound_id)
    if not compound_id:
        return None, "empty_id"
    ik = _normalize_inchikey(compound_id)
    if ik:
        return ik, "inchikey_direct"
    if compound_id.isdigit() and compound_id in cid_to_ik:
        return cid_to_ik[compound_id], "cid"
    if compound_id in chembl_to_ik:
        return chembl_to_ik[compound_id], "chembl_id"
    if compound_id in bindingdb_to_ik:
        return bindingdb_to_ik[compound_id], "bindingdb_id"
    name_lower = compound_id.lower()
    if name_lower in name_to_ik:
        return name_to_ik[name_lower], "name_exact"
    if _HAS_RAPIDFUZZ and name_to_ik:
        best_score = 0
        best_ik = None
        for mn, ik_val in name_to_ik.items():
            score = fuzz.ratio(name_lower, mn)
            if score >= fuzzy_threshold and score > best_score:
                best_score = score
                best_ik = ik_val
        if best_ik:
            return best_ik, f"name_fuzzy_{best_score}"
    return None, "unresolved"


def _build_master_lookups(compound_master: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    cid_to_ik = build_cid_to_inchikey(compound_master)
    chembl_col = _col(compound_master, ["chembl_id"])
    bdb_col = _col(compound_master, ["bindingdb_id"])
    name_col = _col(compound_master, ["name", "compound_name"])
    ik_col = _col(compound_master, ["inchikey", "inchi_key"])
    chembl_to_ik = {}
    bindingdb_to_ik = {}
    name_to_ik = {}
    for _, r in compound_master.iterrows():
        ik = _normalize_inchikey(r.get(ik_col)) if ik_col else None
        if not ik:
            continue
        if chembl_col:
            c = _safe_str(r.get(chembl_col))
            if c:
                chembl_to_ik[c] = ik
        if bdb_col:
            b = _safe_str(r.get(bdb_col))
            if b:
                bindingdb_to_ik[b] = ik
        if name_col:
            n = _safe_str(r.get(name_col))
            if n:
                name_to_ik[n.lower()] = ik
    return cid_to_ik, chembl_to_ik, bindingdb_to_ik, name_to_ik


def load_food_compound_gene_links(repo_root: Path) -> pd.DataFrame:
    """Load food_compound_gene_links.parquet and return edges with compound_id (InChIKey when possible), gene_symbol, source."""
    path = Path(repo_root) / "data" / "processed" / "phase12_genetics" / "food_compound_gene_links.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = _load_csv_or_parquet(path)
    if df is None or df.empty:
        return pd.DataFrame()
    cmp_col = _col(df, ["compound_id", "inchikey", "inchi_key"])
    gene_col = _col(df, ["gene_symbol", "gene_id", "gene"])
    if not cmp_col or not gene_col:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        cmp_val = _safe_str(r.get(cmp_col))
        gene_val = _safe_str(r.get(gene_col))
        if not cmp_val or not gene_val:
            continue
        ik = _normalize_inchikey(cmp_val) if cmp_val else None
        compound_id = ik if ik else cmp_val
        rows.append({
            "compound_id": compound_id,
            "gene_symbol": gene_val.upper(),
            "source": "food_compound_gene_links",
            "evidence_strength": _safe_str(r.get("association")) or "",
            "original_ids": cmp_val,
        })
    out = pd.DataFrame(rows)
    logger.info("load_food_compound_gene_links: %s edges", len(out))
    return out


def load_and_expand_bindingdb_edges(
    repo_root: Path,
    compound_master: pd.DataFrame,
    cid_to_ik: Dict[str, str],
    fuzzy_threshold: int = 95,
) -> pd.DataFrame:
    """
    Load compound_target_edges_bindingdb.parquet; resolve compounds to InChIKey; produce edges with inchikey, gene_symbol, source=bindingdb.
    Target column: use target_name as gene_symbol, or uniprot_id if no target_name (store as gene_symbol for consistency).
    """
    path = Path(repo_root) / "data" / "processed" / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if not path.exists():
        logger.info("BindingDB edges not found: %s", path)
        return pd.DataFrame()
    df = _load_csv_or_parquet(path)
    if df is None or df.empty:
        return pd.DataFrame()
    cmp_col = _col(df, ["compound_id", "inchikey", "inchi_key"])
    target_name_col = _col(df, ["target_name", "gene_symbol", "gene", "gene_id"])
    uniprot_col = _col(df, ["uniprot_id", "uniprot", "uniprot_accession"])
    if not cmp_col:
        return pd.DataFrame()
    gene_col = target_name_col or uniprot_col
    if not gene_col:
        logger.warning("BindingDB: no target/gene column found; columns: %s", list(df.columns))
        return pd.DataFrame()
    cid_to_ik, chembl_to_ik, bindingdb_to_ik, name_to_ik = _build_master_lookups(compound_master)
    resolution_counts: Dict[str, int] = {}
    rows = []
    for _, r in df.iterrows():
        compound_id_raw = r.get(cmp_col)
        if compound_id_raw is None or (isinstance(compound_id_raw, float) and pd.isna(compound_id_raw)):
            continue
        compound_id_raw = str(compound_id_raw).strip()
        ik, reason = _resolve_bindingdb_compound_to_inchikey(
            compound_id_raw,
            cid_to_ik,
            chembl_to_ik,
            bindingdb_to_ik,
            name_to_ik,
            fuzzy_threshold=fuzzy_threshold,
        )
        resolution_counts[reason] = resolution_counts.get(reason, 0) + 1
        if not ik:
            continue
        gene_val = _safe_str(r.get(gene_col))
        if not gene_val:
            continue
        evidence = []
        if "affinity_nM" in df.columns and r.get("affinity_nM") is not None and not pd.isna(r.get("affinity_nM")):
            evidence.append(f"affinity_nM={r.get('affinity_nM')}")
        if "measurement_type" in df.columns and r.get("measurement_type"):
            evidence.append(str(r.get("measurement_type")))
        rows.append({
            "compound_id": ik,
            "gene_symbol": gene_val.upper() if len(gene_val) <= 10 else gene_val,
            "source": "bindingdb",
            "evidence_strength": "; ".join(evidence) if evidence else "",
            "original_ids": compound_id_raw,
        })
    for reason, count in sorted(resolution_counts.items(), key=lambda x: -x[1]):
        logger.info("BindingDB resolution %s: %s", reason, count)
    return pd.DataFrame(rows)


def build_expanded_edges(
    repo_root: Path,
    compound_master: pd.DataFrame,
    include_food_links: bool = True,
    include_bindingdb: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Build raw and canonical compound→gene edge tables and report dict.
    - Start with food_compound_gene_links (if include_food_links).
    - Add BindingDB edges resolved to InChIKey (if include_bindingdb).
    - Canonical: compound_id (InChIKey), gene_symbol, source, evidence_strength (optional), original_ids (optional).
    - Deduplicate; keep evidence columns.
    """
    report: Dict[str, Any] = {
        "n_edges_total": 0,
        "n_compounds_total": 0,
        "n_edges_by_source": {},
        "resolution_rates": {},
        "overlap_with_ingredient_compound": {},
        "overlap_with_previous_compound_gene": {},
    }
    cid_to_ik = build_cid_to_inchikey(compound_master)
    all_edges: List[Dict[str, Any]] = []
    if include_food_links:
        food_df = load_food_compound_gene_links(repo_root)
        if not food_df.empty:
            report["n_edges_by_source"]["food_compound_gene_links"] = len(food_df)
            for _, r in food_df.iterrows():
                all_edges.append(r.to_dict())
    if include_bindingdb:
        bdb_df = load_and_expand_bindingdb_edges(repo_root, compound_master, cid_to_ik)
        if not bdb_df.empty:
            report["n_edges_by_source"]["bindingdb"] = len(bdb_df)
            for _, r in bdb_df.iterrows():
                all_edges.append(r.to_dict())
    if not all_edges:
        raw_df = pd.DataFrame(columns=["compound_id", "gene_symbol", "source", "evidence_strength", "original_ids"])
        report["n_edges_total"] = 0
        report["n_compounds_total"] = 0
        return raw_df, raw_df.copy(), report
    raw_df = pd.DataFrame(all_edges)
    raw_df = raw_df.drop_duplicates(subset=["compound_id", "gene_symbol", "source"], keep="first")
    canonical_df = raw_df[["compound_id", "gene_symbol", "source"]].copy()
    if "evidence_strength" in raw_df.columns:
        canonical_df["evidence_strength"] = raw_df["evidence_strength"]
    if "original_ids" in raw_df.columns:
        canonical_df["original_ids"] = raw_df["original_ids"]
    report["n_edges_total"] = len(canonical_df)
    report["n_compounds_total"] = int(canonical_df["compound_id"].nunique())
    # Overlaps
    canonical_dir = Path(repo_root) / "data" / "processed" / "canonical"
    ic_path = canonical_dir / "ingredient_compound_canonical.csv"
    if ic_path.exists():
        ic_df = _load_csv_or_parquet(ic_path)
        if ic_df is not None and not ic_df.empty:
            ic_col = _col(ic_df, ["compound_id", "inchikey"])
            if ic_col:
                ic_set = set(ic_df[ic_col].dropna().astype(str).str.strip().str.upper())
                cg_set = set(canonical_df["compound_id"].dropna().astype(str).str.strip().str.upper())
                overlap = ic_set & cg_set
                report["overlap_with_ingredient_compound"] = {
                    "n_ic_compounds": len(ic_set),
                    "n_cg_compounds": len(cg_set),
                    "n_overlap": len(overlap),
                    "overlap_vs_cg": round(len(overlap) / len(cg_set), 4) if cg_set else 0,
                    "overlap_vs_ic": round(len(overlap) / len(ic_set), 4) if ic_set else 0,
                }
    prev_cg_path = canonical_dir / "compound_gene_canonical.csv"
    if not prev_cg_path.exists():
        prev_cg_path = canonical_dir / "compound_gene_canonical.parquet"
    if prev_cg_path.exists():
        prev_df = _load_csv_or_parquet(prev_cg_path)
        if prev_df is not None and not prev_df.empty:
            prev_col = _col(prev_df, ["compound_id", "inchikey"])
            if prev_col:
                prev_set = set(prev_df[prev_col].dropna().astype(str).str.strip().str.upper())
                cg_set = set(canonical_df["compound_id"].dropna().astype(str).str.strip().str.upper())
                overlap_prev = prev_set & cg_set
                report["overlap_with_previous_compound_gene"] = {
                    "n_previous_compounds": len(prev_set),
                    "n_expanded_compounds": len(cg_set),
                    "n_overlap": len(overlap_prev),
                    "overlap_vs_expanded": round(len(overlap_prev) / len(cg_set), 4) if cg_set else 0,
                }
    return raw_df, canonical_df, report


def run_expansion_pipeline(
    repo_root: Path,
    output_dir: Optional[Path] = None,
    compound_master_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Full pipeline: load compound_master, build expanded edges, write CSVs and report.
    Returns (raw_df, canonical_df, report).
    """
    repo_root = Path(repo_root).resolve()
    output_dir = output_dir or (repo_root / "data" / "processed" / "canonical")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compound_master_path = compound_master_path or (output_dir / "compound_master.csv")
    compound_master = _load_csv_or_parquet(compound_master_path)
    if compound_master is None or compound_master.empty:
        logger.warning("compound_master not found at %s; expansion will use only existing InChIKeys in sources", compound_master_path)
        compound_master = pd.DataFrame()
    raw_df, canonical_df, report = build_expanded_edges(repo_root, compound_master, include_food_links=True, include_bindingdb=True)
    raw_path = output_dir / "compound_gene_expanded_raw.csv"
    canon_path = output_dir / "compound_gene_expanded_canonical.csv"
    raw_df.to_csv(raw_path, index=False)
    canonical_df.to_csv(canon_path, index=False)
    logger.info("Wrote %s (%s rows) and %s (%s rows)", raw_path.name, len(raw_df), canon_path.name, len(canonical_df))
    report_path = output_dir / "compound_gene_expansion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", report_path.name)
    return raw_df, canonical_df, report
