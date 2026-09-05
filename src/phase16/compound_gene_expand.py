"""
Phase16A: CMP->GENE densification — coherent resolver pipeline (local-only, no external APIs).
Increases compound_gene coverage by: PharmGKB chemicals -> InChIKey (RDKit); BindingDB COCONUT -> InChIKey;
targets -> genes via local uniprot mapping. Outputs compound_gene_expanded_canonical.csv and compound_gene_canonical.csv.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# FooDB normalizer (match phase14)
_FDB_PATTERN = re.compile(r"^(?:FDB[_\-]?|fdb[_\-]?|FOODB\s*:\s*)(\d+)$", re.IGNORECASE)


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
    if df is None or df.empty:
        return None
    low = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "_")
        if k in low:
            return low[k]
    return None


def normalize_fdb_id(x: Optional[str]) -> Optional[str]:
    """Normalize FooDB-style IDs to FDB_<integer>."""
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


def load_food_compound_gene_links(repo_root: Path) -> pd.DataFrame:
    """Load food_compound_gene_links.parquet; return df with compound_id, gene_symbol (normalized)."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    path = processed / "phase12_genetics" / "food_compound_gene_links.parquet"
    if not path.exists():
        for p in repo_root.rglob("food_compound_gene_links.parquet"):
            path = p
            break
        else:
            return pd.DataFrame(columns=["compound_id", "gene_symbol", "compound_name"])
    df = _load_df(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["compound_id", "gene_symbol", "compound_name"])
    cmp_col = _col(df, ["compound_id", "compound id"])
    gene_col = _col(df, ["gene_symbol", "gene", "gene_id"])
    name_col = _col(df, ["compound_name", "compound name", "name", "food_name"])
    if not cmp_col or not gene_col:
        return pd.DataFrame(columns=["compound_id", "gene_symbol", "compound_name"])
    out = []
    for _, r in df.iterrows():
        cid = _safe_str(r.get(cmp_col))
        g = _safe_str(r.get(gene_col))
        if not cid or not g:
            continue
        name = _safe_str(r.get(name_col)) if name_col else ""
        out.append({"compound_id": cid, "gene_symbol": g.upper(), "compound_name": name})
    logger.info("load_food_compound_gene_links: %s rows from %s", len(out), path.name)
    return pd.DataFrame(out)


def build_pharmgkb_chemical_identity(repo_root: Path) -> pd.DataFrame:
    """Load PharmGKB chemicals; return df with chem_name, pubchem_cid, inchi, smiles, inchikey (empty until compute_inchikey), synonyms optional."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    path = processed / "phase12_genetics" / "pharmgkb_chemicals.parquet"
    if not path.exists():
        for p in repo_root.rglob("pharmgkb_chemicals.parquet"):
            path = p
            break
        else:
            return pd.DataFrame(columns=["chem_name", "pubchem_cid", "inchi", "smiles", "inchikey", "synonyms"])
    df = _load_df(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["chem_name", "pubchem_cid", "inchi", "smiles", "inchikey", "synonyms"])
    cl = {c.lower().replace(" ", "_"): c for c in df.columns}
    name_col = cl.get("name") or cl.get("compound_name") or cl.get("chemical_name")
    cid_col = cl.get("pubchem_compound_identifiers") or cl.get("pubchem_cid") or cl.get("cid") or cl.get("pubchem_id")
    inchi_col = cl.get("inchi")
    smiles_col = cl.get("smiles") or cl.get("canonical_smiles")
    syn_col = cl.get("synonyms") or cl.get("generic_names") or cl.get("trade_names")
    out = []
    for _, r in df.iterrows():
        name = _safe_str(r.get(name_col)) if name_col else ""
        cid = r.get(cid_col)
        if cid is not None and not (isinstance(cid, float) and pd.isna(cid)):
            try:
                cid = str(int(float(cid)))
            except (ValueError, TypeError):
                cid = _safe_str(cid)
        else:
            cid = ""
        inchi = _safe_str(r.get(inchi_col)) if inchi_col else ""
        smiles = _safe_str(r.get(smiles_col)) if smiles_col else ""
        syn = _safe_str(r.get(syn_col)) if syn_col else ""
        out.append({
            "chem_name": name,
            "pubchem_cid": cid,
            "inchi": inchi,
            "smiles": smiles,
            "inchikey": "",
            "synonyms": syn,
        })
    df_chem = pd.DataFrame(out)
    logger.info("build_pharmgkb_chemical_identity: %s rows from %s", len(df_chem), path.name)
    return df_chem


def compute_inchikey_from_inchi_smiles(df_chem: pd.DataFrame) -> pd.DataFrame:
    """Fill inchikey using RDKit: InChI -> MolFromInchi -> MolToInchiKey; else SMILES -> MolFromSmiles -> MolToInchiKey.
    If RDKit is missing, raises RuntimeError with exact conda/pip install steps for Windows."""
    try:
        from rdkit import Chem
    except ImportError as e:
        raise RuntimeError(
            "RDKit is required to compute InChIKey from PharmGKB InChI/SMILES. Install it first.\n"
            "Windows (conda): conda install -c conda-forge rdkit\n"
            "Windows (pip):  pip install rdkit\n"
            "Then re-run the Phase16A expansion."
        ) from e
    out = df_chem.copy()
    if "inchikey" not in out.columns:
        out["inchikey"] = ""
    filled = 0
    for i in range(len(out)):
        row = out.iloc[i]
        if row.get("inchikey") and len(_safe_str(row["inchikey"])) >= 25:
            continue
        mol = None
        inchi = _safe_str(row.get("inchi", ""))
        smiles = _safe_str(row.get("smiles", ""))
        if inchi:
            try:
                mol = Chem.MolFromInchi(inchi)
            except Exception:
                pass
        if mol is None and smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                pass
        if mol is not None:
            ik = Chem.inchi.MolToInchiKey(mol)
            if ik:
                out.iloc[i, out.columns.get_loc("inchikey")] = ik
                filled += 1
    logger.info("compute_inchikey_from_inchi_smiles: filled %s of %s rows", filled, len(out))
    return out


def build_foodb_to_inchikey_map(
    repo_root: Path,
    df_chem: pd.DataFrame,
    compound_master: pd.DataFrame,
    df_food: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Build FDB_id -> InChIKey map. Priority: (a) compound_master FDB+inchikey (b) df_food compound_name -> PharmGKB name match (c) df_food PubChem if any -> df_chem CID.
    Returns (fdb_to_ik, report) with match rates and top unmapped."""
    repo_root = Path(repo_root).resolve()
    fdb_to_ik: Dict[str, str] = {}
    report: Dict[str, Any] = {"from_compound_master": 0, "from_name_match_pharmgkb": 0, "from_cid_pharmgkb": 0, "unmapped_fdb": [], "n_total_fdb_seen": 0}

    # (a) compound_master: fdb_id_norm / fdb column -> inchikey
    if not compound_master.empty:
        fdb_col = _col(compound_master, ["fdb_id_norm", "fdb_id", "fdb"])
        ik_col = _col(compound_master, ["compound_id", "inchikey", "inchi_key"])
        if fdb_col and ik_col:
            for _, r in compound_master.iterrows():
                fdb = normalize_fdb_id(r.get(fdb_col)) or _safe_str(r.get(fdb_col))
                if not fdb or "FDB_" not in fdb.upper():
                    continue
                ik = _safe_str(r.get(ik_col))
                if len(ik) >= 25:
                    fdb_to_ik[fdb] = ik
            report["from_compound_master"] = len(fdb_to_ik)

    # Name -> inchikey from df_chem (after compute_inchikey)
    name_to_ik: Dict[str, str] = {}
    if not df_chem.empty and "inchikey" in df_chem.columns:
        for _, r in df_chem.iterrows():
            ik = _safe_str(r.get("inchikey"))
            if len(ik) < 25:
                continue
            name = _safe_str(r.get("chem_name", ""))
            if name:
                name_to_ik[name.casefold().strip()] = ik
            syn = _safe_str(r.get("synonyms", ""))
            for part in syn.replace(";", ",").split(","):
                n = part.strip().casefold()
                if n and n not in name_to_ik:
                    name_to_ik[n] = ik

    # (b) df_food compound_name -> PharmGKB name match
    if df_food is not None and not df_food.empty and "compound_name" in df_food.columns:
        for _, r in df_food.iterrows():
            fdb = normalize_fdb_id(r.get("compound_id"))
            if not fdb or fdb in fdb_to_ik:
                continue
            name = _safe_str(r.get("compound_name", "")).casefold().strip()
            if name and name in name_to_ik:
                fdb_to_ik[fdb] = name_to_ik[name]
                report["from_name_match_pharmgkb"] = report.get("from_name_match_pharmgkb", 0) + 1

    # (c) CID join: if df_food had pubchem/cid we could add; df_chem has pubchem_cid -> inchikey
    cid_to_ik: Dict[str, str] = {}
    if not df_chem.empty and "inchikey" in df_chem.columns and "pubchem_cid" in df_chem.columns:
        for _, r in df_chem.iterrows():
            cid = _safe_str(r.get("pubchem_cid", ""))
            ik = _safe_str(r.get("inchikey", ""))
            if cid and len(ik) >= 25:
                cid_to_ik[cid] = ik

    # FDB from compound_master might have cid; we don't have df_food cid usually. So just report.
    report["n_cid_to_ik_pharmgkb"] = len(cid_to_ik)

    # Collect all FDB IDs we care about (from df_food) for unmapped list
    all_fdb: Set[str] = set()
    if df_food is not None and not df_food.empty:
        for v in df_food["compound_id"].dropna().astype(str).str.strip():
            fdb = normalize_fdb_id(v) or v
            if fdb and ("FDB" in fdb.upper() or v.upper().startswith("FDB")):
                all_fdb.add(fdb)
    report["n_total_fdb_seen"] = len(all_fdb)
    unmapped = [f for f in sorted(all_fdb) if f not in fdb_to_ik]
    report["top_unmapped_fdb"] = unmapped[:50]
    report["n_unmapped_fdb"] = len(unmapped)

    return fdb_to_ik, report


def build_coconut_to_inchikey_map(repo_root: Path) -> Dict[str, str]:
    """Build COCONUT_ID -> InChIKey from inchikey_to_compound_id.json (invert) and any coconut compound tables."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    coconut_to_ik: Dict[str, str] = {}

    # Primary: phase15 JSON
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
                ik_clean = _safe_str(ik).strip().upper()
                if len(ik_clean) < 25:
                    continue
                coconut_to_ik[s] = ik_clean
                if "." in s:
                    base = s.split(".")[0]
                    if base:
                        coconut_to_ik[base] = ik_clean
        except Exception as e:
            logger.warning("build_coconut_to_inchikey_map: could not load JSON: %s", e)

    # Enrich from compound_master coconut_id -> inchikey
    for p in [processed / "canonical" / "compound_master.csv", processed / "canonical" / "compound_master.parquet"]:
        if not p.exists():
            continue
        df = _load_df(p)
        if df is None or df.empty:
            continue
        coco_col = _col(df, ["coconut_id", "coconut_base"])
        ik_col = _col(df, ["compound_id", "inchikey", "inchi_key"])
        if not coco_col or not ik_col:
            continue
        for _, r in df.iterrows():
            coco = _safe_str(r.get(coco_col))
            if not coco or "COCONUT_" not in coco.upper():
                continue
            ik = _safe_str(r.get(ik_col))
            if len(ik) >= 25:
                coconut_to_ik[coco] = ik
                if "." in coco:
                    coconut_to_ik[coco.split(".")[0]] = ik
        break

    logger.info("build_coconut_to_inchikey_map: %s entries", len(coconut_to_ik))
    return coconut_to_ik


def build_uniprot_to_gene_map(repo_root: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Scan data/processed for files with uniprot_id and gene_symbol/gene; use largest by rowcount; include target_clusters/target_functional_clusters. Returns (uniprot_to_gene, report)."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    if not processed.exists():
        return {}, {"chosen_file": None, "n_mappings": 0, "coverage_note": "no data/processed"}

    candidates: List[Tuple[Path, int, Dict[str, str]]] = []
    for path in list(processed.rglob("*.parquet")) + list(processed.rglob("*.csv")):
        if path.stat().st_size > 50_000_000:
            continue
        try:
            df = _load_df(path)
            if df is None or df.empty or len(df) > 500_000:
                continue
        except Exception:
            continue
        u_col = _col(df, ["uniprot_id", "uniprot", "uniprot_accession", "uniprot_accession_x", "uniprot_accession_y"])
        g_col = _col(df, ["gene_symbol", "gene_name", "gene"])
        if not u_col or not g_col:
            continue
        m: Dict[str, str] = {}
        for _, r in df.iterrows():
            u = _safe_str(r.get(u_col)).upper()
            g = _safe_str(r.get(g_col)).upper()
            if u and g and len(u) <= 20 and len(g) <= 30:
                if u not in m or len(g) < len(m.get(u, "")):
                    m[u] = g
        if m:
            candidates.append((path, len(m), m))
    # Prefer target_functional_clusters / target_clusters if present
    target_path = processed / "features" / "target_functional_clusters.csv"
    if not target_path.exists():
        target_path = processed / "features" / "target_clusters.csv"
    if target_path.exists():
        df = _load_df(target_path)
        if df is not None and not df.empty:
            u_col = _col(df, ["uniprot_id", "uniprot", "uniprot_accession"])
            g_col = _col(df, ["gene_symbol", "gene_name", "gene"])
            if u_col and g_col:
                m = {}
                for _, r in df.iterrows():
                    u = _safe_str(r.get(u_col)).upper()
                    g = _safe_str(r.get(g_col)).upper()
                    if u and g:
                        m[u] = g
                if m:
                    candidates.append((target_path, len(m), m))

    if not candidates:
        return {}, {"chosen_file": None, "n_mappings": 0}

    # Largest by mapping count
    candidates.sort(key=lambda x: -x[1])
    best_path, best_n, best_m = candidates[0]
    report = {"chosen_file": str(best_path.relative_to(repo_root)) if repo_root in best_path.parents else str(best_path), "n_mappings": best_n}
    logger.info("build_uniprot_to_gene_map: %s mappings from %s", best_n, best_path.name)
    return best_m, report


def build_bindingdb_compound_gene(
    repo_root: Path,
    coconut_map: Dict[str, str],
    uniprot_to_gene_map: Dict[str, str],
    only_human: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load BindingDB edges; filter human; resolve compound_id -> inchikey via coconut_map; uniprot_id -> gene_symbol. 
    Returns (df with compound_id_raw, compound_id (inchikey), gene_symbol, source=bindingdb, evidence fields), report."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    path = processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if not path.exists():
        for p in repo_root.rglob("compound_target_edges_bindingdb.parquet"):
            path = p
            break
        else:
            return pd.DataFrame(), {"n_total": 0, "n_human": 0, "n_compounds_resolved_to_inchikey": 0, "n_targets_resolved_to_gene": 0}

    df = _load_df(path)
    if df is None or df.empty:
        return pd.DataFrame(), {"n_total": 0}

    cmp_col = _col(df, ["compound_id"])
    uniprot_col = _col(df, ["uniprot_id", "uniprot"])
    target_col = _col(df, ["target_name"])
    organism_col = _col(df, ["organism"])
    affinity_col = _col(df, ["affinity_nm", "affinity_nM"])
    meas_col = _col(df, ["measurement_type", "ki_nm"])
    pubmed_col = _col(df, ["pubmed_id", "pubmed"])

    if not cmp_col:
        return pd.DataFrame(), {"n_total": len(df)}

    n_total = len(df)
    n_human = 0
    n_compounds_resolved = 0
    n_targets_resolved = 0
    rows: List[Dict[str, Any]] = []
    unresolved_targets: Dict[str, int] = {}  # uniprot or target_name -> count

    for _, r in df.iterrows():
        organism = _safe_str(r.get(organism_col)) if organism_col else ""
        if only_human and organism and "homo sapiens" not in organism.lower() and "human" not in organism.lower():
            continue
        n_human += 1
        cid_raw = _safe_str(r.get(cmp_col))
        if not cid_raw:
            continue
        # Resolve to InChIKey
        ik = coconut_map.get(cid_raw)
        if not ik and cid_raw.startswith("COCONUT_") and "." in cid_raw:
            ik = coconut_map.get(cid_raw.split(".")[0])
        if not ik:
            continue
        n_compounds_resolved += 1
        uniprot = _safe_str(r.get(uniprot_col)) if uniprot_col else ""
        target_name = _safe_str(r.get(target_col)) if target_col else ""
        gene = uniprot_to_gene_map.get(uniprot.upper()) if uniprot else ""
        if not gene and target_name and len(target_name) <= 15 and target_name.replace("-", "").isalnum():
            gene = target_name.upper()
        if not gene:
            key = uniprot or target_name or "unknown"
            unresolved_targets[key] = unresolved_targets.get(key, 0) + 1
            continue
        n_targets_resolved += 1
        aff = r.get(affinity_col)
        try:
            aff_val = float(aff) if aff is not None and not (isinstance(aff, float) and pd.isna(aff)) else None
        except (TypeError, ValueError):
            aff_val = None
        meas = _safe_str(r.get(meas_col)) if meas_col else ""
        pubmed = _safe_str(r.get(pubmed_col)) if pubmed_col else ""
        rows.append({
            "compound_id_raw": cid_raw,
            "compound_id": ik,
            "gene_symbol": gene,
            "source": "bindingdb",
            "uniprot_id": uniprot,
            "target_name": target_name,
            "affinity_nM": aff_val,
            "measurement_type": meas,
            "pubmed_id": pubmed,
        })
    top_unresolved = sorted(unresolved_targets.items(), key=lambda x: -x[1])[:20]
    report = {
        "n_total": n_total,
        "n_human": n_human,
        "n_compounds_resolved_to_inchikey": n_compounds_resolved,
        "n_targets_resolved_to_gene": n_targets_resolved,
        "pct_bindingdb_compounds_resolved_to_inchikey": round(100.0 * n_compounds_resolved / n_human, 2) if n_human else 0,
        "pct_bindingdb_targets_resolved_to_gene": round(100.0 * n_targets_resolved / n_human, 2) if n_human else 0,
        "top_unresolved_targets": [t[0] for t in top_unresolved],
    }
    logger.info("build_bindingdb_compound_gene: %s edges (human), %s compounds resolved, %s targets->gene", len(rows), n_compounds_resolved, n_targets_resolved)
    return pd.DataFrame(rows), report


def merge_compound_gene_sources(
    df_food_resolved: pd.DataFrame,
    df_binding_resolved: pd.DataFrame,
    repo_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Merge food + binding; dedupe on (compound_id, gene_symbol); add sources, n_sources, sample_evidence. 
    Returns (expanded_df, canonical_df, report). Canonical = same rows, may drop some provenance for Phase14."""
    repo_root = Path(repo_root).resolve()
    canonical_dir = repo_root / "data" / "processed" / "canonical"
    ing_path = canonical_dir / "ingredient_compound_canonical.csv"

    # Normalize: ensure compound_id, gene_symbol
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "compound_id" not in out.columns and "inchikey" in out.columns:
            out = out.rename(columns={"inchikey": "compound_id"})
        if "gene_symbol" not in out.columns and "gene" in out.columns:
            out = out.rename(columns={"gene": "gene_symbol"})
        return out

    food = _norm(df_food_resolved)
    binding = _norm(df_binding_resolved)

    all_rows: List[Dict[str, Any]] = []
    if not food.empty:
        for _, r in food.iterrows():
            row = {"compound_id": _safe_str(r.get("compound_id")), "gene_symbol": _safe_str(r.get("gene_symbol", "")).upper(), "source": "food_compound_gene_links"}
            if "compound_name" in r:
                row["sample_evidence"] = _safe_str(r.get("compound_name"))[:200]
            all_rows.append(row)
    if not binding.empty:
        for _, r in binding.iterrows():
            row = {"compound_id": _safe_str(r.get("compound_id")), "gene_symbol": _safe_str(r.get("gene_symbol", "")).upper(), "source": "bindingdb"}
            ev = []
            if r.get("target_name"):
                ev.append(str(r.get("target_name")))
            if r.get("uniprot_id"):
                ev.append(str(r.get("uniprot_id")))
            if ev:
                row["sample_evidence"] = " | ".join(ev)[:200]
            all_rows.append(row)

    if not all_rows:
        expanded = pd.DataFrame(columns=["compound_id", "gene_symbol", "source", "sources", "n_sources", "sample_evidence"])
        report = {"n_edges_total": 0, "n_unique_compounds": 0, "n_unique_genes": 0, "n_from_food_links": 0, "n_from_bindingdb": 0}
        return expanded, expanded.copy(), report

    merged = pd.DataFrame(all_rows)
    # Dedup by (compound_id, gene_symbol): keep one row per edge, aggregate sources
    def _agg_sources(s: pd.Series) -> str:
        return "|".join(sorted(set(s.dropna().astype(str).str.strip())))
    def _agg_evidence(s: pd.Series) -> str:
        parts = s.dropna().astype(str).str.strip()
        return " | ".join(parts.head(2).tolist())[:300] if len(parts) else ""
    expanded = merged.groupby(["compound_id", "gene_symbol"], as_index=False).agg(
        source=("source", "first"),
        sample_evidence=("sample_evidence", _agg_evidence),
    )
    expanded["sources"] = merged.groupby(["compound_id", "gene_symbol"])["source"].apply(_agg_sources).values
    expanded["n_sources"] = expanded["sources"].str.count("|").add(1)

    report = {
        "n_edges_total": len(expanded),
        "n_unique_compounds": int(expanded["compound_id"].nunique()),
        "n_unique_genes": int(expanded["gene_symbol"].nunique()),
        "n_from_food_links": int((expanded["sources"].str.contains("food", case=False, na=False)).sum()) if "sources" in expanded.columns else 0,
        "n_from_bindingdb": int((expanded["sources"].str.contains("bindingdb", case=False, na=False)).sum()) if "sources" in expanded.columns else 0,
    }

    # Overlap with ingredient_compound
    overlap_report: Dict[str, Any] = {}
    if ing_path.exists():
        ing_df = _load_df(ing_path)
        if ing_df is not None and not ing_df.empty and "compound_id" in ing_df.columns:
            ic_set = set(ing_df["compound_id"].dropna().astype(str).str.strip().str.upper())
            cg_set = set(expanded["compound_id"].dropna().astype(str).str.strip().str.upper())
            n_overlap = len(ic_set & cg_set)
            overlap_report["overlap_with_ingredient_compound"] = {
                "n_overlap": n_overlap,
                "overlap_vs_cg": round(n_overlap / len(cg_set), 4) if cg_set else 0,
                "overlap_vs_ic": round(n_overlap / len(ic_set), 4) if ic_set else 0,
            }
    report["overlap_with_ingredient_compound"] = overlap_report.get("overlap_with_ingredient_compound", {})

    # Top unresolved (from build_foodb report if passed in merge; we don't have it here so leave empty or add later)
    report["top_unresolved_compounds"] = []
    report["top_unresolved_targets"] = []

    canonical = expanded[["compound_id", "gene_symbol"]].copy()
    if "source" in expanded.columns:
        canonical["source"] = expanded["source"]
    if "sources" in expanded.columns:
        canonical["sources"] = expanded["sources"]
    if "n_sources" in expanded.columns:
        canonical["n_sources"] = expanded["n_sources"]
    if "sample_evidence" in expanded.columns:
        canonical["sample_evidence"] = expanded["sample_evidence"]

    return expanded, canonical, report


def run_full_pipeline(
    repo_root: Path,
    only_human: bool = True,
    write_debug: bool = False,
) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    """Run full Phase16A pipeline: PharmGKB identity + RDKit InChIKey, FooDB map, Coconut map, uniprot->gene, BindingDB, merge, write.
    Returns (expanded_path, canonical_path, report_path, report)."""
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"
    reports_dir = canonical_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) PharmGKB chemical identity + InChIKey (RDKit required)
    df_chem = build_pharmgkb_chemical_identity(repo_root)
    df_chem = compute_inchikey_from_inchi_smiles(df_chem)

    # 2) FooDB -> InChIKey
    compound_master = _load_df(canonical_dir / "compound_master.csv")
    if compound_master is None or compound_master.empty:
        compound_master = _load_df(canonical_dir / "compound_master.parquet")
    if compound_master is None or compound_master.empty:
        compound_master = pd.DataFrame()
    df_food = load_food_compound_gene_links(repo_root)
    fdb_to_ik, foodb_report = build_foodb_to_inchikey_map(repo_root, df_chem, compound_master, df_food)

    # 3) Resolve food links to InChIKey
    food_resolved_rows: List[Dict[str, Any]] = []
    for _, r in df_food.iterrows():
        cid = _safe_str(r.get("compound_id"))
        fdb = normalize_fdb_id(cid) or cid
        ik = fdb_to_ik.get(fdb)
        if not ik and len(cid) >= 25 and cid.replace("-", "").isalnum():
            ik = cid  # already InChIKey
        if not ik:
            continue
        food_resolved_rows.append({
            "compound_id": ik,
            "gene_symbol": _safe_str(r.get("gene_symbol", "")).upper(),
            "compound_name": _safe_str(r.get("compound_name", "")),
        })
    df_food_resolved = pd.DataFrame(food_resolved_rows)

    # 4) Coconut map + uniprot->gene
    coconut_map = build_coconut_to_inchikey_map(repo_root)
    uniprot_to_gene, uniprot_report = build_uniprot_to_gene_map(repo_root)

    # 5) BindingDB compound->gene
    df_binding, binding_report = build_bindingdb_compound_gene(repo_root, coconut_map, uniprot_to_gene, only_human=only_human)

    # 6) Merge and report
    expanded, canonical, merge_report = merge_compound_gene_sources(df_food_resolved, df_binding, repo_root)

    report = {
        "n_edges_total": merge_report["n_edges_total"],
        "n_unique_compounds": merge_report["n_unique_compounds"],
        "n_unique_genes": merge_report["n_unique_genes"],
        "n_from_food_links": merge_report["n_from_food_links"],
        "n_from_bindingdb": merge_report["n_from_bindingdb"],
        "pct_bindingdb_compounds_resolved_to_inchikey": binding_report.get("pct_bindingdb_compounds_resolved_to_inchikey"),
        "pct_bindingdb_targets_resolved_to_gene": binding_report.get("pct_bindingdb_targets_resolved_to_gene"),
        "overlap_with_ingredient_compound": merge_report.get("overlap_with_ingredient_compound", {}),
        "top_unresolved_compounds": foodb_report.get("top_unmapped_fdb", [])[:30],
        "top_unresolved_targets": binding_report.get("top_unresolved_targets", [])[:30],
        "foodb_resolution": foodb_report,
        "uniprot_to_gene": uniprot_report,
        "bindingdb": binding_report,
    }

    # Write
    expanded_path = canonical_dir / "compound_gene_expanded_canonical.csv"
    canonical_path = canonical_dir / "compound_gene_canonical.csv"
    report_path = reports_dir / "compound_gene_expansion_v2_report.json"

    expanded.to_csv(expanded_path, index=False)
    canonical.to_csv(canonical_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if write_debug:
        debug_dir = reports_dir
        df_chem.to_csv(debug_dir / "debug_pharmgkb_chemical_identity.csv", index=False)
        if not df_food_resolved.empty:
            df_food_resolved.to_csv(debug_dir / "debug_food_resolved.csv", index=False)
        if not df_binding.empty:
            df_binding.to_csv(debug_dir / "debug_bindingdb_resolved.csv", index=False)

    logger.info("Wrote %s (%s rows), %s (%s compounds, %s genes), report %s",
                expanded_path.name, len(expanded), canonical_path.name, report["n_unique_compounds"], report["n_unique_genes"], report_path.name)
    return expanded_path, canonical_path, report_path, report
