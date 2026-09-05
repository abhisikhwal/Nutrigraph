"""
Expand compound->gene coverage via BindingDB (Coconut ID -> InChIKey) and optional target->gene mapping.
Writes: compound_target_canonical.csv, compound_gene_expanded_v3_canonical.csv.
Run: python scripts/phase16/expand_compound_gene_v3.py --repo-root .
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s == "nan" else s


def canonicalize_coconut_id(val: Any) -> Optional[str]:
    s = _safe_str(val)
    if not s or "COCONUT_" not in s.upper():
        return None
    return s


def build_coconut_to_inchikey(repo_root: Path) -> Dict[str, str]:
    """Build coconut_id -> inchikey from compound_master and phase15 JSON."""
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"
    coconut_to_ik: Dict[str, str] = {}

    # 1) From compound_master
    for path in [canonical_dir / "compound_master.csv", canonical_dir / "compound_master.parquet"]:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
        except Exception as e:
            logger.warning("Could not load %s: %s", path.name, e)
            continue
        if df.empty or "coconut_id" not in df.columns:
            continue
        for _, r in df.iterrows():
            coco = _safe_str(r.get("coconut_id"))
            ik = _safe_str(r.get("inchikey"))
            if coco and ik and len(ik) >= 25:
                coconut_to_ik[coco] = ik
                coconut_to_ik[coco.strip()] = ik
                if "." in coco:
                    base = coco.split(".")[0]
                    if base and base not in coconut_to_ik:
                        coconut_to_ik[base] = ik
        break

    # 2) From phase15 inchikey_to_compound_id.json (invert: inchikey -> coconut_id => coconut_id -> inchikey)
    json_path = processed / "phase15_coconut" / "inchikey_to_compound_id.json"
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                ik2id = json.load(f)
            for ik, raw_id in ik2id.items():
                if raw_id is None:
                    continue
                coco = canonicalize_coconut_id(raw_id)
                if coco and len(_safe_str(ik)) >= 25:
                    ik_norm = _safe_str(ik).strip().upper()
                    if len(ik_norm) >= 25:
                        coconut_to_ik[coco] = ik_norm
                        if "." in coco:
                            base = coco.split(".")[0]
                            if base and base not in coconut_to_ik:
                                coconut_to_ik[base] = ik_norm
        except Exception as e:
            logger.warning("Could not load inchikey_to_compound_id.json: %s", e)

    return coconut_to_ik


def resolve_coconut_to_inchikey(compound_id: Any, coconut_to_ik: Dict[str, str]) -> Optional[str]:
    s = _safe_str(compound_id)
    if not s:
        return None
    if s in coconut_to_ik:
        return coconut_to_ik[s]
    if s.startswith("COCONUT_"):
        base = s.split(".")[0] if "." in s else s
        if base in coconut_to_ik:
            return coconut_to_ik[base]
    return None


def discover_bindingdb_edges(repo_root: Path) -> Optional[Path]:
    processed = repo_root / "data" / "processed"
    candidates = [
        processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet",
        processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in processed.rglob("compound_target_edges*.parquet"):
        if "bindingdb" in str(p).lower() or p.parent.name == "phase16_bindingdb":
            return p
    return None


def load_target_to_gene_mapping(repo_root: Path) -> Dict[str, str]:
    """If a local mapping (uniprot_id or target_name -> gene_symbol) exists, load it."""
    processed = repo_root / "data" / "processed"
    mapping: Dict[str, str] = {}
    for pattern in ["*uniprot*gene*.csv", "*target*gene*.csv", "*gene*mapping*.csv"]:
        for p in processed.rglob(pattern):
            if not p.is_file():
                continue
            try:
                df = pd.read_csv(p, nrows=5000)
            except Exception:
                continue
            cols = [c.lower() for c in df.columns]
            uniprot_col = next((c for c in df.columns if "uniprot" in c.lower() and "id" in c.lower()), None)
            gene_col = next((c for c in df.columns if "gene" in c.lower() and ("symbol" in c.lower() or c.lower() == "gene")), None)
            if uniprot_col and gene_col:
                for _, r in df.iterrows():
                    u = _safe_str(r.get(uniprot_col))
                    g = _safe_str(r.get(gene_col))
                    if u and g:
                        mapping[u] = g
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand compound-gene via BindingDB and write v3 canonical")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    # Build coconut_id -> inchikey
    coconut_to_ik = build_coconut_to_inchikey(repo_root)
    logger.info("Coconut->InChIKey mapping size: %s", len(coconut_to_ik))

    # Load BindingDB edges
    bdb_path = discover_bindingdb_edges(repo_root)
    if not bdb_path:
        logger.warning("BindingDB edges not found; skipping compound_target and v3")
        return 0

    if bdb_path.suffix.lower() == ".parquet":
        bdb = pd.read_parquet(bdb_path)
    else:
        bdb = pd.read_csv(bdb_path)
    if bdb.empty or "compound_id" not in bdb.columns:
        logger.warning("BindingDB edges empty or missing compound_id")
        return 0

    # Resolve compound_id to InChIKey
    target_col = next((c for c in bdb.columns if "target" in c.lower() and "name" in c.lower()), "target_name")
    uniprot_col = next((c for c in bdb.columns if "uniprot" in c.lower()), "uniprot_id")
    aff_col = next((c for c in bdb.columns if "affinity" in c.lower()), "affinity_nM")
    mtype_col = next((c for c in bdb.columns if "measurement" in c.lower()), "measurement_type")
    src_col = next((c for c in bdb.columns if c.lower() == "source"), "source")

    compound_target_rows: List[Dict[str, Any]] = []
    compound_gene_rows: List[Dict[str, Any]] = []
    resolved_compounds: Set[str] = set()
    unresolved_count = 0

    for _, r in bdb.iterrows():
        raw_cid = r.get("compound_id")
        ik = resolve_coconut_to_inchikey(raw_cid, coconut_to_ik)
        if not ik or len(ik) < 25:
            unresolved_count += 1
            continue
        resolved_compounds.add(ik)
        target_name = _safe_str(r.get(target_col))
        uniprot_id = _safe_str(r.get(uniprot_col))
        affinity = r.get(aff_col)
        if pd.isna(affinity):
            affinity = ""
        else:
            try:
                affinity = float(affinity)
            except (TypeError, ValueError):
                affinity = ""
        measurement_type = _safe_str(r.get(mtype_col))
        source = _safe_str(r.get(src_col)) or "bindingdb"

        compound_target_rows.append({
            "compound_id": ik,
            "target_name": target_name,
            "uniprot_id": uniprot_id,
            "affinity_nM": affinity,
            "measurement_type": measurement_type,
            "source": source,
        })
        # For v3: use target_name as gene_id when no mapping (evidence layer)
        gene_id = target_name or uniprot_id or ""
        compound_gene_rows.append({
            "compound_id": ik,
            "gene_id": gene_id,
            "target_name": target_name,
            "uniprot_id": uniprot_id,
            "source": source,
        })

    # Optional: override gene_id with mapping when available
    target_to_gene = load_target_to_gene_mapping(repo_root)
    if target_to_gene:
        for row in compound_gene_rows:
            u = row.get("uniprot_id", "")
            t = row.get("target_name", "")
            if u and u in target_to_gene:
                row["gene_id"] = target_to_gene[u]
            elif t and t in target_to_gene:
                row["gene_id"] = target_to_gene[t]

    # Drop compound_gene rows without gene_id if we had a mapping and now some are empty
    compound_gene_df = pd.DataFrame(compound_gene_rows)
    if not compound_gene_df.empty:
        compound_gene_df = compound_gene_df[compound_gene_df["gene_id"].astype(str).str.strip() != ""]

    # Write compound_target_canonical.csv
    target_df = pd.DataFrame(compound_target_rows)
    if not target_df.empty:
        target_path = canonical_dir / "compound_target_canonical.csv"
        target_df.to_csv(target_path, index=False)
        logger.info("Wrote %s: rows=%s unique_compounds=%s", target_path.name, len(target_df), target_df["compound_id"].nunique())

    # Write compound_gene_expanded_v3_canonical.csv (prefer compound_id, gene_id for loaders)
    if not compound_gene_df.empty:
        out_cols = ["compound_id", "gene_id"]
        if "source" in compound_gene_df.columns:
            out_cols.append("source")
        v3_path = canonical_dir / "compound_gene_expanded_v3_canonical.csv"
        compound_gene_df[out_cols].to_csv(v3_path, index=False)
        logger.info(
            "Wrote %s: rows=%s unique_compounds=%s unique_genes=%s",
            v3_path.name, len(compound_gene_df), compound_gene_df["compound_id"].nunique(), compound_gene_df["gene_id"].nunique()
        )

    logger.info(
        "BindingDB resolution: resolved=%s unresolved=%s n_unique_compounds_inchikey=%s",
        len(compound_target_rows), unresolved_count, len(resolved_compounds),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
