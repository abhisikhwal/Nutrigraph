"""
Build compound_gene_expanded_v4 using the compound identity bridge.
Resolves food_compound_gene_links (FooDB) and BindingDB edges to InChIKey via bridge.
Writes: compound_gene_expanded_v4_canonical.csv, compound_gene_expanded_v4_report.json.
Report includes overlap_with_ingredient_compound (n_overlap, overlap_vs_cg, overlap_vs_ic).
Run: python scripts/phase16/build_compound_gene_expanded_v4.py --repo-root .
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


def load_bridge_resolver(repo_root: Path) -> Dict[str, str]:
    """Load compound_identity_bridge.csv and build source_id -> inchikey (and FDB normalized)."""
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"
    bridge_path = canonical_dir / "compound_identity_bridge.csv"
    resolver: Dict[str, str] = {}
    if not bridge_path.exists():
        logger.warning("Bridge not found at %s. Run bridge first.", bridge_path)
        return resolver
    try:
        df = pd.read_csv(bridge_path, low_memory=False)
    except Exception as e:
        logger.warning("Could not load bridge: %s", e)
        return resolver
    if df.empty or "source_id" not in df.columns:
        return resolver
    ik_col = next((c for c in df.columns if "inchikey" in c.lower()), None)
    if not ik_col:
        return resolver
    for _, r in df.iterrows():
        sid = _safe_str(r.get("source_id"))
        ik = _safe_str(r.get(ik_col))
        if not sid:
            continue
        if ik and len(ik) >= 25:
            resolver[sid] = ik
            if sid.startswith("COCONUT_") and "." in sid:
                base = sid.split(".")[0]
                if base and base not in resolver:
                    resolver[base] = ik
    # FooDB: also register normalized form
    from src.phase14.compound_identity import normalize_fdb_id, looks_like_fdb_id
    for sid in list(resolver.keys()):
        if looks_like_fdb_id(sid):
            norm = normalize_fdb_id(sid)
            if norm and norm not in resolver:
                resolver[norm] = resolver[sid]
    return resolver


def resolve_to_inchikey(compound_id: Any, resolver: Dict[str, str]) -> Optional[str]:
    from src.phase14.compound_identity import normalize_fdb_id, looks_like_inchikey, normalize_inchikey
    s = _safe_str(compound_id)
    if not s:
        return None
    if looks_like_inchikey(s):
        return normalize_inchikey(s)
    if s in resolver:
        return resolver[s]
    if s.startswith("COCONUT_") and "." in s:
        base = s.split(".")[0]
        if base in resolver:
            return resolver[base]
    fdb = normalize_fdb_id(s)
    if fdb and fdb in resolver:
        return resolver[fdb]
    return None


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compound_gene_expanded_v4 via identity bridge")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    resolver = load_bridge_resolver(repo_root)
    logger.info("Bridge resolver size: %s", len(resolver))

    rows: List[Dict[str, Any]] = []

    # 1) food_compound_gene_links.parquet
    fcg = processed / "phase12_genetics" / "food_compound_gene_links.parquet"
    if fcg.exists():
        df = load_csv_or_parquet(fcg)
        if df is not None and not df.empty:
            cmp_col = next((c for c in df.columns if "compound" in c.lower() and "id" in c.lower()), None)
            gene_col = next((c for c in df.columns if "gene" in c.lower()), None)
            if cmp_col and gene_col:
                for _, r in df.iterrows():
                    raw_cid = r.get(cmp_col)
                    ik = resolve_to_inchikey(raw_cid, resolver)
                    if not ik or len(ik) < 25:
                        continue
                    gene = _safe_str(r.get(gene_col))
                    if not gene:
                        continue
                    rows.append({
                        "compound_id": ik,
                        "gene_symbol": gene,
                        "source": "food_compound_gene_links",
                        "evidence_fields": "",
                    })
    logger.info("From food_compound_gene_links: %s edges", len(rows))

    # 2) BindingDB edges
    bdb = processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if bdb.exists():
        df = load_csv_or_parquet(bdb)
        if df is not None and not df.empty and "compound_id" in df.columns:
            target_col = next((c for c in df.columns if "target" in c.lower() and "name" in c.lower()), "target_name")
            gene_col = next((c for c in df.columns if "gene" in c.lower()), None)
            n_bdb = 0
            for _, r in df.iterrows():
                raw_cid = r.get("compound_id")
                ik = resolve_to_inchikey(raw_cid, resolver)
                if not ik or len(ik) < 25:
                    continue
                gene = _safe_str(r.get(gene_col)) if gene_col else _safe_str(r.get(target_col))
                if not gene:
                    continue
                rows.append({
                    "compound_id": ik,
                    "gene_symbol": gene,
                    "source": "bindingdb",
                    "evidence_fields": "",
                })
                n_bdb += 1
            logger.info("From BindingDB: %s edges", n_bdb)

    if not rows:
        out_df = pd.DataFrame(columns=["compound_id", "gene_symbol", "source", "evidence_fields"])
        out_df.to_csv(canonical_dir / "compound_gene_expanded_v4_canonical.csv", index=False)
        report = {
            "n_edges_total": 0,
            "n_unique_compounds": 0,
            "n_unique_genes": 0,
            "overlap_with_ingredient_compound": {"n_overlap": 0, "overlap_vs_cg": 0.0, "overlap_vs_ic": 0.0},
        }
        with open(canonical_dir / "compound_gene_expanded_v4_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.warning("No edges produced; wrote empty v4 outputs.")
        return 0

    out_df = pd.DataFrame(rows)
    # Dedup by (compound_id, gene_symbol, source)
    out_df = out_df.drop_duplicates(subset=["compound_id", "gene_symbol", "source"], keep="first")
    # Dedup by (compound_id, gene_symbol) keep first (best evidence)
    out_df = out_df.drop_duplicates(subset=["compound_id", "gene_symbol"], keep="first").reset_index(drop=True)

    out_path = canonical_dir / "compound_gene_expanded_v4_canonical.csv"
    out_df.to_csv(out_path, index=False)
    n_cmp = out_df["compound_id"].nunique()
    n_gene = out_df["gene_symbol"].nunique()
    logger.info("Wrote %s: rows=%s unique_compounds=%s unique_genes=%s", out_path.name, len(out_df), n_cmp, n_gene)

    # Overlap with ingredient_compound
    ic_path = canonical_dir / "ingredient_compound_canonical.csv"
    overlap_metrics = {"n_overlap": 0, "overlap_vs_cg": 0.0, "overlap_vs_ic": 0.0, "n_ic_compounds": 0, "n_cg_compounds": n_cmp}
    if ic_path.exists():
        try:
            ic_df = load_csv_or_parquet(ic_path)
            if ic_df is not None and not ic_df.empty and "compound_id" in ic_df.columns:
                ic_set = set(ic_df["compound_id"].dropna().astype(str).str.strip().str.upper())
                cg_set = set(out_df["compound_id"].dropna().astype(str).str.strip().str.upper())
                overlap_set = ic_set & cg_set
                n_overlap = len(overlap_set)
                n_ic = len(ic_set)
                n_cg = max(len(cg_set), 1)
                overlap_metrics = {
                    "n_overlap": n_overlap,
                    "overlap_vs_cg": round(n_overlap / n_cg, 4),
                    "overlap_vs_ic": round(n_overlap / n_ic, 4) if n_ic else 0.0,
                    "n_ic_compounds": n_ic,
                    "n_cg_compounds": len(cg_set),
                }
                logger.info("Overlap with ingredient_compound: n_overlap=%s overlap_vs_cg=%.2f%% overlap_vs_ic=%.2f%%",
                           n_overlap, 100 * overlap_metrics["overlap_vs_cg"], 100 * overlap_metrics["overlap_vs_ic"])
        except Exception as e:
            logger.warning("Could not compute overlap: %s", e)

    report = {
        "n_edges_total": len(out_df),
        "n_unique_compounds": n_cmp,
        "n_unique_genes": n_gene,
        "overlap_with_ingredient_compound": overlap_metrics,
    }
    report_path = canonical_dir / "compound_gene_expanded_v4_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", report_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
