"""
Option 1: Build compound_gene expanded canonical (Strengthen Human Gene Layer).
Loads food_compound_gene_links + BindingDB; resolves compounds (COCONUT->InChIKey) and targets (UniProt/target_name->gene);
writes compound_gene_expanded_canonical.csv and compound_gene_expansion_report.json.
No external API calls. Windows-safe. Asserts with clear messages.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


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


def _build_fdb_to_inchikey(repo_root: Path) -> Dict[str, str]:
    """From compound_identity_bridge (FOODB namespace) build FDB_* -> inchikey."""
    from src.phase14.compound_identity import normalize_fdb_id
    bridge_path = repo_root / "data" / "processed" / "canonical" / "compound_identity_bridge.csv"
    fdb_to_ik: Dict[str, str] = {}
    if not bridge_path.exists():
        return fdb_to_ik
    df = _load_df(bridge_path)
    if df is None or df.empty:
        return fdb_to_ik
    ns_col = next((c for c in df.columns if "namespace" in c.lower()), None)
    sid_col = next((c for c in df.columns if "source_id" in c.lower()), None)
    ik_col = next((c for c in df.columns if "inchikey" in c.lower()), None)
    if not sid_col or not ik_col:
        return fdb_to_ik
    for _, r in df.iterrows():
        if ns_col and _safe_str(r.get(ns_col)).upper() != "FOODB":
            continue
        sid = _safe_str(r.get(sid_col))
        fdb = normalize_fdb_id(sid) or sid
        if not fdb or "FDB_" not in fdb.upper():
            continue
        ik = _safe_str(r.get(ik_col))
        if len(ik) >= 25:
            fdb_to_ik[fdb] = ik
    return fdb_to_ik


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase14: Build compound_gene_expanded canonical (Option 1)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()

    assert (repo_root / "data").exists(), "Repo root must contain data/"
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"
    reports_dir = canonical_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load food_compound_gene_links
    phase12 = processed / "phase12_genetics"
    fcg_path = phase12 / "food_compound_gene_links.parquet"
    if not fcg_path.exists():
        for p in processed.rglob("food_compound_gene_links.parquet"):
            fcg_path = p
            break
    assert fcg_path.exists(), "food_compound_gene_links.parquet not found under data/processed"
    df_food = _load_df(fcg_path)
    assert df_food is not None and not df_food.empty, "food_compound_gene_links.parquet is empty or unreadable"
    cmp_col = next((c for c in df_food.columns if "compound" in c.lower() and "id" in c.lower()), None)
    gene_col = next((c for c in df_food.columns if "gene" in c.lower()), None)
    assert cmp_col and gene_col, "food_compound_gene_links must have compound_id and gene_symbol/gene"

    # 2) Load BindingDB edges
    bdb_path = processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if not bdb_path.exists():
        for p in processed.rglob("compound_target_edges_bindingdb.parquet"):
            bdb_path = p
            break
    assert bdb_path.exists(), "compound_target_edges_bindingdb.parquet not found"
    df_bdb = _load_df(bdb_path)
    assert df_bdb is not None and not df_bdb.empty, "BindingDB parquet is empty or unreadable"
    assert "compound_id" in df_bdb.columns, "BindingDB must have compound_id"

    # 3) Resolve BindingDB compounds to InChIKey
    from src.phase14.bindingdb_compound_resolver import resolve_bindingdb_compounds_batch, build_coconut_to_inchikey_map
    compound_ids_bdb = df_bdb["compound_id"].dropna().astype(str).str.strip().unique().tolist()
    res_df, compound_report = resolve_bindingdb_compounds_batch(compound_ids_bdb, repo_root)
    coconut_to_ik = build_coconut_to_inchikey_map(repo_root)
    resolved_ik = set(res_df[res_df["inchikey"].astype(str).str.len() >= 25]["compound_id_raw"].tolist())  # for reporting

    # 4) Resolve targets to gene_symbol (alias map written to canonical/reports/target_name_alias_map.csv)
    from src.phase14.uniprot_gene_resolver import build_uniprot_to_gene_map, build_target_name_to_gene_map, resolve_target_to_gene, normalize_target_name
    uniprot_to_gene, _ = build_uniprot_to_gene_map(repo_root)
    target_name_to_gene, _ = build_target_name_to_gene_map(repo_root)

    # 5) Build food links rows (FDB -> InChIKey via bridge)
    fdb_to_ik = _build_fdb_to_inchikey(repo_root)
    from src.phase14.compound_identity import normalize_fdb_id
    food_rows: List[Dict[str, Any]] = []
    for _, r in df_food.iterrows():
        cid = _safe_str(r.get(cmp_col))
        g = _safe_str(r.get(gene_col)).upper()
        if not cid or not g:
            continue
        fdb = normalize_fdb_id(cid) or cid
        ik = fdb_to_ik.get(fdb)
        if not ik and len(cid) >= 25 and cid.replace("-", "").replace(" ", "").isalnum():
            from src.phase14.compound_identity import looks_like_inchikey, normalize_inchikey
            if looks_like_inchikey(cid):
                ik = normalize_inchikey(cid)
        if not ik:
            continue
        food_rows.append({
            "compound_id": ik,
            "gene_symbol": g,
            "source": "food_compound_gene_links",
            "evidence_fields": "",
            "resolver_used_compound": "fdb_bridge",
            "resolver_used_target": "direct",
        })

    # 6) Build BindingDB rows (human only; compound resolved to InChIKey; target to gene)
    uniprot_col = next((c for c in df_bdb.columns if "uniprot" in c.lower()), None)
    target_col = next((c for c in df_bdb.columns if "target" in c.lower() and "name" in c.lower()), None)
    organism_col = next((c for c in df_bdb.columns if "organism" in c.lower()), None)
    affinity_col = next((c for c in df_bdb.columns if "affinity" in c.lower()), None)
    meas_col = next((c for c in df_bdb.columns if "measurement" in c.lower()), None)
    pubmed_col = next((c for c in df_bdb.columns if "pubmed" in c.lower()), None)

    res_lookup = res_df.set_index("compound_id_raw")["inchikey"].to_dict()
    resolver_lookup = res_df.set_index("compound_id_raw")["resolver_used"].to_dict() if "resolver_used" in res_df.columns else {}
    binding_rows: List[Dict[str, Any]] = []
    n_bdb_skipped_organism = 0
    n_bdb_skipped_compound = 0
    n_bdb_skipped_target = 0
    unresolved_target_names: List[str] = []
    for _, r in df_bdb.iterrows():
        organism = _safe_str(r.get(organism_col)) if organism_col else ""
        if organism and "homo sapiens" not in organism.lower() and "human" not in organism.lower():
            n_bdb_skipped_organism += 1
            continue
        cid_raw = _safe_str(r.get("compound_id"))
        if not cid_raw:
            continue
        ik = res_lookup.get(cid_raw) or res_lookup.get(cid_raw.split(".")[0] if "." in cid_raw else "")
        if not ik or len(str(ik)) < 25:
            n_bdb_skipped_compound += 1
            continue
        uniprot = _safe_str(r.get(uniprot_col)) if uniprot_col else ""
        target_name = _safe_str(r.get(target_col)) if target_col else ""
        gene, resolver_target, _ = resolve_target_to_gene(uniprot or None, target_name or None, uniprot_to_gene, target_name_to_gene)
        if not gene:
            n_bdb_skipped_target += 1
            if target_name:
                unresolved_target_names.append(normalize_target_name(target_name))
            continue
        ev = []
        if affinity_col and r.get(affinity_col) is not None:
            ev.append("affinity_nM=%s" % r.get(affinity_col))
        if meas_col and r.get(meas_col):
            ev.append("measurement_type=%s" % r.get(meas_col))
        if pubmed_col and r.get(pubmed_col):
            ev.append("pubmed_id=%s" % r.get(pubmed_col))
        if organism:
            ev.append("organism=%s" % organism[:50])
        evidence_fields = "|".join(ev)[:500] if ev else ""
        res_compound = resolver_lookup.get(cid_raw) or resolver_lookup.get(cid_raw.split(".")[0] if "." in cid_raw else "") or "coconut_map"
        binding_rows.append({
            "compound_id": ik,
            "gene_symbol": gene,
            "source": "bindingdb",
            "evidence_fields": evidence_fields,
            "resolver_used_compound": res_compound,
            "resolver_used_target": resolver_target,
        })

    # 7) Merge and dedupe
    all_rows = food_rows + binding_rows
    assert len(all_rows) > 0, "No edges produced: check FDB->InChIKey bridge and BindingDB resolution"
    df_out = pd.DataFrame(all_rows).drop_duplicates(subset=["compound_id", "gene_symbol", "source"], keep="first")
    df_out = df_out.drop_duplicates(subset=["compound_id", "gene_symbol"], keep="first")

    # 8) Overlap vs ingredient_compound_canonical
    ing_path = canonical_dir / "ingredient_compound_canonical.csv"
    overlap_metrics: Dict[str, Any] = {"n_overlap": 0, "overlap_vs_cg": 0.0, "overlap_vs_ic": 0.0, "n_ic_compounds": 0, "n_cg_compounds": len(df_out["compound_id"].unique())}
    if ing_path.exists():
        ing_df = _load_df(ing_path)
        if ing_df is not None and not ing_df.empty and "compound_id" in ing_df.columns:
            ic_set = set(ing_df["compound_id"].dropna().astype(str).str.strip().str.upper())
            ic_set = {x for x in ic_set if x and str(x) != "NAN"}
            cg_set = set(df_out["compound_id"].dropna().astype(str).str.strip().str.upper())
            cg_set = {x for x in cg_set if x and str(x) != "NAN"}
            overlap_metrics["n_ic_compounds"] = len(ic_set)
            overlap_metrics["n_overlap"] = len(ic_set & cg_set)
            overlap_metrics["overlap_vs_cg"] = round(overlap_metrics["n_overlap"] / len(cg_set), 4) if cg_set else 0.0
            overlap_metrics["overlap_vs_ic"] = round(overlap_metrics["n_overlap"] / len(ic_set), 4) if ic_set else 0.0

    # 9) Resolution gaps and top 200 unresolved target names
    rc = compound_report.get("resolver_counts") or {}
    cid_resolved = sum(rc.get(k, 0) for k in ("cid_master", "cid_pharmgkb", "cid_scan", "cid_map"))
    cid_unresolved = rc.get("unresolved", 0) + rc.get("unresolved_coconut", 0) + rc.get("unresolved_empty", 0)
    top_unresolved_target_counts = Counter(unresolved_target_names).most_common(200)
    top_200_unresolved_target_names = [{"normalized_target_name": n, "count": c} for n, c in top_unresolved_target_counts]
    for i, (norm_name, count) in enumerate(top_unresolved_target_counts[:200]):
        logger.info("Unresolved target %d: %s (count=%s)", i + 1, norm_name, count)

    gaps = {
        "cid_resolved": cid_resolved,
        "cid_unresolved": cid_unresolved,
        "target_unresolved": n_bdb_skipped_target,
        "top_200_unresolved_target_names": top_200_unresolved_target_names,
        "compound_resolver_counts": rc,
    }
    with open(reports_dir / "bindingdb_resolution_gaps.json", "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2)
    logger.info("Wrote bindingdb_resolution_gaps.json (cid_resolved=%s, cid_unresolved=%s, target_unresolved=%s)", cid_resolved, cid_unresolved, n_bdb_skipped_target)

    # 10) Report (include diagnostics: bindingdb rows processed, % target/compound resolved, top failure patterns)
    n_from_food = sum(1 for _ in food_rows)
    n_from_binding = sum(1 for _ in binding_rows)
    n_bdb_human = len(df_bdb) - n_bdb_skipped_organism
    pct_compound_resolved = round(100.0 * (compound_report.get("n_resolved") or 0) / max(1, compound_report.get("n_total") or 1), 2)
    pct_target_resolved = round(100.0 * (n_bdb_human - n_bdb_skipped_compound - n_bdb_skipped_target) / max(1, n_bdb_human - n_bdb_skipped_compound), 2) if (n_bdb_human - n_bdb_skipped_compound) > 0 else 0.0
    top_10_failure_patterns = []
    if compound_report.get("resolver_counts"):
        for res, cnt in sorted(compound_report["resolver_counts"].items(), key=lambda x: -x[1])[:10]:
            top_10_failure_patterns.append({"resolver_or_failure": res, "count": cnt})
    report = {
        "n_edges_total": len(df_out),
        "n_unique_compounds": int(df_out["compound_id"].nunique()),
        "n_unique_genes": int(df_out["gene_symbol"].nunique()),
        "n_from_food_compound_gene_links": n_from_food,
        "n_from_bindingdb": n_from_binding,
        "bindingdb_contribution": n_from_binding,
        "bindingdb_rows_processed": n_bdb_human,
        "pct_compound_resolved": pct_compound_resolved,
        "pct_target_resolved": pct_target_resolved,
        "compound_resolution": compound_report,
        "resolver_counts": compound_report.get("resolver_counts"),
        "cid_resolved": cid_resolved,
        "cid_unresolved": cid_unresolved,
        "bindingdb_skipped_organism": n_bdb_skipped_organism,
        "bindingdb_skipped_compound_unresolved": n_bdb_skipped_compound,
        "bindingdb_skipped_target_unresolved": n_bdb_skipped_target,
        "overlap_with_ingredient_compound": overlap_metrics,
        "top_10_failure_patterns": top_10_failure_patterns,
        "top_failure_reasons": {
            "compound_unresolved": "COCONUT/other id not in mapping" if n_bdb_skipped_compound else None,
            "target_unresolved": "uniprot_id missing and target_name not matched" if n_bdb_skipped_target else None,
        },
    }
    report["top_200_unresolved_target_names"] = top_200_unresolved_target_names[:50]

    # 11) Write
    out_csv = canonical_dir / "compound_gene_expanded_canonical.csv"
    df_out.to_csv(out_csv, index=False)
    report_path = reports_dir / "compound_gene_expansion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s: %s edges, %s compounds, %s genes", out_csv.name, len(df_out), report["n_unique_compounds"], report["n_unique_genes"])
    logger.info("BindingDB contribution: %s edges", n_from_binding)
    logger.info("Overlap vs ingredient_compound: n_overlap=%s, overlap_vs_cg=%.2f%%", overlap_metrics["n_overlap"], 100 * overlap_metrics["overlap_vs_cg"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
