"""
Deep diagnosis of BindingDB resolution gaps (compound_unresolved, target_unresolved).
Uses same filtering as expansion pipeline (human organism). Writes:
  bindingdb_resolution_gaps.json
  bindingdb_unresolved_compounds_top.csv
  bindingdb_unresolved_targets_top.csv
  bindingdb_targetname_patterns.csv
No external API calls. Deterministic.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and __import__("pandas").isna(val)):
        return ""
    return str(val).strip()


def _load_df(path: Path):
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            return __import__("pandas").read_parquet(path)
        return __import__("pandas").read_csv(path, low_memory=False, dtype=str)
    except Exception:
        return None


def _load_bindingdb_edges(processed: Path) -> Tuple[Any, Optional[str], Optional[str]]:
    """
    Load BindingDB edges. Returns (dataframe, source_file_basename, error_message).
    Tries: compound_target_edges_bindingdb.parquet, then bindingdb_matched.parquet (with column normalization).
    """
    err = "BindingDB parquet not found (expected data/processed/phase16_bindingdb/compound_target_edges_bindingdb.parquet). Run Phase16 notebook to create it."
    primary = processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if not primary.exists():
        for p in processed.rglob("compound_target_edges_bindingdb.parquet"):
            primary = p
            break
    if primary.exists():
        df = _load_df(primary)
        if df is not None and not df.empty:
            return df, primary.name, None
        if df is not None and df.empty:
            err = f"BindingDB parquet has 0 rows: {primary}"
        else:
            try:
                __import__("pandas").read_parquet(primary)
            except Exception as e:
                err = f"BindingDB parquet unreadable: {primary}: {e}"
            else:
                err = f"BindingDB parquet unreadable: {primary}"

    # Fallback: bindingdb_matched.parquet (different column names)
    fallback = processed / "phase16_bindingdb" / "bindingdb_matched.parquet"
    if fallback.exists():
        df = _load_df(fallback)
        if df is not None and not df.empty:
            # Normalize to expected columns: compound_id, target_name, organism, uniprot_id
            rename = {}
            for c in df.columns:
                if "target" in c.lower() and "name" in c.lower():
                    rename[c] = "target_name"
                elif "organism" in c.lower() or ("source" in c.lower() and "curator" in c.lower()):
                    rename[c] = "organism"
            df = df.rename(columns=rename)
            if "uniprot_id" not in df.columns:
                df["uniprot_id"] = ""
            return df, fallback.name, None
    return None, None, err


def _col(df, names: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    low = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "_")
        if k in low:
            return low[k]
    return None


def _compound_id_pattern(cid: str) -> str:
    """Return a pattern label for compound_id (e.g. COCONUT_, CHEMBL, numeric, etc.)."""
    s = _safe_str(cid)
    if not s:
        return "empty"
    if s.upper().startswith("COCONUT_"):
        return "COCONUT_"
    if s.upper().startswith("CHEMBL"):
        return "CHEMBL"
    if s.upper().startswith("CID") or (s.isdigit() and len(s) <= 12):
        return "CID_or_numeric"
    if s.upper().startswith("DB"):
        return "DB_"
    if len(s) >= 25 and "-" in s and s.replace("-", "").isalnum():
        return "InChIKey_like"
    for prefix in ("SMILES", "INCHI", "ZINC", "DRUGBANK"):
        if s.upper().startswith(prefix):
            return prefix + "_"
    return "other"


def _normalize_target_name(s: str) -> str:
    """Strip bracket content, lower, remove punctuation, collapse spaces."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s*\[[^\]]*\]\s*", " ", s)
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = re.sub(r"[^\w\s\-]", " ", s)
    return " ".join(s.split())


def run_diagnosis(repo_root: Path) -> Dict[str, Any]:
    processed = repo_root / "data" / "processed"
    canonical = processed / "canonical"
    df, source_file, load_err = _load_bindingdb_edges(processed)
    if df is None or df.empty:
        return {"error": load_err or "BindingDB parquet empty", "n_rows": 0}

    organism_col = _col(df, ["organism"])
    cmp_col = "compound_id"
    uniprot_col = _col(df, ["uniprot_id", "uniprot"])
    target_col = _col(df, ["target_name"])

    n_total = len(df)
    skip_organism = 0
    human_rows = []
    for _, r in df.iterrows():
        org = _safe_str(r.get(organism_col)) if organism_col else ""
        if org and "homo sapiens" not in org.lower() and "human" not in org.lower():
            skip_organism += 1
            continue
        human_rows.append(dict(r))

    # Resolve compounds (reuse resolver)
    try:
        from src.phase14.bindingdb_compound_resolver import resolve_bindingdb_compounds_batch
        compound_ids = list({_safe_str(r.get(cmp_col)) for r in human_rows if _safe_str(r.get(cmp_col))})
        res_df, comp_report = resolve_bindingdb_compounds_batch(compound_ids, repo_root)
        resolved_cmp = set(res_df[res_df["inchikey"].astype(str).str.len() >= 25]["compound_id_raw"].tolist())
    except Exception as e:
        resolved_cmp = set()
        comp_report = {"n_resolved": 0, "n_total": 0}
        import traceback
        traceback.print_exc()

    # Resolve targets (reuse resolver)
    try:
        from src.phase14.uniprot_gene_resolver import build_uniprot_to_gene_map, build_target_name_to_gene_map, resolve_target_to_gene
        uniprot_to_gene, _ = build_uniprot_to_gene_map(repo_root)
        target_name_to_gene, _ = build_target_name_to_gene_map(repo_root)
    except Exception as e:
        uniprot_to_gene = {}
        target_name_to_gene = {}
        import traceback
        traceback.print_exc()

    skip_compound = 0
    skip_target = 0
    unresolved_compound_ids: List[str] = []
    unresolved_compound_patterns: Counter = Counter()
    unresolved_targets: List[Dict[str, Any]] = []
    uniprot_pattern_counts: Counter = Counter()
    normalized_target_counts: Counter = Counter()

    for r in human_rows:
        cid = _safe_str(r.get(cmp_col))
        if not cid:
            continue
        if cid not in resolved_cmp:
            skip_compound += 1
            unresolved_compound_ids.append(cid)
            unresolved_compound_patterns[_compound_id_pattern(cid)] += 1
            continue
        uniprot = _safe_str(r.get(uniprot_col)) if uniprot_col else ""
        target_name = _safe_str(r.get(target_col)) if target_col else ""
        gene, _, _ = resolve_target_to_gene(uniprot or None, target_name or None, uniprot_to_gene, target_name_to_gene)
        if not gene:
            skip_target += 1
            uniprot_pattern = "missing" if not uniprot else "present"
            uniprot_pattern_counts[uniprot_pattern] += 1
            norm = _normalize_target_name(target_name)
            if norm:
                normalized_target_counts[norm] += 1
            unresolved_targets.append({"uniprot_id": uniprot, "target_name": target_name, "normalized": norm})

    # Top 30 compound patterns with 50 examples each
    pattern_to_examples: Dict[str, List[str]] = {}
    seen_per_pattern: Dict[str, Set[str]] = {}
    for cid in unresolved_compound_ids:
        pat = _compound_id_pattern(cid)
        seen_per_pattern.setdefault(pat, set())
        if len(seen_per_pattern[pat]) < 50:
            seen_per_pattern[pat].add(cid)
    for pat, examples in seen_per_pattern.items():
        pattern_to_examples[pat] = sorted(examples)[:50]

    # Check compound_master for unresolved IDs
    master_path = canonical / "compound_master.csv"
    master_columns_checked = []
    in_master_count = 0
    if master_path.exists():
        master = _load_df(master_path)
        if master is not None and not master.empty:
            master_columns_checked = list(master.columns)
            for col in ["bindingdb_id", "chembl_id", "cid", "smiles", "name", "fdb_id_norm"]:
                if col not in master.columns:
                    continue
                vals = set(master[col].dropna().astype(str).str.strip().str.lower())
                for cid in unresolved_compound_ids[:500]:
                    cid_low = cid.lower().strip()
                    if cid_low in vals or cid in set(master[col].dropna().astype(str).str.strip()):
                        in_master_count += 1
                        break

    # Target coverage: match normalized target names against targets.parquet and target_functional_clusters
    targets_parquet_path = canonical / "targets.parquet"
    targets_match_count = 0
    clusters_match_count = 0
    top_unmatched_targets: List[Tuple[str, int]] = []
    t_col = None
    clusters_path = processed / "features" / "target_functional_clusters.csv"
    cdf = None
    tc_col = None
    if normalized_target_counts:
        if targets_parquet_path.exists():
            tdf = _load_df(targets_parquet_path)
            if tdf is not None and not tdf.empty:
                t_col = _col(tdf, ["target_name", "pref_name"])
                if t_col:
                    master_norm = {_normalize_target_name(_safe_str(x)) for x in tdf[t_col].dropna()}
                    for norm in normalized_target_counts:
                        if norm in master_norm:
                            targets_match_count += 1
        if clusters_path.exists():
            cdf = _load_df(clusters_path)
            if cdf is not None and not cdf.empty:
                tc_col = _col(cdf, ["sample_targets"])
                if tc_col:
                    cluster_norms = set()
                    for _, row in cdf.iterrows():
                        raw = _safe_str(row.get(tc_col))
                        if not raw:
                            continue
                        try:
                            import ast
                            arr = ast.literal_eval(raw) if raw.startswith("[") else [x.strip() for x in raw.replace("'", "").split(",")]
                        except Exception:
                            arr = [raw]
                        for t in arr:
                            cluster_norms.add(_normalize_target_name(t))
                    for norm in normalized_target_counts:
                        if norm in cluster_norms:
                            clusters_match_count += 1
        all_norms = set(normalized_target_counts.keys())
        matched_in_targets = set()
        matched_in_clusters = set()
        if targets_parquet_path.exists():
            tdf2 = _load_df(targets_parquet_path)
            if tdf2 is not None and not tdf2.empty:
                tc = _col(tdf2, ["target_name", "pref_name"])
                if tc:
                    matched_in_targets = {_normalize_target_name(_safe_str(x)) for x in tdf2[tc].dropna()} & all_norms
        if clusters_path.exists():
            cdf2 = _load_df(clusters_path)
            if cdf2 is not None and not cdf2.empty:
                tcc = _col(cdf2, ["sample_targets"])
                if tcc:
                    for _, row in cdf2.iterrows():
                        raw = _safe_str(row.get(tcc))
                        if not raw:
                            continue
                        try:
                            import ast
                            arr = ast.literal_eval(raw) if raw.startswith("[") else [x.strip() for x in raw.replace("'", "").split(",")]
                        except Exception:
                            arr = [raw]
                        for t in arr:
                            matched_in_clusters.add(_normalize_target_name(t))
                    matched_in_clusters &= all_norms
        matched_any = matched_in_targets | matched_in_clusters
        for norm, cnt in normalized_target_counts.most_common(500):
            if norm not in matched_any:
                top_unmatched_targets.append((norm, cnt))
        top_unmatched_targets = top_unmatched_targets[:200]

    n_human = len(human_rows)
    gaps = {
        "bindingdb_source_file": source_file or "compound_target_edges_bindingdb.parquet",
        "n_total_rows": n_total,
        "n_after_human_filter": n_human,
        "skip_organism_not_human": skip_organism,
        "skip_compound_unresolved": skip_compound,
        "skip_target_unresolved": skip_target,
        "compound_resolution_pct": round(100.0 * comp_report.get("n_resolved", 0) / max(1, comp_report.get("n_total", 1)), 2),
        "compound_pattern_counts": dict(unresolved_compound_patterns.most_common(30)),
        "compound_pattern_examples": pattern_to_examples,
        "compound_master_columns_checked": master_columns_checked,
        "unresolved_compounds_in_master_approx": in_master_count,
        "uniprot_pattern_counts": dict(uniprot_pattern_counts),
        "target_normalized_top200": [{"normalized_target_name": k, "count": v} for k, v in normalized_target_counts.most_common(200)],
        "target_fraction_matched_targets_parquet": round(targets_match_count / max(1, len(normalized_target_counts)), 4) if normalized_target_counts else 0,
        "target_fraction_matched_clusters": round(clusters_match_count / max(1, len(normalized_target_counts)), 4) if normalized_target_counts else 0,
        "top_unmatched_target_names": top_unmatched_targets[:50],
        "unresolved_targets": unresolved_targets[:1000],
    }
    return gaps


def main() -> int:
    import pandas as pd
    repo_root = Path(REPO_ROOT).resolve()
    reports_dir = repo_root / "data" / "processed" / "canonical" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    gaps = run_diagnosis(repo_root)
    if gaps.get("error"):
        print("ERROR:", gaps["error"])
        return 1

    with open(reports_dir / "bindingdb_resolution_gaps.json", "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2)
    print("Wrote bindingdb_resolution_gaps.json")

    # bindingdb_unresolved_compounds_top.csv: pattern, compound_id (one per row for top patterns)
    rows_cmp = []
    for pat, examples in gaps.get("compound_pattern_examples", {}).items():
        for cid in examples:
            rows_cmp.append({"pattern": pat, "compound_id": cid})
    if rows_cmp:
        pd.DataFrame(rows_cmp).to_csv(reports_dir / "bindingdb_unresolved_compounds_top.csv", index=False)
        print("Wrote bindingdb_unresolved_compounds_top.csv")

    # bindingdb_unresolved_targets_top.csv: uniprot_id, target_name, normalized
    rows_tgt = gaps.get("unresolved_targets", [])
    if rows_tgt:
        pd.DataFrame(rows_tgt).to_csv(reports_dir / "bindingdb_unresolved_targets_top.csv", index=False)
        print("Wrote bindingdb_unresolved_targets_top.csv")

    # bindingdb_targetname_patterns.csv: normalized_target_name, count
    patterns_rows = [{"normalized_target_name": k, "count": v} for k, v in gaps.get("target_normalized_top200", [])]
    if not patterns_rows and gaps.get("top_unmatched_target_names"):
        patterns_rows = [{"normalized_target_name": k, "count": v} for k, v in gaps["top_unmatched_target_names"]]
    if patterns_rows:
        pd.DataFrame(patterns_rows).to_csv(reports_dir / "bindingdb_targetname_patterns.csv", index=False)
        print("Wrote bindingdb_targetname_patterns.csv")
    else:
        tn_top = gaps.get("target_normalized_top200", [])
        if tn_top:
            pd.DataFrame(tn_top).to_csv(reports_dir / "bindingdb_targetname_patterns.csv", index=False)
            print("Wrote bindingdb_targetname_patterns.csv")

    print("skip_organism:", gaps.get("skip_organism_not_human"))
    print("skip_compound_unresolved:", gaps.get("skip_compound_unresolved"))
    print("skip_target_unresolved:", gaps.get("skip_target_unresolved"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
