"""
Phase14: Build compound_gene_canonical.csv with InChIKey resolution and provenance.

Resolves compound_gene rows (e.g. food_compound_gene_links.parquet) via:
(a) FDB ID normalize -> compound_master
(b) compound_name exact + fuzzy match -> compound_master
(c) CID fallback

Writes:
- data/processed/canonical/compound_gene_canonical.csv (compound_id=InChIKey, gene, source_file, match_type, match_score, original_compound_id, original_name)
- reports/compound_gene_resolution_stats.json
- reports/unresolved_compound_gene_top.csv
- reports/overlap_after_resolution.json

Run from repo root: python scripts/phase14/build_compound_gene_canonical.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from src.phase14.compound_identity import (
        load_csv_or_parquet,
        build_compound_master_from_sources,
        build_fdb_to_inchikey_with_fallbacks,
        canonicalize_compound_gene_with_provenance,
        write_compound_gene_diagnostics,
    )

    parser = argparse.ArgumentParser(description="Build compound_gene_canonical with provenance")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    processed = repo_root / "data" / "processed"
    canonical_dir = processed / "canonical"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load or build compound_master
    master_path = canonical_dir / "compound_master.csv"
    if master_path.exists():
        compound_master = load_csv_or_parquet(master_path)
    else:
        compound_master = build_compound_master_from_sources(repo_root, None)
    if compound_master is None or compound_master.empty:
        print("ERROR: compound_master not found. Run: python -m src.phase14.compound_identity --repo-root .")
        return 1

    # 2) FDB -> InChIKey mapping
    fdb_to_ik, _, _ = build_fdb_to_inchikey_with_fallbacks(repo_root, compound_master, report_dir=reports_dir)

    # 3) Load compound_gene source (phase12)
    cg_path = processed / "phase12_genetics" / "food_compound_gene_links.parquet"
    if not cg_path.exists():
        cg_path = canonical_dir / "compound_gene_links.csv"
    if not cg_path.exists():
        cg_path = canonical_dir / "compound_gene_links.parquet"
    if not cg_path.exists():
        print("ERROR: food_compound_gene_links.parquet or compound_gene_links not found.")
        return 1

    cg = load_csv_or_parquet(cg_path)
    if cg is None or cg.empty:
        print("ERROR: Could not load compound_gene file.")
        return 1

    cmp_col = next((c for c in cg.columns if "compound" in c.lower() and "id" in c.lower()), None) or cg.columns[0]
    gene_col = next((c for c in cg.columns if c.lower() in ("gene", "gene_symbol", "symbol")), None) or next((c for c in cg.columns if "gene" in c.lower()), cg.columns[1])
    name_col = next((c for c in cg.columns if "compound_name" in c.lower() or c.lower() == "name"), None)
    src_col = next((c for c in cg.columns if "source" in c.lower()), None)
    cols = [cmp_col, gene_col]
    if name_col:
        cols.append(name_col)
    if src_col:
        cols.append(src_col)
    cg_work = cg[cols].copy()
    cg_work = cg_work.dropna(subset=[cmp_col, gene_col], how="all")
    cg_work[cmp_col] = cg_work[cmp_col].astype(str).str.strip()
    cg_work[gene_col] = cg_work[gene_col].astype(str).str.strip().str.upper()
    cg_work = cg_work[(cg_work[cmp_col].str.len() > 0) & (cg_work[gene_col].str.len() > 0)]

    # 4) Canonicalize with provenance
    cg_canonical, cg_stats = canonicalize_compound_gene_with_provenance(
        cg_work, compound_master, fdb_to_ik,
        compound_id_col=cmp_col, gene_col=gene_col,
        compound_name_col=name_col, source_col=src_col,
        source_file_default=cg_path.name,
    )

    # 5) Load ingredient_compound_canonical for overlap
    ing_path = canonical_dir / "ingredient_compound_canonical.csv"
    if ing_path.exists():
        ing_canonical = load_csv_or_parquet(ing_path)
    else:
        ing_canonical = __import__("pandas").DataFrame(columns=["compound_id"])

    if ing_canonical is None:
        ing_canonical = __import__("pandas").DataFrame(columns=["compound_id"])

    # 6) Write outputs
    cg_canonical.to_csv(canonical_dir / "compound_gene_canonical.csv", index=False)
    write_compound_gene_diagnostics(
        reports_dir, cg_stats, cg_work, cg_canonical, ing_canonical,
        compound_id_col=cmp_col, compound_name_col=name_col, gene_col=gene_col,
    )

    # 7) Summary to console
    total = len(cg_work)
    resolved = len(cg_canonical)
    pct = (100.0 * resolved / total) if total else 0
    print("--- compound_gene canonicalization ---")
    print("input rows: %s" % total)
    print("resolved to InChIKey: %s (%.1f%%)" % (resolved, pct))
    print("match routes: %s" % json.dumps(cg_stats, indent=2))
    set_ing = set(ing_canonical["compound_id"].dropna().astype(str).str.strip()) if not ing_canonical.empty and "compound_id" in ing_canonical.columns else set()
    set_cg = set(cg_canonical["compound_id"].dropna().astype(str).str.strip()) if not cg_canonical.empty else set()
    overlap = len(set_ing & set_cg)
    overlap_pct = (100.0 * overlap / len(set_ing)) if set_ing else 0
    print("overlap with ingredient_compound_canonical: %s compounds (%.2f%%)" % (overlap, overlap_pct))
    print("wrote: %s" % (canonical_dir / "compound_gene_canonical.csv"))
    print("wrote: reports/compound_gene_resolution_stats.json, unresolved_compound_gene_top.csv, overlap_after_resolution.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
