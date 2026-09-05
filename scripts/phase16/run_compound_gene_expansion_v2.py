"""
Phase16A: Run CMP->GENE densification pipeline (PharmGKB + BindingDB -> InChIKey, target->gene).
Writes: compound_gene_expanded_canonical.csv, compound_gene_canonical.csv, reports/compound_gene_expansion_v2_report.json.
Usage: python scripts/phase16/run_compound_gene_expansion_v2.py --repo-root .
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase16A: Compound-gene expansion v2 (PharmGKB + BindingDB -> InChIKey)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root (default: script's repo)")
    parser.add_argument("--only-human", action="store_true", default=True, help="Filter BindingDB to human only (default: True)")
    parser.add_argument("--no-only-human", action="store_false", dest="only_human", help="Include non-human BindingDB targets")
    parser.add_argument("--write-debug", action="store_true", help="Write intermediate CSVs to canonical/reports/")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        logger.error("Repo root does not exist: %s", repo_root)
        return 1
    try:
        from src.phase16.compound_gene_expand import run_full_pipeline
        expanded_path, canonical_path, report_path, report = run_full_pipeline(
            repo_root, only_human=args.only_human, write_debug=args.write_debug
        )
    except RuntimeError as e:
        if "RDKit" in str(e):
            logger.error("%s", e)
            return 1
        raise
    # Crisp summary
    n_cmp = report.get("n_unique_compounds", 0)
    n_genes = report.get("n_unique_genes", 0)
    n_edges = report.get("n_edges_total", 0)
    ov = report.get("overlap_with_ingredient_compound", {})
    ov_cg = ov.get("overlap_vs_cg") or 0
    ov_ic = ov.get("overlap_vs_ic") or 0
    print("--- Phase16A summary ---")
    print("compound_gene_expanded_canonical.csv: %s edges, %s unique compounds, %s unique genes" % (n_edges, n_cmp, n_genes))
    print("compound_gene_canonical.csv: same (Phase14 will load this)")
    print("Overlap vs ingredient_compound: overlap_vs_cg=%.2f%%, overlap_vs_ic=%.2f%%" % (100 * ov_cg, 100 * ov_ic))
    print("Report: %s" % report_path)
    if n_cmp < 500:
        print("WARNING: unique compounds (%s) < 500; run identity bridge and check data/processed/canonical/reports/" % n_cmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
