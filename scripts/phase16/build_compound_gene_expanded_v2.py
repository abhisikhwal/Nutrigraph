"""
Build compound_gene_expanded_v2 (canonical compound→gene with target→gene bridging).
Writes: compound_gene_expanded_v2_raw.csv, compound_gene_expanded_v2_canonical.csv, compound_gene_expansion_v2_report.json.
Exit nonzero ONLY if no inchikey resolved for any source (catastrophic). Otherwise always writes outputs + diagnostics.
Run from repo root: python scripts/phase16/build_compound_gene_expanded_v2.py [--repo-root .]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    from src.phase16.compound_gene_expand_v2 import write_expanded_v2

    parser = argparse.ArgumentParser(description="Build compound_gene_expanded_v2 canonical")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "data").exists():
        logger.error("Repo root has no data/: %s", repo_root)
        return 1
    raw_path, canon_path, report_path, report = write_expanded_v2(repo_root)
    print("Wrote:", raw_path, canon_path, report_path)
    print("Summary: n_edges_raw=%s n_edges_canonical=%s n_unique_compounds=%s n_unique_genes=%s" % (
        report.get("n_edges_raw"), report.get("n_edges_canonical"),
        report.get("n_unique_compounds_canonical"), report.get("n_unique_genes")))
    if report.get("sources"):
        print("By source:", report["sources"])
    # Catastrophic failure: no inchikey resolved for any source
    if report.get("n_unique_compounds_canonical", 0) == 0 and report.get("n_edges_raw", 0) > 0:
        logger.error("Canonicalization failed: no inchikey resolved for any source")
        return 1
    if report.get("n_edges_canonical", 0) == 0 and report.get("n_edges_raw", 0) == 0:
        logger.warning("No edges from any source (missing inputs?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
