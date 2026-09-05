"""
Phase16: Build expanded compound→gene canonical dataset.

Runs the full expansion pipeline (food_compound_gene_links + BindingDB resolved to InChIKey),
writes data/processed/canonical/compound_gene_expanded_raw.csv,
compound_gene_expanded_canonical.csv, and compound_gene_expansion_report.json.

No external API calls; repo data only. Deterministic.

Usage (from repo root):
  python scripts/phase16/build_compound_gene_expanded.py
  python scripts/phase16/build_compound_gene_expanded.py --repo-root .
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
    from src.phase16.compound_gene_expansion import run_expansion_pipeline

    parser = argparse.ArgumentParser(description="Build compound_gene_expanded canonical (Phase16)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "data").exists():
        logger.error("Repo root has no data/ dir: %s", repo_root)
        return 1
    logger.info("Phase16 compound-gene expansion: repo_root=%s", repo_root)
    raw_df, canonical_df, report = run_expansion_pipeline(repo_root)
    logger.info("Done. n_edges_total=%s n_compounds_total=%s", report.get("n_edges_total"), report.get("n_compounds_total"))
    if report.get("n_edges_by_source"):
        for src, count in report["n_edges_by_source"].items():
            logger.info("  %s: %s edges", src, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
