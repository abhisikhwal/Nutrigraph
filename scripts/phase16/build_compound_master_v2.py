"""
Build compound_master_v2 from local repo data only.
Writes: data/processed/canonical/compound_master_v2.parquet, .csv, compound_master_v2_diagnostics.json.
Run from repo root: python scripts/phase16/build_compound_master_v2.py [--repo-root .]
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
    from src.phase16.compound_master_v2 import write_compound_master_v2

    parser = argparse.ArgumentParser(description="Build compound_master_v2 (canonical compound table)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "data").exists():
        logger.error("Repo root has no data/: %s", repo_root)
        return 1
    pq_path, csv_path, diag_path = write_compound_master_v2(repo_root)
    print("Wrote:", pq_path, csv_path, diag_path)
    import json
    with open(diag_path) as f:
        d = json.load(f)
    print("Diagnostics: n_rows=%s, n_with_inchikey=%s, n_with_pubchem_cid=%s" % (d.get("n_rows"), d.get("n_with_inchikey"), d.get("n_with_pubchem_cid")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
