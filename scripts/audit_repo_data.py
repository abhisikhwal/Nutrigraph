"""
Audit repo data: scan data/raw and data/processed, write repo_manifest.json.
Run from repo root: python scripts/audit_repo_data.py [--repo-root .]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    from src.utils.data_inventory import build_repo_manifest, write_manifest

    parser = argparse.ArgumentParser(description="Audit repo data and write repo_manifest.json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "data").exists():
        logger.error("Repo root has no data/ dir: %s", repo_root)
        return 1
    out = write_manifest(repo_root)
    print("Wrote:", out)
    manifest = build_repo_manifest(repo_root)
    print("Total files:", manifest["n_files"])
    print("Missing candidates:", manifest["MISSING_CANDIDATES"])
    print("By schema:", list(manifest["by_detected_schema"].keys()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
