"""
Create a safety snapshot BEFORE bridge/resolution changes.
Folder: data/processed/canonical/reports/snapshots/phase14_bridge_YYYYMMDD_HHMMSS/
Copies: build_compound_gene_expanded.py, uniprot_gene_resolver.py, bindingdb_compound_resolver.py,
        compound_identity.py, compound_gene_expansion_report.json (if exists).
If git available: git diff > pre_patch.diff
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    now = datetime.utcnow()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    snap_dir = REPO_ROOT / "data" / "processed" / "canonical" / "reports" / "snapshots" / ("phase14_bridge_%s" % stamp)
    snap_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        (REPO_ROOT / "scripts" / "phase14" / "build_compound_gene_expanded.py", "build_compound_gene_expanded.py"),
        (REPO_ROOT / "src" / "phase14" / "uniprot_gene_resolver.py", "uniprot_gene_resolver.py"),
        (REPO_ROOT / "src" / "phase14" / "bindingdb_compound_resolver.py", "bindingdb_compound_resolver.py"),
        (REPO_ROOT / "src" / "phase14" / "compound_identity.py", "compound_identity.py"),
    ]
    for src, name in files_to_copy:
        if src.exists():
            shutil.copy2(src, snap_dir / name)
            print("Copied", name)
        else:
            print("Skip (missing):", src)

    report_src = REPO_ROOT / "data" / "processed" / "canonical" / "reports" / "compound_gene_expansion_report.json"
    if report_src.exists():
        shutil.copy2(report_src, snap_dir / "compound_gene_expansion_report.json")
        print("Copied compound_gene_expansion_report.json")

    try:
        r = subprocess.run(
            ["git", "diff", "--no-color"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode == 0 and (r.stdout or r.stderr):
            patch_path = snap_dir / "pre_patch.diff"
            with open(patch_path, "w", encoding="utf-8") as f:
                f.write(r.stdout or "")
            print("Wrote pre_patch.diff")
        elif r.returncode != 0:
            print("Git not available or not a repo; skipped pre_patch.diff")
    except Exception as e:
        print("Git diff failed:", e)

    print("Snapshot dir:", snap_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
