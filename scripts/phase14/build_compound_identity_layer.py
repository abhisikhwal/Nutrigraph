"""
Phase14: Build canonical compound identity resolution layer.

Delegates to src.phase14.compound_identity.run_compound_identity_pipeline().
Use: python -m src.phase14.compound_identity --repo-root . --write-scan
  or: python scripts/phase14/build_compound_identity_layer.py [--repo-root .]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    import json
    from src.phase14.compound_identity import run_compound_identity_pipeline

    parser = argparse.ArgumentParser(description="Build Phase14 canonical compound identity layer")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--min-overlap-pct", type=float, default=20.0, help="Fail if overlap %% < this (default 20)")
    parser.add_argument("--write-scan", action="store_true", help="Write compound_scan.json under canonical/")
    parser.add_argument("--smoke", action="store_true", help="Do not exit non-zero; write diagnostics and unresolved report")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    exit_code = run_compound_identity_pipeline(
        repo_root,
        write_scan=args.write_scan,
        min_overlap_pct=args.min_overlap_pct,
        smoke=args.smoke,
    )
    canonical_dir = repo_root / "data" / "processed" / "canonical"
    if (canonical_dir / "compound_identity_diagnostics.json").exists():
        with open(canonical_dir / "compound_identity_diagnostics.json", encoding="utf-8") as f:
            diag = json.load(f)
        print("--- Diagnostics ---")
        print("  pct ingredient compounds resolved: %.2f" % diag.get("pct_ingredient_compounds_resolved", 0) + "%")
        print("  pct compound_gene rows resolved: %.2f" % diag.get("pct_compound_gene_rows_resolved", 0) + "%")
        print("  pct final overlap: %.2f" % diag.get("pct_final_overlap", 0) + "% (%s / %s)" % (diag.get("n_overlap", 0), diag.get("n_ing_cmp_compounds", 0)))
        if exit_code != 0:
            print("ERROR: Overlap < " + str(args.min_overlap_pct) + "%. No fallback. Fix identity resolution or data sources.")
        else:
            print("OK: Canonical mediation identity layer built; overlap >= 20%.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
