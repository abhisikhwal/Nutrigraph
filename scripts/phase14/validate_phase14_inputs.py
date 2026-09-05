"""
Phase14 input validation CLI. Uses src.phase14.validation for logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Repo root when run as script
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DEFAULT))

from src.phase14.validation import run_validation, print_validation_table


def _serialize(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase14 inputs and resolve paths")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT, help="Repo root")
    parser.add_argument("--phase13-dir", type=str, default="data/processed/phase13_interactions_v3_20260206_162122_b_gpu_stable", help="Phase13 dir (relative to repo)")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any required file missing")
    parser.add_argument("--json", type=str, default="", metavar="PATH", help="Write report JSON (e.g. data/processed/phase14_mediation/_diagnostics/validate.json)")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    phase13_dir = Path(args.phase13_dir)

    report, selected, errors = run_validation(repo_root, phase13_dir)

    print_validation_table(report, selected)
    print()
    print("Resolved selected paths (relative to repo_root):")
    for k, v in sorted(selected.items()):
        print(f"  {k}: {v}")

    if args.json:
        out_path = Path(args.json)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_serialize(report), f, indent=2)
        print(f"\nWrote report: {out_path}")

    if args.strict and errors:
        print("\nStrict mode: required inputs missing. Exiting with error.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
