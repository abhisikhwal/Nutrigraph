"""
CLI: Phase14 readiness check. Prints dashboard and RERUN_PHASE14: YES/NO.
Optionally writes reports/readiness_check.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run from repo root; ensure src is on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.phase14.readiness import (
    load_latest_run_dir,
    load_run_fingerprints,
    compute_current_input_fingerprints,
    should_rerun_phase14,
    print_readiness_dashboard,
    run_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase14 readiness: should we rerun?")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root")
    parser.add_argument("--write", action="store_true", help="Write reports/readiness_check.json")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    base_dir = repo_root / "data" / "processed" / "phase14_mediation"

    rerun, reasons, payload = run_readiness(repo_root=repo_root, base_dir=base_dir)
    last_run_dir = load_latest_run_dir(base_dir=base_dir, repo_root=repo_root)
    last_inputs = payload.get("last_run_input_fingerprints") or {}
    last_metrics = payload.get("last_run_metrics") or {}

    print_readiness_dashboard(
        current_inputs=payload["current_input_fingerprints"],
        last_run_dir=last_run_dir,
        last_run_inputs=last_inputs,
        last_metrics=last_metrics,
        rerun=rerun,
        reasons=reasons,
    )

    if args.write:
        out_path = repo_root / "reports" / "readiness_check.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
