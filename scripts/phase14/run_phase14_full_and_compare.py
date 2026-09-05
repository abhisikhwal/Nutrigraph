"""
Run Phase14 bridge pipeline (build compound_gene expanded + full Phase14), then compare metrics to baseline.
Writes data/processed/canonical/reports/phase14_bridge_upgrade_delta.json.
Run from repo root: python scripts/phase14/run_phase14_full_and_compare.py [--repo-root .]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Baseline (from last run before bridge fixes)
DEFAULT_BASELINE = {
    "pct_rows_with_nonzero_propagation": 15.77,
    "n_nonzero": 162,
    "n_rows": 1027,
    "n_unique_compounds_cg": 635,
    "n_overlap_compounds": 32,
    "overlap_vs_cg": 0.0504,
}


def _load_baseline(repo_root: Path) -> dict:
    path = repo_root / "data" / "processed" / "canonical" / "reports" / "phase14_compound_gene_metrics.json"
    baseline = dict(DEFAULT_BASELINE)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if "n_cg_compounds" in data:
                baseline["n_unique_compounds_cg"] = data["n_cg_compounds"]
            if "n_overlap" in data:
                baseline["n_overlap_compounds"] = data["n_overlap"]
            if "overlap_vs_cg" in data:
                baseline["overlap_vs_cg"] = data["overlap_vs_cg"]
        except Exception as e:
            logger.warning("Could not load baseline from %s: %s. Using defaults.", path, e)
    return baseline


def _latest_phase14_run_dir(repo_root: Path) -> Path | None:
    base = repo_root / "data" / "processed" / "phase14_mediation"
    if not base.exists():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("phase14_")]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _collect_metrics_from_run(run_dir: Path) -> dict:
    reports = run_dir / "reports"
    out = {}
    prop_path = reports / "propagation_diagnostics.json"
    if prop_path.exists():
        try:
            with open(prop_path, encoding="utf-8") as f:
                out.update(json.load(f))
        except Exception as e:
            logger.warning("Could not read %s: %s", prop_path, e)
    audit_path = reports / "phase14_coverage_audit.json"
    if audit_path.exists():
        try:
            with open(audit_path, encoding="utf-8") as f:
                audit = json.load(f)
            out["n_unique_compounds_cg"] = audit.get("n_unique_compounds_cg")
            out["n_overlap_compounds"] = audit.get("n_overlap_compounds")
            out["overlap_vs_cg"] = audit.get("overlap_vs_cg")
        except Exception as e:
            logger.warning("Could not read %s: %s", audit_path, e)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase14 full and compare to baseline")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root")
    parser.add_argument("--skip-build", action="store_true", help="Skip build_compound_gene_expanded.py")
    parser.add_argument("--skip-phase14", action="store_true", help="Skip run_phase14.py (only compare latest run)")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    baseline = _load_baseline(repo_root)
    logger.info("Baseline: pct_rows_with_nonzero_propagation=%.2f%%, n_nonzero=%s, n_unique_compounds_cg=%s, n_overlap_compounds=%s, overlap_vs_cg=%s",
                baseline.get("pct_rows_with_nonzero_propagation"),
                baseline.get("n_nonzero"),
                baseline.get("n_unique_compounds_cg"),
                baseline.get("n_overlap_compounds"),
                baseline.get("overlap_vs_cg"))

    if not args.skip_build:
        logger.info("Running build_compound_gene_expanded.py ...")
        r = subprocess.run(
            [sys.executable, "scripts/phase14/build_compound_gene_expanded.py", "--repo-root", "."],
            cwd=repo_root,
            capture_output=False,
        )
        if r.returncode != 0:
            logger.error("build_compound_gene_expanded.py failed with exit code %s", r.returncode)
            return r.returncode

    if not args.skip_phase14:
        logger.info("Running run_phase14.py (full) ...")
        r = subprocess.run(
            [sys.executable, "scripts/phase14/run_phase14.py", "--repo-root", str(repo_root)],
            cwd=repo_root,
            capture_output=False,
        )
        if r.returncode != 0:
            logger.warning("run_phase14.py exited with code %s (e.g. propagation gate). Metrics below from last written reports.", r.returncode)

    run_dir = _latest_phase14_run_dir(repo_root)
    if not run_dir:
        logger.error("No Phase14 run dir found under data/processed/phase14_mediation/")
        return 1
    logger.info("Using run dir: %s", run_dir)

    new_metrics = _collect_metrics_from_run(run_dir)
    if not new_metrics:
        logger.error("No metrics found in %s/reports/", run_dir)
        return 1

    # Delta
    delta = {}
    for key in ["pct_rows_with_nonzero_propagation", "n_nonzero", "n_unique_compounds_cg", "n_overlap_compounds", "overlap_vs_cg"]:
        b = baseline.get(key)
        n = new_metrics.get(key)
        if b is not None and n is not None:
            if isinstance(b, (int, float)) and isinstance(n, (int, float)):
                diff = float(n) - float(b)
                delta[key] = int(round(diff)) if key in ("n_nonzero", "n_overlap_compounds", "n_unique_compounds_cg") else round(diff, 4)
            else:
                delta[key] = n

    # Print delta table
    print("\n--- Phase14 bridge upgrade delta ---")
    print(f"{'Metric':<40} {'Baseline':<15} {'New':<15} {'Delta':<12}")
    print("-" * 82)
    for key in ["n_unique_compounds_cg", "n_overlap_compounds", "overlap_vs_cg", "pct_rows_with_nonzero_propagation", "n_nonzero"]:
        b = baseline.get(key)
        n = new_metrics.get(key)
        d = delta.get(key)
        if b is not None or n is not None:
            print(f"{key:<40} {str(b):<15} {str(n):<15} {str(d):<12}")
    print("-" * 82)

    out = {
        "baseline": baseline,
        "new_metrics": {k: v for k, v in new_metrics.items() if k in list(baseline.keys()) + ["n_rows"]},
        "delta": delta,
        "run_dir": str(run_dir),
    }
    out_path = repo_root / "data" / "processed" / "canonical" / "reports" / "phase14_bridge_upgrade_delta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_path)

    # Acceptance reminder
    pct_new = new_metrics.get("pct_rows_with_nonzero_propagation")
    if pct_new is not None:
        if pct_new >= 25.0:
            logger.info("Acceptance: pct_rows_with_nonzero_propagation >= 25%% met (%.2f%%)", pct_new)
        else:
            logger.warning("Acceptance: pct_rows_with_nonzero_propagation = %.2f%% (target >= 25%%). Use diagnostics to add mapping sources.", pct_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
