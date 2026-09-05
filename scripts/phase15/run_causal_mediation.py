"""
Phase15: Run causal mediation (SEM-style). Exposure -> Mediator -> Outcome.
Outputs causal_mediation_results.csv, causal_mediation_summary.json, summary markdown.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.phase15.causal_mediation import (
    load_pair_mediation,
    run_mediation_per_row,
    run_mediation_by_category,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase15 causal mediation")
    parser.add_argument("--phase14-dir", type=str, default=None, help="Phase14 run/snapshot; default snapshot")
    parser.add_argument("--phase15-dir", type=str, default=None, help="Phase15 out dir; required for writing")
    parser.add_argument("--bootstrap-n", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()

    if args.phase14_dir is None:
        snapshot = repo_root / "data" / "processed" / "milestones" / "phase14" / "v1_working_2026-02-19" / "phase14_20260219_204918"
        phase14_dir = snapshot if snapshot.exists() else repo_root / "data" / "processed" / "phase14_mediation" / "phase14_20260219_204918"
    else:
        phase14_dir = (repo_root / args.phase14_dir.replace("\\", "/")).resolve()
    if not phase14_dir.exists():
        logger.error("Phase14 dir not found: %s", phase14_dir)
        return 1

    if args.phase15_dir is None:
        logger.error("--phase15-dir required for writing outputs.")
        return 1
    phase15_dir = Path(args.phase15_dir).resolve()
    causal_dir = phase15_dir / "causal"
    reports_dir = phase15_dir / "reports"
    causal_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_pair_mediation(phase14_dir)
    if df.empty:
        logger.warning("No pair_category_mediation found")
        summary_json = {"status": "no_data", "n_obs": 0}
        with open(reports_dir / "causal_mediation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2)
        # Always write results CSV (empty placeholder) so downstream expects are met
        pd.DataFrame(columns=["category", "indirect_mean", "indirect_ci_low", "indirect_ci_high", "indirect_p"]).to_csv(
            causal_dir / "causal_mediation_results.csv", index=False
        )
        logger.info("Wrote causal/causal_mediation_results.csv (empty) and reports/causal_mediation_summary.json")
        return 0

    global_row = run_mediation_per_row(
        df,
        exposure_col="shared_compounds_count",
        mediator_col="propagated_pathway_score",
        outcome_col="did",
        bootstrap_n=args.bootstrap_n,
        alpha=args.alpha,
        seed=args.seed,
    )
    by_cat = run_mediation_by_category(
        df,
        exposure_col="shared_compounds_count",
        mediator_col="propagated_pathway_score",
        outcome_col="did",
        bootstrap_n=args.bootstrap_n,
        alpha=args.alpha,
        seed=args.seed,
    )
    results = by_cat if not by_cat.empty else global_row
    results_path = causal_dir / "causal_mediation_results.csv"
    results.to_csv(results_path, index=False)
    logger.info("Wrote %s", results_path)

    summary_json = {
        "phase14_dir": str(phase14_dir),
        "n_rows": int(len(df)),
        "n_categories": int(by_cat["category"].nunique()) if not by_cat.empty else 0,
        "global_indirect_mean": float(global_row["indirect_mean"].iloc[0]) if not global_row.empty else None,
        "global_indirect_p": float(global_row["indirect_p"].iloc[0]) if not global_row.empty else None,
        "warnings": ["Hypothesis generation only. Not medical advice. Assumptions (e.g. no unmeasured confounding) may be violated; observations are not independent samples."],
    }
    with open(reports_dir / "causal_mediation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    summary_md = [
        "# Causal mediation summary",
        "",
        "## Specification",
        "- Exposure: shared_compounds_count (or dose_proxy_AB)",
        "- Mediator: propagated_pathway_score",
        "- Outcome: did (effect size)",
        "",
        "## Limitations",
        "- Observations (pair-category rows) are not independent; use for hypothesis generation only.",
        "- No medical or clinical claims. Not medical advice.",
        "",
        "## Global",
        f"- Indirect effect (mean): {summary_json.get('global_indirect_mean', 'N/A')}",
        f"- Indirect p-value: {summary_json.get('global_indirect_p', 'N/A')}",
    ]
    with open(reports_dir / "causal_mediation_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md))
    logger.info("Causal mediation done. Results in %s", causal_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
