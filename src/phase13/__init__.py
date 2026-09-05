# Phase 13: Ingredient Interaction Atlas (legacy) + Quality-first v3
from .interaction_atlas import (
    build_candidate_pairs,
    build_confounders,
    run_pair_regression,
    bh_fdr,
    bootstrap_stability,
    run_null_tests,
)
from .phase13_config import Phase13Config
from .phase13_utils import bh_fdr as bh_fdr_utils, get_memory_mb, run_manifest_dict, quantile_bins, Timer, _ts
from .phase13_stratified import compute_pair_interactions_sharded
from .phase13_holdouts import run_ingredient_holdout, run_cuisine_holdout, write_holdout_report
from .phase13_regression_audit import run_regression_audit, write_regression_audit
from .phase13_reports import (
    generate_phase13_summary,
    generate_interaction_atlas,
    deliverables_checklist,
    generate_volcano_plot,
    generate_q_histogram,
)

__all__ = [
    "build_candidate_pairs",
    "build_confounders",
    "run_pair_regression",
    "bh_fdr",
    "bootstrap_stability",
    "run_null_tests",
    "Phase13Config",
    "bh_fdr_utils",
    "get_memory_mb",
    "run_manifest_dict",
    "quantile_bins",
    "Timer",
    "_ts",
    "compute_pair_interactions_sharded",
    "run_ingredient_holdout",
    "run_cuisine_holdout",
    "write_holdout_report",
    "run_regression_audit",
    "write_regression_audit",
    "generate_phase13_summary",
    "generate_interaction_atlas",
    "deliverables_checklist",
    "generate_volcano_plot",
    "generate_q_histogram",
]
