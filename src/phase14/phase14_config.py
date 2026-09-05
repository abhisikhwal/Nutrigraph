"""
Phase14: Biological Mediation Layer — config, paths, defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# Default run output base
PHASE14_OUT_BASE = "data/processed/phase14_mediation"

# Phase13 categories (from Phase13B / pathway_bundles)
DEFAULT_CATEGORIES = [
    "apoptosis",
    "cardiovascular",
    "cell_cycle",
    "dna_repair",
    "hormone",
    "immune",
    "metabolism",
    "nervous",
    "other",
    "signaling",
    "transport",
    "cell_signaling",
    "oxidative_stress",
]

# Map atlas category (normalized: lower, spaces->underscores) to pathway_bundles key for propagation.
# Used when atlas category has no pathways in the graph so propagation can use a related bundle.
# Aligns Phase13/atlas category names with pathway_bundles.json keys and avoids missing_bundle zeros.
CATEGORY_SYNONYMS_FOR_PROPAGATION: Dict[str, str] = {
    "signaling": "cell_signaling",
    "nervous": "neurotransmission",
    "immune": "inflammation",
    "cell_cycle": "cell_signaling",
    "dna_repair": "cell_signaling",
    "hormone": "metabolism",
    "transport": "metabolism",
    "other": "metabolism",
    "translation": "transcription",
}

# Mechanistic score weights (sigmoid inputs)
MECH_A1 = 0.4  # log1p(shared_genes)
MECH_A2 = 0.35  # log1p(shared_compounds)
MECH_A3 = 0.25  # log1p(shared_pathways)

# RWR / propagation
RWR_RESTART_PROB = 0.2
RWR_MAX_ITER = 50
RWR_TOL = 1e-6

# Consistency
LAPLACIAN_ITERATIONS = 2
INCOHERENT_DID_SIGN_THRESHOLD = 0.3

# Compound-gene overlap gate: when using compound_gene_expanded_canonical.csv, require at least this overlap_vs_ic (default 1%)
COMPOUND_GENE_REQUIRE_OVERLAP_PCT = 0.01

# Readiness gate for FULL run (after BindingDB expansion overlap_vs_cg is no longer a hard gate):
# - atlas_pair_cov_compound >= ATLAS_PAIR_COV_COMPOUND_MIN
# - pct_rows_with_nonzero_propagation >= PCT_ROWS_NONZERO_PROPAGATION_MIN
# - (ic_gene_coverage >= IC_GENE_COVERAGE_MIN) OR (n_overlap_compounds >= N_OVERLAP_COMPOUNDS_MIN)
# overlap_vs_cg is warning-only.
ATLAS_PAIR_COV_COMPOUND_MIN = 0.30
PCT_ROWS_NONZERO_PROPAGATION_MIN = 25.0
IC_GENE_COVERAGE_MIN = 0.005  # 0.5% of IC compounds appear in compound_gene
N_OVERLAP_COMPOUNDS_MIN = 200
# Mechanistic score saturation warning threshold (pct of rows > 0.95)
MECHANISTIC_SCORE_SATURATION_WARN_PCT = 30.0

# Triplet / beyond-pairwise (Phase15-style, bounded for production)
TRIPLET_MIN_SUPPORT = 50
TRIPLET_MAX_COUNT = 500
TRIPLET_MIN_SUPPORT_FULL = 50
TRIPLET_MIN_SUPPORT_SMOKE = 10
TRIPLET_DROP_TOP_FREQ_PCT = 1.0  # drop ingredients in top 1% by recipe count (avoid staple dominance)
TRIPLET_MAX_TRIPLETS = 50000  # cap for runtime (legacy name; see below for bounded pipeline)
# Bounded pipeline (production on large recipe_ingredients)
TRIPLET_MAX_RECIPES = 200_000
TRIPLET_MAX_UNIQUE_INGREDIENTS = 3000
TRIPLET_MIN_SUPPORT_PAIRS = 50
TRIPLET_MIN_SUPPORT_TRIPLETS = 25
TRIPLET_TOP_K_PAIRS = 20_000
TRIPLET_MAX_TRIPLETS_SCORED = 50_000
TRIPLET_TIME_BUDGET_SECONDS = 600
TRIPLET_RANDOM_SEED = 42

# Glob patterns for file discovery (relative to data/processed)
SIGNATURE_PATTERNS = [
    "phase17_reaggregation/recipe_functional_signatures_v3.parquet",
    "exports_v2/recipes_biological_effects_v2_FINAL.parquet",
    "**/recipe*functional*signature*.parquet",
    "**/recipes_biological*.parquet",
]
RI_PATTERNS = [
    "canonical/recipe_ingredients_expanded_v2.parquet",
    "**/recipe_ingredients*expanded*.parquet",
    "**/recipe_ingredients*.parquet",
]
PATHWAY_PATTERNS = [
    "features/pathway_gene_signatures.parquet",
    "features/pathway_cluster_info.csv",
    "features/pathway_bundles.json",
    "**/pathway*gene*.parquet",
]
METABOLOMICS_PATTERNS = [
    "phase11_metabolomics/*.parquet",
    "phase11_metabolomics/*.csv",
    "**/ingredient*compound*.parquet",
    "**/compound*ingredient*.parquet",
]
GENETICS_PATTERNS = [
    "phase12_genetics/*.parquet",
    "phase12_genetics/*.csv",
    "**/gene*categor*.parquet",
]
BINDING_PATTERNS = [
    "phase16_bindingdb/*.parquet",
    "phase16_bindingdb/*.csv",
    "**/binding*.parquet",
    "**/target*.parquet",
]
TARGET_CLUSTER_PATTERNS = [
    "features/target_functional_clusters.csv",
]


def get_repo_root() -> Path:
    """Infer repo root (parent of src, or cwd)."""
    try:
        import sys
        for p in sys.path:
            path = Path(p).resolve()
            if (path / "src" / "phase14").exists():
                return path
    except Exception:
        pass
    return Path(__file__).resolve().parent.parent.parent


def resolve_processed_dir(repo_root: Optional[Path] = None) -> Path:
    repo_root = repo_root or get_repo_root()
    return repo_root / "data" / "processed"
