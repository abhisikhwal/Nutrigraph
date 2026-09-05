"""
Smoke test for Phase14 propagation alignment.
Asserts: resolved_category_has_pathways >= 90% for sampled atlas rows;
         non-zero propagation exists for multiple categories.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS_CSV = REPO_ROOT / "data" / "processed" / "canonical" / "reports" / "propagation_zero_row_diagnosis.csv"
SAMPLE_SIZE = 20
MIN_PCT_RESOLVED_BUNDLE = 90.0
MIN_CATEGORIES_WITH_NONZERO = 2


def _load_diagnosis_df():
    """Load diagnosis CSV; if missing, run diagnostic script once then load."""
    if DIAGNOSIS_CSV.exists():
        return pd.read_csv(DIAGNOSIS_CSV)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.phase14.diagnose_propagation_zero_rows import main
    exit_code = main()
    if exit_code != 0:
        pytest.skip("Diagnostic script failed; run manually: python scripts/phase14/diagnose_propagation_zero_rows.py")
    if not DIAGNOSIS_CSV.exists():
        pytest.skip("Diagnosis CSV not produced")
    return pd.read_csv(DIAGNOSIS_CSV)


def test_propagation_alignment_resolved_bundle():
    """At least 90% of (up to 20) atlas rows have resolved_category_has_pathways True."""
    df = _load_diagnosis_df()
    if df.empty:
        pytest.skip("No diagnosis rows")
    sample = df.head(SAMPLE_SIZE) if len(df) >= SAMPLE_SIZE else df
    if "resolved_category_has_pathways" not in sample.columns:
        pytest.skip("Diagnosis CSV missing resolved_category_has_pathways column; re-run diagnostic script")
    n = len(sample)
    n_ok = sample["resolved_category_has_pathways"].astype(bool).sum()
    pct = 100.0 * n_ok / n if n else 0
    assert pct >= MIN_PCT_RESOLVED_BUNDLE, (
        f"resolved_category_has_pathways true for {n_ok}/{n} rows ({pct:.1f}%); need >= {MIN_PCT_RESOLVED_BUNDLE}%"
    )


def test_propagation_alignment_multiple_categories_nonzero():
    """Non-zero propagation exists for at least two categories (not only e.g. apoptosis)."""
    df = _load_diagnosis_df()
    if df.empty:
        pytest.skip("No diagnosis rows")
    if "final_propagated_score" not in df.columns or "category" not in df.columns:
        pytest.skip("Diagnosis CSV missing score/category columns")
    nonzero = df[df["final_propagated_score"] > 0]
    if nonzero.empty:
        pytest.fail("No rows with non-zero propagation")
    categories_with_nonzero = nonzero["category"].nunique()
    assert categories_with_nonzero >= MIN_CATEGORIES_WITH_NONZERO, (
        f"Only {categories_with_nonzero} category(ies) have non-zero propagation; need >= {MIN_CATEGORIES_WITH_NONZERO}"
    )


def test_propagation_alignment_summary():
    """Print short summary of propagation alignment (for CI logs)."""
    df = _load_diagnosis_df()
    if df.empty:
        pytest.skip("No diagnosis rows")
    n = len(df)
    pct_nonzero = 100.0 * (df.get("zero_reason", pd.Series()) == "nonzero").sum() / n if n else 0
    pct_resolved = 100.0 * df["resolved_category_has_pathways"].astype(bool).sum() / n if "resolved_category_has_pathways" in df.columns and n else 0
    n_cat_nonzero = df[df["final_propagated_score"] > 0]["category"].nunique() if "final_propagated_score" in df.columns else 0
    print(
        f"Propagation alignment: n_rows={n} pct_nonzero={pct_nonzero:.1f}% "
        f"pct_resolved_bundle={pct_resolved:.1f}% categories_with_nonzero={n_cat_nonzero}"
    )
