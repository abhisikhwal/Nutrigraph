"""
Phase15: Causal mediation analysis (SEM-style).
For each atlas row (ingredient pair, category): exposure -> mediator -> outcome.
Bootstrap CI and p-values; hypothesis generation only, not medical advice.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
np.random.seed(42)


def load_pair_mediation(run_dir: Path) -> pd.DataFrame:
    """Load pair_category_mediation.csv from Phase14 run/snapshot."""
    p = Path(run_dir) / "pair_category_mediation.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _safe_numeric(s: pd.Series, default: float = 0.0) -> np.ndarray:
    out = pd.to_numeric(s, errors="coerce")
    return out.fillna(default).values


def fit_mediation_simple(
    exposure: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
) -> Dict[str, float]:
    """
    Simple two-step OLS:
      mediator ~ exposure
      outcome ~ exposure + mediator
    Returns coefficients and R2; no bootstrap here (done in runner).
    """
    from numpy.linalg import lstsq
    n = len(exposure)
    if n < 10:
        return {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "direct": 0.0, "r2_med": 0.0, "r2_out": 0.0}
    X_med = np.column_stack([np.ones(n), exposure])
    (coef_med, _, _, _) = lstsq(X_med, mediator, rcond=None)
    alpha = coef_med[1] if len(coef_med) > 1 else 0.0
    pred_med = X_med @ coef_med
    r2_med = 1 - np.sum((mediator - pred_med) ** 2) / max(1e-10, np.sum((mediator - mediator.mean()) ** 2))
    X_out = np.column_stack([np.ones(n), exposure, mediator])
    (coef_out, _, _, _) = lstsq(X_out, outcome, rcond=None)
    direct = coef_out[1] if len(coef_out) > 1 else 0.0
    beta = coef_out[2] if len(coef_out) > 2 else 0.0
    pred_out = X_out @ coef_out
    r2_out = 1 - np.sum((outcome - pred_out) ** 2) / max(1e-10, np.sum((outcome - outcome.mean()) ** 2))
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(direct),
        "direct": float(direct),
        "indirect": float(alpha * beta),
        "r2_med": float(r2_med),
        "r2_out": float(r2_out),
    }


def bootstrap_mediation(
    exposure: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap CI and p-value for indirect effect (alpha*beta)."""
    rng = np.random.RandomState(seed)
    n = len(exposure)
    if n < 10:
        return {"indirect_ci_low": 0.0, "indirect_ci_high": 0.0, "indirect_p": 1.0, "warnings": ["n too small"]}
    indirects = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        e, m, o = exposure[idx], mediator[idx], outcome[idx]
        res = fit_mediation_simple(e, m, o)
        indirects.append(res["indirect"])
    indirects = np.array(indirects)
    ci_low = np.percentile(indirects, 100 * alpha / 2)
    ci_high = np.percentile(indirects, 100 * (1 - alpha / 2))
    p = 2 * min(np.mean(indirects <= 0), np.mean(indirects >= 0))
    return {
        "indirect_ci_low": float(ci_low),
        "indirect_ci_high": float(ci_high),
        "indirect_p": float(p),
        "indirect_mean": float(np.mean(indirects)),
    }


def run_mediation_per_row(
    df: pd.DataFrame,
    exposure_col: str = "shared_compounds_count",
    mediator_col: str = "propagated_pathway_score",
    outcome_col: str = "did",
    outcome_proxy_optional: Optional[str] = "p_analytic",
    bootstrap_n: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each (ingA_id, ingB_id, category) we have one row. So we run mediation on the full table
    as one regression (all rows) or per-group. Specification: one global model using all rows.
    """
    if df.empty or exposure_col not in df.columns:
        return pd.DataFrame()
    exposure = _safe_numeric(df[exposure_col])
    mediator = _safe_numeric(df[mediator_col]) if mediator_col in df.columns else np.zeros(len(df))
    outcome = _safe_numeric(df[outcome_col]) if outcome_col in df.columns else np.zeros(len(df))
    if outcome_proxy_optional and outcome_proxy_optional in df.columns:
        op = _safe_numeric(df[outcome_proxy_optional])
        if np.any(np.isfinite(op)):
            outcome = np.where(np.isfinite(outcome) & (outcome != 0), outcome, op)
    n = len(exposure)
    warnings = []
    if n < 30:
        warnings.append("Sample size small; results are hypothesis-generating only.")
    if np.all(mediator == 0):
        warnings.append("Mediator constant zero; mediation effect undefined.")
    res = fit_mediation_simple(exposure, mediator, outcome)
    boot = bootstrap_mediation(exposure, mediator, outcome, n_bootstrap=bootstrap_n, alpha=alpha, seed=seed)
    res.update(boot)
    res["n_obs"] = n
    res["warnings"] = warnings
    row = {**res}
    row["exposure_col"] = exposure_col
    row["mediator_col"] = mediator_col
    row["outcome_col"] = outcome_col
    return pd.DataFrame([row])


def run_mediation_by_category(
    df: pd.DataFrame,
    exposure_col: str = "shared_compounds_count",
    mediator_col: str = "propagated_pathway_score",
    outcome_col: str = "did",
    bootstrap_n: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Run mediation per category; return one row per category with coefficients and bootstrap CI."""
    if df.empty or "category" not in df.columns:
        return pd.DataFrame()
    results = []
    for cat, g in df.groupby("category"):
        exp = _safe_numeric(g[exposure_col]) if exposure_col in g.columns else np.zeros(len(g))
        med = _safe_numeric(g[mediator_col]) if mediator_col in g.columns else np.zeros(len(g))
        out = _safe_numeric(g[outcome_col]) if outcome_col in g.columns else np.zeros(len(g))
        if len(g) < 5:
            results.append({"category": cat, "n_obs": len(g), "indirect_mean": np.nan, "indirect_p": np.nan})
            continue
        res = fit_mediation_simple(exp, med, out)
        boot = bootstrap_mediation(exp, med, out, n_bootstrap=bootstrap_n, alpha=alpha, seed=seed)
        results.append({
            "category": cat,
            "n_obs": len(g),
            "alpha": res["alpha"],
            "beta": res["beta"],
            "direct": res["direct"],
            "indirect_mean": boot["indirect_mean"],
            "indirect_ci_low": boot["indirect_ci_low"],
            "indirect_ci_high": boot["indirect_ci_high"],
            "indirect_p": boot["indirect_p"],
        })
    return pd.DataFrame(results)
