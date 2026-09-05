"""
Phase 13 v3 — Confirmatory regression audit (OLS with controls for top hits).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .phase13_utils import _ts

try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hc3
except ImportError:
    sm = None


def run_regression_audit(
    top_interactions: pd.DataFrame,
    recipe_ingredients: pd.DataFrame,
    signatures: pd.DataFrame,
    confounders: pd.DataFrame,
    y_cols: List[str],
    top_k: int = 300,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    For top_k interactions (by |delta| or q), run OLS: y ~ A + B + A*B + num_ingredients + num_compounds + cuisine_dummies.
    Returns (audit_df, summary_dict).
    """
    if sm is None:
        return pd.DataFrame(), {"error": "statsmodels not available", "timestamp": _ts()}

    # Dedupe by (ingA, ingB); take top by abs(delta) or q
    if "delta" in top_interactions.columns:
        top = top_interactions.sort_values("delta", key=abs, ascending=False).drop_duplicates(subset=["ingA_id", "ingB_id"]).head(top_k)
    else:
        top = top_interactions.drop_duplicates(subset=["ingA_id", "ingB_id"]).head(top_k)

    ri = recipe_ingredients[["recipe_id", "ingredient_id"]].drop_duplicates()
    grouped = ri.groupby("recipe_id")["ingredient_id"].apply(set).to_dict()
    sig_idx = signatures.set_index("recipe_id")
    conf_idx = confounders.set_index("recipe_id")

    rows = []
    for _, row in top.iterrows():
        ing_a, ing_b = row["ingA_id"], row["ingB_id"]
        cat = row.get("category")
        if pd.isna(cat) or cat not in y_cols:
            continue
        set_a = set()
        set_b = set()
        for rid, ings in grouped.items():
            if ing_a in ings:
                set_a.add(rid)
            if ing_b in ings:
                set_b.add(rid)
        relevant = sorted(set_a | set_b)
        if len(relevant) < 30:
            continue
        y = sig_idx.loc[relevant, cat].values.astype(float)
        A = np.array([1 if r in set_a else 0 for r in relevant], dtype=float)
        B = np.array([1 if r in set_b else 0 for r in relevant], dtype=float)
        X = pd.DataFrame({"A": A, "B": B, "A_B": A * B}, index=relevant)
        conf = conf_idx.loc[relevant]
        if "num_ingredients" in conf.columns:
            X["num_ingredients"] = conf["num_ingredients"].values
        if "num_compounds" in conf.columns and conf["num_compounds"].notna().any():
            X["num_compounds"] = conf["num_compounds"].fillna(0).values
        if "cuisine_or_cluster" in conf.columns:
            dums = pd.get_dummies(conf["cuisine_or_cluster"], drop_first=True, dtype=float)
            X = pd.concat([X, dums], axis=1)
        X = sm.add_constant(X, has_constant="add")
        try:
            res = sm.OLS(y, X).fit(cov_type="HC3")
        except Exception:
            try:
                res = sm.OLS(y, X).fit()
            except Exception:
                continue
        if "A_B" not in res.params.index:
            continue
        beta_int = res.params["A_B"]
        se_ols = res.bse["A_B"]
        p_ols = res.pvalues["A_B"]
        delta_strat = row.get("delta", np.nan)
        sign_match = np.sign(beta_int) == np.sign(delta_strat) if not np.isnan(delta_strat) else None
        rows.append({
            "ingA_id": ing_a,
            "ingB_id": ing_b,
            "category": cat,
            "beta_int_ols": float(beta_int),
            "se_ols": float(se_ols),
            "p_ols": float(p_ols),
            "delta_stratified": float(delta_strat),
            "sign_match": sign_match,
            "n_obs": len(relevant),
        })

    audit_df = pd.DataFrame(rows)
    if audit_df.empty:
        summary = {"n_audited": 0, "timestamp": _ts()}
    else:
        sign_match = audit_df["sign_match"].dropna()
        summary = {
            "n_audited": len(audit_df),
            "sign_match_frac": float(sign_match.mean()) if len(sign_match) > 0 else None,
            "delta_beta_correlation": float(audit_df["delta_stratified"].corr(audit_df["beta_int_ols"])) if len(audit_df) > 1 else None,
            "timestamp": _ts(),
        }
    return audit_df, summary


def write_regression_audit(
    output_dir: Path,
    audit_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:
    audit_df.to_csv(output_dir / "regression_audit.csv", index=False)
    with open(output_dir / "regression_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
