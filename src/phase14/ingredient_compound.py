"""
Phase14: Ingredient→Compound bridge — InChIKey-first, multi-source assembly with audit.
Stable identifiers first; name-based matching only as last resort and flagged LOW_CONF.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Default: no fuzzy name matching as main method
ALLOW_FUZZY = False

CANONICAL_COLUMNS = ["ingredient_id", "compound_id", "inchikey", "evidence_source", "evidence_strength", "match_type"]
INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def _normalize_inchikey(s: str) -> Optional[str]:
    """Uppercase, strip; return if valid 27-char InChIKey else None."""
    if not s or pd.isna(s):
        return None
    s = str(s).strip().upper()
    if len(s) == 27 and "-" in s and INCHIKEY_PATTERN.match(s):
        return s
    # Allow without dashes: 14+10+1
    s = s.replace("-", "")
    if len(s) == 25:
        return f"{s[:14]}-{s[14:24]}-{s[24]}"
    return None


def _normalize_ingredient_id(x: Any) -> str:
    s = str(x).strip()
    if s.upper().startswith("ING_"):
        return s
    return f"ING_{s}" if s else ""


def _normalize_compound_id(x: Any) -> str:
    return str(x).strip() if x is not None and not pd.isna(x) else ""


def load_ing_compound_edges(root: Path, scan_df: pd.DataFrame, chosen: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Assemble Ingredient→Compound edges from audit-chosen sources.
    Returns DataFrame with: ingredient_id, compound_id, inchikey (optional),
    evidence_source, evidence_strength, match_type.
    Priority: A) direct INGREDIENT_COMPOUND, B) InChIKey two-step join, C) Food→Compound only with explicit bridge.
    """
    root = Path(root).resolve()
    derived_dir = root / "data" / "processed" / "phase14_mediation" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    cache_pq = derived_dir / "ingredient_compound_links.parquet"
    cache_csv = derived_dir / "ingredient_compound_links.csv"

    # Prefer existing derived cache if it has evidence columns (from a previous run)
    if cache_pq.exists():
        try:
            df = pd.read_parquet(cache_pq)
            if not df.empty and "ingredient_id" in df.columns and "compound_id" in df.columns:
                logger.info("Using cached ingredient_compound_links from %s (%s rows)", cache_pq, len(df))
                return _ensure_canonical_columns(df)
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
    if cache_csv.exists():
        try:
            df = pd.read_csv(cache_csv)
            if not df.empty and "ingredient_id" in df.columns and "compound_id" in df.columns:
                logger.info("Using cached ingredient_compound_links (CSV) from %s (%s rows)", cache_csv, len(df))
                return _ensure_canonical_columns(df)
        except Exception as e:
            logger.warning("Cache CSV load failed: %s", e)

    chosen = chosen or {}
    out: List[Dict[str, Any]] = []

    # A) Direct INGREDIENT_COMPOUND file
    direct_path = chosen.get("ingredient_compound")
    if direct_path:
        path = Path(direct_path) if not isinstance(direct_path, Path) else direct_path
        if path.exists():
            try:
                df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)
                ing_col = next((c for c in df.columns if "ingredient" in c.lower() and "id" in c.lower()), None)
                if not ing_col:
                    ing_col = next((c for c in df.columns if c.lower() == "ing_id"), None)
                cmp_col = next((c for c in df.columns if "compound" in c.lower() and "id" in c.lower()), None)
                if not cmp_col:
                    cmp_col = next((c for c in df.columns if c.lower() in ("inchikey", "inchi_key")), None)
                if ing_col and cmp_col:
                    for _, r in df.iterrows():
                        ing_id = _normalize_ingredient_id(r.get(ing_col))
                        cmp_id = _normalize_compound_id(r.get(cmp_col))
                        if ing_id and cmp_id:
                            out.append({
                                "ingredient_id": ing_id,
                                "compound_id": cmp_id,
                                "inchikey": _normalize_inchikey(r.get("inchikey") or r.get("inchi_key") or ""),
                                "evidence_source": path.name,
                                "evidence_strength": "HIGH",
                                "match_type": "direct_id",
                            })
                    if out:
                        result = _dedupe_edges(pd.DataFrame(out))
                        _write_cache(result, derived_dir)
                        return result
            except Exception as e:
                logger.warning("Direct INGREDIENT_COMPOUND load failed %s: %s", path, e)

    # B) Two-step InChIKey: ingredient_inchikey + inchikey_compound
    ing_ik_path = chosen.get("ingredient_inchikey")
    ik_cmp_path = chosen.get("inchikey_compound")
    if ing_ik_path and ik_cmp_path:
        p1 = Path(ing_ik_path)
        p2 = Path(ik_cmp_path)
        if p1.exists() and p2.exists():
            try:
                df_ing = pd.read_parquet(p1) if p1.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p1)
                df_cmp = pd.read_parquet(p2) if p2.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p2)
                ing_col = next((c for c in df_ing.columns if "ingredient" in c.lower() and "id" in c.lower()), None) or "ingredient_id"
                ik_col_ing = next((c for c in df_ing.columns if "inchikey" in c.lower() or "inchi_key" in c.lower()), None)
                ik_col_cmp = next((c for c in df_cmp.columns if "inchikey" in c.lower() or "inchi_key" in c.lower()), None)
                cmp_col = next((c for c in df_cmp.columns if "compound" in c.lower() and "id" in c.lower()), None) or "compound_id"
                if ik_col_ing and ik_col_cmp:
                    df_ing["_ik"] = df_ing[ik_col_ing].astype(str).apply(lambda x: _normalize_inchikey(x))
                    df_cmp["_ik"] = df_cmp[ik_col_cmp].astype(str).apply(lambda x: _normalize_inchikey(x))
                    df_ing = df_ing[df_ing["_ik"].notna()]
                    df_cmp = df_cmp[df_cmp["_ik"].notna()]
                    merged = df_ing.merge(df_cmp, on="_ik", how="inner")
                    for _, r in merged.iterrows():
                        ing_id = _normalize_ingredient_id(r.get(ing_col))
                        cmp_id = _normalize_compound_id(r.get(cmp_col))
                        if ing_id and cmp_id:
                            out.append({
                                "ingredient_id": ing_id,
                                "compound_id": cmp_id,
                                "inchikey": r["_ik"],
                                "evidence_source": f"inchikey_join:{p1.name}+{p2.name}",
                                "evidence_strength": "MED",
                                "match_type": "inchikey_join",
                            })
                    if out:
                        result = _dedupe_edges(pd.DataFrame(out))
                        _write_cache(result, derived_dir)
                        return result
            except Exception as e:
                logger.warning("InChIKey two-step join failed: %s", e)

    # C) Food→Compound: only if explicit ingredient↔food bridge exists (no fuzzy by default)
    bridge_path = chosen.get("ingredient_food_bridge")
    food_cmp_path = chosen.get("food_compound")
    if bridge_path and food_cmp_path:
        # Load bridge (ingredient_id <-> food_id or ingredient_name <-> food_name) and food_compound
        pass  # Implement when bridge file schema is defined; skip fuzzy

    empty_out = pd.DataFrame(columns=CANONICAL_COLUMNS)
    logger.info("No ingredient→compound edges assembled from audit sources.")
    return empty_out


def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[[c for c in CANONICAL_COLUMNS if c in df.columns]].drop_duplicates(subset=["ingredient_id", "compound_id"])


def _dedupe_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Keep first occurrence; prefer HIGH > MED > LOW strength
    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    df = df.copy()
    df["_order"] = df.get("evidence_strength", "LOW").map(lambda x: order.get(x, 2))
    df = df.sort_values("_order").drop_duplicates(subset=["ingredient_id", "compound_id"], keep="first")
    df = df.drop(columns=["_order"], errors="ignore")
    return _ensure_canonical_columns(df)


def _write_cache(df: pd.DataFrame, derived_dir: Path) -> None:
    if df.empty:
        return
    derived_dir = Path(derived_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(derived_dir / "ingredient_compound_links.parquet", index=False)
        logger.info("Wrote %s", derived_dir / "ingredient_compound_links.parquet")
    except Exception as e:
        logger.warning("Could not write parquet cache: %s", e)
    try:
        df.to_csv(derived_dir / "ingredient_compound_links.csv", index=False)
        logger.info("Wrote %s", derived_dir / "ingredient_compound_links.csv")
    except Exception as e:
        logger.warning("Could not write CSV cache: %s", e)


def validate_ing_compound_edges(
    df: pd.DataFrame,
    recipe_ingredients: pd.DataFrame,
    output_dir: Optional[Path] = None,
    phase14_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Report: #unique ingredients covered, % of recipe_ingredients covered, total edges, distribution.
    If coverage < 25%, log WARNING and set coverage_fail in phase14_summary (and write to phase14_summary.json if output_dir given).
    """
    report: Dict[str, Any] = {
        "n_edges": 0,
        "n_unique_ingredients_covered": 0,
        "pct_recipe_ingredients_covered": 0.0,
        "edges_per_ingredient_dist": [],
        "coverage_fail": False,
    }
    if df.empty or "ingredient_id" not in df.columns:
        report["coverage_fail"] = True
        logger.warning("Ingredient→compound edges empty or missing ingredient_id; coverage_fail=True")
        if output_dir and phase14_summary is not None:
            phase14_summary["ingredient_compound_coverage_fail"] = True
            _write_phase14_summary(output_dir, phase14_summary)
        return report

    n_edges = len(df)
    ings_covered = df["ingredient_id"].nunique()
    report["n_edges"] = n_edges
    report["n_unique_ingredients_covered"] = int(ings_covered)

    if not recipe_ingredients.empty and "ingredient_id" in recipe_ingredients.columns:
        total_ings = recipe_ingredients["ingredient_id"].nunique()
        if total_ings > 0:
            pct = 100.0 * ings_covered / total_ings
            report["pct_recipe_ingredients_covered"] = round(pct, 2)
            if pct < 25:
                report["coverage_fail"] = True
                logger.warning(
                    "Ingredient→compound coverage %.1f%% < 25%%; mediation strength will be weak; coverage_fail=True",
                    pct,
                )
                if phase14_summary is not None:
                    phase14_summary["ingredient_compound_coverage_fail"] = True
                    phase14_summary["ingredient_compound_pct_covered"] = pct
                    if output_dir:
                        _write_phase14_summary(output_dir, phase14_summary)
    else:
        report["pct_recipe_ingredients_covered"] = None

    # Distribution of edges per ingredient
    if not df.empty:
        dist = df.groupby("ingredient_id").size()
        report["edges_per_ingredient_dist"] = [int(x) for x in dist.describe().tolist()] if len(dist) else []

    return report


def _write_phase14_summary(output_dir: Path, summary: Dict[str, Any]) -> None:
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "phase14_summary.json"
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.info("Updated %s with coverage_fail/coverage stats", path)
