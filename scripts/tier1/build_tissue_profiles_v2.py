#!/usr/bin/env python3
"""
Enrichment-de-saturated tissue-localization profiles (v2).

Replaces flat confidence-weighted gene sums (v1) with per-(ingredient, gene)
distinctiveness weights derived from calibrated enrichment + gene IDF +
ingredient-gene specificity. GTEx tissue_weight(g,T) unchanged from v1.

Usage (from repo root):
    python scripts/tier1/build_tissue_profiles_v2.py

Outputs (new only; v1 + MoA untouched):
    data/processed/tier1/ingredient_tissue_profiles_v2.parquet
    data/processed/tier1/tissue_profiles_v2_build_report_v1.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse GTEx parse/join/expression from v1 tissue builder.
_v1_spec = importlib.util.spec_from_file_location("build_tissue_moa_v1", SCRIPT_DIR / "build_tissue_moa_v1.py")
_v1 = importlib.util.module_from_spec(_v1_spec)
assert _v1_spec.loader is not None
_v1_spec.loader.exec_module(_v1)

TIER1 = ROOT / "data/processed/tier1"
ENRICHMENT = TIER1 / "enrichment_weighted_v3_calibrated.parquet"
GENE_SETS_V3 = ROOT / "data/processed/integrated/ingredient_gene_sets_v3.parquet"
GENE_IDF = TIER1 / "weights/gene_idf_weights_20260625_083943.csv"
TISSUE_V1 = TIER1 / "ingredient_tissue_profiles_v1.parquet"
STRING_MAP = ROOT / "data/processed/canonical/ingredient_string_species_v2.parquet"

TISSUE_V2 = TIER1 / "ingredient_tissue_profiles_v2.parquet"
REPORT_OUT = TIER1 / "tissue_profiles_v2_build_report_v1.json"

SAMPLE_INGREDIENTS = ["SP_000052", "SP_000005", "SP_000259", "SP_000235", "SP_000026"]
SPOTCHECK_INGREDIENTS = ["SP_000052", "SP_000259", "SP_000235"]

Q_STRICT = 0.10
Q_RELAX = 0.15
RELAX_MULT = 0.25
FALLBACK_SCALE = 0.02
TOP_K_GENES = 75
DISTINCTIVENESS_POWER = 2.0

WEIGHTING_FORMULA_DOC = """
Per-pathway signal (ingredient i, pathway p):
  pathway_weight(i,p) = weighted_fold_enrichment(p) × max(0, -log10(max(q_value(p), 1e-300))) × tier_mult(p)

  tier_mult = 1.0 when q <= 0.10 (strict significant)
            = 0.25 when 0.10 < q <= 0.15 (relaxed tier; only if strict tier yields no drivers for i)
            = 0.0 otherwise

Per-gene enrichment raw score (before ingredient specificity):
  raw_enrich(i,g) = Σ_{p: g ∈ drivers(p)} pathway_weight(i,p) × confidence(g) × idf_gene(g)

Ingredient-gene specificity (down-weights promiscuous drivers across ingredients):
  spec(g) = log(N_ingredients / |{i : g appears in any q<=0.10 enriched pathway for i}|)

Distinctiveness (top-K truncated, power-compressed, normalized per ingredient):
  raw_distinct(i,g) = raw_enrich(i,g) × spec(g)
  distinctiveness(i,g) = raw_distinct(i,g)^2  among top-75 genes by raw_distinct, then / Σ_g

Fallback (13 ingredients with no q<=0.15 enrichment drivers):
  raw_distinct(i,g) = 0.02 × confidence(g) × idf_gene(g) × spec(g)

Tissue score (same GTEx tissue_weight as v1):
  tissue_weight(g,T) = shifted z-scored log1p(TPM) row-normalized per gene
  tissue_score(i,T) = Σ_g distinctiveness(i,g) × tissue_weight(g,T)
  normalized_score(i,T) = tissue_score(i,T) / Σ_T tissue_score(i,T)
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gene_idf() -> dict[str, float]:
    df = pd.read_csv(GENE_IDF)
    return dict(zip(df["gene_symbol"].astype(str), df["idf_gene"].astype(float)))


def build_gene_ingredient_specificity(enrichment: pd.DataFrame) -> dict[str, float]:
    """How many ingredients have gene g as driver in q<=0.10 enriched pathways."""
    gene_ings: dict[str, set[str]] = defaultdict(set)
    sig = enrichment[enrichment["q_value"] <= Q_STRICT]
    for ing, sub in sig.groupby("ingredient_id"):
        for row in sub["driving_genes_json"]:
            for d in json.loads(row):
                gene_ings[d["gene_symbol"]].add(str(ing))
    n_ing = enrichment["ingredient_id"].nunique()
    return {g: math.log(n_ing / max(1, len(ings))) for g, ings in gene_ings.items()}


def aggregate_pathway_raw(
    sub: pd.DataFrame,
    q_max: float,
    tier_mult: float,
) -> dict[str, float]:
    raw: dict[str, float] = defaultdict(float)
    for _, row in sub[sub["q_value"] <= q_max].iterrows():
        pw = (
            tier_mult
            * max(0.0, -math.log10(max(float(row["q_value"]), 1e-300)))
            * float(row["weighted_fold_enrichment"])
        )
        for d in json.loads(row["driving_genes_json"]):
            g = d["gene_symbol"]
            raw[g] += pw * float(d["confidence"]) * float(d["idf_gene"])
    return dict(raw)


def build_distinctiveness_weights(
    enrichment: pd.DataFrame,
    gene_sets: pd.DataFrame,
    gene_idf: dict[str, float],
    gene_spec: dict[str, float],
    gtex_genes: set[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Returns long DataFrame: ingredient_id, gene_symbol, distinctiveness_weight, evidence, weight_tier
    and tier counts per ingredient.
    """
    n_ing = enrichment["ingredient_id"].nunique()
    default_spec = math.log(float(n_ing))
    rows: list[dict[str, Any]] = []
    tiers: dict[str, str] = {}

    evidence_map = (
        gene_sets.groupby(["ingredient_id", "gene_symbol"])["evidence"]
        .agg(lambda s: s.iloc[0])
        .to_dict()
    )

    for ing_id, sub in enrichment.groupby("ingredient_id"):
        ing = str(ing_id)
        raw = aggregate_pathway_raw(sub, Q_STRICT, 1.0)
        tier = "strict_q010"
        if not raw:
            raw = aggregate_pathway_raw(sub, Q_RELAX, RELAX_MULT)
            tier = "relaxed_q015"
        if not raw:
            gsub = gene_sets[gene_sets["ingredient_id"] == ing_id]
            for _, grow in gsub.iterrows():
                sym = str(grow["gene_symbol"])
                if sym not in gtex_genes:
                    continue
                raw[sym] = raw.get(sym, 0.0) + FALLBACK_SCALE * float(grow["confidence"]) * gene_idf.get(sym, 0.0)
            tier = "fallback_conf_idf"

        scaled = {
            g: (v * gene_spec.get(g, default_spec)) ** DISTINCTIVENESS_POWER
            for g, v in raw.items()
            if v > 0 and g in gtex_genes
        }
        top = dict(sorted(scaled.items(), key=lambda x: (-x[1], x[0]))[:TOP_K_GENES])
        total = sum(top.values())
        if total <= 0:
            tiers[ing] = f"{tier}_empty"
            continue

        tiers[ing] = tier
        for g, w in top.items():
            rows.append(
                {
                    "ingredient_id": ing,
                    "gene_symbol": g,
                    "distinctiveness_weight": w / total,
                    "evidence": evidence_map.get((ing_id, g), evidence_map.get((ing, g), "unknown")),
                    "weight_tier": tier,
                }
            )

    return pd.DataFrame(rows), tiers


def build_tissue_profiles_v2(
    dist_weights: pd.DataFrame,
    expr: pd.DataFrame,
) -> pd.DataFrame:
    tissues = expr.columns.tolist()
    out_rows: list[dict[str, Any]] = []

    for ing_id, grp in dist_weights.groupby("ingredient_id"):
        genes = grp["gene_symbol"].astype(str).tolist()
        w = grp["distinctiveness_weight"].astype(float).values
        evidence = grp["evidence"].astype(str).tolist()
        mask = [g in expr.index for g in genes]
        if not any(mask):
            continue
        genes = [g for g, m in zip(genes, mask) if m]
        w = w[mask]
        evidence = [e for e, m in zip(evidence, mask) if m]
        w = w / w.sum()
        sub = expr.loc[genes].values
        raw = (sub * w.reshape(-1, 1)).sum(axis=0)

        meas_m = np.array([e == "measured" for e in evidence])
        meas_score = (sub[meas_m] * w[meas_m].reshape(-1, 1)).sum(axis=0) if meas_m.any() else np.zeros(len(tissues))
        pred_score = raw - meas_score

        total = float(raw.sum())
        norm = raw / total if total > 1e-12 else np.ones(len(tissues)) / len(tissues)

        ms_tot = float(meas_score.sum())
        ps_tot = float(pred_score.sum())
        denom = ms_tot + ps_tot
        meas_frac = ms_tot / denom if denom > 0 else np.nan
        if np.isnan(meas_frac) or meas_frac < 0.05:
            split = "predicted_dominant"
        elif meas_frac > 0.95:
            split = "measured_dominant"
        else:
            split = "mixed"

        for i, tissue in enumerate(tissues):
            out_rows.append(
                {
                    "ingredient_id": ing_id,
                    "tissue": tissue,
                    "tissue_score": float(raw[i]),
                    "normalized_score": float(norm[i]),
                    "n_genes_contributing": len(genes),
                    "measured_vs_predicted_split": split,
                    "measured_score_component": float(meas_score[i]),
                    "predicted_score_component": float(pred_score[i]),
                    "measured_fraction_of_score": meas_frac,
                    "interpretation_note": _v1.INTERPRETATION_NOTE,
                    "weighting_method": "enrichment_distinctiveness_v2",
                }
            )
    return pd.DataFrame(out_rows)


def profile_correlation(profiles: pd.DataFrame, n: int = 30, seed: int = 42) -> float:
    wide = profiles.pivot_table(
        index="ingredient_id", columns="tissue", values="normalized_score", fill_value=0.0
    )
    if len(wide) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(wide.index.to_numpy(), size=min(n, len(wide)), replace=False)
    sub = wide.loc[sample_ids]
    corr = sub.T.corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    return float(corr.where(mask).stack().mean())


def top_tissues(profiles: pd.DataFrame, ing_id: str, k: int = 5) -> list[dict[str, Any]]:
    sub = profiles[profiles["ingredient_id"] == ing_id].nlargest(k, "normalized_score")
    return sub[["tissue", "normalized_score"]].to_dict(orient="records")


def spotcheck_drivers(
    ing_id: str,
    dist_weights: pd.DataFrame,
    expr: pd.DataFrame,
    top_tissue: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    grp = dist_weights[dist_weights["ingredient_id"] == ing_id].copy()
    if grp.empty or top_tissue not in expr.columns:
        return []
    grp["tissue_contrib"] = grp["gene_symbol"].map(expr[top_tissue]) * grp["distinctiveness_weight"]
    top = grp.nlargest(k, "tissue_contrib")
    return top[["gene_symbol", "distinctiveness_weight", "evidence", "tissue_contrib"]].to_dict(orient="records")


def label_lookup(string_map: pd.DataFrame | None) -> dict[str, str]:
    if string_map is None or string_map.empty:
        return {}
    if "species_node" in string_map.columns and "canonical_name" in string_map.columns:
        return dict(zip(string_map["species_node"].astype(str), string_map["canonical_name"].astype(str)))
    return {}


def main() -> int:
    TIER1.mkdir(parents=True, exist_ok=True)

    gtex_path = _v1.resolve_gtex_path()
    gtex, tissue_cols, gtex_meta = _v1.load_gtex(gtex_path)
    our_genes = _v1.load_our_genes()
    sym_to_ens, _ = _v1.load_hgnc_maps()
    join = _v1.gtex_join_report(gtex, our_genes, sym_to_ens)
    if not join["gate_passed"]:
        print("STOP: GTEx join gate failed.")
        return 1

    expr, _ = _v1.build_expression_matrix(gtex, tissue_cols, join["gene_to_gtex_row"])
    gtex_genes = set(expr.index)

    enrichment = pd.read_parquet(ENRICHMENT)
    gene_sets = pd.read_parquet(GENE_SETS_V3)
    gene_idf = load_gene_idf()
    gene_spec = build_gene_ingredient_specificity(enrichment)

    dist_weights, weight_tiers = build_distinctiveness_weights(
        enrichment, gene_sets, gene_idf, gene_spec, gtex_genes
    )
    profiles_v2 = build_tissue_profiles_v2(dist_weights, expr)
    profiles_v2.to_parquet(TISSUE_V2, index=False)

    profiles_v1 = pd.read_parquet(TISSUE_V1)
    corr_v1 = profile_correlation(profiles_v1)
    corr_v2 = profile_correlation(profiles_v2)

    string_map = pd.read_parquet(STRING_MAP) if STRING_MAP.exists() else None
    names = label_lookup(string_map)

    comparison: list[dict[str, Any]] = []
    for ing_id in SAMPLE_INGREDIENTS:
        comparison.append(
            {
                "ingredient_id": ing_id,
                "label": names.get(ing_id, ing_id),
                "weight_tier": weight_tiers.get(ing_id),
                "n_distinctive_genes": int(
                    dist_weights[dist_weights["ingredient_id"] == ing_id]["gene_symbol"].nunique()
                ),
                "v1_top5": top_tissues(profiles_v1, ing_id),
                "v2_top5": top_tissues(profiles_v2, ing_id),
            }
        )

    top1_v1 = [top_tissues(profiles_v1, i, 1)[0]["tissue"] for i in SAMPLE_INGREDIENTS if i in set(profiles_v1["ingredient_id"])]
    top1_v2 = [top_tissues(profiles_v2, i, 1)[0]["tissue"] for i in SAMPLE_INGREDIENTS if i in set(profiles_v2["ingredient_id"])]

    spotchecks: list[dict[str, Any]] = []
    for ing_id in SPOTCHECK_INGREDIENTS:
        if ing_id not in set(profiles_v2["ingredient_id"]):
            continue
        top_t = profiles_v2[profiles_v2["ingredient_id"] == ing_id].nlargest(1, "normalized_score").iloc[0]["tissue"]
        spotchecks.append(
            {
                "ingredient_id": ing_id,
                "label": names.get(ing_id, ing_id),
                "top_tissue": top_t,
                "top_driving_genes": spotcheck_drivers(ing_id, dist_weights, expr, top_t),
            }
        )

    tier_counts = pd.Series(weight_tiers).value_counts().to_dict()
    overcorrect_flags = {
        "mean_pairwise_corr_dropped": corr_v1 - corr_v2,
        "corr_v1": round(corr_v1, 4),
        "corr_v2": round(corr_v2, 4),
        "top1_all_same_v1": len(set(top1_v1)) == 1,
        "top1_all_same_v2": len(set(top1_v2)) == 1,
        "top1_tissues_v1": top1_v1,
        "top1_tissues_v2": top1_v2,
        "plausible_spotchecks": all(len(s["top_driving_genes"]) > 0 for s in spotchecks),
    }

    report: dict[str, Any] = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "weighting_formula": WEIGHTING_FORMULA_DOC.strip(),
        "parameters": {
            "Q_STRICT": Q_STRICT,
            "Q_RELAX": Q_RELAX,
            "RELAX_MULT": RELAX_MULT,
            "FALLBACK_SCALE": FALLBACK_SCALE,
            "TOP_K_GENES": TOP_K_GENES,
            "DISTINCTIVENESS_POWER": DISTINCTIVENESS_POWER,
            "gene_idf_table": str(GENE_IDF.relative_to(ROOT)),
            "enrichment_input": str(ENRICHMENT.relative_to(ROOT)),
        },
        "gtex_join": {k: v for k, v in join.items() if k != "gene_to_gtex_row"},
        "weight_tier_counts": tier_counts,
        "deconvergence": overcorrect_flags,
        "sample_v1_vs_v2": comparison,
        "spotcheck_top_genes": spotchecks,
        "outputs": {
            "v2_parquet": str(TISSUE_V2.relative_to(ROOT)),
            "v2_sha256": sha256_file(TISSUE_V2),
            "v2_rows": len(profiles_v2),
            "v1_preserved": str(TISSUE_V1.relative_to(ROOT)),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Tissue profiles v2 ===")
    print(f"Formula: enrichment × idf_gene × ingredient-gene spec; top-{TOP_K_GENES}, power={DISTINCTIVENESS_POWER}")
    print(f"Pairwise correlation: v1={corr_v1:.4f} -> v2={corr_v2:.4f} (drop {corr_v1-corr_v2:.4f})")
    print(f"Top-1 tissues v1: {top1_v1}")
    print(f"Top-1 tissues v2: {top1_v2}")
    print(f"Wrote {TISSUE_V2.relative_to(ROOT)}")
    print(f"Report {REPORT_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
