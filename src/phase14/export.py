"""
Phase14: Export mediation CSVs, Neo4j CSVs, and reports.
Neo4j-ready exports with strict headers; sanity checks (no NaN in START/END); high_conf filtered export.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Neo4j import headers (display_name optional, for readability)
NEO4J_NODE_HEADERS = [":ID", ":LABEL", "name", "display_name", "source"]
NEO4J_EDGE_HEADERS = [":START_ID", ":END_ID", ":TYPE"]


def _ts() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _pair_category_mediation_export_df(pair_mediation: pd.DataFrame) -> pd.DataFrame:
    """Build pair_category_mediation with full columns for mechanism queries: ingA_id, ingB_id, category, did, q_global, p_analytic, stability_score, shared_*_count, mediated_path_score, propagated_pathway_score, coherence_score, dose_proxy_*, mechanistic_score, confidence, evidence_breakdown."""
    if pair_mediation is None or pair_mediation.empty:
        return pd.DataFrame()
    df = pair_mediation.copy()
    if "confidence" not in df.columns:
        df["confidence"] = df.get("mediation_confidence", 0.5)
    if "evidence_breakdown" not in df.columns:
        def _breakdown(r):
            c = r.get("shared_compounds_count", 0) or 0
            g = r.get("shared_genes_count", 0) or 0
            p = r.get("shared_pathways_count", 0) or 0
            return f"c:{c},g:{g},p:{p}"
        df["evidence_breakdown"] = df.apply(_breakdown, axis=1)
    priority = [
        "ingA_id", "ingB_id", "category", "did", "q_global", "p_analytic", "stability_score",
        "shared_compounds_count", "shared_genes_count", "shared_pathways_count",
        "mediated_path_score", "propagated_pathway_score", "mechanistic_score",
        "evidence_compounds", "evidence_genes", "evidence_paths",
        "q_weight", "stability_weight", "mediation_strength",
        "coherence_score", "confidence", "evidence_breakdown",
    ]
    dose_cols = [c for c in df.columns if str(c).startswith("dose_proxy")]
    cols = [c for c in priority + dose_cols if c in df.columns]
    return df[cols]


def write_mediation_artifacts(
    output_dir: Path,
    mediation_nodes: pd.DataFrame,
    mediation_edges: pd.DataFrame,
    mediation_edges_scored: Optional[pd.DataFrame] = None,
    pair_category_mediation: Optional[pd.DataFrame] = None,
    run_id: str = "phase14",
) -> None:
    """Write mediation_nodes.csv, mediation_edges.csv, mediation_edges_scored.csv, pair_category_mediation.csv (with confidence, evidence_breakdown)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mediation_nodes.to_csv(output_dir / "mediation_nodes.csv", index=False)
    mediation_edges.to_csv(output_dir / "mediation_edges.csv", index=False)
    if mediation_edges_scored is not None and not mediation_edges_scored.empty:
        mediation_edges_scored.to_csv(output_dir / "mediation_edges_scored.csv", index=False)
    if pair_category_mediation is not None and not pair_category_mediation.empty:
        pcm = _pair_category_mediation_export_df(pair_category_mediation)
        pcm.to_csv(output_dir / "pair_category_mediation.csv", index=False)
    logger.info("[%s] Wrote mediation artifacts to %s", _ts(), output_dir)


def neo4j_nodes_csv(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Produce Neo4j nodes table with strict headers: :ID, :LABEL, name, display_name, source."""
    if nodes_df.empty:
        return pd.DataFrame(columns=NEO4J_NODE_HEADERS)
    df = nodes_df.copy()
    df = df.rename(columns={"node_id": ":ID", "label": ":LABEL"})
    if "name" not in df.columns:
        df["name"] = df[":ID"]
    if "display_name" not in df.columns:
        df["display_name"] = df["name"]
    for h in NEO4J_NODE_HEADERS:
        if h not in df.columns:
            df[h] = None
    return df[[c for c in NEO4J_NODE_HEADERS if c in df.columns]]


def neo4j_edges_csv(edges_df: pd.DataFrame, include_properties: bool = True) -> pd.DataFrame:
    """Produce Neo4j edges table with :START_ID, :END_ID, :TYPE; optional property columns preserved when include_properties=True."""
    if edges_df.empty:
        return pd.DataFrame(columns=NEO4J_EDGE_HEADERS)
    df = edges_df.copy()
    df = df.rename(columns={"source_id": ":START_ID", "target_id": ":END_ID", "edge_type": ":TYPE"})
    for h in NEO4J_EDGE_HEADERS:
        if h not in df.columns:
            df[h] = None
    cols = list(NEO4J_EDGE_HEADERS)
    if include_properties:
        extra = [c for c in df.columns if c not in NEO4J_EDGE_HEADERS]
        cols = [c for c in cols + extra if c in df.columns]
    else:
        cols = [c for c in cols if c in df.columns]
    return df[cols]


def _sanity_check_neo4j_ids(n_df: pd.DataFrame, e_df: pd.DataFrame) -> None:
    """Fail fast if any NaN in :ID, :START_ID, :END_ID; list first 10 bad rows."""
    bad = []
    if ":ID" in n_df.columns:
        na_ids = n_df[n_df[":ID"].isna()]
        if not na_ids.empty:
            bad.append(("nodes", ":ID", na_ids.head(10)))
    for col in (":START_ID", ":END_ID"):
        if col in e_df.columns:
            na = e_df[e_df[col].isna()]
            if not na.empty:
                bad.append(("edges", col, na.head(10)))
    if bad:
        msg = "Neo4j export sanity check failed: NaN in ID columns. First 10 bad rows:\n"
        for which, col, frame in bad:
            msg += f"  {which} {col}: {frame.to_dict('records')}\n"
        raise ValueError(msg)


def _build_mediated_by_edges(
    pair_mediation: pd.DataFrame,
    mediation_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Build INT->PATH MEDIATED_BY edges when shared_pathways_count > 0; include mechanistic_score, mediated_path_score."""
    from .id_normalization import to_category_id, to_interaction_id
    if pair_mediation is None or pair_mediation.empty or mediation_edges.empty:
        return pd.DataFrame()
    map_edges = mediation_edges[mediation_edges["edge_type"] == "MAPS_TO_CATEGORY"]
    if map_edges.empty:
        return pd.DataFrame()
    cat_to_paths: Dict[str, List[str]] = {}
    for _, e in map_edges.iterrows():
        cat_to_paths.setdefault(str(e["target_id"]), []).append(str(e["source_id"]))
    rows = []
    for _, r in pair_mediation.iterrows():
        if (r.get("shared_pathways_count") or 0) <= 0:
            continue
        ing_a, ing_b = str(r.get("ingA_id", "")), str(r.get("ingB_id", ""))
        cat = str(r.get("category", "")).strip()
        cat_id = to_category_id(cat)
        int_id = to_interaction_id(ing_a, ing_b)
        paths = cat_to_paths.get(cat_id, [])
        mech = float(r.get("mechanistic_score", 0) or 0)
        mpath = float(r.get("mediated_path_score", 0) or 0)
        pprop = float(r.get("propagated_pathway_score", 0) or 0)
        for path_id in paths:
            rows.append({
                "source_id": int_id,
                "target_id": path_id,
                "edge_type": "MEDIATED_BY",
                "mechanistic_score": mech,
                "mediated_path_score": mpath,
                "propagated_pathway_score": pprop,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _enrich_affects_with_mechanistic_score(
    e_base: pd.DataFrame,
    pair_mediation: pd.DataFrame,
) -> pd.DataFrame:
    """Add mechanistic_score to AFFECTS edges from pair_mediation so it is queryable on INT->CAT."""
    from .id_normalization import to_category_id, to_interaction_id
    if pair_mediation is None or pair_mediation.empty or e_base.empty:
        return e_base
    key_to_score: Dict[tuple, float] = {}
    for _, r in pair_mediation.iterrows():
        int_id = to_interaction_id(str(r.get("ingA_id", "")), str(r.get("ingB_id", "")))
        cat_id = to_category_id(str(r.get("category", "")).strip())
        key_to_score[(int_id, cat_id)] = float(r.get("mechanistic_score") or 0)
    if "mechanistic_score" not in e_base.columns:
        e_base = e_base.reindex(columns=list(e_base.columns) + ["mechanistic_score"], fill_value=None)
    aff = e_base["edge_type"] == "AFFECTS"
    for idx in e_base.index[aff]:
        sid, tid = e_base.at[idx, "source_id"], e_base.at[idx, "target_id"]
        e_base.at[idx, "mechanistic_score"] = key_to_score.get((sid, tid))
    return e_base


def write_neo4j_export(
    output_dir: Path,
    mediation_nodes: pd.DataFrame,
    mediation_edges: pd.DataFrame,
    pair_mediation: Optional[pd.DataFrame] = None,
    run_id: str = "phase14",
) -> None:
    """Write neo4j/nodes.csv and neo4j/edges.csv. Optionally add MEDIATED_BY edges (INT->PATH) when pair_mediation has shared_pathways_count>0. AFFECTS edges get mechanistic_score from pair_mediation."""
    neo_dir = Path(output_dir) / "neo4j"
    neo_dir.mkdir(parents=True, exist_ok=True)
    n_df = neo4j_nodes_csv(mediation_nodes)
    e_base = mediation_edges.copy()
    e_base = _enrich_affects_with_mechanistic_score(e_base, pair_mediation)
    mediated = _build_mediated_by_edges(pair_mediation, mediation_edges)
    if not mediated.empty:
        e_base = e_base.reindex(
            columns=list(e_base.columns) + ["mediated_path_score", "propagated_pathway_score"],
            fill_value=None,
        )
        e_df = pd.concat([e_base, mediated], ignore_index=True)
    else:
        e_df = e_base
    e_df = neo4j_edges_csv(e_df, include_properties=True)
    _sanity_check_neo4j_ids(n_df, e_df)
    n_df.to_csv(neo_dir / "nodes.csv", index=False)
    e_df.to_csv(neo_dir / "edges.csv", index=False)
    logger.info("[%s] Wrote Neo4j export to %s", _ts(), neo_dir)
    write_ingredient_name_export_validation(n_df, output_dir)
    write_mechanistic_score_validation(e_df, output_dir)
    write_unknown_pathway_report(n_df, e_df, output_dir)


def write_ingredient_name_export_validation(neo4j_nodes_df: pd.DataFrame, output_dir: Path) -> None:
    """
    After Neo4j export, validate Ingredient nodes: total count, how many have name != id, sample 20 rows.
    Writes reports/ingredient_name_export_validation.json.
    """
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "ingredient_name_export_validation.json"
    if neo4j_nodes_df.empty or ":LABEL" not in neo4j_nodes_df.columns:
        report = {"total_ingredient_nodes": 0, "with_name_not_id": 0, "sample_rows": [], "error": "no nodes or no :LABEL"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return
    ing = neo4j_nodes_df[neo4j_nodes_df[":LABEL"].astype(str).str.contains("Ingredient", na=False)]
    total = len(ing)
    if ":ID" not in ing.columns or "name" not in ing.columns:
        report = {"total_ingredient_nodes": total, "with_name_not_id": 0, "sample_rows": [], "error": "missing :ID or name column"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return
    with_name_not_id = int((ing[":ID"].astype(str) != ing["name"].astype(str)).sum())
    sample = ing.head(20)[[":ID", "name"]].fillna("").astype(str)
    sample_rows = [{"id": row[":ID"], "name": row["name"]} for _, row in sample.iterrows()]
    report = {
        "total_ingredient_nodes": total,
        "with_name_not_id": with_name_not_id,
        "sample_rows": sample_rows,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("[%s] Ingredient name validation: %s Ingredient nodes, %s with name!=id -> %s", _ts(), total, with_name_not_id, out_path)


def write_mechanistic_score_validation(neo4j_edges_df: pd.DataFrame, output_dir: Path) -> None:
    """Write reports/mechanistic_score_validation.json: AFFECTS edges with mechanistic_score, sample rows."""
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "mechanistic_score_validation.json"
    if neo4j_edges_df.empty or ":TYPE" not in neo4j_edges_df.columns:
        report = {"affects_total": 0, "affects_with_mechanistic_score": 0, "sample_rows": [], "note": "mechanistic_score on AFFECTS (INT->CAT)"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return
    col_start = ":START_ID" if ":START_ID" in neo4j_edges_df.columns else "source_id"
    col_end = ":END_ID" if ":END_ID" in neo4j_edges_df.columns else "target_id"
    affects = neo4j_edges_df[neo4j_edges_df[":TYPE"].astype(str).str.strip() == "AFFECTS"]
    total = len(affects)
    if "mechanistic_score" not in affects.columns:
        with_score = 0
        sample_rows = []
    else:
        with_score = int(affects["mechanistic_score"].notna().sum())
        sample = affects.nlargest(10, "mechanistic_score", keep="first") if with_score else affects.head(5)
        sample_rows = [
            {"start_id": str(row[col_start]), "end_id": str(row[col_end]), "mechanistic_score": row.get("mechanistic_score")}
            for _, row in sample.iterrows()
        ]
    report = {
        "affects_total": total,
        "affects_with_mechanistic_score": with_score,
        "sample_rows": sample_rows,
        "note": "mechanistic_score on AFFECTS (Interaction->Category); also on MEDIATED_BY (Interaction->Pathway)",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("[%s] Mechanistic score validation: %s AFFECTS edges, %s with score -> %s", _ts(), total, with_score, out_path)


def write_unknown_pathway_report(neo4j_nodes_df: pd.DataFrame, neo4j_edges_df: pd.DataFrame, output_dir: Path) -> None:
    """Write reports/unknown_pathway_report.json: MEDIATED_BY to Unknown pathway count, fraction of interactions only-Unknown."""
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "unknown_pathway_report.json"
    report: Dict[str, Any] = {"mediated_by_total": 0, "mediated_by_to_unknown_pathway": 0, "interactions_with_mediated_by": 0, "interactions_only_unknown_pathway": 0, "fraction_only_unknown": None}
    if neo4j_nodes_df.empty or neo4j_edges_df.empty:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return
    col_end = ":END_ID" if ":END_ID" in neo4j_edges_df.columns else "target_id"
    pathway_nodes = neo4j_nodes_df[neo4j_nodes_df[":LABEL"].astype(str).str.strip().str.contains("Pathway", na=False)]
    disp_col = "display_name" if "display_name" in pathway_nodes.columns else "name"
    unknown_pathway_ids = set(
        pathway_nodes[pathway_nodes[disp_col].fillna("").astype(str).str.strip() == "Unknown pathway"][":ID"].astype(str)
    )
    mediated = neo4j_edges_df[neo4j_edges_df[":TYPE"].astype(str).str.strip() == "MEDIATED_BY"]
    col_start = ":START_ID" if ":START_ID" in neo4j_edges_df.columns else "source_id"
    report["mediated_by_total"] = len(mediated)
    report["mediated_by_to_unknown_pathway"] = int(mediated[mediated[col_end].astype(str).isin(unknown_pathway_ids)].shape[0])
    int_ids = mediated[col_start].unique()
    report["interactions_with_mediated_by"] = len(int_ids)
    only_unknown = 0
    for iid in int_ids:
        targets = set(mediated[mediated[col_start] == iid][col_end].astype(str))
        if targets and targets <= unknown_pathway_ids:
            only_unknown += 1
    report["interactions_only_unknown_pathway"] = only_unknown
    report["fraction_only_unknown"] = round(only_unknown / len(int_ids), 4) if len(int_ids) > 0 else None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("[%s] Unknown pathway report: %s MEDIATED_BY to Unknown, %s interactions only-Unknown (%.1f%%) -> %s",
                _ts(), report["mediated_by_to_unknown_pathway"], only_unknown,
                100 * (report["fraction_only_unknown"] or 0), out_path)


def high_conf_mechanistic_df(
    pair_mediation: pd.DataFrame,
    confidence_min: float = 0.8,
    stability_min: float = 0.5,
    mechanistic_p75_within_category: bool = True,
) -> pd.DataFrame:
    """Filter to high-confidence mechanistic pairs: confidence>=confidence_min, stability_score>=stability_min, mechanistic_score>=p75 within category."""
    if pair_mediation is None or pair_mediation.empty:
        return pd.DataFrame()
    df = pair_mediation.copy()
    conf_col = "confidence" if "confidence" in df.columns else "mediation_confidence"
    if conf_col not in df.columns:
        return pd.DataFrame()
    df = df[df[conf_col].fillna(0) >= confidence_min]
    if "stability_score" in df.columns:
        df = df[df["stability_score"].fillna(0) >= stability_min]
    if df.empty:
        return df
    if mechanistic_p75_within_category and "category" in df.columns and "mechanistic_score" in df.columns:
        p75 = df.groupby("category")["mechanistic_score"].transform(lambda x: x.quantile(0.75))
        df = df[df["mechanistic_score"] >= p75]
    return df


def write_high_conf_export(output_dir: Path, pair_mediation: pd.DataFrame) -> None:
    """Write optional high_conf_mechanistic.csv (confidence>=0.8, stability>=0.5, mechanistic_score>=p75 within category)."""
    output_dir = Path(output_dir)
    df = high_conf_mechanistic_df(pair_mediation)
    if df.empty:
        logger.info("[%s] No high-conf mechanistic pairs; skipping high_conf_mechanistic.csv", _ts())
        return
    out_path = output_dir / "high_conf_mechanistic.csv"
    df.to_csv(out_path, index=False)
    logger.info("[%s] Wrote %s (%s rows)", _ts(), out_path, len(df))


def write_reports(
    output_dir: Path,
    summary: Dict[str, Any],
    top_mediated: Optional[pd.DataFrame] = None,
    run_id: str = "phase14",
) -> None:
    """Write reports/phase14_summary.json and reports/top_mediated_pairs_by_category.csv."""
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "phase14_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if top_mediated is not None and not top_mediated.empty:
        top_mediated.to_csv(report_dir / "top_mediated_pairs_by_category.csv", index=False)
    logger.info("[%s] Wrote reports to %s", _ts(), report_dir)


def write_propagation_diagnostics(output_dir: Path, diagnostics: Dict[str, Any]) -> None:
    """Write reports/propagation_diagnostics.json."""
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "propagation_diagnostics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("[%s] Wrote %s", _ts(), path)


def write_mechanistic_score_diagnostics(
    output_dir: Path,
    pair_mediation: pd.DataFrame,
    saturation_warn_pct: float = 30.0,
) -> None:
    """
    Write reports/mechanistic_score_diagnostics.json with quantiles of mechanistic_score,
    evidence_compounds, evidence_genes, evidence_paths, stability_score, and pct of rows with mechanistic_score > 0.95.
    If > saturation_warn_pct of rows are > 0.95, log a warning suggesting score saturation.
    """
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "mechanistic_score_diagnostics.json"
    if pair_mediation is None or pair_mediation.empty:
        out = {"n_rows": 0, "pct_mechanistic_gt_095": 0.0, "quantiles": {}, "saturation_warning": False}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info("[%s] Wrote %s (no data)", _ts(), path)
        return
    df = pair_mediation
    n = len(df)
    cols = ["mechanistic_score", "evidence_compounds", "evidence_genes", "evidence_paths", "stability_score"]
    quantiles: Dict[str, Dict[str, float]] = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].fillna(0)
        q = s.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
        quantiles[c] = {str(k): round(float(v), 6) for k, v in q.items()}
    pct_gt_095 = 100.0 * (df["mechanistic_score"].fillna(0) > 0.95).sum() / n if "mechanistic_score" in df.columns else 0.0
    out = {
        "n_rows": n,
        "pct_mechanistic_gt_095": round(pct_gt_095, 2),
        "quantiles": quantiles,
        "saturation_warning": pct_gt_095 > saturation_warn_pct,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("[%s] Wrote %s (pct_mechanistic_gt_095=%.1f%%)", _ts(), path, pct_gt_095)
    if pct_gt_095 > saturation_warn_pct:
        logger.warning(
            "[%s] Mechanistic score saturation: %.1f%% of rows have mechanistic_score > 0.95 (threshold %.0f%%). Consider reviewing score calibration.",
            _ts(), pct_gt_095, saturation_warn_pct,
        )
    return out


def write_identity_diagnostics(
    output_dir: Path,
    identity_metrics: Dict[str, Any],
    overlap_compounds_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Write reports/identity_metrics.json and reports/overlap_compounds.csv.
    identity_metrics should include: n_ic_compounds, n_cg_compounds, n_overlap, overlap_vs_cg, overlap_vs_ic,
    gate_passed, gate_reason, and optionally top_overlapping_inchikeys (list of {inchikey, count_in_ic, count_in_cg}).
    overlap_compounds_df: optional DataFrame with columns e.g. compound_id (InChIKey), count_in_ic, count_in_cg, ingredients, genes.
    """
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    # JSON-serializable copy (no sets)
    out_metrics = {k: v for k, v in identity_metrics.items() if k != "overlap_set" and not isinstance(v, set)}
    with open(report_dir / "identity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)
    if overlap_compounds_df is not None and not overlap_compounds_df.empty:
        overlap_compounds_df.to_csv(report_dir / "overlap_compounds.csv", index=False)
    logger.info("[%s] Wrote identity_metrics.json (and overlap_compounds.csv) to %s", _ts(), report_dir)


def write_triplet_reports(
    output_dir: Path,
    triplets_df: pd.DataFrame,
    triplet_category_df: pd.DataFrame,
    top_overall_n: int = 200,
    top_per_category_n: int = 50,
) -> None:
    """Write reports/top_triplets_overall.csv (top by PMI_ABC) and reports/top_triplets_by_category.csv (top per category by triplet_category_score)."""
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not triplets_df.empty and "PMI_ABC" in triplets_df.columns:
        top = triplets_df.nlargest(top_overall_n, "PMI_ABC")
        top.to_csv(report_dir / "top_triplets_overall.csv", index=False)
        logger.info("[%s] Wrote %s (%s rows)", _ts(), report_dir / "top_triplets_overall.csv", len(top))
    if not triplet_category_df.empty and "triplet_category_score" in triplet_category_df.columns:
        top_cat = (
            triplet_category_df[triplet_category_df["triplet_category_score"] > 0]
            .sort_values(["category", "triplet_category_score"], ascending=[True, False])
            .groupby("category", group_keys=False)
            .head(top_per_category_n)
            .reset_index(drop=True)
        )
        top_cat.to_csv(report_dir / "top_triplets_by_category.csv", index=False)
        logger.info("[%s] Wrote %s (%s rows)", _ts(), report_dir / "top_triplets_by_category.csv", len(top_cat))


def write_triplet_neo4j(
    output_dir: Path,
    triplet_nodes: pd.DataFrame,
    triplet_edges: pd.DataFrame,
) -> None:
    """Write neo4j/nodes_triplets.csv and neo4j/edges_triplets.csv with Neo4j import headers."""
    neo_dir = Path(output_dir) / "neo4j"
    neo_dir.mkdir(parents=True, exist_ok=True)
    if triplet_nodes.empty:
        n_df = pd.DataFrame(columns=NEO4J_NODE_HEADERS)
    else:
        n_df = triplet_nodes.copy()
        n_df = n_df.rename(columns={"node_id": ":ID", "label": ":LABEL"})
        if "name" not in n_df.columns:
            n_df["name"] = n_df[":ID"]
        for h in NEO4J_NODE_HEADERS:
            if h not in n_df.columns:
                n_df[h] = None
        n_df = n_df[[c for c in NEO4J_NODE_HEADERS if c in n_df.columns]]
    if triplet_edges.empty:
        e_df = pd.DataFrame(columns=NEO4J_EDGE_HEADERS)
    else:
        e_df = neo4j_edges_csv(triplet_edges, include_properties=True)
    n_df.to_csv(neo_dir / "nodes_triplets.csv", index=False)
    e_df.to_csv(neo_dir / "edges_triplets.csv", index=False)
    logger.info("[%s] Wrote triplet Neo4j export to %s (nodes=%s, edges=%s)", _ts(), neo_dir, len(n_df), len(e_df))


def write_triplet_mining_summary(output_dir: Path, summary: Dict[str, Any]) -> None:
    """Write reports/triplet_mining_summary.json with sampling params, stage counts, elapsed time."""
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "triplet_mining_summary.json"
    # JSON-serializable
    out = {}
    for k, v in summary.items():
        if v is None:
            continue
        if isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif hasattr(v, "item"):  # numpy scalar
            out[k] = v.item()
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("[%s] Wrote %s", _ts(), path)
