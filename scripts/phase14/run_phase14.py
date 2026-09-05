"""
Phase14 headless pipeline: load Phase13 CSVs, build mediation KG, score, export.
Run from repo root: python scripts/phase14/run_phase14.py [--smoke] [--rows 200]
Uses resolve_phase14_inputs for validation and path resolution; fails fast if required inputs missing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("phase14")

DEFAULT_PHASE13_DIR = "data/processed/phase13_interactions_v3_20260206_162122_b_gpu_stable"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase14 Mediation KG pipeline")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root (default: script parent)")
    parser.add_argument("--phase13-dir", type=str, default=DEFAULT_PHASE13_DIR, help="Phase13 dir relative to repo")
    parser.add_argument("--output-run-id", type=str, default=None, help="Run ID subdir (default: phase14_YYYYMMDD_HHMMSS)")
    parser.add_argument("--smoke", action="store_true", help="Run on first N rows only")
    parser.add_argument("--rows", type=int, default=200, help="N rows for smoke mode")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    phase13_dir = Path(args.phase13_dir)

    from src.phase14 import phase14_config as config
    from src.phase14.audit import pick_best_sources, scan_processed_data, write_audit
    from src.phase14.loaders import (
        resolve_phase14_inputs,
        resolved_to_paths,
        load_phase13_csvs,
        load_recipe_ingredients,
        load_pathway_bundles,
        load_pathway_cluster_info,
        load_target_functional_clusters,
        load_ingredient_compound,
        load_compound_gene,
        load_compound_target,
        load_ingredient_id_to_name,
        summarize_missing,
        compute_phase14_coverage_metrics,
        ensure_compound_gene_when_ingredient_compound,
    )
    from src.phase14.mediation_graph import build_mediation_graph
    from src.phase14.mediation_scoring import compute_mediation_features, add_mechanistic_score
    from src.phase14.propagation import build_ingredient_genes_from_tables, propagated_scores_for_pairs, pct_resolved_category_has_pathways
    from src.phase14.consistency import coherence_flag_and_score
    from src.phase14.dose_proxy import add_dose_proxy
    from src.phase14.export import write_mediation_artifacts, write_neo4j_export, write_reports, write_high_conf_export, write_identity_diagnostics, write_propagation_diagnostics, write_mechanistic_score_diagnostics
    from src.phase14.ingredient_compound import validate_ing_compound_edges
    from src.phase14.loaders import build_overlap_compounds_and_identity_metrics

    resolved = resolve_phase14_inputs(repo_root, phase13_dir)
    paths = resolved_to_paths(repo_root, resolved)
    for k, v in sorted(resolved.items()):
        logger.info("  %s: %s", k, v)
    missing = summarize_missing(paths)
    if missing:
        logger.warning("Missing optional inputs: %s", missing)

    # Data audit first: scan processed, pick best sources, write audit report
    run_id = args.output_run_id or ("phase14_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    out_base = repo_root / config.PHASE14_OUT_BASE / run_id
    out_base.mkdir(parents=True, exist_ok=True)
    scan_df = scan_processed_data(repo_root)
    chosen = pick_best_sources(scan_df, repo_root)
    write_audit(scan_df, chosen, out_base, run_id)
    logger.info("Selected sources: ingredient_compound=%s, compound_gene=%s", chosen.get("ingredient_compound"), chosen.get("compound_gene"))
    for line in chosen.get("reasoning", []):
        logger.info("  %s", line)

    phase13_dir_path = paths.get("phase13_dir")
    if not phase13_dir_path or not phase13_dir_path.exists():
        logger.error(
            "Phase14 required input missing: phase13_dir. Expected dir with atlas_confirmed.csv, bootstrap_stability.csv. "
            "Run: python scripts/phase14/validate_phase14_inputs.py --strict --repo-root %s --phase13-dir %s",
            repo_root, phase13_dir,
        )
        return 1
    if not (phase13_dir_path / "atlas_confirmed.csv").exists():
        logger.error("Phase13 dir found but atlas_confirmed.csv missing. Run validate_phase14_inputs.py for details.")
        return 1
    if not (phase13_dir_path / "bootstrap_stability.csv").exists():
        logger.error("Phase13 bootstrap_stability.csv missing. Run validate_phase14_inputs.py for details.")
        return 1

    atlas, stability, kg_edges, kg_nodes = load_phase13_csvs(phase13_dir_path)
    if args.smoke:
        atlas = atlas.head(args.rows)
        logger.info("Smoke mode: %s rows (full mapping sources still used for ingredient->compound)", len(atlas))
    else:
        logger.info("Full run: %s atlas rows", len(atlas))

    recipe_ingredients = load_recipe_ingredients(paths)
    pathway_bundles = load_pathway_bundles(paths)
    pathway_cluster_info = load_pathway_cluster_info(paths)
    target_clusters = load_target_functional_clusters(paths)
    ingredient_compound = load_ingredient_compound(paths)
    compound_gene = load_compound_gene(paths, chosen=chosen, full_run_gate=not args.smoke)
    compound_target = load_compound_target(paths)
    ensure_compound_gene_when_ingredient_compound(
        ingredient_compound,
        compound_gene,
        canonical_expected_path="data/processed/canonical/compound_gene_expanded_canonical.csv",
    )

    ingredient_id_to_name = load_ingredient_id_to_name(repo_root=repo_root)
    mediation_nodes, mediation_edges = build_mediation_graph(
        atlas, kg_edges, kg_nodes,
        pathway_cluster_info, target_clusters, pathway_bundles,
        ingredient_compound=ingredient_compound, compound_gene=compound_gene,
        ingredient_id_to_name=ingredient_id_to_name,
    )
    logger.info("Mediation graph: %s nodes, %s edges", len(mediation_nodes), len(mediation_edges))

    features = compute_mediation_features(atlas, mediation_edges, mediation_nodes, compound_target_df=compound_target)
    pair_mediation = add_mechanistic_score(features, stability)
    ingredient_genes = build_ingredient_genes_from_tables(ingredient_compound, compound_gene)
    prop_scores, prop_diag = propagated_scores_for_pairs(atlas, mediation_edges, ingredient_genes=ingredient_genes)
    pct_resolved = pct_resolved_category_has_pathways(atlas, mediation_edges)
    if isinstance(prop_diag, dict):
        prop_diag["pct_resolved_category_has_pathways"] = pct_resolved
    if not args.smoke and pct_resolved < 50.0:
        raise ValueError(
            "Phase14 category-bundle alignment failed: pct_resolved_category_has_pathways=%.1f%% (need >= 50%%). "
            "Atlas categories do not resolve to pathway_bundles keys with pathways in the graph. "
            "Run: python scripts/phase14/diagnose_propagation_zero_rows.py and check data/processed/canonical/reports/propagation_zero_row_diagnosis.csv (zero_reason=missing_bundle). "
            "Ensure CATEGORY_SYNONYMS_FOR_PROPAGATION in phase14_config.py maps atlas categories to pathway_bundles keys that have pathway clusters."
            % (pct_resolved,)
        )
    pair_mediation = pair_mediation.merge(prop_scores, on=["ingA_id", "ingB_id", "category"], how="left")
    pair_mediation = coherence_flag_and_score(pair_mediation)
    pair_mediation = add_dose_proxy(pair_mediation, recipe_ingredients, mediation_edges)

    # Summary: coverage and source files used
    summary = {
        "run_id": run_id,
        "n_nodes": len(mediation_nodes),
        "n_edges": len(mediation_edges),
        "n_pairs": len(pair_mediation),
        "ingredient_compound_source": chosen.get("ingredient_compound_reason") or "derived_or_audit",
        "compound_gene_source": str(chosen.get("compound_gene") or "none"),
    }
    if not ingredient_compound.empty:
        coverage_report = validate_ing_compound_edges(ingredient_compound, recipe_ingredients, output_dir=out_base, phase14_summary=summary)
        summary["ingredient_compound_n_edges"] = coverage_report.get("n_edges", 0)
        summary["ingredient_compound_pct_covered"] = coverage_report.get("pct_recipe_ingredients_covered")
    if not compound_gene.empty:
        summary["compound_gene_n_edges"] = len(compound_gene)
    cov = compute_phase14_coverage_metrics(ingredient_compound, compound_gene, atlas)
    summary["n_compounds_ingredient_compound"] = cov.get("n_compounds_ingredient_compound", 0)
    summary["n_compounds_compound_gene"] = cov.get("n_compounds_compound_gene", 0)
    summary["n_overlap"] = cov.get("n_overlap", 0)
    summary["overlap_vs_cg"] = cov.get("overlap_vs_cg")
    summary["overlap_vs_ic"] = cov.get("overlap_vs_ic")
    summary["gate_passed"] = cov.get("gate_passed", False)
    summary["gate_reason"] = cov.get("gate_reason", "")
    summary["pct_compound_overlap"] = cov.get("pct_compound_overlap")
    summary["n_ingredients_reach_gene"] = cov.get("n_ingredients_reach_gene", 0)
    summary["pct_ingredients_reach_gene"] = cov.get("pct_ingredients_reach_gene")
    pct_overlap = cov.get("pct_compound_overlap")
    overlap_vs_cg = cov.get("overlap_vs_cg") or 0
    if pct_overlap is not None and pct_overlap < 20 and overlap_vs_cg < 0.20:
        logger.warning(
            "Compound overlap vs IC=%.1f%%, overlap_vs_cg=%.1f%% (informational). See reports/phase14_summary.json",
            pct_overlap, 100 * overlap_vs_cg,
        )
    # Recompute coverage with propagation_stats so summary gate reflects full readiness gate
    cov = compute_phase14_coverage_metrics(ingredient_compound, compound_gene, atlas, propagation_stats=prop_diag)
    summary["gate_passed"] = cov.get("gate_passed", False)
    summary["gate_reason"] = cov.get("gate_reason", "")
    summary["pct_rows_with_nonzero_propagation"] = cov.get("pct_rows_with_nonzero_propagation")
    summary["atlas_pair_cov_compound"] = cov.get("atlas_pair_cov_compound")
    summary["ic_gene_coverage"] = cov.get("ic_gene_coverage")

    write_mediation_artifacts(out_base, mediation_nodes, mediation_edges, None, pair_mediation, run_id)
    write_neo4j_export(out_base, mediation_nodes, mediation_edges, pair_mediation=pair_mediation, run_id=run_id)
    write_high_conf_export(out_base, pair_mediation)
    top = pair_mediation.nlargest(100, "mechanistic_score")
    write_reports(out_base, summary, top, run_id)

    if isinstance(prop_diag, dict):
        prop_diag["diagnosis_csv_path"] = "data/processed/canonical/reports/propagation_zero_row_diagnosis.csv"
        prop_diag["diagnosis_script"] = "python scripts/phase14/diagnose_propagation_zero_rows.py"
    write_propagation_diagnostics(out_base, prop_diag)
    saturation_pct = getattr(config, "MECHANISTIC_SCORE_SATURATION_WARN_PCT", 30.0)
    write_mechanistic_score_diagnostics(out_base, pair_mediation, saturation_warn_pct=saturation_pct)
    from src.phase14.loaders import _canonical_compound_gene_path, phase14_smoke_check_print
    cg_path = _canonical_compound_gene_path(paths)
    phase14_smoke_check_print(
        chosen_cg_file=cg_path.name if cg_path else None,
        n_unique_compounds_ic=cov.get("n_compounds_ingredient_compound"),
        n_unique_compounds_cg=cov.get("n_compounds_compound_gene"),
        n_overlap=cov.get("n_overlap"),
        pct_rows_with_nonzero_propagation=prop_diag.get("pct_rows_with_nonzero_propagation") if isinstance(prop_diag, dict) else None,
    )
    pct_prop = prop_diag.get("pct_rows_with_nonzero_propagation") if isinstance(prop_diag, dict) else None
    if not args.smoke and pct_prop is not None and pct_prop < 25.0:
        n_rows = prop_diag.get("n_rows") if isinstance(prop_diag, dict) else None
        n_nonzero = prop_diag.get("n_nonzero") if isinstance(prop_diag, dict) else None
        diag_csv = prop_diag.get("diagnosis_csv_path", "data/processed/canonical/reports/propagation_zero_row_diagnosis.csv") if isinstance(prop_diag, dict) else "data/processed/canonical/reports/propagation_zero_row_diagnosis.csv"
        diag_script = prop_diag.get("diagnosis_script", "python scripts/phase14/diagnose_propagation_zero_rows.py") if isinstance(prop_diag, dict) else "python scripts/phase14/diagnose_propagation_zero_rows.py"
        raise ValueError(
            "Propagation gate failed for FULL run: pct_rows_with_nonzero_propagation=%.1f%% (need >= 25%%). "
            "n_rows=%s n_nonzero=%s. See %s for per-row zero_reason counts; run: %s. "
            "Ensure compound_gene overlaps ingredient_compound (InChIKey) and category-bundle alignment (see phase14_config.CATEGORY_SYNONYMS_FOR_PROPAGATION)."
            % (pct_prop, n_rows, n_nonzero, diag_csv, diag_script)
        )
    from src.phase14.audit_metrics import (
        compute_phase14_coverage_audit,
        write_phase14_coverage_audit,
        build_top_missing_evidence,
        write_top_missing_evidence,
    )
    coverage_audit = compute_phase14_coverage_audit(
        ingredient_compound, compound_gene, mediation_edges, atlas, pair_mediation,
    )
    write_phase14_coverage_audit(out_base, coverage_audit)
    top_missing = build_top_missing_evidence(
        atlas, pair_mediation, ingredient_compound, compound_gene, mediation_edges, top_n=200,
    )
    write_top_missing_evidence(out_base, top_missing)

    overlap_compounds_df, identity_metrics = build_overlap_compounds_and_identity_metrics(
        ingredient_compound, compound_gene, metrics=cov,
    )
    write_identity_diagnostics(out_base, identity_metrics, overlap_compounds_df)

    # Single summary: overlap %, propagation nonzero %, and top reasons if low
    overlap_pct = (summary.get("overlap_vs_cg") or 0) * 100
    prop_pct = summary.get("pct_rows_with_nonzero_propagation") or prop_diag.get("pct_rows_with_nonzero_propagation") if isinstance(prop_diag, dict) else None
    logger.info(
        "Output: %s | n_pairs=%s n_nodes=%s n_edges=%s | ic=%s cg=%s overlap=%s overlap_vs_cg=%.2f%% | propagation_nonzero=%s",
        out_base, summary.get("n_pairs"), summary.get("n_nodes"), summary.get("n_edges"),
        summary.get("n_compounds_ingredient_compound"), summary.get("n_compounds_compound_gene"),
        summary.get("n_overlap"), overlap_pct, ("%.1f%%" % prop_pct) if prop_pct is not None else "N/A",
    )
    if overlap_pct < 5.0 or (prop_pct is not None and prop_pct < 25.0):
        reasons = []
        if overlap_pct < 5.0:
            reasons.append("low overlap_vs_cg (%.1f%%)" % overlap_pct)
        if prop_pct is not None and prop_pct < 25.0:
            reasons.append("low propagation nonzero (%.1f%%)" % prop_pct)
        logger.warning(
            "Phase14 summary: %s. To improve: run compound identity bridge and build_compound_gene_expanded_v4 (see data/processed/canonical/*.json).",
            "; ".join(reasons),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
