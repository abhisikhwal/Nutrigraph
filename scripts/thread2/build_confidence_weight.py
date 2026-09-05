#!/usr/bin/env python3
"""
Derive calibrated max_nn_tanimoto -> confidence_weight from k-NN validation bands.

RECON + BUILD — produces weight spec, validation plot, weighted edge file.
Does NOT integrate into graph or canonical files.

Usage (from repo root):
    python scripts/thread2/build_confidence_weight.py

Outputs:
    data/processed/thread2/inference/confidence_weight_spec.json
    data/processed/thread2/inference/confidence_weight_report.json
    data/processed/thread2/inference/confidence_weight_curve.png
    data/processed/thread2/inference/predicted_compound_gene_weighted_v1.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from confidence_weight import (  # noqa: E402
    BAND_SPECS,
    ConfidenceWeightFunction,
    DEFAULT_CALIBRATION_PATH,
    multi_evidence_options,
    tier_sanity_check,
)

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = ROOT / "data/processed/thread2/inference"
EDGES_IN = INFERENCE_DIR / "predicted_compound_gene_v1.parquet"
EDGES_OUT = INFERENCE_DIR / "predicted_compound_gene_weighted_v1.parquet"
SPEC_OUT = INFERENCE_DIR / "confidence_weight_spec.json"
REPORT_OUT = INFERENCE_DIR / "confidence_weight_report.json"
PLOT_OUT = INFERENCE_DIR / "confidence_weight_curve.svg"


def write_validation_plot(
    fn: ConfidenceWeightFunction,
    calibration_points: list[dict],
    fit_comparison: dict,
    out_path: Path,
) -> None:
    """Write validation curve as SVG (no matplotlib dependency)."""
    mid_x = [p["similarity_midpoint"] for p in calibration_points]
    mid_y = [p["measured_hit_at_10"] for p in calibration_points]
    iso_y = fit_comparison["isotonic_corrected_hit_at_10"]

    grid = np.linspace(0.0, 1.0, 200)
    curve = fn.apply(grid)

    w, h = 720, 440
    ml, mr, mt, mb = 70, 30, 40, 60
    pw, ph = w - ml - mr, h - mt - mb

    def sx(x: float) -> float:
        return ml + x * pw

    def sy(y: float) -> float:
        return mt + (1.0 - y) * ph

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'<rect width="{100}%" height="{100}%" fill="white"/>',
        f'<text x="{w/2:.0f}" y="24" text-anchor="middle" font-size="14" font-family="sans-serif">'
        "Calibrated similarity → confidence weight (hit@10)</text>",
        f'<text x="{w/2:.0f}" y="42" text-anchor="middle" font-size="11" fill="#555" font-family="sans-serif">'
        "random-split bands; sim &lt; 0.3 → weight 0</text>",
    ]
    # tier bands
    for lo, hi, fill in [(0.3, 0.5, "#f5f5f5"), (0.5, 0.7, "#ececec"), (0.7, 1.0, "#e3e3e3")]:
        lines.append(
            f'<rect x="{sx(lo):.1f}" y="{sy(1):.1f}" width="{sx(hi)-sx(lo):.1f}" '
            f'height="{sy(0)-sy(1):.1f}" fill="{fill}"/>'
        )
    # axes
    lines.append(f'<line x1="{ml}" y1="{sy(0):.1f}" x2="{w-mr}" y2="{sy(0):.1f}" stroke="#333"/>')
    lines.append(f'<line x1="{ml}" y1="{sy(0):.1f}" x2="{ml}" y2="{sy(1):.1f}" stroke="#333"/>')
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        lines.append(f'<text x="{sx(t):.0f}" y="{h-20}" text-anchor="middle" font-size="10" font-family="sans-serif">{t:.1f}</text>')
        lines.append(f'<line x1="{sx(t):.0f}" y1="{sy(0):.0f}" x2="{sx(t):.0f}" y2="{sy(0)+4:.0f}" stroke="#333"/>')
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        lines.append(f'<text x="{ml-8}" y="{sy(t)+4:.0f}" text-anchor="end" font-size="10" font-family="sans-serif">{t:.1f}</text>')
    # floor line
    lines.append(f'<line x1="{sx(0.3):.1f}" y1="{sy(0):.1f}" x2="{sx(0.3):.1f}" y2="{sy(1):.1f}" stroke="#999" stroke-dasharray="4,4"/>')
    # curve
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(grid, curve))
    lines.append(f'<polyline fill="none" stroke="#2ca02c" stroke-width="2.5" points="{pts}"/>')
    # measured points
    for x, y in zip(mid_x, mid_y):
        lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="#1f77b4"/>')
    for x, y in zip(mid_x, iso_y):
        lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="none" stroke="#ff7f0e" stroke-width="2"/>')
    lines.append(f'<text x="{ml}" y="{h-4}" font-size="10" font-family="sans-serif">● measured  ○ isotonic  — weight curve</text>')
    lines.append("</svg>")

    out_path = out_path.with_suffix(".svg") if out_path.suffix == ".png" else out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def weight_distribution_summary(weights: np.ndarray) -> dict:
    qs = np.quantile(weights, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "n": int(len(weights)),
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "quantiles": {
            "p0": float(qs[0]),
            "p10": float(qs[1]),
            "p25": float(qs[2]),
            "p50": float(qs[3]),
            "p75": float(qs[4]),
            "p90": float(qs[5]),
            "p100": float(qs[6]),
        },
    }


def main() -> int:
    print("=== Confidence weight calibration (RECON + BUILD) ===", flush=True)

    fn, build_report = ConfidenceWeightFunction.from_calibration(DEFAULT_CALIBRATION_PATH)
    fit = build_report["fit_comparison"]

    print("\n--- Calibration points (random split) ---", flush=True)
    for p in build_report["calibration_points"]:
        print(
            f"  {p['band']:12s} mid={p['similarity_midpoint']:.2f}  "
            f"hit@10={p['measured_hit_at_10']:.4f}  n={p['n_test_compounds']:,}",
            flush=True,
        )
    print(f"\n  Caveat: {build_report['calibration_meta']['scaffold_inflation_note']}", flush=True)

    print(f"\n--- Fit comparison ---", flush=True)
    print(f"  Chosen: {build_report['chosen_method']}", flush=True)
    print(f"  Reason: {build_report['chosen_reason']}", flush=True)
    print(f"  Piecewise-linear RMSE on band midpoints: {fit['piecewise_linear_rmse']:.4f}", flush=True)
    if fit.get("logistic", {}).get("available"):
        print(f"  Logistic RMSE: {fit['logistic']['rmse_on_band_midpoints']:.4f}", flush=True)

    print("\n--- Chosen function (knots) ---", flush=True)
    spec = fn.to_spec()
    for x, y in zip(spec["knots_similarity"], spec["knots_weight"]):
        print(f"  sim={x:.2f} -> weight={y:.4f}", flush=True)

    sanity = tier_sanity_check(fn)
    print("\n--- Tier sanity check ---", flush=True)
    for row in sanity["probe_weights"]:
        extra = ""
        if "tier_aggregate_hit_at_10" in row:
            extra = f"  tier_hit@10={row['tier_aggregate_hit_at_10']:.3f}  delta={row['delta_weight_minus_tier']:+.3f}"
        print(f"  {row['probe']:18s} sim={row['similarity']:.2f}  weight={row['weight']:.4f}{extra}", flush=True)
    if sanity["flags"]:
        print("  FLAGS:", flush=True)
        for f in sanity["flags"]:
            print(f"    - {f}", flush=True)
    else:
        print("  No tier/curve contradictions flagged.", flush=True)

    # Apply to predicted edges
    print(f"\n--- Applying to {EDGES_IN} ---", flush=True)
    if not EDGES_IN.exists():
        print(f"ERROR: missing {EDGES_IN}", file=sys.stderr)
        return 1
    edges = pd.read_parquet(EDGES_IN)
    edges["confidence_weight"] = fn.apply(edges["max_nn_tanimoto"].to_numpy())
    edges.to_parquet(EDGES_OUT, index=False)
    print(f"Wrote {EDGES_OUT} ({len(edges):,} rows)", flush=True)

    all_w = edges["confidence_weight"].to_numpy()
    dist_all = weight_distribution_summary(all_w)
    dist_distinctive = weight_distribution_summary(
        edges.loc[edges["is_distinctive"], "confidence_weight"].to_numpy()
    )

    print("\n--- Weight distribution (all edges) ---", flush=True)
    print(json.dumps(dist_all, indent=2), flush=True)
    print("\n--- Weight distribution (distinctive edges only) ---", flush=True)
    print(json.dumps(dist_distinctive, indent=2), flush=True)

    multi = multi_evidence_options()

    write_validation_plot(fn, build_report["calibration_points"], fit, PLOT_OUT)
    print(f"\nWrote validation plot: {PLOT_OUT}", flush=True)

    report = {
        "calibration_points": build_report["calibration_points"],
        "calibration_meta": build_report["calibration_meta"],
        "fit_comparison": fit,
        "chosen_function": spec,
        "tier_sanity_check": sanity,
        "weight_distribution": {
            "all_edges": dist_all,
            "distinctive_edges": dist_distinctive,
        },
        "multi_evidence_combination": multi,
        "outputs": {
            "weighted_edges": str(EDGES_OUT),
            "weight_spec": str(SPEC_OUT),
            "validation_plot": str(PLOT_OUT),
        },
    }
    SPEC_OUT.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {SPEC_OUT}", flush=True)
    print(f"Wrote {REPORT_OUT}", flush=True)

    print("\n--- Multi-evidence combination (next step; not implemented) ---", flush=True)
    print(f"  Recommendation: {multi['recommendation']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
