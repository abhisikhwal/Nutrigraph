"""
Calibrated max_nn_tanimoto -> confidence_weight mapping for predicted edges.

Derived from knn_similarity_validation.json per-band hit@10.
Similarity < 0.3 -> weight 0 (withheld).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_PATH = ROOT / "data/processed/thread2/recon/knn_similarity_validation.json"
DEFAULT_WEIGHT_SPEC_PATH = ROOT / "data/processed/thread2/inference/confidence_weight_spec.json"

CalibrationSplit = Literal["random_split", "scaffold_split"]

# Band label -> (lo, hi, midpoint) for calibration table
BAND_SPECS: list[tuple[str, float, float, float]] = [
    ("[0.3-0.4)", 0.3, 0.4, 0.35),
    ("[0.4-0.5)", 0.4, 0.5, 0.45),
    ("[0.5-0.6)", 0.5, 0.6, 0.55),
    ("[0.6-0.7)", 0.6, 0.7, 0.65),
    ("[0.7-0.8)", 0.7, 0.8, 0.75),
    ("[0.8-0.9)", 0.8, 0.9, 0.85),
    ("[0.9-1.0]", 0.9, 1.0, 0.95),
]

SIM_FLOOR = 0.3


def _band_points_from_table(
    per_band: dict[str, Any],
    split_used: CalibrationSplit,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for label, lo, hi, mid in BAND_SPECS:
        row = per_band[label]
        points.append(
            {
                "band": label,
                "similarity_lo": lo,
                "similarity_hi": hi,
                "similarity_midpoint": mid,
                "measured_hit_at_10": float(row["hit@10"]),
                "n_test_compounds": int(row["n"]),
            }
        )
    return points


def _proportional_scaffold_adjustment(
    random_points: list[dict[str, Any]],
    random_overall: float,
    scaffold_overall: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scale band hit@10 toward scaffold overall while preserving floor and monotonicity."""
    scale = scaffold_overall / random_overall if random_overall > 0 else 1.0
    adjusted: list[dict[str, Any]] = []
    for p in random_points:
        hit = float(p["measured_hit_at_10"]) * scale
        adjusted.append({**p, "measured_hit_at_10": hit})
    # Re-enforce monotonicity on adjusted hits before knot fit
    hits = np.array([p["measured_hit_at_10"] for p in adjusted], dtype=float)
    hits = _isotonic_non_decreasing(hits)
    for i, p in enumerate(adjusted):
        p["measured_hit_at_10"] = float(hits[i])
    meta = {
        "calibration_approach": "proportional_scaffold_scaling",
        "scale_factor": float(scale),
        "random_overall_hit_at_10": random_overall,
        "scaffold_overall_hit_at_10": scaffold_overall,
        "note": (
            "Per-band scaffold hit@10 unavailable; scaled random-split band table by "
            f"{scale:.4f} so implied overall matches scaffold ({scaffold_overall:.4f})."
        ),
    }
    return adjusted, meta


def load_calibration_points(
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    split_used: CalibrationSplit = "random_split",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract per-band hit@10 midpoints from validation JSON."""
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    random_split = data["random_split"]
    scaffold = data.get("scaffold_split", {})
    random_overall = float(random_split["overall_hit_at_10"])
    scaffold_overall = float(scaffold.get("overall_hit_at_10", random_overall))

    random_points = _band_points_from_table(random_split["per_band"], "random_split")
    scaffold_per_band = scaffold.get("per_band")
    approach_meta: dict[str, Any] = {}

    if split_used == "scaffold_split" and scaffold_per_band:
        points = _band_points_from_table(scaffold_per_band, "scaffold_split")
        approach_meta = {
            "calibration_approach": "direct_scaffold_per_band",
            "note": (
                "Per-band hit@10 from Murcko-scaffold split "
                f"({scaffold.get('test_n_evaluated', '?')} test compounds evaluated)."
            ),
        }
    elif split_used == "scaffold_split":
        points, approach_meta = _proportional_scaffold_adjustment(
            random_points, random_overall, scaffold_overall
        )
    else:
        points = random_points

    meta = {
        "source": str(calibration_path),
        "split_used": split_used,
        "scaffold_per_band_available": bool(scaffold_per_band),
        "scaffold_overall_hit_at_10": scaffold_overall,
        "random_overall_hit_at_10": random_overall,
        "scaffold_inflation_random_minus_scaffold": random_overall - scaffold_overall,
        **approach_meta,
    }
    if split_used == "random_split":
        meta["scaffold_inflation_note"] = (
            "Per-band scaffold hit@10 not used; random-split band table. "
            f"Overall scaffold hit@10 ({scaffold_overall}) is "
            f"~{random_overall - scaffold_overall:.1%} below random — band weights may be optimistic."
        )
    return points, meta


def _isotonic_non_decreasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators isotonic regression (non-decreasing)."""
    try:
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        x = np.arange(len(y), dtype=float)
        return ir.fit_transform(x, y)
    except ImportError:
        # Fallback: cumulative max (coarse but monotone)
        return np.maximum.accumulate(y)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_candidate_functions(
    midpoints: np.ndarray,
    hit_at_10: np.ndarray,
) -> dict[str, Any]:
    """Compare monotone piecewise-linear (isotonic-corrected) vs logistic."""
    y_iso = _isotonic_non_decreasing(hit_at_10)

    # Knots: floor at 0.3 -> 0, then band midpoints, then 1.0
    knots_x = np.concatenate([[SIM_FLOOR], midpoints, [1.0]])
    knots_y = np.concatenate([[0.0], y_iso, [max(float(y_iso[-1]), float(y_iso[-2]))]])

    def piecewise_linear(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.interp(x, knots_x, knots_y)
        out = np.where(x < SIM_FLOOR, 0.0, out)
        return np.clip(out, 0.0, 1.0)

    # Logistic on midpoints only (soft, may need monotone clip)
    logistic_info: dict[str, Any] = {"available": False}
    logistic_rmse = np.inf
    logistic_fn = None
    try:
        from scipy.optimize import curve_fit

        def logistic(x, l_cap, k, x0):
            return l_cap / (1.0 + np.exp(-k * (x - x0)))

        p0 = [1.0, 10.0, 0.55]
        bounds = ([0.5, 0.1, 0.3], [1.0, 50.0, 0.9])
        popt, _ = curve_fit(logistic, midpoints, hit_at_10, p0=p0, bounds=bounds, maxfev=10000)
        l_cap, k, x0 = popt

        def logistic_fn(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            raw = logistic(x, l_cap, k, x0)
            raw = np.maximum.accumulate(raw) if x.ndim == 0 else raw  # not valid for array
            out = logistic(x, l_cap, k, x0)
            out = np.where(x < SIM_FLOOR, 0.0, out)
            return np.clip(out, 0.0, 1.0)

        logistic_rmse = _rmse(hit_at_10, logistic(midpoints, *popt))
        logistic_info = {
            "available": True,
            "formula": "L / (1 + exp(-k * (x - x0))) with x0=similarity",
            "params": {"L_cap": float(l_cap), "k": float(k), "x0": float(x0)},
            "rmse_on_band_midpoints": logistic_rmse,
        }
    except Exception as exc:
        logistic_info["error"] = str(exc)

    pl_rmse = _rmse(hit_at_10, piecewise_linear(midpoints))

    chosen = "monotone_piecewise_linear"
    reason = (
        "Isotonic-corrected hit@10 at band midpoints, linearly interpolated between knots "
        f"(RMSE={pl_rmse:.4f} on midpoints). No parametric assumption; knots are transparent. "
        "Logistic "
        + (
            f"RMSE={logistic_rmse:.4f} — worse or less interpretable."
            if logistic_info.get("available")
            else "fit skipped/failed."
        )
    )

    return {
        "chosen_method": chosen,
        "chosen_reason": reason,
        "raw_hit_at_10": hit_at_10.tolist(),
        "isotonic_corrected_hit_at_10": y_iso.tolist(),
        "knots_similarity": knots_x.tolist(),
        "knots_weight": knots_y.tolist(),
        "piecewise_linear_rmse": pl_rmse,
        "logistic": logistic_info,
        "piecewise_linear": {
            "description": (
                "confidence_weight(sim) = 0 if sim < 0.3 else "
                "linear_interp(sim, knots_x, knots_y) clipped to [0,1]"
            ),
            "knots_x": knots_x.tolist(),
            "knots_y": knots_y.tolist(),
        },
        "apply_fn": piecewise_linear,
    }


class ConfidenceWeightFunction:
    """Reusable calibrated weight function."""

    def __init__(self, knots_x: np.ndarray, knots_y: np.ndarray, sim_floor: float = SIM_FLOOR):
        self.knots_x = np.asarray(knots_x, dtype=float)
        self.knots_y = np.asarray(knots_y, dtype=float)
        self.sim_floor = sim_floor
        self._spec_version = "v1"
        self._calibration_meta: dict[str, Any] = {}

    @classmethod
    def from_calibration(
        cls,
        calibration_path: Path = DEFAULT_CALIBRATION_PATH,
        split_used: CalibrationSplit = "random_split",
        spec_version: str = "v1",
    ) -> tuple[ConfidenceWeightFunction, dict[str, Any]]:
        points, meta = load_calibration_points(calibration_path, split_used=split_used)
        midpoints = np.array([p["similarity_midpoint"] for p in points], dtype=float)
        hit = np.array([p["measured_hit_at_10"] for p in points], dtype=float)
        fit = fit_candidate_functions(midpoints, hit)
        fn = cls(np.array(fit["knots_similarity"]), np.array(fit["knots_weight"]))
        fn._spec_version = spec_version
        fn._calibration_meta = meta
        report = {
            "calibration_points": points,
            "calibration_meta": meta,
            "fit_comparison": {k: v for k, v in fit.items() if k != "apply_fn"},
            "chosen_method": fit["chosen_method"],
            "chosen_reason": fit["chosen_reason"],
        }
        return fn, report

    def __call__(self, max_nn_tanimoto: float) -> float:
        return float(self.apply(np.array([max_nn_tanimoto]))[0])

    def apply(self, max_nn_tanimoto: np.ndarray) -> np.ndarray:
        x = np.asarray(max_nn_tanimoto, dtype=float)
        out = np.interp(x, self.knots_x, self.knots_y)
        out = np.where(x < self.sim_floor, 0.0, out)
        return np.clip(out, 0.0, 1.0)

    def to_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "version": getattr(self, "_spec_version", "v1"),
            "sim_floor": self.sim_floor,
            "method": "monotone_piecewise_linear",
            "knots_similarity": self.knots_x.tolist(),
            "knots_weight": self.knots_y.tolist(),
            "formula": (
                "weight=0 if max_nn_tanimoto < 0.3 else "
                "np.interp(max_nn_tanimoto, knots_similarity, knots_weight)"
            ),
        }
        if self._calibration_meta:
            spec["calibration"] = self._calibration_meta
        return spec

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> ConfidenceWeightFunction:
        return cls(
            np.array(spec["knots_similarity"]),
            np.array(spec["knots_weight"]),
            spec.get("sim_floor", SIM_FLOOR),
        )


def tier_sanity_check(
    fn: ConfidenceWeightFunction,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    split_used: CalibrationSplit = "random_split",
) -> dict[str, Any]:
    """Compare continuous weights to tier aggregate hit@10 guardrails."""
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if split_used == "scaffold_split" and data.get("scaffold_split", {}).get("tier_summary"):
        tier_stats = data["scaffold_split"]["tier_summary"]
        tier_hits = {
            "withhold": tier_stats["withhold_lt_0.3"]["hit@10"],
            "predicted_low": tier_stats["predicted_low_0.3_to_0.5"]["hit@10"],
            "predicted_moderate": tier_stats["predicted_moderate_0.5_to_0.7"]["hit@10"],
            "predicted_high": tier_stats["predicted_high_ge_0.7"]["hit@10"],
        }
    else:
        tiers = data["confidence_tier_recommendation"]["evidence_based_thresholds"]
        tier_hits = {
            "withhold": tiers["withhold"]["measured_hit_at_10"],
            "predicted_low": tiers["predicted_low"]["measured_hit_at_10"],
            "predicted_moderate": tiers["predicted_moderate"]["measured_hit_at_10"],
            "predicted_high": tiers["predicted_high"]["measured_hit_at_10"],
        }

    checks = []
    probe_points = [
        ("below_withhold", 0.25, 0.0, tier_hits["withhold"]),
        ("low_band_mid", 0.35, None, None),
        ("low_tier_mid", 0.40, None, tier_hits["predicted_low"]),
        ("operating_0.45", 0.45, None, None),
        ("operating_0.55", 0.55, None, None),
        ("moderate_tier_mid", 0.60, None, tier_hits["predicted_moderate"]),
        ("high_threshold", 0.70, None, tier_hits["predicted_high"]),
        ("high_mid", 0.75, None, None),
        ("very_high", 0.85, None, None),
        ("near_identity", 0.95, None, None),
        ("unity", 1.00, None, None),
    ]
    for name, sim, expect_exact, tier_hit in probe_points:
        w = fn(sim)
        entry = {"probe": name, "similarity": sim, "weight": w}
        if expect_exact is not None:
            entry["expected"] = expect_exact
            entry["ok"] = abs(w - expect_exact) < 1e-9
        if tier_hit is not None:
            entry["tier_aggregate_hit_at_10"] = tier_hit
            entry["delta_weight_minus_tier"] = w - tier_hit
        checks.append(entry)

    flags = []
    w70 = fn(0.70)
    if w70 < 0.95:
        flags.append(f"At sim=0.70 weight={w70:.3f} below expected ~0.99")
    w55 = fn(0.55)
    w45 = fn(0.45)
    if not (w45 <= w55):
        flags.append(f"Non-monotone: w(0.45)={w45:.3f} > w(0.55)={w55:.3f}")
    w35 = fn(0.35)
    if w35 > 0.85:
        flags.append(f"At sim=0.35 weight={w35:.3f} unusually high vs tier low ~0.69")

    return {"probe_weights": checks, "flags": flags, "monotonic_ok": len(flags) == 0}


def multi_evidence_options() -> dict[str, Any]:
    return {
        "options": [
            {
                "name": "max_weight",
                "formula": "gene_score(ingredient, g) = max(edge.confidence_weight for edges predicting g)",
                "pros": "Conservative; one strong neighbor dominates.",
                "cons": "Ignores corroboration from multiple moderate edges.",
            },
            {
                "name": "noisy_or",
                "formula": "gene_score = 1 - prod(1 - w_i) over edges predicting g",
                "pros": "Standard independent-evidence combine; more edges saturate toward 1.",
                "cons": "Can inflate quickly with many moderate edges.",
            },
            {
                "name": "sum_capped",
                "formula": "gene_score = min(1, sum(w_i)) over edges predicting g",
                "pros": "Simple, interpretable accumulation; cap prevents >1.",
                "cons": "Linear add assumes equal marginal value for each edge.",
            },
        ],
        "recommendation": (
            "noisy_or for enrichment aggregation: it treats each compound-level prediction as "
            "independent structural evidence, rewards corroboration without unbounded linear "
            "inflation, and stays in [0,1]. Use max_weight as a sensitivity bound."
        ),
    }
