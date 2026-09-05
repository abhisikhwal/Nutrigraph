"""
Phase14 readiness: decide whether to rerun Phase14 (FULL) based on
input file hashes, key metrics, and presence of outputs.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def hash_file(path: Path, chunk: int = 65536) -> str:
    """Compute SHA256 of file. Raises FileNotFoundError if missing."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_hash(path: Path) -> Optional[str]:
    """Return SHA256 or None if file missing."""
    try:
        return hash_file(path)
    except Exception:
        return None


def _discover_phase13_dir(repo_root: Path) -> Optional[Path]:
    """Find phase13 dir with atlas_confirmed.csv and bootstrap_stability.csv."""
    processed = repo_root / "data" / "processed"
    if not processed.exists():
        return None
    for d in sorted(processed.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and "phase13" in d.name.lower():
            if (d / "atlas_confirmed.csv").exists() and (d / "bootstrap_stability.csv").exists():
                return d
    return None


def _get_compound_gene_source_path(repo_root: Path, run_dir: Optional[Path]) -> Optional[Path]:
    """Resolve compound_gene source: from run's phase14_summary, or canonical/reports, or fallback."""
    if run_dir and (run_dir / "reports" / "phase14_summary.json").exists():
        try:
            with open(run_dir / "reports" / "phase14_summary.json", "r", encoding="utf-8") as f:
                s = json.load(f)
            raw = s.get("compound_gene_source") or ""
            if raw:
                p = Path(raw)
                if p.is_absolute() and p.exists():
                    return p
                rel = str(raw).replace("\\", "/")
                if not rel.startswith("/"):
                    cand = repo_root / rel
                    if cand.exists():
                        return cand
        except Exception:
            pass
    metrics_path = repo_root / "data" / "processed" / "canonical" / "reports" / "phase14_compound_gene_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            src = m.get("source_file")
            if src:
                cand = repo_root / "data" / "processed" / "canonical" / src
                if cand.exists():
                    return cand
        except Exception:
            pass
    for name in ["compound_gene_expanded_canonical.csv", "compound_gene_canonical.csv"]:
        cand = repo_root / "data" / "processed" / "canonical" / name
        if cand.exists():
            return cand
    return None


def load_latest_run_dir(
    base_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Return the most recent Phase14 run directory under base_dir.
    Prefer timestamp in directory name (phase14_YYYYMMDD_HHMMSS), else mtime.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    if base_dir is None:
        base_dir = repo_root / "data" / "processed" / "phase14_mediation"
    base_dir = Path(base_dir).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        return None
    runs: List[Tuple[float, Path]] = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        if not (d / "reports" / "phase14_summary.json").exists():
            continue
        # Prefer name-based timestamp (phase14_20260219_191353)
        try:
            parts = d.name.split("_")
            if len(parts) >= 3 and parts[0].lower() == "phase14":
                date_part = parts[1]
                time_part = parts[2]
                if len(date_part) == 8 and len(time_part) == 6:
                    ts = float(f"{date_part}{time_part}")
                    runs.append((ts, d))
                    continue
        except Exception:
            pass
        runs.append((d.stat().st_mtime, d))
    if not runs:
        return None
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0][1]


def compute_current_input_fingerprints(repo_root: Optional[Path] = None) -> Dict[str, str]:
    """
    Compute SHA256 for canonical inputs currently used (same set as snapshot).
    Returns dict key -> sha256 (only keys for existing files).
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root).resolve()
    out: Dict[str, str] = {}

    ic = repo_root / "data" / "processed" / "canonical" / "ingredient_compound_canonical.csv"
    if ic.exists():
        out["ingredient_compound_canonical"] = hash_file(ic)

    cg = _get_compound_gene_source_path(repo_root, None)
    if cg and cg.exists():
        out["compound_gene_source"] = hash_file(cg)

    pb = repo_root / "data" / "processed" / "features" / "pathway_bundles.json"
    if pb.exists():
        out["pathway_bundles"] = hash_file(pb)

    p13 = _discover_phase13_dir(repo_root)
    if p13:
        for fname, key in [("atlas_confirmed.csv", "atlas_confirmed"), ("bootstrap_stability.csv", "bootstrap_stability")]:
            f = p13 / fname
            if f.exists():
                out[key] = hash_file(f)

    return out


def load_run_fingerprints(run_dir: Path, repo_root: Optional[Path] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Load input hashes and key metrics for a run.
    If run_dir contains MANIFEST.json (e.g. milestone snapshot), use input_hashes from it
    and metrics from run reports. Else compute input hashes from run's reported paths and load metrics from reports.
    Returns (input_hashes dict key->sha256, metrics dict).
    """
    run_dir = Path(run_dir).resolve()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root).resolve()
    input_hashes: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}

    manifest_path = run_dir / "MANIFEST.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            ih = manifest.get("input_hashes") or {}
            for k, v in ih.items():
                if isinstance(v, dict) and v.get("sha256"):
                    input_hashes[k] = v["sha256"]
                elif isinstance(v, str):
                    input_hashes[k] = v
        except Exception as e:
            logger.warning("Could not read MANIFEST.json: %s", e)

    if not input_hashes:
        # Compute from current canonical paths (same as snapshot)
        ic = repo_root / "data" / "processed" / "canonical" / "ingredient_compound_canonical.csv"
        if ic.exists():
            input_hashes["ingredient_compound_canonical"] = hash_file(ic)
        cg = _get_compound_gene_source_path(repo_root, run_dir)
        if cg and cg.exists():
            input_hashes["compound_gene_source"] = hash_file(cg)
        pb = repo_root / "data" / "processed" / "features" / "pathway_bundles.json"
        if pb.exists():
            input_hashes["pathway_bundles"] = hash_file(pb)
        p13 = _discover_phase13_dir(repo_root)
        if p13:
            for fname, key in [("atlas_confirmed.csv", "atlas_confirmed"), ("bootstrap_stability.csv", "bootstrap_stability")]:
                f = p13 / fname
                if f.exists():
                    input_hashes[key] = hash_file(f)

    # Load metrics from run reports
    for name, key in [
        ("reports/phase14_summary.json", "summary"),
        ("reports/propagation_diagnostics.json", "propagation"),
    ]:
        p = run_dir / name
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if key == "summary":
                    metrics.update(data)
                else:
                    metrics["pct_rows_with_nonzero_propagation"] = data.get("pct_rows_with_nonzero_propagation")
                    metrics["n_nonzero"] = data.get("n_nonzero")
            except Exception:
                pass

    return input_hashes, metrics


def should_rerun_phase14(
    current_inputs: Dict[str, str],
    last_run_inputs: Dict[str, str],
    last_metrics: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Gate: should we rerun Phase14 (FULL)?
    - No run exists => YES
    - Any canonical input hash changed => YES
    - Last run pct_rows_with_nonzero_propagation < 25% => YES
    - Else => NO
    Returns (rerun: bool, reasons: list of strings).
    """
    reasons: List[str] = []

    if not last_run_inputs and not last_metrics:
        reasons.append("No previous run found")
        return True, reasons

    for key, current_sha in current_inputs.items():
        last_sha = last_run_inputs.get(key)
        if last_sha is None:
            reasons.append(f"New input not in last run: {key}")
            return True, reasons
        if current_sha != last_sha:
            reasons.append(f"Input hash changed: {key}")
            return True, reasons

    pct = last_metrics.get("pct_rows_with_nonzero_propagation")
    if pct is not None:
        try:
            if float(pct) < 25.0:
                reasons.append(f"Last run propagation nonzero % ({pct}) < 25%")
                return True, reasons
        except (TypeError, ValueError):
            pass

    reasons.append("Inputs unchanged and last run propagation acceptable")
    return False, reasons


def print_readiness_dashboard(
    current_inputs: Dict[str, str],
    last_run_dir: Optional[Path],
    last_run_inputs: Dict[str, str],
    last_metrics: Dict[str, Any],
    rerun: bool,
    reasons: List[str],
) -> None:
    """Print a dashboard-style summary and RERUN_PHASE14 line."""
    print("=" * 60)
    print("Phase14 Readiness Dashboard")
    print("=" * 60)
    print("Current input fingerprints (SHA256):")
    for k, v in sorted(current_inputs.items()):
        print(f"  {k}: {v[:16]}...")
    print()
    if last_run_dir:
        print(f"Latest run dir: {last_run_dir}")
        print("Last run input hashes:")
        for k, v in sorted(last_run_inputs.items()):
            match = "OK" if current_inputs.get(k) == v else "CHANGED"
            print(f"  {k}: {v[:16] if v else 'N/A'}... [{match}]")
        print("Last run key metrics:")
        print(f"  pct_rows_with_nonzero_propagation: {last_metrics.get('pct_rows_with_nonzero_propagation', 'N/A')}")
        print(f"  overlap_vs_cg: {last_metrics.get('overlap_vs_cg', 'N/A')}")
        print(f"  n_overlap: {last_metrics.get('n_overlap', 'N/A')}")
        print(f"  n_pairs: {last_metrics.get('n_pairs', 'N/A')}")
    else:
        print("No previous Phase14 run found.")
    print()
    print("Reasons:", "; ".join(reasons))
    print()
    result = "YES" if rerun else "NO"
    print(f"RERUN_PHASE14: {result}")
    print("=" * 60)


def run_readiness(
    repo_root: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Run full readiness check. Returns (should_rerun, reasons, payload for JSON).
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root).resolve()
    if base_dir is None:
        base_dir = repo_root / "data" / "processed" / "phase14_mediation"

    current_inputs = compute_current_input_fingerprints(repo_root)
    last_run_dir = load_latest_run_dir(base_dir=base_dir, repo_root=repo_root)
    last_run_inputs: Dict[str, str] = {}
    last_metrics: Dict[str, Any] = {}
    if last_run_dir:
        last_run_inputs, last_metrics = load_run_fingerprints(last_run_dir, repo_root)

    rerun, reasons = should_rerun_phase14(current_inputs, last_run_inputs, last_metrics)
    payload = {
        "rerun_phase14": rerun,
        "reasons": reasons,
        "current_input_fingerprints": current_inputs,
        "last_run_dir": str(last_run_dir) if last_run_dir else None,
        "last_run_input_fingerprints": last_run_inputs,
        "last_run_metrics": last_metrics,
    }
    return rerun, reasons, payload
