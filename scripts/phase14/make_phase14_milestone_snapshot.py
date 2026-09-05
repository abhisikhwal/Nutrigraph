"""
Phase14 milestone snapshot: freeze a successful run and its inputs.
Validates run-dir, computes SHA256 for canonical inputs and run outputs,
copies to data/processed/milestones/phase14/<tag>/<run_id>/ with MANIFEST.json and SUMMARY.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Key files that must exist in run-dir (or we fail validation)
REQUIRED_REPORTS = ["phase14_summary.json", "propagation_diagnostics.json"]
REQUIRED_TOP_LEVEL = ["high_conf_mechanistic.csv"]
# At least one of pair_mediation / pair_category_mediation
PAIR_MEDIATION_NAMES = ["pair_category_mediation.csv", "pair_mediation.csv"]
NEO4J_NAMES = ["nodes.csv", "edges.csv"]


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


def validate_run_dir(run_dir: Path) -> Tuple[bool, List[str]]:
    """Check run-dir has reports/, neo4j/, and key files. Return (ok, list of missing)."""
    errors: List[str] = []
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        return False, [f"run-dir not found or not a directory: {run_dir}"]

    reports_dir = run_dir / "reports"
    neo_dir = run_dir / "neo4j"
    if not reports_dir.is_dir():
        errors.append("reports/ missing")
    if not neo_dir.is_dir():
        errors.append("neo4j/ missing")

    for name in REQUIRED_REPORTS:
        if not (reports_dir / name).exists():
            errors.append(f"reports/{name} missing")
    for name in REQUIRED_TOP_LEVEL:
        if not (run_dir / name).exists():
            errors.append(f"{name} missing")

    pair_found = any((run_dir / n).exists() for n in PAIR_MEDIATION_NAMES)
    if not pair_found:
        errors.append("neither pair_category_mediation.csv nor pair_mediation.csv found")

    for name in NEO4J_NAMES:
        if not (neo_dir / name).exists():
            errors.append(f"neo4j/{name} missing")

    return len(errors) == 0, errors


def discover_phase13_dir(repo_root: Path) -> Optional[Path]:
    """Find phase13 dir with atlas_confirmed.csv and bootstrap_stability.csv."""
    processed = repo_root / "data" / "processed"
    if not processed.exists():
        return None
    for d in sorted(processed.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and "phase13" in d.name.lower():
            if (d / "atlas_confirmed.csv").exists() and (d / "bootstrap_stability.csv").exists():
                return d
    return None


def get_compound_gene_source_path(repo_root: Path, run_dir: Path) -> Optional[Path]:
    """Resolve compound_gene source file from run reports (phase14_summary or compound_gene_metrics)."""
    summary_path = run_dir / "reports" / "phase14_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            raw = s.get("compound_gene_source") or ""
            if raw:
                p = Path(raw)
                if p.is_absolute() and p.exists():
                    return p
                # relative to repo
                rel = str(raw).replace("\\", "/")
                if not rel.startswith("/"):
                    cand = repo_root / rel
                    if cand.exists():
                        return cand
        except Exception as e:
            logger.warning("Could not read compound_gene_source from summary: %s", e)
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
    fallback = repo_root / "data" / "processed" / "canonical" / "compound_gene_expanded_canonical.csv"
    if fallback.exists():
        return fallback
    return None


def compute_input_hashes(repo_root: Path, run_dir: Path, include_large: bool) -> Dict[str, Dict[str, Any]]:
    """Build input_hashes dict: key -> {path, sha256, size_bytes} for canonical inputs."""
    repo_root = repo_root.resolve()
    run_dir = run_dir.resolve()
    out: Dict[str, Dict[str, Any]] = {}

    # ingredient_compound_canonical
    ic = repo_root / "data" / "processed" / "canonical" / "ingredient_compound_canonical.csv"
    if ic.exists():
        out["ingredient_compound_canonical"] = {
            "path": str(ic.relative_to(repo_root)) if repo_root in ic.resolve().parents else str(ic),
            "sha256": safe_hash(ic),
            "size_bytes": ic.stat().st_size,
        }

    # compound_gene source
    cg = get_compound_gene_source_path(repo_root, run_dir)
    if cg and cg.exists():
        try:
            rel = str(cg.relative_to(repo_root)) if repo_root in cg.resolve().parents else str(cg)
        except ValueError:
            rel = str(cg)
        out["compound_gene_source"] = {
            "path": rel,
            "sha256": safe_hash(cg),
            "size_bytes": cg.stat().st_size,
        }

    # pathway_bundles
    pb = repo_root / "data" / "processed" / "features" / "pathway_bundles.json"
    if pb.exists():
        out["pathway_bundles"] = {
            "path": str(pb.relative_to(repo_root)),
            "sha256": safe_hash(pb),
            "size_bytes": pb.stat().st_size,
        }

    # atlas_confirmed + bootstrap_stability
    p13 = discover_phase13_dir(repo_root)
    if p13:
        for name, key in [("atlas_confirmed.csv", "atlas_confirmed"), ("bootstrap_stability.csv", "bootstrap_stability")]:
            f = p13 / name
            if f.exists():
                out[key] = {
                    "path": str(f.relative_to(repo_root)),
                    "sha256": safe_hash(f),
                    "size_bytes": f.stat().st_size,
                }

    return out


def copy_file_and_record(
    src: Path,
    dest_dir: Path,
    dest_name: str,
    manifest_entries: List[Dict],
    repo_root: Path,
) -> None:
    """Copy src to dest_dir/dest_name and append to manifest_entries with size and sha256."""
    if not src.exists():
        manifest_entries.append({"source": str(src), "dest": dest_name, "error": "missing"})
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dest_name
    shutil.copy2(src, dest)
    size = dest.stat().st_size
    sha = hash_file(dest)
    rel = str(src.relative_to(repo_root)) if repo_root in src.resolve().parents else str(src)
    manifest_entries.append({"source": rel, "dest": dest_name, "size_bytes": size, "sha256": sha})


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase14 milestone snapshot")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to phase14 run directory")
    parser.add_argument("--tag", type=str, required=True, help="Human tag e.g. v1_working_2026-02-19")
    parser.add_argument("--include-large", action="store_true", help="Include large canonical parquets; default only key CSVs/JSON")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned actions only")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    run_dir = (repo / args.run_dir.replace("\\", "/")).resolve()
    tag = args.tag.strip()
    if not tag:
        logger.error("--tag must be non-empty")
        return 1

    ok, missing = validate_run_dir(run_dir)
    if not ok:
        logger.error("Run-dir validation failed: %s", missing)
        return 1
    logger.info("Run-dir validated: %s", run_dir)

    run_id = run_dir.name
    snapshot_base = repo / "data" / "processed" / "milestones" / "phase14" / tag / run_id
    if args.dry_run:
        logger.info("DRY RUN: would create %s", snapshot_base)
        input_hashes = compute_input_hashes(repo, run_dir, args.include_large)
        logger.info("Input hashes: %s", list(input_hashes.keys()))
        return 0

    snapshot_base.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()

    # Copy reports/
    reports_src = run_dir / "reports"
    reports_dest = snapshot_base / "reports"
    if reports_src.is_dir():
        if reports_dest.exists():
            shutil.rmtree(reports_dest)
        shutil.copytree(reports_src, reports_dest)
        for f in reports_dest.rglob("*"):
            if f.is_file():
                rel = f.relative_to(snapshot_base)
                manifest_entries.append({
                    "source": str(run_dir / "reports" / f.relative_to(reports_dest)),
                    "dest": str(rel).replace("\\", "/"),
                    "size_bytes": f.stat().st_size,
                    "sha256": hash_file(f),
                })
        logger.info("Copied reports/ -> %s", reports_dest)

    # Copy neo4j/
    neo_src = run_dir / "neo4j"
    neo_dest = snapshot_base / "neo4j"
    if neo_src.is_dir():
        if neo_dest.exists():
            shutil.rmtree(neo_dest)
        shutil.copytree(neo_src, neo_dest)
        for f in neo_dest.rglob("*"):
            if f.is_file():
                rel = f.relative_to(snapshot_base)
                manifest_entries.append({
                    "source": str(run_dir / "neo4j" / f.relative_to(neo_dest)),
                    "dest": str(rel).replace("\\", "/"),
                    "size_bytes": f.stat().st_size,
                    "sha256": hash_file(f),
                })
        logger.info("Copied neo4j/ -> %s", neo_dest)

    # Copy high_conf_mechanistic.csv and pair mediation
    for name in REQUIRED_TOP_LEVEL:
        src = run_dir / name
        if src.exists():
            copy_file_and_record(src, snapshot_base, name, manifest_entries, repo)
    for name in PAIR_MEDIATION_NAMES:
        src = run_dir / name
        if src.exists():
            copy_file_and_record(src, snapshot_base, name, manifest_entries, repo)
            break

    # Input hashes and small inputs/ folder
    input_hashes = compute_input_hashes(repo, run_dir, args.include_large)
    inputs_dest = snapshot_base / "inputs"
    inputs_dest.mkdir(parents=True, exist_ok=True)
    size_limit_skip = 500_000_000  # 500 MB - skip copying if larger unless include_large
    for key, info in input_hashes.items():
        path_str = info.get("path", "")
        if path_str.startswith("data"):
            src = repo / path_str
        else:
            src = Path(path_str) if Path(path_str).is_absolute() else repo / path_str
        if src.exists() and info.get("sha256"):
            size = info.get("size_bytes", 0)
            if size <= size_limit_skip or args.include_large:
                dest_name = src.name
                copy_file_and_record(src, inputs_dest, dest_name, manifest_entries, repo)
            else:
                manifest_entries.append({
                    "source": path_str,
                    "dest": f"inputs/{src.name} (skipped, size={size})",
                    "size_bytes": size,
                    "sha256": info["sha256"],
                    "skipped_large": True,
                })
    logger.info("Input hashes: %s", list(input_hashes.keys()))

    # MANIFEST.json
    manifest = {
        "tag": tag,
        "run_id": run_id,
        "created_at": created_at,
        "run_dir": str(run_dir),
        "milestone_dir": str(snapshot_base),
        "input_hashes": input_hashes,
        "files": manifest_entries,
    }
    manifest_path = snapshot_base / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote %s", manifest_path)

    # SUMMARY.md from reports
    summary_metrics: Dict[str, Any] = {}
    try:
        with open(run_dir / "reports" / "phase14_summary.json", "r", encoding="utf-8") as f:
            summary_metrics = json.load(f)
    except Exception:
        pass
    try:
        with open(run_dir / "reports" / "propagation_diagnostics.json", "r", encoding="utf-8") as f:
            prop = json.load(f)
        summary_metrics["pct_rows_with_nonzero_propagation"] = prop.get("pct_rows_with_nonzero_propagation")
        summary_metrics["n_nonzero"] = prop.get("n_nonzero")
    except Exception:
        pass
    mech_path = run_dir / "high_conf_mechanistic.csv"
    if mech_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(mech_path, nrows=100000)
            if "mechanistic_score" in df.columns:
                m = df["mechanistic_score"].fillna(0)
                summary_metrics["share_mechanistic_gt_07"] = round(100.0 * (m > 0.7).mean(), 2)
        except Exception:
            pass

    summary_lines = [
        "# Phase14 Milestone Snapshot",
        f"- **tag**: {tag}",
        f"- **run_id**: {run_id}",
        f"- **created_at**: {created_at}",
        "",
        "## Key metrics",
        f"- pct_rows_with_nonzero_propagation: {summary_metrics.get('pct_rows_with_nonzero_propagation', 'N/A')}",
        f"- overlap_vs_cg: {summary_metrics.get('overlap_vs_cg', 'N/A')}",
        f"- n_overlap: {summary_metrics.get('n_overlap', summary_metrics.get('n_compounds_ingredient_compound', 'N/A'))}",
        f"- share mechanistic_score > 0.7: {summary_metrics.get('share_mechanistic_gt_07', 'N/A')}%",
        f"- n_nodes: {summary_metrics.get('n_nodes', 'N/A')}",
        f"- n_edges: {summary_metrics.get('n_edges', 'N/A')}",
        f"- n_pairs: {summary_metrics.get('n_pairs', 'N/A')}",
    ]
    summary_path = snapshot_base / "SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    logger.info("Wrote %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
