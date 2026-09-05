"""
Phase15 master runner. Runs (1) build_training_graph, (2) node2vec, (3) pykeen, (4) gnn, (5) validate, (6) causal.
Writes all outputs under data/processed/phase15_embeddings/<timestamp>/.
Use --run-dir to override Phase14 source (default: snapshot).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def default_phase14_dir(repo_root: Path) -> Path:
    snapshot = repo_root / "data" / "processed" / "milestones" / "phase14" / "v1_working_2026-02-19" / "phase14_20260219_204918"
    if snapshot.exists():
        return snapshot
    return repo_root / "data" / "processed" / "phase14_mediation" / "phase14_20260219_204918"


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> int:
    cwd = cwd or REPO_ROOT
    logger.info("Run: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, shell=(sys.platform == "win32")).returncode


def run_check_env(repo_root: Path) -> int:
    """Run check_env.py; return 0 if all deps OK, else 1."""
    return run_cmd([
        sys.executable,
        str(repo_root / "scripts" / "phase15" / "check_env.py"),
    ], cwd=repo_root)


def check_dependencies() -> Tuple[bool, bool, bool]:
    """Return (node2vec_ok, pykeen_ok, gnn_ok)."""
    node2vec_ok = False
    try:
        from node2vec import Node2Vec
        node2vec_ok = True
    except ImportError:
        pass
    pykeen_ok = False
    try:
        import pykeen
        pykeen_ok = True
    except ImportError:
        pass
    gnn_ok = False
    try:
        import torch
        import torch_geometric
        gnn_ok = True
    except ImportError:
        pass
    return node2vec_ok, pykeen_ok, gnn_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase15 master runner")
    parser.add_argument("--run-dir", type=str, default=None, help="Phase14 run/snapshot dir; default snapshot")
    parser.add_argument("--out-dir", type=str, default=None, help="Output dir; default phase15_embeddings/<timestamp>")
    parser.add_argument("--skip-node2vec", action="store_true")
    parser.add_argument("--skip-pykeen", action="store_true")
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--skip-causal", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    phase14_dir = (repo_root / args.run_dir.replace("\\", "/")).resolve() if args.run_dir else default_phase14_dir(repo_root)
    if not phase14_dir.exists():
        logger.error("Phase14 dir not found: %s", phase14_dir)
        return 1
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = repo_root / "data" / "processed" / "phase15_embeddings" / f"phase15_{stamp}"
    else:
        out_dir = (repo_root / args.out_dir.replace("\\", "/")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "causal").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)
    logger.info("Phase14 source: %s", phase14_dir)
    logger.info("Phase15 out: %s", out_dir)

    # Run environment check first (logs actionable install commands if missing)
    run_check_env(repo_root)

    node2vec_ok, pykeen_ok, gnn_ok = check_dependencies()
    missing_dependencies = []
    if not node2vec_ok:
        missing_dependencies.append({"package": "node2vec", "install": "pip install node2vec"})
    if not pykeen_ok:
        missing_dependencies.append({"package": "pykeen", "install": "pip install pykeen"})
    if not gnn_ok:
        missing_dependencies.append({
            "package": "torch + torch_geometric",
            "install": "pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install torch-geometric",
        })

    # (1) Build training graph
    ret = run_cmd([
        sys.executable,
        str(repo_root / "scripts" / "phase15" / "build_training_graph.py"),
        "--run-dir", str(phase14_dir),
        "--out-dir", str(out_dir),
    ])
    if ret != 0:
        logger.error("build_training_graph failed")
        return ret

    steps_succeeded = ["build_training_graph"]
    steps_skipped = []
    steps_failed = []

    # (2) Node2Vec
    if not args.skip_node2vec and node2vec_ok:
        ret = run_cmd([
            sys.executable,
            str(repo_root / "scripts" / "phase15" / "run_node2vec.py"),
            "--phase15-dir", str(out_dir),
        ])
        if ret == 0:
            steps_succeeded.append("node2vec")
        else:
            steps_failed.append("node2vec")
    elif not args.skip_node2vec and not node2vec_ok:
        steps_skipped.append("node2vec (not installed)")
        logger.warning("node2vec skipped: not installed. pip install node2vec")

    # (3) PyKEEN
    if not args.skip_pykeen and pykeen_ok:
        ret = run_cmd([
            sys.executable,
            str(repo_root / "scripts" / "phase15" / "run_pykeen.py"),
            "--phase15-dir", str(out_dir),
        ])
        if ret == 0:
            steps_succeeded.append("pykeen")
        else:
            steps_failed.append("pykeen")
    elif not args.skip_pykeen and not pykeen_ok:
        steps_skipped.append("pykeen (not installed)")
        logger.warning("PyKEEN skipped: not installed. pip install pykeen")

    # (4) GNN
    if not args.skip_gnn and gnn_ok:
        ret = run_cmd([
            sys.executable,
            str(repo_root / "scripts" / "phase15" / "run_gnn_linkpred.py"),
            "--phase15-dir", str(out_dir),
        ])
        if ret == 0:
            steps_succeeded.append("gnn")
        else:
            steps_failed.append("gnn")
    elif not args.skip_gnn and not gnn_ok:
        steps_skipped.append("gnn (not installed)")
        logger.warning("GNN skipped: not installed. pip install torch torch-geometric")

    # (5) Validate
    if not args.skip_validate:
        ret = run_cmd([
            sys.executable,
            str(repo_root / "scripts" / "phase15" / "validate_predictions.py"),
            "--phase15-dir", str(out_dir),
        ])
        if ret == 0:
            steps_succeeded.append("validate")
        else:
            steps_failed.append("validate")

    # (6) Causal mediation
    if not args.skip_causal:
        ret = run_cmd([
            sys.executable,
            str(repo_root / "scripts" / "phase15" / "run_causal_mediation.py"),
            "--phase14-dir", str(phase14_dir),
            "--phase15-dir", str(out_dir),
        ])
        if ret == 0:
            steps_succeeded.append("causal")
        else:
            steps_failed.append("causal")

    # Readiness / metrics summary
    readiness = {
        "phase15_out_dir": str(out_dir),
        "phase14_source": str(phase14_dir),
        "steps_succeeded": steps_succeeded,
        "steps_skipped": steps_skipped,
        "steps_failed": steps_failed,
        "phase15_readiness": "complete" if "build_training_graph" in steps_succeeded else "incomplete",
        "missing_dependencies": missing_dependencies if missing_dependencies else [],
        "install_commands": [d["install"] for d in missing_dependencies] if missing_dependencies else [],
    }
    reports_dir = out_dir / "reports"
    build_report = reports_dir / "build_training_graph_report.json"
    if build_report.exists():
        try:
            with open(build_report, "r", encoding="utf-8") as f:
                r = json.load(f)
            readiness["n_nodes"] = r.get("n_nodes")
            readiness["n_edges"] = r.get("n_edges")
            readiness["overlap_vs_cg"] = r.get("overlap_vs_cg")
            readiness["overlap_vs_ic"] = r.get("overlap_vs_ic")
        except Exception:
            pass
    with open(reports_dir / "phase15_readiness.json", "w", encoding="utf-8") as f:
        json.dump(readiness, f, indent=2)
    logger.info("Phase15 readiness written to %s", reports_dir / "phase15_readiness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
