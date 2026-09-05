"""
Phase15: Agent to explain an ingredient pair. Pulls pair_category_mediation + evidence trails.
Output: structured explanation JSON + short plain-text report. No API calls if internet unavailable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_pair_mediation(run_dir: Path) -> pd.DataFrame:
    p = Path(run_dir) / "pair_category_mediation.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_evidence_trail(
    repo_root: Path,
    ing_a: str,
    ing_b: str,
    top_compounds: int = 10,
    top_genes: int = 10,
) -> Dict[str, Any]:
    """Shared compounds and genes from canonical CSVs."""
    ic_path = repo_root / "data" / "processed" / "canonical" / "ingredient_compound_canonical.csv"
    cg_path = repo_root / "data" / "processed" / "canonical" / "compound_gene_expanded_canonical.csv"
    if not cg_path.exists():
        cg_path = repo_root / "data" / "processed" / "canonical" / "compound_gene_canonical.csv"
    out = {"shared_compounds": [], "shared_genes": []}
    if not ic_path.exists():
        return out
    ic = pd.read_csv(ic_path)
    comp_a = set(ic.loc[ic["ingredient_id"] == ing_a, "compound_id"].dropna().astype(str))
    comp_b = set(ic.loc[ic["ingredient_id"] == ing_b, "compound_id"].dropna().astype(str))
    shared = list(comp_a & comp_b)[:top_compounds]
    out["shared_compounds"] = shared
    if cg_path.exists():
        cg = pd.read_csv(cg_path)
        genes = set()
        for c in shared:
            g = cg.loc[cg["compound_id"] == c, "gene_symbol"].dropna().astype(str)
            genes.update(g)
        out["shared_genes"] = list(genes)[:top_genes]
    return out


def build_explanation(
    ing_a: str,
    ing_b: str,
    run_dir: Path,
    repo_root: Path,
    top_categories: int = 10,
) -> Dict[str, Any]:
    """Structured explanation: pair, categories, mechanistic_score, evidence trail."""
    pair_med = load_pair_mediation(run_dir)
    if pair_med.empty:
        return {"ing_a": ing_a, "ing_b": ing_b, "categories": [], "mechanistic_scores": {}, "evidence": {}}
    subset = pair_med[(pair_med["ingA_id"] == ing_a) & (pair_med["ingB_id"] == ing_b)]
    if subset.empty:
        subset = pair_med[(pair_med["ingA_id"] == ing_b) & (pair_med["ingB_id"] == ing_a)]
    categories = []
    mechanistic_scores = {}
    for _, row in subset.iterrows():
        cat = row.get("category", "")
        categories.append(cat)
        mechanistic_scores[cat] = float(row.get("mechanistic_score", 0))
    categories = sorted(categories, key=lambda c: mechanistic_scores.get(c, 0), reverse=True)[:top_categories]
    evidence = load_evidence_trail(repo_root, ing_a, ing_b, top_compounds=10, top_genes=10)
    return {
        "ing_a": ing_a,
        "ing_b": ing_b,
        "categories": categories,
        "mechanistic_scores": {c: mechanistic_scores.get(c, 0) for c in categories},
        "evidence": evidence,
        "shared_compounds_count": len(evidence["shared_compounds"]),
        "shared_genes_count": len(evidence["shared_genes"]),
    }


def explanation_to_text(expl: Dict[str, Any]) -> str:
    """Short plain-text report."""
    lines = [
        f"Ingredient pair: {expl['ing_a']} + {expl['ing_b']}",
        "",
        "Top categories (mechanistic_score):",
    ]
    for c in expl.get("categories", [])[:10]:
        sc = expl.get("mechanistic_scores", {}).get(c, 0)
        lines.append(f"  - {c}: {sc:.3f}")
    lines.append("")
    lines.append("Evidence trail:")
    lines.append(f"  Shared compounds (top 10): {expl.get('evidence', {}).get('shared_compounds', [])[:10]}")
    lines.append(f"  Shared genes (top 10): {expl.get('evidence', {}).get('shared_genes', [])[:10]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase15 agent: explain ingredient pair")
    parser.add_argument("--ing-a", type=str, required=True)
    parser.add_argument("--ing-b", type=str, required=True)
    parser.add_argument("--run-dir", type=str, default=None, help="Phase14 run/snapshot")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-txt", type=str, default=None)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    if args.run_dir is None:
        snapshot = repo_root / "data" / "processed" / "milestones" / "phase14" / "v1_working_2026-02-19" / "phase14_20260219_204918"
        run_dir = snapshot if snapshot.exists() else repo_root / "data" / "processed" / "phase14_mediation" / "phase14_20260219_204918"
    else:
        run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        logger.error("Run dir not found: %s", run_dir)
        return 1

    expl = build_explanation(args.ing_a, args.ing_b, run_dir, repo_root)
    text = explanation_to_text(expl)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(expl, f, indent=2)
        logger.info("Wrote %s", out_path)
    if args.out_txt:
        out_path = Path(args.out_txt).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
