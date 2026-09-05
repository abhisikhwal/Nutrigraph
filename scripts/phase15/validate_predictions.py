"""
Phase15: Validate predicted compound->gene edges against ChEMBL/DrugBank if present locally.
Outputs validation_report.json, validated_predictions.csv, unmatched_predictions.csv.
If files missing, report gracefully with exact missing file names.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(phase15_dir: Path) -> pd.DataFrame:
    """Load PyKEEN and/or GNN top predictions (compound_id/head_str, gene/tail_str)."""
    predictions_dir = phase15_dir / "predictions"
    dfs = []
    for fname in ["pykeen_top_predictions_compound_gene.csv", "gnn_top_predictions_compound_gene.csv"]:
        p = predictions_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            if "head_str" in df.columns and "tail_str" in df.columns:
                df = df.rename(columns={"head_str": "compound_id", "tail_str": "gene_symbol"})
            elif "compound_id" not in df.columns and "head_int" in df.columns:
                continue
            df["source"] = fname.replace("_top_predictions_compound_gene.csv", "")
            dfs.append(df[["compound_id", "gene_symbol", "score", "source"]].drop_duplicates(subset=["compound_id", "gene_symbol"]))
    if not dfs:
        return pd.DataFrame(columns=["compound_id", "gene_symbol", "score", "source"])
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["compound_id", "gene_symbol"])


def load_chembl_targets(path: Path) -> Optional[pd.DataFrame]:
    """Expected: inchikey or chembl_id, gene_symbol (or target_symbol)."""
    if not path.exists():
        return None
    df = pd.read_csv(path, nrows=100000)
    cols = list(df.columns)
    id_col = None
    for c in ["inchikey", "compound_id", "chembl_id", "mol_id"]:
        if c in cols:
            id_col = c
            break
    if id_col is None:
        id_col = cols[0]
    gene_col = None
    for c in ["gene_symbol", "target_symbol", "gene", "symbol"]:
        if c in cols:
            gene_col = c
            break
    if gene_col is None:
        gene_col = cols[1] if len(cols) > 1 else None
    if gene_col is None:
        return None
    return df[[id_col, gene_col]].rename(columns={id_col: "compound_id", gene_col: "gene_symbol"}).drop_duplicates().dropna()


def load_drugbank_targets(path: Path) -> Optional[pd.DataFrame]:
    """Expected: inchikey or drugbank_id, gene_symbol."""
    if not path.exists():
        return None
    df = pd.read_csv(path, nrows=100000)
    cols = list(df.columns)
    id_col = next((c for c in ["inchikey", "compound_id", "drugbank_id", "InChI Key"] if c in cols), cols[0])
    gene_col = next((c for c in ["gene_symbol", "Gene Name", "target_symbol", "symbol"] if c in cols), (cols[1] if len(cols) > 1 else None))
    if gene_col is None:
        return None
    return df[[id_col, gene_col]].rename(columns={id_col: "compound_id", gene_col: "gene_symbol"}).drop_duplicates().dropna()


def validate_against_ref(pred: pd.DataFrame, ref: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match pred (compound_id, gene_symbol) to ref. Return (validated, unmatched)."""
    ref_set = set(zip(ref["compound_id"].astype(str).str.strip().str.upper(), ref["gene_symbol"].astype(str).str.strip().str.upper()))
    validated = []
    unmatched = []
    for _, row in pred.iterrows():
        c = str(row["compound_id"]).strip().upper()
        g = str(row["gene_symbol"]).strip().upper()
        if (c, g) in ref_set:
            validated.append(row.to_dict())
        else:
            unmatched.append(row.to_dict())
    return pd.DataFrame(validated), pd.DataFrame(unmatched)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase15 validate predictions vs ChEMBL/DrugBank")
    parser.add_argument("--phase15-dir", type=str, required=True)
    parser.add_argument("--chembl-path", type=str, default=None)
    parser.add_argument("--drugbank-path", type=str, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    phase15_dir = Path(args.phase15_dir).resolve()
    predictions_dir = phase15_dir / "predictions"
    validation_dir = phase15_dir / "validation"
    reports_dir = phase15_dir / "reports"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(phase15_dir)
    if pred.empty:
        logger.warning("No prediction files found in %s", predictions_dir)
        report = {"status": "no_predictions", "missing_files": [], "validated_count": 0, "unmatched_count": 0}
        with open(validation_dir / "validation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return 0

    chembl_path = Path(args.chembl_path) if args.chembl_path else repo_root / "data" / "external" / "chembl_targets.csv"
    drugbank_path = Path(args.drugbank_path) if args.drugbank_path else repo_root / "data" / "external" / "drugbank_targets.csv"
    missing = []
    if not chembl_path.exists():
        missing.append(str(chembl_path))
    if not drugbank_path.exists():
        missing.append(str(drugbank_path))

    validated_list = []
    unmatched_df = pred.copy()
    if chembl_path.exists():
        ref_chembl = load_chembl_targets(chembl_path)
        if ref_chembl is not None and not ref_chembl.empty:
            v, u = validate_against_ref(pred, ref_chembl)
            v["reference"] = "chembl"
            validated_list.append(v)
            unmatched_df = u
    if drugbank_path.exists():
        ref_db = load_drugbank_targets(drugbank_path)
        if ref_db is not None and not ref_db.empty:
            v, u = validate_against_ref(unmatched_df, ref_db)
            v["reference"] = "drugbank"
            validated_list.append(v)
            unmatched_df = u

    validated_df = pd.concat(validated_list, ignore_index=True) if validated_list else pd.DataFrame()
    validated_df.to_csv(validation_dir / "validated_predictions.csv", index=False)
    unmatched_df.to_csv(validation_dir / "unmatched_predictions.csv", index=False)

    report = {
        "status": "ok" if not missing else "partial",
        "missing_files": missing,
        "predictions_loaded": int(len(pred)),
        "validated_count": int(len(validated_df)),
        "unmatched_count": int(len(unmatched_df)),
        "chembl_used": chembl_path.exists(),
        "drugbank_used": drugbank_path.exists(),
    }
    with open(validation_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    summary = [
        "# Validation report",
        f"- Predictions loaded: {report['predictions_loaded']}",
        f"- Validated (ChEMBL/DrugBank): {report['validated_count']}",
        f"- Unmatched: {report['unmatched_count']}",
    ]
    if missing:
        summary.append("")
        summary.append("## Missing files (optional)")
        for m in missing:
            summary.append(f"- {m}")
    with open(validation_dir / "validation_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    logger.info("Validated %d unmatched %d; missing refs: %s", report["validated_count"], report["unmatched_count"], missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
