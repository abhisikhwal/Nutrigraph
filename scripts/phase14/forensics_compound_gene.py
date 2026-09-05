"""
Phase14 forensics: scan repo for compound->gene/target candidate files.
Produces data/processed/phase14_mediation/_forensics/candidate_files.csv and schemas/<safe_name>.json.
Run from repo root: python scripts/phase14/forensics_compound_gene.py [--repo-root .]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Filename patterns that suggest compound-gene/target content
FILENAME_KEYWORDS = [
    "binding", "bindingdb", "target", "uniprot", "gene", "genetics",
    "chemical_gene", "compound_gene", "ctd", "chembl", "gene_chemical",
]

# Compound identifier column names (any match)
COMPOUND_COL_NAMES = [
    "compound_id", "chemical_id", "inchikey", "inchi_key", "InChIKey", "pubchem_cid", "cid", "chembl_id",
]

# Gene/target column names
GENE_COL_NAMES = [
    "gene", "gene_symbol", "symbol", "target", "target_name", "uniprot", "uniprot_id",
]


def _path_safe_name(p: Path) -> str:
    """Safe filename from path for schema JSON (Windows-safe)."""
    s = str(p).replace("\\", "_").replace("/", "_").replace(":", "_")
    s = re.sub(r"[^\w\-_.]", "_", s)
    return s[:120] + (".json" if len(s) > 120 else "")


def _filename_matches(path: Path) -> bool:
    name = path.name.lower()
    return any(kw in name for kw in FILENAME_KEYWORDS)


def _has_compound_and_gene_columns(columns: list) -> tuple[bool, list]:
    """Return (has_both, list of detected key column names)."""
    cl = {c.lower(): c for c in columns}
    compound_col = None
    for k in COMPOUND_COL_NAMES:
        if k.lower() in cl:
            compound_col = cl[k.lower()]
            break
    gene_col = None
    for k in GENE_COL_NAMES:
        if k.lower() in cl:
            gene_col = cl[k.lower()]
            break
    key_cols = []
    if compound_col:
        key_cols.append(compound_col)
    if gene_col:
        key_cols.append(gene_col)
    return bool(compound_col and gene_col), key_cols


def _infer_phase(path: Path, root: Path) -> str:
    """Infer phase/folder from path relative to root."""
    try:
        rel = path.relative_to(root)
        parts = rel.parts
    except ValueError:
        return "unknown"
    if len(parts) >= 2:
        # data/processed/phase12_... or canonical/...
        folder = parts[1] if len(parts) > 2 else parts[0]
        return folder
    return "unknown"


def _sample_load(path: Path, n: int = 5) -> tuple[object, int | None]:
    """Load up to n rows; return (df or None, total_rows or None)."""
    import pandas as pd
    if not path.exists():
        return None, None
    suf = path.suffix.lower()
    try:
        if suf == ".csv":
            df = pd.read_csv(path, nrows=n)
            try:
                full = pd.read_csv(path)
                total = len(full)
            except Exception:
                total = None
            return df, total
        if suf in (".parquet", ".pq"):
            df = pd.read_parquet(path)
            total = len(df)
            if total > n:
                df = df.head(n)
            return df, total
    except Exception:
        return None, None
    return None, None


def run_forensics(repo_root: Path) -> Path:
    """Scan data/processed, write candidate_files.csv and schemas/*.json. Returns forensics dir."""
    import pandas as pd

    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    forensics_dir = processed / "phase14_mediation" / "_forensics"
    schemas_dir = forensics_dir / "schemas"
    forensics_dir.mkdir(parents=True, exist_ok=True)
    schemas_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for ext in ("*.parquet", "*.pq", "*.csv"):
        for path in processed.rglob(ext):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            phase = _infer_phase(path, repo_root)
            try:
                path_rel = str(path.relative_to(repo_root))
            except ValueError:
                path_rel = str(path)
            # Load sample to get columns and row count
            sample, n_rows = _sample_load(path, 5)
            if sample is None:
                candidates.append({
                    "path": str(path),
                    "path_rel": path_rel,
                    "size": size,
                    "phase": phase,
                    "detected_type": "load_failed",
                    "n_rows": None,
                    "columns": "",
                    "key_columns_detected": "",
                })
                continue
            df = sample
            columns = list(df.columns)
            by_filename = _filename_matches(path)
            has_keys, key_cols = _has_compound_and_gene_columns(columns)
            if by_filename or has_keys:
                detected_type = "compound_gene_by_columns" if has_keys else "compound_gene_by_filename"
                if not has_keys:
                    key_cols = []
            else:
                detected_type = "skipped_no_compound_gene_columns"
                key_cols = []

            candidates.append({
                "path": str(path),
                "path_rel": path_rel,
                "size": size,
                "phase": phase,
                "detected_type": detected_type,
                "n_rows": n_rows,
                "columns": "|".join(columns),
                "key_columns_detected": "|".join(key_cols) if key_cols else "",
            })

            # Write schema JSON for candidate (has_keys or by_filename)
            if has_keys or (by_filename and columns):
                safe = _path_safe_name(path)
                if not safe.endswith(".json"):
                    safe = safe + ".json"
                schema = {
                    "path": str(path),
                    "columns": columns,
                    "dtypes": {c: str(df.dtypes.get(c, "")) for c in columns},
                    "sample_rows": df.head(5).fillna("").astype(str).to_dict("records"),
                    "key_columns_detected": key_cols,
                }
                with open(schemas_dir / safe, "w", encoding="utf-8") as f:
                    json.dump(schema, f, indent=2)
            elif by_filename and not has_keys:
                safe = _path_safe_name(path)
                if not safe.endswith(".json"):
                    safe = safe + ".json"
                schema = {
                    "path": str(path),
                    "columns": columns,
                    "dtypes": {c: str(df.dtypes.get(c, "")) for c in columns},
                    "sample_rows": df.head(5).fillna("").astype(str).to_dict("records"),
                    "key_columns_detected": [],
                    "note": "filename match but no compound+gene columns detected",
                }
                with open(schemas_dir / safe, "w", encoding="utf-8") as f:
                    json.dump(schema, f, indent=2)

    # Filter to actual candidates (compound_gene by columns or by filename with keys)
    candidate_df = pd.DataFrame(candidates)
    candidate_df.to_csv(forensics_dir / "candidate_files.csv", index=False)

    # Rank: explicit compound-gene > genetics > bindingdb > other
    def rank_row(r):
        path_lower = (r.get("path") or "").lower()
        rel = (r.get("path_rel") or "").lower()
        if "compound_gene" in path_lower or "chemical_gene" in path_lower or "gene_chemical" in path_lower:
            return 0
        if "phase12" in rel and "genetics" in rel:
            return 1
        if "bindingdb" in path_lower or "binding" in path_lower:
            return 2
        if "phase16" in rel:
            return 2
        return 3

    valid = candidate_df[candidate_df["detected_type"].isin(("compound_gene_by_columns", "compound_gene_by_filename"))].copy()
    if not valid.empty:
        valid["_rank"] = valid.apply(rank_row, axis=1)
        valid = valid.sort_values(["_rank", "n_rows"], ascending=[True, False])
        print("Ranked compound->gene/target sources (best first):")
        for i, (_, row) in enumerate(valid.head(20).iterrows(), 1):
            print(f"  {i}. [{row['detected_type']}] {row['path_rel']} (rows={row['n_rows']}, keys={row['key_columns_detected']})")
    else:
        print("No compound->gene/target candidate files found.")

    return forensics_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase14 forensics: scan for compound->gene candidates")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root")
    args = parser.parse_args()
    out = run_forensics(args.repo_root)
    print(f"Forensics output: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
