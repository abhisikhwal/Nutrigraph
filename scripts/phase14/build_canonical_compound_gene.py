"""
Phase14: Build canonical compound->gene file from forensics + normalized sources.
Run from repo root: python scripts/phase14/build_canonical_compound_gene.py [--repo-root .]
Writes data/processed/canonical/compound_gene_links.parquet and .csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_csv_or_parquet(p: Path):
    import pandas as pd
    p = Path(p)
    if not p.exists():
        return None
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return None


def _load_ingredient_compound_keys_and_ings(repo_root: Path) -> tuple:
    """Load ingredient_compounds (or derived); return (set of compound keys, df with ingredient_id + compound_key if available)."""
    root = Path(repo_root).resolve()
    candidates = [
        root / "data" / "processed" / "canonical" / "ingredient_compounds.parquet",
        root / "data" / "processed" / "canonical" / "ingredient_compounds.csv",
        root / "data" / "processed" / "phase14_mediation" / "derived" / "ingredient_compound_links.parquet",
        root / "data" / "processed" / "phase14_mediation" / "derived" / "ingredient_compound_links.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = _load_csv_or_parquet(path)
        if df is None or df.empty:
            continue
        cl = {c.lower(): c for c in df.columns}
        keys = set()
        key_col = None
        if "inchikey" in cl or "inchi_key" in cl:
            key_col = cl.get("inchikey") or cl.get("inchi_key")
            keys = set(df[key_col].dropna().astype(str).str.strip().str.upper())
        if "compound_id" in cl:
            keys |= set(df[cl["compound_id"]].dropna().astype(str).str.strip())
            if key_col is None:
                key_col = cl["compound_id"]
        keys = {k for k in keys if k and str(k).lower() != "nan"}
        ing_col = cl.get("ingredient_id")
        if ing_col and key_col:
            return keys, df[[ing_col, key_col]].drop_duplicates()
        return keys, None
    return set(), None


def _unique_inchikey_or_cid_from_ingredient_compounds(repo_root: Path) -> set:
    keys, _ = _load_ingredient_compound_keys_and_ings(repo_root)
    return keys


def run_build(repo_root: Path) -> int:
    import pandas as pd
    from src.phase14.normalize_sources import canonicalize_compound_gene

    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    forensics_dir = processed / "phase14_mediation" / "_forensics"
    canonical_dir = processed / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    # a) Run forensics
    forensics_script = repo_root / "scripts" / "phase14" / "forensics_compound_gene.py"
    if forensics_script.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("forensics_compound_gene", forensics_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run_forensics(repo_root)
    else:
        print("Forensics script not found; reading existing candidate_files.csv if present.")

    candidate_path = forensics_dir / "candidate_files.csv"
    if not candidate_path.exists():
        print("No candidate_files.csv; run forensics first.")
        return 1
    cand = pd.read_csv(candidate_path)
    valid = cand[cand["detected_type"].isin(("compound_gene_by_columns", "compound_gene_by_filename"))].copy()
    if valid.empty:
        print("No compound->gene candidates found. Create genetics/binding files and re-run forensics.")
        return 1

    def rank_path(path_rel: str) -> int:
        r = (path_rel or "").lower()
        if "compound_gene" in r or "chemical_gene" in r or "gene_chemical" in r:
            return 0
        if "phase12" in r and "genetics" in r:
            return 1
        if "bindingdb" in r or "phase16" in r or "binding" in r:
            return 2
        return 3

    valid["_rank"] = valid["path_rel"].astype(str).apply(rank_path)
    valid = valid.sort_values(["_rank", "n_rows"], ascending=[True, False])

    # b) Pick best sources, c) normalize, d) union + dedup on (inchikey, gene)
    all_frames = []
    for _, row in valid.iterrows():
        path = Path(row["path"])
        if not path.exists():
            continue
        df = _load_csv_or_parquet(path)
        if df is None or df.empty:
            continue
        out = canonicalize_compound_gene(df, str(path))
        if out.empty:
            continue
        all_frames.append(out)
    if not all_frames:
        print("No rows after canonicalization.")
        return 1
    combined = pd.concat(all_frames, ignore_index=True)
    combined["inchikey"] = combined["inchikey"].astype(str).str.strip().str.upper()
    if "compound_id" not in combined.columns:
        combined["compound_id"] = combined["inchikey"]
    combined["compound_id"] = combined["compound_id"].astype(str).str.strip().str.upper()
    combined["gene"] = combined["gene"].astype(str).str.strip().str.upper()
    combined = combined[combined["gene"].str.len() > 0]
    combined = combined[combined["compound_id"].str.len() > 0]
    combined = combined.drop_duplicates(subset=["compound_id", "gene"])

    out_pq = canonical_dir / "compound_gene_links.parquet"
    out_csv = canonical_dir / "compound_gene_links.csv"
    combined.to_parquet(out_pq, index=False)
    combined.to_csv(out_csv, index=False)
    print(f"Wrote {out_pq} ({len(combined)} rows)")
    print(f"Wrote {out_csv}")

    # e) Coverage stats (join key = compound_id in canonical file)
    ing_cmp_keys = _unique_inchikey_or_cid_from_ingredient_compounds(repo_root)
    cg_keys = set(combined["compound_id"].dropna().astype(str).str.strip().str.upper())
    cg_keys = {k for k in cg_keys if k}
    cg_inchikey = set(combined["inchikey"].dropna().astype(str).str.strip().str.upper()) if "inchikey" in combined.columns else set()
    cg_inchikey = {k for k in cg_inchikey if k}
    overlap = len(ing_cmp_keys & cg_keys)
    overlap_pct = (100.0 * overlap / len(ing_cmp_keys)) if ing_cmp_keys else 0.0
    print("--- Coverage ---")
    print(f"#unique compound keys in ingredient_compounds (or derived): {len(ing_cmp_keys)}")
    print(f"#unique compound_id in compound_gene_links: {len(cg_keys)}")
    print(f"Overlap count: {overlap}")
    print(f"Overlap %: {overlap_pct:.2f}%")

    if ing_cmp_keys and overlap_pct < 1.0:
        print("WARNING: Overlap < 1%. Likely identifier mismatch (e.g. ingredient_compounds use CID/FDB ID, compound_gene_links use InChIKey). Consider adding an InChIKey lookup for ingredient compounds.")

    _, ing_cmp_df = _load_ingredient_compound_keys_and_ings(repo_root)
    if ing_cmp_df is not None and len(ing_cmp_df.columns) >= 2:
        ing_col = ing_cmp_df.columns[0]
        key_col = ing_cmp_df.columns[1]
        keys_in_cg = cg_inchikey
        ing_cmp_df = ing_cmp_df.dropna(subset=[key_col])
        ing_cmp_df["_key_upper"] = ing_cmp_df[key_col].astype(str).str.strip().str.upper()
        ings_with_gene = set(ing_cmp_df[ing_cmp_df["_key_upper"].isin(cg_keys)][ing_col].dropna().astype(str))
        print(f"#ingredients reaching >=1 gene (via compound): {len(ings_with_gene)}")
    else:
        print("#ingredients reaching >=1 gene: N/A (no ingredient_compound with ingredient_id)")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical compound->gene file")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo root")
    args = parser.parse_args()
    return run_build(args.repo_root)


if __name__ == "__main__":
    sys.exit(main())
