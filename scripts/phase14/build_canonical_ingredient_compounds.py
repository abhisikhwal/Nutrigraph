"""
Phase14: Build canonical Ingredient->Compound links aligned to Phase13 ING_ IDs and compound_gene_links.
Run from repo root: python scripts/phase14/build_canonical_ingredient_compounds.py [--repo-root .]
Output: data/processed/canonical/ingredient_compound_links.csv (required), optional .parquet.
Fails with nonzero exit and audit_report.json if overlap < 1%% or atlas ingredient coverage < 30%%.
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


def _normalize_name(s: str) -> str:
    if not isinstance(s, str) or str(s).strip() == "":
        return ""
    s = re.sub(r"[^\w\s]", " ", str(s).lower().strip())
    return " ".join(s.split())


def _discover_ing_cmp_sources(repo_root: Path) -> list[Path]:
    """Discover raw ING->CMP sources under data/processed (ingredient_compounds, coconut, etc.)."""
    processed = repo_root / "data" / "processed"
    if not processed.exists():
        return []
    candidates = []
    for pat in ("**/ingredient_compounds*.parquet", "**/ingredient_compounds*.csv",
                "**/ingredient_compound*.parquet", "**/ingredient_compound*.csv"):
        for p in processed.glob(pat):
            if p.is_file():
                candidates.append(p)
    seen = set()
    out = []
    for p in sorted(candidates, key=lambda x: (len(str(x)), str(x))):
        key = (p.name, p.stat().st_size)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def run_build(repo_root: Path, allow_fuzzy: bool = False, smoke_recipes: int | None = None) -> int:
    import pandas as pd
    from src.phase14.id_normalization import to_ingredient_id

    repo_root = Path(repo_root).resolve()
    canonical_dir = repo_root / "data" / "processed" / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    report_dir = canonical_dir / "phase14_audit"
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Ingredient lookup from recipe_ingredients ---
    ri_path = canonical_dir / "recipe_ingredients_expanded_v2.parquet"
    if not ri_path.exists():
        ri_path = repo_root / "data" / "processed" / "canonical" / "recipe_ingredients_expanded_v2.csv"
    if not ri_path.exists():
        print("ERROR: recipe_ingredients_expanded_v2 not found. Need it for authoritative ingredient IDs.")
        return 1
    ri = _load_csv_or_parquet(ri_path)
    if ri is None or ri.empty:
        print("ERROR: Could not load recipe_ingredients.")
        return 1
    if smoke_recipes and ri is not None and "recipe_id" in ri.columns:
        recipe_sample = ri["recipe_id"].drop_duplicates().head(smoke_recipes)
        ri = ri[ri["recipe_id"].isin(recipe_sample)]
    id_col = "ingredient_id" if "ingredient_id" in ri.columns else None
    name_col = next((c for c in ri.columns if "ingredient" in c.lower() and "raw" in c.lower()), None)
    if not name_col:
        name_col = next((c for c in ri.columns if "ingredient" in c.lower() and "name" in c.lower()), None)
    if not id_col:
        print("ERROR: recipe_ingredients missing ingredient_id.")
        return 1
    lookup = ri[[id_col]].drop_duplicates()
    lookup["ingredient_id_canonical"] = lookup[id_col].astype(str).apply(lambda x: to_ingredient_id(x))
    if name_col and name_col in ri.columns:
        name_df = ri[[id_col, name_col]].drop_duplicates()
        name_df = name_df.groupby(id_col, as_index=False).first()
        lookup = lookup.merge(name_df, on=id_col, how="left")
        lookup["_name_norm"] = lookup[name_col].fillna("").astype(str).apply(_normalize_name)
    else:
        lookup["_name_norm"] = ""
    lookup = lookup.drop_duplicates(subset=["ingredient_id_canonical"])

    # --- 2) Load raw ingredient_compounds ---
    raw_path = canonical_dir / "ingredient_compounds.parquet"
    if not raw_path.exists():
        raw_path = canonical_dir / "ingredient_compounds.csv"
    sources = _discover_ing_cmp_sources(repo_root)
    if not raw_path.exists() and sources:
        raw_path = sources[0]
    if not raw_path.exists():
        print("ERROR: No ingredient_compounds.parquet/csv found under data/processed/canonical or discovered.")
        return 1
    raw = _load_csv_or_parquet(raw_path)
    if raw is None or raw.empty:
        print("ERROR: Could not load raw ingredient_compounds.")
        return 1

    # Detect columns: ingredient side
    cl = {c.lower(): c for c in raw.columns}
    raw_ing_col = next((cl[k] for k in ["ingredient_id", "ing_id", "ingredientid"] if k in cl), None)
    if not raw_ing_col:
        raw_ing_col = next((cl[k] for k in ["ingredient_name", "ingredient_raw", "food_name", "name"] if k in cl), None)
    raw_cmp_col = next((cl[k] for k in ["compound_id", "compoundid", "inchikey", "inchi_key"] if k in cl), None)
    if not raw_cmp_col:
        raw_cmp_col = next((c for c in raw.columns if "compound" in c.lower() or "inchikey" in c.lower()), None)
    if not raw_ing_col or not raw_cmp_col:
        print("ERROR: Raw file missing ingredient and compound columns. Columns:", list(raw.columns))
        with open(report_dir / "audit_report.json", "w") as f:
            json.dump({"error": "column_mismatch", "columns": list(raw.columns)}, f, indent=2)
        return 1

    # Build edges: join to lookup
    raw = raw[[raw_ing_col, raw_cmp_col]].dropna(how="all")
    raw = raw.astype(str).apply(lambda x: x.str.strip())
    raw = raw[(raw[raw_ing_col] != "") & (raw[raw_cmp_col] != "")]

    # Join by id: exact match raw ingredient -> lookup ingredient_id
    lookup_ids = lookup[[id_col, "ingredient_id_canonical"]].drop_duplicates()
    raw["_raw_ing"] = raw[raw_ing_col].astype(str).str.strip()
    merged = raw.merge(lookup_ids, left_on="_raw_ing", right_on=id_col, how="inner")
    if merged.empty:
        raw["ingredient_id_canonical"] = raw[raw_ing_col].astype(str).apply(lambda x: to_ingredient_id(x))
        merged = raw.merge(lookup_ids[["ingredient_id_canonical"]].drop_duplicates(), on="ingredient_id_canonical", how="inner")
    if merged.empty and "_name_norm" in lookup.columns and lookup["_name_norm"].str.len().gt(0).any():
        raw["_raw_norm"] = raw[raw_ing_col].astype(str).apply(_normalize_name)
        lookup_named = lookup[lookup["_name_norm"] != ""][["_name_norm", "ingredient_id_canonical"]].drop_duplicates()
        merged = raw.merge(lookup_named, left_on="_raw_norm", right_on="_name_norm", how="inner")
    if merged.empty and allow_fuzzy:
        try:
            from rapidfuzz import fuzz
            rows = []
            raw_names = raw[raw_ing_col].drop_duplicates().tolist()
            for r in raw.itertuples(index=False):
                r_ing = getattr(r, raw_ing_col)
                r_cmp = getattr(r, raw_cmp_col)
                best = None
                best_score = 0
                for _, row in lookup.iterrows():
                    if row["_name_norm"] == "":
                        continue
                    sc = fuzz.ratio(_normalize_name(r_ing), row["_name_norm"])
                    if sc >= 90 and sc > best_score:
                        best_score = sc
                        best = row["ingredient_id_canonical"]
                if best:
                    rows.append({"ingredient_id": best, "compound_id": r_cmp})
            if rows:
                merged = pd.DataFrame(rows)
        except ImportError:
            pass
    if merged.empty:
        print("ERROR: No rows after joining raw to ingredient lookup. Check ingredient_id/name alignment.")
        with open(report_dir / "audit_report.json", "w") as f:
            json.dump({"error": "join_empty", "raw_columns": [raw_ing_col, raw_cmp_col], "lookup_columns": list(lookup.columns)}, f, indent=2)
        return 1

    canon_col = "ingredient_id_canonical" if "ingredient_id_canonical" in merged.columns else None
    if not canon_col:
        canon_col = [c for c in merged.columns if "ingredient" in c.lower() and "id" in c.lower()]
        canon_col = canon_col[0] if canon_col else merged.columns[0]
    out = merged[[canon_col, raw_cmp_col]].drop_duplicates()
    out = out.rename(columns={canon_col: "ingredient_id", raw_cmp_col: "compound_id"})
    out["ingredient_id"] = out["ingredient_id"].astype(str)
    out["compound_id"] = out["compound_id"].astype(str).str.strip().str.upper()
    out = out[(out["ingredient_id"].str.len() > 0) & (out["compound_id"].str.len() > 0)]

    # Normalize compound_id to match compound_gene_links: prefer InChIKey-style, else keep as-is
    def _looks_inchikey(s: str) -> bool:
        s = str(s).strip().upper()
        return len(s) >= 25 and "-" in s and s.replace("-", "").isalnum()
    out["compound_id"] = out["compound_id"].apply(lambda x: x.strip().upper() if _looks_inchikey(x) else x.strip())

    # --- 3) Overlap with compound_gene_links ---
    cg_path = canonical_dir / "compound_gene_links.csv"
    if not cg_path.exists():
        cg_path = canonical_dir / "compound_gene_links.parquet"
    if not cg_path.exists():
        print("WARNING: compound_gene_links not found; cannot enforce overlap.")
        cg_compound_ids = set()
    else:
        cg = _load_csv_or_parquet(cg_path)
        if cg is None or cg.empty:
            cg_compound_ids = set()
        else:
            cmp_col_cg = next((c for c in cg.columns if c.lower() in ("compound_id", "inchikey")), cg.columns[0])
            cg_compound_ids = set(cg[cmp_col_cg].dropna().astype(str).str.strip().str.upper())
            cg_compound_ids = {x for x in cg_compound_ids if x}

    n_unique_compounds_ing_cmp = out["compound_id"].nunique()
    n_unique_compounds_cmp_gene = len(cg_compound_ids)
    cmp_ing_set = set(out["compound_id"].unique())
    overlap_set = cmp_ing_set & cg_compound_ids
    overlap_pct = (100.0 * len(overlap_set) / len(cmp_ing_set)) if cmp_ing_set else 0.0

    # Atlas coverage (if atlas available)
    atlas_path = repo_root / "data" / "processed" / "phase13_interactions_v3_20260206_162122_b_gpu_stable" / "atlas_confirmed.csv"
    if not atlas_path.exists():
        dirs = list((repo_root / "data" / "processed").glob("phase13*"))
        if dirs:
            atlas_path = dirs[0] / "atlas_confirmed.csv"
    atlas_ing_coverage_pct = None
    atlas_rows_both_pct = None
    if atlas_path.exists():
        atlas = _load_csv_or_parquet(atlas_path)
        if atlas is not None and not atlas.empty and "ingA_id" in atlas.columns and "ingB_id" in atlas.columns:
            atlas_ings = set(atlas["ingA_id"].dropna().astype(str)) | set(atlas["ingB_id"].dropna().astype(str))
            atlas_ings_canonical = {to_ingredient_id(x) for x in atlas_ings}
            mapped_ings = set(out["ingredient_id"].unique())
            n_atlas = len(atlas_ings_canonical)
            n_mapped = len(atlas_ings_canonical & mapped_ings)
            atlas_ing_coverage_pct = (100.0 * n_mapped / n_atlas) if n_atlas else 0
            both_mapped = atlas.apply(
                lambda r: to_ingredient_id(str(r["ingA_id"])) in mapped_ings and to_ingredient_id(str(r["ingB_id"])) in mapped_ings,
                axis=1
            )
            atlas_rows_both_pct = (100.0 * both_mapped.sum() / len(atlas)) if len(atlas) else 0

    # --- 4) Enforce overlap and coverage ---
    audit = {
        "n_unique_compounds_ing_cmp": int(n_unique_compounds_ing_cmp),
        "n_unique_compounds_cmp_gene": int(n_unique_compounds_cmp_gene),
        "overlap_count": len(overlap_set),
        "overlap_pct": round(overlap_pct, 2),
        "atlas_ingredient_coverage_pct": atlas_ing_coverage_pct,
        "atlas_rows_both_sides_mapped_pct": atlas_rows_both_pct,
        "example_compound_ids_ing_cmp": list(cmp_ing_set)[:20],
        "example_compound_ids_cmp_gene": list(cg_compound_ids)[:20],
        "example_matched": list(overlap_set)[:20],
    }
    with open(report_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("--- Coverage ---")
    print(f"n_unique_compounds_ing_cmp: {n_unique_compounds_ing_cmp}")
    print(f"n_unique_compounds_cmp_gene: {n_unique_compounds_cmp_gene}")
    print(f"overlap %: {overlap_pct:.2f}%")
    if atlas_ing_coverage_pct is not None:
        print(f"atlas ingredient coverage %: {atlas_ing_coverage_pct:.2f}%")
    if atlas_rows_both_pct is not None:
        print(f"atlas rows both sides mapped %: {atlas_rows_both_pct:.2f}%")

    fail = False
    if overlap_pct < 1.0 and cg_compound_ids:
        print("FAIL: Overlap with compound_gene_links < 1%. See audit_report.json for example IDs.")
        fail = True
    if atlas_ing_coverage_pct is not None and atlas_ing_coverage_pct < 30:
        print("FAIL: Atlas ingredient coverage < 30%.")
        fail = True
    if fail:
        return 1

    # Write output
    out_csv = canonical_dir / "ingredient_compound_links.csv"
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out)} rows, {out['ingredient_id'].nunique()} unique ingredients)")
    try:
        out.to_parquet(canonical_dir / "ingredient_compound_links.parquet", index=False)
        print(f"Wrote {canonical_dir / 'ingredient_compound_links.parquet'}")
    except Exception as e:
        print(f"Parquet write skipped: {e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical ingredient->compound links")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--allow-fuzzy", action="store_true", help="Allow fuzzy name match if exact join is low")
    parser.add_argument("--smoke-recipes", type=int, default=None, help="Use only first N recipes (for testing)")
    args = parser.parse_args()
    return run_build(args.repo_root, allow_fuzzy=args.allow_fuzzy, smoke_recipes=args.smoke_recipes)


if __name__ == "__main__":
    sys.exit(main())
