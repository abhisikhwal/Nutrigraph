#!/usr/bin/env python3
"""
Species remap BUILD (Phase 1-3) — exact FooDB species matching with human-review gate.

Re-maps recoverable ingredient strings onto canonical species nodes (Latin binomial)
using exact/normalized/modifier-stripped FooDB matching + curated trap disambiguation.
NO fuzzy string scoring. Does NOT merge into 223-graph or touch measured canonical files.

Usage (from repo root):
    python scripts/species_remap/build_species_remap.py

Outputs (data/processed/species_remap/ only):
    species_nodes_v1.parquet
    ingredient_string_species_map_v1.parquet
    auto_accepted_v1.parquet
    ambiguous_for_review_v1.parquet
    ambiguous_for_review_v1.json
    composites_for_recipe_layer_v1.parquet
    unmatched_v1.parquet
    junk_discarded_v1.json
    species_remap_report_v1.json

STOP after Phase 3 — ambiguous set awaits human review before merge.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.species_remap.species_match import (  # noqa: E402
    FooDBSpeciesIndex,
    load_foodb_index,
    match_ingredient_string,
    normalize_name,
)

OUT_DIR = ROOT / "data/processed/species_remap"
RAW_FULL = ROOT / "data/interim/recipenlg/recipe_ingredients_raw_full.parquet"
MAPPED_FULL = ROOT / "data/interim/recipenlg/recipe_ingredients_mapped_full.parquet"
INGREDIENTS = ROOT / "data/processed/canonical/ingredients.parquet"
IC223 = ROOT / "data/processed/canonical/ingredient_compound_canonical.csv"
FOODB_FOOD = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Food.csv"
FOODB_CONTENT = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Content.csv"
FOODB_COMPOUND = ROOT / "data/processed/canonical/compound_master.csv"
MEASURED_CG = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"
PREDICTED_CG = ROOT / "data/processed/thread2/inference/predicted_compound_gene_weighted_v2.parquet"

AMBIGUOUS_FREQ_THRESHOLD = 3  # export all ambiguous >= this; report total count
AUTO_SAMPLE_N = 20
STAPLES = ["salt", "sugar", "flour", "water", "milk", "chicken", "pepper", "rice", "garlic", "onion", "butter", "eggs"]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (pd.Int64Dtype,)):
        return obj
    return str(obj)


def load_string_universe() -> pd.DataFrame:
    """All distinct ingredient strings ranked by recipe frequency."""
    raw = pd.read_parquet(RAW_FULL)
    freq = (
        raw.groupby("ingredient_raw", as_index=False)
        .size()
        .rename(columns={"size": "recipe_occurrences"})
        .sort_values("recipe_occurrences", ascending=False)
    )
    freq["ingredient_string"] = freq["ingredient_raw"].astype(str).str.strip()
    freq = freq.drop(columns=["ingredient_raw"])

    # Re-validate 223: add canonical names not already present as strings
    ing = pd.read_parquet(INGREDIENTS)
    ic223_ids = set(pd.read_csv(IC223)["ingredient_id"].astype(str).unique())
    names223 = ing.loc[ing["ingredient_id"].astype(str).isin(ic223_ids), "canonical_name"].dropna().astype(str)
    existing = set(freq["ingredient_string"].str.lower())
    extra_rows = []
    for name in names223.unique():
        if name.lower() not in existing:
            extra_rows.append({"ingredient_string": name, "recipe_occurrences": 0, "source": "223_canonical_revalidate"})
    if extra_rows:
        freq = pd.concat([freq.assign(source="raw_full"), pd.DataFrame(extra_rows)], ignore_index=True)
    else:
        freq = freq.assign(source="raw_full")
    return freq


def load_old_fuzzy_context() -> tuple[dict[str, str | None], dict[str, bool], dict[str, str | None]]:
    """Old fuzzy map: string -> ingredient_id, in_223 flag, matched_name."""
    mapped = pd.read_parquet(MAPPED_FULL)
    ic223_ids = set(pd.read_csv(IC223)["ingredient_id"].astype(str).unique())
    best = (
        mapped.dropna(subset=["ingredient_id"])
        .groupby(["ingredient_raw", "ingredient_id"])
        .size()
        .reset_index(name="n")
        .sort_values(["ingredient_raw", "n"], ascending=[True, False])
        .drop_duplicates("ingredient_raw")
    )
    ing = pd.read_parquet(INGREDIENTS, columns=["ingredient_id", "canonical_name"])
    ing_lu = dict(zip(ing["ingredient_id"].astype(str), ing["canonical_name"].astype(str)))
    old_id: dict[str, str | None] = {}
    old_name: dict[str, str | None] = {}
    in223: dict[str, bool] = {}
    for _, row in best.iterrows():
        s = str(row["ingredient_raw"]).strip()
        iid = str(row["ingredient_id"])
        old_id[s] = iid
        old_name[s] = ing_lu.get(iid)
        in223[s] = iid in ic223_ids
    return old_id, in223, old_name


def build_food_compound_inchikeys(content_csv: Path, compound_master_csv: Path) -> dict[int, set[str]]:
    """Map food_id -> set of InChIKeys via FooDB Content source_id -> compound_master.fdb_id."""
    cm = pd.read_csv(
        compound_master_csv,
        usecols=["inchikey", "fdb_id_norm"],
        engine="python",
        on_bad_lines="skip",
    )
    cm = cm.dropna(subset=["inchikey", "fdb_id_norm"])

    def fdb_num(x: str) -> int | None:
        s = str(x).strip().upper().replace("FDB_", "").replace("FDB", "")
        try:
            return int(s)
        except ValueError:
            return None

    fdb_to_ik: dict[int, str] = {}
    for _, row in cm.iterrows():
        num = fdb_num(row["fdb_id_norm"])
        if num is not None:
            fdb_to_ik[num] = str(row["inchikey"]).strip()

    food_iks: dict[int, set[str]] = defaultdict(set)
    for chunk in pd.read_csv(content_csv, usecols=["food_id", "source_id", "source_type"], chunksize=500_000):
        sub = chunk[chunk["source_type"].astype(str).str.lower() == "compound"]
        for fid, sid in zip(sub["food_id"].dropna(), sub["source_id"].dropna()):
            ik = fdb_to_ik.get(int(sid))
            if ik:
                food_iks[int(fid)].add(ik)
    return dict(food_iks)


def load_corpus_inchikeys() -> tuple[set[str], set[str], set[str]]:
    measured = set(pd.read_csv(MEASURED_CG, usecols=["compound_id"])["compound_id"].dropna().astype(str))
    ic223 = set(pd.read_csv(IC223, usecols=["compound_id"])["compound_id"].dropna().astype(str))
    pred = pd.read_parquet(PREDICTED_CG, columns=["dark_compound_inchikey"])
    predicted = set(pred["dark_compound_inchikey"].dropna().astype(str))
    return measured, predicted, ic223


def assign_species_node_ids(df: pd.DataFrame, index: FooDBSpeciesIndex) -> pd.DataFrame:
    """Assign SP_* ids to unique food_ids in auto_accepted rows."""
    accepted = df[df["bucket"] == "auto_accepted"].dropna(subset=["foodb_food_id"])
    uniq_fids = sorted(accepted["foodb_food_id"].astype(int).unique())
    fid_to_sp = {fid: f"SP_{i:06d}" for i, fid in enumerate(uniq_fids, start=1)}
    nodes = []
    for fid in uniq_fids:
        rec = index.get(fid)
        if rec:
            nodes.append(
                {
                    "species_node_id": fid_to_sp[fid],
                    "canonical_name": rec.name,
                    "latin_name": rec.latin_name,
                    "foodb_food_id": fid,
                    "n_compounds_foodb": rec.n_compounds,
                    "n_nutrients_foodb": rec.n_nutrients,
                }
            )
    nodes_df = pd.DataFrame(nodes)
    df = df.copy()
    df["species_node_id"] = df["foodb_food_id"].map(lambda x: fid_to_sp.get(int(x)) if pd.notna(x) else None)
    return df, nodes_df


def build_report(
    df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    index: FooDBSpeciesIndex,
    total_pairs: int,
    old_id: dict[str, str | None],
    old_name: dict[str, str | None],
    in223: dict[str, bool],
    food_iks: dict[int, set[str]],
    measured_ik: set[str],
    predicted_ik: set[str],
    ic223_ik: set[str],
) -> dict[str, Any]:
    corpus_all = measured_ik | predicted_ik | ic223_ik

    bucket_counts = df.groupby("bucket").agg(
        n_strings=("ingredient_string", "count"),
        recipe_occurrences=("recipe_occurrences", "sum"),
    )
    bucket_summary = {}
    for bucket, row in bucket_counts.iterrows():
        occ = int(row["recipe_occurrences"])
        bucket_summary[bucket] = {
            "n_strings": int(row["n_strings"]),
            "recipe_occurrences": occ,
            "occurrence_pct_of_corpus": round(100 * occ / total_pairs, 2),
        }

    auto = df[df["bucket"] == "auto_accepted"].copy()
    auto_species = auto["foodb_food_id"].nunique()
    auto_with_compounds = 0
    auto_name_only = 0
    for fid in auto["foodb_food_id"].dropna().astype(int).unique():
        rec = index.get(fid)
        if rec and rec.n_compounds > 0:
            auto_with_compounds += 1
        else:
            auto_name_only += 1

    # Staple verification
    staples_report = []
    for staple in STAPLES:
        rows = df[df["ingredient_string"].str.lower() == staple]
        if rows.empty:
            staples_report.append({"ingredient_string": staple, "status": "not_in_corpus"})
            continue
        r = rows.iloc[0]
        rec = index.get(int(r["foodb_food_id"])) if pd.notna(r.get("foodb_food_id")) else None
        staples_report.append(
            {
                "ingredient_string": staple,
                "bucket": r["bucket"],
                "match_method": r["match_method"],
                "foodb_food_id": int(r["foodb_food_id"]) if pd.notna(r.get("foodb_food_id")) else None,
                "species_name": rec.name if rec else None,
                "latin_name": rec.latin_name if rec else None,
                "n_compounds": rec.n_compounds if rec else 0,
                "old_fuzzy_canonical": old_name.get(staple),
                "old_fuzzy_ingredient_id": old_id.get(staple),
                "corrected": r["bucket"] == "auto_accepted" and old_name.get(staple) != (rec.name if rec else None),
            }
        )

    # Auto-accepted sample
    auto_sorted = auto.sort_values("recipe_occurrences", ascending=False)
    sample = []
    for _, r in auto_sorted.head(AUTO_SAMPLE_N).iterrows():
        rec = index.get(int(r["foodb_food_id"]))
        sample.append(
            {
                "ingredient_string": r["ingredient_string"],
                "recipe_occurrences": int(r["recipe_occurrences"]),
                "foodb_food_id": int(r["foodb_food_id"]),
                "canonical_name": rec.name if rec else None,
                "latin_name": rec.latin_name if rec else None,
                "match_method": r["match_method"],
            }
        )

    # 223 re-validation
    reval = df[df["in_old_223_graph"] == True]  # noqa: E712
    reval_corrected = 0
    reval_staple_fixes = []
    for _, r in reval.iterrows():
        s = r["ingredient_string"]
        if r["bucket"] == "auto_accepted" and old_name.get(s) and pd.notna(r.get("foodb_food_id")):
            rec = index.get(int(r["foodb_food_id"]))
            if rec and normalize_name(old_name[s]) != normalize_name(rec.name):
                reval_corrected += 1
                if r["recipe_occurrences"] >= 100:
                    reval_staple_fixes.append(
                        {
                            "ingredient_string": s,
                            "recipe_occurrences": int(r["recipe_occurrences"]),
                            "old_fuzzy": old_name.get(s),
                            "new_species": rec.name,
                            "foodb_food_id": int(r["foodb_food_id"]),
                        }
                    )

    # Corpus overlap for newly recovered auto-accepted species
    auto_fids = set(auto["foodb_food_id"].dropna().astype(int).unique())
    overlap_stats = {"n_species": len(auto_fids), "mechanistically_live": 0, "thin_mechanism": 0, "no_corpus_overlap": 0}
    overlap_examples = {"live": [], "thin": []}
    for fid in sorted(auto_fids):
        iks = food_iks.get(fid, set())
        n_measured = len(iks & measured_ik)
        n_predicted = len(iks & predicted_ik)
        n_corpus = len(iks & corpus_all)
        rec = index.get(fid)
        entry = {
            "foodb_food_id": fid,
            "name": rec.name if rec else None,
            "n_foodb_compounds": len(iks),
            "n_corpus_overlap": n_corpus,
            "n_measured_overlap": n_measured,
            "n_predicted_overlap": n_predicted,
        }
        if n_corpus >= 10:
            overlap_stats["mechanistically_live"] += 1
            if len(overlap_examples["live"]) < 10:
                overlap_examples["live"].append(entry)
        elif n_corpus > 0:
            overlap_stats["thin_mechanism"] += 1
            if len(overlap_examples["thin"]) < 10:
                overlap_examples["thin"].append(entry)
        else:
            overlap_stats["no_corpus_overlap"] += 1

    # Projected coverage vs current 223
    in223_occ = int(df.loc[df["in_old_223_graph"] == True, "recipe_occurrences"].sum())  # noqa: E712
    auto_occ = int(auto["recipe_occurrences"].sum())
    amb = df[df["bucket"] == "ambiguous"]
    amb_occ = int(amb["recipe_occurrences"].sum())
    amb_gated = amb[amb["recipe_occurrences"] >= AMBIGUOUS_FREQ_THRESHOLD]
    current_223_strings = int((df["in_old_223_graph"] == True).sum())  # noqa: E712

    report = {
        "phase": "3_STOP_FOR_HUMAN_REVIEW",
        "status": "NOT_MERGED_INTO_223",
        "inputs": {
            "raw_full": str(RAW_FULL.relative_to(ROOT)),
            "foodb_food": str(FOODB_FOOD.relative_to(ROOT)),
            "ingredient_compound_canonical_223": str(IC223.relative_to(ROOT)),
        },
        "corpus": {
            "distinct_ingredient_strings": int(len(df)),
            "total_recipe_occurrences": total_pairs,
            "current_223_graph_ingredients": 223,
            "current_223_mapped_strings_in_raw_full": current_223_strings,
            "current_223_recipe_occurrences_raw_full": in223_occ,
            "current_223_occurrence_pct": round(100 * in223_occ / total_pairs, 2),
        },
        "bucket_summary": bucket_summary,
        "auto_accepted": {
            "n_strings": int(len(auto)),
            "n_unique_species": int(auto_species),
            "recipe_occurrences": auto_occ,
            "occurrence_pct": round(100 * auto_occ / total_pairs, 2),
            "species_with_compound_data": auto_with_compounds,
            "species_name_only": auto_name_only,
            "sample_top_20": sample,
        },
        "ambiguous": {
            "n_strings_total": int(len(amb)),
            "recipe_occurrences_total": amb_occ,
            "n_strings_exported": int(len(amb)) if len(amb) <= 100 else int(len(amb_gated)),
            "freq_threshold": AMBIGUOUS_FREQ_THRESHOLD,
            "note": "Full ambiguous list in ambiguous_for_review_v1.parquet (gated) — rule on candidates before merge",
        },
        "composite": {
            "n_strings": int((df["bucket"] == "composite").sum()),
            "recipe_occurrences": int(df.loc[df["bucket"] == "composite", "recipe_occurrences"].sum()),
            "routed_to": "composites_for_recipe_layer_v1.parquet",
        },
        "unmatched": {
            "n_strings": int((df["bucket"] == "unmatched").sum()),
            "recipe_occurrences": int(df.loc[df["bucket"] == "unmatched", "recipe_occurrences"].sum()),
            "top_by_frequency": df[df["bucket"] == "unmatched"]
            .sort_values("recipe_occurrences", ascending=False)
            .head(30)[["ingredient_string", "recipe_occurrences"]]
            .to_dict(orient="records"),
        },
        "junk": {
            "n_strings": int((df["bucket"] == "junk").sum()),
            "recipe_occurrences": int(df.loc[df["bucket"] == "junk", "recipe_occurrences"].sum()),
        },
        "staple_verification": staples_report,
        "revalidate_223": {
            "strings_previously_in_223_graph": int(len(reval)),
            "auto_accepted_with_different_species_than_fuzzy": reval_corrected,
            "high_frequency_corrections": sorted(reval_staple_fixes, key=lambda x: -x["recipe_occurrences"])[:25],
        },
        "projected_coverage": {
            "auto_accepted_unique_species": int(auto_species),
            "auto_accepted_occurrence_pct": round(100 * auto_occ / total_pairs, 2),
            "if_all_ambiguous_approved_occurrence_pct": round(100 * (auto_occ + amb_occ) / total_pairs, 2),
            "if_ambiguous_gated_approved_occurrence_pct": round(
                100 * (auto_occ + int(amb_gated["recipe_occurrences"].sum())) / total_pairs, 2
            ),
            "vs_current_223_occurrence_pct": round(100 * in223_occ / total_pairs, 2),
            "note": "Approving ambiguous requires human ruling per candidate; projections are upper bounds",
        },
        "corpus_overlap_new_species": {
            **overlap_stats,
            "honest_note": (
                "Recovering staples correctly (e.g. salt) may still yield thin mechanism if FooDB compounds "
                "do not overlap measured/predicted corpus. n_corpus_overlap uses measured+predicted+223 compound sets."
            ),
            "examples_mechanistically_live": overlap_examples["live"],
            "examples_thin_mechanism": overlap_examples["thin"],
        },
        "match_method_counts": dict(Counter(auto["match_method"].dropna())),
        "next_step": "Human review: verify staples in auto_accepted sample, rule on ambiguous_for_review_v1.parquet, then approve merge separately",
    }
    return _json_safe(report)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[species_remap] Loading FooDB index...")
    index = load_foodb_index(FOODB_FOOD, FOODB_CONTENT)

    print("[species_remap] Loading ingredient string universe...")
    freq = load_string_universe()
    total_pairs = int(freq["recipe_occurrences"].sum())
    old_id, in223, old_name = load_old_fuzzy_context()

    print(f"[species_remap] Matching {len(freq):,} distinct strings...")
    rows = []
    for _, row in freq.iterrows():
        s = row["ingredient_string"]
        occ = int(row["recipe_occurrences"])
        result = match_ingredient_string(s, index)
        rec = index.get(result.food_id) if result.food_id else None
        rows.append(
            {
                "ingredient_string": s,
                "recipe_occurrences": occ,
                "source": row.get("source", "raw_full"),
                "bucket": result.bucket,
                "foodb_food_id": result.food_id,
                "canonical_name": rec.name if rec else None,
                "latin_name": rec.latin_name if rec else None,
                "match_method": result.match_method,
                "match_confidence": result.match_confidence,
                "modifier_stripped": result.stripped,
                "pre_class": result.pre_class,
                "n_compounds_foodb": rec.n_compounds if rec else 0,
                "candidate_species_json": json.dumps(result.candidates) if result.candidates else None,
                "old_fuzzy_ingredient_id": old_id.get(s),
                "old_fuzzy_canonical_name": old_name.get(s),
                "in_old_223_graph": bool(in223.get(s, False)),
            }
        )

    df = pd.DataFrame(rows)
    df, nodes_df = assign_species_node_ids(df, index)

    print("[species_remap] Computing corpus overlap...")
    food_iks = build_food_compound_inchikeys(FOODB_CONTENT, FOODB_COMPOUND)
    measured_ik, predicted_ik, ic223_ik = load_corpus_inchikeys()

    # Write outputs
    nodes_df.to_parquet(OUT_DIR / "species_nodes_v1.parquet", index=False)
    df.to_parquet(OUT_DIR / "ingredient_string_species_map_v1.parquet", index=False)

    auto = df[df["bucket"] == "auto_accepted"].copy()
    auto.to_parquet(OUT_DIR / "auto_accepted_v1.parquet", index=False)

    amb = df[df["bucket"] == "ambiguous"].copy()
    amb_gated = amb[amb["recipe_occurrences"] >= AMBIGUOUS_FREQ_THRESHOLD].copy()
    amb_export_df = amb if len(amb) <= 100 else amb_gated
    amb_export_df.to_parquet(OUT_DIR / "ambiguous_for_review_v1.parquet", index=False)
    amb_export = amb_export_df.sort_values("recipe_occurrences", ascending=False)
    amb_json = []
    for _, r in amb_export.iterrows():
        amb_json.append(
            {
                "ingredient_string": r["ingredient_string"],
                "recipe_occurrences": int(r["recipe_occurrences"]),
                "match_method": r["match_method"],
                "modifier_stripped": r["modifier_stripped"],
                "candidate_species": json.loads(r["candidate_species_json"]) if r["candidate_species_json"] else [],
            }
        )
    (OUT_DIR / "ambiguous_for_review_v1.json").write_text(json.dumps(amb_json, indent=2), encoding="utf-8")

    comp = df[df["bucket"] == "composite"].copy()
    comp.to_parquet(OUT_DIR / "composites_for_recipe_layer_v1.parquet", index=False)

    unmatched = df[df["bucket"] == "unmatched"].copy()
    unmatched.to_parquet(OUT_DIR / "unmatched_v1.parquet", index=False)

    junk = df[df["bucket"] == "junk"]
    junk_summary = {
        "n_strings": int(len(junk)),
        "recipe_occurrences": int(junk["recipe_occurrences"].sum()),
        "examples": junk.sort_values("recipe_occurrences", ascending=False)
        .head(50)[["ingredient_string", "recipe_occurrences"]]
        .to_dict(orient="records"),
    }
    (OUT_DIR / "junk_discarded_v1.json").write_text(json.dumps(junk_summary, indent=2), encoding="utf-8")

    report = build_report(
        df, nodes_df, index, total_pairs, old_id, old_name, in223,
        food_iks, measured_ik, predicted_ik, ic223_ik,
    )
    (OUT_DIR / "species_remap_report_v1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== SPECIES REMAP Phase 3 — STOP FOR REVIEW ===")
    for bucket, info in report["bucket_summary"].items():
        print(f"  {bucket:15s}  strings={info['n_strings']:>6,}  occ={info['recipe_occurrences']:>8,}  ({info['occurrence_pct_of_corpus']}%)")
    print(f"\n  Auto-accepted unique species: {report['auto_accepted']['n_unique_species']}")
    print(f"  Ambiguous for review: {report['ambiguous']['n_strings_exported']} / {report['ambiguous']['n_strings_total']}")
    print(f"\n  Outputs -> {OUT_DIR}")
    print("  STOP — do not merge into 223 until human review complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
