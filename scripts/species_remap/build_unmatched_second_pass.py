#!/usr/bin/env python3
"""
Second-pass recovery BUILD for high-frequency UNMATCHED ingredient strings.

Applies approved pepper ruling, triages unmatched head, proposes curated FooDB aliases
for recoverable real foods. STOP at Phase 4 for human review — no merge.

Usage (from repo root):
    python scripts/species_remap/build_unmatched_second_pass.py

Outputs (data/processed/species_remap/ only):
    pepper_ruling_applied_v1.parquet
    unmatched_head_triage_v1.parquet
    unmatched_second_pass_proposed_v1.parquet
    processed_no_phytochem_v1.parquet
    judgment_calls_for_review_v1.json
    unmatched_second_pass_report_v1.json
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

from scripts.species_remap.build_species_remap import (  # noqa: E402
    build_food_compound_inchikeys,
    load_corpus_inchikeys,
)
from scripts.species_remap.pepper_ruling import apply_pepper_ruling
from scripts.species_remap.second_pass_proposals import (
    lookup_processed,
    lookup_proposed,
    species_info,
)
from scripts.species_remap.species_match import load_foodb_index

OUT_DIR = ROOT / "data/processed/species_remap"
FOODB_FOOD = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Food.csv"
FOODB_CONTENT = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Content.csv"
COMPOUND_MASTER = ROOT / "data/processed/canonical/compound_master.csv"

HEAD_N = 500  # triage top-N by frequency
HEAD_MIN_OCC = 50  # also include all strings >= this occurrence
CORPUS_LIVE_THRESHOLD = 10
CORPUS_THIN_THRESHOLD = 1


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def corpus_overlap_for_food(
    food_id: int,
    food_iks: dict[int, set[str]],
    corpus: set[str],
    measured: set[str],
    predicted: set[str],
) -> dict[str, int]:
    iks = food_iks.get(food_id, set())
    return {
        "n_foodb_compounds_resolved": len(iks),
        "n_corpus_overlap": len(iks & corpus),
        "n_measured_overlap": len(iks & measured),
        "n_predicted_overlap": len(iks & predicted),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_foodb_index(FOODB_FOOD, FOODB_CONTENT)
    food_iks = build_food_compound_inchikeys(FOODB_CONTENT, COMPOUND_MASTER)
    measured, predicted, ic223 = load_corpus_inchikeys()
    corpus = measured | predicted | ic223

    full_map = pd.read_parquet(OUT_DIR / "ingredient_string_species_map_v1.parquet")
    unmatched = pd.read_parquet(OUT_DIR / "unmatched_v1.parquet")
    ambiguous = pd.read_parquet(OUT_DIR / "ambiguous_for_review_v1.parquet")
    auto = pd.read_parquet(OUT_DIR / "auto_accepted_v1.parquet")

    total_pairs = int(full_map["recipe_occurrences"].sum())
    unmatched_total_occ = int(unmatched["recipe_occurrences"].sum())

    # --- Apply pepper ruling to ambiguous strings ---
    pepper_rows = []
    for _, row in ambiguous.iterrows():
        ruling = apply_pepper_ruling(row["ingredient_string"])
        if ruling is None:
            continue
        fid = ruling["proposed_foodb_id"]
        rec = index.get(fid)
        ov = corpus_overlap_for_food(fid, food_iks, corpus, measured, predicted)
        pepper_rows.append(
            {
                "ingredient_string": row["ingredient_string"],
                "recipe_occurrences": int(row["recipe_occurrences"]),
                "proposed_foodb_id": fid,
                "proposed_species": ruling["proposed_species"],
                "latin_name": ruling["latin_name"],
                "reasoning": ruling["reasoning"],
                "judgment_call": ruling.get("judgment_call", False),
                "judgment_note": ruling.get("judgment_note"),
                "n_compounds_foodb": rec.n_compounds if rec else 0,
                **{f"corpus_{k}": v for k, v in ov.items()},
                "source": "pepper_ruling_applied",
            }
        )
    pepper_df = pd.DataFrame(pepper_rows)
    pepper_df.to_parquet(OUT_DIR / "pepper_ruling_applied_v1.parquet", index=False)

    # --- Triage unmatched head ---
    unm_sorted = unmatched.sort_values("recipe_occurrences", ascending=False)
    head_by_rank = unm_sorted.head(HEAD_N)
    head_by_occ = unm_sorted[unm_sorted["recipe_occurrences"] >= HEAD_MIN_OCC]
    head = pd.concat([head_by_rank, head_by_occ]).drop_duplicates("ingredient_string")
    head_occ = int(head["recipe_occurrences"].sum())

    triage_rows = []
    proposed_rows = []
    processed_rows = []
    judgment_rows = []

    for _, row in head.iterrows():
        s = row["ingredient_string"]
        occ = int(row["recipe_occurrences"])

        proc = lookup_processed(s)
        if proc:
            note, thin_fid = proc
            rec = index.get(thin_fid) if thin_fid else None
            ov = corpus_overlap_for_food(thin_fid, food_iks, corpus, measured, predicted) if thin_fid else {}
            triage_rows.append(
                {
                    "ingredient_string": s,
                    "recipe_occ": occ,
                    "bucket": "processed_no_phytochem",
                    "note": note,
                    "proposed_foodb_id": thin_fid,
                    "proposed_species": rec.name if rec else None,
                }
            )
            proc_entry = {
                "ingredient_string": s,
                "recipe_occ": occ,
                "category": "processed_no_phytochem",
                "note": note,
                "optional_foodb_id": thin_fid,
                "optional_foodb_name": rec.name if rec else None,
                "n_compounds_foodb": rec.n_compounds if rec else 0,
            }
            if ov:
                proc_entry.update({f"corpus_{k}": v for k, v in ov.items()})
            processed_rows.append(proc_entry)
            continue

        prop = lookup_proposed(s)
        if prop:
            fid = int(prop["proposed_foodb_id"])
            sp, latin = species_info(fid)
            rec = index.get(fid)
            ov = corpus_overlap_for_food(fid, food_iks, corpus, measured, predicted)
            entry = {
                "ingredient_string": s,
                "recipe_occ": occ,
                "proposed_foodb_id": fid,
                "proposed_species": sp or (rec.name if rec else None),
                "latin_name": latin or (rec.latin_name if rec else None),
                "reasoning": prop["reasoning"],
                "judgment_call": bool(prop.get("judgment_call", False)),
                "judgment_note": prop.get("judgment_note"),
                "has_compound_data": bool(rec and rec.n_compounds > 0),
                "n_compounds_foodb": rec.n_compounds if rec else 0,
                **{f"corpus_{k}": v for k, v in ov.items()},
            }
            triage_rows.append({**entry, "bucket": "recoverable_real_food", "recipe_occurrences": occ})
            proposed_rows.append(entry)
            if entry["judgment_call"]:
                judgment_rows.append(entry)
            continue

        triage_rows.append(
            {
                "ingredient_string": s,
                "recipe_occ": occ,
                "recipe_occurrences": occ,
                "bucket": "still_unmatched",
                "note": "No proposed alias; no FooDB exact/parent match in second-pass table",
            }
        )

    triage_df = pd.DataFrame(triage_rows)
    triage_df.to_parquet(OUT_DIR / "unmatched_head_triage_v1.parquet", index=False)

    proposed_df = pd.DataFrame(proposed_rows).sort_values("recipe_occ", ascending=False)
    proposed_df.to_parquet(OUT_DIR / "unmatched_second_pass_proposed_v1.parquet", index=False)

    processed_df = pd.DataFrame(processed_rows).sort_values("recipe_occ", ascending=False)
    processed_df.to_parquet(OUT_DIR / "processed_no_phytochem_v1.parquet", index=False)

    # Pepper judgment calls (hot pepper -> Capsicum vs Chili)
    pepper_judgments = pepper_df[pepper_df["judgment_call"] == True].to_dict(orient="records")  # noqa: E712
    all_judgments = judgment_rows + pepper_judgments
    (OUT_DIR / "judgment_calls_for_review_v1.json").write_text(
        json.dumps(_json_safe(all_judgments), indent=2), encoding="utf-8"
    )

    # Bucket stats on head
    bucket_stats = {}
    for bucket, g in triage_df.groupby("bucket"):
        bucket_stats[bucket] = {
            "n_strings": int(len(g)),
            "recipe_occurrences": int(g["recipe_occ"].sum() if "recipe_occ" in g else g["recipe_occurrences"].sum()),
        }
    for b in bucket_stats:
        bucket_stats[b]["pct_of_unmatched_total"] = round(100 * bucket_stats[b]["recipe_occurrences"] / unmatched_total_occ, 2)
        bucket_stats[b]["pct_of_unmatched_head"] = round(100 * bucket_stats[b]["recipe_occurrences"] / head_occ, 2)

    # Corpus overlap for proposed recoverable species
    prop_fids = proposed_df["proposed_foodb_id"].dropna().astype(int).unique() if len(proposed_df) else []
    live, thin, none = 0, 0, 0
    for fid in prop_fids:
        n = len(food_iks.get(fid, set()) & corpus)
        if n >= CORPUS_LIVE_THRESHOLD:
            live += 1
        elif n >= CORPUS_THIN_THRESHOLD:
            thin += 1
        else:
            none += 1

    # Projected coverage
    auto_occ = int(auto["recipe_occurrences"].sum())
    auto_species = int(auto["foodb_food_id"].nunique())
    pepper_occ = int(pepper_df["recipe_occurrences"].sum()) if len(pepper_df) else 0
    prop_occ = int(proposed_df["recipe_occ"].sum()) if len(proposed_df) else 0
    prop_unique_fids = set(proposed_df["proposed_foodb_id"].astype(int).tolist()) if len(proposed_df) else set()
    pepper_fids = set(pepper_df["proposed_foodb_id"].astype(int).tolist()) if len(pepper_df) else set()
    auto_fids = set(auto["foodb_food_id"].dropna().astype(int).tolist())

    merged_species = len(auto_fids | pepper_fids | prop_unique_fids)
    merged_occ = auto_occ + pepper_occ + prop_occ
    merged_occ_pct = round(100 * merged_occ / total_pairs, 2)

    report = {
        "phase": "4_STOP_FOR_HUMAN_REVIEW",
        "status": "NOT_MERGED",
        "triage_scope": {
            "unmatched_total_strings": int(len(unmatched)),
            "unmatched_total_occurrences": unmatched_total_occ,
            "unmatched_pct_of_corpus": round(100 * unmatched_total_occ / total_pairs, 2),
            "head_n_by_rank": HEAD_N,
            "head_min_occurrence": HEAD_MIN_OCC,
            "head_strings_triaged": int(len(head)),
            "head_occurrences": head_occ,
            "head_pct_of_unmatched_occ": round(100 * head_occ / unmatched_total_occ, 2),
            "zipf_note": f"Top {HEAD_N} strings cover {round(100 * int(head_by_rank['recipe_occurrences'].sum()) / unmatched_total_occ, 1)}% of unmatched occurrences",
        },
        "pepper_ruling_applied": {
            "n_strings": int(len(pepper_df)),
            "recipe_occurrences": pepper_occ,
            "occurrence_pct_of_corpus": round(100 * pepper_occ / total_pairs, 2),
            "by_target": dict(Counter(pepper_df["proposed_foodb_id"].astype(int).tolist())) if len(pepper_df) else {},
            "output": "pepper_ruling_applied_v1.parquet",
        },
        "head_bucket_summary": bucket_stats,
        "recoverable_real_food": {
            "n_proposed_aliases": int(len(proposed_df)),
            "recipe_occurrences": prop_occ,
            "n_unique_foodb_species": int(len(prop_unique_fids)),
            "n_judgment_calls": int(proposed_df["judgment_call"].sum()) if len(proposed_df) else 0,
            "top_20_by_frequency": proposed_df.head(20).to_dict(orient="records") if len(proposed_df) else [],
            "output": "unmatched_second_pass_proposed_v1.parquet",
        },
        "processed_no_phytochem": {
            "n_strings_in_head": int(len(processed_df)),
            "recipe_occurrences": int(processed_df["recipe_occ"].sum()) if len(processed_df) else 0,
            "top_examples": processed_df.head(15).to_dict(orient="records") if len(processed_df) else [],
            "policy": "Marked honestly as outside phytochemical mechanism model; not force-matched",
            "output": "processed_no_phytochem_v1.parquet",
        },
        "still_unmatched_head": {
            "n_strings": int((triage_df["bucket"] == "still_unmatched").sum()),
            "top_by_frequency": triage_df[triage_df["bucket"] == "still_unmatched"]
            .sort_values("recipe_occ", ascending=False)
            .head(25)
            .to_dict(orient="records"),
        },
        "corpus_overlap_proposed_recoverable": {
            "n_unique_species_proposed": int(len(prop_fids)),
            "mechanistically_live_gte_10": live,
            "thin_mechanism_1_to_9": thin,
            "no_corpus_overlap": none,
            "honest_note": "Some recoverable foods (cheese, spirits, syrup) may be real nodes but thin on measured/predicted corpus",
        },
        "projected_coverage_if_approved": {
            "current_pass1_auto_species": auto_species,
            "current_pass1_occurrence_pct": round(100 * auto_occ / total_pairs, 2),
            "plus_pepper_ruling_occurrence_pct": round(100 * (auto_occ + pepper_occ) / total_pairs, 2),
            "plus_second_pass_proposed_occurrence_pct": merged_occ_pct,
            "projected_unique_species": merged_species,
            "incremental_species_vs_pass1": merged_species - auto_species,
            "note": "Occurrence counts sum strings (same recipe may count multiple strings); species count is unique foodb_food_id",
        },
        "next_step": "Review unmatched_second_pass_proposed_v1.parquet and judgment_calls_for_review_v1.json; rule on judgment calls; approve merge separately",
    }
    (OUT_DIR / "unmatched_second_pass_report_v1.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    print("\n=== UNMATCHED SECOND PASS Phase 4 — STOP FOR REVIEW ===")
    print(f"  Head triaged: {len(head)} strings ({head_occ:,} occ, {100*head_occ/unmatched_total_occ:.1f}% of unmatched)")
    print(f"  Pepper ruling applied: {len(pepper_df)} strings ({pepper_occ:,} occ)")
    for bucket, info in bucket_stats.items():
        print(f"  {bucket:25s}  n={info['n_strings']:>4}  occ={info['recipe_occurrences']:>7,}  ({info['pct_of_unmatched_head']:.1f}% of head)")
    print(f"\n  Proposed recoverable aliases: {len(proposed_df)} ({prop_occ:,} occ)")
    print(f"  Judgment calls: {len(all_judgments)}")
    print(f"  Projected coverage if approved: {merged_occ_pct}% (vs pass1 {100*auto_occ/total_pairs:.1f}%)")
    print(f"  Outputs -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
