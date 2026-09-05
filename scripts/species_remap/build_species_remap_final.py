#!/usr/bin/env python3
"""
Finalize species remap — assemble merge-ready set with preparation labels.

Combines pass-1 auto-accepted + pepper ruling + approved second-pass recoverables.
Does NOT merge into 223-graph or measured canonical files.

Usage (from repo root):
    python scripts/species_remap/build_species_remap_final.py

Outputs (data/processed/species_remap/ only):
    species_remap_final_v1.parquet
    species_remap_final_nodes_v1.parquet
    species_remap_merge_impact_v1.json
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
from scripts.species_remap.preparation_labels import derive_preparation_label, merge_annotation
from scripts.species_remap.species_match import load_foodb_index, normalize_name

OUT_DIR = ROOT / "data/processed/species_remap"
FOODB_FOOD = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Food.csv"
FOODB_CONTENT = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Content.csv"
COMPOUND_MASTER = ROOT / "data/processed/canonical/compound_master.csv"
IC223 = ROOT / "data/processed/canonical/ingredient_compound_canonical.csv"
INGREDIENTS = ROOT / "data/processed/canonical/ingredients.parquet"
MEASURED_CG = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"

CORPUS_LIVE = 10


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def load_final_rows() -> list[dict[str, Any]]:
    auto = pd.read_parquet(OUT_DIR / "auto_accepted_v1.parquet")
    pepper = pd.read_parquet(OUT_DIR / "pepper_ruling_applied_v1.parquet")
    second = pd.read_parquet(OUT_DIR / "unmatched_second_pass_proposed_v1.parquet")

    rows: list[dict[str, Any]] = []

    for _, r in auto.iterrows():
        rows.append(
            {
                "ingredient_string": r["ingredient_string"],
                "recipe_occurrences": int(r["recipe_occurrences"]),
                "foodb_id": int(r["foodb_food_id"]),
                "canonical_name": r["canonical_name"],
                "latin_name": r["latin_name"] if pd.notna(r["latin_name"]) else None,
                "match_method": r["match_method"],
                "source_pass": "pass1_auto_accepted",
                "old_fuzzy_ingredient_id": r.get("old_fuzzy_ingredient_id"),
                "old_fuzzy_canonical_name": r.get("old_fuzzy_canonical_name"),
                "in_old_223_graph": bool(r.get("in_old_223_graph", False)),
            }
        )

    for _, r in pepper.iterrows():
        rows.append(
            {
                "ingredient_string": r["ingredient_string"],
                "recipe_occurrences": int(r["recipe_occurrences"]),
                "foodb_id": int(r["proposed_foodb_id"]),
                "canonical_name": r["proposed_species"],
                "latin_name": r["latin_name"] if pd.notna(r["latin_name"]) else None,
                "match_method": "pepper_ruling",
                "source_pass": "pepper_ruling",
                "old_fuzzy_ingredient_id": None,
                "old_fuzzy_canonical_name": None,
                "in_old_223_graph": False,
            }
        )

    for _, r in second.iterrows():
        rows.append(
            {
                "ingredient_string": r["ingredient_string"],
                "recipe_occurrences": int(r["recipe_occ"]),
                "foodb_id": int(r["proposed_foodb_id"]),
                "canonical_name": r["proposed_species"],
                "latin_name": r["latin_name"] if pd.notna(r["latin_name"]) else None,
                "match_method": "second_pass_alias",
                "source_pass": "second_pass_recoverable",
                "old_fuzzy_ingredient_id": None,
                "old_fuzzy_canonical_name": None,
                "in_old_223_graph": False,
                "second_pass_reasoning": r.get("reasoning"),
            }
        )

    return rows


def assign_species_nodes(df: pd.DataFrame, index) -> tuple[pd.DataFrame, pd.DataFrame]:
    uniq_fids = sorted(df["foodb_id"].astype(int).unique())
    fid_to_node = {fid: f"SP_{i:06d}" for i, fid in enumerate(uniq_fids, start=1)}
    nodes = []
    for fid in uniq_fids:
        rec = index.get(fid)
        nodes.append(
            {
                "species_node_id": fid_to_node[fid],
                "foodb_id": fid,
                "canonical_name": rec.name if rec else None,
                "latin_name": rec.latin_name if rec else None,
                "n_compounds_foodb": rec.n_compounds if rec else 0,
            }
        )
    df = df.copy()
    df["species_node"] = df["foodb_id"].map(fid_to_node)
    return df, pd.DataFrame(nodes)


def mechanism_tier(n_corpus: int) -> str:
    if n_corpus >= CORPUS_LIVE:
        return "mechanistically_live"
    if n_corpus >= 1:
        return "thin_mechanism"
    return "no_corpus_overlap"


def build_impact_report(
    final: pd.DataFrame,
    nodes: pd.DataFrame,
    total_pairs: int,
    food_iks: dict[int, set[str]],
    measured: set[str],
    predicted: set[str],
    ic223_ik: set[str],
) -> dict[str, Any]:
    corpus = measured | predicted | ic223_ik
    ic223_ids = set(pd.read_csv(IC223)["ingredient_id"].astype(str).unique())
    ing = pd.read_parquet(INGREDIENTS)
    ing223 = ing[ing["ingredient_id"].astype(str).isin(ic223_ids)]

    # Species-level mechanism tiers
    species_tiers: dict[int, str] = {}
    for fid in final["foodb_id"].astype(int).unique():
        n = len(food_iks.get(fid, set()) & corpus)
        species_tiers[fid] = mechanism_tier(n)

    final = final.copy()
    final["mechanism_tier"] = final["foodb_id"].map(species_tiers)

    tier_counts = Counter(final["mechanism_tier"])
    tier_occ = final.groupby("mechanism_tier")["recipe_occurrences"].sum().to_dict()

    # Preparation retention stats
    prep_by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fid in final["foodb_id"].astype(int).unique():
        sub = final[final["foodb_id"] == fid]
        preps = sub["preparation_label"].dropna().unique().tolist()
        if preps:
            rec_name = sub["canonical_name"].iloc[0]
            prep_by_species[rec_name] = [
                {
                    "preparation_label": p,
                    "n_strings": int((sub["preparation_label"] == p).sum()),
                    "recipe_occurrences": int(sub.loc[sub["preparation_label"] == p, "recipe_occurrences"].sum()),
                }
                for p in sorted(preps)
            ]

    prep_summary = []
    for species_name, items in sorted(prep_by_species.items(), key=lambda x: -sum(i["recipe_occurrences"] for i in x[1])):
        prep_summary.append(
            {
                "species": species_name,
                "foodb_id": int(final.loc[final["canonical_name"] == species_name, "foodb_id"].iloc[0]),
                "n_distinct_preparation_labels": len(items),
                "n_strings_with_preparation_label": int(sum(i["n_strings"] for i in items)),
                "preparations": items[:20],
            }
        )

    # 223 re-map impact via strings previously in 223 graph
    in223 = final[final["in_old_223_graph"] == True]  # noqa: E712
    remapped_223 = []
    for _, r in in223.iterrows():
        old = r.get("old_fuzzy_canonical_name")
        new = r.get("canonical_name")
        if pd.notna(old) and pd.notna(new) and normalize_name(str(old)) != normalize_name(str(new)):
            remapped_223.append(
                {
                    "ingredient_string": r["ingredient_string"],
                    "recipe_occurrences": int(r["recipe_occurrences"]),
                    "old_fuzzy_canonical": old,
                    "new_species": new,
                    "foodb_id": int(r["foodb_id"]),
                    "preparation_label": r.get("preparation_label"),
                }
            )

    staples_check = []
    for s in ["salt", "sugar", "flour", "water", "milk", "chicken", "rice", "black pepper"]:
        hit = final[final["ingredient_string"].str.lower() == s]
        if len(hit):
            r = hit.iloc[0]
            staples_check.append(
                {
                    "ingredient_string": s,
                    "foodb_id": int(r["foodb_id"]),
                    "species": r["canonical_name"],
                    "preparation_label": r.get("preparation_label"),
                    "old_fuzzy": r.get("old_fuzzy_canonical_name"),
                }
            )

    measured_cg_path = str(IC223.relative_to(ROOT))
    measured_edges = len(pd.read_csv(IC223))

    return {
        "phase": "FINAL_MERGE_READY_STOP",
        "status": "NOT_MERGED_INTO_GRAPH",
        "final_set": {
            "n_ingredient_strings": int(len(final)),
            "n_unique_species_nodes": int(final["foodb_id"].nunique()),
            "recipe_occurrences": int(final["recipe_occurrences"].sum()),
            "recipe_occurrence_pct_of_corpus": round(100 * final["recipe_occurrences"].sum() / total_pairs, 2),
            "by_source_pass": dict(Counter(final["source_pass"])),
        },
        "mechanism_tier_breakdown": {
            "by_string_count": dict(tier_counts),
            "by_recipe_occurrences": {k: int(v) for k, v in tier_occ.items()},
            "unique_species_mechanistically_live": int(sum(1 for t in species_tiers.values() if t == "mechanistically_live")),
            "unique_species_thin": int(sum(1 for t in species_tiers.values() if t == "thin_mechanism")),
            "unique_species_no_overlap": int(sum(1 for t in species_tiers.values() if t == "no_corpus_overlap")),
        },
        "preparation_label_retention": {
            "n_strings_with_preparation_label": int(final["preparation_label"].notna().sum()),
            "n_strings_without_preparation_label": int(final["preparation_label"].isna().sum()),
            "n_distinct_preparation_labels": int(final["preparation_label"].nunique(dropna=True)),
            "top_species_by_preparation_diversity": prep_summary[:15],
            "sus_scrofa_preparations": prep_by_species.get("Domestic pig") or prep_by_species.get(
                final.loc[final["foodb_id"] == 549, "canonical_name"].iloc[0] if (final["foodb_id"] == 549).any() else ""
            ),
            "flour_preparations": next((p["preparations"] for p in prep_summary if p.get("foodb_id") == 825), []),
        },
        "merge_annotations": final[final["merge_annotation"].notna()][
            ["ingredient_string", "foodb_id", "canonical_name", "merge_annotation"]
        ].to_dict(orient="records"),
        "pre_merge_impact": {
            "current_graph_ingredient_count": 223,
            "projected_species_node_count_after_merge": int(final["foodb_id"].nunique()),
            "projected_ingredient_string_mappings": int(len(final)),
            "ingredient_id_expansion_note": (
                "Merge would replace fuzzy ING_* identity with foodb_id species nodes; "
                "ingredient_compound links would be rebuilt from FooDB Content per species, not from old fuzzy ING_* names."
            ),
            "strings_previously_in_223_fuzzy_set": int(in223.shape[0]),
            "strings_with_different_species_than_fuzzy": len(remapped_223),
            "high_frequency_223_corrections": sorted(remapped_223, key=lambda x: -x["recipe_occurrences"])[:30],
            "staple_verification": staples_check,
            "measured_compound_gene_edges_UNCHANGED": {
                "confirmed": True,
                "measured_edge_file": measured_cg_path,
                "measured_edge_count": measured_edges,
                "measured_cg_normalized_file": str(MEASURED_CG.relative_to(ROOT)),
                "statement": (
                    "This build does NOT modify ingredient_compound_canonical.csv, compound_gene_expanded_canonical*.csv, "
                    "or any measured compound→gene edge. Remap only prepares ingredient_string→species (FooDB) identity. "
                    "Gene sets, enrichment, and graph merge are separate explicit steps."
                ),
            },
        },
        "excluded_not_in_final": {
            "processed_no_phytochem": "processed_no_phytochem_v1.parquet (baking powder/soda/yeast/additives)",
            "composites": "composites_for_recipe_layer_v1.parquet (recipe layer, later)",
            "still_unmatched": "unmatched_v1.parquet minus approved second-pass recoverables",
        },
        "outputs": {
            "merge_ready_mapping": "species_remap_final_v1.parquet",
            "species_nodes": "species_remap_final_nodes_v1.parquet",
            "impact_report": "species_remap_merge_impact_v1.json",
        },
        "next_step": "Review species_remap_merge_impact_v1.json and species_remap_final_v1.parquet; approve explicit merge step separately",
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_foodb_index(FOODB_FOOD, FOODB_CONTENT)
    food_iks = build_food_compound_inchikeys(FOODB_CONTENT, COMPOUND_MASTER)
    measured, predicted, ic223_ik = load_corpus_inchikeys()
    corpus = measured | predicted | ic223_ik

    full_map = pd.read_parquet(OUT_DIR / "ingredient_string_species_map_v1.parquet")
    total_pairs = int(full_map["recipe_occurrences"].sum())

    rows = load_final_rows()
    df = pd.DataFrame(rows)

    # Preparation labels + merge annotations
    df["preparation_label"] = df.apply(
        lambda r: derive_preparation_label(r["ingredient_string"], int(r["foodb_id"])), axis=1
    )
    df["merge_annotation"] = df["ingredient_string"].map(merge_annotation)

    # Corpus overlap per row (species-level)
    def row_corpus(fid: int) -> int:
        return len(food_iks.get(int(fid), set()) & corpus)

    df["corpus_overlap_n"] = df["foodb_id"].map(row_corpus)
    df["mechanism_tier"] = df["corpus_overlap_n"].map(mechanism_tier)

    df, nodes = assign_species_nodes(df, index)

    # Output schema — user-required columns first
    out_cols = [
        "ingredient_string",
        "species_node",
        "latin_name",
        "foodb_id",
        "preparation_label",
        "match_method",
        "canonical_name",
        "recipe_occurrences",
        "source_pass",
        "merge_annotation",
        "mechanism_tier",
        "corpus_overlap_n",
        "in_old_223_graph",
        "old_fuzzy_canonical_name",
        "old_fuzzy_ingredient_id",
    ]
    final = df[out_cols].sort_values("recipe_occurrences", ascending=False)
    final.to_parquet(OUT_DIR / "species_remap_final_v1.parquet", index=False)
    nodes.to_parquet(OUT_DIR / "species_remap_final_nodes_v1.parquet", index=False)

    report = build_impact_report(final, nodes, total_pairs, food_iks, measured, predicted, ic223_ik)
    (OUT_DIR / "species_remap_merge_impact_v1.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    print("\n=== SPECIES REMAP FINAL — MERGE-READY (NOT MERGED) ===")
    print(f"  Ingredient strings: {len(final):,}")
    print(f"  Unique species nodes: {final['foodb_id'].nunique()}")
    print(f"  Recipe coverage: {report['final_set']['recipe_occurrence_pct_of_corpus']}%")
    print(f"  With preparation_label: {report['preparation_label_retention']['n_strings_with_preparation_label']}")
    print(f"  Mechanism live strings: {report['mechanism_tier_breakdown']['by_string_count'].get('mechanistically_live', 0)}")
    print(f"  223 fuzzy corrections: {report['pre_merge_impact']['strings_with_different_species_than_fuzzy']}")
    print(f"  Measured CG edges touched: NO ({report['pre_merge_impact']['measured_compound_gene_edges_UNCHANGED']['measured_edge_count']} unchanged)")
    print(f"  Outputs -> {OUT_DIR}")
    print("  STOP — await explicit merge approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
