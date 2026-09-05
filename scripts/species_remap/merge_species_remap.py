#!/usr/bin/env python3
"""
MERGE species remap into ingredient layer (v2).

Backs up v1 identity files, builds ingredient_compound_canonical_v2 from
463 FooDB species nodes + Content compounds. Does NOT modify measured
compound→gene edges or integrated/predicted layers.

Usage (from repo root):
    python scripts/species_remap/merge_species_remap.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
from scripts.species_remap.species_match import load_foodb_index

CANONICAL = ROOT / "data/processed/canonical"
ARCHIVE = CANONICAL / "archive" / "pre_species_merge"
SPECIES_REMAP = ROOT / "data/processed/species_remap"
INTERIM = ROOT / "data/interim/recipenlg"
INTEGRATED = ROOT / "data/processed/integrated"
THREAD2_INF = ROOT / "data/processed/thread2/inference"

FINAL_MAP = SPECIES_REMAP / "species_remap_final_v1.parquet"
FINAL_NODES = SPECIES_REMAP / "species_remap_final_nodes_v1.parquet"
RAW_FULL = INTERIM / "recipe_ingredients_raw_full.parquet"
FOODB_CONTENT = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Content.csv"
COMPOUND_MASTER = CANONICAL / "compound_master.csv"
FOODB_FOOD = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Food.csv"

V1_ICC = CANONICAL / "ingredient_compound_canonical.csv"
V2_ICC_CSV = CANONICAL / "ingredient_compound_canonical_v2.csv"
V2_ICC_PQ = CANONICAL / "ingredient_compound_canonical_v2.parquet"
V2_NODES = CANONICAL / "species_nodes_v2.parquet"
V2_STRING_MAP = CANONICAL / "ingredient_string_species_v2.parquet"
VERIFY_OUT = CANONICAL / "species_merge_verification_v1.json"
PROVENANCE = CANONICAL / "INGREDIENT_LAYER_PROVENANCE.md"

MEASURED_CG = CANONICAL / "compound_gene_expanded_canonical_normalized.csv"
STAPLES = {
    "salt": 666,
    "sugar": 670,
    "flour": 825,
    "water": 685,
    "milk": 632,
    "chicken": 334,
    "rice": 125,
    "black pepper": 139,
}

BACKUP_FILES = [
    V1_ICC,
    CANONICAL / "ingredients.parquet",
    INTERIM / "recipe_ingredients_mapped_full.parquet",
    INTERIM / "recipe_ingredients_mapped.parquet",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(paths: list[Path]) -> dict[str, Any]:
    out = {}
    for p in paths:
        if p.exists():
            out[str(p.relative_to(ROOT))] = {
                "exists": True,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "rows": _row_count(p),
            }
        else:
            out[str(p.relative_to(ROOT))] = {"exists": False}
    return out


def _row_count(p: Path) -> int | None:
    if p.suffix.lower() == ".csv":
        return sum(1 for _ in open(p, encoding="utf-8", errors="replace")) - 1
    if p.suffix.lower() == ".parquet":
        return len(pd.read_parquet(p))
    return None


def backup_originals() -> dict[str, Any]:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    manifest_before = file_manifest(BACKUP_FILES)
    copied = []
    for src in BACKUP_FILES:
        if not src.exists():
            continue
        dst = ARCHIVE / src.name
        if dst.exists():
            if sha256_file(src) != sha256_file(dst):
                raise RuntimeError(f"Backup exists but differs from source: {dst}")
        else:
            shutil.copy2(src, dst)
        if sha256_file(src) != sha256_file(dst):
            raise RuntimeError(f"Backup integrity failed for {src.name}")
        copied.append(str(dst.relative_to(ROOT)))

    # Also write explicit v1 rename backup for ICC
    icc_backup = CANONICAL / "ingredient_compound_canonical_pre_species_merge.csv"
    if not icc_backup.exists():
        shutil.copy2(V1_ICC, icc_backup)
    if sha256_file(V1_ICC) != sha256_file(icc_backup):
        raise RuntimeError("pre_species_merge CSV backup hash mismatch")

    manifest_after = file_manifest([ARCHIVE / p.name for p in BACKUP_FILES if p.exists()] + [icc_backup])
    return {
        "archive_dir": str(ARCHIVE.relative_to(ROOT)),
        "icc_backup": str(icc_backup.relative_to(ROOT)),
        "files_copied": copied,
        "manifest_source": manifest_before,
        "manifest_backup": manifest_after,
    }


def build_v2_ingredient_compound(
    nodes: pd.DataFrame,
    food_iks: dict[int, set[str]],
) -> pd.DataFrame:
    rows = []
    for _, n in nodes.iterrows():
        sp = n["species_node_id"]
        fid = int(n["foodb_id"])
        for ik in sorted(food_iks.get(fid, set())):
            rows.append({"ingredient_id": sp, "compound_id": ik, "foodb_id": fid})
    return pd.DataFrame(rows)


def build_species_nodes_v2(final_map: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    prep_agg = (
        final_map.groupby(["species_node", "foodb_id"])
        .agg(
            preparation_labels=("preparation_label", lambda s: sorted({x for x in s.dropna().astype(str)})),
            n_strings=("ingredient_string", "count"),
            n_preparation_labels=("preparation_label", "nunique"),
        )
        .reset_index()
        .rename(columns={"species_node": "species_node_id"})
    )
    out = nodes.merge(prep_agg, on=["species_node_id", "foodb_id"], how="left")
    out["preparation_labels"] = out["preparation_labels"].apply(lambda x: x if isinstance(x, list) else [])
    return out


def verify_merge(
    v2_icc: pd.DataFrame,
    final_map: pd.DataFrame,
    nodes_v2: pd.DataFrame,
    backup_info: dict[str, Any],
    pre_hashes: dict[str, str],
) -> dict[str, Any]:
    raw = pd.read_parquet(RAW_FULL)
    total_pairs = len(raw)

    # Coverage: sum occurrences of mapped strings in corpus
    mapped_strings = set(final_map["ingredient_string"])
    covered = raw[raw["ingredient_raw"].isin(mapped_strings)]
    coverage_pct = round(100 * len(covered) / total_pairs, 2)

    # Staples
    staples = []
    for name, expected_fid in STAPLES.items():
        hit = final_map[final_map["ingredient_string"].str.lower() == name]
        ok = False
        sp = None
        if len(hit):
            row = hit.iloc[0]
            ok = int(row["foodb_id"]) == expected_fid
            sp = row["species_node"]
        staples.append(
            {
                "ingredient_string": name,
                "expected_foodb_id": expected_fid,
                "actual_foodb_id": int(hit.iloc[0]["foodb_id"]) if len(hit) else None,
                "species_node": sp,
                "pass": ok,
            }
        )

    measured, predicted, ic223 = load_corpus_inchikeys()
    corpus = measured | predicted | ic223

    food_iks = build_food_compound_inchikeys(FOODB_CONTENT, COMPOUND_MASTER)
    no_overlap_fids = (
        final_map[final_map["mechanism_tier"] == "no_corpus_overlap"]["foodb_id"].astype(int).unique().tolist()
    )
    no_overlap_present = sorted(
        int(fid)
        for fid in no_overlap_fids
        if fid in set(nodes_v2["foodb_id"].astype(int))
    )

    post_hashes = file_manifest(
        [
            MEASURED_CG,
            V1_ICC,
            INTEGRATED / "compound_gene_integrated_v1.parquet",
            INTEGRATED / "ingredient_gene_sets_v2.parquet",
            THREAD2_INF / "predicted_compound_gene_weighted_v2.parquet",
        ]
    )

    unchanged_proof = {}
    for rel, info in pre_hashes.items():
        post = post_hashes.get(rel, {})
        unchanged_proof[rel] = {
            "pre_sha256": info.get("sha256"),
            "post_sha256": post.get("sha256"),
            "unchanged": info.get("sha256") == post.get("sha256"),
            "pre_rows": info.get("rows"),
            "post_rows": post.get("rows"),
        }

    n_prep_labels = int(final_map["preparation_label"].nunique(dropna=True))
    pig = sorted(final_map.loc[final_map["foodb_id"] == 549, "preparation_label"].dropna().unique().tolist())

    return {
        "merge_status": "COMPLETE_NOT_REBUILDING_GENE_SETS",
        "ingredient_layer_v2": {
            "n_species_nodes": int(nodes_v2["species_node_id"].nunique()),
            "n_species_with_compound_edges": int(v2_icc["ingredient_id"].nunique()),
            "n_species_zero_compound_edges": int(nodes_v2["species_node_id"].nunique() - v2_icc["ingredient_id"].nunique()),
            "n_compound_ids": int(v2_icc["compound_id"].nunique()),
            "n_ingredient_compound_edges": int(len(v2_icc)),
            "n_ingredient_strings_mapped": int(len(final_map)),
            "recipe_occurrence_coverage_pct": coverage_pct,
            "expected_species_count": 463,
            "species_count_match": int(nodes_v2["species_node_id"].nunique()) == 463,
        },
        "staple_spot_check": staples,
        "all_staples_pass": all(s["pass"] for s in staples),
        "measured_mechanism_unchanged_proof": unchanged_proof,
        "all_mechanism_files_unchanged": all(v["unchanged"] for v in unchanged_proof.values() if v.get("pre_sha256")),
        "preparation_labels": {
            "n_distinct_preparation_labels": n_prep_labels,
            "expected": 449,
            "match": n_prep_labels == 449,
            "sus_scrofa_preparation_labels": sorted(pig),
            "n_sus_scrofa_preparations": len(pig),
        },
        "no_corpus_overlap_species": {
            "expected_count": 18,
            "present_in_v2_nodes": no_overlap_present,
            "count_present": len(no_overlap_present),
            "all_present": len(no_overlap_present) == 18,
        },
        "backup_integrity": backup_info,
        "v1_preserved": {
            "ingredient_compound_canonical_csv": str(V1_ICC.relative_to(ROOT)),
            "unchanged": unchanged_proof.get(str(V1_ICC.relative_to(ROOT)), {}).get("unchanged"),
        },
    }


def write_provenance(backup_info: dict[str, Any], verify: dict[str, Any]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = f"""# Ingredient Layer Provenance — Species Remap Merge

**Merge date:** {ts}  
**Source:** `data/processed/species_remap/species_remap_final_v1.parquet`  
**Status:** v2 ingredient layer written; v1 originals preserved.

## What changed

- Ingredient identity remapped from fuzzy 223-ingredient graph → **463 FooDB species nodes** (`SP_*` ids).
- **2,078** clean ingredient strings with preparation labels retained (`preparation_label` column).
- **561** strings corrected vs. old fuzzy 223 mapping (including staples: salt, sugar, flour, water, milk, chicken, rice, pepper).
- Fuzzy mapper superseded by exact FooDB species matching + curated aliases (see species_remap pipeline).

## What did NOT change

- **Measured compound→gene edges** (`compound_gene_expanded_canonical_normalized.csv`) — untouched.
- Integrated/predicted compound→gene layers — untouched.
- Ingredient gene sets and enrichment — **not rebuilt** (next step).

## v2 outputs

| File | Role |
|------|------|
| `ingredient_compound_canonical_v2.csv` / `.parquet` | Species node → compound InChIKey (FooDB Content) |
| `species_nodes_v2.parquet` | 463 species nodes + preparation label lists |
| `ingredient_string_species_v2.parquet` | String → species mapping with preparation labels |
| `species_merge_verification_v1.json` | Post-merge verification report |

## v1 backups (revert path)

| Backup | Path |
|--------|------|
| Archive copy | `{backup_info["archive_dir"]}/` |
| ICC explicit backup | `{backup_info["icc_backup"]}` |

**One-line revert:** Point downstream consumers back to `ingredient_compound_canonical.csv` (v1) and delete or ignore `*_v2*` files; measured mechanism files were never modified.

## Verification summary

- Species count: {verify["ingredient_layer_v2"]["n_species_nodes"]} (expected 463)
- Recipe coverage: {verify["ingredient_layer_v2"]["recipe_occurrence_coverage_pct"]}%
- Staples pass: {verify["all_staples_pass"]}
- Measured CG unchanged: {verify["all_mechanism_files_unchanged"]}
- Preparation labels: {verify["preparation_labels"]["n_distinct_preparation_labels"]}
"""
    PROVENANCE.write_text(text, encoding="utf-8")


def main() -> int:
    print("[merge] Phase 1: backup originals...")
    backup_info = backup_originals()
    print(f"  Backups OK -> {ARCHIVE}")

    pre_hash_paths = [
        MEASURED_CG,
        V1_ICC,
        INTEGRATED / "compound_gene_integrated_v1.parquet",
        INTEGRATED / "ingredient_gene_sets_v2.parquet",
        THREAD2_INF / "predicted_compound_gene_weighted_v2.parquet",
    ]
    pre_hashes = file_manifest([p for p in pre_hash_paths if p.exists()])

    print("[merge] Phase 2: load species remap final...")
    final_map = pd.read_parquet(FINAL_MAP)
    nodes = pd.read_parquet(FINAL_NODES)
    assert final_map["foodb_id"].nunique() == 463, "Expected 463 species"
    assert len(nodes) == 463

    print("[merge] Building FooDB food->compound index...")
    food_iks = build_food_compound_inchikeys(FOODB_CONTENT, COMPOUND_MASTER)

    print("[merge] Building ingredient_compound_canonical_v2...")
    v2_icc = build_v2_ingredient_compound(nodes, food_iks)
    v2_icc.to_parquet(V2_ICC_PQ, index=False)
    v2_icc[["ingredient_id", "compound_id"]].to_csv(V2_ICC_CSV, index=False)
    print(f"  v2 edges: {len(v2_icc):,} | species: {v2_icc['ingredient_id'].nunique()} | compounds: {v2_icc['compound_id'].nunique()}")

    nodes_v2 = build_species_nodes_v2(final_map, nodes)
    nodes_v2.to_parquet(V2_NODES, index=False)

    string_map = final_map[
        [
            "ingredient_string",
            "species_node",
            "latin_name",
            "foodb_id",
            "preparation_label",
            "match_method",
            "canonical_name",
            "recipe_occurrences",
            "merge_annotation",
            "mechanism_tier",
            "source_pass",
        ]
    ].copy()
    string_map.to_parquet(V2_STRING_MAP, index=False)

    print("[merge] Phase 3: verify...")
    verify = verify_merge(v2_icc, final_map, nodes_v2, backup_info, pre_hashes)
    VERIFY_OUT.write_text(json.dumps(verify, indent=2), encoding="utf-8")
    write_provenance(backup_info, verify)

    print("\n=== SPECIES MERGE COMPLETE ===")
    il = verify["ingredient_layer_v2"]
    print(f"  Species nodes: {il['n_species_nodes']} (match 463: {il['species_count_match']})")
    print(f"  ICC v2 edges: {il['n_ingredient_compound_edges']:,}")
    print(f"  Recipe coverage: {il['recipe_occurrence_coverage_pct']}%")
    print(f"  Staples pass: {verify['all_staples_pass']}")
    print(f"  Measured CG unchanged: {verify['all_mechanism_files_unchanged']}")
    print(f"  Preparation labels: {verify['preparation_labels']['n_distinct_preparation_labels']}")
    print(f"  No-overlap species present: {verify['no_corpus_overlap_species']['count_present']}/18")
    print(f"  Provenance: {PROVENANCE.relative_to(ROOT)}")
    print("  STOP — gene sets / enrichment NOT rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
