#!/usr/bin/env python3
"""
Build structure-bearing compound->gene training corpus from ChEMBL + BindingDB.

Strict policy (approved). Uses data/interim/mappings/uniprot_to_hgnc.parquet.
Output: data/processed/corpus/compound_target_corpus_v1.parquet (+ build report JSON).

Does NOT modify any canonical files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHEMBL_DB = ROOT / "data/raw/chembl/chembl_36/chembl_36_sqlite/chembl_36.db"
BDB_TSV = ROOT / "data/raw/BindingDB_All.tsv"
UNIPROT_MAP = ROOT / "data/interim/mappings/uniprot_to_hgnc.parquet"
INGREDIENT_COMPOUNDS = ROOT / "data/processed/canonical/ingredient_compound_canonical.csv"
OUT_DIR = ROOT / "data/processed/corpus"
OUT_PARQUET = OUT_DIR / "compound_target_corpus_v1.parquet"
OUT_REPORT = OUT_DIR / "compound_target_corpus_v1_build_report.json"
CHEMBL_RAW_CACHE = OUT_DIR / "_chembl_raw_activity.parquet"
BDB_RAW_CACHE = OUT_DIR / "_bindingdb_raw_activity.parquet"

POTENCY_NM = 10_000.0
STD_TYPES = ("Ki", "Kd", "IC50", "EC50")
BDB_MEAS_COLS = {
    "Ki": "Ki (nM)",
    "Kd": "Kd (nM)",
    "IC50": "IC50 (nM)",
    "EC50": "EC50 (nM)",
}

POTENCY_RE = re.compile(r"^[<>=]?\s*([0-9]*\.?[0-9]+)")


def parse_affinity_nM(raw: Any) -> float | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "n/a"}:
        return None
    m = POTENCY_RE.match(s.replace(",", ""))
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if v <= 0 or v > POTENCY_NM:
        return None
    return v


def to_nM(value: float, units: str) -> float | None:
    u = (units or "").strip()
    if u == "nM":
        return value
    if u == "uM":
        return value * 1000.0
    return None


def load_uniprot_map(path: Path) -> tuple[dict[str, str], set[str]]:
    df = pd.read_parquet(path)
    ok = df[df["mapping_type"].isin(["primary_canonical", "secondary_accession"])].copy()
    amb = set(df[df["mapping_type"] == "ambiguous_multi_hgnc"]["uniprot_accession"].astype(str).str.upper())
    mp: dict[str, str] = {}
    for _, r in ok.iterrows():
        acc = str(r["uniprot_accession"]).strip().upper()
        sym = str(r["hgnc_symbol"]).strip()
        if acc and sym and acc not in mp:
            mp[acc] = sym
    return mp, amb


def chembl_scale_report() -> dict[str, int]:
    import sqlite3

    con = sqlite3.connect(str(CHEMBL_DB))
    cur = con.cursor()
    potency_sql = (
        "((a.standard_units='nM' AND a.standard_value<=?) "
        "OR (a.standard_units='uM' AND a.standard_value<=?))"
    )
    base_join = (
        "FROM activities a "
        "JOIN assays s ON a.assay_id=s.assay_id "
        "JOIN target_dictionary td ON s.tid=td.tid"
    )
    stages: list[tuple[str, str, tuple]] = [
        ("all_with_tid", f"SELECT COUNT(*) {base_join} WHERE s.tid IS NOT NULL", ()),
        (
            "single_protein_human",
            f"SELECT COUNT(*) {base_join} "
            "WHERE td.target_type=? AND td.organism=?",
            ("SINGLE PROTEIN", "Homo sapiens"),
        ),
        (
            "plus_measured_std_type",
            f"SELECT COUNT(*) {base_join} "
            "WHERE td.target_type=? AND td.organism=? "
            "AND a.standard_value IS NOT NULL "
            "AND a.standard_type IN (?,?,?,?)",
            ("SINGLE PROTEIN", "Homo sapiens", *STD_TYPES),
        ),
        (
            "plus_potency_10uM",
            f"SELECT COUNT(*) {base_join} "
            "WHERE td.target_type=? AND td.organism=? "
            "AND a.standard_value IS NOT NULL "
            "AND a.standard_type IN (?,?,?,?) "
            f"AND {potency_sql}",
            ("SINGLE PROTEIN", "Homo sapiens", *STD_TYPES, POTENCY_NM, 10.0),
        ),
        (
            "plus_inchikey_and_smiles",
            f"SELECT COUNT(*) {base_join} "
            "JOIN compound_structures cs ON a.molregno=cs.molregno "
            "WHERE td.target_type=? AND td.organism=? "
            "AND a.standard_value IS NOT NULL "
            "AND a.standard_type IN (?,?,?,?) "
            f"AND {potency_sql} "
            "AND cs.standard_inchi_key IS NOT NULL AND cs.standard_inchi_key!='' "
            "AND cs.canonical_smiles IS NOT NULL AND cs.canonical_smiles!=''",
            ("SINGLE PROTEIN", "Homo sapiens", *STD_TYPES, POTENCY_NM, 10.0),
        ),
    ]
    out: dict[str, int] = {}
    for name, sql, params in stages:
        cur.execute(sql, params)
        out[name] = int(cur.fetchone()[0])
    cur.execute(
        f"""
        SELECT COUNT(DISTINCT cs.standard_inchi_key || '|' || csq.accession)
        {base_join}
        JOIN compound_structures cs ON a.molregno=cs.molregno
        JOIN target_components tc ON td.tid=tc.tid
        JOIN component_sequences csq ON tc.component_id=csq.component_id
        WHERE td.target_type=? AND td.organism=?
        AND a.standard_value IS NOT NULL
        AND a.standard_type IN (?,?,?,?)
        AND {potency_sql}
        AND cs.standard_inchi_key IS NOT NULL AND cs.standard_inchi_key!=''
        AND csq.accession IS NOT NULL AND csq.accession!=''
        """,
        ("SINGLE PROTEIN", "Homo sapiens", *STD_TYPES, POTENCY_NM, 10.0),
    )
    out["distinct_inchikey_uniprot_pairs"] = int(cur.fetchone()[0])
    con.close()
    return out


def extract_chembl(
    uniprot_map: dict[str, str],
    ambiguous: set[str],
    batch_size: int = 250_000,
) -> pd.DataFrame:
    import sqlite3

    con = sqlite3.connect(str(CHEMBL_DB))
    cur = con.cursor()
    cur.execute("SELECT MIN(activity_id), MAX(activity_id) FROM activities")
    min_id, max_id = cur.fetchone()
    potency_sql = (
        "((a.standard_units='nM' AND a.standard_value<=?) "
        "OR (a.standard_units='uM' AND a.standard_value<=?))"
    )
    query = f"""
        SELECT
            a.activity_id,
            cs.standard_inchi_key,
            cs.canonical_smiles,
            cs.standard_inchi,
            csq.accession,
            td.pref_name,
            a.standard_type,
            a.standard_value,
            a.standard_units
        FROM activities a
        JOIN assays s ON a.assay_id=s.assay_id
        JOIN target_dictionary td ON s.tid=td.tid
        JOIN compound_structures cs ON a.molregno=cs.molregno
        JOIN target_components tc ON td.tid=tc.tid
        JOIN component_sequences csq ON tc.component_id=csq.component_id
        WHERE a.activity_id >= ? AND a.activity_id < ?
          AND td.target_type=?
          AND td.organism=?
          AND a.standard_value IS NOT NULL
          AND a.standard_type IN (?,?,?,?)
          AND {potency_sql}
          AND cs.standard_inchi_key IS NOT NULL AND cs.standard_inchi_key!=''
          AND cs.canonical_smiles IS NOT NULL AND cs.canonical_smiles!=''
          AND csq.accession IS NOT NULL AND csq.accession!=''
    """
    rows: list[dict[str, Any]] = []
    dropped_no_map = dropped_amb = dropped_no_structure = 0
    start = int(min_id)
    end_max = int(max_id) + 1
    print(f"ChEMBL batch extraction activity_id {start}..{end_max}", flush=True)
    while start < end_max:
        stop = min(start + batch_size, end_max)
        cur.execute(
            query,
            (
                start,
                stop,
                "SINGLE PROTEIN",
                "Homo sapiens",
                *STD_TYPES,
                POTENCY_NM,
                10.0,
            ),
        )
        batch = cur.fetchall()
        if batch:
            print(f"  batch {start}-{stop}: raw rows {len(batch):,}", flush=True)
        for (
            _aid,
            inchikey,
            smiles,
            inchi,
            accession,
            pref_name,
            std_type,
            std_val,
            std_units,
        ) in batch:
            if not smiles or not str(smiles).strip():
                dropped_no_structure += 1
                continue
            acc = str(accession).strip().upper()
            if acc in ambiguous:
                dropped_amb += 1
                continue
            gene = uniprot_map.get(acc)
            if not gene:
                dropped_no_map += 1
                continue
            nm = to_nM(float(std_val), str(std_units))
            if nm is None or nm > POTENCY_NM:
                continue
            rows.append(
                {
                    "compound_inchikey": str(inchikey).strip(),
                    "compound_smiles": str(smiles).strip(),
                    "compound_inchi": str(inchi).strip() if inchi else "",
                    "gene_symbol": gene,
                    "uniprot_accession": acc,
                    "source": "chembl",
                    "measurement_type": str(std_type),
                    "standard_value_nM": nm,
                    "target_pref_name": str(pref_name).strip() if pref_name else "",
                }
            )
        start = stop
    con.close()
    print(
        f"ChEMBL kept activity rows {len(rows):,}; "
        f"dropped unmapped {dropped_no_map:,}, ambiguous {dropped_amb:,}, no_structure {dropped_no_structure:,}",
        flush=True,
    )
    return pd.DataFrame(rows)


def extract_bindingdb(uniprot_map: dict[str, str], ambiguous: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dropped = defaultdict(int)
    with BDB_TSV.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ik = (row.get("Ligand InChI Key") or "").strip()
            if not ik:
                dropped["no_inchikey"] += 1
                continue
            org = (row.get("Target Source Organism According to Curator or DataSource") or "").strip().lower()
            if org and ("homo sapiens" not in org and "human" not in org):
                dropped["non_human"] += 1
                continue
            acc = (
                (row.get("UniProt (SwissProt) Primary ID of Target Chain 1") or "").strip()
                or (row.get("UniProt (TrEMBL) Primary ID of Target Chain 1") or "").strip()
            ).upper()
            if not acc or " " in acc:
                dropped["bad_uniprot"] += 1
                continue
            if acc in ambiguous:
                dropped["ambiguous"] += 1
                continue
            gene = uniprot_map.get(acc)
            if not gene:
                dropped["unmapped"] += 1
                continue
            smiles = (row.get("Ligand SMILES") or "").strip()
            inchi = (row.get("Ligand InChI") or "").strip()
            if not smiles:
                dropped["no_smiles"] += 1
                continue
            tname = (row.get("Target Name") or "").strip()
            picked = False
            for mtype, col in BDB_MEAS_COLS.items():
                val = parse_affinity_nM(row.get(col))
                if val is None:
                    continue
                rows.append(
                    {
                        "compound_inchikey": ik,
                        "compound_smiles": smiles,
                        "compound_inchi": inchi,
                        "gene_symbol": gene,
                        "uniprot_accession": acc,
                        "source": "bindingdb",
                        "measurement_type": mtype,
                        "standard_value_nM": val,
                        "target_pref_name": tname,
                    }
                )
                picked = True
            if not picked:
                dropped["no_potent_measurement"] += 1
    print(f"BindingDB activity rows kept {len(rows):,}; drops {dict(dropped)}", flush=True)
    return pd.DataFrame(rows)


def aggregate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (compound_inchikey, gene_symbol, source): median potency."""
    if df.empty:
        return df
    group_cols = ["compound_inchikey", "gene_symbol", "source"]
    med = (
        df.groupby(group_cols, as_index=False)["standard_value_nM"]
        .median()
        .rename(columns={"standard_value_nM": "_median_nM"})
    )
    df = df.merge(med, on=group_cols, how="inner")
    df["_diff"] = (df["standard_value_nM"] - df["_median_nM"]).abs()
    idx = df.groupby(group_cols)["_diff"].idxmin()
    reps = df.loc[idx].copy()
    reps["standard_value_nM"] = reps["_median_nM"]
    reps = reps.drop(columns=["_median_nM", "_diff"])
    reps = reps[reps["compound_smiles"].astype(str).str.strip() != ""]
    cols = [
        "compound_inchikey",
        "compound_smiles",
        "compound_inchi",
        "gene_symbol",
        "uniprot_accession",
        "source",
        "measurement_type",
        "standard_value_nM",
        "target_pref_name",
    ]
    return reps[cols].reset_index(drop=True)


def ingredient_coverage(corpus: pd.DataFrame) -> dict[str, Any]:
    ic = pd.read_csv(INGREDIENT_COMPOUNDS, usecols=["ingredient_id", "compound_id"])
    comp_counts = ic.groupby("compound_id")["ingredient_id"].nunique()
    distinctive = set(comp_counts[comp_counts == 1].index.astype(str))
    corpus_comp = set(corpus["compound_inchikey"].astype(str))
    ing_comp = set(ic["compound_id"].astype(str))
    ing_with_target = ing_comp & corpus_comp
    dist_with_target = distinctive & corpus_comp
    ing_ids_with_any = set(
        ic[ic["compound_id"].astype(str).isin(corpus_comp)]["ingredient_id"].astype(str)
    )
    return {
        "ingredient_compounds_total": len(ing_comp),
        "ingredient_compounds_with_corpus_target": len(ing_with_target),
        "ingredients_with_any_corpus_compound": len(ing_ids_with_any),
        "ingredients_total": int(ic["ingredient_id"].nunique()),
        "distinctive_compounds_total": len(distinctive),
        "distinctive_compounds_with_measured_target": len(dist_with_target),
        "distinctive_compounds_without_measured_target": len(distinctive - corpus_comp),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-only", action="store_true", help="Report projected scale only")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate from cached raw activity parquets (skip extraction)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    uniprot_map, ambiguous = load_uniprot_map(UNIPROT_MAP)
    print(f"UniProt map: {len(uniprot_map):,} resolvable; {len(ambiguous):,} ambiguous excluded", flush=True)

    chembl_scale = chembl_scale_report()
    print("ChEMBL scale stages:", json.dumps(chembl_scale, indent=2), flush=True)

    if args.scale_only:
        print("Scale-only mode; stopping before extraction.", flush=True)
        return 0

    if args.aggregate_only and CHEMBL_RAW_CACHE.exists() and BDB_RAW_CACHE.exists():
        print("Loading cached raw activity parquets...", flush=True)
        chembl_raw = pd.read_parquet(CHEMBL_RAW_CACHE)
        bdb_raw = pd.read_parquet(BDB_RAW_CACHE)
    else:
        chembl_raw = extract_chembl(uniprot_map, ambiguous)
        chembl_raw.to_parquet(CHEMBL_RAW_CACHE, index=False)
        print(f"Cached {CHEMBL_RAW_CACHE}", flush=True)
        bdb_raw = extract_bindingdb(uniprot_map, ambiguous)
        bdb_raw.to_parquet(BDB_RAW_CACHE, index=False)
        print(f"Cached {BDB_RAW_CACHE}", flush=True)

    combined_raw = pd.concat([chembl_raw, bdb_raw], ignore_index=True)
    print(f"Combined raw activity rows: {len(combined_raw):,}", flush=True)

    print("Aggregating to compound-gene pairs (ChEMBL)...", flush=True)
    corpus_chembl = aggregate_pairs(chembl_raw)
    print(f"  ChEMBL pairs: {len(corpus_chembl):,}", flush=True)
    print("Aggregating to compound-gene pairs (BindingDB)...", flush=True)
    corpus_bdb = aggregate_pairs(bdb_raw)
    print(f"  BindingDB pairs: {len(corpus_bdb):,}", flush=True)
    corpus = (
        pd.concat([corpus_chembl, corpus_bdb], ignore_index=True)
        .sort_values(["source", "compound_inchikey", "gene_symbol"])
        .reset_index(drop=True)
    )
    corpus.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET} ({len(corpus):,} pairs)", flush=True)

    has_structure = corpus["compound_smiles"].astype(str).str.len() > 0
    ing_cov = ingredient_coverage(corpus)

    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "chembl": {
                "target_type": "SINGLE PROTEIN",
                "organism": "Homo sapiens",
                "standard_types": list(STD_TYPES),
                "potency_max_nM": POTENCY_NM,
            },
            "bindingdb": {
                "require_inchikey": True,
                "human_only": True,
                "require_uniprot": True,
                "potency_max_nM": POTENCY_NM,
                "measurement_columns": BDB_MEAS_COLS,
            },
            "aggregation": "median standard_value_nM per (compound_inchikey, gene_symbol, source); "
            "measurement_type from row closest to median; structure from first non-null per compound",
        },
        "scale_stages": {"chembl": chembl_scale},
        "corpus": {
            "path": str(OUT_PARQUET.relative_to(ROOT)),
            "total_pairs": int(len(corpus)),
            "unique_compounds": int(corpus["compound_inchikey"].nunique()),
            "unique_genes": int(corpus["gene_symbol"].nunique()),
            "by_source": corpus.groupby("source")
            .agg(pairs=("gene_symbol", "size"), compounds=("compound_inchikey", "nunique"), genes=("gene_symbol", "nunique"))
            .reset_index()
            .to_dict(orient="records"),
            "compounds_with_valid_smiles": int(has_structure.sum()),
            "trainable_fraction": round(float(has_structure.mean()), 4),
        },
        "ingredient_coverage": ing_cov,
        "held_out_readiness": {
            "compound_key": "compound_inchikey",
            "structure_field": "compound_smiles",
            "all_rows_have_inchikey": bool((corpus["compound_inchikey"].astype(str).str.len() > 0).all()),
            "all_rows_have_smiles": bool(has_structure.all()),
            "split_by_compound_feasible": True,
            "note": "GroupKFold or compound-level holdout by compound_inchikey ensures test compounds never appear in train.",
        },
        "raw_activity_rows": {
            "chembl": int(len(chembl_raw)),
            "bindingdb": int(len(bdb_raw)),
            "combined": int(len(combined_raw)),
        },
    }

    with OUT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["corpus"], indent=2), flush=True)
    print(json.dumps(report["ingredient_coverage"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
