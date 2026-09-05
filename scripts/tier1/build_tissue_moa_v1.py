#!/usr/bin/env python3
"""
GTEx tissue-localization profiles + measured-only ChEMBL MoA annotation.

Part 0: verify GTEx parse + gene join (gate >= 90%).
Part 1: ingredient × tissue profiles from v3 gene sets + GTEx (new output only).
Part 2: ChEMBL action_type on measured compound→gene edges, same-gene only.

Usage (from repo root):
    python scripts/tier1/build_tissue_moa_v1.py

Outputs:
    data/processed/tier1/ingredient_tissue_profiles_v1.parquet
    data/processed/tier1/measured_moa_annotation_v1.parquet
    data/processed/tier1/tissue_moa_build_report_v1.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TIER1 = ROOT / "data/processed/tier1"

GTEX_CANDIDATES = [
    ROOT / "data/raw/gtex/gene_median_tpm.gct",
    ROOT / "data/raw/gtex/gene_median_tpm.gct.gz",
    ROOT / "data/raw/gtex.gct",
]
GENE_SETS_V3 = ROOT / "data/processed/integrated/ingredient_gene_sets_v3.parquet"
INTEGRATED_CG = ROOT / "data/processed/integrated/compound_gene_integrated_v1.parquet"
MEASURED_CG = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"
CHEMBL_DB = ROOT / "data/raw/chembl/chembl_36/chembl_36_sqlite/chembl_36.db"
HGNC_PATH = ROOT / "data/interim/mappings/hgnc_complete_set.txt"
STRING_MAP = ROOT / "data/processed/canonical/ingredient_string_species_v2.parquet"

TISSUE_OUT = TIER1 / "ingredient_tissue_profiles_v1.parquet"
MOA_OUT = TIER1 / "measured_moa_annotation_v1.parquet"
REPORT_OUT = TIER1 / "tissue_moa_build_report_v1.json"

JOIN_GATE_MIN = 0.90
INTERPRETATION_NOTE = (
    "tissue_score reflects where TARGET GENES are expressed in GTEx (human baseline); "
    "NOT proof the ingredient/compound reaches or acts in that tissue (no PK/absorption model)."
)

SAMPLE_INGREDIENTS = ["SP_000052", "SP_000005", "SP_000259", "SP_000235", "SP_000026"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_gtex_path() -> Path:
    for p in GTEX_CANDIDATES:
        if p.exists():
            return p
    gtex_dir = ROOT / "data/raw/gtex"
    if gtex_dir.is_dir():
        for p in sorted(gtex_dir.glob("*.gct*")):
            return p
    raise FileNotFoundError(f"No GTEx GCT found among {GTEX_CANDIDATES}")


def read_gct_header(path: Path) -> tuple[int, int, list[str]]:
    opener = gzip_open if str(path).endswith(".gz") else lambda p, m: p.open(m, encoding="utf-8")
    with opener(path, "r") as f:
        line1 = f.readline()
        if not line1.startswith("#1.2"):
            raise ValueError(f"Expected GCT #1.2 header, got: {line1[:40]!r}")
        dims = f.readline().strip().split("\t")
        n_genes, n_cols = int(dims[0]), int(dims[1])
        header = f.readline().strip().split("\t")
    return n_genes, n_cols, header


def gzip_open(path: Path, mode: str):
    import gzip

    return gzip.open(path, mode, encoding="utf-8")


def load_gtex(path: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    n_genes, n_cols, header = read_gct_header(path)
    gtex = pd.read_csv(path, sep="\t", skiprows=2)
    if gtex.shape[0] != n_genes:
        raise ValueError(f"GCT row count mismatch: header {n_genes}, parsed {gtex.shape[0]}")
    tissue_cols = [c for c in header if c not in ("Name", "Description")]
    meta = {
        "path": str(path.relative_to(ROOT)),
        "n_genes": n_genes,
        "n_cols_field": n_cols,
        "n_tissue_columns": len(tissue_cols),
        "tissue_names": tissue_cols,
        "sample_rows": gtex.iloc[:3, : min(6, gtex.shape[1])].to_dict(orient="records"),
    }
    return gtex, tissue_cols, meta


def load_our_genes() -> set[str]:
    df = pd.read_parquet(INTEGRATED_CG, columns=["gene_symbol"])
    return set(df["gene_symbol"].astype(str).dropna())


def load_hgnc_maps() -> tuple[dict[str, str], dict[str, str]]:
    """symbol_upper -> ensembl_gene_id; uniprot_acc -> hgnc symbol (1:1 approved only)."""
    sym_to_ens: dict[str, str] = {}
    acc_to_sym: dict[str, str] = {}
    acc_ambiguous: set[str] = set()

    with HGNC_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status", "").strip() != "Approved":
                continue
            sym = row.get("symbol", "").strip()
            ens = row.get("ensembl_gene_id", "").strip()
            if sym and ens:
                sym_to_ens[sym.upper()] = ens
            uniprots = row.get("uniprot_ids", "").strip()
            if not sym or not uniprots:
                continue
            for acc in uniprots.split("|"):
                acc = acc.strip().upper()
                if not acc:
                    continue
                if acc in acc_to_sym and acc_to_sym[acc] != sym:
                    acc_ambiguous.add(acc)
                else:
                    acc_to_sym[acc] = sym
    for acc in acc_ambiguous:
        acc_to_sym.pop(acc, None)
    return sym_to_ens, acc_to_sym


def gtex_join_report(gtex: pd.DataFrame, our_genes: set[str], sym_to_ens: dict[str, str]) -> dict[str, Any]:
    gtex = gtex.copy()
    gtex["Description_upper"] = gtex["Description"].astype(str).str.upper()
    gtex["ens_base"] = gtex["Name"].astype(str).str.split(".").str[0]

    symbol_index = set(gtex["Description_upper"])
    matched_symbol = sorted(g for g in our_genes if g.upper() in symbol_index)
    unmatched = sorted(g for g in our_genes if g.upper() not in symbol_index)

    matched_ensembl: list[str] = []
    still_unmatched: list[str] = []
    ens_index = set(gtex["ens_base"])
    for g in unmatched:
        ens = sym_to_ens.get(g.upper())
        if ens and ens in ens_index:
            matched_ensembl.append(g)
        else:
            still_unmatched.append(g)

    matched_all = matched_symbol + matched_ensembl
    rate = len(matched_all) / len(our_genes) if our_genes else 0.0

    # gene -> row index for expression lookup
    gene_to_idx: dict[str, int] = {}
    desc_to_idx = dict(zip(gtex["Description_upper"], gtex.index))
    ens_to_idx = dict(zip(gtex["ens_base"], gtex.index))
    for g in matched_symbol:
        gene_to_idx[g] = int(desc_to_idx[g.upper()])
    for g in matched_ensembl:
        ens = sym_to_ens[g.upper()]
        gene_to_idx[g] = int(ens_to_idx[ens])

    return {
        "our_gene_count": len(our_genes),
        "matched_on_symbol": len(matched_symbol),
        "matched_on_ensembl_fallback": len(matched_ensembl),
        "matched_total": len(matched_all),
        "unmatched_total": len(still_unmatched),
        "match_rate": round(rate, 4),
        "gate_passed": rate >= JOIN_GATE_MIN,
        "matched_examples": matched_all[:5],
        "unmatched_genes": still_unmatched,
        "gene_to_gtex_row": gene_to_idx,
    }


def build_expression_matrix(
    gtex: pd.DataFrame,
    tissue_cols: list[str],
    gene_to_idx: dict[str, int],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Per-gene tissue weights from log1p(TPM), z-score across tissues, then shift+normalize
    to a non-negative tissue distribution per gene (relative enrichment, not absolute TPM).
    """
    genes = sorted(gene_to_idx)
    mat = gtex.loc[[gene_to_idx[g] for g in genes], tissue_cols].astype(float).values
    log_mat = np.log1p(mat)
    mu = log_mat.mean(axis=1, keepdims=True)
    sd = log_mat.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z_mat = (log_mat - mu) / sd
    # Shift so each gene's minimum tissue weight is 0, then row-normalize to sum to 1.
    z_pos = z_mat - z_mat.min(axis=1, keepdims=True)
    row_sum = z_pos.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-8, 1.0, row_sum)
    prop = z_pos / row_sum
    expr = pd.DataFrame(prop, index=genes, columns=tissue_cols)
    return expr, genes


def build_tissue_profiles(
    gene_sets: pd.DataFrame,
    expr: pd.DataFrame,
) -> pd.DataFrame:
    tissues = expr.columns.tolist()
    rows: list[dict[str, Any]] = []

    for ing_id, grp in gene_sets.groupby("ingredient_id"):
        genes = grp["gene_symbol"].astype(str)
        conf = grp["confidence"].astype(float)
        evidence = grp["evidence"].astype(str)
        mask = genes.isin(expr.index)
        if not mask.any():
            continue
        g_list = genes[mask].tolist()
        c_list = conf[mask].values
        e_list = evidence[mask].tolist()
        sub = expr.loc[g_list].values
        weights = c_list.reshape(-1, 1)
        raw = (sub * weights).sum(axis=0)
        measured_mask = np.array([e == "measured" for e in e_list])
        measured_score = (sub[measured_mask] * weights[measured_mask]).sum(axis=0) if measured_mask.any() else np.zeros(len(tissues))
        predicted_score = raw - measured_score
        total = raw.sum()
        if total <= 1e-12:
            norm = np.ones(len(tissues)) / len(tissues)
        else:
            norm = raw / total
        ms_total = float(measured_score.sum())
        ps_total = float(predicted_score.sum())
        split_denom = ms_total + ps_total
        measured_frac = ms_total / split_denom if split_denom > 0 else np.nan
        if np.isnan(measured_frac) or measured_frac < 0.05:
            split_label = "predicted_dominant"
        elif measured_frac > 0.95:
            split_label = "measured_dominant"
        else:
            split_label = "mixed"

        for i, tissue in enumerate(tissues):
            rows.append(
                {
                    "ingredient_id": ing_id,
                    "tissue": tissue,
                    "tissue_score": float(raw[i]),
                    "normalized_score": float(norm[i]),
                    "n_genes_contributing": int(len(g_list)),
                    "measured_vs_predicted_split": split_label,
                    "measured_score_component": float(measured_score[i]),
                    "predicted_score_component": float(predicted_score[i]),
                    "measured_fraction_of_score": measured_frac,
                    "interpretation_note": INTERPRETATION_NOTE,
                }
            )
    return pd.DataFrame(rows)


def tissue_sanity(profiles: pd.DataFrame, string_map: pd.DataFrame | None) -> dict[str, Any]:
    name_lookup: dict[str, str] = {}
    if string_map is not None and not string_map.empty:
        id_col = "species_node" if "species_node" in string_map.columns else "ingredient_id"
        label_col = "canonical_name" if "canonical_name" in string_map.columns else None
        if label_col and id_col in string_map.columns:
            name_lookup = dict(zip(string_map[id_col].astype(str), string_map[label_col].astype(str)))

    samples: list[dict[str, Any]] = []
    ids = [i for i in SAMPLE_INGREDIENTS if i in set(profiles["ingredient_id"])]
    if len(ids) < 5:
        extra = profiles["ingredient_id"].drop_duplicates().head(5 - len(ids)).tolist()
        ids = ids + [i for i in extra if i not in ids]

    top_tissue_counts: dict[str, int] = defaultdict(int)
    for ing_id in ids[:5]:
        sub = profiles[profiles["ingredient_id"] == ing_id].nlargest(5, "normalized_score")
        top_tissue_counts[sub.iloc[0]["tissue"]] += 1
        samples.append(
            {
                "ingredient_id": ing_id,
                "label": name_lookup.get(ing_id, ing_id),
                "top_5_tissues": sub[["tissue", "normalized_score"]].to_dict(orient="records"),
            }
        )

    top5_sets = []
    for ing_id in ids[:5]:
        tset = set(profiles[profiles["ingredient_id"] == ing_id].nlargest(5, "normalized_score")["tissue"])
        top5_sets.append(tset)
    jaccards = []
    for i in range(len(top5_sets)):
        for j in range(i + 1, len(top5_sets)):
            a, b = top5_sets[i], top5_sets[j]
            jaccards.append(len(a & b) / len(a | b) if a | b else 0.0)
    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

    wide = profiles.pivot_table(
        index="ingredient_id", columns="tissue", values="normalized_score", fill_value=0.0
    )
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(wide.index.to_numpy(), size=min(30, len(wide)), replace=False)
    sub = wide.loc[sample_ids]
    corr = sub.T.corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    mean_pairwise_corr = float(corr.where(mask).stack().mean())

    liver_top = top_tissue_counts.get("Liver", 0) + top_tissue_counts.get("Liver_Portal_Tract", 0)
    testis_top = top_tissue_counts.get("Testis", 0)
    collapse_flag = liver_top >= 4 or testis_top >= 4 or mean_jaccard > 0.85 or mean_pairwise_corr > 0.9

    return {
        "sample_ingredient_top5": samples,
        "top1_tissue_counts_among_samples": dict(top_tissue_counts),
        "mean_top5_jaccard": round(mean_jaccard, 4),
        "mean_pairwise_profile_correlation_n30": round(mean_pairwise_corr, 4),
        "global_top_tissues_by_mean_score": (
            profiles.groupby("tissue")["normalized_score"].mean().nlargest(5).round(6).to_dict()
        ),
        "collapse_to_single_tissue_flag": collapse_flag,
        "normalization_ok": not collapse_flag,
        "convergence_note": (
            "High cross-ingredient profile correlation is expected when v3 gene sets share "
            "large predicted target universes (PRKCA/CNR/LPAR hubs); tissue layer reflects "
            "where those shared targets are expressed, not ingredient-specific absorption."
            if mean_pairwise_corr > 0.85
            else None
        ),
    }


def build_measured_moa(acc_to_sym: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    measured = pd.read_csv(MEASURED_CG)
    measured["compound_id"] = measured["compound_id"].astype(str).str.upper()
    measured["gene_symbol"] = measured["gene_symbol"].astype(str).str.upper()

    conn = sqlite3.connect(CHEMBL_DB)
    dm = pd.read_sql(
        """
        SELECT dm.molregno, dm.action_type, dm.tid, cs.standard_inchi_key, cs2.accession
        FROM drug_mechanism dm
        JOIN molecule_dictionary md ON dm.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN target_components tc ON dm.tid = tc.tid
        JOIN component_sequences cs2 ON tc.component_id = cs2.component_id
        WHERE cs.standard_inchi_key IS NOT NULL
          AND cs2.component_type = 'PROTEIN'
          AND cs2.accession IS NOT NULL
          AND dm.action_type IS NOT NULL
        """,
        conn,
    )
    conn.close()

    dm["standard_inchi_key"] = dm["standard_inchi_key"].str.upper()
    dm["accession"] = dm["accession"].str.upper()
    dm["gene_symbol"] = dm["accession"].map(acc_to_sym)
    dm = dm[dm["gene_symbol"].notna()]
    dm = dm.rename(columns={"standard_inchi_key": "compound_id"})
    dm_edges = dm[["compound_id", "gene_symbol", "action_type"]].drop_duplicates()

    merged = measured.merge(dm_edges, on=["compound_id", "gene_symbol"], how="inner")
    out = merged[["compound_id", "gene_symbol", "action_type"]].rename(columns={"compound_id": "compound"})
    out["source"] = "chembl_measured"
    out = out.drop_duplicates()

    predicted_n = 0
    if INTEGRATED_CG.exists():
        integ = pd.read_parquet(INTEGRATED_CG, columns=["compound_id", "gene_symbol", "source"])
        pred = integ[integ["source"] == "predicted"]
        pred_keys = set(zip(pred["compound_id"].str.upper(), pred["gene_symbol"].str.upper()))
        moa_keys = set(zip(out["compound"], out["gene_symbol"]))
        predicted_n = len(moa_keys & pred_keys)

    report = {
        "measured_edges_total": len(measured),
        "annotated_edges": len(out),
        "annotation_fraction": round(len(out) / len(measured), 4) if len(measured) else 0.0,
        "action_type_breakdown": out["action_type"].value_counts().to_dict(),
        "predicted_edges_annotated": predicted_n,
        "zero_predicted_confirmed": predicted_n == 0,
    }
    return out, report


def main() -> int:
    TIER1.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "normalization": {
            "expression_transform": "log1p(GTEx_median_TPM)",
            "per_gene": "z-score across tissues, shift to non-negative, row-normalize to tissue proportions",
            "rationale": (
                "log1p compresses TPM skew; per-gene z-scoring captures relative tissue enrichment; "
                "shifting and row-normalizing yields non-negative tissue weights per gene so "
                "housekeeping absolute TPM does not dominate and scores do not cancel across tissues."
            ),
            "ingredient_profile": "normalized_score = tissue_score / sum(tissue_score) per ingredient",
        },
        "interpretation_note": INTERPRETATION_NOTE,
    }

    gtex_path = resolve_gtex_path()
    gtex, tissue_cols, gtex_meta = load_gtex(gtex_path)
    report["part0_gtex"] = gtex_meta

    our_genes = load_our_genes()
    sym_to_ens, acc_to_sym = load_hgnc_maps()
    join = gtex_join_report(gtex, our_genes, sym_to_ens)
    report["part0_join"] = {k: v for k, v in join.items() if k != "gene_to_gtex_row"}

    print("=== PART 0: GTEx join gate ===")
    print(f"GTEx: {gtex_meta['n_genes']} genes, {gtex_meta['n_tissue_columns']} tissues")
    print(f"Match rate: {join['matched_total']}/{join['our_gene_count']} = {join['match_rate']:.2%}")
    print(f"Gate (>={JOIN_GATE_MIN:.0%}): {'PASS' if join['gate_passed'] else 'FAIL'}")

    if not join["gate_passed"]:
        report["stopped_at"] = "part0_join_gate"
        REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("STOP: join rate below gate.")
        return 1

    # Part 1
    expr, _ = build_expression_matrix(gtex, tissue_cols, join["gene_to_gtex_row"])
    gene_sets = pd.read_parquet(GENE_SETS_V3)
    profiles = build_tissue_profiles(gene_sets, expr)
    profiles.to_parquet(TISSUE_OUT, index=False)
    report["part1_tissue"] = {
        "output": str(TISSUE_OUT.relative_to(ROOT)),
        "rows": len(profiles),
        "ingredients": int(profiles["ingredient_id"].nunique()),
        "tissues": int(profiles["tissue"].nunique()),
        "sha256": sha256_file(TISSUE_OUT),
    }

    string_map = pd.read_parquet(STRING_MAP) if STRING_MAP.exists() else None
    sanity = tissue_sanity(profiles, string_map)
    report["part1_sanity"] = sanity

    print("\n=== PART 1: tissue profiles ===")
    print(f"Wrote {TISSUE_OUT.name}: {len(profiles):,} rows")
    for s in sanity["sample_ingredient_top5"]:
        print(f"  {s['label']} ({s['ingredient_id']}):")
        for t in s["top_5_tissues"][:3]:
            print(f"    {t['tissue']}: {t['normalized_score']:.4f}")

    # Part 2
    moa, moa_report = build_measured_moa(acc_to_sym)
    moa.to_parquet(MOA_OUT, index=False)
    report["part2_moa"] = {
        "output": str(MOA_OUT.relative_to(ROOT)),
        "sha256": sha256_file(MOA_OUT),
        **moa_report,
    }

    print("\n=== PART 2: measured MoA ===")
    print(f"Annotated {moa_report['annotated_edges']}/{moa_report['measured_edges_total']} "
          f"({moa_report['annotation_fraction']:.2%})")
    print(f"Predicted edges annotated: {moa_report['predicted_edges_annotated']}")

    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {REPORT_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
