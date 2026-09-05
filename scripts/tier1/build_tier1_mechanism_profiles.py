"""
Build Tier 1 single-ingredient mechanism profiles to pathway level.

Pinned inputs only:
- data/processed/canonical/ingredient_compound_canonical.csv
- data/processed/canonical/compound_gene_expanded_canonical_normalized.csv
- data/interim/pathways/gene_pathway_mappings.parquet

Outputs (new location only):
- data/processed/tier1/ingredient_mechanism_profiles.parquet
- data/processed/tier1/ingredient_mechanism_profiles.jsonl
- data/processed/tier1/tier1_profile_build_report.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IC_PATH = ROOT / "data/processed/canonical/ingredient_compound_canonical.csv"
CG_PATH = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"
GPM_PATH = ROOT / "data/interim/pathways/gene_pathway_mappings.parquet"

OUT_DIR = ROOT / "data/processed/tier1"
OUT_PARQUET = OUT_DIR / "ingredient_mechanism_profiles.parquet"
OUT_JSONL = OUT_DIR / "ingredient_mechanism_profiles.jsonl"
OUT_REPORT = OUT_DIR / "tier1_profile_build_report.json"

EXPECTED_FUNNEL = {
    "ingredients_with_compound": 223,
    "ingredients_reach_gene": 222,
    "ingredients_reach_pathway": 222,
}


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ic = pd.read_csv(IC_PATH)
    cg = pd.read_csv(CG_PATH)
    gpm = pd.read_parquet(GPM_PATH)
    return ic, cg, gpm


def _normalize(ic: pd.DataFrame, cg: pd.DataFrame, gpm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ic2 = ic.copy()
    cg2 = cg.copy()
    gpm2 = gpm.copy()

    ic2["ingredient_id"] = ic2["ingredient_id"].astype(str).str.strip()
    ic2["compound_id"] = ic2["compound_id"].astype(str).str.strip().str.upper()

    cg2["compound_id"] = cg2["compound_id"].astype(str).str.strip().str.upper()
    cg2["gene_symbol"] = cg2["gene_symbol"].astype(str).str.strip()
    cg2["raw_gene_name"] = cg2["raw_gene_name"].astype(str).str.strip()
    cg2["source"] = cg2["source"].astype(str).str.strip()

    gpm2["gene_symbol"] = gpm2["gene_symbol"].astype(str).str.strip()
    gpm2["pathway_id"] = gpm2["pathway_id"].astype(str).str.strip()
    gpm2["database"] = gpm2["database"].astype(str).str.strip()
    gpm2["uniprot_accession"] = gpm2["uniprot_accession"].astype(str).str.strip()

    return ic2, cg2, gpm2


def _build_maps(
    ic: pd.DataFrame, cg: pd.DataFrame, gpm: pd.DataFrame
) -> tuple[dict[str, dict[str, int]], dict[str, list[dict]], dict[str, list[dict]]]:
    ing_to_compound_counts = (
        ic.groupby(["ingredient_id", "compound_id"]).size().rename("edge_count").reset_index()
    )
    ing_compound_map: dict[str, dict[str, int]] = {}
    for _, row in ing_to_compound_counts.iterrows():
        ing = row["ingredient_id"]
        cmp_id = row["compound_id"]
        cnt = int(row["edge_count"])
        ing_compound_map.setdefault(ing, {})[cmp_id] = cnt

    cg2 = cg[
        [
            "compound_id",
            "gene_symbol",
            "raw_gene_name",
            "source",
            "evidence_fields",
            "resolver_used_compound",
            "resolver_used_target",
        ]
    ].drop_duplicates()
    cmp_gene_map: dict[str, list[dict]] = {}
    for _, row in cg2.iterrows():
        cmp_gene_map.setdefault(row["compound_id"], []).append(
            {
                "gene_symbol": row["gene_symbol"],
                "raw_gene_name": row["raw_gene_name"],
                "source": row["source"],
                "evidence_fields": None if pd.isna(row["evidence_fields"]) else str(row["evidence_fields"]),
                "resolver_used_compound": None
                if pd.isna(row["resolver_used_compound"])
                else str(row["resolver_used_compound"]),
                "resolver_used_target": None
                if pd.isna(row["resolver_used_target"])
                else str(row["resolver_used_target"]),
            }
        )

    gpm2 = gpm[["gene_symbol", "pathway_id", "database", "uniprot_accession"]].drop_duplicates()
    gene_pathway_map: dict[str, list[dict]] = {}
    for _, row in gpm2.iterrows():
        gene_pathway_map.setdefault(row["gene_symbol"], []).append(
            {
                "pathway_id": row["pathway_id"],
                "database": row["database"],
                "uniprot_accession": row["uniprot_accession"],
            }
        )

    return ing_compound_map, cmp_gene_map, gene_pathway_map


def main() -> int:
    ic_raw, cg_raw, gpm_raw = _load_inputs()
    ic, cg, gpm = _normalize(ic_raw, cg_raw, gpm_raw)

    ing_compound_map, cmp_gene_map, gene_pathway_map = _build_maps(ic, cg, gpm)
    genes_with_pathway = set(gene_pathway_map.keys())

    ingredients_with_compound = sorted(ing_compound_map.keys())
    ingredients_with_gene: list[str] = []
    ingredients_with_pathway: list[str] = []
    compounds_but_no_gene: list[str] = []

    profiles: list[dict] = []
    profile_rows: list[dict] = []

    for ing in ingredients_with_compound:
        compounds = ing_compound_map[ing]
        compounds_out = [
            {"compound_id": cmp_id, "edge_count": int(cnt)}
            for cmp_id, cnt in sorted(compounds.items(), key=lambda kv: kv[0])
        ]

        # Aggregate genes for this ingredient via compounds.
        gene_agg: dict[str, dict] = {}
        for cmp_id in compounds:
            for g in cmp_gene_map.get(cmp_id, []):
                gene = g["gene_symbol"]
                entry = gene_agg.setdefault(
                    gene,
                    {
                        "gene_symbol": gene,
                        "raw_gene_names": set(),
                        "sources": set(),
                        "compound_ids": set(),
                        "pathways": [],
                        "no_annotated_pathway": False,
                    },
                )
                entry["raw_gene_names"].add(g["raw_gene_name"])
                entry["sources"].add(g["source"])
                entry["compound_ids"].add(cmp_id)

        if not gene_agg:
            compounds_but_no_gene.append(ing)
            continue

        ingredients_with_gene.append(ing)

        pathways_agg: dict[tuple[str, str], dict] = {}
        n_genes_with_pathway = 0
        n_genes_no_pathway = 0
        for gene, g_entry in gene_agg.items():
            pws = gene_pathway_map.get(gene, [])
            if pws:
                n_genes_with_pathway += 1
                gene_pw = []
                for p in pws:
                    k = (p["pathway_id"], p["database"])
                    pe = pathways_agg.setdefault(
                        k,
                        {
                            "pathway_id": p["pathway_id"],
                            "database": p["database"],
                            "genes": set(),
                            "uniprot_accessions": set(),
                        },
                    )
                    pe["genes"].add(gene)
                    if p["uniprot_accession"]:
                        pe["uniprot_accessions"].add(p["uniprot_accession"])
                    gene_pw.append(
                        {
                            "pathway_id": p["pathway_id"],
                            "database": p["database"],
                            "uniprot_accession": p["uniprot_accession"],
                        }
                    )
                g_entry["pathways"] = gene_pw
                g_entry["no_annotated_pathway"] = False
            else:
                n_genes_no_pathway += 1
                g_entry["pathways"] = []
                g_entry["no_annotated_pathway"] = True

        if pathways_agg:
            ingredients_with_pathway.append(ing)

        genes_out = []
        for gene in sorted(gene_agg):
            g_entry = gene_agg[gene]
            genes_out.append(
                {
                    "gene_symbol": gene,
                    "raw_gene_name": sorted(g_entry["raw_gene_names"]),
                    "source": sorted(g_entry["sources"]),
                    "compound_ids": sorted(g_entry["compound_ids"]),
                    "pathways": g_entry["pathways"],
                    "no_annotated_pathway": g_entry["no_annotated_pathway"],
                }
            )

        pathways_out = []
        for k in sorted(pathways_agg):
            p_entry = pathways_agg[k]
            pathways_out.append(
                {
                    "pathway_id": p_entry["pathway_id"],
                    "database": p_entry["database"],
                    "genes": sorted(p_entry["genes"]),
                    "uniprot_accessions": sorted(p_entry["uniprot_accessions"]),
                }
            )

        profile = {
            "ingredient_id": ing,
            "ingredient_name": None,
            "compounds": compounds_out,
            "genes": genes_out,
            "pathways": pathways_out,
            "summary": {
                "n_compounds": len(compounds_out),
                "n_genes": len(genes_out),
                "n_genes_with_pathway": n_genes_with_pathway,
                "n_genes_no_pathway": n_genes_no_pathway,
                "n_pathways": len(pathways_out),
            },
        }
        profiles.append(profile)
        profile_rows.append(
            {
                "ingredient_id": ing,
                "ingredient_name": None,
                "n_compounds": profile["summary"]["n_compounds"],
                "n_genes": profile["summary"]["n_genes"],
                "n_genes_with_pathway": profile["summary"]["n_genes_with_pathway"],
                "n_genes_no_pathway": profile["summary"]["n_genes_no_pathway"],
                "n_pathways": profile["summary"]["n_pathways"],
                "compounds_json": json.dumps(compounds_out, ensure_ascii=True),
                "genes_json": json.dumps(genes_out, ensure_ascii=True),
                "pathways_json": json.dumps(pathways_out, ensure_ascii=True),
            }
        )

    # Funnel checks (must match verified values).
    funnel = {
        "ingredients_with_compound": len(ingredients_with_compound),
        "ingredients_reach_gene": len(ingredients_with_gene),
        "ingredients_reach_pathway": len(ingredients_with_pathway),
    }
    if funnel != EXPECTED_FUNNEL:
        raise RuntimeError(
            f"Funnel mismatch. Expected {EXPECTED_FUNNEL}, got {funnel}. "
            "Stopping per requirement."
        )

    no_pathway_only_profiles = [
        p["ingredient_id"]
        for p in profiles
        if p["summary"]["n_genes"] > 0 and p["summary"]["n_genes_with_pathway"] == 0
    ]
    if no_pathway_only_profiles:
        raise RuntimeError(
            "Found profiles with only no-pathway genes; expected zero. "
            f"Examples: {no_pathway_only_profiles[:10]}"
        )

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(profile_rows).sort_values("ingredient_id").to_parquet(OUT_PARQUET, index=False)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for p in sorted(profiles, key=lambda x: x["ingredient_id"]):
            fh.write(json.dumps(p, ensure_ascii=True) + "\n")

    # Build report
    s = pd.DataFrame(profile_rows)
    dist = {
        "n_compounds": {
            "min": int(s["n_compounds"].min()),
            "median": float(s["n_compounds"].median()),
            "max": int(s["n_compounds"].max()),
        },
        "n_genes": {
            "min": int(s["n_genes"].min()),
            "median": float(s["n_genes"].median()),
            "max": int(s["n_genes"].max()),
        },
        "n_pathways": {
            "min": int(s["n_pathways"].min()),
            "median": float(s["n_pathways"].median()),
            "max": int(s["n_pathways"].max()),
        },
    }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "opened_input_files": [str(IC_PATH), str(CG_PATH), str(GPM_PATH)],
        "forbidden_inputs_opened": [],
        "funnel": funnel,
        "profiles_built": len(profiles),
        "compounds_but_no_gene_ingredients": sorted(compounds_but_no_gene),
        "no_pathway_only_profiles_count": len(no_pathway_only_profiles),
        "summary_distribution": dist,
        "gene_join": {
            "cg_unique_genes": int(cg["gene_symbol"].nunique()),
            "join_to_gpm": int(len(set(cg["gene_symbol"].unique()) & genes_with_pathway)),
        },
        "outputs": {
            "parquet": str(OUT_PARQUET),
            "jsonl": str(OUT_JSONL),
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
