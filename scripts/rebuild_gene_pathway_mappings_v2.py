"""
Rebuild gene_pathway_mappings for the normalized compound→gene target set.

Uses the exact Phase 5 notebook method (UniProt REST, organism_id:9606).
Writes gene_pathway_mappings_v2.parquet — does NOT overwrite the original.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CG_PATH = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"
OLD_GPM = ROOT / "data/interim/pathways/gene_pathway_mappings.parquet"
OUT_GPM = ROOT / "data/interim/pathways/gene_pathway_mappings_v2.parquet"
REPORT_PATH = ROOT / "reports/gene_pathway_mappings_v2_proposal.json"

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
BATCH_SIZE = 100
RATE_LIMIT_SEC = 1.0
REQUEST_TIMEOUT = 30

# Names explicitly left unmapped in gene normalization (expected no-pathway).
EXCLUDED_UNMAPPED = frozenset(
    {
        "E-SELECTIN",
        "INTEGRASE",
        "TRYPSIN",
        "LANA1",
        "D3-HUMAN",
        "MDRC4",
        "CYP1A",
        "CYCLIN-Y",
        "E6",
        "HEPARANASE",
        "NINEIN",
        "PLECTIN",
    }
)


def get_uniprot_pathways_phase5(gene_symbols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Phase 5 notebook logic (notebooks/phase5_pathway_mapping.ipynb Section 4).
  """
    gene_pathway_mappings: list[dict] = []
    batch_log: list[dict] = []
    n_batches = (len(gene_symbols) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(gene_symbols), BATCH_SIZE):
        batch = gene_symbols[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        query = " OR ".join([f"(gene:{gene})" for gene in batch])
        params = {
            "query": f"({query}) AND (organism_id:9606)",
            "fields": "accession,gene_names,xref_reactome,go_p",
            "format": "tsv",
            "size": 500,
        }
        entry = {
            "batch": batch_num,
            "n_genes": len(batch),
            "genes": batch,
            "status": "ok",
            "n_result_rows": 0,
            "error": None,
        }
        try:
            response = requests.get(UNIPROT_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            lines = response.text.strip().split("\n")
            if len(lines) > 1:
                entry["n_result_rows"] = len(lines) - 1
                for line in lines[1:]:
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        accession = parts[0]
                        gene_names = parts[1].split(" ")[0] if parts[1] else None
                        reactome_ids = parts[2].split(";") if len(parts) > 2 and parts[2] else []
                        go_ids = parts[3].split(";") if len(parts) > 3 and parts[3] else []

                        for rid in reactome_ids:
                            if rid.strip():
                                gene_pathway_mappings.append(
                                    {
                                        "gene_symbol": gene_names,
                                        "uniprot_accession": accession,
                                        "pathway_id": rid.strip(),
                                        "database": "reactome",
                                    }
                                )
                        for gid in go_ids:
                            if gid.strip():
                                gene_pathway_mappings.append(
                                    {
                                        "gene_symbol": gene_names,
                                        "uniprot_accession": accession,
                                        "pathway_id": gid.strip(),
                                        "database": "gene_ontology",
                                    }
                                )
            time.sleep(RATE_LIMIT_SEC)
        except requests.exceptions.Timeout:
            entry["status"] = "timeout"
            entry["error"] = "timeout"
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
        batch_log.append(entry)
        print(
            f"  batch {batch_num}/{n_batches}: {entry['status']}"
            f" genes={len(batch)} result_rows={entry['n_result_rows']}"
            + (f" error={entry['error']}" if entry["error"] else "")
        )

    return pd.DataFrame(gene_pathway_mappings), {
        "n_batches": n_batches,
        "batches_ok": sum(1 for b in batch_log if b["status"] == "ok"),
        "batches_failed": [b for b in batch_log if b["status"] != "ok"],
        "batch_log": batch_log,
    }


def main() -> int:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"Loading target genes from {CG_PATH}")
    cg = pd.read_csv(CG_PATH)
    target_genes = sorted(cg["gene_symbol"].dropna().astype(str).str.strip().unique())
    n_edges = len(cg)
    print(f"Target gene set: {len(target_genes)} genes, {n_edges} compound-gene edges")

    print("Querying UniProt (Phase 5 method)...")
    gpm_new, query_meta = get_uniprot_pathways_phase5(target_genes)

    if gpm_new.empty and query_meta["batches_ok"] == 0:
        print("ERROR: UniProt query produced no data and all batches failed.")
        report = {
            "timestamp_utc": ts,
            "status": "failed",
            "query_meta": query_meta,
            "target_genes_n": len(target_genes),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    OUT_GPM.parent.mkdir(parents=True, exist_ok=True)
    gpm_new.to_parquet(OUT_GPM, index=False)
    print(f"Wrote {OUT_GPM} ({len(gpm_new):,} rows)")

    old = pd.read_parquet(OLD_GPM)
    target_set = set(target_genes)
    old_genes_all = set(old["gene_symbol"].dropna().astype(str).str.strip())
    new_genes_all = set(gpm_new["gene_symbol"].dropna().astype(str).str.strip())

    mapped_new = {g for g in target_set if g in new_genes_all}
    mapped_old = {g for g in target_set if g in old_genes_all}
    unmapped_new = sorted(target_set - mapped_new)
    unmapped_old = sorted(target_set - mapped_old)

    gained_genes = sorted(mapped_new - mapped_old)
    lost_genes = sorted(mapped_old - mapped_new)

    unmapped_excluded = [g for g in unmapped_new if g in EXCLUDED_UNMAPPED]
    unmapped_other = [g for g in unmapped_new if g not in EXCLUDED_UNMAPPED]

    sample = gpm_new.head(10).to_dict(orient="records")

    report = {
        "timestamp_utc": ts,
        "status": "partial" if query_meta["batches_failed"] else "complete",
        "method": "Phase 5 notebook Section 4 — UniProt REST search, organism_id:9606",
        "source_target_file": str(CG_PATH),
        "output_file": str(OUT_GPM),
        "original_gpm": str(OLD_GPM),
        "target_genes_n": len(target_genes),
        "compound_gene_edges_n": n_edges,
        "query_meta": {
            "n_batches": query_meta["n_batches"],
            "batches_ok": query_meta["batches_ok"],
            "n_batches_failed": len(query_meta["batches_failed"]),
            "batches_failed": query_meta["batches_failed"],
        },
        "new_table": {
            "total_rows": int(len(gpm_new)),
            "unique_genes": int(gpm_new["gene_symbol"].nunique()),
            "unique_pathways": int(gpm_new["pathway_id"].nunique()),
            "reactome_rows": int((gpm_new["database"] == "reactome").sum()),
            "go_rows": int((gpm_new["database"] == "gene_ontology").sum()),
        },
        "old_table": {
            "total_rows": int(len(old)),
            "unique_genes": int(old["gene_symbol"].nunique()),
            "unique_pathways": int(old["pathway_id"].nunique()),
        },
        "target_coverage": {
            "with_pathway_new": len(mapped_new),
            "with_pathway_old": len(mapped_old),
            "without_pathway_new": len(unmapped_new),
            "without_pathway_old": len(unmapped_old),
            "coverage_pct_new": round(100 * len(mapped_new) / len(target_set), 1),
            "coverage_pct_old": round(100 * len(mapped_old) / len(target_set), 1),
            "delta_genes_with_pathway": len(mapped_new) - len(mapped_old),
        },
        "genes_gained_pathway_membership": gained_genes,
        "genes_lost_pathway_membership_vs_old_on_target_set": lost_genes,
        "unmapped_new_excluded_by_design": unmapped_excluded,
        "unmapped_new_other": unmapped_other,
        "unmapped_new_all": unmapped_new,
        "sample_rows": sample,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("sample_rows", "genes_gained_pathway_membership", "unmapped_new_all", "query_meta")}, indent=2))
    print(f"Full report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
