"""
Completeness re-pull: Phase 5 UniProt method with smaller batches (no 500-row truncation).
Writes gene_pathway_mappings_v2_full.parquet; does not overwrite gpm or capped v2.
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
CAPPED_V2 = ROOT / "data/interim/pathways/gene_pathway_mappings_v2.parquet"
OUT_FULL = ROOT / "data/interim/pathways/gene_pathway_mappings_v2_full.parquet"
REPORT_PATH = ROOT / "reports/gene_pathway_mappings_v2_full_proposal.json"

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
BATCH_SIZE = 25
ROW_CAP = 500
RATE_LIMIT_SEC = 1.0
REQUEST_TIMEOUT = 30
MATERIAL_DELTA_MIN = 10  # pathways added to count as "materially" truncated in capped run


def get_uniprot_pathways_phase5(
    gene_symbols: list[str],
    batch_size: int = BATCH_SIZE,
) -> tuple[pd.DataFrame, dict]:
    gene_pathway_mappings: list[dict] = []
    batch_log: list[dict] = []
    n_batches = (len(gene_symbols) + batch_size - 1) // batch_size

    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i : i + batch_size]
        batch_num = i // batch_size + 1
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
            "hit_row_cap": False,
            "error": None,
        }
        try:
            response = requests.get(UNIPROT_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            lines = response.text.strip().split("\n")
            if len(lines) > 1:
                n_rows = len(lines) - 1
                entry["n_result_rows"] = n_rows
                entry["hit_row_cap"] = n_rows >= ROW_CAP
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
        cap_flag = " CAP-HIT" if entry.get("hit_row_cap") else ""
        print(
            f"  batch {batch_num}/{n_batches}: {entry['status']}"
            f" genes={len(batch)} result_rows={entry['n_result_rows']}{cap_flag}"
            + (f" error={entry['error']}" if entry["error"] else "")
        )

    return pd.DataFrame(gene_pathway_mappings), {
        "batch_size": batch_size,
        "n_batches": n_batches,
        "batches_ok": sum(1 for b in batch_log if b["status"] == "ok"),
        "batches_failed": [b for b in batch_log if b["status"] != "ok"],
        "batches_hit_cap": [b for b in batch_log if b.get("hit_row_cap")],
        "rows_per_batch": [
            {"batch": b["batch"], "n_genes": b["n_genes"], "n_result_rows": b["n_result_rows"], "hit_cap": b.get("hit_row_cap", False)}
            for b in batch_log
        ],
        "batch_log": batch_log,
    }


def pathway_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    return (
        df.groupby("gene_symbol")["pathway_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )


def main() -> int:
    ts = datetime.now(timezone.utc).isoformat()
    cg = pd.read_csv(CG_PATH)
    target_genes = sorted(cg["gene_symbol"].dropna().astype(str).str.strip().unique())
    target_set = set(target_genes)
    n_edges = len(cg)
    shbg_edges = int((cg["gene_symbol"] == "SHBG").sum())

    print(f"Target genes: {len(target_genes)}, batch_size={BATCH_SIZE}")
    gpm_full, query_meta = get_uniprot_pathways_phase5(target_genes)

    if gpm_full.empty and query_meta["batches_ok"] == 0:
        report = {"timestamp_utc": ts, "status": "failed", "query_meta": query_meta}
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    gpm_full.to_parquet(OUT_FULL, index=False)
    print(f"Wrote {OUT_FULL} ({len(gpm_full):,} rows)")

    capped = pd.read_parquet(CAPPED_V2)
    capped_counts = pathway_counts(capped)
    full_counts = pathway_counts(gpm_full)

    full_genes_all = set(gpm_full["gene_symbol"].dropna().astype(str).str.strip())
    capped_genes_all = set(capped["gene_symbol"].dropna().astype(str).str.strip())
    mapped_full = {g for g in target_set if g in full_genes_all}
    mapped_capped = {g for g in target_set if g in capped_genes_all}

    # Truncation recovery: target genes with more pathways in full vs capped.
    truncation_deltas: list[dict] = []
    for gene in sorted(target_set):
        old_n = capped_counts.get(gene, 0)
        new_n = full_counts.get(gene, 0)
        if new_n > old_n:
            truncation_deltas.append(
                {
                    "gene_symbol": gene,
                    "capped_pathway_count": old_n,
                    "full_pathway_count": new_n,
                    "delta": new_n - old_n,
                }
            )
    truncation_deltas.sort(key=lambda x: -x["delta"])
    materially_truncated = [d for d in truncation_deltas if d["delta"] >= MATERIAL_DELTA_MIN]

    batches_hit_cap = query_meta["batches_hit_cap"]
    no_cap_hit = len(batches_hit_cap) == 0

    report = {
        "timestamp_utc": ts,
        "status": "complete" if no_cap_hit and not query_meta["batches_failed"] else "partial",
        "method": "Phase 5 Section 4 — UniProt REST, organism_id:9606, fields=accession,gene_names,xref_reactome,go_p",
        "batch_size": BATCH_SIZE,
        "row_cap": ROW_CAP,
        "no_batch_hit_row_cap": no_cap_hit,
        "batches_hit_row_cap": batches_hit_cap,
        "rows_per_batch": query_meta["rows_per_batch"],
        "source_target_file": str(CG_PATH),
        "capped_v2_file": str(CAPPED_V2),
        "output_file": str(OUT_FULL),
        "target_genes_n": len(target_genes),
        "compound_gene_edges_n": n_edges,
        "query_meta": {
            "n_batches": query_meta["n_batches"],
            "batches_ok": query_meta["batches_ok"],
            "n_batches_failed": len(query_meta["batches_failed"]),
            "batches_failed": query_meta["batches_failed"],
        },
        "coverage": {
            "capped_v2_target_with_pathway": len(mapped_capped),
            "full_target_with_pathway": len(mapped_full),
            "capped_pct": round(100 * len(mapped_capped) / len(target_set), 1),
            "full_pct": round(100 * len(mapped_full) / len(target_set), 1),
            "delta_genes": len(mapped_full) - len(mapped_capped),
        },
        "table_comparison": {
            "capped_v2": {
                "total_rows": int(len(capped)),
                "unique_genes": int(capped["gene_symbol"].nunique()),
                "unique_pathways": int(capped["pathway_id"].nunique()),
            },
            "full_v2": {
                "total_rows": int(len(gpm_full)),
                "unique_genes": int(gpm_full["gene_symbol"].nunique()),
                "unique_pathways": int(gpm_full["pathway_id"].nunique()),
                "delta_rows": int(len(gpm_full) - len(capped)),
                "delta_genes": int(gpm_full["gene_symbol"].nunique() - capped["gene_symbol"].nunique()),
                "delta_pathways": int(gpm_full["pathway_id"].nunique() - capped["pathway_id"].nunique()),
            },
        },
        "truncation_recovery": {
            "genes_with_any_increase": truncation_deltas,
            "genes_materially_truncated_in_capped_run": materially_truncated,
            "material_delta_threshold": MATERIAL_DELTA_MIN,
        },
        "shbg": {
            "in_target_gene_set": "SHBG" in target_set,
            "compound_gene_edges": shbg_edges,
            "pathway_rows_in_full": int((gpm_full["gene_symbol"] == "SHBG").sum()),
            "pathway_rows_in_capped": int((capped["gene_symbol"] == "SHBG").sum()),
            "note": "SHBG remains in compound-gene file with edges; zero UniProt pathway annotations expected",
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        k: v
        for k, v in report.items()
        if k
        not in (
            "rows_per_batch",
            "truncation_recovery",
            "batches_hit_row_cap",
            "query_meta",
        )
    }
    summary["n_genes_with_pathway_increase"] = len(truncation_deltas)
    summary["n_materially_truncated"] = len(materially_truncated)
    print(json.dumps(summary, indent=2))
    print(f"Full report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
