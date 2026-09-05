#!/usr/bin/env python3
"""
Build authoritative UniProt accession -> HGNC symbol mapping table.

Sources (priority):
  1. HGNC complete set (uniprot_ids cross-references on Approved genes)
  2. UniProt human idmapping (Gene_Name, only when symbol is HGNC-approved)

No fuzzy matching. Ambiguous accessions are flagged, not silently resolved.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "interim" / "mappings"
OUT_PARQUET = OUT_DIR / "uniprot_to_hgnc.parquet"
OUT_REPORT = OUT_DIR / "uniprot_to_hgnc_build_report.json"
HGNC_CACHE = OUT_DIR / "hgnc_complete_set.txt"

HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
UNIPROT_IDMAPPING_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz"
)
UA = {"User-Agent": "global-food-genome-uniprot-hgnc-map/1.0"}


def parse_pipe_field(value: str) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return [p.strip() for p in s.split("|") if p.strip()]


def normalize_accession(acc: str) -> str:
    return acc.strip().upper()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"Downloaded {dest.name}: {len(data):,} bytes", flush=True)


def load_hgnc_approved_symbols(path: Path) -> set[str]:
    approved: set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status", "").strip() == "Approved":
                sym = row.get("symbol", "").strip()
                if sym:
                    approved.add(sym)
    return approved


def build_hgnc_uniprot_mappings(path: Path) -> tuple[list[dict], dict[str, set[str]]]:
    """
    Returns rows and accession -> set(hgnc_symbols) for ambiguity detection.
  Only Approved genes with uniprot_ids are used.
    """
    acc_to_symbols: dict[str, set[str]] = defaultdict(set)
    rows: list[dict] = []

    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status", "").strip() != "Approved":
                continue
            symbol = row.get("symbol", "").strip()
            if not symbol:
                continue
            for acc in parse_pipe_field(row.get("uniprot_ids", "")):
                norm = normalize_accession(acc)
                if not norm or len(norm) > 20:
                    continue
                acc_to_symbols[norm].add(symbol)

    for acc, symbols in sorted(acc_to_symbols.items()):
        if len(symbols) == 1:
            rows.append(
                {
                    "uniprot_accession": acc,
                    "hgnc_symbol": next(iter(symbols)),
                    "source": "HGNC_complete_set",
                    "mapping_type": "primary_canonical",
                }
            )
        else:
            for sym in sorted(symbols):
                rows.append(
                    {
                        "uniprot_accession": acc,
                        "hgnc_symbol": sym,
                        "source": "HGNC_complete_set",
                        "mapping_type": "ambiguous_multi_hgnc",
                    }
                )

    return rows, acc_to_symbols


def stream_uniprot_idmapping(
    url: str,
    approved_symbols: set[str],
    existing: dict[str, str],
    ambiguous: set[str],
) -> list[dict]:
    """
    Add secondary mappings from UniProt idmapping Gene_Name entries.
    Only when Gene_Name is an HGNC-approved symbol and accession is not ambiguous.
    """
    added: list[dict] = []
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for raw in gz:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                acc, db, val = parts[0], parts[1], parts[2]
                if db != "Gene_Name":
                    continue
                norm = normalize_accession(acc)
                if not norm or norm in ambiguous:
                    continue
                if norm in existing:
                    continue
                gene = val.strip()
                if gene not in approved_symbols:
                    continue
                existing[norm] = gene
                added.append(
                    {
                        "uniprot_accession": norm,
                        "hgnc_symbol": gene,
                        "source": "UniProt_idmapping_HUMAN_9606",
                        "mapping_type": "secondary_accession",
                    }
                )
    return added


def build_old_local_map(root: Path) -> dict[str, str]:
    """Reconstruct the legacy ~579-entry map from canonical target tables."""
    frames = []
    for rel in [
        "data/processed/canonical/targets.parquet",
        "data/processed/canonical/target_pathways.parquet",
        "data/processed/canonical/target_pathways_enhanced.parquet",
        "data/processed/canonical/target_pathways_functional.parquet",
        "data/processed/canonical/compound_targets.parquet",
    ]:
        fp = root / rel
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        u = next((c for c in df.columns if "uniprot" in c.lower()), None)
        g = next(
            (c for c in df.columns if "gene_symbol" in c.lower() or c.lower() in ["gene", "gene_name"]),
            None,
        )
        if u and g:
            t = df[[u, g]].dropna().copy()
            t.columns = ["u", "g"]
            frames.append(t)
    if not frames:
        return {}
    m = pd.concat(frames, ignore_index=True)
    mp: dict[str, str] = {}
    for _, r in m.iterrows():
        u = str(r["u"]).strip().upper()
        g = str(r["g"]).strip().upper()
        if u and g and u != "NAN" and g != "NAN" and len(u) <= 20 and len(g) <= 20 and u not in mp:
            mp[u] = g
    return mp


def sample_chembl_accessions(root: Path, limit: int = 5000) -> set[str]:
    import sqlite3

    db = root / "data/raw/chembl/chembl_36/chembl_36_sqlite/chembl_36.db"
    if not db.exists():
        return set()
    con = sqlite3.connect(str(db))
    cur = con.cursor()
    cur.execute(
        """
        SELECT DISTINCT csq.accession
        FROM activities a
        JOIN assays s ON a.assay_id = s.assay_id
        JOIN target_dictionary td ON s.tid = td.tid
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_sequences csq ON tc.component_id = csq.component_id
        WHERE td.target_type = ?
          AND td.organism = ?
          AND COALESCE(s.confidence_score, 0) >= ?
          AND a.standard_value IS NOT NULL
          AND a.standard_type IN (?, ?, ?, ?)
          AND (
            (a.standard_units = ? AND a.standard_value <= ?)
            OR (a.standard_units = ? AND a.standard_value <= ?)
          )
          AND csq.accession IS NOT NULL
          AND csq.accession != ''
        LIMIT ?
        """,
        (
            "SINGLE PROTEIN",
            "Homo sapiens",
            8,
            "Ki",
            "Kd",
            "IC50",
            "EC50",
            "nM",
            10000,
            "uM",
            10,
            limit,
        ),
    )
    acc = {normalize_accession(r[0]) for r in cur.fetchall() if r[0]}
    con.close()
    return acc


def sample_bindingdb_accessions(root: Path, limit: int = 5000) -> set[str]:
    import csv

    path = root / "data/raw/BindingDB_All.tsv"
    if not path.exists():
        return set()
    acc: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        i_org = h.index("Target Source Organism According to Curator or DataSource")
        i_u1 = h.index("UniProt (SwissProt) Primary ID of Target Chain 1")
        i_u2 = h.index("UniProt (TrEMBL) Primary ID of Target Chain 1")
        for row in r:
            if len(acc) >= limit:
                break
            org = (row[i_org] if i_org < len(row) else "").strip().lower()
            if org and ("homo sapiens" not in org and "human" not in org):
                continue
            u = (
                (row[i_u1] if i_u1 < len(row) else "").strip()
                or (row[i_u2] if i_u2 < len(row) else "").strip()
            )
            if u:
                acc.add(normalize_accession(u))
    return acc


def resolve_rate(accessions: set[str], mp: dict[str, str], ambiguous: set[str]) -> dict:
    if not accessions:
        return {"n": 0, "resolved": 0, "rate": 0.0, "ambiguous": 0}
    resolved = sum(1 for a in accessions if a in mp)
    amb = sum(1 for a in accessions if a in ambiguous)
    return {
        "n": len(accessions),
        "resolved": resolved,
        "rate": round(resolved / len(accessions), 4),
        "ambiguous": amb,
        "unresolved": len(accessions) - resolved - amb,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HGNC_CACHE.exists():
        download_file(HGNC_URL, HGNC_CACHE)
    else:
        print(f"Using cached {HGNC_CACHE}", flush=True)

    approved_symbols = load_hgnc_approved_symbols(HGNC_CACHE)
    print(f"HGNC approved symbols: {len(approved_symbols):,}", flush=True)

    hgnc_rows, acc_to_symbols = build_hgnc_uniprot_mappings(HGNC_CACHE)
    ambiguous_accs = {acc for acc, syms in acc_to_symbols.items() if len(syms) > 1}
    canonical_map: dict[str, str] = {
        r["uniprot_accession"]: r["hgnc_symbol"]
        for r in hgnc_rows
        if r["mapping_type"] == "primary_canonical"
    }
    print(
        f"HGNC primary mappings: {len(canonical_map):,}; ambiguous accessions: {len(ambiguous_accs):,}",
        flush=True,
    )

    print("Streaming UniProt human idmapping for secondary accessions...", flush=True)
    uniprot_rows = stream_uniprot_idmapping(
        UNIPROT_IDMAPPING_URL,
        approved_symbols,
        canonical_map,
        ambiguous_accs,
    )
    print(f"UniProt secondary mappings added: {len(uniprot_rows):,}", flush=True)

    all_rows = hgnc_rows + uniprot_rows
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(
        subset=["uniprot_accession", "hgnc_symbol", "mapping_type"], keep="first"
    )
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET} ({len(df):,} rows)", flush=True)

    # Coverage verification
    old_map = build_old_local_map(ROOT)
    chembl_acc = sample_chembl_accessions(ROOT)
    bdb_acc = sample_bindingdb_accessions(ROOT)
    union_acc = chembl_acc | bdb_acc

    cg = pd.read_csv(
        ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv",
        usecols=["gene_symbol"],
    )
    existing_genes = set(cg["gene_symbol"].dropna().astype(str).str.strip())

    hgnc_gene_in_map = existing_genes & set(canonical_map.values()) | (
        existing_genes & approved_symbols
    )

    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "hgnc": {"url": HGNC_URL, "cache": str(HGNC_CACHE.relative_to(ROOT))},
            "uniprot_idmapping": {"url": UNIPROT_IDMAPPING_URL},
        },
        "table": {
            "path": str(OUT_PARQUET.relative_to(ROOT)),
            "total_rows": int(len(df)),
            "unique_accessions": int(df["uniprot_accession"].nunique()),
            "unique_hgnc_symbols": int(df["hgnc_symbol"].nunique()),
            "by_mapping_type": df["mapping_type"].value_counts().to_dict(),
            "by_source": df["source"].value_counts().to_dict(),
        },
        "ambiguities": {
            "n_ambiguous_accessions": len(ambiguous_accs),
            "examples": sorted(ambiguous_accs)[:20],
        },
        "coverage": {
            "old_local_map_size": len(old_map),
            "canonical_resolvable_map_size": len(canonical_map),
            "chembl_p3_sample": {
                "old_map": resolve_rate(chembl_acc, old_map, set()),
                "new_map": resolve_rate(chembl_acc, canonical_map, ambiguous_accs),
            },
            "bindingdb_human_sample": {
                "old_map": resolve_rate(bdb_acc, old_map, set()),
                "new_map": resolve_rate(bdb_acc, canonical_map, ambiguous_accs),
            },
            "union_sample": {
                "old_map": resolve_rate(union_acc, old_map, set()),
                "new_map": resolve_rate(union_acc, canonical_map, ambiguous_accs),
            },
        },
        "round_trip": {
            "genes_in_compound_gene_normalized": len(existing_genes),
            "genes_in_hgnc_approved": len(existing_genes & approved_symbols),
            "genes_with_uniprot_in_hgnc_table": int(
                df[df["hgnc_symbol"].isin(existing_genes)]["hgnc_symbol"].nunique()
            ),
        },
    }

    with OUT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUT_REPORT}", flush=True)
    print(json.dumps(report["coverage"], indent=2), flush=True)
    print(json.dumps(report["table"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
