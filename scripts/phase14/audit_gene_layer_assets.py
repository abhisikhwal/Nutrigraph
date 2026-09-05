"""
Option 1 audit: Locate existing assets for Strengthen Human Gene Layer.
Outputs: data/processed/canonical/reports/audit_gene_layer_assets.json + console print.
No external API calls. Windows-safe paths.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_df(path: Path):
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            import pandas as pd
            return pd.read_parquet(path)
        import pandas as pd
        return pd.read_csv(path, low_memory=False, nrows=5)
    except Exception:
        return None

def _col(df, names):
    if df is None or df.empty:
        return None
    low = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "_")
        if k in low:
            return low[k]
    return None

def run_audit(repo_root: Path) -> dict:
    processed = repo_root / "data" / "processed"
    raw = repo_root / "data" / "raw"
    canonical = processed / "canonical"
    report = {
        "coconut": {"found": [], "key_columns": [], "notes": ""},
        "foodb_registry": {"found": [], "key_columns": [], "notes": ""},
        "phase12_genetics": {"found": [], "key_columns": {}, "notes": ""},
        "phase16_bindingdb": {"found": [], "key_columns": {}, "notes": ""},
        "uniprot_gene_mapping": {"found": [], "key_columns": {}, "notes": ""},
        "chembl": {"found": [], "key_columns": {}, "notes": ""},
    }

    # COCONUT
    coco_json = processed / "phase15_coconut" / "inchikey_to_compound_id.json"
    if coco_json.exists():
        try:
            with open(coco_json, encoding="utf-8") as f:
                d = json.load(f)
            n_coconut = sum(1 for v in d.values() if v and str(v).upper().startswith("COCONUT_"))
            report["coconut"]["found"].append(str(coco_json.relative_to(repo_root)))
            report["coconut"]["key_columns"] = ["InChIKey -> compound_id (numeric or COCONUT_CNP...)"]
            report["coconut"]["notes"] = "Invert for COCONUT_id->InChIKey. n_coconut_ids=%s" % n_coconut
        except Exception as e:
            report["coconut"]["notes"] = "Error: %s" % e
    for p in [canonical / "compound_master.csv", canonical / "compound_master.parquet"]:
        if p.exists():
            df = _load_df(p)
            if df is not None and "coconut_id" in df.columns:
                report["coconut"]["found"].append(str(p.relative_to(repo_root)))
                break
    bridge = canonical / "compound_identity_bridge.csv"
    if bridge.exists():
        df = _load_df(bridge)
        if df is not None and "source_id" in df.columns and "inchikey" in df.columns:
            report["coconut"]["found"].append(str(bridge.relative_to(repo_root)) + " (namespace COCONUT)")

    # FooDB
    reg = processed / "phase14_chemical_identity" / "compound_registry_lookup.json"
    if reg.exists():
        report["foodb_registry"]["found"].append(str(reg.relative_to(repo_root)))
        report["foodb_registry"]["key_columns"] = ["InChIKey -> compound_id, name, source"]
    for p in [canonical / "compound_master.csv", canonical / "compound_master.parquet"]:
        if p.exists():
            report["foodb_registry"]["found"].append(str(p.relative_to(repo_root)))
            break

    # Phase12 genetics
    phase12 = processed / "phase12_genetics"
    for name in ["food_compound_gene_links.parquet", "pharmgkb_chemicals.parquet"]:
        p = phase12 / name
        if p.exists():
            df = _load_df(p)
            report["phase12_genetics"]["found"].append(str(p.relative_to(repo_root)))
            if df is not None and not df.empty:
                report["phase12_genetics"]["key_columns"][name] = list(df.columns)
    if not report["phase12_genetics"]["found"]:
        for p in processed.rglob("food_compound_gene_links.parquet"):
            report["phase12_genetics"]["found"].append(str(p.relative_to(repo_root)))
            break
        for p in processed.rglob("pharmgkb_chemicals.parquet"):
            report["phase12_genetics"]["found"].append(str(p.relative_to(repo_root)))
            break

    # Phase16 BindingDB
    bdb = processed / "phase16_bindingdb" / "compound_target_edges_bindingdb.parquet"
    if bdb.exists():
        df = _load_df(bdb)
        report["phase16_bindingdb"]["found"].append(str(bdb.relative_to(repo_root)))
        if df is not None and not df.empty:
            report["phase16_bindingdb"]["key_columns"] = list(df.columns)

    # UniProt -> Gene (local)
    candidates = [
        canonical / "targets.parquet",
        canonical / "target_pathways.parquet",
        canonical / "compound_targets.parquet",
        processed / "features" / "target_functional_clusters.csv",
    ]
    for p in candidates:
        if p.exists():
            df = _load_df(p)
            if df is not None and not df.empty:
                u = _col(df, ["uniprot_id", "uniprot_accession", "uniprot_accession_x", "uniprot_accession_y"])
                g = _col(df, ["gene_symbol", "gene_name", "gene"])
                if u or g:
                    report["uniprot_gene_mapping"]["found"].append(str(p.relative_to(repo_root)))
                    report["uniprot_gene_mapping"]["key_columns"][p.name] = {"uniprot": u, "gene": g}
    for p in list(processed.rglob("*.parquet")) + list(processed.rglob("*.csv")):
        if p.stat().st_size > 50_000_000:
            continue
        if p in candidates:
            continue
        try:
            df = _load_df(p)
            if df is None or df.empty or len(df) > 100000:
                continue
        except Exception:
            continue
        u = _col(df, ["uniprot_id", "uniprot_accession"])
        g = _col(df, ["gene_symbol", "gene_name", "gene"])
        if u and g:
            report["uniprot_gene_mapping"]["found"].append(str(p.relative_to(repo_root)))
            if p.name not in report["uniprot_gene_mapping"]["key_columns"]:
                report["uniprot_gene_mapping"]["key_columns"][p.name] = {"uniprot": u, "gene": g}

    # ChEMBL
    for p in [canonical / "compound_targets.parquet", canonical / "targets.parquet", raw / "chembl"]:
        if p.exists():
            report["chembl"]["found"].append(str(p.relative_to(repo_root)))
            if p.suffix.lower() in (".parquet", ".csv"):
                df = _load_df(p)
                if df is not None and not df.empty:
                    report["chembl"]["key_columns"][p.name] = list(df.columns)

    return report

def main():
    repo_root = Path(REPO_ROOT).resolve()
    if not (repo_root / "data").exists():
        print("ERROR: no data/ dir at repo root:", repo_root)
        return 1
    report = run_audit(repo_root)
    out_dir = repo_root / "data" / "processed" / "canonical" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_gene_layer_assets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("--- Audit: Strengthen Human Gene Layer ---")
    print("COCONUT:", report["coconut"]["found"] or "NOT FOUND", "|", report["coconut"].get("notes", ""))
    print("FooDB registry:", report["foodb_registry"]["found"] or "NOT FOUND")
    print("Phase12 genetics:", report["phase12_genetics"]["found"] or "NOT FOUND")
    print("Phase16 BindingDB:", report["phase16_bindingdb"]["found"] or "NOT FOUND")
    print("UniProt->Gene mapping:", report["uniprot_gene_mapping"]["found"] or "NOT FOUND")
    print("ChEMBL:", report["chembl"]["found"] or "NOT FOUND")
    print("Report written:", out_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
