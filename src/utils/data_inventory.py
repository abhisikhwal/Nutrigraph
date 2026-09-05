"""
Repo data inventory: scan data/raw and data/processed for key datasets.
Detects files by path patterns and column signatures; writes repo_manifest.json.
No web APIs; local-only. Windows-path safe (pathlib).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Signatures: (path_contains_patterns, required_columns_lower) -> dataset_key
DATASET_SIGNATURES = [
    (["pharmgkb_chemicals", ".parquet"], {"name", "pharmgkb", "pubchem", "cid"}, "pharmgkb_chemicals"),
    (["pharmgkb_chemicals", ".csv"], {"name", "pharmgkb", "pubchem", "cid"}, "pharmgkb_chemicals"),
    (["food_compound_gene_links", ".parquet"], {"compound", "gene"}, "food_compound_gene_links"),
    (["food_compound_gene_links", ".csv"], {"compound", "gene"}, "food_compound_gene_links"),
    (["gene_chemical_links"], {"compound", "gene", "chemical"}, "pharmgkb_chemical_gene_links"),
    (["compound_target_edges_bindingdb", ".parquet"], {"compound", "target"}, "bindingdb_edges"),
    (["compound_target_edges_bindingdb", ".csv"], {"compound", "target"}, "bindingdb_edges"),
    (["chembl", ".parquet"], {"chembl", "compound"}, "chembl_compound"),
    (["chembl", ".csv"], {"chembl", "compound"}, "chembl_compound"),
    (["inchikey_to_compound_id", ".json"], set(), "coconut_inchikey_mapping"),
    (["compound_registry", "coconut"], {"coconut", "inchikey"}, "coconut_registry"),
    (["ingredient_compound", ".parquet"], {"ingredient", "compound"}, "ingredient_compounds"),
    (["ingredient_compound", ".csv"], {"ingredient", "compound"}, "ingredient_compounds"),
    (["compound_metabolite_links", ".parquet"], {"compound", "metabolite"}, "metabolomics_links"),
    (["compound_metabolite_links", ".csv"], {"compound", "metabolite"}, "metabolomics_links"),
    (["compound_master", ".csv"], {"inchikey", "compound"}, "compound_master"),
    (["compound_master", ".parquet"], {"inchikey", "compound"}, "compound_master"),
    (["targets", ".parquet"], {"target", "uniprot", "gene"}, "chembl_targets"),
    (["targets", ".csv"], {"target", "uniprot", "gene"}, "chembl_targets"),
    (["pathway_gene", ".parquet"], {"pathway", "gene"}, "pathway_gene"),
    (["pathway_gene_signatures"], {"pathway", "gene"}, "pathway_gene_signatures"),
]


def _normalize_columns_for_signature(cols: List[str]) -> Set[str]:
    """Lowercase and reduce to 'word stems' for flexible matching (e.g. compound_id -> compound)."""
    out: Set[str] = set()
    for c in cols:
        c = c.lower().replace("_", " ").replace("-", " ")
        for part in c.split():
            if len(part) > 2:
                out.add(part)
    return out


def _detect_dataset(path: Path, columns: List[str]) -> Optional[str]:
    path_str = path.as_posix().lower()
    col_stems = _normalize_columns_for_signature(columns)
    for path_parts, required_stems, key in DATASET_SIGNATURES:
        if not all(p in path_str for p in path_parts):
            continue
        if required_stems and not (required_stems <= col_stems):
            # Check partial: at least one of compound/target/gene etc
            if key == "pharmgkb_chemicals" and ("name" in col_stems or "pharmgkb" in col_stems):
                return key
            if key == "bindingdb_edges" and ("compound" in col_stems and ("target" in col_stems or "uniprot" in col_stems)):
                return key
            if required_stems & col_stems:
                return key
        else:
            return key
    return None


def _sample_row_count(path: Path, max_csv_lines: int = 2_000_000) -> Optional[int]:
    """Light sampling: parquet full count; CSV line count if file small else None."""
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            import pandas as pd
            df = pd.read_parquet(path, columns=[])
            return len(df)
        if path.suffix.lower() == ".csv":
            if path.stat().st_size > 100 * 1024 * 1024:
                return None
            import pandas as pd
            df = pd.read_csv(path, nrows=50000, low_memory=False, dtype=str)
            return len(df) if len(df) < 50000 else None
    except Exception:
        return None
    return None


def _get_columns(path: Path) -> List[str]:
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            import pandas as pd
            df = pd.read_parquet(path, columns=[])
            return list(df.columns)
        if path.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(path, nrows=0)
            return list(df.columns)
    except Exception:
        pass
    return []


def _scan_dir(root: Path, extensions: Tuple[str, ...] = (".parquet", ".pq", ".csv", ".json")) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            out.append(p)
    return out


def build_repo_manifest(repo_root: Path) -> Dict[str, Any]:
    """
    Recursively scan data/raw and data/processed; detect datasets; write manifest.
    Returns manifest dict (also suitable for repo_manifest.json).
    """
    repo_root = Path(repo_root).resolve()
    data_raw = repo_root / "data" / "raw"
    data_processed = repo_root / "data" / "processed"
    canonical_dir = data_processed / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    files: List[Dict[str, Any]] = []
    seen_rel: Set[str] = set()
    missing_candidates: List[str] = []

    expected_patterns = [
        "pharmgkb_chemicals.parquet",
        "food_compound_gene_links.parquet",
        "compound_target_edges_bindingdb.parquet",
        "chembl*.parquet",
        "inchikey_to_compound_id.json",
        "ingredient_compound*.parquet",
        "compound_metabolite_links.parquet",
        "compound_master.csv",
    ]

    for base in (data_raw, data_processed):
        if not base.exists():
            continue
        for path in _scan_dir(base):
            try:
                rel = path.relative_to(repo_root).as_posix()
            except ValueError:
                rel = path.as_posix()
            if rel in seen_rel:
                continue
            seen_rel.add(rel)
            size = path.stat().st_size
            columns: List[str] = []
            row_count: Optional[int] = None
            detected: Optional[str] = None
            if path.suffix.lower() in (".parquet", ".pq", ".csv"):
                columns = _get_columns(path)
                row_count = _sample_row_count(path)
                detected = _detect_dataset(path, columns)
            elif path.suffix.lower() == ".json":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        row_count = len(data)
                    elif isinstance(data, list):
                        row_count = len(data)
                    else:
                        row_count = None
                except Exception:
                    row_count = None
                detected = "coconut_inchikey_mapping" if "inchikey" in path.name.lower() or "compound" in path.name.lower() else None
            entry = {
                "path": rel,
                "size_bytes": size,
                "columns": columns[:50],
                "row_count": row_count,
                "detected_schema": detected,
            }
            files.append(entry)

    # Build MISSING_CANDIDATES from expected patterns
    found_names = {Path(e["path"]).name for e in files}
    for pat in expected_patterns:
        if "*" in pat:
            if not any(pat.replace("*", "") in n for n in found_names):
                missing_candidates.append(pat)
        else:
            if pat not in found_names and not any(pat in p for p in [e["path"] for e in files]):
                missing_candidates.append(pat)

    manifest = {
        "repo_root": str(repo_root),
        "files": files,
        "n_files": len(files),
        "MISSING_CANDIDATES": missing_candidates,
        "by_detected_schema": {},
    }
    for e in files:
        d = e.get("detected_schema")
        if d:
            manifest["by_detected_schema"].setdefault(d, []).append(e["path"])

    return manifest


def write_manifest(repo_root: Path, output_path: Optional[Path] = None) -> Path:
    """Build manifest and write to data/processed/canonical/repo_manifest.json (or output_path)."""
    manifest = build_repo_manifest(repo_root)
    out = output_path or (Path(repo_root) / "data" / "processed" / "canonical" / "repo_manifest.json")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote repo_manifest.json: %s files, MISSING_CANDIDATES: %s", manifest["n_files"], len(manifest["MISSING_CANDIDATES"]))
    return out
