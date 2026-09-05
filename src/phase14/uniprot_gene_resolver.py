"""
Option 1: UniProt target resolution (local only, no web).
- uniprot_id -> gene_symbol from local files
- target_name -> gene with normalization, alias table, optional fuzzy95
Build canonical mapping; write target_name_alias_map.csv. No external API calls.
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Strip these prefixes only when the remainder is non-empty and looks safe (e.g. not just "of")
_STRIP_PREFIXES = re.compile(
    r"^(?:dimer\s+of\s+|protein\s+|the\s+)?(.+)$",
    re.IGNORECASE,
)


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _load_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False, dtype=str)
    except Exception as e:
        logger.warning("Load failed %s: %s", path, e)
        return None


def _col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    low = {c.lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "_")
        if k in low:
            return low[k]
    return None


def normalize_target_name(s: str) -> str:
    """
    Robust normalization for target_name matching:
    - lowercase
    - remove bracketed suffixes like " [501-599]" or "(human)"
    - strip safe prefixes: "Dimer of", "Protein", "The"
    - remove punctuation except hyphens inside token-like parts
    - collapse whitespace
    """
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s*\[[^\]]*\]\s*", " ", s)
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    m = _STRIP_PREFIXES.match(s)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"[^\w\s\-]", " ", s)
    return " ".join(s.split())


# Backward compatibility
def _normalize_target_name(s: str) -> str:
    return normalize_target_name(s)


def build_uniprot_to_gene_map(repo_root: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Scan data/processed for files with uniprot_id + gene_symbol/gene_name; build uniprot_id -> gene_symbol.
    """
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    canonical = processed / "canonical"
    uniprot_to_gene: Dict[str, str] = {}
    report: Dict[str, Any] = {"sources": [], "n_mappings": 0}

    candidates = [
        canonical / "targets.parquet",
        canonical / "target_pathways.parquet",
        canonical / "target_pathways_enhanced.parquet",
        canonical / "compound_targets.parquet",
        canonical / "target_pathways_functional.parquet",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = _load_df(p)
        if df is None or df.empty or len(df) > 500_000:
            continue
        u_col = _col(df, ["uniprot_id", "uniprot_accession", "uniprot_accession_x", "uniprot_accession_y"])
        g_col = _col(df, ["gene_symbol", "gene_name", "gene"])
        if not u_col or not g_col:
            continue
        for _, r in df.iterrows():
            u = _safe_str(r.get(u_col)).upper()
            g = _safe_str(r.get(g_col)).upper()
            if u and g and len(u) <= 20 and len(g) <= 30:
                if u not in uniprot_to_gene or len(g) < len(uniprot_to_gene.get(u, "")):
                    uniprot_to_gene[u] = g
        report["sources"].append(str(p.relative_to(repo_root)))
    report["n_mappings"] = len(uniprot_to_gene)
    logger.info("build_uniprot_to_gene_map: %s mappings from %s", len(uniprot_to_gene), report["sources"])
    return uniprot_to_gene, report


def build_target_name_to_gene_map(
    repo_root: Path,
    write_alias_csv: bool = True,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Build target_name -> gene_symbol from target_functional_clusters, targets.parquet, and any
    target tables under data/processed. Creates normalized + raw keys. Optionally writes
    target_name_alias_map.csv (normalized_target_name, raw_examples, gene_symbol, source_file, confidence).
    """
    repo_root = Path(repo_root).resolve()
    processed = repo_root / "data" / "processed"
    features = processed / "features"
    canonical = processed / "canonical"
    target_to_gene: Dict[str, str] = {}
    raw_examples: Dict[str, List[str]] = {}
    source_file_map: Dict[str, str] = {}
    confidence_map: Dict[str, str] = {}

    def add(norm_key: str, gene: str, raw: str, source: str, confidence: str = "exact"):
        if not norm_key or not gene or len(gene) > 15:
            return
        if norm_key not in target_to_gene or len(gene) < len(target_to_gene.get(norm_key, "")):
            target_to_gene[norm_key] = gene.upper()
            raw_examples.setdefault(norm_key, []).append(raw[:200])
            if len(raw_examples[norm_key]) > 5:
                raw_examples[norm_key] = raw_examples[norm_key][:5]
            source_file_map[norm_key] = source
            confidence_map[norm_key] = confidence

    # target_functional_clusters.csv
    clusters_path = features / "target_functional_clusters.csv"
    if clusters_path.exists():
        df = _load_df(clusters_path)
        if df is not None and not df.empty:
            genes_col = _col(df, ["sample_genes", "gene_symbol"])
            targets_col = _col(df, ["sample_targets", "target_name"])
            if genes_col and targets_col:
                for _, r in df.iterrows():
                    genes_raw = _safe_str(r.get(genes_col))
                    targets_raw = _safe_str(r.get(targets_col))
                    if not genes_raw or not targets_raw:
                        continue
                    try:
                        genes = ast.literal_eval(genes_raw) if (genes_raw.startswith("[") or genes_raw.startswith("(")) else [x.strip() for x in genes_raw.replace("'", "").split(",")]
                        targets = ast.literal_eval(targets_raw) if (targets_raw.startswith("[") or targets_raw.startswith("(")) else [x.strip() for x in targets_raw.replace("'", "").split(",")]
                    except Exception:
                        genes = [x.strip() for x in genes_raw.split(",")]
                        targets = [x.strip() for x in targets_raw.split(",")]
                    for i, t in enumerate(targets):
                        if not t:
                            continue
                        g = genes[i] if i < len(genes) else (genes[0] if genes else "")
                        if not g or len(g) > 15:
                            continue
                        add(normalize_target_name(t), g.upper(), t, clusters_path.name, "cluster")

    # targets.parquet: raw + normalized
    for p in [canonical / "targets.parquet"]:
        if not p.exists():
            continue
        df = _load_df(p)
        if df is None or df.empty:
            continue
        t_col = _col(df, ["target_name", "pref_name"])
        g_col = _col(df, ["gene_name", "gene_symbol", "gene"])
        if t_col and g_col:
            for _, r in df.iterrows():
                raw_t = _safe_str(r.get(t_col))
                t = normalize_target_name(raw_t)
                g = _safe_str(r.get(g_col)).upper()
                if t and g and len(g) <= 15:
                    add(t, g, raw_t, p.name, "targets_parquet")
        break

    # Scan other target-like tables under data/processed (quick)
    for p in list((processed / "canonical").glob("target*.parquet")) + list((processed / "canonical").glob("target*.csv")):
        if p.name.startswith("target_pathways") or p == canonical / "targets.parquet":
            continue
        df = _load_df(p)
        if df is None or df.empty or len(df) > 100_000:
            continue
        t_col = _col(df, ["target_name", "pref_name"])
        g_col = _col(df, ["gene_symbol", "gene_name", "gene"])
        if t_col and g_col:
            for _, r in df.iterrows():
                raw_t = _safe_str(r.get(t_col))
                t = normalize_target_name(raw_t)
                g = _safe_str(r.get(g_col)).upper()
                if t and g and len(g) <= 15:
                    add(t, g, raw_t, p.name, "scan")

    report: Dict[str, Any] = {"sources": list(set(source_file_map.values())), "n_mappings": len(target_to_gene)}

    if write_alias_csv and target_to_gene:
        alias_rows = []
        for norm_key, gene in target_to_gene.items():
            alias_rows.append({
                "normalized_target_name": norm_key,
                "raw_examples": "|".join(raw_examples.get(norm_key, [])[:3]),
                "gene_symbol": gene,
                "source_file": source_file_map.get(norm_key, ""),
                "confidence": confidence_map.get(norm_key, "exact"),
            })
        reports_dir = canonical / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / "target_name_alias_map.csv"
        try:
            pd.DataFrame(alias_rows).to_csv(out_path, index=False)
            logger.info("Wrote %s", out_path.name)
        except Exception as e:
            logger.warning("Could not write target_name_alias_map.csv: %s", e)

    logger.info("build_target_name_to_gene_map: %s mappings", len(target_to_gene))
    return target_to_gene, report


def resolve_target_to_gene(
    uniprot_id: Optional[str],
    target_name: Optional[str],
    uniprot_to_gene: Dict[str, str],
    target_name_to_gene: Dict[str, str],
    fuzzy_threshold: int = 95,
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Resolve (uniprot_id, target_name) to gene_symbol.
    Returns (gene_symbol, resolver_used, matched_key).
    Uses uniprot first, then normalized target_name, then fuzzy match (rapidfuzz >= fuzzy_threshold) as "fuzzy95".
    """
    u = _safe_str(uniprot_id).upper() if uniprot_id else ""
    if u and u in uniprot_to_gene:
        return uniprot_to_gene[u], "uniprot_map", None
    t = normalize_target_name(_safe_str(target_name)) if target_name else ""
    if t and t in target_name_to_gene:
        return target_name_to_gene[t], "target_name_map", t
    if t:
        for key, gene in target_name_to_gene.items():
            if key in t or t in key:
                return gene, "target_name_substring", key
        if len(t) <= 10 and t.replace("-", "").isalnum():
            return t.upper(), "target_name_as_symbol", t
        try:
            from rapidfuzz import fuzz
            best_score = 0
            best_key = None
            best_gene = None
            for key, gene in target_name_to_gene.items():
                score = fuzz.ratio(t, key)
                if score >= fuzzy_threshold and score > best_score:
                    best_score = score
                    best_key = key
                    best_gene = gene
            if best_gene:
                return best_gene, "fuzzy95", best_key
        except ImportError:
            pass
    return None, "unresolved", None


def resolve_target_to_gene_legacy(
    uniprot_id: Optional[str],
    target_name: Optional[str],
    uniprot_to_gene: Dict[str, str],
    target_name_to_gene: Dict[str, str],
) -> Tuple[Optional[str], str]:
    """Backward-compatible: returns (gene, resolver_used) without matched_key."""
    gene, resolver, _ = resolve_target_to_gene(uniprot_id, target_name, uniprot_to_gene, target_name_to_gene)
    return gene, resolver
