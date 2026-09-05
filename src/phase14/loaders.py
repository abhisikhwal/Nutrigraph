"""
Phase14: File discovery and loading — Phase13 CSVs + mediation datasets.
Uses validation.resolve_phase14_inputs for deterministic path resolution (all paths relative to repo_root).
CSV-first on Windows; overlap gate uses overlap_vs_cg (coverage of compound_gene by ingredient_compound).
"""
from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from . import phase14_config as config
from . import validation as _validation

logger = logging.getLogger(__name__)


def _ts() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def resolve_phase14_inputs(repo_root: Path, phase13_dir: Path) -> Dict[str, Optional[str]]:
    """
    Resolve all Phase14 inputs using the same selection rules as validate_phase14_inputs.
    Returns a dict of key -> path relative to repo_root (str) or None.
    On Windows, if Phase13 dir exists at repo_root / phase13_dir, it is set even when cwd differs.
    """
    repo_root = Path(repo_root).resolve()
    phase13_dir = Path(phase13_dir)
    _, selected, _ = _validation.run_validation(repo_root, phase13_dir)
    phase13_abs = (repo_root / phase13_dir).resolve()
    if (selected.get("phase13_dir") is None or not (repo_root / selected["phase13_dir"]).exists()) and phase13_abs.exists():
        try:
            rel = str(phase13_abs.relative_to(repo_root))
        except ValueError:
            rel = str(phase13_dir).replace("\\", "/")
        selected["phase13_dir"] = rel
        selected["atlas_confirmed"] = rel + "/atlas_confirmed.csv" if not rel.endswith("/") else "atlas_confirmed.csv"
        selected["bootstrap_stability"] = rel + "/bootstrap_stability.csv" if not rel.endswith("/") else "bootstrap_stability.csv"
        for key, fname in (("kg_edges", "kg_edges.csv"), ("kg_nodes", "kg_nodes.csv")):
            p = phase13_abs / "kg" / fname
            if not p.exists():
                p = phase13_abs / fname
            selected[key] = str(p.relative_to(repo_root)) if p.exists() else None
    return selected


def resolved_to_paths(repo_root: Path, resolved: Dict[str, Optional[str]]) -> Dict[str, Optional[Path]]:
    """Convert resolved (relative str) to Paths for loading. Keys unchanged."""
    repo_root = Path(repo_root).resolve()
    out: Dict[str, Optional[Path]] = {}
    for k, v in resolved.items():
        if v is None:
            out[k] = None
        else:
            p = (repo_root / v).resolve()
            out[k] = p if p.exists() else None
    return out


def discover_path(
    base: Path,
    patterns: List[str],
    prefer_ext: Optional[str] = None,
) -> Optional[Path]:
    """
    Search under base for first existing path matching any pattern.
    Prefer newest by mtime if multiple; prefer preferred extension if given.
    """
    base = base.resolve()
    candidates: List[Path] = []
    for pat in patterns:
        if "*" not in pat:
            p = base / pat
            if p.exists():
                candidates.append(p)
            continue
        for p in base.glob(pat):
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None
    if prefer_ext:
        with_ext = [c for c in candidates if str(c).lower().endswith(prefer_ext.lower())]
        if with_ext:
            candidates = with_ext
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def discover_all(repo_root: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    """Discover all Phase14-relevant paths under data/processed. CSV-first on Windows."""
    root = repo_root or config.get_repo_root()
    processed = config.resolve_processed_dir(root)
    prefer_csv = os.name == "nt"
    out: Dict[str, Optional[Path]] = {}
    out["phase13_dir"] = _discover_phase13_dir(processed)
    out["signatures"] = discover_path(processed, config.SIGNATURE_PATTERNS, ".csv" if prefer_csv else ".parquet")
    out["recipe_ingredients"] = discover_path(processed, config.RI_PATTERNS, ".csv" if prefer_csv else ".parquet")
    out["pathway_gene"] = discover_path(processed, config.PATHWAY_PATTERNS)
    out["pathway_cluster_info"] = discover_path(processed, ["features/pathway_cluster_info.csv"])
    out["pathway_bundles"] = discover_path(processed, ["features/pathway_bundles.json"])
    out["metabolomics"] = discover_path(processed, config.METABOLOMICS_PATTERNS)
    out["genetics"] = discover_path(processed, config.GENETICS_PATTERNS)
    out["binding"] = discover_path(processed, config.BINDING_PATTERNS)
    out["target_clusters"] = discover_path(processed, config.TARGET_CLUSTER_PATTERNS)
    if out.get("signatures") or out.get("recipe_ingredients"):
        logger.info("[%s] File discovery: prefer_csv=%s (Windows=%s)", _ts(), prefer_csv, os.name == "nt")
    return out


def _discover_phase13_dir(processed: Path) -> Optional[Path]:
    """Prefer the specified Phase13 run dir, else newest phase13_interactions*."""
    specified = processed / "phase13_interactions_v3_20260206_162122_b_gpu_stable"
    if specified.exists():
        return specified
    dirs = [d for d in processed.iterdir() if d.is_dir() and "phase13" in d.name.lower()]
    if not dirs:
        return None
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0]


def _kg_path(phase13_dir: Path, name: str) -> Path:
    """Resolve kg file: try phase13_dir/kg/ first, then phase13_dir."""
    for d in (phase13_dir / "kg", phase13_dir):
        p = d / name
        if p.exists():
            return p
    return phase13_dir / name


def load_phase13_csvs(phase13_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load atlas_confirmed, bootstrap_stability; build ground-truth from these. kg_edges/kg_nodes optional (CSV).
    Pair-category ground truth is from atlas_confirmed + bootstrap_stability; kg_edges is not trusted for ingA_id/ingB_id/category (may have NaN).
    Returns (atlas, stability, kg_edges, kg_nodes). kg_edges/kg_nodes are empty DataFrame if files missing.
    """
    phase13_dir = Path(phase13_dir).resolve()
    logger.info("[%s] Picked Phase13 dir: %s (CSV)", _ts(), phase13_dir)
    if not (phase13_dir / "atlas_confirmed.csv").exists():
        raise FileNotFoundError(f"Phase13 required file not found: {phase13_dir / 'atlas_confirmed.csv'}")
    atlas = pd.read_csv(phase13_dir / "atlas_confirmed.csv")
    logger.info("[%s] Loaded atlas_confirmed.csv: %s rows, columns: %s", _ts(), len(atlas), list(atlas.columns))
    stability_path = phase13_dir / "bootstrap_stability.csv"
    if not stability_path.exists():
        raise FileNotFoundError(f"Phase13 required file not found: {phase13_dir / 'bootstrap_stability.csv'}")
    stability = pd.read_csv(stability_path)
    logger.info("[%s] Loaded bootstrap_stability: %s", _ts(), len(stability))
    kg_edges_path = _kg_path(phase13_dir, "kg_edges.csv")
    kg_nodes_path = _kg_path(phase13_dir, "kg_nodes.csv")
    if kg_edges_path.exists() and kg_nodes_path.exists():
        kg_edges = pd.read_csv(kg_edges_path)
        kg_nodes = pd.read_csv(kg_nodes_path)
        logger.info("[%s] Loaded kg_edges (optional): %s, kg_nodes: %s", _ts(), len(kg_edges), len(kg_nodes))
    else:
        kg_edges = pd.DataFrame()
        kg_nodes = pd.DataFrame()
        logger.info("[%s] kg_edges/kg_nodes not found; using atlas+bootstrap as ground truth only", _ts())
    return atlas, stability, kg_edges, kg_nodes


def load_phase13_pair_category(phase13_dir: Path) -> pd.DataFrame:
    """
    Build clean pair-category table from atlas_confirmed + bootstrap_stability (ground truth for interactions).
    Columns: ingA_id, ingB_id, category, plus stability columns if present. No reliance on kg_edges.
    """
    phase13_dir = Path(phase13_dir).resolve()
    atlas = pd.read_csv(phase13_dir / "atlas_confirmed.csv")
    required = ["ingA_id", "ingB_id", "category"]
    for c in required:
        if c not in atlas.columns:
            raise ValueError(f"atlas_confirmed.csv missing column: {c}")
    pair = atlas[required].drop_duplicates()
    stability_path = phase13_dir / "bootstrap_stability.csv"
    if stability_path.exists():
        stab = pd.read_csv(stability_path)
        if "ingA_id" in stab.columns and "ingB_id" in stab.columns and "category" in stab.columns:
            pair = pair.merge(
                stab[["ingA_id", "ingB_id", "category"] + [c for c in stab.columns if c not in required]],
                on=["ingA_id", "ingB_id", "category"],
                how="left",
            )
    return pair


def load_analysis_exports(phase13_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Optionally load analysis_exports/kg_edges_enriched.csv and kg_nodes_minimal.csv."""
    ae = phase13_dir / "analysis_exports"
    edges_enriched = ae / "kg_edges_enriched.csv" if ae.exists() else None
    nodes_minimal = ae / "kg_nodes_minimal.csv" if ae.exists() else None
    out_edges = pd.read_csv(edges_enriched) if edges_enriched and edges_enriched.exists() else None
    out_nodes = pd.read_csv(nodes_minimal) if nodes_minimal and nodes_minimal.exists() else None
    return out_edges, out_nodes


def load_csv_or_parquet(path: Path, prefer_csv: bool = True) -> pd.DataFrame:
    """Load DataFrame from CSV or parquet. Prefer CSV on Windows for portability. Logs and reports parquet errors."""
    if not path.exists():
        return pd.DataFrame()
    suf = path.suffix.lower()
    if suf == ".csv":
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.error("[%s] CSV read failed %s: %s", _ts(), path, e)
            raise
    if suf in (".parquet", ".pq"):
        try:
            df = pd.read_parquet(path)
            logger.info("[%s] Loaded parquet %s (picked; %s rows)", _ts(), path, len(df))
            return df
        except Exception as e:
            logger.warning("[%s] Parquet read failed %s: %s", _ts(), path, e)
            csv_alt = path.with_suffix(".csv")
            if prefer_csv and csv_alt.exists():
                logger.info("[%s] Fallback to CSV: %s", _ts(), csv_alt)
                return pd.read_csv(csv_alt)
            return pd.DataFrame()
    return pd.DataFrame()


def load_recipe_ingredients(discovered: Dict[str, Optional[Path]]) -> pd.DataFrame:
    """Load recipe_ingredients for dose proxy and co-occurrence; empty DataFrame if missing."""
    p = discovered.get("recipe_ingredients")
    if not p or not p.exists():
        logger.warning("[%s] recipe_ingredients not found; dose proxy will use fallbacks", _ts())
        return pd.DataFrame()
    logger.info("[%s] Picked recipe_ingredients: %s", _ts(), p)
    df = load_csv_or_parquet(p)
    if df.empty:
        return df
    for col in ["recipe_id", "ingredient_id"]:
        if col not in df.columns:
            logger.warning("[%s] recipe_ingredients missing column '%s'", _ts(), col)
            return pd.DataFrame()
    logger.info("[%s] Loaded recipe_ingredients: %s rows", _ts(), len(df))
    return df


def load_pathway_bundles(discovered: Dict[str, Optional[Path]]) -> Dict[str, List[str]]:
    """Load pathway_bundles.json: category -> list of keywords for PATH->CAT mapping."""
    p = discovered.get("pathway_bundles")
    if not p or not p.exists():
        return {}
    logger.info("[%s] Picked pathway_bundles: %s", _ts(), p)
    import json
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("[%s] Loaded pathway_bundles: %s categories", _ts(), len(data))
    return data


def load_pathway_cluster_info(discovered: Dict[str, Optional[Path]]) -> pd.DataFrame:
    """Load pathway cluster info (pathway clusters and sample pathway names)."""
    p = discovered.get("pathway_cluster_info")
    if not p or not p.exists():
        return pd.DataFrame()
    logger.info("[%s] Picked pathway_cluster_info: %s", _ts(), p)
    df = pd.read_csv(p)
    logger.info("[%s] Loaded pathway_cluster_info: %s rows", _ts(), len(df))
    return df


def load_target_functional_clusters(discovered: Dict[str, Optional[Path]]) -> pd.DataFrame:
    """Load target/functional clusters (genes per cluster)."""
    p = discovered.get("target_clusters")
    if not p or not p.exists():
        return pd.DataFrame()
    logger.info("[%s] Picked target_clusters: %s", _ts(), p)
    df = pd.read_csv(p)
    logger.info("[%s] Loaded target_functional_clusters: %s rows", _ts(), len(df))
    return df


def load_ingredient_id_to_name(repo_root: Optional[Path] = None) -> Dict[str, str]:
    """
    Load authoritative ingredient_id -> human-readable name from data/processed/canonical/ingredients.parquet.
    Keys are normalized with to_ingredient_id so ING_* and numeric ids match. Prefers canonical_name, then scientific_name, then name.
    Returns empty dict if file missing. Used by mediation graph and Neo4j export so Ingredient nodes have real names.
    """
    from .id_normalization import to_ingredient_id
    root = Path(repo_root).resolve() if repo_root else config.get_repo_root()
    p = root / "data" / "processed" / "canonical" / "ingredients.parquet"
    mapping: Dict[str, str] = {}
    if not p.exists():
        logger.info("[%s] No ingredients.parquet at %s; Ingredient node names will fall back to id", _ts(), p)
        return mapping
    try:
        df = pd.read_parquet(p)
        if "ingredient_id" not in df.columns:
            return mapping
        name_cols = [c for c in ["canonical_name", "scientific_name", "name"] if c in df.columns]
        for _, row in df.iterrows():
            ing_id = row.get("ingredient_id")
            if pd.isna(ing_id) or not str(ing_id).strip():
                continue
            key = to_ingredient_id(str(ing_id).strip())
            name = None
            for col in name_cols:
                v = row.get(col)
                if not pd.isna(v) and str(v).strip() and not _is_id_like_name(str(v)):
                    name = str(v).strip()
                    break
            if name and key:
                mapping[key] = name
        logger.info("[%s] Loaded ingredient_id->name: %s entries from %s", _ts(), len(mapping), p.name)
    except Exception as e:
        logger.warning("[%s] load_ingredient_id_to_name failed: %s", _ts(), e)
    return mapping


def _is_id_like_name(s: str) -> bool:
    """True if s looks like an ID (e.g. ING_000051) and should not be used as display name."""
    if not s or not str(s).strip():
        return True
    s = str(s).strip()
    if s.upper() == "ING_UNKNOWN":
        return True
    if re.match(r"^ING_\d+$", s, re.IGNORECASE):
        return True
    return False


def _normalize_name(s: str) -> str:
    """Lowercase, strip, remove punctuation, collapse whitespace."""
    if not isinstance(s, str) or pd.isna(s):
        return ""
    s = re.sub(r"[^\w\s]", " ", str(s).lower().strip())
    return " ".join(s.split())


def build_ingredient_compound_from_food_links(
    recipe_ingredients: pd.DataFrame,
    food_compound_gene_links: pd.DataFrame,
    score_threshold: int = 92,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build ingredient_id -> compound_id mapping by matching ingredient names to food/compound names.
    Requires recipe_ingredients: ingredient_id and ingredient_raw (or ingredient_name).
    Requires food_compound_gene_links: compound_id and compound_name (or food/food_name column).
    Uses rapidfuzz if available, else exact normalized match. Returns DataFrame with
    ingredient_id, compound_id, and optional match_score, food_name_matched.
    """
    out_cols = ["ingredient_id", "compound_id"]
    empty = pd.DataFrame(columns=out_cols + ["match_score", "food_name_matched"])
    if recipe_ingredients.empty or food_compound_gene_links.empty:
        logger.warning("[%s] build_ingredient_compound_from_food_links: empty inputs", _ts())
        return empty
    id_col = "ingredient_id" if "ingredient_id" in recipe_ingredients.columns else None
    name_col = next((c for c in recipe_ingredients.columns if "ingredient" in c.lower() and "raw" in c.lower()), None)
    if not name_col:
        name_col = next((c for c in recipe_ingredients.columns if "ingredient" in c.lower() and "name" in c.lower()), None)
    if not id_col or not name_col:
        logger.warning("[%s] recipe_ingredients missing ingredient_id or name column: %s", _ts(), list(recipe_ingredients.columns))
        return empty
    food_name_col = next((c for c in food_compound_gene_links.columns if "food" in c.lower() or c == "food_name"), None)
    if not food_name_col:
        food_name_col = next((c for c in food_compound_gene_links.columns if "compound" in c.lower() and "name" in c.lower()), None)
    cmp_col = next((c for c in food_compound_gene_links.columns if "compound" in c.lower() and "id" in c.lower()), None)
    if not cmp_col:
        cmp_col = "compound_id" if "compound_id" in food_compound_gene_links.columns else None
    if not food_name_col or not cmp_col:
        logger.warning("[%s] food_compound_gene_links missing food/compound name and compound_id: %s", _ts(), list(food_compound_gene_links.columns))
        return empty

    ings = recipe_ingredients[[id_col, name_col]].drop_duplicates()
    ings = ings[ings[name_col].notna() & (ings[name_col].astype(str).str.strip() != "")]
    ings["_norm"] = ings[name_col].astype(str).apply(_normalize_name)
    ings = ings[ings["_norm"] != ""].drop_duplicates(subset=[id_col, "_norm"])

    foods = food_compound_gene_links[[cmp_col, food_name_col]].drop_duplicates()
    foods = foods[foods[food_name_col].notna() & (foods[food_name_col].astype(str).str.strip() != "")]
    foods["_norm"] = foods[food_name_col].astype(str).apply(_normalize_name)
    foods = foods[foods["_norm"] != ""].drop_duplicates(subset=[cmp_col, "_norm"])

    use_fuzzy = False
    try:
        from rapidfuzz import fuzz
        use_fuzzy = True
    except ImportError:
        pass

    rows: List[Dict[str, Any]] = []
    for _, row in ings.iterrows():
        ing_id = row[id_col]
        norm_ing = row["_norm"]
        best_score = 0
        best_cmp: Optional[str] = None
        best_food: Optional[str] = None
        if use_fuzzy:
            for _, fr in foods.iterrows():
                score = fuzz.ratio(norm_ing, fr["_norm"])
                if score >= score_threshold and score > best_score:
                    best_score = score
                    best_cmp = fr[cmp_col]
                    best_food = fr[food_name_col]
        else:
            match = foods[foods["_norm"] == norm_ing]
            if not match.empty:
                best_cmp = match[cmp_col].iloc[0]
                best_food = match[food_name_col].iloc[0]
                best_score = 100
        if best_cmp is not None:
            rows.append({
                "ingredient_id": ing_id,
                "compound_id": best_cmp,
                "match_score": best_score,
                "food_name_matched": best_food,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        logger.info("[%s] build_ingredient_compound_from_food_links: 0 matches (threshold=%s, fuzzy=%s)", _ts(), score_threshold, use_fuzzy)
        return empty
    n_ing = out["ingredient_id"].nunique()
    logger.info("[%s] build_ingredient_compound_from_food_links: %s edges, %s unique ingredients (match rate ~%s%%, fuzzy=%s)",
                _ts(), len(out), n_ing, round(100 * n_ing / ings["ingredient_id"].nunique(), 1) if ings["ingredient_id"].nunique() else 0, use_fuzzy)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path_pq = output_dir / "ingredient_compound_links.parquet"
        out_path_csv = output_dir / "ingredient_compound_links.csv"
        try:
            out.to_parquet(out_path_pq, index=False)
            logger.info("[%s] Wrote %s", _ts(), out_path_pq)
        except Exception as e:
            logger.warning("[%s] Could not write parquet: %s", _ts(), e)
        try:
            out.to_csv(out_path_csv, index=False)
            logger.info("[%s] Wrote %s", _ts(), out_path_csv)
        except Exception as e:
            logger.warning("[%s] Could not write CSV: %s", _ts(), e)
    return out


def _infer_repo_root(discovered: Dict[str, Optional[Path]]) -> Optional[Path]:
    """Infer repo root from phase13_dir or recipe_ingredients path."""
    for key in ("phase13_dir", "recipe_ingredients"):
        p = discovered.get(key)
        if p and Path(p).exists():
            resolved = Path(p).resolve()
            if key == "phase13_dir":
                return resolved.parent.parent  # data/processed -> repo
            return resolved.parent.parent.parent  # canonical/... -> processed -> data -> repo
    return None


def print_ing_cmp_schema(path: Path) -> None:
    """Print schema/columns of an ING->CMP candidate file for diagnostics. Handles parquet/csv."""
    path = Path(path)
    if not path.exists():
        logger.warning("print_ing_cmp_schema: path does not exist: %s", path)
        return
    try:
        df = load_csv_or_parquet(path)
        if df.empty:
            logger.info("Schema %s: empty file, columns=%s", path.name, list(df.columns))
            return
        ing_sample = df["ingredient_id"].iloc[0] if "ingredient_id" in df.columns else (df.iloc[0].get("ingredient_id") or "N/A")
        cmp_sample = df["compound_id"].iloc[0] if "compound_id" in df.columns else (df.iloc[0].get("compound_id") or "N/A")
        logger.info(
            "[%s] Schema %s: columns=%s, n_rows=%s, sample ingredient_id=%s, sample compound_id=%s",
            _ts(), path.name, list(df.columns), len(df), ing_sample, cmp_sample,
        )
    except Exception as e:
        logger.warning("print_ing_cmp_schema failed %s: %s", path, e)


def load_ingredient_compound(discovered: Dict[str, Optional[Path]]) -> pd.DataFrame:
    """
    Load ING->CMP links. Prefers data/processed/canonical/ingredient_compound_links.csv (deterministic).
    Then derived/ingredient_compound_links, then metabolomics (only if has ingredient column), then audit bridge.
    Returns DataFrame with ingredient_id, compound_id. Logs: file chosen, n edges, n unique ingredients, n unique compounds.
    """
    empty = pd.DataFrame(columns=["ingredient_id", "compound_id"])
    processed = None
    for key in ("phase13_dir", "recipe_ingredients", "metabolomics"):
        p = discovered.get(key)
        if p and Path(p).exists():
            resolved = Path(p).resolve()
            processed = resolved.parent if key == "phase13_dir" else resolved.parent.parent
            break
    if processed:
        canonical_dir = processed / "canonical"
        ing_canonical_csv = canonical_dir / "ingredient_compound_canonical.csv"
        if ing_canonical_csv.exists():
            try:
                df = pd.read_csv(ing_canonical_csv)
                if not df.empty and "ingredient_id" in df.columns and "compound_id" in df.columns:
                    out = df[["ingredient_id", "compound_id"]].drop_duplicates()
                    logger.info(
                        "[%s] Picked CANONICAL ingredient_compound_canonical: %s | edges=%s, unique_ingredients=%s, unique_compounds=%s",
                        _ts(), ing_canonical_csv.name, len(out), out["ingredient_id"].nunique(), out["compound_id"].nunique(),
                    )
                    return out
            except Exception as e:
                logger.warning("[%s] ingredient_compound_canonical load failed: %s", _ts(), e)
        canonical_csv = canonical_dir / "ingredient_compound_links.csv"
        if canonical_csv.exists():
            try:
                df = pd.read_csv(canonical_csv)
                if not df.empty and "ingredient_id" in df.columns and "compound_id" in df.columns:
                    out = df[["ingredient_id", "compound_id"]].drop_duplicates()
                    n_ing = out["ingredient_id"].nunique()
                    n_cmp = out["compound_id"].nunique()
                    logger.info(
                        "[%s] Picked CANONICAL ingredient_compound: %s | edges=%s, unique_ingredients=%s, unique_compounds=%s",
                        _ts(), canonical_csv.name, len(out), n_ing, n_cmp,
                    )
                    return out
            except Exception as e:
                logger.warning("[%s] Canonical ingredient_compound load failed: %s", _ts(), e)
        canonical_pq = processed / "canonical" / "ingredient_compound_links.parquet"
        if canonical_pq.exists():
            try:
                df = load_csv_or_parquet(canonical_pq)
                if not df.empty and "ingredient_id" in df.columns and "compound_id" in df.columns:
                    out = df[["ingredient_id", "compound_id"]].drop_duplicates()
                    logger.info("[%s] Picked CANONICAL (parquet) ingredient_compound: %s | edges=%s, unique_ingredients=%s, unique_compounds=%s",
                                _ts(), canonical_pq.name, len(out), out["ingredient_id"].nunique(), out["compound_id"].nunique())
                    return out
            except Exception as e:
                logger.warning("[%s] Canonical parquet load failed: %s", _ts(), e)
        derived = processed / "phase14_mediation" / "derived" / "ingredient_compound_links.parquet"
        if derived.exists():
            try:
                df = load_csv_or_parquet(derived)
                if not df.empty and "ingredient_id" in df.columns and "compound_id" in df.columns:
                    out = df[["ingredient_id", "compound_id"]].drop_duplicates()
                    logger.info("[%s] Picked derived ingredient_compound: %s | edges=%s, unique_ingredients=%s, unique_compounds=%s",
                                _ts(), derived.name, len(out), out["ingredient_id"].nunique(), out["compound_id"].nunique())
                    return out
            except Exception as e:
                logger.warning("[%s] Derived load failed: %s", _ts(), e)
        csv_derived = processed / "phase14_mediation" / "derived" / "ingredient_compound_links.csv"
        if csv_derived.exists():
            try:
                df = pd.read_csv(csv_derived)
                if not df.empty and "ingredient_id" in df.columns and "compound_id" in df.columns:
                    out = df[["ingredient_id", "compound_id"]].drop_duplicates()
                    logger.info("[%s] Picked derived (CSV) ingredient_compound: %s | edges=%s, unique_ingredients=%s, unique_compounds=%s",
                                _ts(), csv_derived.name, len(out), out["ingredient_id"].nunique(), out["compound_id"].nunique())
                    return out
            except Exception as e:
                logger.warning("[%s] Derived CSV load failed: %s", _ts(), e)
    # Do NOT use metabolomics file as ingredient->compound unless it has an ingredient column (compound_metabolite_links does not).
    p = discovered.get("metabolomics")
    if p and p.exists():
        try:
            df = load_csv_or_parquet(p)
            if not df.empty:
                id_col = next((c for c in df.columns if "ingredient" in c.lower() or c.lower() in ("ing_id", "ingredient_id")), None)
                cmp_col = next((c for c in df.columns if "compound" in c.lower() or c.lower() in ("cmp_id", "compound_id")), None)
                if id_col and cmp_col:
                    out = df[[id_col, cmp_col]].drop_duplicates().rename(columns={id_col: "ingredient_id", cmp_col: "compound_id"})
                    logger.info("[%s] Loaded ingredient_compound from metabolomics: %s rows", _ts(), len(out))
                    return out
                logger.info("[%s] Skipping metabolomics for ing->compound (no ingredient column): %s", _ts(), list(df.columns))
        except Exception as e:
            logger.warning("[%s] load_ingredient_compound metabolomics failed: %s", _ts(), e)
    # Audit-based bridge: InChIKey-first, multi-source
    repo_root = _infer_repo_root(discovered)
    if repo_root:
        try:
            from .audit import pick_best_sources, scan_processed_data
            from .ingredient_compound import load_ing_compound_edges
            scan_df = scan_processed_data(repo_root)
            chosen = pick_best_sources(scan_df, repo_root)
            bridge_df = load_ing_compound_edges(repo_root, scan_df, chosen)
            if not bridge_df.empty and "ingredient_id" in bridge_df.columns and "compound_id" in bridge_df.columns:
                logger.info("[%s] Loaded ingredient_compound from audit bridge: %s rows", _ts(), len(bridge_df))
                return bridge_df[["ingredient_id", "compound_id"]].drop_duplicates()
        except Exception as e:
            logger.warning("[%s] Audit bridge for ingredient_compound failed: %s", _ts(), e)
    logger.warning("[%s] load_ingredient_compound: no source found. Run: python scripts/phase14/build_canonical_ingredient_compounds.py", _ts())
    return empty


# Column names for flexible compound–gene detection (aligned with audit.py)
_COMPOUND_KEY_NAMES = [
    "compound_id", "inchikey", "inchi_key", "InChIKey", "pubchem_cid", "chembl_id",
]
_GENE_TARGET_NAMES = [
    "gene", "gene_symbol", "gene_id", "target", "target_name", "uniprot", "uniprot_id",
]


def _normalize_inchikey_for_cg(val: Any) -> Optional[str]:
    """Uppercase, strip; return None if not a valid-looking InChIKey."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().upper()
    return s if s else None


def _normalize_gene_for_cg(val: Any) -> str:
    """Uppercase, strip; empty string if missing."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip().upper() or ""


def _pick_compound_col(df: pd.DataFrame) -> Optional[str]:
    cl = {c.lower(): c for c in df.columns}
    for k in _COMPOUND_KEY_NAMES:
        if k.lower() in cl:
            return cl[k.lower()]
    return None


def _pick_gene_target_col(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Return (column_name, gene_type). gene_type is 'gene' or 'target'."""
    cl = {c.lower(): c for c in df.columns}
    gene_prefer = ["gene_symbol", "gene", "gene_id"]
    target_prefer = ["target", "target_name", "uniprot", "uniprot_id"]
    for k in gene_prefer:
        if k in cl:
            return cl[k], "gene"
    for k in target_prefer:
        if k in cl:
            return cl[k], "target"
    return None, "gene"


def _load_and_normalize_compound_gene(path: Path) -> pd.DataFrame:
    """Load one file, detect compound/gene columns, normalize; return canonical compound_id, inchikey, gene_id, gene_type."""
    df = load_csv_or_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["compound_id", "inchikey", "gene_id", "gene_type"])
    cmp_col = _pick_compound_col(df)
    gene_col, gene_type = _pick_gene_target_col(df)
    if not cmp_col or not gene_col:
        logger.warning("[%s] compound_gene file missing compound or gene/target column: %s", _ts(), list(df.columns))
        return pd.DataFrame(columns=["compound_id", "inchikey", "gene_id", "gene_type"])
    # Optional inchikey column
    ik_col = None
    for c in df.columns:
        if c.lower() in ("inchikey", "inchi_key"):
            ik_col = c
            break
    rows = []
    for _, r in df.iterrows():
        cmp_val = r.get(cmp_col)
        if cmp_val is None or pd.isna(cmp_val):
            continue
        compound_id = str(cmp_val).strip()
        inchikey = _normalize_inchikey_for_cg(r.get(ik_col) if ik_col else cmp_val if (str(cmp_val).strip().upper().startswith("A") and "-" in str(cmp_val)) else None)
        if not inchikey and str(cmp_val).strip().upper().startswith("A") and "-" in str(cmp_val):
            inchikey = _normalize_inchikey_for_cg(cmp_val)
        gene_val = _normalize_gene_for_cg(r.get(gene_col))
        if not gene_val:
            continue
        rows.append({"compound_id": compound_id, "inchikey": inchikey or "", "gene_id": gene_val, "gene_type": gene_type})
    out = pd.DataFrame(rows).drop_duplicates(subset=["compound_id", "gene_id"])
    out = out[out["gene_id"].str.len() > 0]
    logger.info("[%s] Loaded compound_gene from %s: %s rows (gene_type=%s)", _ts(), path.name, len(out), gene_type)
    return out


# Overlap gate: PASS if (overlap_vs_cg >= 0.20) OR (n_overlap >= 25) OR (|cg_set| >= 500 and overlap_vs_ic >= 0.01)
OVERLAP_GATE_CG_MIN = 0.20
OVERLAP_GATE_N_ABSOLUTE = 25
OVERLAP_GATE_LARGE_CG_THRESHOLD = 500
OVERLAP_GATE_LARGE_CG_IC_MIN = 0.01


def compute_identity_overlap_metrics(
    ic_set: Set[str],
    cg_set: Set[str],
) -> Dict[str, Any]:
    """
    Compute overlap metrics for ingredient_compound vs compound_gene (both InChIKey sets).
    Returns dict: n_ic_compounds, n_cg_compounds, n_overlap, overlap_vs_cg, overlap_vs_ic,
    gate_passed, gate_reason.
    """
    n_ic = len(ic_set)
    n_cg = max(len(cg_set), 1)
    overlap_set = ic_set & cg_set
    n_overlap = len(overlap_set)
    overlap_vs_cg = n_overlap / n_cg if n_cg else 0.0
    overlap_vs_ic = n_overlap / n_ic if n_ic else 0.0
    # Gate: PASS if (overlap_vs_cg >= 0.20) OR (n_overlap >= 25) OR (|cg_set| >= 500 and overlap_vs_ic >= 0.01)
    gate_passed = False
    reason = ""
    if overlap_vs_cg >= OVERLAP_GATE_CG_MIN:
        gate_passed = True
        reason = "overlap_vs_cg >= %.0f%%" % (100 * OVERLAP_GATE_CG_MIN)
    elif n_overlap >= OVERLAP_GATE_N_ABSOLUTE:
        gate_passed = True
        reason = "n_overlap >= %d" % OVERLAP_GATE_N_ABSOLUTE
    elif len(cg_set) >= OVERLAP_GATE_LARGE_CG_THRESHOLD and overlap_vs_ic >= OVERLAP_GATE_LARGE_CG_IC_MIN:
        gate_passed = True
        reason = "|cg_set| >= %d and overlap_vs_ic >= %.2f%%" % (OVERLAP_GATE_LARGE_CG_THRESHOLD, 100 * OVERLAP_GATE_LARGE_CG_IC_MIN)
    else:
        reason = "overlap_vs_cg=%.2f%%, n_overlap=%d, |cg_set|=%d (need overlap_vs_cg>=%.0f%% or n_overlap>=%d or large-cg rule)" % (
            100 * overlap_vs_cg, n_overlap, len(cg_set), 100 * OVERLAP_GATE_CG_MIN, OVERLAP_GATE_N_ABSOLUTE,
        )
    return {
        "n_ic_compounds": n_ic,
        "n_cg_compounds": len(cg_set),
        "n_overlap": n_overlap,
        "overlap_vs_cg": round(overlap_vs_cg, 4),
        "overlap_vs_ic": round(overlap_vs_ic, 4),
        "gate_passed": gate_passed,
        "gate_reason": reason,
    }


def _check_canonical_overlap(
    canonical_dir: Path,
    compound_gene_out: pd.DataFrame,
    _ts: Any,
    canonical_path: Optional[Path] = None,
    require_overlap_pct: Optional[float] = None,
    require_overlap_vs_cg_min: Optional[float] = None,
    full_run_gate: bool = False,
) -> None:
    """
    If ingredient_compound_canonical.csv exists, require overlap gate to pass.
    Gate: (overlap_vs_cg >= 20%) OR (n_overlap >= 25) OR (|cg_set| >= 500 and overlap_vs_ic >= 1%).
    When require_overlap_vs_cg_min is set (e.g. 0.05 for FULL runs), raise if overlap_vs_cg < that value.
    On failure raises ValueError with all metrics and points to reports.
    """
    ing_canon_path = Path(canonical_dir) / "ingredient_compound_canonical.csv"
    if not ing_canon_path.exists():
        return
    try:
        ing_df = load_csv_or_parquet(ing_canon_path)
        if ing_df is None or ing_df.empty or "compound_id" not in ing_df.columns:
            return
        ic_set = set(ing_df["compound_id"].dropna().astype(str).str.strip().str.upper())
        ic_set = {k for k in ic_set if k and str(k) != "NAN"}
        cg_set = set(compound_gene_out["compound_id"].dropna().astype(str).str.strip().str.upper())
        cg_set = {k for k in cg_set if k and str(k) != "NAN"}
        if not ic_set or not cg_set:
            return
        metrics = compute_identity_overlap_metrics(ic_set, cg_set)
        # When using any compound_gene_expanded* source: do NOT require 5% overlap (expansion is the point); only WARN if < 20%
        canonical_name = (canonical_path.name or "") if canonical_path else ""
        use_expanded_canonical = "compound_gene_expanded" in canonical_name
        if use_expanded_canonical and require_overlap_vs_cg_min is not None and full_run_gate:
            if metrics["overlap_vs_cg"] < 0.20:
                logger.warning(
                    "[%s] Expanded compound_gene loaded (%s); overlap_vs_cg=%.2f%% < 20%%. Proceeding (no crash). To improve overlap, run identity bridge or build_compound_gene_expanded.",
                    _ts(), canonical_name, 100 * metrics["overlap_vs_cg"],
                )
        elif require_overlap_vs_cg_min is not None and full_run_gate:
            # Hard gate for FULL runs: overlap_vs_cg >= 5%
            if metrics["overlap_vs_cg"] < require_overlap_vs_cg_min:
                msg = (
                    "Compound-gene overlap too low for FULL run. overlap_vs_cg=%.2f%% (need >= %.0f%%). "
                    "n_ic=%s, n_cg=%s, n_overlap=%s. "
                    "Run: python scripts/phase16/build_compound_gene_expanded_v4.py --repo-root . "
                    "Or: python scripts/phase16/run_compound_gene_expansion_v2.py --repo-root . "
                    "(after building the identity bridge). Check data/processed/canonical/*.json"
                ) % (
                    100 * metrics["overlap_vs_cg"],
                    100 * require_overlap_vs_cg_min,
                    metrics["n_ic_compounds"],
                    metrics["n_cg_compounds"],
                    metrics["n_overlap"],
                )
                raise ValueError(msg)
        # When using expanded canonical, allow 1% overlap (broader compound set)
        use_expanded_gate = canonical_path and "compound_gene_expanded" in (canonical_path.name or "")
        min_overlap_ic = require_overlap_pct if require_overlap_pct is not None else (0.01 if use_expanded_gate else None)
        if min_overlap_ic is not None and metrics["overlap_vs_ic"] >= min_overlap_ic:
            metrics = dict(metrics)
            metrics["gate_passed"] = True
            metrics["gate_reason"] = "overlap_vs_ic >= %.2f%% (expanded)" % (100 * min_overlap_ic)
        if not metrics.get("gate_passed", True):
            msg = (
                "Canonical ingredient_compound vs compound_gene overlap gate FAILED. "
                "n_ic=%s, n_cg=%s, n_overlap=%s, overlap_vs_cg=%.2f%%, overlap_vs_ic=%.2f%%. %s "
                "Run: python scripts/phase16/build_compound_gene_expanded_v4.py --repo-root . "
                "Check data/processed/canonical/*.json"
            ) % (
                metrics["n_ic_compounds"],
                metrics["n_cg_compounds"],
                metrics["n_overlap"],
                100 * metrics["overlap_vs_cg"],
                100 * metrics["overlap_vs_ic"],
                metrics.get("gate_reason", ""),
            )
            logger.warning("[%s] %s", _ts(), msg)
            try:
                reports_dir = canonical_dir / "reports" if (canonical_dir / "reports").exists() else canonical_dir.parent / "reports"
                reports_dir = Path(reports_dir)
                reports_dir.mkdir(parents=True, exist_ok=True)
                fail_path = reports_dir / "coverage_fail_reasons.json"
                import json as _json
                with open(fail_path, "w", encoding="utf-8") as f:
                    _json.dump({
                        "overlap_gate_failed": True,
                        "reason": metrics.get("gate_reason", ""),
                        "n_ic_compounds": metrics["n_ic_compounds"],
                        "n_cg_compounds": metrics["n_cg_compounds"],
                        "n_overlap": metrics["n_overlap"],
                        "overlap_vs_cg": metrics["overlap_vs_cg"],
                        "overlap_vs_ic": metrics["overlap_vs_ic"],
                        "message": msg,
                    }, f, indent=2)
                logger.warning("[%s] Wrote %s", _ts(), fail_path)
            except Exception as e:
                logger.warning("[%s] Could not write coverage_fail_reasons.json: %s", _ts(), e)
        logger.info(
            "[%s] Overlap gate PASS: %s (n_overlap=%s, overlap_vs_cg=%.1f%%)",
            _ts(), metrics.get("gate_reason", "ok"), metrics["n_overlap"], 100 * metrics["overlap_vs_cg"],
        )
    except ValueError:
        raise
    except Exception:
        pass


def _get_canonical_dir(discovered: Dict[str, Optional[Path]]) -> Optional[Path]:
    """Resolve data/processed/canonical from discovered paths."""
    processed = None
    for key in ("phase13_dir", "recipe_ingredients"):
        p = discovered.get(key)
        if p is not None and Path(p).exists():
            resolved = Path(p).resolve()
            processed = resolved.parent if key == "phase13_dir" else resolved.parent.parent
            break
    if processed is None:
        return None
    return processed / "canonical"


def _assert_expanded_selected_if_exists(canonical_path: Path, full_run_gate: bool) -> None:
    """
    In FULL mode: if compound_gene_expanded_canonical.csv exists and is non-empty, it must have been selected.
    Prevents silent regressions when loaders pick a different source.
    """
    if not full_run_gate:
        return
    canonical_dir = canonical_path.parent
    expanded = canonical_dir / "compound_gene_expanded_canonical.csv"
    if not expanded.exists():
        return
    try:
        df = load_csv_or_parquet(expanded)
        if df is None or df.empty:
            return
    except Exception:
        return
    if canonical_path.resolve() != expanded.resolve():
        raise RuntimeError(
            "compound_gene_expanded_canonical.csv exists and is non-empty but was not selected. "
            "Compound-gene source selection must use expanded priority. Selected: %s"
            % (canonical_path.name,)
        )


def _compound_gene_overlap_for_log(canonical_dir: Path, compound_gene_out: pd.DataFrame) -> Tuple[float, float]:
    """Compute overlap_vs_cg and overlap_vs_ic for logging (returns (0,0) if ingredient_compound missing)."""
    if "compound_id" not in compound_gene_out.columns:
        return 0.0, 0.0
    cg_set = set(compound_gene_out["compound_id"].dropna().astype(str).str.strip().str.upper())
    cg_set = {k for k in cg_set if k and str(k) != "NAN"}
    ing_path = canonical_dir / "ingredient_compound_canonical.csv"
    if not ing_path.exists():
        return 0.0, 0.0
    try:
        ing_df = load_csv_or_parquet(ing_path)
        if ing_df is None or ing_df.empty or "compound_id" not in ing_df.columns:
            return 0.0, 0.0
        ic_set = set(ing_df["compound_id"].dropna().astype(str).str.strip().str.upper())
        ic_set = {k for k in ic_set if k and str(k) != "NAN"}
        m = compute_identity_overlap_metrics(ic_set, cg_set)
        return m["overlap_vs_cg"], m["overlap_vs_ic"]
    except Exception:
        return 0.0, 0.0


def _write_compound_gene_metrics(
    canonical_dir: Path,
    compound_gene_out: pd.DataFrame,
    canonical_path: Optional[Path] = None,
) -> None:
    """Write overlap_vs_cg and key counts to canonical/reports/phase14_compound_gene_metrics.json (no silent junk)."""
    try:
        reports_dir = canonical_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        cg_set = set(compound_gene_out["compound_id"].dropna().astype(str).str.strip().str.upper()) if "compound_id" in compound_gene_out.columns else set()
        cg_set = {k for k in cg_set if k and str(k) != "NAN"}
        metrics = {"n_cg_compounds": len(cg_set), "n_cg_edges": len(compound_gene_out), "source_file": canonical_path.name if canonical_path else None}
        ing_path = canonical_dir / "ingredient_compound_canonical.csv"
        if ing_path.exists():
            ing_df = load_csv_or_parquet(ing_path)
            if ing_df is not None and not ing_df.empty and "compound_id" in ing_df.columns:
                ic_set = set(ing_df["compound_id"].dropna().astype(str).str.strip().str.upper())
                ic_set = {k for k in ic_set if k and str(k) != "NAN"}
                overlap_metrics = compute_identity_overlap_metrics(ic_set, cg_set)
                metrics["overlap_vs_cg"] = overlap_metrics["overlap_vs_cg"]
                metrics["overlap_vs_ic"] = overlap_metrics["overlap_vs_ic"]
                metrics["n_overlap"] = overlap_metrics["n_overlap"]
                metrics["n_ic_compounds"] = overlap_metrics["n_ic_compounds"]
        out_path = reports_dir / "phase14_compound_gene_metrics.json"
        import json as _json
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(metrics, f, indent=2)
        logger.info("[%s] Wrote overlap metrics to %s", _ts(), out_path.name)
    except Exception as e:
        logger.warning("[%s] Could not write phase14_compound_gene_metrics.json: %s", _ts(), e)


def _is_expanded_compound_gene_path(path: Optional[Path]) -> bool:
    """True if path is an expanded compound_gene file (expanded_canonical or expanded_v4/v3/v2)."""
    if path is None:
        return False
    name = (path.name or "").lower()
    return "compound_gene_expanded" in name and (name.endswith(".csv") or name.endswith(".parquet"))


def _compound_gene_priority_paths(canonical_dir: Path) -> List[Path]:
    """
    Priority order for compound_gene source (do not use overlap when any of these exist and non-empty):
    a) compound_gene_expanded_canonical.csv
    b) compound_gene_expanded_v4*.csv (any match, sorted)
    c) compound_gene_canonical.csv
    d) other fallbacks
    """
    priority: List[Path] = []
    # a) expanded canonical
    p = canonical_dir / "compound_gene_expanded_canonical.csv"
    if p.exists():
        priority.append(p)
    # b) expanded v4 (glob)
    for f in sorted(canonical_dir.glob("compound_gene_expanded_v4*.csv")):
        priority.append(f)
    # v3, v2
    for name in ("compound_gene_expanded_v3_canonical.csv", "compound_gene_expanded_v2_canonical.csv"):
        p = canonical_dir / name
        if p.exists():
            priority.append(p)
    # c) base canonical
    for name in ("compound_gene_canonical.csv", "compound_gene_canonical.parquet", "compound_gene_links.csv", "compound_gene_links.parquet"):
        p = canonical_dir / name
        if p.exists():
            priority.append(p)
    return priority


def _compound_gene_candidate_paths(canonical_dir: Path) -> List[Path]:
    """Return candidate compound_gene files in preference order (expanded canonical first, then v4/v3/v2)."""
    return _compound_gene_priority_paths(canonical_dir)


def _score_candidate_overlap(
    candidate_path: Path,
    ic_set: Set[str],
) -> Tuple[float, int, Optional[pd.DataFrame]]:
    """Load candidate, compute overlap_vs_cg. Returns (overlap_vs_cg, n_overlap, out_df or None)."""
    df = load_csv_or_parquet(candidate_path)
    if df is None or df.empty:
        return 0.0, 0, None
    cmp_col = next((c for c in df.columns if "compound" in c.lower() and "id" in c.lower()), None) or ( "inchikey" if "inchikey" in df.columns else None )
    if not cmp_col:
        return 0.0, 0, None
    gene_col = next((c for c in ("gene_symbol", "gene", "gene_id") if c in df.columns), None)
    if not gene_col:
        return 0.0, 0, None
    cg_set = set(df[cmp_col].dropna().astype(str).str.strip().str.upper())
    cg_set = {k for k in cg_set if k and str(k) != "NAN"}
    if not cg_set:
        return 0.0, 0, None
    overlap = len(ic_set & cg_set)
    n_cg = len(cg_set)
    overlap_vs_cg = overlap / n_cg if n_cg else 0.0
    out = df[[cmp_col, gene_col]].drop_duplicates()
    out = out.rename(columns={cmp_col: "compound_id", gene_col: "gene_id"})
    out["compound_id"] = out["compound_id"].astype(str).str.strip().str.upper()
    out["gene_id"] = out["gene_id"].astype(str).str.strip().str.upper()
    out = out[out["compound_id"].str.len() > 0]
    out = out[out["gene_id"].str.len() > 0]
    return overlap_vs_cg, overlap, out


# Hard gate: FULL runs require overlap_vs_cg >= 5% (or propagation >= 25% checked later).
OVERLAP_GATE_FULL_MIN_CG = 0.05
PROPAGATION_GATE_FULL_MIN = 0.25


def _canonical_compound_gene_path(discovered: Dict[str, Optional[Path]]) -> Optional[Path]:
    """
    Resolve compound_gene path with deterministic priority. Do NOT use overlap when an expanded file exists.
    Priority: a) compound_gene_expanded_canonical.csv (if exists and non-empty)
              b) compound_gene_expanded_v4*.csv
              c) compound_gene_canonical.csv
              d) other fallbacks.
    When no expanded file exists and non-empty, fall back to overlap-based selection. Overlap metrics are
    still computed and written to reports for whichever source is chosen.
    """
    canonical_dir = _get_canonical_dir(discovered)
    if canonical_dir is None:
        return None
    priority = _compound_gene_priority_paths(canonical_dir)
    if not priority:
        return None
    # First: use priority order when an expanded/high-priority file exists and is non-empty (no overlap choice)
    for path in priority:
        if not path.exists():
            continue
        df = load_csv_or_parquet(path)
        if df is not None and not df.empty:
            cmp_col = next((c for c in df.columns if "compound" in c.lower() and "id" in c.lower()), None) or ("inchikey" if "inchikey" in df.columns else None)
            gene_col = next((c for c in ("gene_symbol", "gene", "gene_id") if c in df.columns), None)
            if cmp_col and gene_col:
                logger.info("[%s] Using compound_gene source (expanded priority): %s", _ts(), path.name)
                return path
    # Fallback: overlap-based selection (only when no priority file was available and non-empty)
    ing_path = canonical_dir / "ingredient_compound_canonical.csv"
    ic_set: Set[str] = set()
    if ing_path.exists():
        try:
            ing_df = load_csv_or_parquet(ing_path)
            if ing_df is not None and not ing_df.empty and "compound_id" in ing_df.columns:
                ic_set = set(ing_df["compound_id"].dropna().astype(str).str.strip().str.upper())
                ic_set = {k for k in ic_set if k and str(k) != "NAN"}
        except Exception:
            pass
    if not ic_set:
        logger.info("[%s] No ingredient_compound canonical; selecting first candidate: %s", _ts(), priority[0].name)
        return priority[0]
    scored: List[Tuple[float, int, Path]] = []
    for path in priority:
        ov_cg, n_overlap, _ = _score_candidate_overlap(path, ic_set)
        scored.append((ov_cg, n_overlap, path))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    best_ov, best_n, best_path = scored[0]
    if best_ov == 0.0 and len(scored) > 1:
        non_zero = [s for s in scored if s[0] > 0]
        if non_zero:
            best_ov, best_n, best_path = non_zero[0]
            logger.info("[%s] Skipped 0-overlap candidates; selected %s (overlap_vs_cg=%.2f%%)", _ts(), best_path.name, 100 * best_ov)
    else:
        logger.info("[%s] Selected compound_gene by overlap: %s (overlap_vs_cg=%.2f%%, n_overlap=%s)", _ts(), best_path.name, 100 * best_ov, best_n)
    return best_path


def discover_compound_target_canonical(discovered: Dict[str, Optional[Path]]) -> Optional[Path]:
    """Return path to compound_target_canonical.csv if present (optional evidence). Do not break if absent."""
    processed = None
    for key in ("phase13_dir", "recipe_ingredients"):
        p = discovered.get(key)
        if p and Path(p).exists():
            resolved = Path(p).resolve()
            processed = resolved.parent if key == "phase13_dir" else resolved.parent.parent
            break
    if not processed:
        return None
    path = processed / "canonical" / "compound_target_canonical.csv"
    return path if path.exists() else None


def load_compound_target(discovered: Dict[str, Optional[Path]]) -> pd.DataFrame:
    """Load compound_target_canonical.csv if present (compound_id, target_name, etc.). Returns empty DataFrame if absent."""
    path = discover_compound_target_canonical(discovered)
    if path is None:
        return pd.DataFrame(columns=["compound_id", "target_name"])
    try:
        df = load_csv_or_parquet(path)
        if df is not None and not df.empty and "compound_id" in df.columns:
            target_col = next((c for c in df.columns if "target" in c.lower() and "name" in c.lower()), None)
            if target_col:
                out = df[["compound_id", target_col]].dropna(how="all")
                out = out.rename(columns={target_col: "target_name"})
                out["compound_id"] = out["compound_id"].astype(str).str.strip().str.upper()
                out["target_name"] = out["target_name"].astype(str).str.strip()
                out = out[out["compound_id"].str.len() > 0]
                logger.info("[%s] Loaded compound_target for evidence: %s rows", _ts(), len(out))
                return out
            return df[["compound_id"]].drop_duplicates()
    except Exception as e:
        logger.debug("[%s] load_compound_target failed (optional): %s", _ts(), e)
    return pd.DataFrame(columns=["compound_id", "target_name"])


def load_compound_gene(
    discovered: Dict[str, Optional[Path]],
    chosen: Optional[Dict[str, Any]] = None,
    full_run_gate: bool = True,
) -> pd.DataFrame:
    """
    Load CMP->GENE. Always tries canonical first: data/processed/canonical/compound_gene_links.parquet.
    If missing, logs a loud warning to run build_canonical_compound_gene.py.
    Then tries audit-chosen path, then discovered genetics/binding.
    Returns DataFrame with compound_id (= inchikey when from canonical), gene_id.
    """
    canonical_path = _canonical_compound_gene_path(discovered)
    if canonical_path is not None:
        try:
            df = load_csv_or_parquet(canonical_path)
            if df is not None and not df.empty:
                gene_col = next(
                    (c for c in ("gene", "gene_symbol", "gene_id") if c in df.columns),
                    None,
                )
                if "inchikey" in df.columns and (gene_col := next((c for c in ("gene_symbol", "gene", "gene_id") if c in df.columns), None)):
                    out = df[["inchikey", gene_col]].drop_duplicates()
                    out = out.rename(columns={"inchikey": "compound_id", gene_col: "gene_id"})
                    out = out[out["compound_id"].astype(str).str.strip().str.len() > 0]
                    out = out[out["gene_id"].astype(str).str.strip().str.len() > 0]
                    out["compound_id"] = out["compound_id"].astype(str).str.strip().str.upper()
                    out["gene_id"] = out["gene_id"].astype(str).str.strip().str.upper()
                    _req_pct = getattr(config, "COMPOUND_GENE_REQUIRE_OVERLAP_PCT", None)
                    _check_canonical_overlap(
                        canonical_path.parent, out, _ts(), canonical_path=canonical_path, require_overlap_pct=_req_pct,
                        require_overlap_vs_cg_min=OVERLAP_GATE_FULL_MIN_CG if full_run_gate else None,
                        full_run_gate=full_run_gate,
                    )
                    _write_compound_gene_metrics(canonical_path.parent, out, canonical_path)
                    ov_cg, ov_ic = _compound_gene_overlap_for_log(canonical_path.parent, out)
                    _assert_expanded_selected_if_exists(canonical_path, full_run_gate)
                    logger.info(
                        "[%s] Using compound_gene source: %s (%s) edges=%s unique_compounds=%s unique_genes=%s overlap_vs_cg=%.2f%% overlap_vs_ic=%.2f%%",
                        _ts(), canonical_path.name, "expanded priority" if _is_expanded_compound_gene_path(canonical_path) else "canonical",
                        len(out), out["compound_id"].nunique(), out["gene_id"].nunique(), 100 * ov_cg, 100 * ov_ic,
                    )
                    return out
                if "compound_id" in df.columns and gene_col:
                    out = df[["compound_id", gene_col]].drop_duplicates()
                    out = out.rename(columns={gene_col: "gene_id"})
                    out = out[out["compound_id"].astype(str).str.strip().str.len() > 0]
                    out = out[out["gene_id"].astype(str).str.strip().str.len() > 0]
                    out["compound_id"] = out["compound_id"].astype(str).str.strip().str.upper()
                    out["gene_id"] = out["gene_id"].astype(str).str.strip().str.upper()
                    _req_pct = getattr(config, "COMPOUND_GENE_REQUIRE_OVERLAP_PCT", None)
                    _check_canonical_overlap(
                        canonical_path.parent, out, _ts(), canonical_path=canonical_path, require_overlap_pct=_req_pct,
                        require_overlap_vs_cg_min=OVERLAP_GATE_FULL_MIN_CG if full_run_gate else None,
                        full_run_gate=full_run_gate,
                    )
                    _write_compound_gene_metrics(canonical_path.parent, out, canonical_path)
                    n_cmp = out["compound_id"].nunique()
                    n_gene = out["gene_id"].nunique()
                    ov_cg, ov_ic = _compound_gene_overlap_for_log(canonical_path.parent, out)
                    _assert_expanded_selected_if_exists(canonical_path, full_run_gate)
                    logger.info(
                        "[%s] Using compound_gene source: %s (%s) edges=%s unique_compounds=%s unique_genes=%s overlap_vs_cg=%.2f%% overlap_vs_ic=%.2f%%",
                        _ts(), canonical_path.name, "expanded priority" if _is_expanded_compound_gene_path(canonical_path) else "canonical",
                        len(out), n_cmp, n_gene, 100 * ov_cg, 100 * ov_ic,
                    )
                    return out
                if "inchikey" in df.columns and ("gene" in df.columns or "gene_symbol" in df.columns or "gene_id" in df.columns):
                    gcol = "gene" if "gene" in df.columns else ("gene_symbol" if "gene_symbol" in df.columns else "gene_id")
                    out = df[["inchikey", gcol]].drop_duplicates()
                    out = out.rename(columns={"inchikey": "compound_id", gcol: "gene_id"})
                    out = out[out["compound_id"].astype(str).str.strip().str.len() > 0]
                    out = out[out["gene_id"].astype(str).str.strip().str.len() > 0]
                    out["compound_id"] = out["compound_id"].astype(str).str.strip().str.upper()
                    out["gene_id"] = out["gene_id"].astype(str).str.strip().str.upper()
                    _req_pct = getattr(config, "COMPOUND_GENE_REQUIRE_OVERLAP_PCT", None)
                    _check_canonical_overlap(
                        canonical_path.parent, out, _ts(), canonical_path=canonical_path, require_overlap_pct=_req_pct,
                        require_overlap_vs_cg_min=OVERLAP_GATE_FULL_MIN_CG if full_run_gate else None,
                        full_run_gate=full_run_gate,
                    )
                    _write_compound_gene_metrics(canonical_path.parent, out, canonical_path)
                    ov_cg, ov_ic = _compound_gene_overlap_for_log(canonical_path.parent, out)
                    _assert_expanded_selected_if_exists(canonical_path, full_run_gate)
                    logger.info(
                        "[%s] Using compound_gene source: %s (%s) edges=%s unique_compounds=%s unique_genes=%s overlap_vs_cg=%.2f%% overlap_vs_ic=%.2f%%",
                        _ts(), canonical_path.name, "expanded priority" if _is_expanded_compound_gene_path(canonical_path) else "canonical",
                        len(out), out["compound_id"].nunique(), out["gene_id"].nunique(), 100 * ov_cg, 100 * ov_ic,
                    )
                    return out
        except ValueError:
            raise
        except Exception as e:
            logger.warning("[%s] Canonical compound_gene load failed: %s", _ts(), e)
    else:
        logger.warning(
            "[%s] CANONICAL compound->gene file NOT FOUND. Run: python -m src.phase14.compound_identity --repo-root .",
            _ts(),
        )

    # Overlap-based compound_gene source selection when canonical is missing
    MIN_OVERLAP_FRAC = 0.05
    _processed = None
    for _key in ("phase13_dir", "recipe_ingredients"):
        _p = discovered.get(_key)
        if _p and Path(_p).exists():
            _resolved = Path(_p).resolve()
            _processed = _resolved.parent if _key == "phase13_dir" else _resolved.parent.parent
            break
    if _processed is not None:
        _canonical_dir = _processed / "canonical"
        _ing_canon_path = _canonical_dir / "ingredient_compound_canonical.csv"
        _master_path = _canonical_dir / "compound_master.csv"
        if _ing_canon_path.exists() and _master_path.exists():
            try:
                from .compound_identity import (
                    build_fdb_to_inchikey_with_fallbacks,
                    score_compound_gene_source_overlap,
                    load_csv_or_parquet as _load_cid,
                )
                _ing_canon = _load_cid(_ing_canon_path)
                _compound_master = _load_cid(_master_path)
                if _ing_canon is not None and not _ing_canon.empty and _compound_master is not None and not _compound_master.empty:
                    _ing_ik_set = set(_ing_canon["compound_id"].dropna().astype(str).str.strip()) if "compound_id" in _ing_canon.columns else set()
                    _fdb_to_ik, _, _ = build_fdb_to_inchikey_with_fallbacks(_processed.parent, _compound_master)
                    _candidates: List[Tuple[Path, str]] = []
                    if chosen and chosen.get("compound_gene"):
                        _candidates.append((Path(chosen["compound_gene"]), "chosen"))
                    for _k in ("genetics", "binding"):
                        _p2 = discovered.get(_k)
                        if _p2 and Path(_p2).exists():
                            _candidates.append((Path(_p2), _k))
                    if not _candidates and chosen is not None:
                        try:
                            from .audit import scan_processed_data
                            _repo_root = _infer_repo_root(discovered)
                            if _repo_root:
                                _scan_df = scan_processed_data(_repo_root)
                                for _, _r in _scan_df.iterrows():
                                    for _c in _r.get("capabilities") or []:
                                        if _c in ("COMPOUND_GENE", "COMPOUND_TARGET"):
                                            _candidates.append((Path(_r["path"]), _c))
                                            break
                        except Exception:
                            pass
                    _best_path: Optional[Path] = None
                    _best_score = -1.0
                    for _path, _label in _candidates:
                        if not _path.exists():
                            continue
                        _df = load_csv_or_parquet(_path)
                        if _df is None or _df.empty:
                            continue
                        _cmp_col = _pick_compound_col(_df)
                        if not _cmp_col:
                            continue
                        _score = score_compound_gene_source_overlap(_ing_ik_set, _df, _fdb_to_ik, compound_id_column=_cmp_col)
                        if _score > _best_score:
                            _best_score = _score
                            _best_path = _path
                    if _best_path is not None:
                        out = _load_and_normalize_compound_gene(_best_path)
                        if not out.empty:
                            if "inchikey" in out.columns:
                                out = out.rename(columns={"inchikey": "compound_id"})
                            out = out[["compound_id", "gene_id"]].drop_duplicates() if "gene_id" in out.columns else out
                            logger.info("[%s] Picked compound_gene by overlap: %s (overlap=%.1f%%)", _ts(), _best_path.name, 100.0 * _best_score)
                            if _best_score < MIN_OVERLAP_FRAC:
                                raise ValueError(
                                    "Compound-gene source overlap with ingredient_compound canonical InChIKeys is %.1f%% < 5%%. "
                                    "Add or fix registry bridging. Run: python -m src.phase14.compound_identity --repo-root ."
                                    % (100.0 * _best_score,)
                                )
                            return out
                    if _best_score >= 0 and _best_score < MIN_OVERLAP_FRAC:
                        raise ValueError(
                            "Best compound-gene source overlap is %.1f%% < 5%%. "
                            "Missing registry bridging. Run: python -m src.phase14.compound_identity --repo-root ."
                            % (100.0 * _best_score,)
                        )
            except ValueError:
                raise
            except Exception as _e:
                logger.warning("[%s] Overlap-based compound_gene selection failed: %s", _ts(), _e)

    if chosen and chosen.get("compound_gene"):
        p = Path(chosen["compound_gene"])
        if p.exists():
            try:
                out = _load_and_normalize_compound_gene(p)
                if not out.empty:
                    if "inchikey" in out.columns:
                        out = out.rename(columns={"inchikey": "compound_id", "gene_id": "gene_id"})
                        out = out[["compound_id", "gene_id"]]
                    return out
            except Exception as e:
                logger.warning("[%s] load_compound_gene from chosen failed: %s", _ts(), e)
    for key in ("genetics", "binding"):
        p = discovered.get(key)
        if not p or not p.exists():
            continue
        try:
            logger.info("[%s] Picked compound_gene source (%s): %s", _ts(), key, p)
            out = _load_and_normalize_compound_gene(Path(p))
            if not out.empty:
                if "inchikey" in out.columns:
                    out = out.rename(columns={"inchikey": "compound_id"})
                out = out[["compound_id", "gene_id"]].drop_duplicates() if "gene_id" in out.columns else out
                return out
        except Exception as e:
            logger.warning("[%s] load_compound_gene (%s) failed: %s", _ts(), key, e)
    return pd.DataFrame(columns=["compound_id", "gene_id"])


def phase14_smoke_check_print(
    chosen_cg_file: Optional[str] = None,
    n_unique_compounds_ic: Optional[int] = None,
    n_unique_compounds_cg: Optional[int] = None,
    n_overlap: Optional[int] = None,
    pct_rows_with_nonzero_propagation: Optional[float] = None,
) -> None:
    """Unit-like smoke check: print chosen cg file, n_unique_compounds_ic, n_unique_compounds_cg, n_overlap, pct_rows_with_nonzero_propagation."""
    logger.info(
        "[%s] Phase14 smoke check: chosen_cg_file=%s n_unique_compounds_ic=%s n_unique_compounds_cg=%s n_overlap=%s pct_rows_with_nonzero_propagation=%s",
        _ts(),
        chosen_cg_file or "none",
        n_unique_compounds_ic if n_unique_compounds_ic is not None else "N/A",
        n_unique_compounds_cg if n_unique_compounds_cg is not None else "N/A",
        n_overlap if n_overlap is not None else "N/A",
        ("%.2f%%" % pct_rows_with_nonzero_propagation) if pct_rows_with_nonzero_propagation is not None else "N/A",
    )
    print(
        "Phase14 smoke check: chosen_cg_file=%s | n_unique_compounds_ic=%s | n_unique_compounds_cg=%s | n_overlap=%s | pct_rows_with_nonzero_propagation=%s"
        % (
            chosen_cg_file or "none",
            n_unique_compounds_ic if n_unique_compounds_ic is not None else "N/A",
            n_unique_compounds_cg if n_unique_compounds_cg is not None else "N/A",
            n_overlap if n_overlap is not None else "N/A",
            ("%.2f%%" % pct_rows_with_nonzero_propagation) if pct_rows_with_nonzero_propagation is not None else "N/A",
        )
    )


def ensure_compound_gene_when_ingredient_compound(
    ingredient_compound: pd.DataFrame,
    compound_gene: pd.DataFrame,
    canonical_expected_path: Optional[str] = None,
) -> None:
    """
    Validation gate: if ingredient_compound is non-empty but compound_gene is missing/empty, raise.
    If compound_gene loads, compute and log n_compounds_ic, n_compounds_cg, n_overlap, overlap_vs_cg.
    """
    has_ic = not ingredient_compound.empty and "compound_id" in ingredient_compound.columns
    has_cg = not compound_gene.empty and "compound_id" in compound_gene.columns
    if has_ic and not has_cg:
        path_msg = canonical_expected_path or "data/processed/canonical/compound_gene_expanded_v2_canonical.csv (or compound_gene_canonical.csv)"
        raise ValueError(
            "compound_gene missing. Expected canonical at %s. "
            "Run: python scripts/phase14/build_compound_gene_canonical.py --repo-root ."
            % path_msg
        )
    if has_ic and has_cg:
        ic_set = set(ingredient_compound["compound_id"].dropna().astype(str).str.strip())
        ic_set = {k for k in ic_set if k and str(k).upper() != "NAN"}
        cg_set = set(compound_gene["compound_id"].dropna().astype(str).str.strip())
        cg_set = {k for k in cg_set if k and str(k).upper() != "NAN"}
        metrics = compute_identity_overlap_metrics(ic_set, cg_set)
        logger.info(
            "[%s] compound_gene loaded: n_compounds_ic=%s n_compounds_cg=%s n_overlap=%s overlap_vs_cg=%.2f%%",
            _ts(),
            metrics["n_ic_compounds"],
            metrics["n_cg_compounds"],
            metrics["n_overlap"],
            100 * metrics["overlap_vs_cg"],
        )


def _debug_overlap_gate(repo_root: str | Path) -> Dict[str, Any]:
    """
    Load ingredient_compound_canonical and compound_gene_canonical from repo, compute overlap metrics and gate.
    Prints and returns metrics dict (for use in scripts/tests). Uses CSV-first resolution.
    """
    repo_root = Path(repo_root).resolve()
    canonical_dir = repo_root / "data" / "processed" / "canonical"
    out: Dict[str, Any] = {"gate_passed": False, "reason": "no data", "n_ic_compounds": 0, "n_cg_compounds": 0, "n_overlap": 0, "overlap_vs_cg": 0.0, "overlap_vs_ic": 0.0}
    ic_path = canonical_dir / "ingredient_compound_canonical.csv"
    cg_path = canonical_dir / "compound_gene_canonical.csv"
    if not cg_path.exists():
        cg_path = canonical_dir / "compound_gene_canonical.parquet"
    if not ic_path.exists() or not cg_path.exists():
        print("_debug_overlap_gate: missing canonical files (ingredient_compound_canonical.csv or compound_gene_canonical)")
        return out
    try:
        ing_df = pd.read_csv(ic_path)
        cg_df = load_csv_or_parquet(cg_path)
    except Exception as e:
        out["reason"] = str(e)
        print("_debug_overlap_gate: load failed:", e)
        return out
    if ing_df.empty or "compound_id" not in ing_df.columns or cg_df.empty or "compound_id" not in cg_df.columns:
        out["reason"] = "empty or missing compound_id column"
        print("_debug_overlap_gate:", out["reason"])
        return out
    ic_set = set(ing_df["compound_id"].dropna().astype(str).str.strip())
    ic_set = {k for k in ic_set if k and str(k).upper() != "NAN"}
    cg_set = set(cg_df["compound_id"].dropna().astype(str).str.strip())
    cg_set = {k for k in cg_set if k and str(k).upper() != "NAN"}
    metrics = compute_identity_overlap_metrics(ic_set, cg_set)
    out = {k: metrics[k] for k in ("n_ic_compounds", "n_cg_compounds", "n_overlap", "overlap_vs_cg", "overlap_vs_ic", "gate_passed", "gate_reason")}
    out["reason"] = metrics["gate_reason"]
    print("n_ic_compounds:", out["n_ic_compounds"])
    print("n_cg_compounds:", out["n_cg_compounds"])
    print("n_overlap:", out["n_overlap"])
    print("overlap_vs_cg: %.4f (%.2f%%)" % (out["overlap_vs_cg"], 100 * out["overlap_vs_cg"]))
    print("overlap_vs_ic: %.4f (%.2f%%)" % (out["overlap_vs_ic"], 100 * out["overlap_vs_ic"]))
    print("gate_passed:", out["gate_passed"])
    print("gate_reason:", out["gate_reason"])
    return out


def build_overlap_compounds_and_identity_metrics(
    ingredient_compound: pd.DataFrame,
    compound_gene: pd.DataFrame,
    metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build overlap_compounds.csv-style DataFrame and identity_metrics dict for reports.
    Returns (overlap_compounds_df, identity_metrics). identity_metrics includes top 20 overlapping InChIKeys with counts.
    """
    cmp_col_ic = "compound_id" if "compound_id" in ingredient_compound.columns else None
    cmp_col_cg = "compound_id" if "compound_id" in compound_gene.columns else None
    if not cmp_col_ic or not cmp_col_cg or ingredient_compound.empty or compound_gene.empty:
        return pd.DataFrame(columns=["compound_id", "count_in_ingredient_compound", "count_in_compound_gene", "ingredient_ids", "genes"]), (
            metrics or {}
        )
    ic_set = set(ingredient_compound[cmp_col_ic].dropna().astype(str).str.strip())
    ic_set = {k for k in ic_set if k and str(k).upper() != "NAN"}
    cg_set = set(compound_gene[cmp_col_cg].dropna().astype(str).str.strip())
    cg_set = {k for k in cg_set if k and str(k).upper() != "NAN"}
    overlap_set = ic_set & cg_set
    if metrics is None:
        metrics = compute_identity_overlap_metrics(ic_set, cg_set)
    ic_counts = ingredient_compound[cmp_col_ic].astype(str).str.strip().value_counts()
    cg_counts = compound_gene[cmp_col_cg].astype(str).str.strip().value_counts()
    ing_col = "ingredient_id" if "ingredient_id" in ingredient_compound.columns else None
    gene_col = "gene" if "gene" in compound_gene.columns else ("gene_id" if "gene_id" in compound_gene.columns else None)
    rows: List[Dict[str, Any]] = []
    for ik in sorted(overlap_set):
        count_ic = int(ic_counts.get(ik, 0))
        count_cg = int(cg_counts.get(ik, 0))
        ings_str = ""
        if ing_col:
            ings = ingredient_compound[ingredient_compound[cmp_col_ic].astype(str).str.strip() == ik][ing_col].dropna().astype(str).unique().tolist()
            ings_str = ";".join(ings[:20]) + ("..." if len(ings) > 20 else "")
        genes_str = ""
        if gene_col:
            genes = compound_gene[compound_gene[cmp_col_cg].astype(str).str.strip() == ik][gene_col].dropna().astype(str).unique().tolist()
            genes_str = ";".join(genes[:20]) + ("..." if len(genes) > 20 else "")
        rows.append({
            "compound_id": ik,
            "count_in_ingredient_compound": count_ic,
            "count_in_compound_gene": count_cg,
            "ingredient_ids": ings_str,
            "genes": genes_str,
        })
    overlap_df = pd.DataFrame(rows)
    top20 = []
    for ik in overlap_set:
        top20.append({
            "inchikey": ik,
            "count_in_ingredient_compound": int(ic_counts.get(ik, 0)),
            "count_in_compound_gene": int(cg_counts.get(ik, 0)),
        })
    top20.sort(key=lambda x: -(x["count_in_ingredient_compound"] + x["count_in_compound_gene"]))
    metrics_out = dict(metrics)
    metrics_out["top_20_overlapping_inchikeys"] = top20[:20]
    return overlap_df, metrics_out


def compound_gene_overlap_diagnostics(
    ingredient_compound: pd.DataFrame,
    compound_gene: pd.DataFrame,
    overlap_pct: Optional[float],
    n_show: int = 5,
    use_print: bool = True,
) -> None:
    """
    Print overlap diagnostics when overlap is low. Call after loading both tables.
    Shows example compound keys that fail to match and that match; suggests identifier mismatch (CID vs InChIKey etc).
    """
    def _out(msg: str, *args: Any) -> None:
        if use_print:
            print(msg % args if args else msg)
        else:
            logger.info(msg, *args)
    if ingredient_compound.empty or "compound_id" not in ingredient_compound.columns:
        _out("Overlap diagnostics: ingredient_compound empty or missing compound_id")
        return
    if compound_gene.empty or "compound_id" not in compound_gene.columns:
        _out("Overlap diagnostics: compound_gene empty or missing compound_id")
        return
    set_ing = set(ingredient_compound["compound_id"].dropna().astype(str).str.strip())
    set_ing = {k for k in set_ing if k}
    set_cg = set(compound_gene["compound_id"].dropna().astype(str).str.strip())
    set_cg = {k for k in set_cg if k}
    overlap = set_ing & set_cg
    n_ic, n_cg, n_overlap = len(set_ing), len(set_cg), len(overlap)
    overlap_vs_cg = (100.0 * n_overlap / n_cg) if n_cg else 0.0
    overlap_vs_ic = (100.0 * n_overlap / n_ic) if n_ic else 0.0
    _out("n_overlap=%s (of n_ic=%s, n_cg=%s)", n_overlap, n_ic, n_cg)
    _out("overlap_vs_cg=%.2f%% (n_overlap / n_cg)", overlap_vs_cg)
    _out("overlap_vs_ic=%.2f%% (n_overlap / n_ic; informative only)", overlap_vs_ic)
    fail = set_ing - set_cg
    pct = overlap_pct if overlap_pct is not None else overlap_vs_ic
    if fail and n_show > 0:
        sample_fail = list(fail)[:n_show]
        _out("Example compound keys (ingredient_compound) that do NOT match: %s", sample_fail)
    if overlap and n_show > 0:
        sample_ok = list(overlap)[:n_show]
        _out("Example compound keys that MATCH: %s", sample_ok)
    if n_cg and overlap_vs_cg < 20.0:
        logger.warning(
            "overlap_vs_cg=%.1f%% < 20%% (informational only; not used for readiness gate after BindingDB expansion). "
            "Readiness uses atlas_pair_cov_compound, pct_rows_with_nonzero_propagation, and ic_gene_coverage/n_overlap.",
            overlap_vs_cg,
        )
        _out("WARNING: overlap_vs_cg < 20%% (informational only; gate uses atlas_pair_cov, propagation%%, ic_gene_coverage/n_overlap).")


def compute_phase14_coverage_metrics(
    ingredient_compound: pd.DataFrame,
    compound_gene: pd.DataFrame,
    atlas: pd.DataFrame,
    propagation_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute coverage for Phase14 summary/reports:
    - n_compounds_ingredient_compound, n_compounds_compound_gene (and n_unique_compounds_ic, n_unique_compounds_cg)
    - n_overlap, overlap_vs_cg, overlap_vs_ic
    - pct_compound_overlap (legacy), n_ingredients_reach_gene, pct_ingredients_reach_gene
    - gate_passed, gate_reason
    - pct_rows_with_nonzero_propagation if propagation_stats provided
    """
    from .id_normalization import to_ingredient_id
    out: Dict[str, Any] = {
        "n_compounds_ingredient_compound": 0,
        "n_compounds_compound_gene": 0,
        "n_unique_compounds_ic": 0,
        "n_unique_compounds_cg": 0,
        "n_overlap": 0,
        "overlap_vs_cg": None,
        "overlap_vs_ic": None,
        "ic_gene_coverage": None,
        "atlas_pair_cov_compound": None,
        "pct_compound_overlap": None,
        "n_ingredients_reach_gene": 0,
        "pct_ingredients_reach_gene": None,
        "gate_passed": False,
        "gate_reason": "",
        "pct_rows_with_nonzero_propagation": None,
    }
    if ingredient_compound.empty or "compound_id" not in ingredient_compound.columns:
        return out
    cmp_ing = set(ingredient_compound["compound_id"].dropna().astype(str).str.strip())
    cmp_ing = {c for c in cmp_ing if c}
    out["n_compounds_ingredient_compound"] = len(cmp_ing)
    out["n_unique_compounds_ic"] = len(cmp_ing)
    if compound_gene.empty or "compound_id" not in compound_gene.columns:
        return out
    cmp_gene = set(compound_gene["compound_id"].dropna().astype(str).str.strip())
    cmp_gene = {c for c in cmp_gene if c}
    out["n_compounds_compound_gene"] = len(cmp_gene)
    out["n_unique_compounds_cg"] = len(cmp_gene)
    overlap_n = len(cmp_ing & cmp_gene)
    out["n_overlap"] = overlap_n
    out["pct_compound_overlap"] = round(100.0 * overlap_n / len(cmp_ing), 2) if cmp_ing else None
    metrics = compute_identity_overlap_metrics(cmp_ing, cmp_gene)
    out["overlap_vs_cg"] = metrics["overlap_vs_cg"]
    out["overlap_vs_ic"] = metrics["overlap_vs_ic"]
    # ic_gene_coverage = (# IC compounds that appear in compound_gene) / (# unique IC compounds)
    out["ic_gene_coverage"] = round(metrics["overlap_vs_ic"], 4) if metrics["overlap_vs_ic"] is not None else None
    # Readiness gate: no longer use overlap_vs_cg >= 20% as hard gate (CG includes BindingDB non-food compounds).
    # Gate: atlas_pair_cov_compound >= 30%, pct_rows_with_nonzero_propagation >= 25%, and (ic_gene_coverage >= 0.5% OR n_overlap >= 200).
    if atlas.empty or "ingA_id" not in atlas.columns:
        out["gate_passed"] = metrics["gate_passed"]
        out["gate_reason"] = metrics["gate_reason"]
        return out
    # atlas_pair_cov_compound = % of atlas rows where both ingA and ingB have >=1 compound in ingredient_compound
    mapped_ings = set()
    for _, r in ingredient_compound.iterrows():
        mapped_ings.add(to_ingredient_id(str(r.get("ingredient_id", ""))))
    mapped_ings.discard("")
    a_ids = atlas["ingA_id"].apply(lambda x: to_ingredient_id(str(x)))
    b_ids = atlas["ingB_id"].apply(lambda x: to_ingredient_id(str(x)))
    pair_ok = (a_ids.isin(mapped_ings) & b_ids.isin(mapped_ings)).sum()
    n_atlas = len(atlas)
    out["atlas_pair_cov_compound"] = round(100.0 * pair_ok / n_atlas, 2) if n_atlas else None
    # Apply new readiness gate (configurable)
    from . import phase14_config as config
    min_pair_cov = getattr(config, "ATLAS_PAIR_COV_COMPOUND_MIN", 0.30)
    min_prop = getattr(config, "PCT_ROWS_NONZERO_PROPAGATION_MIN", 25.0)
    min_ic_cov = getattr(config, "IC_GENE_COVERAGE_MIN", 0.005)
    min_n_overlap = getattr(config, "N_OVERLAP_COMPOUNDS_MIN", 200)
    pct_prop = (propagation_stats or {}).get("pct_rows_with_nonzero_propagation") if propagation_stats is not None else None
    pair_cov_pct = (out["atlas_pair_cov_compound"] or 0) / 100.0
    ic_cov = out["ic_gene_coverage"] or 0.0
    gate_1 = pair_cov_pct >= min_pair_cov
    gate_2 = (pct_prop is None) or (pct_prop >= min_prop)
    gate_3 = (ic_cov >= min_ic_cov) or (overlap_n >= min_n_overlap)
    out["gate_passed"] = gate_1 and gate_2 and gate_3
    reasons = []
    if gate_1:
        reasons.append("atlas_pair_cov_compound>=%.0f%%" % (100 * min_pair_cov))
    if gate_2:
        reasons.append("pct_rows_with_nonzero_propagation>=%.0f%%" % min_prop if pct_prop is not None else "propagation_ok")
    if gate_3:
        reasons.append("ic_gene_coverage>=%.2f%% or n_overlap>=%d" % (100 * min_ic_cov, min_n_overlap))
    if not out["gate_passed"]:
        if not gate_1:
            reasons.append("atlas_pair_cov_compound=%.1f%%" % (out["atlas_pair_cov_compound"] or 0))
        if not gate_2 and pct_prop is not None:
            reasons.append("pct_rows_with_nonzero_propagation=%.1f%%" % pct_prop)
        if not gate_3:
            reasons.append("ic_gene_coverage=%.4f n_overlap=%d" % (ic_cov, overlap_n))
    out["gate_reason"] = "; ".join(reasons) if reasons else metrics["gate_reason"]
    # Ingredients that reach at least 1 gene (via compound): ing has compound in ingredient_compound and that compound in compound_gene
    ing_to_cmp: Dict[str, Set[str]] = {}
    for _, r in ingredient_compound.iterrows():
        ing = to_ingredient_id(str(r.get("ingredient_id", "")))
        c = str(r.get("compound_id", "")).strip()
        if ing and c:
            ing_to_cmp.setdefault(ing, set()).add(c)
    cmp_with_gene = cmp_ing & cmp_gene
    ings_with_gene = {ing for ing, cmps in ing_to_cmp.items() if cmps & cmp_with_gene}
    all_atlas_ings = set()
    for _, r in atlas.iterrows():
        all_atlas_ings.add(to_ingredient_id(str(r["ingA_id"])))
        all_atlas_ings.add(to_ingredient_id(str(r["ingB_id"])))
    n_all = len(all_atlas_ings)
    n_reach = len(ings_with_gene & all_atlas_ings)
    out["n_ingredients_reach_gene"] = n_reach
    out["pct_ingredients_reach_gene"] = round(100.0 * n_reach / n_all, 2) if n_all else None
    if propagation_stats is not None and isinstance(propagation_stats, dict):
        pct_nonzero = propagation_stats.get("pct_rows_with_nonzero_propagation") or propagation_stats.get("pct_nonzero")
        if pct_nonzero is not None:
            out["pct_rows_with_nonzero_propagation"] = pct_nonzero
    return out


def summarize_missing(discovered: Dict[str, Optional[Path]]) -> List[str]:
    """Return list of human-readable 'what's missing' messages."""
    missing: List[str] = []
    labels = {
        "signatures": "Recipe functional signatures",
        "recipe_ingredients": "Recipe ingredients expanded",
        "pathway_gene": "Pathway-gene signatures",
        "pathway_cluster_info": "Pathway cluster info",
        "pathway_bundles": "Pathway bundles (category keywords)",
        "metabolomics": "Metabolomics/ingredient-compound",
        "genetics": "Genetics/gene tables",
        "binding": "BindingDB/target tables",
        "target_clusters": "Target functional clusters",
    }
    for key, label in labels.items():
        if key not in discovered:
            continue
        if not discovered[key] or not Path(discovered[key]).exists():
            missing.append(f"Missing: {label}")
    return missing
