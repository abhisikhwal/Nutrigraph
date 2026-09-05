"""
Phase14: Display-name enrichment for Neo4j export.
Enriches node names for readability without changing graph structure or scoring.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _is_compound_id_like(s: str) -> bool:
    """True if s looks like CMP_* or InChIKey (no human name)."""
    if not s or not str(s).strip():
        return True
    s = str(s).strip()
    if s.upper().startswith("CMP_"):
        return True
    # InChIKey: 27 chars, two parts separated by hyphen
    if len(s) >= 27 and "-" in s and re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", s, re.IGNORECASE):
        return True
    return False


def load_compound_id_to_display_name(repo_root: Path) -> Dict[str, str]:
    """
    Build compound_id -> display name from compound_master (inchikey, compound_master_id, name).
    Keys: inchikey, CMP_* ids if present. Value: name column or first non-empty name-like field.
    """
    root = Path(repo_root).resolve()
    out: Dict[str, str] = {}
    for name in ["compound_master.csv", "compound_master.parquet", "compound_master_v2.csv"]:
        p = root / "data" / "processed" / "canonical" / name
        if not p.exists():
            continue
        try:
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, nrows=500000, low_memory=False)
            name_col = next((c for c in df.columns if c.lower() in ("name", "compound_name", "name_norm")), None)
            id_cols = [c for c in df.columns if "inchikey" in c.lower() or "compound_master_id" in c.lower() or c.lower() == "compound_id"]
            if not name_col or not id_cols:
                continue
            for _, row in df.iterrows():
                disp = row.get(name_col)
                if pd.isna(disp) or not str(disp).strip() or _is_compound_id_like(str(disp)):
                    continue
                disp = str(disp).strip()[:128]
                for col in id_cols:
                    v = row.get(col)
                    if pd.isna(v) or not str(v).strip():
                        continue
                    k = str(v).strip()
                    if k and k not in out:
                        out[k] = disp
            break
        except Exception:
            continue
    return out


def pathway_unknown_to_better_label(repo_root: Path) -> Dict[str, str]:
    """
    Build pathway_id -> better display label for clusters that normalize to 'Unknown pathway'.
    Uses pathway_cluster_info top_terms (first 3-5 terms) for cluster_id 0 so export shows a descriptive label.
    """
    from .id_normalization import to_pathway_id
    root = Path(repo_root).resolve()
    pci_path = root / "data" / "processed" / "features" / "pathway_cluster_info.csv"
    out: Dict[str, str] = {}
    if not pci_path.exists():
        return out
    try:
        df = pd.read_csv(pci_path, nrows=100)
        if "cluster_id" not in df.columns:
            return out
        top_col = "top_terms" if "top_terms" in df.columns else df.columns[2] if len(df.columns) > 2 else None
        for _, row in df.iterrows():
            cid = row.get("cluster_id")
            pid = to_pathway_id(f"cluster_{cid}", index=int(cid) if isinstance(cid, (int, float)) else None)
            label = str(row.get("auto_label", "")).strip()
            if not label or "unknown" in label.lower():
                if top_col:
                    terms = row.get(top_col, "")
                    if isinstance(terms, str) and terms.startswith("["):
                        import ast
                        try:
                            arr = ast.literal_eval(terms)
                            if isinstance(arr, list) and arr:
                                first = [str(x) for x in arr[:5] if x and str(x).strip() and str(x).lower() not in ("unknown", "pathway")]
                                if first:
                                    out[pid] = " / ".join(first)[:80]
                        except Exception:
                            pass
            if pid not in out and (not label or "unknown" in label.lower()):
                out[pid] = f"Pathway cluster {cid}"
    except Exception:
        pass
    return out


def clean_pathway_display_name(raw: str) -> str:
    """
    Create clean_display_name for Pathway nodes:
    - Replace underscores with spaces
    - Collapse repeated tokens/phrases (e.g. 'pathway pathway' -> 'pathway')
    - Normalize 'unknown pathway_unknown_pathway' -> 'Unknown pathway'
    - Preserve original raw in separate field; this returns the clean string only.
    """
    if not raw or not str(raw).strip():
        return str(raw) if raw else ""
    s = str(raw).strip()
    s = s.replace("_", " ")
    # Collapse repeated words (case-insensitive)
    words = s.split()
    seen: Dict[str, bool] = {}
    unique: list[str] = []
    for w in words:
        key = w.lower()
        if key not in seen:
            seen[key] = True
            unique.append(w)
    s = " ".join(unique)
    # Normalize "unknown pathway" pattern
    if re.match(r"^unknown\s+pathway(\s+unknown\s+pathway)*$", s, re.IGNORECASE) or s.lower() == "unknown pathway":
        return "Unknown pathway"
    if s.lower().startswith("unknown pathway"):
        return "Unknown pathway"
    return s[:128]


def enrich_display_names(
    nodes_df: pd.DataFrame,
    repo_root: Path,
    compound_id_to_name: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add or fill display_name for each node type. Modifies a copy of nodes_df.
    Returns (enriched_df, stats) where stats has counts and sample before/after.
    """
    if nodes_df.empty or "label" not in nodes_df.columns:
        return nodes_df.copy(), {"error": "empty or no label column"}

    df = nodes_df.copy()
    if "name" not in df.columns:
        df["name"] = df.get("node_id", df.get(":ID", pd.Series(dtype=object)))
    if "display_name" not in df.columns:
        df["display_name"] = df["name"].astype(str)

    node_id_col = "node_id" if "node_id" in df.columns else ":ID"
    if node_id_col not in df.columns:
        return df, {"error": "no node_id or :ID"}

    stats: Dict[str, Any] = {
        "compound_improved": 0,
        "pathway_improved": 0,
        "category_improved": 0,
        "ingredient_improved": 0,
        "compound_before_after": [],
        "pathway_before_after": [],
        "category_before_after": [],
    }

    root = Path(repo_root).resolve()
    cmp_map = compound_id_to_name if compound_id_to_name is not None else load_compound_id_to_display_name(root)

    for idx, row in df.iterrows():
        label = str(row.get("label", "")).strip().lower()
        nid = row.get(node_id_col)
        name = str(row.get("name", "")).strip()
        disp = str(row.get("display_name", "")).strip()

        if label == "compound":
            if _is_compound_id_like(name) and (not disp or _is_compound_id_like(disp)):
                new_name = cmp_map.get(str(nid).strip()) or cmp_map.get(name) or name
                if new_name != name and not _is_compound_id_like(new_name):
                    if stats["compound_improved"] < 5:
                        stats["compound_before_after"].append({"before": name[:50], "after": new_name[:50]})
                    stats["compound_improved"] += 1
                    df.at[idx, "display_name"] = new_name
                else:
                    df.at[idx, "display_name"] = name
            else:
                df.at[idx, "display_name"] = disp or name

        elif label == "pathway":
            clean = clean_pathway_display_name(name)
            if clean != name:
                if stats["pathway_improved"] < 5:
                    stats["pathway_before_after"].append({"before": name[:60], "after": clean[:60]})
                stats["pathway_improved"] += 1
            df.at[idx, "display_name"] = clean

        elif label == "category":
            # Human-readable: replace underscores with spaces, title-case each word
            raw = (disp or name or "").strip()
            clean = raw.replace("_", " ").strip()
            if clean:
                clean = " ".join(w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper() for w in clean.split())
            if clean != raw:
                if stats["category_improved"] < 5:
                    stats["category_before_after"].append({"before": raw[:40], "after": clean[:40]})
                stats["category_improved"] += 1
            df.at[idx, "display_name"] = clean or raw

        elif label == "ingredient":
            df.at[idx, "display_name"] = name or str(nid)
            if name and name != str(nid):
                stats["ingredient_improved"] += 1

        else:
            # Interaction, Gene: keep name as display
            df.at[idx, "display_name"] = disp or name

    return df, stats
