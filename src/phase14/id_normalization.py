"""
Phase14: Canonical ID normalization — ING_, CMP_, GENE_, PATH_, CAT_, INT_.
"""
from __future__ import annotations

import re
from typing import Optional

# Unsafe chars for IDs (keep alphanumeric, underscore, hyphen in normalized form)
UNSAFE_PATTERN = re.compile(r"[^\w\-]", re.ASCII)


def normalize_for_id(raw: str, uppercase_genes: bool = False) -> str:
    """
    Strip whitespace, replace spaces with underscore, remove unsafe chars.
    If uppercase_genes, uppercase the result (for gene symbols).
    """
    if not isinstance(raw, str) or not raw:
        return ""
    s = str(raw).strip().replace(" ", "_")
    s = UNSAFE_PATTERN.sub("", s)
    if uppercase_genes:
        s = s.upper()
    return s


def to_ingredient_id(raw: str) -> str:
    """Ensure ingredient ID has ING_ prefix."""
    s = str(raw).strip()
    if s.upper().startswith("ING_"):
        return s
    normalized = normalize_for_id(s, uppercase_genes=False)
    if normalized.isdigit():
        return f"ING_{normalized.zfill(6)}"
    return f"ING_{normalized}" if normalized else "ING_unknown"


def to_compound_id(raw: str, index: Optional[int] = None) -> str:
    """Canonical compound ID: CMP_xxxxxx (index or normalized name)."""
    s = str(raw).strip()
    if s.upper().startswith("CMP_"):
        return s
    if index is not None:
        return f"CMP_{index:06d}"
    normalized = normalize_for_id(s)[:32]
    if not normalized:
        return "CMP_000000"
    return f"CMP_{normalized}"


def to_gene_id(symbol_or_entrez: str) -> str:
    """Canonical gene ID: GENE_<symbol or entrez>, uppercase symbol."""
    s = str(symbol_or_entrez).strip()
    if s.upper().startswith("GENE_"):
        return s
    normalized = normalize_for_id(s, uppercase_genes=True)
    if not normalized:
        return "GENE_unknown"
    return f"GENE_{normalized}"


def to_pathway_id(raw: str, index: Optional[int] = None) -> str:
    """Canonical pathway ID: PATH_<id or normalized name>."""
    s = str(raw).strip()
    if s.upper().startswith("PATH_"):
        return s
    if index is not None:
        return f"PATH_{index:06d}"
    normalized = normalize_for_id(s)[:64]
    if not normalized:
        return "PATH_unknown"
    return f"PATH_{normalized}"


def to_category_id(category: str) -> str:
    """Canonical category ID: CAT_<category>."""
    s = str(category).strip().lower().replace(" ", "_")
    s = UNSAFE_PATTERN.sub("", s)
    if not s:
        return "CAT_other"
    return f"CAT_{s}" if not s.startswith("cat_") else s


def to_interaction_id(ing_a: str, ing_b: str) -> str:
    """Canonical interaction ID: INT_<ingA>_<ingB>, ordered so A <= B for consistency."""
    a = to_ingredient_id(ing_a)
    b = to_ingredient_id(ing_b)
    if a > b:
        a, b = b, a
    return f"INT_{a}_{b}"


def to_triplet_id(ing_a: str, ing_b: str, ing_c: str) -> str:
    """Canonical triplet interaction ID: INT3_<ingA>_<ingB>_<ingC>, sorted."""
    ings = sorted([to_ingredient_id(ing_a), to_ingredient_id(ing_b), to_ingredient_id(ing_c)])
    return f"INT3_{ings[0]}_{ings[1]}_{ings[2]}"
