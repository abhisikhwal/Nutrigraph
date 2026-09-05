"""Preparation-label retention for species-collapsed ingredient mappings."""
from __future__ import annotations

import re

from scripts.species_remap.species_match import normalize_name, strip_modifiers

# Cured / cut pork preparations → parent Sus scrofa (549)
PORK_PREPARATION_EXACT: dict[str, str] = {
    "bacon": "bacon",
    "ham": "ham",
    "prosciutto": "prosciutto",
    "pancetta": "pancetta",
    "canadian bacon": "canadian_bacon",
    "bacon bits": "bacon_bits",
    "bacon drippings": "bacon_drippings",
    "bacon fat": "bacon_fat",
    "ham hock": "ham_hock",
    "ham bone": "ham_bone",
    "smoked ham": "smoked_ham",
    "spiral ham": "spiral_ham",
    "deli ham": "deli_ham",
    "black forest ham": "black_forest_ham",
    "serrano ham": "serrano_ham",
    "country ham": "country_ham",
    "ground pork": "ground_pork",
    "pork chops": "pork_chops",
    "pork chop": "pork_chop",
    "pork loin": "pork_loin",
    "pork tenderloin": "pork_tenderloin",
    "pork shoulder": "pork_shoulder",
    "pork butt": "pork_butt",
    "pork": "pork",
}

# Flour varieties → parent Flour (825)
FLOUR_PREPARATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^all-purpose$", re.I), "all_purpose_fragment"),
    (re.compile(r"all[- ]purpose", re.I), "all_purpose"),
    (re.compile(r"bread flour", re.I), "bread"),
    (re.compile(r"cake flour", re.I), "cake"),
    (re.compile(r"pastry flour", re.I), "pastry"),
    (re.compile(r"self[- ]rising", re.I), "self_rising"),
    (re.compile(r"whole wheat", re.I), "whole_wheat"),
    (re.compile(r"wheat flour", re.I), "wheat"),
    (re.compile(r"white flour", re.I), "white"),
    (re.compile(r"plain flour", re.I), "plain"),
]

SUGAR_FOOD_ID = 670
SUGAR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"brown sugar", re.I), "brown"),
    (re.compile(r"powdered sugar|icing sugar|confectioners", re.I), "powdered"),
    (re.compile(r"granulated sugar", re.I), "granulated"),
    (re.compile(r"raw sugar|turbinado|demerara", re.I), "raw"),
    (re.compile(r"white sugar", re.I), "white"),
]

RICE_FOOD_ID = 125
RICE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"basmati", re.I), "basmati"),
    (re.compile(r"jasmine", re.I), "jasmine"),
    (re.compile(r"arborio", re.I), "arborio"),
    (re.compile(r"brown rice", re.I), "brown"),
    (re.compile(r"white rice", re.I), "white"),
    (re.compile(r"wild rice", re.I), "wild"),
    (re.compile(r"long grain", re.I), "long_grain"),
    (re.compile(r"short grain", re.I), "short_grain"),
]

MILK_FOOD_ID = 632
MILK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"skim|nonfat|fat free", re.I), "skim"),
    (re.compile(r"whole milk", re.I), "whole"),
    (re.compile(r"2%|two percent", re.I), "2_percent"),
    (re.compile(r"1%|one percent", re.I), "1_percent"),
    (re.compile(r"semi[- ]skim", re.I), "semi_skimmed"),
    (re.compile(r"low fat|low-fat", re.I), "low_fat"),
]

BUTTER_FOOD_ID = 667
PEPPER_SPICE_FOOD_ID = 139
CAPSICUM_FOOD_ID = 40

MERGE_ANNOTATIONS: dict[str, str] = {
    normalize_name("salmon"): "family_level_coarse: Salmonidae (504) is family-level, not species-specific",
    normalize_name("all-purpose"): "truncated_fragment: string is truncated recipe fragment for all-purpose flour",
}


def derive_preparation_label(ingredient_string: str, foodb_id: int) -> str | None:
    """Return preparation/variety label when string collapses to a parent species node."""
    n = normalize_name(ingredient_string)

    if foodb_id == 549:
        if n in PORK_PREPARATION_EXACT:
            return PORK_PREPARATION_EXACT[n]
        for key, label in PORK_PREPARATION_EXACT.items():
            if key in n or n in key:
                return label
        if "pork" in n:
            return re.sub(r"[^a-z0-9]+", "_", n.strip())[:64]

    if foodb_id == 825:
        for pat, label in FLOUR_PREPARATION_PATTERNS:
            if pat.search(n):
                return label
        if n == "flour":
            return None
        stripped = strip_modifiers(ingredient_string)
        if stripped != "flour" and "flour" in n:
            return re.sub(r"[^a-z0-9]+", "_", stripped)[:64]

    if foodb_id == SUGAR_FOOD_ID:
        for pat, label in SUGAR_PATTERNS:
            if pat.search(n):
                return label
        if n != "sugar":
            return re.sub(r"[^a-z0-9]+", "_", strip_modifiers(ingredient_string))[:64] if n != "sugar" else None

    if foodb_id == RICE_FOOD_ID:
        for pat, label in RICE_PATTERNS:
            if pat.search(n):
                return label

    if foodb_id == MILK_FOOD_ID:
        for pat, label in MILK_PATTERNS:
            if pat.search(n):
                return label

    if foodb_id == BUTTER_FOOD_ID:
        if "unsalted" in n:
            return "unsalted"
        if "salted" in n:
            return "salted"

    if foodb_id == PEPPER_SPICE_FOOD_ID:
        if "black" in n:
            return "black"
        if "white" in n:
            return "white"
        if "ground" in n or "freshly ground" in n or "cracked" in n:
            return "ground"
        if n == "pepper":
            return "black_pepper_default"

    if foodb_id == CAPSICUM_FOOD_ID:
        if "paprika" in n:
            return "paprika_ground"
        if "bell" in n:
            return "bell"
        for color in ("green", "red", "yellow", "orange"):
            if f"{color} pepper" in n or f"{color} bell" in n:
                return color
        if n in ("peppers", "pepper") or n.endswith(" peppers"):
            return "peppers_plural"

    # Generic: retain modifier-stripped form when it differs from bare species token
    stripped = strip_modifiers(ingredient_string)
    if stripped and stripped != n and len(stripped) > 2:
        base_tokens = {"flour", "sugar", "rice", "milk", "butter", "pepper", "onion", "chicken", "beef"}
        if not any(stripped == b for b in base_tokens):
            return re.sub(r"[^a-z0-9]+", "_", stripped)[:64]

    return None


def merge_annotation(ingredient_string: str) -> str | None:
    return MERGE_ANNOTATIONS.get(normalize_name(ingredient_string))
