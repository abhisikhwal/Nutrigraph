"""Approved pepper disambiguation ruling (human-locked)."""
from __future__ import annotations

import re

from scripts.species_remap.species_match import normalize_name, singularize, strip_modifiers

# FooDB ids
PIPER_NIGRUM = 139
CAPSICUM = 40
CHILI = 731

VEG_MARKERS = re.compile(
    r"\b(bell|capsicum|sweet|jalapeno|jalapeño|habanero|serrano|poblano|anaheim|"
    r"chipotle|piquillo|pepperoncini|banana pepper)\b",
    re.I,
)
COLOR_PEPPER = re.compile(r"\b(green|red|yellow|orange)\s+pepper", re.I)
HOT_MARKERS = re.compile(r"\b(hot|chili|chile|chilli)\b", re.I)


def apply_pepper_ruling(ingredient_string: str) -> dict | None:
    """
    Return dict with foodb_food_id, species_name, latin_name, reasoning if ruled; else None.
    Locked rules:
      - unqualified pepper / freshly ground pepper / ground pepper -> Piper nigrum (139)
      - hot/chili/bell/color/plural peppers -> Capsicum (40) or Chili (731)
    """
    raw = str(ingredient_string).strip()
    n = normalize_name(raw)
    stripped = strip_modifiers(raw)
    base = singularize(stripped)

    if "pepper" not in n:
        return None

    # Explicit spice forms already handled in pass 1; skip
    if any(x in n for x in ("black pepper", "white pepper", "cayenne", "peppercorn")):
        return None

    # Vegetable / chili cues
    if VEG_MARKERS.search(n) or COLOR_PEPPER.search(n):
        fid = CAPSICUM
        return _result(fid, "Capsicum-qualified pepper -> vegetable Capsicum", n)

    if HOT_MARKERS.search(n) and "pepper" in n:
        fid = CHILI if "chili" in n or "chile" in n or "chilli" in n else CAPSICUM
        name = "Chili" if fid == CHILI else "Pepper"
        return _result(
            fid,
            f"Hot/chili pepper cue -> {name} ({'Capsicum/Chili ruling'})",
            n,
            judgment_call=fid == CAPSICUM and "hot" in n,
            judgment_note="hot pepper -> Chili (731) vs Capsicum (40)" if fid == CAPSICUM else None,
        )

    if base == "peppers" or n.endswith(" peppers") or n == "peppers":
        return _result(CAPSICUM, "Plural peppers -> vegetable Capsicum (40)", n)

    # Spice: unqualified or ground (without hot/bell/color)
    if base == "pepper" or n in ("pepper", "peppers"):
        if n == "peppers":
            return _result(CAPSICUM, "Plural peppers -> vegetable Capsicum (40)", n)
        return _result(PIPER_NIGRUM, "Unqualified pepper -> Piper nigrum (139)", n)

    if n in ("freshly ground pepper", "fresh ground pepper", "ground pepper", "freshly grated pepper", "grated pepper"):
        return _result(PIPER_NIGRUM, "Ground/grated pepper (unqualified) -> Piper nigrum (139)", n)

    if stripped == "pepper" and "ground" in n and "hot" not in n:
        return _result(PIPER_NIGRUM, "Modifier-stripped to pepper after ground -> Piper nigrum (139)", n)

    return None


def _result(
    food_id: int,
    reasoning: str,
    ingredient_norm: str,
    judgment_call: bool = False,
    judgment_note: str | None = None,
) -> dict:
    names = {
        PIPER_NIGRUM: ("Pepper (Spice)", "Piper nigrum"),
        CAPSICUM: ("Pepper", "Capsicum annuum"),
        CHILI: ("Chili", None),
    }
    species, latin = names[food_id]
    return {
        "proposed_foodb_id": food_id,
        "proposed_species": species,
        "latin_name": latin,
        "reasoning": reasoning,
        "judgment_call": judgment_call,
        "judgment_note": judgment_note,
        "ingredient_norm": ingredient_norm,
    }
