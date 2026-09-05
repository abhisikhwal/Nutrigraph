"""Convert parsed units to grams using USDA food_portion + category fallbacks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Direct mass conversions (to grams)
MASS_TO_GRAM: dict[str, float] = {
    "gram": 1.0,
    "g": 1.0,
    "kilogram": 1000.0,
    "kg": 1000.0,
    "ounce": 28.3495,
    "oz": 28.3495,
    "pound": 453.592,
    "lb": 453.592,
    "lbs": 453.592,
    "milliliter": 1.0,  # water-like default; overridden by FDC portion when available
    "ml": 1.0,
    "liter": 1000.0,
    "l": 1000.0,
    "trace": 1.0,  # amount already in grams from trace policy
    "pinch": 1.0,
    "dash": 1.0,
    "handful": 1.0,
    "garnish": 1.0,
}

# Category defaults when FDC-specific portion missing (grams per 1 unit)
CATEGORY_UNIT_GRAMS: dict[str, dict[str, float]] = {
    "spice": {"teaspoon": 2.5, "tablespoon": 7.5, "cup": 100.0, "clove": 3.0},
    "liquid": {"cup": 240.0, "tablespoon": 15.0, "teaspoon": 5.0, "milliliter": 1.0},
    "grain": {"cup": 185.0, "tablespoon": 12.0},
    "legume_dry": {"cup": 192.0, "tablespoon": 12.0},
    "legume_cooked": {"cup": 198.0, "tablespoon": 12.3},
    "flour": {"cup": 125.0, "tablespoon": 8.0},
    "sugar": {"cup": 200.0, "tablespoon": 12.5},
    "fat": {"cup": 227.0, "tablespoon": 14.0, "teaspoon": 4.7},
    "vegetable": {"cup": 130.0, "head": 600.0, "bunch": 100.0},
    "egg": {"egg": 50.0, "piece": 50.0},
    "generic": {"cup": 240.0, "tablespoon": 15.0, "teaspoon": 5.0, "slice": 30.0,
                "piece": 50.0, "clove": 3.0, "can": 400.0, "package": 200.0},
}

UNIT_TOKENS = re.compile(
    r"\b(cup|cups|tbsp|tablespoon|tsp|teaspoon|oz|ounce|lb|pound|"
    r"clove|cloves|slice|slices|piece|pieces|egg|eggs|can|head|bunch|sprig|stick)\b",
    re.I,
)


@dataclass
class MassResult:
    ingredient_id: str | None
    ingredient_name: str
    mass_g: float | None
    conversion_method: str
    conversion_notes: list[str] = field(default_factory=list)
    fdc_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingredient_id": self.ingredient_id,
            "ingredient_name": self.ingredient_name,
            "mass_g": self.mass_g,
            "conversion_method": self.conversion_method,
            "conversion_notes": self.conversion_notes,
            "fdc_id": self.fdc_id,
        }


class MassConverter:
    """FDC-linked unit → gram conversion."""

    def __init__(
        self,
        portion_df: pd.DataFrame,
        fdc_by_ingredient: dict[str, int],
        ingredient_categories: dict[str, str] | None = None,
    ) -> None:
        self.fdc_by_ingredient = fdc_by_ingredient
        self.ingredient_categories = ingredient_categories or {}
        self.portion_index: dict[int, dict[str, float]] = {}
        self._build_portion_index(portion_df)

    def _build_portion_index(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            try:
                fid = int(row["fdc_id"])
            except (ValueError, TypeError):
                continue
            amount = float(row.get("amount") or 1.0) or 1.0
            grams = float(row.get("gram_weight") or 0)
            if grams <= 0:
                continue
            per_unit = grams / amount
            units: list[str] = []
            mu = str(row.get("name", "") or "").strip().lower()
            if mu and mu != "undetermined":
                units.append(mu)
            mod = str(row.get("modifier") or "").strip().lower()
            if mod:
                units.append(mod)
                for tok in UNIT_TOKENS.findall(mod):
                    units.append(tok.lower())
            desc = str(row.get("portion_description") or "").strip().lower()
            if desc:
                for tok in UNIT_TOKENS.findall(desc):
                    units.append(tok.lower())
            bucket = self.portion_index.setdefault(fid, {})
            for u in units:
                key = _norm_unit(u)
                if key and key not in bucket:
                    bucket[key] = per_unit

    @staticmethod
    def coverage_report(
        ingredient_ids: list[str],
        fdc_by_ingredient: dict[str, int],
        portion_index: dict[int, dict[str, float]],
    ) -> dict[str, Any]:
        has_fdc = [i for i in ingredient_ids if fdc_by_ingredient.get(i)]
        with_portion = [
            i for i in has_fdc
            if portion_index.get(int(fdc_by_ingredient[i]))
        ]
        volume_units = {"cup", "tablespoon", "teaspoon", "ounce", "clove", "egg", "slice"}
        with_volume = [
            i for i in has_fdc
            if volume_units & set(portion_index.get(int(fdc_by_ingredient[i]), {}))
        ]
        return {
            "n_ingredients": len(ingredient_ids),
            "n_with_fdc": len(has_fdc),
            "n_with_any_portion": len(with_portion),
            "n_with_volume_or_count_portion": len(with_volume),
            "fdc_coverage_pct": round(100 * len(has_fdc) / max(len(ingredient_ids), 1), 2),
            "portion_coverage_pct": round(100 * len(with_portion) / max(len(ingredient_ids), 1), 2),
            "volume_count_portion_pct": round(100 * len(with_volume) / max(len(ingredient_ids), 1), 2),
        }

    def convert(
        self,
        ingredient_id: str | None,
        ingredient_name: str,
        amount: float | None,
        unit: str | None,
    ) -> MassResult:
        notes: list[str] = []
        if amount is None:
            return MassResult(
                ingredient_id, ingredient_name, None, "unconvertible",
                ["no amount parsed"], self.fdc_by_ingredient.get(ingredient_id or ""),
            )
        unit_key = _norm_unit(unit or "")
        if unit_key in MASS_TO_GRAM and unit_key not in {"trace", "pinch", "dash", "handful", "garnish"}:
            mass = amount * MASS_TO_GRAM[unit_key]
            return MassResult(
                ingredient_id, ingredient_name, round(mass, 3), "direct_mass",
                notes, self.fdc_by_ingredient.get(ingredient_id or ""),
            )
        if unit_key in {"trace", "pinch", "dash", "handful", "garnish"}:
            return MassResult(
                ingredient_id, ingredient_name, round(amount, 3), "trace_nominal",
                ["vague quantity nominal grams"], self.fdc_by_ingredient.get(ingredient_id or ""),
            )

        fdc_id = self.fdc_by_ingredient.get(ingredient_id or "")
        if fdc_id:
            per_g = self.portion_index.get(int(fdc_id), {}).get(unit_key)
            if per_g:
                return MassResult(
                    ingredient_id, ingredient_name, round(amount * per_g, 3),
                    "fdc_portion", [f"FDC {fdc_id} {unit_key}={per_g}g"], int(fdc_id),
                )

        cat = self.ingredient_categories.get(ingredient_id or "", "generic")
        cat_table = CATEGORY_UNIT_GRAMS.get(cat, CATEGORY_UNIT_GRAMS["generic"])
        if unit_key in cat_table:
            notes.append(f"category fallback ({cat})")
            return MassResult(
                ingredient_id, ingredient_name, round(amount * cat_table[unit_key], 3),
                "category_fallback", notes, int(fdc_id) if fdc_id else None,
            )
        if unit_key in CATEGORY_UNIT_GRAMS["generic"]:
            notes.append("generic unit fallback")
            return MassResult(
                ingredient_id, ingredient_name,
                round(amount * CATEGORY_UNIT_GRAMS["generic"][unit_key], 3),
                "generic_fallback", notes, int(fdc_id) if fdc_id else None,
            )
        return MassResult(
            ingredient_id, ingredient_name, None, "unconvertible",
            [f"no conversion for unit '{unit_key}'"], int(fdc_id) if fdc_id else None,
        )


def _norm_unit(u: str) -> str:
    u = u.strip().lower().rstrip(".")
    aliases = {
        "c": "cup", "c.": "cup", "cups": "cup",
        "tbsp": "tablespoon", "tbsp.": "tablespoon", "tablespoons": "tablespoon",
        "tsp": "teaspoon", "tsp.": "teaspoon", "teaspoons": "teaspoon",
        "oz": "ounce", "oz.": "ounce", "ounces": "ounce",
        "lb": "pound", "lbs": "pound", "pounds": "pound",
        "g": "gram", "grams": "gram", "kg": "kilogram",
        "ml": "milliliter", "cloves": "clove", "eggs": "egg",
        "slices": "slice", "pieces": "piece", "sprigs": "sprig", "sticks": "stick",
        "cans": "can",
    }
    return aliases.get(u, u)


def infer_category(canonical_name: str, node_type: str | None = None) -> str:
    n = canonical_name.lower()
    if any(x in n for x in ("turmeric", "cumin", "coriander", "pepper", "spice", "clove", "cardamom", "masala")):
        return "spice"
    if any(x in n for x in ("lentil", "dal", "bean", "pea", "chickpea")):
        return "legume_dry" if "raw" in n or "dry" in n else "legume_cooked"
    if any(x in n for x in ("flour", "wheat")):
        return "flour"
    if any(x in n for x in ("sugar", "honey", "syrup")):
        return "sugar"
    if any(x in n for x in ("oil", "butter", "ghee", "fat")):
        return "fat"
    if "egg" in n:
        return "egg"
    if any(x in n for x in ("milk", "water", "broth", "sauce", "juice")):
        return "liquid"
    if any(x in n for x in ("rice", "oat", "grain")):
        return "grain"
    return "generic"
