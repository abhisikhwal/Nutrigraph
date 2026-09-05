"""FDA Daily Value reference and nutrient aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# FDA Daily Values (2016 food label, general adult population)
# Source: 21 CFR 101.9 / FDA Labeling Nutrient Content Claims
FDA_DAILY_VALUES: dict[str, dict[str, Any]] = {
    "Energy": {"nutrient_id": 1008, "dv_amount": 2000.0, "unit": "KCAL", "label": "Calories"},
    "Total Fat": {"nutrient_id": 1004, "dv_amount": 78.0, "unit": "G", "label": "Total fat"},
    "Saturated Fat": {"nutrient_id": 1258, "dv_amount": 20.0, "unit": "G", "label": "Saturated fat"},
    "Cholesterol": {"nutrient_id": 1253, "dv_amount": 300.0, "unit": "MG", "label": "Cholesterol"},
    "Sodium, Na": {"nutrient_id": 1093, "dv_amount": 2300.0, "unit": "MG", "label": "Sodium"},
    "Carbohydrate, by difference": {"nutrient_id": 1005, "dv_amount": 275.0, "unit": "G", "label": "Carbohydrate"},
    "Fiber, total dietary": {"nutrient_id": 1079, "dv_amount": 28.0, "unit": "G", "label": "Dietary fiber"},
    "Protein": {"nutrient_id": 1003, "dv_amount": 50.0, "unit": "G", "label": "Protein"},
    "Vitamin D (D2 + D3)": {"nutrient_id": 1114, "dv_amount": 20.0, "unit": "UG", "label": "Vitamin D"},
    "Calcium, Ca": {"nutrient_id": 1087, "dv_amount": 1300.0, "unit": "MG", "label": "Calcium"},
    "Iron, Fe": {"nutrient_id": 1089, "dv_amount": 18.0, "unit": "MG", "label": "Iron"},
    "Potassium, K": {"nutrient_id": 1092, "dv_amount": 4700.0, "unit": "MG", "label": "Potassium"},
    "Vitamin C, total ascorbic acid": {"nutrient_id": 1162, "dv_amount": 90.0, "unit": "MG", "label": "Vitamin C"},
    "Folate, DFE": {"nutrient_id": 1190, "dv_amount": 400.0, "unit": "UG", "label": "Folate (DFE)"},
    "Vitamin A, RAE": {"nutrient_id": 1106, "dv_amount": 900.0, "unit": "UG", "label": "Vitamin A"},
    "Vitamin B-12": {"nutrient_id": 1178, "dv_amount": 2.4, "unit": "UG", "label": "Vitamin B12"},
    "Magnesium, Mg": {"nutrient_id": 1090, "dv_amount": 420.0, "unit": "MG", "label": "Magnesium"},
    "Zinc, Zn": {"nutrient_id": 1095, "dv_amount": 11.0, "unit": "MG", "label": "Zinc"},
}

# Aliases for joining nutrient names from profiles
NUTRIENT_NAME_ALIASES: dict[str, str] = {
    "Iron, Fe": "Iron, Fe",
    "Iron": "Iron, Fe",
    "Protein": "Protein",
    "Carbohydrate, by difference": "Carbohydrate, by difference",
    "Fiber, total dietary": "Fiber, total dietary",
    "Sodium, Na": "Sodium, Na",
    "Potassium, K": "Potassium, K",
    "Folate, DFE": "Folate, DFE",
    "Folate, total": "Folate, DFE",
    "Vitamin C, total ascorbic acid": "Vitamin C, total ascorbic acid",
}


@dataclass
class NutrientPanel:
    basis: str  # per_recipe | per_serving
    n_servings: float | None
    servings_known: bool
    totals: dict[str, float] = field(default_factory=dict)
    percent_dv: dict[str, float] = field(default_factory=dict)
    source: str = "FDA Daily Values (2016 label, 21 CFR 101.9)"

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for key, pct in sorted(self.percent_dv.items(), key=lambda x: -x[1]):
            meta = FDA_DAILY_VALUES.get(key, {})
            rows.append({
                "nutrient": meta.get("label", key),
                "nutrient_key": key,
                "total_amount": round(self.totals.get(key, 0), 3),
                "dv_amount": meta.get("dv_amount"),
                "unit": meta.get("unit"),
                "percent_dv": round(pct, 1),
            })
        return {
            "basis": self.basis,
            "n_servings": self.n_servings,
            "servings_known": self.servings_known,
            "source": self.source,
            "nutrients": rows,
        }


def build_nutrient_lookup(nutrient_df: pd.DataFrame) -> dict[str, dict[int, dict[str, Any]]]:
    """ingredient_id -> nutrient_id -> {amount per 100g, unit, name}."""
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for _, r in nutrient_df.iterrows():
        sid = str(r["species_id"])
        nid = int(r["nutrient_id"])
        out.setdefault(sid, {})[nid] = {
            "amount": float(r["amount"]),
            "unit": str(r["unit"]),
            "name": str(r["nutrient_name"]),
        }
    return out


def _sanitize_nutrient_amount(name: str, amount: float, unit: str) -> tuple[float, str]:
    """Fix profile nutrition rows where mg/kcal minerals were stored with wrong unit labels."""
    u = unit.upper()
    n = name.lower()
    if ("calor" in n or "energy" in n) and u == "G" and amount > 20:
        return amount, "KCAL"
    if "cholesterol" in n and u == "G":
        return amount, "MG"
    mineral = any(x in n for x in ("iron", "sodium", "calcium", "potassium", "magnesium", "zinc"))
    if mineral and u == "G" and amount > 1.0:
        return amount, "MG"
    if mineral and u == "G" and amount > 0.05:
        return amount * 1000, "MG"
    return amount, u


def compute_nutrient_panel(
    ingredients: list[dict[str, Any]],
    nutrient_lookup: dict[str, dict[int, dict[str, Any]]],
    n_servings: float | None = None,
) -> NutrientPanel:
    """Sum mass-scaled nutrients; return % DV."""
    totals_by_name: dict[str, float] = {}
    for ing in ingredients:
        iid = ing.get("ingredient_id")
        mass = ing.get("mass_g")
        if not iid or mass is None or mass <= 0:
            continue
        per100 = nutrient_lookup.get(iid, {})
        for nid, meta in per100.items():
            dv_key = _match_dv_key(meta["name"], nid)
            if not dv_key:
                continue
            amt = meta["amount"] * (mass / 100.0)
            amt, unit = _sanitize_nutrient_amount(dv_key, amt, meta["unit"])
            totals_by_name[dv_key] = totals_by_name.get(dv_key, 0.0) + _normalize_amount(amt, unit, FDA_DAILY_VALUES[dv_key]["unit"])

    servings_known = n_servings is not None and n_servings > 0
    divisor = n_servings if servings_known else 1.0
    basis = "per_serving" if servings_known else "per_recipe"

    pct_dv: dict[str, float] = {}
    display_totals: dict[str, float] = {}
    for key, total in totals_by_name.items():
        scaled = total / divisor
        display_totals[key] = scaled
        dv = FDA_DAILY_VALUES[key]["dv_amount"]
        pct_dv[key] = 100.0 * scaled / dv if dv else 0.0

    return NutrientPanel(
        basis=basis,
        n_servings=n_servings,
        servings_known=servings_known,
        totals=display_totals,
        percent_dv=pct_dv,
    )


def _match_dv_key(name: str, nutrient_id: int) -> str | None:
    for key, meta in FDA_DAILY_VALUES.items():
        if meta["nutrient_id"] == nutrient_id:
            return key
    alias = NUTRIENT_NAME_ALIASES.get(name)
    if alias and alias in FDA_DAILY_VALUES:
        return alias
    if name in FDA_DAILY_VALUES:
        return name
    return None


def _normalize_amount(amount: float, from_unit: str, to_unit: str) -> float:
    fu, tu = from_unit.upper(), to_unit.upper()
    if fu == tu:
        return amount
    # common conversions
    if fu in {"G", "GRAM", "GRAMS"} and tu == "MG":
        return amount * 1000
    if fu == "MG" and tu == "G":
        return amount / 1000
    if fu in {"UG", "MCG"} and tu == "MG":
        return amount / 1000
    if fu == "MG" and tu in {"UG", "MCG"}:
        return amount * 1000
    if fu == "KCAL" and tu == "G":
        return amount  # ignore mislabeled energy
    return amount
