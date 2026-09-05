"""Unified dose engine for batch recipes and live user input."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "scripts/product") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts/product"))

from dose.quantity_parser import ParsedQuantity, parse_ingredient_line
from dose.mass_converter import MassConverter, MassResult, infer_category
from dose.nutrient_rdi import (
    FDA_DAILY_VALUES,
    NutrientPanel,
    _sanitize_nutrient_amount,
    build_nutrient_lookup,
    compute_nutrient_panel,
)
from dose.mechanism_contribution import (
    HeroAnalysis,
    compute_mechanism_contribution,
    load_theme_context,
)

DEFAULT_PROFILES = ROOT / "data/processed/product/ingredient_profiles_v2.jsonl"
DEFAULT_LOOKUP = ROOT / "data/processed/product/indexes/ingredient_lookup.json"
DEFAULT_NUTRIENTS = ROOT / "data/processed/product/nutrients/species_nutrient_profiles_production.parquet"
DEFAULT_FDC_MAP = ROOT / "data/processed/product/nutrients/species_fdc_map_production.parquet"
DEFAULT_PORTION = ROOT / "data/processed/product/nutrients/food_portion.parquet"
DEFAULT_INDEX = ROOT / "data/processed/product/indexes"


@dataclass
class IngredientDose:
    raw: str
    parsed: ParsedQuantity
    ingredient_id: str | None
    canonical_name: str | None
    mass: MassResult
    match_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "parsed": self.parsed.to_dict(),
            "ingredient_id": self.ingredient_id,
            "canonical_name": self.canonical_name,
            "mass_g": self.mass.mass_g,
            "mass_conversion": self.mass.to_dict(),
            "match_type": self.match_type,
        }


@dataclass
class RecipeDoseResult:
    recipe_id: str
    recipe_label: str
    source: str
    n_servings: float | None
    servings_known: bool
    ingredients: list[IngredientDose] = field(default_factory=list)
    nutrient_panel: NutrientPanel | None = None
    mechanism: HeroAnalysis | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_label": self.recipe_label,
            "source": self.source,
            "n_servings": self.n_servings,
            "servings_known": self.servings_known,
            "ingredients": [i.to_dict() for i in self.ingredients],
            "nutrient_panel": self.nutrient_panel.to_dict() if self.nutrient_panel else None,
            "mechanism": self.mechanism.to_dict() if self.mechanism else None,
            "warnings": self.warnings,
        }


class DoseEngine:
    """parse → resolve → mass → RDI + relative mechanism contribution."""

    def __init__(
        self,
        profiles_path: Path = DEFAULT_PROFILES,
        lookup_path: Path = DEFAULT_LOOKUP,
        nutrient_path: Path = DEFAULT_NUTRIENTS,
        fdc_map_path: Path = DEFAULT_FDC_MAP,
        portion_path: Path = DEFAULT_PORTION,
        index_dir: Path = DEFAULT_INDEX,
    ) -> None:
        self.profiles: dict[str, dict[str, Any]] = {}
        with profiles_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    p = json.loads(line)
                    iid = p["ingredient"].get("ingredient_id", p["ingredient"]["species_id"])
                    self.profiles[iid] = p

        lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
        self.aliases: dict[str, str] = lookup.get("aliases", {})
        self.by_id = lookup.get("by_ingredient_id", lookup.get("by_species_id", {}))

        # FDC map: SP from production + ING fdc_id from profiles
        fdc_by_ing: dict[str, int] = {}
        if fdc_map_path.exists():
            fmap = pd.read_parquet(fdc_map_path)
            for _, r in fmap.iterrows():
                if pd.notna(r.get("fdc_id")):
                    fdc_by_ing[str(r["species_id"])] = int(r["fdc_id"])
        for iid, prof in self.profiles.items():
            fdc = prof["ingredient"].get("fdc_id")
            if fdc and iid not in fdc_by_ing:
                fdc_by_ing[iid] = int(fdc)

        categories = {
            iid: infer_category(p["ingredient"]["canonical_name"], p["ingredient"].get("node_type"))
            for iid, p in self.profiles.items()
        }
        portion_df = pd.read_parquet(portion_path) if portion_path.exists() else pd.DataFrame()
        self.mass_converter = MassConverter(portion_df, fdc_by_ing, categories)
        self.fdc_by_ingredient = fdc_by_ing
        self.portion_index = self.mass_converter.portion_index

        nutr_df = pd.read_parquet(nutrient_path) if nutrient_path.exists() else pd.DataFrame()
        self.nutrient_lookup = build_nutrient_lookup(nutr_df)
        # extend ING nodes from profile nutrition if missing in production parquet
        self._extend_nutrients_from_profiles()

        self.theme_meta, self.theme_index = load_theme_context(index_dir)

    def _extend_nutrients_from_profiles(self) -> None:
        name_to_id = {v.get("label", k): k for k, v in FDA_DAILY_VALUES.items()}
        name_to_id.update({k: k for k in FDA_DAILY_VALUES})
        for iid, prof in self.profiles.items():
            if iid in self.nutrient_lookup:
                continue
            nutr = prof.get("nutrition", {})
            if not nutr.get("available"):
                continue
            bucket: dict[int, dict[str, Any]] = {}
            for n in nutr.get("top_nutrients", []):
                nm = str(n.get("nutrient_name", ""))
                matched = name_to_id.get(nm) or nm
                meta = FDA_DAILY_VALUES.get(matched)
                if meta:
                    amt, unit = _sanitize_nutrient_amount(meta.get("label", nm), float(n.get("amount") or 0), str(n.get("unit", meta["unit"])))
                    bucket[meta["nutrient_id"]] = {
                        "amount": amt,
                        "unit": unit,
                        "name": meta.get("label", nm),
                    }
            if bucket:
                self.nutrient_lookup[iid] = bucket

    def resolve_ingredient(self, name: str) -> tuple[str | None, str]:
        key = re.sub(r"\s+", " ", name.strip().lower())
        iid = self.aliases.get(key)
        if iid and iid in self.profiles:
            return iid, "exact"
        for alias, cand in self.aliases.items():
            if len(alias) >= 4 and (key in alias or alias in key):
                if cand in self.profiles:
                    return cand, "alias_fuzzy"
        return None, "unmapped"

    def analyze_ingredient_lines(
        self,
        lines: list[str],
        recipe_id: str = "live",
        recipe_label: str = "User recipe",
        source: str = "live_input",
        n_servings: float | None = None,
    ) -> RecipeDoseResult:
        """Core path for live pasted recipes and batch adapters."""
        doses: list[IngredientDose] = []
        ing_for_nutrients: list[dict[str, Any]] = []
        warnings: list[str] = []

        for line in lines:
            parsed = parse_ingredient_line(line)
            iid, match_type = self.resolve_ingredient(parsed.ingredient_name)
            cname = None
            if iid:
                cname = self.profiles[iid]["ingredient"]["canonical_name"]
            mass = self.mass_converter.convert(iid, parsed.ingredient_name, parsed.amount, parsed.unit)
            if mass.mass_g is None:
                warnings.append(f"Unconverted: {line}")
            doses.append(IngredientDose(
                raw=line, parsed=parsed, ingredient_id=iid, canonical_name=cname,
                mass=mass, match_type=match_type,
            ))
            ing_for_nutrients.append({
                "ingredient_id": iid,
                "mass_g": mass.mass_g,
                "canonical_name": cname,
            })

        panel = compute_nutrient_panel(ing_for_nutrients, self.nutrient_lookup, n_servings)
        if not panel.servings_known:
            warnings.append("Serving count unknown — nutrient panel reported per-recipe.")

        mech = compute_mechanism_contribution(
            ing_for_nutrients, self.profiles, self.theme_meta, self.theme_index,
        )

        return RecipeDoseResult(
            recipe_id=recipe_id,
            recipe_label=recipe_label,
            source=source,
            n_servings=n_servings,
            servings_known=panel.servings_known,
            ingredients=doses,
            nutrient_panel=panel,
            mechanism=mech,
            warnings=warnings,
        )

    def coverage_report(self) -> dict[str, Any]:
        ids = list(self.profiles.keys())
        mass_cov = MassConverter.coverage_report(ids, self.fdc_by_ingredient, self.portion_index)
        mass_cov["note"] = (
            "Volume/count conversion uses FDC food_portion when available; "
            "category/generic fallbacks otherwise."
        )
        return {
            "universe_nodes": len(ids),
            "fdc_map": mass_cov,
            "rdi_source": "FDA Daily Values (2016 label, 21 CFR 101.9)",
        }
