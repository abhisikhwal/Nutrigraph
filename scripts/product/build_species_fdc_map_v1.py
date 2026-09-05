#!/usr/bin/env python3
"""
Stage 1 — species → USDA FDC mapping with human-review gate.

Phases A–C (default): clean pool, rule templates, classification → review artifacts.
Phase D (--phase compose): join approved map to food_nutrient (after human approval).

Outputs (new layer only; does not touch mechanism graph / profiles / indexes):
  data/processed/product/nutrients/fdc_clean_pool_v1.parquet
  data/processed/product/nutrients/fdc_description_patterns_v1.json
  data/processed/product/nutrients/species_fdc_mapping_review_v1.json
  data/processed/product/nutrients/species_fdc_auto_confident_v1.parquet
  data/processed/product/nutrients/species_fdc_mapping_report_v1.json

After approval, also:
  data/processed/product/nutrients/species_fdc_map_approved_v1.parquet  (human-edited)
  data/processed/product/nutrients/species_nutrient_profiles_v1.parquet
  data/processed/product/nutrients/species_nutrient_summary_v1.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/product/nutrients"
SPECIES_PATH = ROOT / "data/processed/canonical/species_nodes_v2.parquet"
USDA_ZIP = ROOT / "data/raw/usda_fooddata.zip"
USDA_PREFIX = "FoodData_Central_csv_2024-10-31/"
FOOD_NUTRIENT_PARQUET = ROOT / "data/raw/phase9_expansion/usda_food_nutrient.parquet"
REVIEW_PATH = OUT_DIR / "species_fdc_mapping_review_v1.json"
APPROVED_PATH = OUT_DIR / "species_fdc_map_approved_v1.parquet"

VARIANT_SELECTION_POLICY = (
    "When multiple FDC entries match a species, prefer the raw/base form (e.g. ', raw', "
    "'mature seeds, raw', 'Spices, X, ground') over cooked, canned, frozen, or salted variants. "
    "Foundation Foods rank above SR Legacy when scores tie. "
    "Applied defaults are flagged default_raw_preference=true for audit."
)

# Canonical name → primary FDC search term(s)
NAME_TO_FDC_TERMS: dict[str, list[str]] = {
    "Garden onion": ["onions"],
    "Kiwi": ["kiwifruit"],
    "Soy bean": ["soybeans"],
    "Common bean": ["beans"],
    "Common pea": ["peas"],
    "Garden tomato": ["tomatoes"],
    "Cherry tomato": ["tomatoes"],
    "Brussel sprouts": ["brussels sprouts"],
    "Swede": ["rutabaga"],
    "Cashew nut": ["cashew nuts"],
    "Pecan nut": ["pecans"],
    "Brazil nut": ["brazilnuts"],
    "Peanut": ["peanuts"],
    "Ceylon cinnamon": ["cinnamon"],
    "Star anise": ["anise seed"],
    "Black walnut": ["walnuts"],
    "Sweet bay": ["bay leaf"],
    "Rocket salad": ["arugula"],
    "Rapini": ["broccoli raab"],
    "Daikon radish": ["radishes"],
    "Black radish": ["radishes"],
    "Celery stalks": ["celery"],
    "Chicory leaves": ["chicory"],
    "White cabbage": ["cabbage"],
    "Black cabbage": ["cabbage"],
    "Chinese cabbage": ["cabbage", "pak-choi"],
    "Savoy cabbage": ["cabbage", "savoy"],
    "Green onion": ["onions"],
    "Red onion": ["onions"],
    "Green bell pepper": ["peppers, sweet"],
    "Red bell pepper": ["peppers, sweet"],
    "Yellow bell pepper": ["peppers, sweet"],
    "Orange bell pepper": ["peppers, sweet"],
    "Cubanelle pepper": ["peppers, sweet"],
    "Jalapeno pepper": ["peppers, jalapeno"],
    "Pepper (Spice)": ["spices, pepper"],
    "Green apple": ["apples"],
    "Green grape": ["grapes"],
    "Red grape": ["grapes"],
    "Highbush blueberry": ["blueberries"],
    "Red raspberry": ["raspberries"],
    "Black raspberry": ["raspberries"],
    "American cranberry": ["cranberries"],
    "Lingonberry": ["cranberries"],
    "Asian pear": ["pears"],
    "Yellow zucchini": ["squash, summer"],
    "Japanese pumpkin": ["pumpkin"],
    "Broad bean": ["fava beans"],
    "Black-eyed pea": ["cowpeas"],
    "Common bean": ["beans, kidney, all types, mature seeds, raw"],
    "Green lentil": ["lentils"],
    "Yellow wax bean": ["beans, snap"],
    "Green bean": ["beans, snap"],
    "Pea shoots": ["peas"],
    "Wild leek": ["leeks"],
    "Chinese chives": ["chives"],
    "Milk (Cow)": ["milk, whole"],
    "Cattle (Beef, Veal)": ["beef, composite"],
    "Domestic pig": ["pork, fresh, loin"],
    "Salmonidae (Salmon, Trout)": ["fish, salmon"],
    "Common mushroom": ["mushrooms"],
    "Breadfruit": ["breadfruit"],
    "Other bread": ["bread"],
    "Red tea": ["beverages, tea, black, brewed"],
    "Black tea": ["beverages, tea, black, brewed"],
    "Green tea": ["beverages, tea, green, brewed"],
    "Herbal tea": ["beverages, tea, herb"],
    "Grape wine": ["alcoholic beverage, wine, table, all"],
    "Whisky": ["alcoholic beverage, distilled, whiskey"],
    "Port wine": ["alcoholic beverage, wine, dessert"],
    "Plain cream cheese": ["cheese, cream"],
    "Greek feta cheese": ["cheese, feta"],
    "Processed cheese": ["cheese, pasteurized process"],
    "Rapeseed oil": ["oil, canola"],
    "Cooking oil": ["oil, olive"],
    "Sour orange": ["oranges"],
    "Root vegetables": ["turnips"],
    "Other candy": ["candies"],
    "Baby food": ["babyfood"],
    "Coffee": ["beverages, coffee, brewed"],
    "Banana": ["bananas"],
    "Lentils": ["lentils"],
    "Common pea": ["peas, green"],
    "Shallot": ["onions, raw"],
    "Almond": ["nuts, almonds"],
    "Coconut": ["nuts, coconut meat, raw"],
    "Peanut": ["peanuts"],
    "Shrimp": ["crustaceans, shrimp"],
    "Scallop": ["mollusks, scallop"],
    "Rabbit": ["game meat, rabbit"],
    "Green zucchini": ["squash, summer, zucchini"],
    "Parsnip": ["parsnips"],
    "Leek": ["leeks, raw"],
    "Walnut": ["nuts, walnuts"],
    "Hazelnut": ["nuts, hazelnuts"],
    "Pistachio": ["nuts, pistachio"],
    "Flaxseed": ["seeds, flaxseed"],
    "Lima bean": ["beans, lima, large, mature seeds, raw"],
    "Saffron": ["spices, saffron"],
    "Sweet bay": ["spices, bay leaf"],
    "Cumin": ["spices, cumin seed"],
    "Cardamom": ["spices, cardamom"],
}

SPICE_HERB_NAMES = {
    "turmeric", "cumin", "coriander", "cardamom", "saffron", "nutmeg", "clove", "cloves",
    "cinnamon", "ginger", "horseradish", "wasabi", "paprika", "chili", "fenugreek", "anise",
    "caraway", "dill", "tarragon", "thyme", "sage", "basil", "oregano", "marjoram", "rosemary",
    "bay", "mint", "parsley", "chervil", "lavender", "lemongrass", "mace", "mustard", "poppy",
    "savory", "sumac", "vanilla", "allspice", "capers", "chives", "garlic", "hops", "mugwort",
    "pepper", "safflower", "sesame", "spearmint", "star anise", "sweet basil", "common thyme",
    "common sage", "common oregano", "mexican oregano", "italian oregano", "sweet marjoram",
    "lemon balm", "lemon verbena", "lemon thyme", "sweet bay", "ceylon cinnamon", "black mustard",
    "white mustard", "chinese mustard", "summer savory", "winter savory", "curry", "galangal",
}

JUDGMENT_SPECIES_IDS: set[str] = set()  # filled at runtime for meat groups etc.

MEAT_REPRESENTATIVE_DEFAULTS: dict[str, dict[str, Any]] = {
    "SP_000237": {
        "fdc_pattern": "Beef, composite of trimmed retail cuts, separable lean only, trimmed to 0\" fat, all grades, raw",
        "rationale": "Species-level Bos taurus → composite retail lean cut (generic beef).",
    },
    "SP_000259": {
        "fdc_pattern": "Pork, fresh, loin, whole, separable lean only, raw",
        "rationale": "Species-level Sus scrofa → loin as representative lean cut.",
    },
    "SP_000189": {
        "fdc_pattern": "Chicken, broilers or fryers, breast, meat only, raw",
        "rationale": "Gallus gallus → boneless breast as common reference cut.",
    },
    "SP_000235": {
        "fdc_pattern": "Fish, salmon, Atlantic, wild, raw",
        "rationale": "Family-level Salmonidae → Atlantic salmon as proxy (explicit tradeoff).",
    },
    "SP_000030": {
        "fdc_pattern": "Beverages, tea, black, brewed, prepared with tap water",
        "rationale": "Camellia sinensis → brewed black tea beverage (not dry leaf).",
    },
    "SP_000288": {
        "fdc_pattern": "Milk, whole, 3.25% milkfat, with added vitamin D",
        "rationale": "Generic cow milk → whole milk with standard fortification.",
    },
    "SP_000287": {
        "fdc_pattern": "Cheese, cheddar",
        "rationale": "Generic cheese species → cheddar as reference.",
    },
}


GENERIC_NAME_TOKENS = {
    "garden", "common", "sweet", "sour", "green", "red", "black", "white", "yellow",
    "chinese", "american", "highbush", "wild", "other", "plain", "greek", "italian",
    "mexican", "ceylon", "domestic", "garden", "nut", "spice", "bell", "hot",
}


HEAD_NOUNS = {
    "cabbage", "onion", "tomato", "pepper", "bean", "pea", "lettuce", "apple", "grape",
    "cherry", "berry", "nut", "seed", "fish", "milk", "cheese", "bread", "rice", "wheat",
    "mustard", "orange", "lime", "lemon", "mango", "banana", "potato", "carrot", "squash",
}


def name_modifiers(name: str) -> list[str]:
    toks = [
        t
        for t in norm(name).split()
        if t not in GENERIC_NAME_TOKENS and t not in HEAD_NOUNS and len(t) >= 4
    ]
    return toks


def modifiers_satisfied(name: str, description: str) -> bool:
    mods = name_modifiers(name)
    if not mods:
        return True
    d = norm(description)
    return any(m in d for m in mods)


@dataclass
class FdcCandidate:
    fdc_id: int
    description: str
    data_type: str
    score: float
    rule_id: str
    default_raw_preference: bool = False


@dataclass
class SpeciesMapping:
    species_id: str
    canonical_name: str
    latin_name: str | None
    category: str
    classification: str
    fdc_id: int | None = None
    fdc_description: str | None = None
    fdc_data_type: str | None = None
    rule_id: str | None = None
    default_raw_preference: bool = False
    n_strings: int = 0
    candidates: list[FdcCandidate] = field(default_factory=list)
    judgment_rationale: str | None = None
    notes: str | None = None


def norm(s: str) -> str:
    s = str(s).lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves"):
        return token[:-3] + "f"
    if token.endswith("oes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def infer_category(name: str, latin: str | None, prep_labels: list[str]) -> str:
    n = norm(name)
    latin_l = norm(latin or "")

    if any(x in n for x in ("wine", "whisky", "whiskey", "vermouth", "sherry", "spirit", "cider", "beer", "rum", "vodka", "gin")):
        return "beverage_alcohol"
    if "coffee" in n:
        return "beverage_coffee"
    if "tea" in n or (latin_l and "camellia sinensis" in latin_l):
        return "beverage_tea"
    if any(x in n for x in ("milk", "cream", "yogurt", "butter", "cheese")):
        return "dairy"
    if re.search(r"\begg\b", n) and "eggplant" not in n:
        return "egg"
    if n in SPICE_HERB_NAMES or any(sp in n for sp in ("spice", "herb")) or n in {
        "cumin", "cardamom", "saffron", "nutmeg", "clove", "allspice", "fenugreek", "wasabi",
    }:
        return "spice_herb"
    if any(x in n for x in ("rabbit", "hare")):
        return "game"
    if any(x in n for x in ("beef", "cattle", "veal", "bison", "buffalo", "elk", "venison", "boar", "lamb", "mutton")):
        return "meat"
    if any(x in n for x in ("chicken", "turkey", "duck", "goose", "pheasant", "quail", "hen")):
        return "poultry"
    if "salmonidae" in n or ("fish" in latin_l and "idae" in latin_l):
        return "fish_group"
    if any(x in n for x in (
        "salmon", "trout", "tuna", "cod", "haddock", "sardine", "anchovy", "mackerel",
        "herring", "bass", "perch", "carp", "eel", "garfish", "turbot", "mackerel",
    )):
        return "fish"
    if any(x in n for x in (
        "crab", "lobster", "shrimp", "prawn", "oyster", "mussel", "clam", "squid", "octopus",
        "cuttlefish", "scallop", "crayfish",
    )):
        return "shellfish"
    if "peanut" in n:
        return "nut"
    if any(x in n for x in ("nut", "almond", "walnut", "pecan", "cashew", "hazelnut", "chestnut", "pistachio", "coconut")):
        return "nut"
    if re.search(r"\b(bean|peas?\b|lentil|chickpea|soy)\b", n) and "soybean oil" not in n and "soy sauce" not in n:
        return "legume"
    if any(x in n for x in ("oat", "rice", "wheat", "barley", "rye", "corn", "millet", "quinoa", "sorghum")):
        return "grain"
    if "seed" in n or n in {"sunflower", "safflower", "sesame", "flax", "flaxseed", "pumpkin"}:
        return "seed"
    if any(x in n for x in (
        "apple", "apricot", "avocado", "banana", "berry", "cherry", "grape", "melon", "citrus",
        "orange", "lemon", "lime", "mango", "papaya", "pineapple", "fig", "date", "plum", "peach", "pear",
        "kiwi", "guava", "lychee", "persimmon", "pomegranate", "quince",
    )):
        return "fruit"
    if any(x in n for x in ("bread", "candy", "chip", "marzipan", "remoulade", "junket", "dripping", "biscuit")):
        return "processed"
    if "mushroom" in n or "agaricus" in latin_l or "pleurotus" in latin_l:
        return "mushroom"
    if any(x in n for x in ("kombu", "seaweed", "irish moss", "algae")):
        return "algae"
    return "vegetable"


def search_terms(name: str, prep_labels: list[str]) -> list[str]:
    if name in NAME_TO_FDC_TERMS:
        return NAME_TO_FDC_TERMS[name]
    terms: list[str] = []
    n = norm(name)
    terms.append(n)
    for w in n.split():
        if len(w) >= 3:
            terms.append(singularize(w))
    for p in prep_labels or []:
        terms.append(norm(p.replace("_", " ")))
    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def score_candidate(desc: str, data_type: str, rule_id: str) -> float:
    d = desc.lower()
    score = 0.0
    if rule_id.startswith("template_exact"):
        score += 80
    elif rule_id.startswith("template_"):
        score += 50
    else:
        score += 20
    if re.search(r",\s*raw\b", d) or d.endswith(" raw"):
        score += 45
    # Generic plural raw form (e.g. "Onions, raw", "Bananas, raw")
    if re.search(r",\s*raw$", d) and not any(x in d for x in ("red", "yellow", "white", "sweet", "green", "frozen")):
        score += 20
    if "mature seeds, raw" in d:
        score += 40
    if d.startswith("spices, ") and ("ground" in d or "dried" in d):
        score += 35
    if data_type == "foundation_food":
        score += 12
    if "without salt" in d:
        score -= 5
    if "with salt" in d:
        score -= 20
    if "cooked" in d:
        score -= 35
    if "canned" in d:
        score -= 30
    if "frozen" in d:
        score -= 25
    if any(x in d for x in ("restaurant", "snacks", "babyfood", "fast foods", "kraft", "pillsbury")):
        score -= 80
    if "brand" in d or "zespri" in d or "ataulfo" in d:
        score -= 25
    if "oil" in d and "soybean oil" not in d and rule_id != "template_oil":
        score -= 10
    return score


def apply_rule_templates(category: str, name: str, terms: list[str], clean: pd.DataFrame) -> list[FdcCandidate]:
    cands: dict[int, FdcCandidate] = {}
    cn = norm(name)

    def add_hits(mask: pd.Series, rule_id: str) -> None:
        hits = clean[mask]
        for _, row in hits.iterrows():
            fid = int(row["fdc_id"])
            sc = score_candidate(row["description"], row["data_type"], rule_id)
            prev = cands.get(fid)
            if prev is None or sc > prev.score:
                cands[fid] = FdcCandidate(
                    fdc_id=fid,
                    description=row["description"],
                    data_type=row["data_type"],
                    score=sc,
                    rule_id=rule_id,
                )

    for term in terms:
        t = norm(term)
        if not t:
            continue
        # Full FDC phrase overrides (from NAME_TO_FDC_TERMS)
        if any(term.startswith(p) for p in ("spices,", "nuts,", "beans,", "beverages,", "crustaceans,", "mollusks,", "game meat,", "squash,")):
            add_hits(clean["norm_desc"].str.contains(t, regex=False, na=False), "template_fdc_phrase")
            continue
        if category == "spice_herb":
            for suffix in ("ground", "dried"):
                pat = f"spices, {t}, {suffix}"
                add_hits(clean["norm_desc"] == norm(pat), f"template_spice_{suffix}")
            add_hits(clean["norm_desc"] == norm(f"spices, {t}"), "template_spice_exact")
            add_hits(clean["norm_desc"] == norm(f"spices, {t} seed"), "template_spice_seed")
            add_hits(clean["norm_desc"].str.match(rf"^{re.escape(t)}(,|\s|$)"), "template_spice_name")
        elif category == "nut":
            for prefix in ("nuts, ", "nut butter, "):
                add_hits(
                    clean["norm_desc"].str.contains(rf"^{re.escape(prefix)}{re.escape(t)}", regex=True, na=False),
                    "template_nut",
                )
            if t == "peanut" or t == "peanuts":
                add_hits(clean["norm_desc"].str.contains(norm("peanuts, raw"), regex=False, na=False), "template_peanut_raw")
            if t == "almond" or t == "almonds":
                add_hits(
                    clean["norm_desc"].str.contains(norm("nuts, almonds"), regex=False, na=False),
                    "template_almond",
                )
            if t == "coconut":
                add_hits(
                    clean["norm_desc"].str.contains("^nuts, coconut meat, raw", regex=True, na=False),
                    "template_coconut",
                )
        elif category == "seed":
            add_hits(
                clean["norm_desc"].str.contains(rf"^seeds, {re.escape(t)}", regex=True, na=False),
                "template_seed",
            )
        elif category == "fish":
            add_hits(
                clean["norm_desc"].str.contains(rf"^fish, {re.escape(t)}", regex=True, na=False),
                "template_fish",
            )
        elif category == "shellfish":
            add_hits(
                clean["norm_desc"].str.contains(
                    rf"^(?:crustaceans|mollusks), {re.escape(t)}", regex=True, na=False
                ),
                "template_shellfish",
            )
            if t == "shrimp":
                add_hits(
                    clean["norm_desc"].str.contains("^crustaceans, shrimp", regex=True, na=False),
                    "template_shrimp",
                )
            if t == "scallop":
                add_hits(
                    clean["norm_desc"].str.contains("^mollusks, scallop", regex=True, na=False),
                    "template_scallop",
                )
        elif category == "legume":
            for pat in (
                rf"^{re.escape(t)}, raw$",
                rf"^{re.escape(t)}, mature seeds, raw$",
                rf"^beans, {re.escape(t)}, mature seeds, raw$",
                rf"^{re.escape(t)}, sprouted, raw$",
                rf"^beans, snap, {re.escape(t)}, raw$",
            ):
                add_hits(clean["norm_desc"].str.match(pat, na=False), "template_legume_raw")
            if t in {"bean", "beans", "common bean"} or "bean" in cn:
                add_hits(
                    clean["norm_desc"].str.contains(
                        norm("beans, kidney, all types, mature seeds, raw"), regex=False, na=False
                    ),
                    "template_kidney_bean",
                )
        elif category == "grain":
            add_hits(
                clean["norm_desc"].str.contains(rf"^{re.escape(t)}.*raw", regex=True, na=False),
                "template_grain_raw",
            )
        elif category == "fruit":
            for pat in (
                rf"^{re.escape(t)}, raw$",
                rf"^{re.escape(t)}s, raw$",
                rf"^{re.escape(t)}es, raw$",
            ):
                add_hits(clean["norm_desc"].str.match(pat, na=False), "template_fruit_raw")
            add_hits(
                clean["norm_desc"].str.match(rf"^{re.escape(t)}(,|\s)", na=False),
                "template_produce",
            )
        elif category == "beverage_coffee":
            add_hits(
                clean["norm_desc"].str.contains(norm("beverages, coffee, brewed"), regex=False, na=False),
                "template_coffee",
            )
        elif category in ("vegetable",):
            add_hits(
                clean["norm_desc"].str.match(rf"^{re.escape(t)}, raw$", na=False),
                "template_exact_raw",
            )
            add_hits(
                clean["norm_desc"].str.match(rf"^{re.escape(t)}s, raw$", na=False),
                "template_plural_raw",
            )
            add_hits(
                clean["norm_desc"].str.match(rf"^{re.escape(t)}(,|\s)", na=False),
                "template_produce",
            )
        else:
            add_hits(
                clean["norm_desc"].str.contains(rf"\b{re.escape(t)}\b", regex=True, na=False),
                "template_generic",
            )

    # Extra templates by category
    if category == "vegetable" and "lettuce" in cn:
        add_hits(clean["norm_desc"].str.contains("lettuce.*raw", regex=True, na=False), "template_lettuce")
    if category == "vegetable" and "spinach" in cn:
        add_hits(clean["norm_desc"] == "spinach, raw", "template_exact_raw")

    ranked = sorted(cands.values(), key=lambda c: (-c.score, c.description))
    if ranked:
        top = ranked[0].score
        for c in ranked:
            c.default_raw_preference = c.score >= top - 5 and c.score >= top * 0.85
    return ranked


def classify_species(row: pd.Series, clean: pd.DataFrame) -> SpeciesMapping:
    species_id = row["species_node_id"]
    name = row["canonical_name"]
    latin = row.get("latin_name")
    prep = row.get("preparation_labels")
    if prep is None or (isinstance(prep, float) and pd.isna(prep)):
        prep = []
    elif isinstance(prep, str):
        prep = [prep]
    elif hasattr(prep, "tolist"):
        prep = prep.tolist()
    elif not isinstance(prep, list):
        prep = list(prep) if prep else []
    n_strings = int(row.get("n_strings") or 0)
    category = infer_category(name, latin, prep)

    mapping = SpeciesMapping(
        species_id=species_id,
        canonical_name=name,
        latin_name=latin if pd.notna(latin) else None,
        category=category,
        classification="NO-MATCH",
        n_strings=n_strings,
    )

    # Predefined judgment species with explicit defaults
    if species_id in MEAT_REPRESENTATIVE_DEFAULTS:
        spec = MEAT_REPRESENTATIVE_DEFAULTS[species_id]
        pat = norm(spec["fdc_pattern"])
        hits = clean[clean["norm_desc"] == pat]
        if hits.empty:
            hits = clean[clean["norm_desc"].str.contains(re.escape(pat[:40]), regex=True, na=False)]
        cands = [
            FdcCandidate(
                fdc_id=int(r["fdc_id"]),
                description=r["description"],
                data_type=r["data_type"],
                score=score_candidate(r["description"], r["data_type"], "judgment_default"),
                rule_id="judgment_default",
                default_raw_preference=True,
            )
            for _, r in hits.iterrows()
        ]
        alts = apply_rule_templates(category, name, search_terms(name, prep), clean)[:8]
        mapping.candidates = cands + [c for c in alts if c.fdc_id not in {x.fdc_id for x in cands}]
        mapping.classification = "JUDGMENT"
        mapping.judgment_rationale = spec["rationale"]
        if cands:
            mapping.fdc_id = cands[0].fdc_id
            mapping.fdc_description = cands[0].description
            mapping.fdc_data_type = cands[0].data_type
            mapping.rule_id = "judgment_default"
            mapping.default_raw_preference = True
        return mapping

    # Other judgment categories: meat/poultry/fish_group/dairy/game at species level
    if category in ("meat", "poultry", "fish_group", "dairy", "beverage_tea", "beverage_alcohol", "processed", "game"):
        cands = apply_rule_templates(category, name, search_terms(name, prep), clean)
        mapping.candidates = cands[:15]
        mapping.classification = "JUDGMENT"
        mapping.judgment_rationale = (
            f"Category '{category}' requires representative/generic FDC entry choice at species level."
        )
        if cands:
            mapping.fdc_id = cands[0].fdc_id
            mapping.fdc_description = cands[0].description
            mapping.fdc_data_type = cands[0].data_type
            mapping.rule_id = cands[0].rule_id
            mapping.default_raw_preference = cands[0].default_raw_preference
        else:
            mapping.classification = "NO-MATCH"
        return mapping

    cands = apply_rule_templates(category, name, search_terms(name, prep), clean)
    mapping.candidates = cands[:15]

    if not cands:
        mapping.classification = "NO-MATCH"
        return mapping

    top_tier = [c for c in cands if c.score >= cands[0].score - 5]
    # Require species-specific modifiers when present (e.g. savoy, romaine)
    if len(top_tier) == 1 and not modifiers_satisfied(name, top_tier[0].description):
        mapping.classification = "AMBIGUOUS"
        mapping.fdc_id = top_tier[0].fdc_id
        mapping.fdc_description = top_tier[0].description
        mapping.fdc_data_type = top_tier[0].data_type
        mapping.rule_id = top_tier[0].rule_id
        mapping.default_raw_preference = top_tier[0].default_raw_preference
        mapping.notes = "Single score-tier match but species modifier not in FDC description."
        return mapping
    if len(top_tier) == 1:
        mapping.classification = "AUTO-CONFIDENT"
        mapping.fdc_id = top_tier[0].fdc_id
        mapping.fdc_description = top_tier[0].description
        mapping.fdc_data_type = top_tier[0].data_type
        mapping.rule_id = top_tier[0].rule_id
        mapping.default_raw_preference = top_tier[0].default_raw_preference
    else:
        mapping.classification = "AMBIGUOUS"
        # propose top scorer as default but do not treat as approved
        mapping.fdc_id = cands[0].fdc_id
        mapping.fdc_description = cands[0].description
        mapping.fdc_data_type = cands[0].data_type
        mapping.rule_id = cands[0].rule_id
        mapping.default_raw_preference = cands[0].default_raw_preference
        mapping.notes = f"{len(top_tier)} candidates within 5-point score band; needs human ruling."

    return mapping


def load_clean_pool() -> pd.DataFrame:
    with zipfile.ZipFile(USDA_ZIP) as z:
        food = pd.read_csv(
            io.BytesIO(z.read(USDA_PREFIX + "food.csv")),
            usecols=["fdc_id", "data_type", "description", "food_category_id"],
        )
        ff = pd.read_csv(io.BytesIO(z.read(USDA_PREFIX + "foundation_food.csv")))
        sl = pd.read_csv(io.BytesIO(z.read(USDA_PREFIX + "sr_legacy_food.csv")))
    clean_ids = set(ff["fdc_id"]) | set(sl["fdc_id"])
    clean = food[food["fdc_id"].isin(clean_ids)].copy()
    clean["norm_desc"] = clean["description"].map(norm)
    return clean


def build_pattern_library(clean: pd.DataFrame) -> dict[str, Any]:
    patterns: Counter[str] = Counter()
    prefix_examples: dict[str, list[str]] = defaultdict(list)

    def bump(label: str, desc: str) -> None:
        patterns[label] += 1
        if len(prefix_examples[label]) < 3:
            prefix_examples[label].append(desc)

    for desc in clean["description"].astype(str):
        d = desc
        dl = d.lower()
        if d.startswith("Spices, "):
            bump("Spices, {name}, {ground|dried}", d)
        elif d.startswith("Nuts, "):
            bump("Nuts, {name}, {form}", d)
        elif d.startswith("Seeds, "):
            bump("Seeds, {name}, {form}", d)
        elif d.startswith("Fish, "):
            bump("Fish, {species}, {raw|cooked}", d)
        elif d.startswith("Crustaceans, "):
            bump("Crustaceans, {species}, {form}", d)
        elif d.startswith("Beef,"):
            bump("Beef, {cut|composite}, {raw|cooked}", d)
        elif d.startswith("Pork,"):
            bump("Pork, {cut}, {raw|cooked}", d)
        elif d.startswith("Chicken,"):
            bump("Chicken, {cut}, {raw|cooked}", d)
        elif d.startswith("Lamb,"):
            bump("Lamb, {cut}, {raw|cooked}", d)
        elif d.startswith("Milk,"):
            bump("Milk, {type}", d)
        elif d.startswith("Cheese,"):
            bump("Cheese, {type}", d)
        elif d.startswith("Egg,"):
            bump("Egg, {form}", d)
        elif d.startswith("Beverages,"):
            bump("Beverages, {type}", d)
        elif ", raw" in dl:
            bump("{produce}, raw", d)
        elif "cooked" in dl:
            bump("{food}, cooked, {method}", d)
        elif "canned" in dl:
            bump("{food}, canned", d)
        elif "frozen" in dl:
            bump("{food}, frozen", d)
        elif "mature seeds, raw" in dl:
            bump("{legume}, mature seeds, raw", d)
        else:
            bump("other", d)

    return {
        "pattern_counts": dict(patterns.most_common(30)),
        "examples": {k: v for k, v in prefix_examples.items()},
        "variant_selection_policy": VARIANT_SELECTION_POLICY,
    }


def mapping_to_review_dict(m: SpeciesMapping) -> dict[str, Any]:
    d = asdict(m)
    d["candidates"] = [asdict(c) for c in m.candidates]
    return d


def run_review_phase() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    species = pd.read_parquet(SPECIES_PATH)
    clean = load_clean_pool()

    clean.to_parquet(OUT_DIR / "fdc_clean_pool_v1.parquet", index=False)
    patterns = build_pattern_library(clean)
    (OUT_DIR / "fdc_description_patterns_v1.json").write_text(
        json.dumps(patterns, indent=2), encoding="utf-8"
    )

    mappings = [classify_species(row, clean) for _, row in species.iterrows()]
    counts = Counter(m.classification for m in mappings)

    auto = [m for m in mappings if m.classification == "AUTO-CONFIDENT"]
    auto_df = pd.DataFrame(
        [
            {
                "species_id": m.species_id,
                "canonical_name": m.canonical_name,
                "fdc_id": m.fdc_id,
                "fdc_description": m.fdc_description,
                "fdc_data_type": m.fdc_data_type,
                "rule_id": m.rule_id,
                "default_raw_preference": m.default_raw_preference,
            }
            for m in auto
        ]
    )
    auto_df.to_parquet(OUT_DIR / "species_fdc_auto_confident_v1.parquet", index=False)

    # Coverage projection: auto + judgment defaults + ambiguous top pick (optimistic/pessimistic)
    proj_auto = counts["AUTO-CONFIDENT"]
    proj_judgment = counts["JUDGMENT"]  # has proposed default
    proj_ambiguous = counts["AMBIGUOUS"]  # needs ruling
    proj_none = counts["NO-MATCH"]

    review = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "C",
        "status": "AWAITING_HUMAN_REVIEW",
        "variant_selection_policy": VARIANT_SELECTION_POLICY,
        "phase_a": {
            "clean_pool_size": len(clean),
            "data_type_breakdown": clean["data_type"].value_counts().to_dict(),
            "branded_excluded": True,
        },
        "phase_b_counts": dict(counts),
        "coverage_projection": {
            "auto_confident_now": proj_auto,
            "judgment_with_proposed_default": proj_judgment,
            "ambiguous_needs_ruling": proj_ambiguous,
            "no_match": proj_none,
            "optimistic_after_rulings": proj_auto + proj_judgment + proj_ambiguous,
            "pessimistic_composition_ready": proj_auto,
            "notes": (
                "Optimistic assumes all JUDGMENT defaults accepted and all AMBIGUOUS resolved. "
                "NO-MATCH species need FooDB fallback or manual entry."
            ),
        },
        "auto_confident_sample_n30": [
            mapping_to_review_dict(m)
            for m in auto[:30]
        ],
        "auto_confident_all": [mapping_to_review_dict(m) for m in auto],
        "ambiguous": [mapping_to_review_dict(m) for m in mappings if m.classification == "AMBIGUOUS"],
        "judgment": [mapping_to_review_dict(m) for m in mappings if m.classification == "JUDGMENT"],
        "no_match": [
            mapping_to_review_dict(m)
            for m in sorted(
                [m for m in mappings if m.classification == "NO-MATCH"],
                key=lambda x: -x.n_strings,
            )
        ],
    }

    REVIEW_PATH.write_text(json.dumps(review, indent=2), encoding="utf-8")

    # Human-friendly review tables
    amb_rows = []
    for item in review["ambiguous"]:
        for rank, cand in enumerate(item.get("candidates", [])[:8], start=1):
            amb_rows.append(
                {
                    "species_id": item["species_id"],
                    "canonical_name": item["canonical_name"],
                    "n_strings": item["n_strings"],
                    "candidate_rank": rank,
                    "fdc_id": cand["fdc_id"],
                    "fdc_description": cand["description"],
                    "score": cand["score"],
                    "proposed_default": rank == 1,
                }
            )
    if amb_rows:
        pd.DataFrame(amb_rows).to_csv(OUT_DIR / "species_fdc_review_ambiguous_v1.csv", index=False)

    jud_rows = []
    for item in review["judgment"]:
        jud_rows.append(
            {
                "species_id": item["species_id"],
                "canonical_name": item["canonical_name"],
                "category": item["category"],
                "n_strings": item["n_strings"],
                "proposed_fdc_id": item.get("fdc_id"),
                "proposed_fdc_description": item.get("fdc_description"),
                "judgment_rationale": item.get("judgment_rationale"),
            }
        )
    pd.DataFrame(jud_rows).to_csv(OUT_DIR / "species_fdc_review_judgment_v1.csv", index=False)

    pd.DataFrame(review["no_match"]).to_csv(OUT_DIR / "species_fdc_review_no_match_v1.csv", index=False)

    report = {
        "generated_at": review["generated_at"],
        "clean_pool_size": len(clean),
        "species_total": len(species),
        "classification_counts": dict(counts),
        "coverage_projection": review["coverage_projection"],
        "outputs": {
            "clean_pool": str(OUT_DIR / "fdc_clean_pool_v1.parquet"),
            "patterns": str(OUT_DIR / "fdc_description_patterns_v1.json"),
            "review": str(REVIEW_PATH),
            "auto_confident": str(OUT_DIR / "species_fdc_auto_confident_v1.parquet"),
        },
        "next_step": (
            "Human review: rule on ambiguous + judgment lists, write species_fdc_map_approved_v1.parquet, "
            "then run --phase compose"
        ),
    }
    (OUT_DIR / "species_fdc_mapping_report_v1.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


# Nutrient groups for Phase D
MACRO_IDS = {
    "Energy", "Protein", "Total lipid (fat)", "Carbohydrate, by difference",
    "Fiber, total dietary", "Water", "Total Sugars", "Sugars, added", "Ash",
}
MINERAL_NAMES = {
    "Calcium, Ca", "Iron, Fe", "Magnesium, Mg", "Phosphorus, P", "Potassium, K",
    "Sodium, Na", "Zinc, Zn", "Copper, Cu", "Manganese, Mn", "Selenium, Se",
}
VITAMIN_KEYWORDS = (
    "Vitamin", "Retinol", "Carotene", "Tocopherol", "Folate", "Folic", "Niacin",
    "Riboflavin", "Thiamin", "Pantothen", "Biotin", "Choline", "Phylloquinone",
    "Menaquinone", "Cholecalciferol", "Ergocalciferol", "Ascorbic", "Cobalamin", "Pyridox",
)
AMINO_ACID_KEYWORDS = (
    "Tryptophan", "Threonine", "Isoleucine", "Leucine", "Lysine", "Methionine",
    "Cystine", "Phenylalanine", "Tyrosine", "Valine", "Arginine", "Histidine",
    "Alanine", "Aspartic", "Glutamic", "Glycine", "Proline", "Serine", "Hydroxyproline",
)


def nutrient_group(name: str) -> str:
    if name in MACRO_IDS or any(k in name for k in ("Energy", "Protein", "Fiber", "Water", "Carbohydrate", "Sugars")):
        if "Fatty" not in name and "Amino" not in name:
            return "macro"
    if name in MINERAL_NAMES or re.search(r", (Ca|Fe|Mg|P|K|Na|Zn|Cu|Mn|Se)$", name):
        return "mineral"
    if any(k in name for k in VITAMIN_KEYWORDS):
        return "vitamin"
    if any(k in name for k in AMINO_ACID_KEYWORDS):
        return "amino_acid"
    if "Fatty acid" in name or re.search(r"\b\d+:\d+", name):
        return "fatty_acid"
    return "other"


def run_compose_phase() -> dict[str, Any]:
    if not APPROVED_PATH.exists():
        raise FileNotFoundError(
            f"Approved mapping not found: {APPROVED_PATH}. "
            "Create from review artifact after human rulings."
        )
    approved = pd.read_parquet(APPROVED_PATH)
    clean = pd.read_parquet(OUT_DIR / "fdc_clean_pool_v1.parquet")
    fn = pd.read_parquet(FOOD_NUTRIENT_PARQUET)
    with zipfile.ZipFile(USDA_ZIP) as z:
        nutrients = pd.read_csv(io.BytesIO(z.read(USDA_PREFIX + "nutrient.csv")))

    merged = approved.merge(
        clean[["fdc_id", "data_type", "description"]].rename(
            columns={"description": "fdc_description_clean", "data_type": "fdc_data_type_clean"}
        ),
        on="fdc_id",
        how="left",
    )
    if "fdc_description" not in merged.columns:
        merged["fdc_description"] = merged["fdc_description_clean"]
    if "fdc_data_type" not in merged.columns:
        merged["fdc_data_type"] = merged.get("fdc_data_type", merged["fdc_data_type_clean"])

    fn_sub = fn[fn["fdc_id"].isin(set(merged["fdc_id"]))].merge(
        nutrients[["id", "name", "unit_name"]],
        left_on="nutrient_id",
        right_on="id",
        how="left",
    )
    sub = merged.merge(fn_sub, on="fdc_id", how="inner")
    sub = sub.rename(columns={"name": "nutrient_name", "unit_name": "unit"})
    sub["nutrient_group"] = sub["nutrient_name"].map(lambda x: nutrient_group(str(x)))
    sub["basis"] = "per_100g"
    sub["provenance"] = sub["fdc_data_type"].map(
        lambda x: "foundation_lab_quality" if x == "foundation_food" else "sr_legacy"
    )

    out_cols = [
        "species_id", "canonical_name", "fdc_id", "fdc_description", "fdc_data_type",
        "mapping_class", "provenance", "nutrient_id", "nutrient_name", "nutrient_group",
        "amount", "unit", "basis",
    ]
    profiles = sub[out_cols].copy()
    profiles.to_parquet(OUT_DIR / "species_nutrient_profiles_v1.parquet", index=False)

    completeness = (
        profiles.groupby(["species_id", "canonical_name", "fdc_data_type", "mapping_class"])
        .agg(
            n_nutrients=("nutrient_id", "nunique"),
            n_macro=("nutrient_group", lambda s: (s == "macro").sum()),
            n_mineral=("nutrient_group", lambda s: (s == "mineral").sum()),
            n_vitamin=("nutrient_group", lambda s: (s == "vitamin").sum()),
            n_amino_acid=("nutrient_group", lambda s: (s == "amino_acid").sum()),
            n_fatty_acid=("nutrient_group", lambda s: (s == "fatty_acid").sum()),
            n_other=("nutrient_group", lambda s: (s == "other").sum()),
        )
        .reset_index()
    )
    completeness.to_parquet(OUT_DIR / "species_nutrient_completeness_v1.parquet", index=False)

    verify_species = ["Lettuce", "Spinach", "Turmeric"]
    key_nutrients = [
        "Water", "Fiber, total dietary", "Potassium, K", "Folate, DFE", "Folate, food",
        "Folic acid", "Calcium, Ca", "Iron, Fe", "Magnesium, Mg", "Protein",
        "Carbohydrate, by difference", "Total lipid (fat)", "Energy",
        "Vitamin C, total ascorbic acid", "Vitamin A, IU",
    ]
    verification = {}
    for sn in verify_species:
        chunk = profiles[profiles["canonical_name"].str.lower() == sn.lower()]
        if chunk.empty:
            verification[sn] = {"error": "not mapped"}
            continue
        fdc_id = int(chunk["fdc_id"].iloc[0])
        by_group = {}
        for grp, gdf in chunk.groupby("nutrient_group"):
            by_group[grp] = gdf[["nutrient_name", "amount", "unit"]].sort_values("nutrient_name").to_dict("records")
        verification[sn] = {
            "species_id": chunk["species_id"].iloc[0],
            "fdc_id": fdc_id,
            "fdc_description": chunk["fdc_description"].iloc[0],
            "fdc_data_type": chunk["fdc_data_type"].iloc[0],
            "mapping_class": chunk["mapping_class"].iloc[0],
            "key_nutrients": chunk[chunk["nutrient_name"].isin(key_nutrients)][
                ["nutrient_name", "amount", "unit"]
            ].sort_values("nutrient_name").to_dict("records"),
            "nutrients_by_group": by_group,
            "nutrient_counts_by_group": chunk.groupby("nutrient_group")["nutrient_id"].nunique().to_dict(),
        }

    deduped_path = OUT_DIR / "species_fdc_no_match_deduped_v1.json"
    no_match_gaps = []
    if deduped_path.exists():
        deduped = json.loads(deduped_path.read_text(encoding="utf-8"))
        no_match_gaps = deduped.get("true_no_match", [])

    all_species = pd.read_parquet(ROOT / "data/processed/canonical/species_nodes_v2.parquet")
    mapped_ids = set(approved["species_id"])
    unmapped = all_species[~all_species["species_node_id"].isin(mapped_ids)]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "species_with_composition": int(profiles["species_id"].nunique()),
        "species_total": 463,
        "total_nutrient_rows": len(profiles),
        "mapping_class_counts": approved["mapping_class"].value_counts().to_dict(),
        "fdc_data_type_species_counts": approved["fdc_data_type"].value_counts().to_dict(),
        "foundation_food_species": int((approved["fdc_data_type"] == "foundation_food").sum()),
        "sr_legacy_species": int((approved["fdc_data_type"] == "sr_legacy_food").sum()),
        "completeness_stats": {
            "median_n_nutrients": float(completeness["n_nutrients"].median()),
            "min_n_nutrients": int(completeness["n_nutrients"].min()),
            "max_n_nutrients": int(completeness["n_nutrients"].max()),
        },
        "verification": verification,
        "gaps": {
            "true_no_match_count": len(no_match_gaps) if no_match_gaps else int(len(unmapped)),
            "true_no_match_top_by_recipe_freq": [
                {"species_id": x.get("species_id"), "canonical_name": x.get("canonical_name"),
                 "n_strings": x.get("n_strings"), "category": x.get("category")}
                for x in (no_match_gaps[:30] if no_match_gaps else [])
            ],
            "unmapped_count": int(len(unmapped)),
            "sparse_coverage": {
                "amino_acids": "~63% of clean FDC foods; expect gaps for produce",
                "vitamin_d": "Measured in subset of foods only",
            },
        },
    }
    (OUT_DIR / "species_nutrient_summary_v1.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build species→FDC map (Stage 1)")
    parser.add_argument(
        "--phase",
        choices=["review", "compose"],
        default="review",
        help="review = phases A–C (STOP gate); compose = phase D after approval",
    )
    args = parser.parse_args()
    if args.phase == "review":
        report = run_review_phase()
        print(json.dumps(report, indent=2))
    else:
        summary = run_compose_phase()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
