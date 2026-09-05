#!/usr/bin/env python3
"""
Stage 1 v2 — species → USDA FDC mapping with matcher guards + FooDB fallback.

Fixes v1 pollution: head-noun guard, category gate, modifier blocklist,
form enforcement, branded/restaurant pool exclusion, recovery templates.

Outputs (v2 only; v1 preserved):
  data/processed/product/nutrients/fdc_clean_pool_v2.parquet
  data/processed/product/nutrients/species_fdc_mapping_review_v2.json
  data/processed/product/nutrients/species_fdc_map_v2.parquet
  data/processed/product/nutrients/species_nutrient_profiles_v2.parquet
  data/processed/product/nutrients/species_nutrient_summary_v2.json
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
FOODB_CONTENT = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Content.csv"
FOODB_NUTRIENT = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Nutrient.csv"
V1_RULINGS_PATH = OUT_DIR / "species_fdc_rulings_v1.json"
V2_RULINGS_PATH = OUT_DIR / "species_fdc_rulings_v2.json"
PRODUCTION_MANIFEST_PATH = OUT_DIR / "species_nutrient_production_manifest.json"

VARIANT_SELECTION_POLICY = (
    "Prefer raw/base form (', raw', 'mature seeds, raw', 'Spices, X, ground/dried'). "
    "Reject nectar/juice/canned/cooked when raw exists. Foundation Foods tie-break over SR Legacy. "
    "Branded/restaurant/fast-food entries excluded from matchable pool."
)

# ---------------------------------------------------------------------------
# Category taxonomy (species → guard category)
# ---------------------------------------------------------------------------
SPECIES_CATEGORIES = (
    "spice_herb", "vegetable", "fruit", "nut_seed", "legume", "grain",
    "meat", "fish_seafood", "dairy_egg", "beverage", "oil", "fungus", "algae", "processed", "other",
)

# FDC food_category_id → allowed per species category
SPECIES_TO_FDC_CATS: dict[str, set[int]] = {
    "spice_herb": {2, 11},
    "vegetable": {11, 9},
    "fruit": {9, 11},
    "nut_seed": {12, 4},
    "legume": {16, 11},
    "grain": {20, 8},
    "meat": {13, 10, 17, 5},
    "fish_seafood": {15},
    "dairy_egg": {1},
    "beverage": {14, 28},
    "oil": {4, 24},
    "fungus": {11},
    "algae": {11, 24},
    "processed": {6, 7, 8, 18, 19, 20, 22, 23},
    "other": {11, 9, 12, 16, 20},
}

EXCLUDED_FDC_CATEGORY_IDS = {21, 25, 26}  # Fast Foods, Restaurant Foods, etc.

BRAND_PATTERNS = re.compile(
    r"(restaurant|fast foods|pizza hut|taco bell|cracker barrel|hot pockets|"
    r"applebee|denny'?s|olive garden|on the border|kraft|pillsbury|nestle|hershey|"
    r"baby mum|shake n bake|smart soup|archway|smucker|waffle, buttermilk, frozen|"
    r"bolthouse|mum mum|5th avenue|breyers|lifeway|hot pockets)",
    re.I,
)

BAD_FORM = re.compile(
    r"\b(nectar|juice\b|juice,|canned|cooked|fried|breaded|sweetened|"
    r"microwaved|restaurant|babyfood|frozen, ready|prepared with|extract|"
    r"flour\b|chips\b|sandwich|platter)\b",
    re.I,
)

COLLISION_MODIFIERS = {
    "sweet", "lemon", "blue", "wild", "black", "red", "green", "orange", "hot", "white", "bell",
}

GENERIC_NAME_TOKENS = {
    "garden", "common", "sweet", "sour", "green", "red", "black", "white", "yellow",
    "chinese", "american", "highbush", "wild", "other", "plain", "greek", "italian",
    "mexican", "ceylon", "domestic", "nut", "spice", "bell", "hot", "lemon", "blue",
    "orange", "baby", "herbal", "cooking", "processed", "winter", "summer", "garden",
}

SPICE_HERB_NAMES = {
    "turmeric", "cumin", "coriander", "cardamom", "saffron", "nutmeg", "clove", "cloves",
    "cinnamon", "ginger", "horseradish", "wasabi", "paprika", "chili", "fenugreek", "anise",
    "caraway", "dill", "tarragon", "thyme", "sage", "basil", "oregano", "marjoram", "rosemary",
    "bay", "mint", "parsley", "chervil", "lavender", "mace", "mustard", "poppy", "savory",
    "sumac", "vanilla", "allspice", "capers", "chives", "garlic", "hops", "mugwort", "pepper",
    "safflower", "sesame", "spearmint", "star anise", "sweet basil", "common thyme", "common sage",
    "common oregano", "mexican oregano", "italian oregano", "sweet marjoram", "lemon balm",
    "lemon verbena", "lemon thyme", "sweet bay", "ceylon cinnamon", "black mustard",
    "white mustard", "chinese mustard", "summer savory", "winter savory", "curry", "galangal",
    "lemongrass", "lemon grass", "peppermint", "salt", "vinegar", "horseradish",
}

FORM_EXCEPTIONS: dict[str, str] = {
    "olive": "no raw olive in FDC; canned ripe acceptable",
    "capers": "only canned form in FDC",
}

JUDGMENT_FDC_PATTERNS: dict[str, str] = {
    "Yogurt": "Yogurt, plain, whole milk",
    "Cream": "Cream, fluid, heavy whipping",
    "Ginger": "Ginger root, raw",
    "Butter": "Butter, without salt",
    "Mozzarella cheese": "Cheese, mozzarella, whole milk",
    "Parmesan cheese": "Cheese, parmesan, grated",
    "Turkey": "Turkey, whole, meat and skin, raw",
    "Grape wine": "Alcoholic beverage, wine, table, all",
    "Beer": "Alcoholic beverage, beer, regular, all",
    "Cheddar Cheese": "Cheese, cheddar",
    "Buttermilk": "Milk, buttermilk, fluid, cultured, lowfat",
    "Butternut": "Squash, winter, butternut, raw",
    "Butternut squash": "Squash, winter, butternut, raw",
    "Coconut milk": "Nuts, coconut milk, raw (liquid expressed from grated meat and water)",
    "Almond milk": "Beverages, almond milk, unsweetened, shelf stable",
    "Soy milk": "Soy milk, unsweetened, plain, shelf stable",
    "Soy cream": "Soy milk, unsweetened, plain, shelf stable",
    "Bison": "Game meat, bison, separable lean only, raw",
    "Buffalo": "Game meat, buffalo, water, raw",
    "Elk": "Game meat, elk, ground, raw",
    "Butternut": "Nuts, butternuts, dried",
    "Pasta": "Pasta, dry, unenriched",
    "Rum": "Alcoholic beverage, distilled, rum",
    "Vodka": "Alcoholic beverage, distilled, vodka, 80 proof",
    "Gin": "Alcoholic beverage, distilled, gin, 90 proof",
    "Whisky": "Alcoholic beverage, distilled, all (gin, rum, vodka, whiskey) 86 proof",
    "Rabbit": "Game meat, rabbit, domesticated, composite of cuts, raw",
    "Sour cream": "Cream, sour, cultured",
    "Guinea hen": "Guinea hen, meat and skin, raw",
    "Ice cream": "Ice creams, vanilla",
    "Tortilla chip": "Snacks, tortilla chips, plain, white corn, salted",
    "Buffalo": "Game meat, bison, ground, raw",
    "Elk": "Game meat, elk, ground, raw",
    "Pheasant": "Pheasant, leg, meat only, raw",
    "Whelk": "Mollusks, whelk, unspecified, raw",
    "Apple cider": "Beverages, apple juice, unsweetened",
    "Madeira wine": "Alcoholic beverage, wine, cooking",
    "Soy yogurt": "Yogurt, plain, skim milk",
    "Pita bread": "Bread, pita, whole-wheat",
    "Plain cream cheese": "Cheese, cream",
    "Swiss cheese": "Cheese, swiss",
    "Cottage cheese": "Cheese, cottage, creamed, large or small curd",
    "Greek feta cheese": "Cheese, feta",
    "Herbal tea": "Beverages, tea, herb, brewed, chamomile",
    "Bison": "Game meat, bison, ground, raw",
    "Breadfruit": "Breadfruit, raw",
    "Quail": "Quail, breast, meat only, raw",
    "Port wine": "Alcoholic beverage, wine, dessert, dry",
    "Dessert wine": "Alcoholic beverage, wine, dessert, dry",
    "Processed cheese": "Cheese, pasteurized process, American, without di sodium phosphate",
    "Cocoa butter": "Oil, cocoa butter",
    "Soy milk": "Soymilk (all flavors), unsweetened, with added calcium, vitamins A and D",
    "Red tea": "Beverages, tea, black, brewed, prepared with tap water",
    "Cape gooseberry": "Groundcherries, (cape-gooseberries or poha), raw",
    "Ice cream": "Ice creams, vanilla",
    "Potato chip": "Snacks, potato chips, plain, salted",
}

OUT_OF_SCOPE_JUDGMENT = {
    "Biscuit", "Candy bar", "Other candy", "Meatball", "Hamburger", "Meatloaf",
    "Other bread",
    "Pizza", "Burrito", "Taco", "Taco shell", "Potato chip", "Tortilla chip",
    "Pancake", "Cape gooseberry", "Apple cider",
    "Citrus", "Nuts", "Fruits", "Green vegetables", "Root vegetables",
}

MEAT_REPRESENTATIVE_DEFAULTS: dict[str, dict[str, Any]] = {
    "SP_000237": {"fdc_pattern": "Beef, composite of trimmed retail cuts, separable lean only, trimmed to 0\" fat, all grades, raw"},
    "SP_000259": {"fdc_pattern": "Pork, fresh, loin, whole, separable lean only, raw"},
    "SP_000189": {"fdc_pattern": "Chicken, broiler or fryers, breast, skinless, boneless, meat only, raw"},
    "SP_000235": {"fdc_pattern": "Fish, salmon, Atlantic, wild, raw"},
    "SP_000030": {"fdc_pattern": "Beverages, tea, black, brewed, prepared with tap water"},
    "SP_000288": {"fdc_pattern": "Milk, whole, 3.25% milkfat, with added vitamin D"},
    "SP_000287": {"fdc_pattern": "Cheese, cheddar"},
}

NAME_TO_FDC_TERMS: dict[str, list[str]] = {
    "Garden onion": ["onions"],
    "White onion": ["onions"],
    "Kiwi": ["kiwifruit"],
    "Soy bean": ["soybeans"],
    "Garden tomato": ["tomatoes, red, ripe"],
    "Cherry tomato": ["tomatoes, red, ripe"],
    "Brussel sprouts": ["brussels sprouts"],
    "Swede": ["rutabaga"],
    "Cashew nut": ["cashew nuts"],
    "Pecan nut": ["pecans"],
    "Brazil nut": ["brazilnuts"],
    "Peanut": ["peanuts"],
    "Ceylon cinnamon": ["cinnamon"],
    "Star anise": ["anise seed"],
    "Black walnut": ["walnuts, black"],
    "Sweet bay": ["bay leaf"],
    "Rocket salad": ["arugula"],
    "Rapini": ["broccoli raab"],
    "Daikon radish": ["radishes, oriental"],
    "Black radish": ["radishes, oriental"],
    "Celery stalks": ["celery"],
    "Chicory leaves": ["chicory"],
    "Savoy cabbage": ["cabbage, savoy"],
    "Green onion": ["onions"],
    "Red onion": ["onions"],
    "Green bell pepper": ["peppers, sweet"],
    "Red bell pepper": ["peppers, sweet"],
    "Yellow bell pepper": ["peppers, sweet"],
    "Orange bell pepper": ["peppers, sweet"],
    "Cubanelle pepper": ["peppers, sweet"],
    "Jalapeno pepper": ["peppers, jalapeno"],
    "Pepper (Spice)": ["spices, pepper, black"],
    "Highbush blueberry": ["blueberries"],
    "Red raspberry": ["raspberries"],
    "Black raspberry": ["raspberries"],
    "American cranberry": ["cranberries"],
    "Asian pear": ["pears"],
    "Green zucchini": ["squash, summer, zucchini"],
    "Yellow zucchini": ["squash, summer, zucchini"],
    "Japanese pumpkin": ["pumpkin"],
    "Broad bean": ["beans, fava"],
    "Black-eyed pea": ["cowpeas"],
    "Adzuki bean": ["beans, adzuki"],
    "Mung bean": ["mung beans"],
    "Winged bean": ["winged beans"],
    "Cannellini bean": ["beans, white"],
    "Green lentil": ["lentils"],
    "Yellow wax bean": ["beans, snap"],
    "Green bean": ["beans, snap"],
    "Wild leek": ["leeks"],
    "Leek": ["leeks"],
    "Shallot": ["shallots"],
    "Chickpea": ["chickpeas"],
    "Flaxseed": ["flaxseed"],
    "Fig": ["figs"],
    "Clam": ["mollusks, clam"],
    "Squid": ["mollusks, squid"],
    "Haddock": ["fish, haddock"],
    "Grouper": ["fish, grouper"],
    "Snapper": ["fish, snapper"],
    "Anchovy": ["fish, anchovy"],
    "Jicama": ["jicama"],
    "Boysenberry": ["boysenberries"],
    "Shiitake": ["mushrooms, shiitake"],
    "Kombu": ["seaweed, wakame"],
    "Wakame": ["seaweed, wakame"],
    "Chestnut": ["chestnuts"],
    "Kumquat": ["kumquats"],
    "Quince": ["quinces"],
    "Loquat": ["loquats"],
    "Nectarine": ["nectarines"],
    "Longan": ["longans"],
    "Plantain": ["plantains"],
    "Sunflower": ["sunflower seed"],
    "Lima bean": ["lima beans"],
    "Common pea": ["peas, green"],
    "Chinese chives": ["chives"],
    "Milk (Cow)": ["milk, whole"],
    "Cattle (Beef, Veal)": ["beef, composite"],
    "Domestic pig": ["pork, fresh, loin"],
    "Salmonidae (Salmon, Trout)": ["fish, salmon"],
    "Common mushroom": ["mushrooms"],
    "Breadfruit": ["breadfruit"],
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
    "Rapeseed oil": ["canola"],
    "Cooking oil": ["olive"],
    "Olive oil": ["olive"],
    "Soybean oil": ["soybean"],
    "Sesame oil": ["sesame"],
    "Coconut oil": ["coconut"],
    "Avocado oil": ["avocado"],
    "Peanut oil": ["peanut"],
    "Grapeseed oil": ["grapeseed"],
    "Sunflower oil": ["sunflower"],
    "Green apple": ["apples, raw, with skin"],
    "Orange bell pepper": ["peppers, sweet, red"],
    "Green grape": ["grapes, green"],
    "Red grape": ["grapes, red"],
    "Acorn squash": ["squash, winter, acorn"],
    "Coffee": ["beverages, coffee, brewed"],
    "Banana": ["bananas"],
    "Lentils": ["lentils"],
    "Almond": ["almonds"],
    "Coconut": ["coconut meat"],
    "Shrimp": ["crustaceans, shrimp"],
    "Scallop": ["mollusks, scallop"],
    "Rabbit": ["game meat, rabbit"],
    "Parsnip": ["parsnips"],
    "Walnut": ["walnuts"],
    "Hazelnut": ["hazelnuts"],
    "Pistachio": ["pistachio nuts"],
    "Saffron": ["spices, saffron"],
    "Cumin": ["spices, cumin seed"],
    "Cardamom": ["spices, cardamom"],
    "Sweet basil": ["basil"],
    "Sweet marjoram": ["spices, marjoram"],
    "Lemon balm": ["balm"],
    "Lemon verbena": ["verbena"],
    "Lemon thyme": ["thyme"],
    "Lemon": ["lemons"],
    "Lemon grass": ["lemon grass"],
    "Corn": ["corn, sweet"],
    "Corn chip": ["snacks, corn-based"],
    "Corn grits": ["corn grits"],
    "Papaya": ["papayas"],
    "Peach": ["peaches"],
    "Guava": ["guavas"],
    "Potato": ["potatoes"],
    "Baked potato": ["potatoes, baked"],
    "Spelt": ["spelt"],
    "Orange roughy": ["fish, roughy, orange"],
    "Sweet orange": ["oranges"],
    "Sweet cherry": ["cherries, sweet"],
    "Sour cherry": ["cherries, sour"],
    "Apple": ["apples"],
    "Blackberry": ["blackberries"],
    "Grapefruit": ["grapefruit"],
    "Lime": ["limes"],
    "Pomegranate": ["pomegranates"],
    "Tamarind": ["tamarind"],
    "Passion fruit": ["passion-fruit"],
    "Strawberry": ["strawberries"],
    "Honey": ["honey"],
    "Wild boar": ["game meat, boar, wild"],
    "Blue cheese": ["cheese, blue"],
    "Lettuce": ["lettuce, cos or romaine"],
    "Olive": ["olives, ripe, canned"],
    "Turmeric": ["spices, turmeric"],
    "Spinach": ["spinach"],
    "Garlic": ["garlic"],
    "Broccoli": ["broccoli"],
    "Monkfish": ["fish, monkfish"],
    "Swordfish": ["fish, swordfish"],
    "Bluefish": ["fish, bluefish"],
    "Sablefish": ["fish, sablefish"],
    "Lingcod": ["fish, lingcod"],
    "Cuttlefish": ["mollusks, cuttlefish"],
    "Crayfish": ["crustaceans, crayfish"],
    "Snow crab": ["crustaceans, crab, blue"],
    "Yellowfin tuna": ["fish, tuna, fresh, yellowfin"],
    "Whitefish": ["fish, whitefish"],
    "Sockeye salmon": ["fish, salmon, sockeye"],
    "Pink salmon": ["fish, salmon, pink"],
    "Atlantic salmon": ["fish, salmon, atlantic"],
    "Pacific cod": ["fish, cod, pacific"],
    "Lake trout": ["fish, trout"],
    "Oyster mushroom": ["mushrooms, oyster"],
    "Chanterelle": ["mushrooms, chanterelle"],
    "Persimmon": ["persimmons"],
    "Date": ["dates"],
    "Pine nut": ["pine nuts"],
    "Macadamia nut": ["macadamia nuts"],
    "Wheat": ["wheat flour"],
    "Rye": ["rye flour"],
    "Gooseberry": ["gooseberries"],
    "Sorrel": ["sorrel"],
    "Black raisin": ["raisins"],
    "Snail": ["snails"],
    "Conch": ["mollusks, conch"],
    "Miso": ["miso"],
    "Pancake": ["pancakes"],
    "Cornbread": ["cornbread"],
    "Pigeon pea": ["pigeon peas"],
    "Acorn": ["squash, winter, acorn"],
    "Spanish mackerel": ["fish, mackerel, spanish"],
    "Walleye": ["fish, walleye"],
    "Black-eyed pea": ["cowpeas"],
    "Marshmallow": ["marshmallow"],
    "Meringue": ["meringue"],
    "Nopal": ["nopales"],
    "Crab": ["crustaceans, crab"],
}


@dataclass
class FdcCandidate:
    fdc_id: int
    description: str
    data_type: str
    food_category_id: int | None
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
    guard_failures: list[str] = field(default_factory=list)
    needs_human_ruling: bool = False


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
    if token.endswith("es") and len(token) > 3 and not token.endswith("ses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def token_match(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if singularize(a) == singularize(b):
        return True
    if a.startswith(b + " ") or b.startswith(a + " "):
        return True
    return False


def infer_species_category(name: str, latin: str | None, prep_labels: list[str]) -> str:
    n = norm(name)
    latin_l = norm(latin or "")

    if n == "butternut" and latin_l and "juglans" in latin_l:
        return "nut_seed"
    if n == "butternut squash" or (n == "butternut" and latin_l and "cucurbita" in latin_l):
        return "vegetable"

    if n in SPICE_HERB_NAMES or any(sp in n for sp in (
        "turmeric", "cumin", "cardamom", "saffron", "nutmeg", "clove", "basil", "thyme",
        "oregano", "marjoram", "rosemary", "sage", "parsley", "dill", "tarragon", "chervil",
        "fenugreek", "anise", "caraway", "peppermint", "spearmint", "verbena", "balm",
    )):
        return "spice_herb"
    if any(x in n for x in ("wine", "whisky", "whiskey", "vermouth", "sherry", "spirit", "beer", "rum", "vodka", "gin", "sake")):
        return "beverage"
    if "cider" in n and "vinegar" not in n:
        return "beverage"
    if re.search(r"\boil\b", n) or n == "cocoa butter":
        return "oil"
    if "coffee" in n:
        return "beverage"
    if "tea" in n or (latin_l and "camellia sinensis" in latin_l):
        return "beverage"
    if any(x in n for x in ("almond milk", "soy milk", "soy cream", "coconut milk")):
        return "beverage"
    if re.search(r"\b(milk|cream|yogurt|butter|cheese)\b", n) and "coconut milk" not in n and "butternut" not in n and "cocoa butter" not in n and "almond milk" not in n and "soy milk" not in n and "soy cream" not in n:
        return "dairy_egg"
    if "buttermilk" in n:
        return "dairy_egg"
    if re.search(r"\begg\b", n) and "eggplant" not in n and "egg roll" not in n:
        return "dairy_egg"
    if "mushroom" in n or any(x in latin_l for x in ("agaricus", "pleurotus", "lentinula", "cantharellus")):
        return "fungus"
    if any(x in n for x in ("kombu", "seaweed", "irish moss", "wakame", "algae")):
        return "algae"
    if "peanut" in n:
        return "nut_seed"
    if any(x in n for x in ("almond", "walnut", "pecan", "cashew", "hazelnut", "chestnut", "pistachio", "coconut", "pine nut", "macadamia")):
        return "nut_seed"
    if "seed" in n or n in {"sunflower", "safflower", "sesame", "flax", "flaxseed", "chia"}:
        return "nut_seed"
    if re.search(r"\b(bean|pea|lentil|chickpea|soy)\b", n) and "soybean oil" not in n and "soy sauce" not in n:
        return "legume"
    if "squash" in n or (latin_l and "cucurbita" in latin_l):
        return "vegetable"
    if any(re.search(rf"\b{re.escape(x)}\b", n) for x in (
        "oat", "rice", "wheat", "barley", "rye", "corn", "millet", "quinoa", "sorghum", "spelt", "bulgur", "couscous",
    )):
        return "grain"
    if "salmonidae" in n or ("fish" in latin_l and "idae" in latin_l):
        return "fish_seafood"
    if any(x in n for x in (
        "salmon", "trout", "tuna", "cod", "haddock", "sardine", "anchovy", "mackerel",
        "herring", "bass", "perch", "carp", "eel", "garfish", "turbot", "grouper", "snapper",
        "monkfish", "swordfish", "bluefish", "sablefish", "lingcod", "roughy", "walleye",
    )):
        return "fish_seafood"
    if any(x in n for x in (
        "crab", "lobster", "shrimp", "prawn", "oyster", "mussel", "clam", "squid", "octopus",
        "cuttlefish", "scallop", "crayfish", "whelk", "conch", "snail",
    )):
        return "fish_seafood"
    if any(x in n for x in ("rabbit", "hare", "boar", "bison", "buffalo", "elk", "pheasant", "quail", "guinea hen")):
        return "meat"
    if any(x in n for x in ("chicken", "turkey", "duck")) or re.search(r"\bgoose\b", n):
        return "meat"
    if any(x in n for x in ("beef", "cattle", "veal", "pork", "pig", "lamb", "mutton")):
        return "meat"
    if any(x in n for x in (
        "apple", "apricot", "avocado", "banana", "berry", "cherry", "grape", "melon", "citrus",
        "orange", "lemon", "lime", "mango", "papaya", "pineapple", "fig", "date", "plum", "peach", "pear",
        "kiwi", "guava", "lychee", "persimmon", "pomegranate", "quince", "boysenberry", "gooseberry",
        "kumquat", "loquat", "nectarine", "longan", "plantain",
    )):
        return "fruit"
    if any(x in n for x in (
        "bread", "pizza", "pasta", "candy", "cookie", "cake", "biscuit", "cracker", "chip",
        "waffle", "hot dog", "taco", "burrito", "dumpling", "falafel", "hummus", "margarine",
        "marshmallow", "meringue", "miso", "pancake", "cornbread",
    )):
        return "processed"
    return "vegetable"


def species_head_terms(name: str) -> list[str]:
    toks = [t for t in norm(name).split() if t not in GENERIC_NAME_TOKENS and len(t) >= 3]
    if not toks:
        toks = [t for t in norm(name).split() if len(t) >= 3]
    terms: list[str] = []
    for t in toks:
        terms.extend([t, singularize(t)])
    if len(toks) >= 2:
        terms.extend([toks[-1], singularize(toks[-1])])
    return list(dict.fromkeys(terms))


def fdc_head_term(desc: str) -> str:
    d = desc.lower().strip()
    if d.startswith("spices, "):
        parts = [p.strip() for p in desc.split(",")]
        return parts[1] if len(parts) > 1 else parts[0]
    if d.startswith("nuts, "):
        return d.split(",")[1].strip().split()[0] if "," in d else d[5:].split()[0]
    if d.startswith("seeds, "):
        return d.split(",")[1].strip().split()[0] if "," in d else d[7:].split()[0]
    if d.startswith("fish, ") or d.startswith("crustaceans, ") or d.startswith("mollusks, "):
        return d.split(",")[1].strip().split()[0] if "," in d else ""
    if d.startswith("beverages, ") or d.startswith("alcoholic beverage"):
        return "beverage"
    if d.startswith("oil, "):
        parts = [p.strip() for p in desc.split(",")]
        return parts[1].split()[0].lower() if len(parts) > 1 else "oil"
    if d.startswith("game meat, "):
        return d.split(",")[1].strip().split()[0] if "," in d else "game"
    return d.split(",")[0].strip()


def head_noun_ok(name: str, desc: str) -> bool:
    desc_n = norm(desc)
    terms = species_head_terms(name)
    nname = norm(name)

    if desc.lower().startswith("spices, "):
        spice_part = norm(desc.split(",")[1] if "," in desc else "")
        return any(token_match(t, spice_part) or t in spice_part for t in terms)

    proxies = {
        "cattle": ["beef"],
        "domestic pig": ["pork"],
        "salmonidae": ["salmon", "fish"],
        "milk (cow)": ["milk"],
        "cheese": ["cheese"],
        "olive": ["olives"],
        "pepper (spice)": ["pepper", "spices"],
        "wild boar": ["boar", "game"],
        "blue cheese": ["cheese", "blue"],
        "garden onion": ["onions", "onion"],
        "white onion": ["onions", "onion"],
        "kiwi": ["kiwifruit"],
        "rocket salad": ["arugula"],
        "rapini": ["broccoli raab", "raab"],
        "soy bean": ["soybeans", "soybean"],
        "cooking oil": ["oil"],
        "olive oil": ["oil", "olive"],
        "soybean oil": ["oil", "soybean"],
        "sesame oil": ["oil", "sesame"],
        "coconut oil": ["oil", "coconut"],
        "avocado oil": ["oil", "avocado"],
        "peanut oil": ["oil", "peanut"],
        "grapeseed oil": ["oil", "grapeseed"],
        "rapeseed oil": ["oil", "canola"],
        "cocoa butter": ["oil", "cocoa"],
        "green zucchini": ["zucchini", "squash"],
        "yellow zucchini": ["zucchini", "squash"],
        "cherry tomato": ["tomato", "tomatoes"],
        "green apple": ["apple", "apples"],
        "orange bell pepper": ["pepper", "peppers"],
        "green bell pepper": ["pepper", "peppers"],
        "red bell pepper": ["pepper", "peppers"],
        "yellow bell pepper": ["pepper", "peppers"],
        "green grape": ["grape", "grapes"],
        "red grape": ["grape", "grapes"],
        "acorn squash": ["squash", "acorn"],
        "pecan nut": ["pecans", "pecan"],
        "cashew nut": ["cashew"],
        "pine nut": ["pine"],
        "macadamia nut": ["macadamia"],
        "brazil nut": ["brazilnut"],
        "black walnut": ["walnut"],
        "sweet potato": ["sweet potatoes", "potato"],
        "irish moss": ["moss", "seaweed"],
        "jicama": ["yambean", "jicama"],
        "kombu": ["wakame", "seaweed"],
    }
    for k, alts in proxies.items():
        if k in nname:
            head = norm(fdc_head_term(desc))
            if any(token_match(a, head) or a in desc_n for a in alts):
                return True

    if desc_n.startswith("alcoholic beverage") or desc_n.startswith("beverages"):
        return any(t in desc_n for t in terms if len(t) >= 3)

    if desc.lower().startswith("oil,"):
        return any(t in desc_n for t in terms if len(t) >= 3)

    head = norm(fdc_head_term(desc))
    for t in sorted(terms, key=len, reverse=True):
        if len(t) < 4 and t not in {"fig", "pea", "soy", "tea", "oat", "rye", "kiwi", "lime", "clam", "pear"}:
            continue
        if token_match(t, head):
            return True
        if desc_n.startswith(t + ",") or desc_n.startswith(t + " "):
            return True
        if desc_n.startswith(singularize(t) + "s,") or desc_n.startswith(singularize(t) + "s "):
            return True

    return False


def modifier_collision_reject(name: str, desc: str, head_ok: bool) -> bool:
    if head_ok:
        return False
    name_toks = set(norm(name).split())
    modifier_hits = name_toks & COLLISION_MODIFIERS
    if not modifier_hits:
        return False
    head = norm(fdc_head_term(desc))
    significant = [t for t in species_head_terms(name) if t not in COLLISION_MODIFIERS]
    if not significant:
        return True
    if any(m in head for m in modifier_hits) and not any(token_match(t, head) for t in significant):
        return True
    return False


def category_ok(species_cat: str, fdc_cat: int | None) -> bool:
    if fdc_cat is None or pd.isna(fdc_cat):
        return True
    allowed = SPECIES_TO_FDC_CATS.get(species_cat, {11})
    return int(fdc_cat) in allowed


def find_raw_alternative(clean: pd.DataFrame, name: str, category: str, current_fdc: int) -> bool:
    terms = species_head_terms(name)
    for t in sorted(terms, key=len, reverse=True):
        if len(t) < 3:
            continue
        hits = clean[clean["norm_desc"].str.contains(rf"\b{re.escape(t)}\b", regex=True, na=False)]
        if category == "spice_herb":
            hits = hits[hits["description"].str.startswith("Spices,", na=False)]
        raw = hits[
            hits["description"].str.contains(
                r",\s*raw\b|mature seeds, raw|ground|dried|Seeds,",
                case=False, na=False, regex=True,
            )
        ]
        raw = raw[~raw["description"].str.contains(BAD_FORM, na=False)]
        if len(raw) and int(raw.iloc[0]["fdc_id"]) != current_fdc:
            return True
    return False


def form_ok(name: str, desc: str, clean: pd.DataFrame, category: str, fdc_id: int) -> bool:
    n = norm(name)
    if n in FORM_EXCEPTIONS or any(k in n for k in FORM_EXCEPTIONS):
        return True
    if not BAD_FORM.search(desc):
        return True
    if "beverages," in desc.lower() and category == "beverage":
        return True
    return not find_raw_alternative(clean, name, category, fdc_id)


def passes_guards(
    name: str,
    category: str,
    desc: str,
    fdc_cat: int | None,
    fdc_id: int,
    clean: pd.DataFrame,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if BRAND_PATTERNS.search(desc):
        failures.append("branded_restaurant")
    if fdc_cat is not None and not pd.isna(fdc_cat) and int(fdc_cat) in EXCLUDED_FDC_CATEGORY_IDS:
        failures.append("excluded_fdc_category")

    head_ok = head_noun_ok(name, desc)
    if not head_ok:
        failures.append("head_noun_mismatch")
    if modifier_collision_reject(name, desc, head_ok):
        failures.append("modifier_collision")
    if not category_ok(category, fdc_cat):
        failures.append("category_mismatch")
    if not form_ok(name, desc, clean, category, fdc_id):
        failures.append("form_violation")
    return len(failures) == 0, failures


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
        terms.append(norm(str(p).replace("_", " ")))
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def score_candidate(desc: str, data_type: str, rule_id: str) -> float:
    d = desc.lower()
    score = 80.0 if rule_id.startswith("template_exact") else 50.0 if rule_id.startswith("template_") else 20.0
    if re.search(r",\s*raw\b", d):
        score += 45
    if "mature seeds, raw" in d:
        score += 40
    if d.startswith("spices, ") and ("ground" in d or "dried" in d):
        score += 35
    if data_type == "foundation_food":
        score += 12
    if "cooked" in d:
        score -= 35
    if "canned" in d:
        score -= 30
    if "frozen" in d:
        score -= 25
    if BAD_FORM.search(desc):
        score -= 40
    if "salad or cooking" in d:
        score += 20
    if "extra light" in d or "extra virgin" in d:
        score -= 8
    if BRAND_PATTERNS.search(desc):
        score -= 100
    return score


def apply_rule_templates(category: str, name: str, terms: list[str], clean: pd.DataFrame) -> list[FdcCandidate]:
    cands: dict[int, FdcCandidate] = {}
    cn = norm(name)
    desc_col = clean["description"]

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
                    food_category_id=int(row["food_category_id"]) if pd.notna(row.get("food_category_id")) else None,
                    score=sc,
                    rule_id=rule_id,
                )

    for term in terms:
        t = norm(term)
        if not t:
            continue
        if any(term.lower().startswith(p) for p in (
            "spices,", "nuts,", "beans,", "beverages,", "crustaceans,", "mollusks,",
            "game meat,", "squash,", "seeds,", "fish,", "oil,", "cheese,", "milk,",
            "alcoholic beverage,", "yogurt,", "snacks,", "mushrooms,", "seaweed,",
            "chickpeas", "lima beans", "plantains", "boysenberries",
            "tomatoes,", "apples,", "peppers,", "grapes,", "squash,",
        )) or "," in term:
            add_hits(desc_col.str.contains(term, case=False, regex=False, na=False), "template_fdc_phrase")
            continue

        if category == "spice_herb":
            for suffix in ("ground", "dried"):
                add_hits(desc_col.str.fullmatch(rf"Spices, {re.escape(term)}, {suffix}", case=False, na=False), f"template_spice_{suffix}")
            add_hits(desc_col.str.fullmatch(rf"Spices, {re.escape(term)}", case=False, na=False), "template_spice_exact")
            add_hits(desc_col.str.fullmatch(rf"Spices, {re.escape(term)} seed", case=False, na=False), "template_spice_seed")
            add_hits(desc_col.str.match(rf"^{re.escape(term)}(,|\s)", case=False, na=False), "template_spice_name")
        elif category == "nut_seed":
            add_hits(desc_col.str.contains(rf"^Nuts, .*{re.escape(term)}", case=False, regex=True, na=False), "template_nut")
            add_hits(desc_col.str.contains(rf"^Seeds, .*{re.escape(term)}", case=False, regex=True, na=False), "template_seed")
            if t in {"peanut", "peanuts"}:
                add_hits(desc_col.str.contains("Peanuts, raw", case=False, na=False), "template_peanut")
        elif category == "fish_seafood":
            add_hits(desc_col.str.contains(rf"^Fish, .*{re.escape(term)}", case=False, regex=True, na=False), "template_fish")
            add_hits(desc_col.str.contains(rf"^(?:Crustaceans|Mollusks), .*{re.escape(term)}", case=False, regex=True, na=False), "template_shellfish")
        elif category == "legume":
            patterns = [
                rf"^{re.escape(term)}s?, raw$",
                rf"^{re.escape(term)}s?, mature seeds, raw$",
                rf"^Beans, {re.escape(term)}.*mature seeds, raw$",
                rf"^{re.escape(term)}s? \(.*\), mature seeds, raw$",
                rf"^Beans, snap, {re.escape(term)}, raw$",
            ]
            if "chickpea" in t:
                patterns.append(r"^Chickpeas \(.*\), mature seeds, raw$")
            for pat in patterns:
                add_hits(desc_col.str.match(pat, case=False, na=False), "template_legume_raw")
        elif category == "grain":
            add_hits(desc_col.str.contains(rf"{re.escape(term)}.*(?:raw|uncooked)", case=False, regex=True, na=False), "template_grain_raw")
            if "corn" in t:
                add_hits(desc_col.str.contains("Corn, sweet", case=False, na=False), "template_corn_sweet")
        elif category == "fruit":
            for pat in (rf"^{re.escape(term)}, raw$", rf"^{re.escape(term)}s, raw$", rf"^{re.escape(term)}es, raw$"):
                add_hits(desc_col.str.match(pat, case=False, na=False), "template_fruit_raw")
        elif category == "oil":
            oil_term = term
            if "oil" in norm(name) and name not in NAME_TO_FDC_TERMS:
                oil_term = re.sub(r"\s*oil\s*$", "", norm(name)).strip() or term
            add_hits(
                desc_col.str.match(rf"^Oil, {re.escape(oil_term)}", case=False, na=False),
                "template_oil",
            )
            add_hits(
                desc_col.str.match(rf"^Oil, {re.escape(oil_term)}, salad or cooking$", case=False, na=False),
                "template_oil_cooking",
            )
        elif category == "beverage":
            add_hits(desc_col.str.contains(rf"Beverages,.*{re.escape(term)}|Alcoholic beverage.*{re.escape(term)}", case=False, regex=True, na=False), "template_beverage")
        elif category == "dairy_egg":
            add_hits(desc_col.str.contains(rf"^(?:Milk|Cheese|Butter|Cream|Yogurt|Egg),.*{re.escape(term)}", case=False, regex=True, na=False), "template_dairy")
        elif category == "meat":
            add_hits(desc_col.str.contains(rf"^(?:Beef|Pork|Chicken|Turkey|Lamb|Game meat),.*{re.escape(term)}", case=False, regex=True, na=False), "template_meat")
        elif category == "fungus":
            add_hits(desc_col.str.contains(rf"^Mushrooms, .*{re.escape(term)}", case=False, regex=True, na=False), "template_mushroom")
        elif category == "algae":
            add_hits(desc_col.str.contains(rf"^Seaweed, .*{re.escape(term)}", case=False, regex=True, na=False), "template_algae")
        else:
            add_hits(desc_col.str.match(rf"^{re.escape(term)}, raw$", case=False, na=False), "template_exact_raw")
            add_hits(desc_col.str.match(rf"^{re.escape(term)}s, raw$", case=False, na=False), "template_plural_raw")
            add_hits(desc_col.str.match(rf"^{re.escape(term)}(,|\s)", case=False, na=False), "template_produce")

    if category == "vegetable" and "lettuce" in cn:
        add_hits(desc_col.str.contains("Lettuce.*raw", case=False, regex=True, na=False), "template_lettuce")
    if category == "vegetable" and "spinach" in cn:
        add_hits(desc_col == "Spinach, raw", "template_exact_raw")

    ranked = sorted(cands.values(), key=lambda c: (-c.score, c.description))
    if ranked:
        top = ranked[0].score
        for c in ranked:
            c.default_raw_preference = c.score >= top - 5 and c.score >= top * 0.85
    return ranked


def filter_guarded_candidates(
    name: str, category: str, cands: list[FdcCandidate], clean: pd.DataFrame,
) -> list[FdcCandidate]:
    out: list[FdcCandidate] = []
    for c in cands:
        ok, _ = passes_guards(name, category, c.description, c.food_category_id, c.fdc_id, clean)
        if ok:
            out.append(c)
    return out


def classify_species(row: pd.Series, clean: pd.DataFrame) -> SpeciesMapping:
    species_id = row["species_node_id"]
    name = row["canonical_name"]
    latin = row.get("latin_name")
    prep = row.get("preparation_labels")
    if prep is None or (isinstance(prep, float) and pd.isna(prep)):
        prep = []
    elif hasattr(prep, "tolist"):
        prep = prep.tolist()
    elif not isinstance(prep, list):
        prep = list(prep) if prep else []
    n_strings = int(row.get("n_strings") or 0)
    category = infer_species_category(name, latin, prep)

    mapping = SpeciesMapping(
        species_id=species_id,
        canonical_name=name,
        latin_name=latin if pd.notna(latin) else None,
        category=category,
        classification="NO-MATCH",
        n_strings=n_strings,
    )

    if species_id in MEAT_REPRESENTATIVE_DEFAULTS:
        spec = MEAT_REPRESENTATIVE_DEFAULTS[species_id]
        pat = spec["fdc_pattern"]
        hits = clean[clean["description"] == pat]
        if hits.empty:
            hits = clean[clean["description"].str.contains(re.escape(pat[:35]), case=False, regex=True, na=False)]
        cands = [
            FdcCandidate(
                fdc_id=int(r["fdc_id"]),
                description=r["description"],
                data_type=r["data_type"],
                food_category_id=int(r["food_category_id"]) if pd.notna(r.get("food_category_id")) else None,
                score=90.0,
                rule_id="judgment_default",
                default_raw_preference=True,
            )
            for _, r in hits.iterrows()
        ]
        guarded = filter_guarded_candidates(name, category, cands, clean)
        mapping.candidates = guarded[:15]
        mapping.classification = "JUDGMENT"
        mapping.judgment_rationale = "Predefined representative FDC entry."
        if guarded:
            mapping.fdc_id = guarded[0].fdc_id
            mapping.fdc_description = guarded[0].description
            mapping.fdc_data_type = guarded[0].data_type
            mapping.rule_id = guarded[0].rule_id
        return mapping

    if name in OUT_OF_SCOPE_JUDGMENT:
        mapping.classification = "OUT_OF_SCOPE"
        mapping.notes = "Processed/composite food with no meaningful species-level FDC base form."
        return mapping

    if name in JUDGMENT_FDC_PATTERNS:
        pat = JUDGMENT_FDC_PATTERNS[name]
        hits = clean[clean["description"] == pat]
        if hits.empty:
            hits = clean[clean["description"].str.contains(re.escape(pat[:30]), case=False, regex=True, na=False)]
        cands = [
            FdcCandidate(
                fdc_id=int(r["fdc_id"]),
                description=r["description"],
                data_type=r["data_type"],
                food_category_id=int(r["food_category_id"]) if pd.notna(r.get("food_category_id")) else None,
                score=85.0,
                rule_id="judgment_pattern",
                default_raw_preference=True,
            )
            for _, r in hits.iterrows()
        ]
        guarded = filter_guarded_candidates(name, category, cands, clean)
        if guarded:
            mapping.classification = "JUDGMENT"
            mapping.candidates = guarded
            mapping.fdc_id = guarded[0].fdc_id
            mapping.fdc_description = guarded[0].description
            mapping.fdc_data_type = guarded[0].data_type
            mapping.rule_id = guarded[0].rule_id
            mapping.judgment_rationale = "Judgment species with guarded default pattern."
            return mapping
        mapping.classification = "JUDGMENT"
        mapping.needs_human_ruling = True
        mapping.notes = "Judgment pattern found no guarded match."
        return mapping

    if category in ("meat", "fish_seafood", "dairy_egg", "beverage", "processed"):
        raw_cands = apply_rule_templates(category, name, search_terms(name, prep), clean)
        guarded = filter_guarded_candidates(name, category, raw_cands, clean)
        mapping.candidates = guarded[:15]
        mapping.classification = "JUDGMENT" if guarded else "NO-MATCH"
        mapping.judgment_rationale = f"Category '{category}' requires representative FDC choice."
        if guarded:
            mapping.fdc_id = guarded[0].fdc_id
            mapping.fdc_description = guarded[0].description
            mapping.fdc_data_type = guarded[0].data_type
            mapping.rule_id = guarded[0].rule_id
            if len(guarded) > 1 and guarded[0].score - guarded[1].score < 5:
                mapping.needs_human_ruling = True
        return mapping

    raw_cands = apply_rule_templates(category, name, search_terms(name, prep), clean)
    guarded = filter_guarded_candidates(name, category, raw_cands, clean)
    mapping.candidates = guarded[:15]

    if not guarded:
        mapping.classification = "NO-MATCH"
        if raw_cands:
            mapping.notes = f"{len(raw_cands)} unguarded candidates rejected by guards."
            mapping.guard_failures = ["all_candidates_rejected"]
        return mapping

    top_tier = [c for c in guarded if c.score >= guarded[0].score - 5]
    if len(top_tier) == 1:
        mapping.classification = "AUTO-CONFIDENT"
        c = top_tier[0]
        mapping.fdc_id = c.fdc_id
        mapping.fdc_description = c.description
        mapping.fdc_data_type = c.data_type
        mapping.rule_id = c.rule_id
        mapping.default_raw_preference = c.default_raw_preference
    else:
        mapping.classification = "AMBIGUOUS"
        c = guarded[0]
        mapping.fdc_id = c.fdc_id
        mapping.fdc_description = c.description
        mapping.fdc_data_type = c.data_type
        mapping.rule_id = c.rule_id
        mapping.default_raw_preference = c.default_raw_preference
        mapping.notes = f"{len(top_tier)} guarded candidates within score band."
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
    clean["food_category_id"] = pd.to_numeric(clean["food_category_id"], errors="coerce")
    clean["norm_desc"] = clean["description"].map(norm)
    before = len(clean)
    clean = clean[~clean["food_category_id"].isin(EXCLUDED_FDC_CATEGORY_IDS)]
    clean = clean[~clean["description"].str.contains(BRAND_PATTERNS, na=False)]
    clean = clean.reset_index(drop=True)
    clean.attrs["excluded_branded_count"] = before - len(clean)
    return clean


def mapping_to_dict(m: SpeciesMapping) -> dict[str, Any]:
    d = asdict(m)
    d["candidates"] = [asdict(c) for c in m.candidates]
    return d


def load_v1_human_rulings() -> dict[str, dict[str, Any]]:
    if not V1_RULINGS_PATH.exists():
        return {}
    cfg = json.loads(V1_RULINGS_PATH.read_text(encoding="utf-8"))
    return {
        sid: r for sid, r in cfg.get("rulings", {}).items()
        if r.get("mapping_class") == "human_ruling"
    }


def load_v2_rulings() -> tuple[set[str], dict[str, dict[str, Any]]]:
    if not V2_RULINGS_PATH.exists():
        return set(), {}
    cfg = json.loads(V2_RULINGS_PATH.read_text(encoding="utf-8"))
    out_of_scope = set(cfg.get("out_of_scope", []))
    rulings = cfg.get("rulings", {})
    return out_of_scope, rulings


def build_approved_map(mappings: list[SpeciesMapping], clean: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    human = load_v1_human_rulings()
    v2_oos, v2_rulings = load_v2_rulings()
    rows: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {
        "auto_confident": 0,
        "ambiguous_resolved": 0,
        "judgment_auto": 0,
        "human_ruling_preserved": 0,
        "human_ruling_v2": 0,
        "out_of_scope": 0,
        "foodb_fallback": 0,
        "unmapped": 0,
        "needs_human": [],
        "out_of_scope_ids": sorted(v2_oos),
    }

    lookup = clean.set_index("fdc_id")
    mapping_by_id = {m.species_id: m for m in mappings}

    for sid in v2_oos:
        m = mapping_by_id.get(sid)
        if m:
            meta["out_of_scope"] += 1

    for sid, ruling in v2_rulings.items():
        m = mapping_by_id.get(sid)
        if not m:
            continue
        fid = int(ruling["fdc_id"])
        if fid not in lookup.index:
            continue
        rows[sid] = {
            "species_id": sid,
            "canonical_name": ruling.get("canonical_name", m.canonical_name),
            "fdc_id": fid,
            "fdc_description": lookup.loc[fid, "description"],
            "fdc_data_type": lookup.loc[fid, "data_type"],
            "mapping_class": ruling.get("mapping_class", "human_ruling_v2"),
            "composition_source": "fdc",
            "coverage": "full",
            "note": ruling.get("note"),
        }
        meta["human_ruling_v2"] += 1

    for m in mappings:
        if m.species_id in rows or m.species_id in v2_oos:
            continue
        if m.species_id in human:
            r = human[m.species_id]
            fid = int(r["fdc_id"])
            if fid in lookup.index:
                desc = lookup.loc[fid, "description"]
                fdc_cat = lookup.loc[fid, "food_category_id"]
                ok, fails = passes_guards(m.canonical_name, m.category, desc, fdc_cat, fid, clean)
                if ok or m.canonical_name in ("Olive", "Capers"):
                    rows[m.species_id] = {
                        "species_id": m.species_id,
                        "canonical_name": m.canonical_name,
                        "fdc_id": fid,
                        "fdc_description": desc,
                        "fdc_data_type": lookup.loc[fid, "data_type"],
                        "mapping_class": "human_ruling_v1",
                        "composition_source": "fdc",
                        "coverage": "full",
                    }
                    meta["human_ruling_preserved"] += 1
                    continue

        if m.classification == "OUT_OF_SCOPE":
            meta["out_of_scope"] += 1
            continue

        if m.classification == "AUTO-CONFIDENT" and m.fdc_id:
            rows[m.species_id] = {
                "species_id": m.species_id,
                "canonical_name": m.canonical_name,
                "fdc_id": m.fdc_id,
                "fdc_description": m.fdc_description,
                "fdc_data_type": m.fdc_data_type,
                "mapping_class": "auto_confident",
                "composition_source": "fdc",
                "coverage": "full",
            }
            meta["auto_confident"] += 1
            continue

        if m.classification in ("AMBIGUOUS", "JUDGMENT") and m.fdc_id and not m.needs_human_ruling:
            rows[m.species_id] = {
                "species_id": m.species_id,
                "canonical_name": m.canonical_name,
                "fdc_id": m.fdc_id,
                "fdc_description": m.fdc_description,
                "fdc_data_type": m.fdc_data_type,
                "mapping_class": "ambiguous_resolved" if m.classification == "AMBIGUOUS" else "judgment_auto",
                "composition_source": "fdc",
                "coverage": "full",
            }
            if m.classification == "AMBIGUOUS":
                meta["ambiguous_resolved"] += 1
            else:
                meta["judgment_auto"] += 1
            continue

        if m.needs_human_ruling:
            meta["needs_human"].append({
                "species_id": m.species_id,
                "canonical_name": m.canonical_name,
                "category": m.category,
                "proposed_fdc_id": m.fdc_id,
                "proposed_fdc_description": m.fdc_description,
                "notes": m.notes,
            })

    species = pd.read_parquet(SPECIES_PATH)
    mapped_ids = set(rows.keys())
    for _, sp in species.iterrows():
        sid = sp["species_node_id"]
        if sid in mapped_ids:
            continue
        meta["unmapped"] += 1

    df = pd.DataFrame(list(rows.values()))
    return df, meta


def load_foodb_nutrients() -> pd.DataFrame:
    content = pd.read_csv(FOODB_CONTENT, low_memory=False)
    nutrients = pd.read_csv(FOODB_NUTRIENT)
    nut = content[content["source_type"].astype(str).str.lower() == "nutrient"].copy()
    nut = nut.merge(nutrients, left_on="source_id", right_on="id", how="inner", suffixes=("", "_nut"))
    nut["nutrient_name"] = nut["name"]
    nut["amount"] = pd.to_numeric(nut["standard_content"], errors="coerce")
    nut["unit"] = nut["orig_unit"].fillna("mg/100g")
    nut["preparation_type"] = nut["preparation_type"].fillna("other")
    pref = nut.sort_values(
        by=["food_id", "nutrient_name", "preparation_type"],
        key=lambda s: s.map(lambda x: 0 if str(x).lower() == "raw" else 1) if s.name == "preparation_type" else s,
    )
    pref = pref.drop_duplicates(subset=["food_id", "nutrient_name"], keep="first")
    return pref


def apply_foodb_fallback(
    approved: pd.DataFrame, species: pd.DataFrame, exclude_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclude_ids = exclude_ids or set()
    mapped_ids = set(approved["species_id"]) if len(approved) else set()
    unmapped = species[
        ~species["species_node_id"].isin(mapped_ids | exclude_ids)
    ].copy()
    foodb = load_foodb_nutrients()
    fb_rows = []
    profile_parts = []
    for _, sp in unmapped.iterrows():
        fid = int(sp["foodb_id"]) if pd.notna(sp["foodb_id"]) else None
        if not fid:
            continue
        sub = foodb[foodb["food_id"] == fid]
        if sub.empty:
            continue
        fb_rows.append({
            "species_id": sp["species_node_id"],
            "canonical_name": sp["canonical_name"],
            "fdc_id": None,
            "fdc_description": None,
            "fdc_data_type": None,
            "foodb_id": fid,
            "mapping_class": "foodb_fallback",
            "composition_source": "foodb",
            "coverage": "partial",
            "n_foodb_nutrients": int(sub["nutrient_name"].nunique()),
        })
        for _, nr in sub.iterrows():
            profile_parts.append({
                "species_id": sp["species_node_id"],
                "canonical_name": sp["canonical_name"],
                "fdc_id": None,
                "fdc_description": f"FooDB food_id={fid}",
                "fdc_data_type": None,
                "mapping_class": "foodb_fallback",
                "provenance": "foodb_partial",
                "composition_source": "foodb",
                "coverage": "partial",
                "nutrient_id": f"FDBN_{int(nr['source_id'])}",
                "nutrient_name": nr["nutrient_name"],
                "nutrient_group": "foodb",
                "amount": nr["amount"],
                "unit": nr["unit"],
                "basis": "per_100g",
            })
    fb_df = pd.DataFrame(fb_rows)
    if len(approved):
        out_map = pd.concat([approved, fb_df], ignore_index=True)
    else:
        out_map = fb_df
    profiles = pd.DataFrame(profile_parts)
    return out_map, profiles


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


def nutrient_group(name: str) -> str:
    if name in MACRO_IDS or any(k in name for k in ("Energy", "Protein", "Fiber", "Water", "Carbohydrate", "Sugars")):
        if "Fatty" not in name and "Amino" not in name:
            return "macro"
    if name in MINERAL_NAMES:
        return "mineral"
    if any(k in name for k in VITAMIN_KEYWORDS):
        return "vitamin"
    if "Fatty acid" in name:
        return "fatty_acid"
    return "other"


def get_out_of_scope_ids(mappings: list[SpeciesMapping]) -> set[str]:
    v2_oos, _ = load_v2_rulings()
    oos = set(v2_oos)
    for m in mappings:
        if m.classification == "OUT_OF_SCOPE":
            oos.add(m.species_id)
    return oos


def report_unmapped_species(
    full_map: pd.DataFrame, species: pd.DataFrame, oos_ids: set[str],
) -> list[dict[str, Any]]:
    mapped = set(full_map["species_id"]) if len(full_map) else set()
    rows = []
    for _, sp in species.iterrows():
        sid = sp["species_node_id"]
        if sid in mapped or sid in oos_ids:
            continue
        rows.append({
            "species_id": sid,
            "canonical_name": sp["canonical_name"],
            "latin_name": sp.get("latin_name") if pd.notna(sp.get("latin_name")) else None,
            "n_strings": int(sp.get("n_strings") or 0),
            "foodb_id": int(sp["foodb_id"]) if pd.notna(sp.get("foodb_id")) else None,
        })
    return sorted(rows, key=lambda x: -x["n_strings"])


def lock_production_outputs(summary: dict[str, Any], unmapped_report: list[dict[str, Any]]) -> dict[str, Any]:
    """Copy locked v2 artifacts to production filenames and write manifest."""
    import shutil

    prod_map = OUT_DIR / "species_fdc_map_production.parquet"
    prod_profiles = OUT_DIR / "species_nutrient_profiles_production.parquet"
    prod_summary = OUT_DIR / "species_nutrient_summary_production.json"

    shutil.copy2(OUT_DIR / "species_fdc_map_v2.parquet", prod_map)
    shutil.copy2(OUT_DIR / "species_nutrient_profiles_v2.parquet", prod_profiles)
    shutil.copy2(OUT_DIR / "species_nutrient_summary_v2.json", prod_summary)

    manifest = {
        "status": "LOCKED",
        "version": "v2.1",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "rulings": str(V2_RULINGS_PATH),
        "coverage": summary.get("coverage"),
        "compose": summary.get("compose"),
        "unmapped_species": unmapped_report,
        "unmapped_count": len(unmapped_report),
        "artifacts": {
            "species_fdc_map": str(prod_map),
            "species_nutrient_profiles": str(prod_profiles),
            "species_nutrient_summary": str(prod_summary),
            "fdc_clean_pool": str(OUT_DIR / "fdc_clean_pool_v2.parquet"),
            "review": str(OUT_DIR / "species_fdc_mapping_review_v2.json"),
            "v2_map": str(OUT_DIR / "species_fdc_map_v2.parquet"),
            "v2_profiles": str(OUT_DIR / "species_nutrient_profiles_v2.parquet"),
        },
        "notes": "Production nutrient layer. v1 artifacts preserved for comparison.",
    }
    PRODUCTION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_compose_phase(approved: pd.DataFrame, exclude_ids: set[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    fdc_map = approved[approved["composition_source"] == "fdc"].copy()
    clean = pd.read_parquet(OUT_DIR / "fdc_clean_pool_v2.parquet")
    fn = pd.read_parquet(FOOD_NUTRIENT_PARQUET)
    with zipfile.ZipFile(USDA_ZIP) as z:
        nutrients = pd.read_csv(io.BytesIO(z.read(USDA_PREFIX + "nutrient.csv")))

    merged = fdc_map.merge(
        clean[["fdc_id", "data_type", "description"]].rename(columns={"description": "fdc_description_clean"}),
        on="fdc_id", how="left",
    )
    if "fdc_description" not in merged.columns or merged["fdc_description"].isna().any():
        merged["fdc_description"] = merged["fdc_description"].fillna(merged["fdc_description_clean"])

    fn_sub = fn[fn["fdc_id"].isin(set(merged["fdc_id"]))].merge(
        nutrients[["id", "name", "unit_name"]], left_on="nutrient_id", right_on="id", how="left",
    )
    sub = merged.merge(fn_sub, on="fdc_id", how="inner")
    sub = sub.rename(columns={"name": "nutrient_name", "unit_name": "unit"})
    sub["nutrient_group"] = sub["nutrient_name"].map(lambda x: nutrient_group(str(x)))
    sub["basis"] = "per_100g"
    sub["provenance"] = sub["fdc_data_type"].map(
        lambda x: "foundation_lab_quality" if x == "foundation_food" else "sr_legacy"
    )
    sub["composition_source"] = "fdc"
    sub["coverage"] = "full"

    out_cols = [
        "species_id", "canonical_name", "fdc_id", "fdc_description", "fdc_data_type",
        "mapping_class", "provenance", "composition_source", "coverage",
        "nutrient_id", "nutrient_name", "nutrient_group", "amount", "unit", "basis",
    ]
    fdc_profiles = sub[out_cols].copy()

    species = pd.read_parquet(SPECIES_PATH)
    full_map, foodb_profiles = apply_foodb_fallback(approved, species, exclude_ids=exclude_ids)
    profiles = pd.concat([fdc_profiles, foodb_profiles], ignore_index=True) if len(foodb_profiles) else fdc_profiles

    profiles.to_parquet(OUT_DIR / "species_nutrient_profiles_v2.parquet", index=False)
    full_map.to_parquet(OUT_DIR / "species_fdc_map_v2.parquet", index=False)

    return profiles, {
        "fdc_species": int(fdc_map["species_id"].nunique()),
        "foodb_fallback_species": int((full_map["composition_source"] == "foodb").sum()),
        "total_with_composition": int(full_map["species_id"].nunique()),
        "total_nutrient_rows": len(profiles),
    }


def run_review_phase() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    species = pd.read_parquet(SPECIES_PATH)
    clean = load_clean_pool()
    clean.to_parquet(OUT_DIR / "fdc_clean_pool_v2.parquet", index=False)

    mappings = [classify_species(row, clean) for _, row in species.iterrows()]
    counts = Counter(m.classification for m in mappings)

    approved_fdc, build_meta = build_approved_map(mappings, clean)
    oos_ids = get_out_of_scope_ids(mappings)
    species_all = pd.read_parquet(SPECIES_PATH)
    full_map, _ = apply_foodb_fallback(approved_fdc, species_all, exclude_ids=oos_ids)
    unmapped_report = report_unmapped_species(full_map, species_all, oos_ids)

    regression_targets = [
        "Sweet basil", "Lemon balm", "Papaya", "Pizza", "Spelt", "Blue cheese", "Wild boar",
        "Turmeric", "Lettuce", "Spinach", "Garlic", "Broccoli", "Almond",
    ]
    v1 = pd.read_parquet(OUT_DIR / "species_fdc_map_approved_v1.parquet") if (OUT_DIR / "species_fdc_map_approved_v1.parquet").exists() else None
    regression = {}
    for name in regression_targets:
        m = next((x for x in mappings if x.canonical_name == name), None)
        v2_row = full_map[full_map["canonical_name"] == name]
        entry = {
            "v2_classification": m.classification if m else None,
            "v2_fdc": v2_row.iloc[0]["fdc_description"] if len(v2_row) and pd.notna(v2_row.iloc[0].get("fdc_description")) else (
                f"FooDB fallback (foodb_id={v2_row.iloc[0].get('foodb_id')})" if len(v2_row) else None
            ),
            "v2_source": v2_row.iloc[0]["composition_source"] if len(v2_row) else None,
        }
        if v1 is not None:
            v1_row = v1[v1["canonical_name"] == name]
            if len(v1_row):
                entry["v1_fdc"] = v1_row.iloc[0]["fdc_description"]
        regression[name] = entry

    no_match_names = [m.canonical_name for m in mappings if m.classification == "NO-MATCH"]
    recovered_from_audit = [
        n for n in (
            "Leek", "Shallot", "Chickpea", "Flaxseed", "Pecan nut", "Fig", "Clam", "Cashew nut",
            "Jicama", "Boysenberry", "Shiitake", "Haddock", "Grouper", "Snapper",
        )
        if n not in no_match_names and n in set(full_map["canonical_name"])
    ]

    review = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "v2.1",
        "guards": [
            "head_noun_plural_tolerant",
            "category_consistency",
            "modifier_collision_blocklist",
            "form_preference_enforced",
            "branded_restaurant_pool_exclusion",
        ],
        "category_taxonomy": list(SPECIES_CATEGORIES),
        "clean_pool_size": len(clean),
        "excluded_branded_or_restaurant": int(clean.attrs.get("excluded_branded_count", 0)),
        "classification_counts": dict(counts),
        "approved_build_meta": build_meta,
        "coverage": {
            "fdc_mapped": int((full_map["composition_source"] == "fdc").sum()),
            "foodb_fallback": int((full_map["composition_source"] == "foodb").sum()),
            "out_of_scope": len(oos_ids),
            "still_unmapped": len(unmapped_report),
            "total_with_composition": int(full_map["species_id"].nunique()),
        },
        "unmapped_species": unmapped_report,
        "recovered_no_match_examples": recovered_from_audit,
        "no_match_remaining": no_match_names[:40],
        "no_match_count": len(no_match_names),
        "regression_check": regression,
        "needs_human_ruling": build_meta.get("needs_human", []),
        "auto_confident": [mapping_to_dict(m) for m in mappings if m.classification == "AUTO-CONFIDENT"],
        "ambiguous": [mapping_to_dict(m) for m in mappings if m.classification == "AMBIGUOUS"],
        "judgment": [mapping_to_dict(m) for m in mappings if m.classification == "JUDGMENT"],
        "no_match": [mapping_to_dict(m) for m in mappings if m.classification == "NO-MATCH"],
        "out_of_scope": [mapping_to_dict(m) for m in mappings if m.classification == "OUT_OF_SCOPE"],
    }
    (OUT_DIR / "species_fdc_mapping_review_v2.json").write_text(json.dumps(review, indent=2), encoding="utf-8")

    approved_fdc.to_parquet(OUT_DIR / "species_fdc_map_fdc_only_v2.parquet", index=False)
    full_map.to_parquet(OUT_DIR / "species_fdc_map_v2.parquet", index=False)

    compose_stats = run_compose_phase(full_map, exclude_ids=oos_ids)

    summary = {
        "generated_at": review["generated_at"],
        "version": "v2.1",
        "status": "LOCKED",
        "classification_counts": dict(counts),
        "approved_build_meta": build_meta,
        "coverage": review["coverage"],
        "unmapped_species": unmapped_report,
        "regression_check": regression,
        "compose": compose_stats[1],
        "outputs": {
            "clean_pool_v2": str(OUT_DIR / "fdc_clean_pool_v2.parquet"),
            "map_v2": str(OUT_DIR / "species_fdc_map_v2.parquet"),
            "profiles_v2": str(OUT_DIR / "species_nutrient_profiles_v2.parquet"),
            "review_v2": str(OUT_DIR / "species_fdc_mapping_review_v2.json"),
        },
    }
    (OUT_DIR / "species_nutrient_summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = lock_production_outputs(summary, unmapped_report)
    summary["production_manifest"] = manifest
    (OUT_DIR / "species_nutrient_summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build species→FDC map v2")
    parser.add_argument("--phase", choices=["review", "compose", "all"], default="all")
    args = parser.parse_args()
    if args.phase in ("review", "all"):
        summary = run_review_phase()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
