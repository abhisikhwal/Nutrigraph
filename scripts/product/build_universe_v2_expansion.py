#!/usr/bin/env python3
"""
Build ingredient universe v2 expansion (review gate — does NOT merge into mechanism graph).

Outputs to data/processed/product/universe_v2/:
  ingredient_nodes_v2_draft.parquet
  ingredient_nodes_v2_draft.json
  blend_constituents_v2_draft.json
  latin_recovery_v2_draft.json
  alias_additions_v2_draft.json
  universe_v2_expansion_report.json
  UNIVERSE_V2_REVIEW.md
"""
from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/product/universe_v2"
RAW = ROOT / "data/raw/new_recipes"
CANON = ROOT / "data/processed/canonical"
PROD = ROOT / "data/processed/product"
FOODB_PATH = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Food.csv"
FDC_POOL_PATH = PROD / "nutrients/fdc_clean_pool_v2.parquet"

# ---------------------------------------------------------------------------
# Reuse recon normalization / clustering (from unmapped_characterize.py)
# ---------------------------------------------------------------------------

QTY_PREFIX = re.compile(
    r"^[\d\s/\.¼½¾]+(?:cup|cups|tablespoon|tablespoons|teaspoon|teaspoons|tbsp|tsp|"
    r"oz|ounce|ounces|pound|pounds|lb|lbs|gram|grams|g|kg|ml|l|liter|litre|inch|inches|"
    r"clove|cloves|sprig|sprigs|pinch|bunch|can|cans|package|packages|slice|slices|head|"
    r"heads|stick|sticks|piece|pieces|bag|bags|jar|jars|bottle|bottles|packet|packets|"
    r"ounce|ounces|fluid)\b[\s,-]*",
    re.I,
)
LEADING_NUM = re.compile(r"^[\d\s/\.¼½¾-]+")
JUNK_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^\d+\s*(cup|cups|tbsp|tsp|oz|lb|g|ml)s?$", re.I),
    re.compile(r"^(to taste|as needed|as required|optional|na|none|null|n/a)$", re.I),
    re.compile(r"^(for garnish|for topping|for the sauce|for the dough|for filling)$", re.I),
    re.compile(r"^[\W_]+$"),
    re.compile(r"^.{0,1}$"),
]
MODIFIERS = {
    "large", "small", "medium", "fresh", "dried", "dry", "frozen", "canned", "chopped",
    "diced", "minced", "sliced", "grated", "ground", "crushed", "whole", "raw", "ripe",
    "optional", "organic", "unsalted", "salted", "boneless", "skinless", "lean", "fat-free",
    "low-fat", "reduced-fat", "nonfat", "light", "dark", "white", "black", "green", "yellow",
    "red", "brown", "extra", "virgin", "cold-pressed", "warm", "hot", "cold", "soft",
    "firm", "thin", "thick", "finely", "roughly", "coarsely", "packed", "heaping",
    "scant", "level", "generous", "about", "approx", "approximately", "peeled", "seeded",
    "deveined", "trimmed", "halved", "quartered", "cubed", "shredded", "crumbled",
    "melted", "softened", "room", "temperature", "boiling", "cooked", "uncooked",
    "instant", "quick", "old-fashioned", "plain", "unsweetened", "sweetened",
}
DESCRIPTOR_SUFFIX = re.compile(
    r"\s+(to taste|as needed|as required|optional|for garnish|for serving|or to taste|"
    r"adjustable|enough|if needed)$",
    re.I,
)
HEAD_RULES = [
    (re.compile(r"soy\s+sauce", re.I), "soy sauce"),
    (re.compile(r"fish\s+sauce", re.I), "fish sauce"),
    (re.compile(r"oyster\s+sauce", re.I), "oyster sauce"),
    (re.compile(r"hoisin\s+sauce", re.I), "hoisin sauce"),
    (re.compile(r"worcestershire\s+sauce", re.I), "worcestershire sauce"),
    (re.compile(r"hot\s+sauce", re.I), "hot sauce"),
    (re.compile(r"baking\s+powder", re.I), "baking powder"),
    (re.compile(r"baking\s+soda", re.I), "baking soda"),
    (re.compile(r"cooking\s+spray", re.I), "cooking spray"),
    (re.compile(r"active\s+dry\s+yeast|dry\s+yeast|instant\s+yeast", re.I), "yeast"),
    (re.compile(r"curry\s+leaves?", re.I), "curry leaf"),
    (re.compile(r"kaffir\s+lime\s+leaves?", re.I), "kaffir lime leaf"),
    (re.compile(r"lemongrass|lemon\s+grass", re.I), "lemongrass"),
    (re.compile(r"gochujang", re.I), "gochujang"),
    (re.compile(r"gochugaru", re.I), "gochugaru"),
    (re.compile(r"mirin", re.I), "mirin"),
    (re.compile(r"garam\s+masala", re.I), "garam masala"),
    (re.compile(r"biryani\s+masala", re.I), "biryani masala"),
    (re.compile(r"curry\s+powder", re.I), "curry powder"),
    (re.compile(r"italian\s+seasoning", re.I), "italian seasoning"),
    (re.compile(r"cajun\s+seasoning", re.I), "cajun seasoning"),
    (re.compile(r"five\s+spice|five-spice", re.I), "five spice"),
    (re.compile(r"all[- ]purpose\s+flour", re.I), "flour"),
    (re.compile(r"plain\s+flour", re.I), "flour"),
    (re.compile(r"self[- ]raising\s+flour", re.I), "flour"),
    (re.compile(r"garlic\s+cloves?|cloves?\s+garlic|minced\s+garlic|garlic\s+minced", re.I), "garlic"),
    (re.compile(r"large\s+eggs?|egg\s+large", re.I), "egg"),
    (re.compile(r"vanilla\s+extract", re.I), "vanilla extract"),
    (re.compile(r"lemon\s+juice|lime\s+juice", re.I), "citrus juice"),
    (re.compile(r"olive\s+oil\s+cooking\s+spray", re.I), "cooking spray"),
    (re.compile(r"brinjal", re.I), "eggplant"),
    (re.compile(r"capsicum|bell\s+pepper", re.I), "bell pepper"),
    (re.compile(r"beetroot|beet\s+root", re.I), "beetroot"),
    (re.compile(r"coriander\s+leaves|cilantro", re.I), "coriander"),
    (re.compile(r"paneer", re.I), "paneer"),
    (re.compile(r"\bghee\b", re.I), "ghee"),
    (re.compile(r"jaggery", re.I), "jaggery"),
    (re.compile(r"galangal", re.I), "galangal"),
    (re.compile(r"\bamla\b", re.I), "amla"),
    (re.compile(r"bottle\s+gourd|lauki", re.I), "bottle gourd"),
    (re.compile(r"bok\s+choy|pak\s+choi", re.I), "bok choy"),
    (re.compile(r"ajwain|carom\s+seeds?", re.I), "ajwain"),
    (re.compile(r"tumeric|turmeric", re.I), "turmeric"),
    (re.compile(r"mayonnaise|mayonaise", re.I), "mayonnaise"),
    (re.compile(r"ketchup", re.I), "ketchup"),
    (re.compile(r"vegetable\s+broth|veggie\s+broth|vegetable\s+stock", re.I), "vegetable broth"),
    (re.compile(r"chicken\s+broth|chicken\s+stock", re.I), "chicken broth"),
    (re.compile(r"\bsalsa\b", re.I), "salsa"),
]

ALIAS_OF_EXISTING: dict[str, str | None] = {
    "brinjal": "eggplant", "eggplant": "eggplant", "aubergine": "eggplant",
    "capsicum": "bell pepper", "bell pepper": "bell pepper",
    "beetroot": None,
    "tumeric": "turmeric", "turmeric": "turmeric",
    "coriander": "coriander", "cilantro": "coriander",
    "plain flour": "flour", "all-purpose flour": "flour", "all purpose flour": "flour",
    "self raising flour": "flour", "self-raising flour": "flour",
    "garlic": "garlic", "large egg": "egg", "eggs": "egg", "egg": "egg",
    "purple onion": "onion", "yellow onion": "onion", "red onion": "onion",
    "diced tomatoes": "tomato", "grated parmesan cheese": "parmesan cheese",
    "shredded cheddar cheese": "cheddar cheese",
    "chili powder": "chili", "garlic powder": "garlic", "dried oregano": "oregano",
    "vanilla extract": "vanilla", "lemon juice": "lemon", "lime juice": "lime",
    "flour": "flour", "all purpose flour": "flour",
    "tomato paste": "garden tomato", "tomato sauce": "garden tomato",
    "green onion": "onion", "spring onion": "onion", "scallion": "onion",
    "green bell pepper": "bell pepper", "red bell pepper": "bell pepper",
    "yellow bell pepper": "bell pepper",
    "yoghurt": "yogurt", "yogurt": "yogurt",
    "chilly": "chili", "chilli": "chili",
    "lemongrass": "lemon grass",
}

# Tier D: explicit alias head -> existing canonical name (resolved to SP_* at runtime)
TIER_D_ALIASES: dict[str, str] = {
    "brinjal": "eggplant", "aubergine": "eggplant",
    "capsicum": "bell pepper",
    "tumeric": "turmeric",
    "cilantro": "coriander",
    "plain flour": "flour", "all-purpose flour": "flour", "all purpose flour": "flour",
    "self raising flour": "flour", "self-raising flour": "flour",
    "flour": "flour",
    "garlic": "garlic", "large egg": "egg", "eggs": "egg", "egg": "egg",
    "purple onion": "onion", "yellow onion": "onion", "red onion": "onion",
    "green onion": "onion", "spring onion": "onion", "scallion": "onion",
    "green bell pepper": "bell pepper", "red bell pepper": "bell pepper",
    "yellow bell pepper": "bell pepper",
    "diced tomatoes": "garden tomato", "tomato paste": "garden tomato", "tomato sauce": "garden tomato",
    "grated parmesan cheese": "parmesan cheese", "shredded cheddar cheese": "cheddar cheese",
    "chili powder": "chili", "garlic powder": "garlic", "dried oregano": "oregano",
    "vanilla extract": "vanilla", "lemon juice": "lemon", "lime juice": "lime",
    "coriander": "coriander",
    "bell pepper": "bell pepper",
    "yoghurt": "yogurt", "yogurt": "yogurt",
    "chilly": "chili", "chilli": "chili",
    "lemongrass": "lemon grass", "lemon grass": "lemon grass",
    # high-confidence alias heads from recon (curated subset of 52 fuzzy-detected)
    "turmeric": "turmeric", "tumeric": "turmeric",
    "parmesan": "parmesan cheese", "mozzarella": "mozzarella cheese",
    "cheddar": "cheddar cheese", "feta": "greek feta cheese", "greek feta": "greek feta cheese",
    "pita": "pita bread", "clam": "clam", "cod": "pacific cod",
    "soy": "soy bean", "curry": "curry powder",
    "brinjal": "eggplant", "aubergine": "eggplant",
    "capsicum": "green bell pepper",
    "tomato paste": "garden tomato", "tomato sauce": "garden tomato",
    "diced tomatoes": "garden tomato",
    "garlic powder": "garlic", "chili powder": "chili", "dried oregano": "oregano",
    "vanilla extract": "vanilla",
    "ice cub": "ice", "ice cube": "ice", "ice cubes": "ice",
}

# Latin recovery candidates (Tier C policy b — whole species)
LATIN_RECOVERY: dict[str, dict[str, Any]] = {
    "curry leaf": {
        "latin": "Murraya koenigii",
        "synonyms": ["curry leaf", "curry leaves", "sweet neem leaf", "kadi patta"],
        "category": "spice_herb",
    },
    "galangal": {
        "latin": "Alpinia galanga",
        "synonyms": ["galangal", "galanga", "greater galangal", "laos"],
        "category": "spice_herb",
    },
    "ajwain": {
        "latin": "Trachyspermum ammi",
        "synonyms": ["ajwain", "carom seed", "carom seeds", "bishop's weed", "ajowan"],
        "category": "spice_herb",
    },
    "amla": {
        "latin": "Phyllanthus emblica",
        "synonyms": ["amla", "indian gooseberry", "emblic", "amalaki"],
        "category": "fruit",
    },
    "bok choy": {
        "latin": "Brassica rapa subsp. chinensis",
        "synonyms": ["bok choy", "pak choi", "pak choy", "chinese cabbage", "bok choi"],
        "category": "vegetable",
    },
    "lemongrass": {
        "latin": "Cymbopogon citratus",
        "synonyms": ["lemongrass", "lemon grass", "citronella grass"],
        "category": "spice_herb",
    },
    "kaffir lime leaf": {
        "latin": "Citrus hystrix",
        "synonyms": ["kaffir lime leaf", "makrut lime leaf", "kaffir lime leaves"],
        "category": "spice_herb",
    },
    "bottle gourd": {
        "latin": "Lagenaria siceraria",
        "synonyms": ["bottle gourd", "lauki", "calabash", "opo squash"],
        "category": "vegetable",
    },
    "tomatillo": {
        "latin": "Physalis philadelphica",
        "synonyms": ["tomatillo", "mexican husk tomato", "tomatillos"],
        "category": "vegetable",
    },
    "beetroot": {
        "latin": "Beta vulgaris",
        "synonyms": ["beetroot", "beet root", "red beet", "beet"],
        "category": "vegetable",
    },
    "prosciutto": {
        "latin": None,
        "synonyms": ["prosciutto", "prosciutto ham", "parma ham"],
        "category": "processed",
    },
    "gochugaru": {
        "latin": "Capsicum annuum",
        "synonyms": ["gochugaru", "korean red pepper flakes", "korean chili flakes"],
        "category": "spice_herb",
    },
}

# Blend decompositions (Tier C policy b) — standard published approximations
# Sources noted in blend_constituents output
BLEND_COMPOSITIONS: dict[str, dict[str, Any]] = {
    "garam masala": {
        "source": "Typical North Indian garam masala blend (McGee, Spiceography, common commercial labels)",
        "constituents": [
            {"name": "cumin", "weight": 0.20},
            {"name": "coriander", "weight": 0.20},
            {"name": "cardamom", "weight": 0.15},
            {"name": "cloves", "weight": 0.10},
            {"name": "pepper", "weight": 0.10},
            {"name": "cinnamon", "weight": 0.15},
            {"name": "nutmeg", "weight": 0.05},
            {"name": "mace", "weight": 0.05},
        ],
    },
    "biryani masala": {
        "source": "Common biryani masala formulations (Tarla Dalal, commercial blends)",
        "constituents": [
            {"name": "cumin", "weight": 0.15},
            {"name": "coriander", "weight": 0.15},
            {"name": "cardamom", "weight": 0.10},
            {"name": "cloves", "weight": 0.08},
            {"name": "cinnamon", "weight": 0.10},
            {"name": "pepper", "weight": 0.08},
            {"name": "nutmeg", "weight": 0.05},
            {"name": "mace", "weight": 0.05},
            {"name": "turmeric", "weight": 0.12},
            {"name": "ginger", "weight": 0.10},
            {"name": "garlic", "weight": 0.07},
            {"name": "chili", "weight": 0.05},
        ],
    },
    "curry powder": {
        "source": "British-style curry powder (McCormick, Wikipedia standard blend)",
        "constituents": [
            {"name": "turmeric", "weight": 0.30},
            {"name": "coriander", "weight": 0.20},
            {"name": "cumin", "weight": 0.15},
            {"name": "fenugreek", "weight": 0.10},
            {"name": "pepper", "weight": 0.05},
            {"name": "ginger", "weight": 0.05},
            {"name": "mustard", "weight": 0.05},
            {"name": "cinnamon", "weight": 0.05},
            {"name": "cloves", "weight": 0.03},
            {"name": "cardamom", "weight": 0.02},
        ],
    },
    "italian seasoning": {
        "source": "Standard Italian seasoning blend (McCormick, Spiceography)",
        "constituents": [
            {"name": "oregano", "weight": 0.30},
            {"name": "basil", "weight": 0.25},
            {"name": "thyme", "weight": 0.15},
            {"name": "rosemary", "weight": 0.15},
            {"name": "marjoram", "weight": 0.10},
            {"name": "sage", "weight": 0.05},
        ],
    },
    "cajun seasoning": {
        "source": "Typical Cajun/Creole seasoning (Tony Chachere-style, AllRecipes standard)",
        "constituents": [
            {"name": "paprika", "weight": 0.30},
            {"name": "garlic", "weight": 0.15},
            {"name": "onion", "weight": 0.10},
            {"name": "pepper", "weight": 0.10},
            {"name": "oregano", "weight": 0.10},
            {"name": "thyme", "weight": 0.10},
            {"name": "salt", "weight": 0.10},
            {"name": "chili", "weight": 0.05},
        ],
    },
    "five spice": {
        "source": "Chinese five-spice powder standard (Wikipedia, Fuchsia Dunlop)",
        "constituents": [
            {"name": "star anise", "weight": 0.25},
            {"name": "cloves", "weight": 0.20},
            {"name": "cinnamon", "weight": 0.20},
            {"name": "pepper", "weight": 0.20},
            {"name": "fennel", "weight": 0.15},
        ],
    },
    "pumpkin pie spice": {
        "source": "Standard pumpkin pie spice (McCormick)",
        "constituents": [
            {"name": "cinnamon", "weight": 0.50},
            {"name": "ginger", "weight": 0.20},
            {"name": "nutmeg", "weight": 0.15},
            {"name": "allspice", "weight": 0.10},
            {"name": "cloves", "weight": 0.05},
        ],
    },
    "chaat masala": {
        "source": "Typical chaat masala blend (Tarla Dalal, commercial MDH-style)",
        "constituents": [
            {"name": "cumin", "weight": 0.20},
            {"name": "coriander", "weight": 0.15},
            {"name": "mango", "weight": 0.15},
            {"name": "pepper", "weight": 0.10},
            {"name": "ginger", "weight": 0.10},
            {"name": "salt", "weight": 0.15},
            {"name": "chili", "weight": 0.10},
            {"name": "mint", "weight": 0.05},
        ],
    },
    "taco seasoning": {
        "source": "Standard taco seasoning mix (McCormick)",
        "constituents": [
            {"name": "chili", "weight": 0.25},
            {"name": "cumin", "weight": 0.20},
            {"name": "paprika", "weight": 0.15},
            {"name": "garlic", "weight": 0.10},
            {"name": "onion", "weight": 0.10},
            {"name": "oregano", "weight": 0.10},
            {"name": "salt", "weight": 0.10},
        ],
    },
    "old bay seasoning": {
        "source": "Old Bay Seasoning ingredient list (McCormick)",
        "constituents": [
            {"name": "paprika", "weight": 0.20},
            {"name": "pepper", "weight": 0.15},
            {"name": "salt", "weight": 0.25},
            {"name": "mustard", "weight": 0.10},
            {"name": "celery", "weight": 0.10},
            {"name": "ginger", "weight": 0.05},
            {"name": "cloves", "weight": 0.05},
            {"name": "bay", "weight": 0.05},
            {"name": "cardamom", "weight": 0.05},
        ],
    },
}

# Curated FDC patterns for composites/additives (same discipline as build_species_fdc_map_v2)
COMPOSITE_FDC_PATTERNS: dict[str, str] = {
    "soy sauce": "Sauce, soy, ready-to-serve",
    "fish sauce": "Sauce, fish, ready-to-serve",
    "oyster sauce": "Sauce, oyster, ready-to-serve",
    "hoisin sauce": "Sauce, hoisin, ready-to-serve",
    "worcestershire sauce": "Sauce, worcestershire",
    "ketchup": "Catsup",
    "mayonnaise": "Salad dressing, mayonnaise, regular",
    "salsa": "Sauce, salsa, ready-to-serve",
    "hot sauce": "Sauce, hot chile, sriracha",
    "sriracha": "Sauce, hot chile, sriracha",
    "baking powder": "Leavening agents, baking powder, double-acting, sodium aluminum sulfate",
    "baking soda": "Leavening agents, baking soda",
    "yeast": "Leavening agents, yeast, baker's, compressed",
    "cooking spray": "Oil, cooking, spray",
    "ghee": "Butter, Clarified butter (ghee)",
    "vegetable broth": "Soup, vegetable broth, ready to serve",
    "chicken broth": "Soup, chicken broth, ready-to-serve",
    "beef broth": "Soup, beef broth, bouillon and consomme, canned, ready-to-serve",
    "cream cheese": "Cheese, cream",
    "sour cream": "Cream, sour, cultured",
    "peanut butter": "Peanut butter, smooth style, with salt",
    "tomato paste": "Tomato products, canned, paste, without salt added",
    "barbecue sauce": "Sauce, barbecue",
    "bbq sauce": "Sauce, barbecue",
    "teriyaki sauce": "Sauce, teriyaki, ready-to-serve",
    "tahini": "Seeds, sesame butter, tahini, from roasted and toasted kernels",
    "jaggery": "Sugars, granulated",
    "paneer": "Cheese, paneer",
    "prosciutto": "Ham, prosciutto, imported",
    "gochujang": "Sauce, chili, hot",
    "corn starch": "Cornstarch",
    "cornstarch": "Cornstarch",
    "cream of tartar": "Leavening agents, cream of tartar",
    "xanthan gum": "Xanthan gum",
    "gelatin": "Gelatin, dry powder, unsweetened",
    "bok choy": "Cabbage, bok choy, raw",
    "tomatillo": "Tomatillos, raw",
    "beetroot": "Beets, raw",
    "lemongrass": "Lemon grass (citronella), raw",
    "bottle gourd": "Gourd, white-flowered (calabash), raw",
}

COMPOSITE_FOODB_PATTERNS: dict[str, str] = {
    "soy sauce": "Soy sauce",
    "fish sauce": None,  # no FooDB entry; do not proxy to soy sauce
    "ketchup": "Ketchup",
    "miso": "Miso",
    "tamarind": "Tamarind",
    "lemongrass": "Lemon grass",
    "bok choy": "Chinese cabbage",
    "tomatillo": "Mexican groundcherry",
    "beetroot": "Red beetroot",
    "bottle gourd": "Calabash",
    "worcestershire sauce": "Worcestershire sauce",
    "hoisin sauce": "Hoisin sauce",
    "oyster sauce": "Oyster sauce",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def ascii_norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower().strip())


def strip_quantity(raw: str) -> str:
    s = str(raw).strip()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"^[\u00bc-\u00be\u2150-\u215e\ufffd\s]+", "", s)
    s = QTY_PREFIX.sub("", s)
    s = LEADING_NUM.sub("", s).strip(" ,-")
    s = re.sub(r"^[^\w]+", "", s)
    s = DESCRIPTOR_SUFFIX.sub("", s)
    return norm(s)


def strip_modifiers_tokens(s: str) -> str:
    toks = s.split()
    while toks and toks[0] in MODIFIERS:
        toks = toks[1:]
    while toks and toks[-1] in MODIFIERS:
        toks = toks[:-1]
    return " ".join(toks) if toks else s


def singularize(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("es") and len(tok) > 3:
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def cluster_head(raw: str) -> str:
    s = strip_quantity(raw)
    s = strip_modifiers_tokens(s)
    for pat, head in HEAD_RULES:
        if pat.search(s):
            return head
    s = re.sub(
        r"\s+(chopped|diced|minced|sliced|grated|ground|crushed|peeled|seeded|optional)$",
        "",
        s,
    )
    return s if s else norm(raw)


def cluster_key(head: str) -> str:
    return " ".join(singularize(t) for t in head.split())


def is_junk(raw: str) -> bool:
    s = norm(raw)
    if not s or len(s) <= 1:
        return True
    for pat in JUNK_PATTERNS:
        if pat.match(s):
            return True
    if re.match(r"^for the\b", s):
        return True
    if re.match(r"^[\d\s/\.¼½¾-]+$", s):
        return True
    if "taste" in s and ("salt" in s or "pepper" in s) and len(s.split()) <= 6:
        return True
    return False


def classify_type(head: str, raw: str) -> str:
    h = head.lower()
    if h in ALIAS_OF_EXISTING and ALIAS_OF_EXISTING[h] and ALIAS_OF_EXISTING[h].lower() in canonical_lower:
        return "alias_existing"
    r = norm(raw)
    for cn in canonical_lower:
        if len(cn) >= 5 and (cn == h or cn == r or (cn in r and len(cn) / max(len(r), 1) > 0.6)):
            if h != cn:
                return "alias_existing"
    if any(x in h for x in (
        "baking powder", "baking soda", "yeast", "cream of tartar", "xanthan gum",
        "cooking spray", "food coloring", "corn starch", "cornstarch", "baking mix",
        "leavening", "gelatin", "pectin", "arrowroot", "tapioca starch",
    )):
        return "leavening_additive"
    if any(x in h for x in (
        "masala", "curry powder", "italian seasoning", "cajun seasoning", "five spice",
        "pumpkin pie spice", "taco seasoning", "ranch seasoning", "seasoning mix",
        "biryani masala", "chaat masala", "garam masala", "old bay",
    )) or h.endswith(" seasoning"):
        return "spice_blend"
    if any(x in h for x in (
        "soy sauce", "fish sauce", "oyster sauce", "hoisin", "worcestershire", "mirin",
        "ketchup", "mayonnaise", "salsa", "hot sauce", "barbecue sauce", "bbq sauce",
        "teriyaki", "tahini", "miso paste", "gochujang", "gochugaru", "sriracha",
        "fish stock", "chicken broth", "vegetable broth", "beef broth", "stock",
        "cream cheese", "sour cream", "whipped cream", "condensed milk", "evaporated milk",
        "coconut milk", "almond milk", "soy milk", "paneer", "ghee", "jaggery",
        "peanut butter", "vinegar", "wine", "beer", "rum", "vodka", "whiskey", "bourbon", "sake",
        "tomato paste", "tomato sauce", "passata", "bouillon", "broth", "gravy", "marinade",
        "dressing", "ranch", "prosciutto",
    )):
        return "composite_processed"
    return "whole_species"


def node_type_from_class(ingredient_class: str) -> str:
    if ingredient_class == "spice_blend":
        return "blend"
    if ingredient_class == "leavening_additive":
        return "additive"
    if ingredient_class == "composite_processed":
        return "composite"
    return "species"


def data_status(has_foodb: bool, has_fdc: bool) -> str:
    if has_foodb and has_fdc:
        return "full"
    if has_fdc:
        return "nutrition_only"
    if has_foodb:
        return "mechanism_only"
    return "name_only"


# Globals populated in main()
canonical_lower: dict[str, str] = {}
canonical_by_id: dict[str, str] = {}
aliases: dict[str, str] = {}
string_map: dict[str, str] = {}
foodb_df: pd.DataFrame
fdc_pool: pd.DataFrame
foodb_by_name: dict[str, tuple[int, str, str | None]]
fdc_by_norm: dict[str, pd.Series]


def engine_resolve(raw: str) -> str | None:
    key = norm(raw)
    sid = aliases.get(key) or string_map.get(key)
    if sid:
        return sid
    for alias, cand in aliases.items():
        if key in alias or alias in key:
            return cand
    return None


def find_foodb(head: str, extra_terms: list[str] | None = None) -> tuple[int | None, str | None, str | None]:
    terms = [head.lower()] + [t.lower() for t in (extra_terms or [])]
    best: tuple[int, str, str | None, int] | None = None
    for t in terms:
        if not t:
            continue
        for name, (fid, cname, latin) in foodb_by_name.items():
            if t == name:
                return fid, cname, "exact"
        for name, (fid, cname, latin) in foodb_by_name.items():
            if t in name or name in t:
                score = abs(len(name) - len(t))
                if best is None or score < best[3]:
                    best = (fid, cname, f"substring:{name}", score)
        for name, (fid, cname, latin) in foodb_by_name.items():
            if latin and t in str(latin).lower():
                return fid, cname, f"latin:{latin}"
    if best:
        return best[0], best[1], best[2]
    return None, None, None


def find_fdc(head: str, pattern_override: str | None = None) -> tuple[int | None, str | None, str | None]:
    def pick_best(hits: pd.DataFrame) -> pd.Series:
        if hits.empty:
            raise ValueError("empty")
        # prefer foundation_food, then shorter description (more specific)
        scored = hits.copy()
        scored["_score"] = scored.apply(
            lambda r: (10 if r.data_type == "foundation_food" else 0) - len(str(r.description)) * 0.01,
            axis=1,
        )
        return scored.sort_values("_score", ascending=False).iloc[0]

    if pattern_override:
        pat = pattern_override.lower()
        hits = fdc_pool[fdc_pool["norm_desc"].str.contains(re.escape(pat), regex=True, na=False)]
        if len(hits):
            row = pick_best(hits)
            return int(row.fdc_id), str(row.description), "pattern_override"
    if head in COMPOSITE_FDC_PATTERNS:
        pat = COMPOSITE_FDC_PATTERNS[head].lower()
        hits = fdc_pool[fdc_pool["norm_desc"].str.contains(re.escape(pat), regex=True, na=False)]
        if len(hits):
            row = pick_best(hits)
            return int(row.fdc_id), str(row.description), "curated_pattern"
    h = head.lower()
    hits = fdc_pool[fdc_pool["norm_desc"].str.contains(re.escape(h), regex=True, na=False)]
    if len(hits):
        row = pick_best(hits)
        return int(row.fdc_id), str(row.description), "substring"
    toks = [t for t in re.split(r"[\s/,-]+", h) if len(t) >= 3 and t not in {"the", "and", "for", "with", "sauce"}]
    if not toks:
        toks = [t for t in h.split() if len(t) >= 3]
    if toks:
        mask = fdc_pool["norm_desc"].apply(
            lambda d: sum(1 for t in toks if t in d) >= max(1, len(toks) - (0 if len(toks) <= 2 else 1))
        )
        hits = fdc_pool[mask]
        if len(hits):
            row = pick_best(hits)
            return int(row.fdc_id), str(row.description), "token"
    return None, None, None


def resolve_species_id(canonical_name: str) -> str | None:
    cn = canonical_name.lower().strip()
    if cn in canonical_lower:
        return canonical_lower[cn]
    # canonical name aliases for blend constituents -> existing 463 names
    constituent_aliases = {
        "pepper": "pepper",
        "black pepper": "pepper",
        "clove": "cloves",
        "onion": "onion",
        "mustard": "mustard",
        "black mustard": "mustard",
        "bay": "bay",
        "salt": "salt",
        "star anise": "star anise",
        "allspice": "allspice",
    }
    if cn in constituent_aliases:
        target = constituent_aliases[cn].lower()
        if target in canonical_lower:
            return canonical_lower[target]
    for name, sid in canonical_lower.items():
        if cn == name:
            return sid
        if len(cn) >= 4 and (cn in name or name in cn):
            return sid
    return None


def extract_unmapped() -> tuple[dict[str, int], dict[str, str], dict[str, list], Counter[str]]:
    all_raw: list[str] = []
    for fn, col, split in [("Food_Recipe.csv", "ingredients_name", ","), ("recipes5.csv", "ingredients", ",")]:
        df = pd.read_csv(RAW / fn)
        for v in df[col].dropna():
            for p in str(v).split(split):
                all_raw.append(p.strip())
    for fn in ["recipes2.json", "recipes3.json"]:
        for rec in json.loads((RAW / fn).read_text(encoding="utf-8")):
            all_raw.extend(rec["ingredients"])
    for v in pd.read_csv(RAW / "recipes4.csv", usecols=["ingredients_canonical"])["ingredients_canonical"].dropna():
        try:
            arr = ast.literal_eval(v)
        except Exception:
            arr = [v]
        all_raw.extend(str(x) for x in arr)

    occ: Counter[str] = Counter()
    raw_ex: dict[str, str] = {}
    for x in all_raw:
        if not x:
            continue
        n = norm(x)
        occ[n] += 1
        raw_ex.setdefault(n, x)

    unmapped = {n: c for n, c in occ.items() if not engine_resolve(raw_ex[n])}
    variant_groups: dict[str, list] = defaultdict(list)
    real_heads: Counter[str] = Counter()
    for n, c in unmapped.items():
        raw = raw_ex[n]
        if is_junk(raw):
            continue
        head = cluster_key(cluster_head(raw))
        variant_groups[head].append((n, c, raw))
        real_heads[head] += c
    return unmapped, raw_ex, variant_groups, real_heads


def build_existing_nodes(species_df: pd.DataFrame) -> list[dict[str, Any]]:
    nodes = []
    for _, r in species_df.iterrows():
        nodes.append({
            "ingredient_id": str(r.species_node_id),
            "canonical_name": str(r.canonical_name),
            "latin_name": r.latin_name if pd.notna(r.latin_name) else None,
            "node_type": "species",
            "data_status": "full",  # existing species assumed full in production layer
            "foodb_id": int(r.foodb_id),
            "fdc_id": None,
            "fdc_description": None,
            "recipe_occurrences_new_datasets": 0,
            "expansion_tier": "existing",
            "constituents": None,
            "alias_strings": [],
            "notes": "Existing 463-species node; unchanged",
            "review_status": "locked",
        })
    return nodes


def main() -> None:
    global canonical_lower, canonical_by_id, aliases, string_map, foodb_df, fdc_pool
    global foodb_by_name, fdc_by_norm

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    species_df = pd.read_parquet(CANON / "species_nodes_v2.parquet")
    sm = pd.read_parquet(CANON / "ingredient_string_species_v2.parquet")
    lookup = json.loads((PROD / "indexes/ingredient_lookup.json").read_text(encoding="utf-8"))
    fdc_map = pd.read_parquet(PROD / "nutrients/species_fdc_map_v2.parquet")

    aliases = {k.strip().lower(): v for k, v in lookup["aliases"].items()}
    string_map = {}
    for _, r in sm.iterrows():
        string_map[str(r.ingredient_string).strip().lower()] = str(r.species_node)
        if pd.notna(r.canonical_name):
            string_map[str(r.canonical_name).strip().lower()] = str(r.species_node)
    canonical_by_id = {str(r.species_node_id): str(r.canonical_name) for _, r in species_df.iterrows()}
    canonical_lower = {v.lower(): k for k, v in canonical_by_id.items()}

    foodb_df = pd.read_csv(FOODB_PATH)
    foodb_by_name = {}
    for _, row in foodb_df.iterrows():
        cname = str(row["name"]).strip()
        fn = cname.lower()
        latin = row["name_scientific"] if pd.notna(row["name_scientific"]) else None
        foodb_by_name[fn] = (int(row["id"]), cname, latin)

    fdc_pool = pd.read_parquet(FDC_POOL_PATH)
    fdc_pool["norm_desc"] = fdc_pool["description"].apply(ascii_norm)

    # attach FDC to existing nodes from production map
    fdc_by_species = {str(r.species_id): r for _, r in fdc_map.iterrows() if pd.notna(r.fdc_id)}

    unmapped, raw_ex, variant_groups, real_heads = extract_unmapped()

    alias_heads: set[str] = set()
    for head in variant_groups:
        if head in TIER_D_ALIASES:
            alias_heads.add(head)
            continue
        target = ALIAS_OF_EXISTING.get(head)
        if target and target.lower() in canonical_lower:
            alias_heads.add(head)

    new_heads = {h for h in variant_groups if h not in alias_heads}
    new_head_occ = {h: real_heads[h] for h in new_heads}

    existing_nodes = build_existing_nodes(species_df)
    for node in existing_nodes:
        sid = node["ingredient_id"]
        if sid in fdc_by_species:
            fr = fdc_by_species[sid]
            node["fdc_id"] = int(fr.fdc_id)
            node["fdc_description"] = str(fr.fdc_description)

    new_nodes: list[dict[str, Any]] = []
    next_ing_id = 1
    latin_recovery_results: list[dict[str, Any]] = []
    blend_records: list[dict[str, Any]] = []
    tier_ab_report: list[dict[str, Any]] = []
    name_only_report: list[dict[str, Any]] = []
    processed_heads: set[str] = set()

    def alloc_id() -> str:
        nonlocal next_ing_id
        iid = f"ING_{next_ing_id:06d}"
        next_ing_id += 1
        return iid

    def add_node(
        head: str,
        tier: str,
        ingredient_class: str,
        ntype: str | None = None,
        extra_foodb_terms: list[str] | None = None,
        force_name_only: bool = False,
        constituents: list[dict] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        occ = new_head_occ.get(head, 0)
        ntype = ntype or node_type_from_class(ingredient_class)
        alias_strings = [raw for _, _, raw in variant_groups.get(head, [])]

        if force_name_only:
            fb_id = fb_name = fb_method = None
            fdc_id = fdc_desc = fdc_method = None
            status = "name_only"
        elif constituents:
            fb_id = fb_name = fb_method = None
            fdc_id = fdc_desc = fdc_method = None
            status = "full"  # inherited via constituents
            notes = (notes + " Mechanism/nutrition inherited from constituent species.").strip()
        else:
            fb_pat = COMPOSITE_FOODB_PATTERNS.get(head)
            if fb_pat:
                fb_id, fb_name, fb_method = find_foodb(fb_pat)
            else:
                fb_id, fb_name, fb_method = find_foodb(head, extra_foodb_terms)
            # reject dishonest cross-ingredient proxies
            if head == "fish sauce" and fb_name and "soy" in fb_name.lower():
                fb_id = fb_name = fb_method = None
            if fb_name and fb_name.lower() in {"sauce"} and head != "sauce":
                fb_id = fb_name = fb_method = None
            fdc_id, fdc_desc, fdc_method = find_fdc(head)
            status = data_status(fb_id is not None, fdc_id is not None)

        node = {
            "ingredient_id": alloc_id(),
            "canonical_name": head,
            "latin_name": (LATIN_RECOVERY.get(head) or {}).get("latin"),
            "node_type": ntype,
            "data_status": status,
            "foodb_id": fb_id,
            "foodb_name": fb_name,
            "foodb_match_method": fb_method,
            "fdc_id": fdc_id,
            "fdc_description": fdc_desc,
            "fdc_match_method": fdc_method,
            "recipe_occurrences_new_datasets": occ,
            "expansion_tier": tier,
            "ingredient_class": ingredient_class,
            "constituents": constituents,
            "alias_strings": alias_strings[:20],
            "n_alias_strings": len(alias_strings),
            "notes": notes,
            "review_status": "draft",
        }
        new_nodes.append(node)
        processed_heads.add(head)
        return node

    # --- Tier A + B: data-backed at >=50 occ ---
    for head in sorted(new_heads, key=lambda h: -new_head_occ[h]):
        occ = new_head_occ[head]
        if occ < 50:
            continue
        exemplar = variant_groups[head][0][2]
        iclass = classify_type(head, exemplar)
        if iclass == "alias_existing":
            continue
        if iclass == "spice_blend":
            continue  # handled in Tier C blends
        if head in LATIN_RECOVERY and iclass == "whole_species":
            continue  # handled in Latin recovery pass

        fb_pat = COMPOSITE_FOODB_PATTERNS.get(head)
        fb_id, _, _ = find_foodb(fb_pat or head)
        fdc_id, _, _ = find_fdc(head)
        has_data = fb_id is not None or fdc_id is not None
        if not has_data:
            continue

        tier = "A" if occ >= 500 else "B"
        node = add_node(head, tier, iclass)
        tier_ab_report.append({
            "head": head,
            "tier": tier,
            "occurrences": occ,
            "ingredient_class": iclass,
            "node_type": node["node_type"],
            "data_status": node["data_status"],
            "foodb_id": node["foodb_id"],
            "foodb_name": node["foodb_name"],
            "fdc_id": node["fdc_id"],
            "fdc_description": node["fdc_description"],
            "ingredient_id": node["ingredient_id"],
        })

    # --- Tier C policy b: Latin recovery for whole species ---
    for head, meta in LATIN_RECOVERY.items():
        if head not in new_heads or new_head_occ.get(head, 0) < 50:
            continue
        if head in processed_heads:
            continue
        exemplar = variant_groups[head][0][2]
        iclass = classify_type(head, exemplar)
        search_terms = [meta["latin"]] if meta.get("latin") else []
        search_terms.extend(meta.get("synonyms", []))
        fb_id, fb_name, fb_method = find_foodb(head, search_terms)
        fdc_id, fdc_desc, fdc_method = find_fdc(head)
        status = data_status(fb_id is not None, fdc_id is not None)

        rec = {
            "head": head,
            "latin_binomial": meta.get("latin"),
            "synonyms_searched": search_terms,
            "foodb_id": fb_id,
            "foodb_name": fb_name,
            "foodb_match_method": fb_method,
            "fdc_id": fdc_id,
            "fdc_description": fdc_desc,
            "fdc_match_method": fdc_method,
            "data_status": status,
            "occurrences": new_head_occ[head],
            "recovered": status in ("full", "mechanism_only", "nutrition_only"),
        }
        latin_recovery_results.append(rec)

        notes = f"Latin recovery: {meta.get('latin') or 'n/a'}"
        if fb_id and head == "fish sauce" and fb_name == "Soy sauce":
            notes += " WARNING: FooDB has no dedicated fish sauce; proxy not used."
        node = add_node(
            head,
            "C_latin",
            iclass,
            extra_foodb_terms=search_terms,
            notes=notes,
        )
        # overwrite with Latin search results
        node["foodb_id"] = fb_id
        node["foodb_name"] = fb_name
        node["foodb_match_method"] = fb_method
        node["fdc_id"] = fdc_id
        node["fdc_description"] = fdc_desc
        node["fdc_match_method"] = fdc_method
        node["data_status"] = status
        node["latin_name"] = meta.get("latin")

    # --- Tier C policy b: blend decomposition ---
    for head in sorted(new_heads, key=lambda h: -new_head_occ.get(h, 0)):
        occ = new_head_occ.get(head, 0)
        if occ < 50:
            continue
        exemplar = variant_groups[head][0][2]
        if classify_type(head, exemplar) != "spice_blend":
            continue
        if head in processed_heads:
            continue

        # find or infer blend composition
        blend_key = head
        for bk in BLEND_COMPOSITIONS:
            if bk in head or head in bk:
                blend_key = bk
                break

        if blend_key not in BLEND_COMPOSITIONS:
            # generic seasoning — skip decomposition, name-only
            node = add_node(head, "C_blend_unresolved", "spice_blend", force_name_only=True,
                            notes="No curated blend composition; name-only pending human curation.")
            name_only_report.append({"head": head, "occurrences": occ, "reason": "blend_unresolved"})
            continue

        comp_def = BLEND_COMPOSITIONS[blend_key]
        resolved_constituents = []
        unresolved = []
        for c in comp_def["constituents"]:
            sid = resolve_species_id(c["name"])
            resolved_constituents.append({
                "constituent_name": c["name"],
                "weight": c["weight"],
                "species_id": sid,
                "resolved": sid is not None,
            })
            if sid is None:
                unresolved.append(c["name"])

        blend_rec = {
            "blend_head": head,
            "blend_key": blend_key,
            "occurrences": occ,
            "source": comp_def["source"],
            "constituents": resolved_constituents,
            "n_constituents": len(resolved_constituents),
            "n_resolved": sum(1 for x in resolved_constituents if x["resolved"]),
            "unresolved_constituents": unresolved,
            "inheritance": "mechanism_and_nutrition_from_constituents",
        }
        blend_records.append(blend_rec)

        notes = f"Blend decomposed per: {comp_def['source']}"
        if unresolved:
            notes += f" Unresolved constituents: {', '.join(unresolved)}"
        node = add_node(
            head,
            "C_blend",
            "spice_blend",
            ntype="blend",
            constituents=resolved_constituents,
            notes=notes,
        )
        node["data_status"] = "full" if not unresolved else "mechanism_only"

    # --- Tier C policy a: name-only composites/additives at >=50 ---
    for head in sorted(new_heads, key=lambda h: -new_head_occ.get(h, 0)):
        occ = new_head_occ.get(head, 0)
        if occ < 50 or head in processed_heads:
            continue
        exemplar = variant_groups[head][0][2]
        iclass = classify_type(head, exemplar)
        if iclass not in ("composite_processed", "leavening_additive", "whole_species"):
            continue
        fb_id, _, _ = find_foodb(head)
        fdc_id, _, _ = find_fdc(head)
        if fb_id or fdc_id:
            continue  # should have been Tier A/B

        node = add_node(
            head,
            "C_nameonly",
            iclass,
            force_name_only=True,
            notes="Recognized ingredient; no FooDB/USDA match found. Backfill candidate — no proxy forced.",
        )
        name_only_report.append({
            "head": head,
            "occurrences": occ,
            "ingredient_class": iclass,
            "ingredient_id": node["ingredient_id"],
            "reason": "no_foodb_no_fdc",
        })

    # --- Tier D: alias additions ---
    alias_additions: list[dict[str, Any]] = []
    for head in sorted(alias_heads):
        target_canonical = TIER_D_ALIASES.get(head)
        if not target_canonical:
            # infer from ALIAS_OF_EXISTING
            target_canonical = ALIAS_OF_EXISTING.get(head)
        if not target_canonical:
            continue
        sid = resolve_species_id(target_canonical)
        if not sid:
            continue
        strings = [raw for _, _, raw in variant_groups[head]]
        alias_additions.append({
            "alias_head": head,
            "target_canonical_name": target_canonical,
            "target_species_id": sid,
            "recipe_occurrences": real_heads[head],
            "example_strings": strings[:10],
            "n_strings": len(variant_groups[head]),
        })

    # Regional parenthetical aliases from Food_Recipe.csv
    food_recipe = pd.read_csv(RAW / "Food_Recipe.csv")
    paren_aliases: list[dict[str, Any]] = []
    paren_re = re.compile(r"\(([^)]+)\)")
    for v in food_recipe["ingredients_name"].dropna():
        for part in str(v).split(","):
            m = paren_re.search(part)
            if not m:
                continue
            regional = norm(m.group(1))
            base = norm(re.sub(r"\([^)]*\)", "", part))
            sid = engine_resolve(part) or resolve_species_id(base)
            if regional and sid and regional not in aliases:
                paren_aliases.append({
                    "alias": regional,
                    "source_string": part.strip(),
                    "target_species_id": sid,
                    "target_canonical": canonical_by_id.get(sid),
                })

    # dedupe paren aliases
    seen_paren: set[str] = set()
    unique_paren = []
    for pa in paren_aliases:
        if pa["alias"] in seen_paren:
            continue
        seen_paren.add(pa["alias"])
        unique_paren.append(pa)

    # --- Coverage projection ---
    total_new_occ = sum(new_head_occ.values())
    mapped_occ = 0
    for head in new_heads:
        if head in processed_heads:
            mapped_occ += new_head_occ[head]
        elif head in alias_heads:
            mapped_occ += real_heads[head]

    all_nodes = existing_nodes + new_nodes
    nodes_df = pd.DataFrame(all_nodes)

    # Summary stats
    new_by_type = Counter(n["node_type"] for n in new_nodes)
    new_by_status = Counter(n["data_status"] for n in new_nodes)
    new_by_tier = Counter(n["expansion_tier"] for n in new_nodes)

    report = {
        "version": "universe_v2_draft",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_gate": "STOP — human review required before merge into mechanism graph or profiles",
        "existing_species_count": len(existing_nodes),
        "new_nodes_count": len(new_nodes),
        "total_universe_size": len(all_nodes),
        "alias_additions_count": len(alias_additions),
        "paren_alias_candidates_count": len(unique_paren),
        "new_nodes_by_type": dict(new_by_type),
        "new_nodes_by_data_status": dict(new_by_status),
        "new_nodes_by_tier": dict(new_by_tier),
        "tier_a_count": sum(1 for n in new_nodes if n["expansion_tier"] == "A"),
        "tier_b_count": sum(1 for n in new_nodes if n["expansion_tier"] == "B"),
        "tier_c_latin_count": sum(1 for n in new_nodes if n["expansion_tier"] == "C_latin"),
        "tier_c_blend_count": sum(1 for n in new_nodes if n["expansion_tier"] in ("C_blend", "C_blend_unresolved")),
        "tier_c_nameonly_count": sum(1 for n in new_nodes if n["expansion_tier"] == "C_nameonly"),
        "latin_recovery": {
            "attempted": len(latin_recovery_results),
            "recovered_with_data": sum(1 for r in latin_recovery_results if r["recovered"]),
            "still_name_only": sum(1 for r in latin_recovery_results if not r["recovered"]),
        },
        "coverage_projection": {
            "unmapped_occurrences_total": total_new_occ,
            "projected_mapped_occurrences": mapped_occ,
            "projected_coverage_pct": round(100 * mapped_occ / max(total_new_occ, 1), 2),
            "alias_heads_occurrences": sum(real_heads[h] for h in alias_heads),
        },
        "data_backing_summary_new_nodes": {
            "full": new_by_status.get("full", 0),
            "nutrition_only": new_by_status.get("nutrition_only", 0),
            "mechanism_only": new_by_status.get("mechanism_only", 0),
            "name_only": new_by_status.get("name_only", 0),
        },
    }

    # Write outputs
    nodes_df.to_parquet(OUT_DIR / "ingredient_nodes_v2_draft.parquet", index=False)
    with open(OUT_DIR / "ingredient_nodes_v2_draft.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": all_nodes}, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "blend_constituents_v2_draft.json", "w", encoding="utf-8") as f:
        json.dump({"blends": blend_records, "composition_library": BLEND_COMPOSITIONS}, f, indent=2)

    with open(OUT_DIR / "latin_recovery_v2_draft.json", "w", encoding="utf-8") as f:
        json.dump({"recoveries": latin_recovery_results}, f, indent=2)

    alias_out = {
        "tier_d_alias_heads": alias_additions,
        "regional_parenthetical_candidates": unique_paren[:100],
        "note": "Draft alias additions — NOT merged into ingredient_lookup.json",
    }
    with open(OUT_DIR / "alias_additions_v2_draft.json", "w", encoding="utf-8") as f:
        json.dump(alias_out, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "tier_ab_nodes_v2_draft.json", "w", encoding="utf-8") as f:
        json.dump({"tier_a_b_nodes": tier_ab_report}, f, indent=2)

    with open(OUT_DIR / "name_only_nodes_v2_draft.json", "w", encoding="utf-8") as f:
        json.dump({"name_only": name_only_report}, f, indent=2)

    with open(OUT_DIR / "universe_v2_expansion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown review doc
    md_lines = [
        "# Universe v2 Expansion — REVIEW GATE (DRAFT)",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "**STOP — do not merge into mechanism graph or regenerate profiles until Latin recoveries and blend decompositions are reviewed.**",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Existing species (unchanged) | {report['existing_species_count']} |",
        f"| New nodes (draft) | {report['new_nodes_count']} |",
        f"| **Total universe size** | **{report['total_universe_size']}** |",
        f"| Alias additions (Tier D) | {report['alias_additions_count']} |",
        f"| Projected new-dataset coverage | {report['coverage_projection']['projected_coverage_pct']}% |",
        "",
        "## New nodes by type",
        "",
    ]
    for k, v in sorted(new_by_type.items()):
        md_lines.append(f"- **{k}**: {v}")
    md_lines.extend(["", "## New nodes by data_status", ""])
    for k, v in sorted(new_by_status.items()):
        md_lines.append(f"- **{k}**: {v}")

    md_lines.extend(["", "## Tier A+B nodes (data-backed)", ""])
    md_lines.append("| Tier | Head | Occ | Type | Status | FooDB | FDC |")
    md_lines.append("|------|------|----:|------|--------|-------|-----|")
    for r in sorted(tier_ab_report, key=lambda x: (-x["occurrences"], x["head"])):
        fb = r["foodb_name"] or "—"
        fdc = (r["fdc_description"] or "—")[:40]
        md_lines.append(
            f"| {r['tier']} | {r['head']} | {r['occurrences']} | {r['node_type']} | {r['data_status']} | {fb} | {fdc} |"
        )

    md_lines.extend(["", "## Latin recovery (Tier C policy b — whole species)", ""])
    md_lines.append("| Head | Latin | Status | FooDB | FDC | Recovered? |")
    md_lines.append("|------|-------|--------|-------|-----|------------|")
    for r in latin_recovery_results:
        md_lines.append(
            f"| {r['head']} | {r['latin_binomial'] or '—'} | {r['data_status']} | "
            f"{r['foodb_name'] or '—'} | {(r['fdc_description'] or '—')[:35]} | {'Yes' if r['recovered'] else 'No'} |"
        )

    md_lines.extend(["", "## Blend decompositions (Tier C policy b)", ""])
    for b in blend_records:
        md_lines.append(f"### {b['blend_head']} ({b['occurrences']} occ)")
        md_lines.append(f"Source: {b['source']}")
        md_lines.append("")
        md_lines.append("| Constituent | Weight | Species ID | Resolved |")
        md_lines.append("|-------------|-------:|------------|----------|")
        for c in b["constituents"]:
            md_lines.append(
                f"| {c['constituent_name']} | {c['weight']:.0%} | {c['species_id'] or '—'} | {'Yes' if c['resolved'] else 'No'} |"
            )
        md_lines.append("")

    md_lines.extend(["", "## Name-only nodes (Tier C policy a — no proxy forced)", ""])
    for r in sorted(name_only_report, key=lambda x: -x["occurrences"])[:40]:
        md_lines.append(f"- **{r['head']}** ({r['occurrences']} occ) — {r['reason']}")

    md_lines.extend(["", "## Tier D alias additions", ""])
    for a in alias_additions:
        md_lines.append(
            f"- `{a['alias_head']}` -> {a['target_canonical_name']} ({a['target_species_id']}, {a['recipe_occurrences']} occ)"
        )

    (OUT_DIR / "UNIVERSE_V2_REVIEW.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote review artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
