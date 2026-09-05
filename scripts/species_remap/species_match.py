"""Exact FooDB species matching — no fuzzy string scoring."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# --- classification patterns (ported from recon) ---

QUANTITY_ONLY = re.compile(
    r"^[\d\s./\-+]+(cup|cups|tbsp|tsp|oz|lb|lbs|pound|teaspoon|tablespoon|ounce|"
    r"gram|grams|kg|ml|liter|litre|clove|cloves|pinch|dash|can|cans|package|"
    r"packages|slice|slices|piece|pieces|head|bunch|sprig|sprigs|stick|sticks)s?$",
    re.I,
)
JUNK_PATTERNS = [
    re.compile(r"^[\d\s./\-+]+$"),
    re.compile(r"^\W+$"),
    re.compile(r"^.{1,2}$"),
    re.compile(r"^\d+\s*$"),
]
MODIFIER_PREFIX = re.compile(
    r"^(fresh|dried|dry|frozen|canned|instant|low[- ]fat|fat[- ]free|reduced[- ]fat|"
    r"nonfat|skim|whole|organic|raw|roasted|toasted|smoked|unsalted|salted|"
    r"granulated|powdered|ground|crushed|minced|chopped|diced|sliced|shredded|"
    r"grated|peeled|boneless|skinless|lean|extra[- ]virgin|light|dark|large|small|medium|"
    r"semi[- ]skimmed|low[- ]sodium|reduced[- ]sodium|sugar[- ]free|no[- ]salt|"
    r"prepared|condensed|evaporated|unsweetened|sweetened|vanilla|plain|"
    r"unsalted|salted|ripe|mature|young|baby|new|old|hot|cold|warm|"
    r"soft|hard|thick|thin|finely|coarsely|roughly|thinly|freshly)\s+",
    re.I,
)
COMPOSITE_CONNECTORS = re.compile(r"\b(and|with|or|plus|&|\+)\b|,\s*\w", re.I)
DISH_SUFFIX = re.compile(
    r"\b(curry|soup|stew|salad|sandwich|burger|pizza|pasta|sauce|gravy|"
    r"casserole|pie|cake|bread|broth|stock|mix|blend|seasoning|marinade|"
    r"dressing|dip|spread|smoothie|cocktail|latte|muffin|cookie|brownie|"
    r"omelet|omelette|frittata|stir[- ]fry|stir fry|tacos|burrito|wrap|"
    r"lasagna|risotto|paella|chowder|gumbo|jambalaya|salsa|pesto|"
    r"stuffing|filling|topping|glaze|frosting|batter|dough|crust|"
    r"substitute|extract|essence|concentrate|bouillon|gravy|"
    r"mayonnaise|mayo|ketchup|relish|chutney|aioli|vinaigrette|"
    r"barbecue|bbq|teriyaki|marinara|alfredo|hollandaise|"
    r"cheese sauce|cream sauce|tomato sauce|soy sauce|fish sauce|"
    r"hot sauce|worcestershire|tabasco|sriracha)\b",
    re.I,
)
COMPOSITE_EXACT = frozenset(
    {
        "soy sauce",
        "fish sauce",
        "oyster sauce",
        "hoisin sauce",
        "worcestershire sauce",
        "hot sauce",
        "barbecue sauce",
        "bbq sauce",
        "tomato sauce",
        "tomato paste",
        "tomato puree",
        "chicken broth",
        "chicken stock",
        "beef broth",
        "beef stock",
        "vegetable broth",
        "vegetable stock",
        "bone broth",
        "cream of mushroom soup",
        "cream of chicken soup",
        "cream of celery soup",
        "italian dressing",
        "ranch dressing",
        "french dressing",
        "salad dressing",
        "pumpkin pie spice",
        "apple pie spice",
        "poultry seasoning",
        "italian seasoning",
        "herbs de provence",
        "herbes de provence",
        "lemon juice",
        "lime juice",
        "orange juice",
        "apple juice",
        "coconut milk",
        "almond milk",
        "soy milk",
        "evaporated milk",
        "condensed milk",
        "sweetened condensed milk",
        "half and half",
        "half-and-half",
        "sour cream",
        "cream cheese",
        "peanut butter",
        "almond butter",
        "baking mix",
        "pancake mix",
        "cake mix",
        "brownie mix",
        "cookie mix",
        "stuffing mix",
        "onion soup mix",
        "taco seasoning",
        "chili powder",
        "curry powder",
        "garlic powder",
        "onion powder",
        "celery salt",
        "garlic salt",
        "seasoned salt",
        "lemon pepper",
        "apple cider vinegar",
        "balsamic vinegar",
        "red wine vinegar",
        "white wine vinegar",
        "rice vinegar",
    }
)
BRANDISH = re.compile(
    r"\b(kraft|heinz|campbell|knorr|maggi|nestle|philadelphia|cool whip|"
    r"velveeta|oreo|fritos|doritos|hidden valley|old bay|tabasco|sriracha|"
    r"betty crocker|duncan hines|pillsbury|jell[- ]o|jello)\b",
    re.I,
)

# Curated alias -> FooDB food_id (explicit trap disambiguation; no fuzzy)
CURATED_ALIASES: dict[str, int] = {
    # staples — correct fuzzy-map failures
    "salt": 666,
    "table salt": 666,
    "kosher salt": 666,
    "sea salt": 666,
    "rock salt": 666,
    "iodized salt": 666,
    "coarse salt": 666,
    "fine salt": 666,
    "pickling salt": 666,
    "sugar": 670,
    "white sugar": 670,
    "granulated sugar": 670,
    "caster sugar": 670,
    "castor sugar": 670,
    "superfine sugar": 670,
    "brown sugar": 670,
    "light brown sugar": 670,
    "dark brown sugar": 670,
    "raw sugar": 670,
    "turbinado sugar": 670,
    "demerara sugar": 670,
    "flour": 825,
    "plain flour": 825,
    "white flour": 825,
    "wheat flour": 825,
    "bread flour": 825,
    "cake flour": 825,
    "pastry flour": 825,
    "water": 685,
    "rice": 125,
    "white rice": 125,
    "brown rice": 125,
    "long grain rice": 125,
    "short grain rice": 125,
    "basmati rice": 125,
    "jasmine rice": 125,
    "chicken": 334,
    "chicken breast": 334,
    "chicken breasts": 334,
    "chicken thigh": 334,
    "chicken thighs": 334,
    "chicken leg": 334,
    "chicken legs": 334,
    "chicken wing": 334,
    "chicken wings": 334,
    "chicken drumstick": 334,
    "chicken drumsticks": 334,
    "whole chicken": 334,
    "rotisserie chicken": 334,
    "milk": 632,
    "whole milk": 632,
    "skim milk": 632,
    "low fat milk": 632,
    "low-fat milk": 632,
    "2% milk": 632,
    "1% milk": 632,
    "semi skimmed milk": 632,
    "semi-skimmed milk": 632,
    "egg": 633,
    "eggs": 633,
    "large egg": 633,
    "large eggs": 633,
    "egg white": 633,
    "egg whites": 633,
    "egg yolk": 633,
    "egg yolks": 633,
    "butter": 667,
    "unsalted butter": 667,
    "salted butter": 667,
    "garlic": 8,
    "garlic clove": 8,
    "garlic cloves": 8,
    "onion": 6,
    "onions": 6,
    "yellow onion": 6,
    "olive oil": 941,
    "extra virgin olive oil": 941,
    "extra-virgin olive oil": 941,
    "vegetable oil": 804,
    "cooking oil": 804,
    "canola oil": 804,
    "corn oil": 804,
    "sunflower oil": 804,
    "tomato": 171,
    "tomatoes": 171,
    "ripe tomato": 171,
    "ripe tomatoes": 171,
    "cherry tomato": 172,
    "cherry tomatoes": 172,
    "parsley": 131,
    "fresh parsley": 131,
    "vanilla": 195,
    "vanilla bean": 195,
    "honey": 643,
    "cream": 669,
    "heavy cream": 669,
    "whipping cream": 669,
    "yogurt": 634,
    "plain yogurt": 634,
    "greek yogurt": 634,
    "cheese": 631,
    "cream cheese": 966,
    "sour cream": 985,
    "cheddar cheese": 967,
    "cheddar": 967,
    "parmesan cheese": 968,
    "parmesan": 968,
    "mozzarella cheese": 965,
    "mozzarella": 965,
    "potato": 175,
    "potatoes": 175,
    "carrot": 245,
    "carrots": 245,
    "celery": 215,
    "spinach": 178,
    "broccoli": 34,
    "cauliflower": 31,
    "mushroom": 561,
    "mushrooms": 561,
    "corn": 205,
    "lemon": 54,
    "lime": 53,
    "apple": 105,
    "banana": 208,
    "avocado": 130,
    "coconut": 341,
    "almond": 148,
    "walnut": 622,
    "pecan": 44,
    "peanut": 16,
    "sesame": 170,
    "tofu": 718,
    "lentil": 98,
    "lentils": 98,
    "chickpea": 47,
    "chickpeas": 47,
    "bean": 134,
    "beans": 134,
    "black bean": 134,
    "black beans": 134,
    "kidney bean": 134,
    "kidney beans": 134,
    "pinto bean": 134,
    "pinto beans": 134,
    "turmeric": 68,
    "cumin": 67,
    "coriander": 61,
    "cilantro": 61,
    "cinnamon": 586,
    "ginger": 206,
    "nutmeg": 118,
    "clove": 179,
    "cloves": 179,
    "cardamom": 74,
    "basil": 119,
    "sweet basil": 119,
    "oregano": 124,
    "thyme": 183,
    "rosemary": 159,
    "dill": 13,
    "mint": 112,
    "spearmint": 112,
    "shallot": 243,
    "shallots": 243,
    "leek": 7,
    "leeks": 7,
    "sage": 165,
    "bay leaf": 97,
    "bay leaves": 97,
    "scallion": 995,
    "scallions": 995,
    "green onion": 995,
    "green onions": 995,
    "spring onion": 995,
    "spring onions": 995,
    "apples": 105,
    "oranges": 57,
    "orange": 57,
    "boiling water": 685,
    "oil": 804,
    "clove garlic": 8,
    "cloves garlic": 8,
    "garlic clove": 8,
    "garlic cloves": 8,
    "beef": 506,
    "ground beef": 506,
    "veal": 506,
    "steak": 506,
    "shrimp": 546,
    "vinegar": 645,
    "wine": 626,
    "red wine": 626,
    "white wine": 626,
    "beer": 268,
    "soy bean": 85,
    "soybean": 85,
    "soybeans": 85,
    "wheat": 575,
    "zucchini": 907,
    "squash": 284,
    "butternut squash": 321,
    # pepper traps — spice
    "black pepper": 139,
    "white pepper": 139,
    "ground pepper": 139,
    "ground black pepper": 139,
    "cracked black pepper": 139,
    "peppercorn": 139,
    "peppercorns": 139,
    "cayenne": 731,
    "cayenne pepper": 731,
    "chili": 731,
    "chile": 731,
    "chili pepper": 731,
    "chile pepper": 731,
    "red pepper flakes": 139,
    "crushed red pepper": 139,
    "crushed red pepper flakes": 139,
}

# Strings that must never auto-map via exact FooDB name (wrong food same name)
BLOCKLIST_EXACT: frozenset[str] = frozenset(
    {
        "dead sea salt",
        "vanilla sugar",
        "lekor",
        "waterblommetjiebredie",
        "liquorice",
        "licorice",
        "instant chicken broth",
    }
)

# Pepper-vegetable color variants -> specific bell peppers when unambiguous
PEPPER_VEG_ALIASES: dict[str, int] = {
    "green bell pepper": 909,
    "green bell peppers": 909,
    "red bell pepper": 912,
    "red bell peppers": 912,
    "yellow bell pepper": 910,
    "yellow bell peppers": 910,
    "orange bell pepper": 911,
    "orange bell peppers": 911,
    "bell pepper": 40,
    "bell peppers": 40,
    "sweet pepper": 40,
    "sweet peppers": 40,
    "green pepper": 909,
    "green peppers": 909,
    "red pepper": 912,
    "red peppers": 912,
    "yellow pepper": 910,
    "yellow peppers": 910,
}


@dataclass
class FoodRecord:
    food_id: int
    name: str
    latin_name: str | None
    n_compounds: int = 0
    n_nutrients: int = 0


@dataclass
class MatchResult:
    bucket: str
    food_id: int | None = None
    match_method: str | None = None
    match_confidence: float = 0.0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    stripped: str | None = None
    pre_class: str | None = None


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    out = re.sub(r"[^\w\s\-]", " ", s.lower().strip())
    return " ".join(out.split())


def singularize(s: str) -> str:
    words = s.split()
    if not words:
        return s
    w = words[-1]
    if len(w) > 3 and w.endswith("ies"):
        words[-1] = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("oes"):
        words[-1] = w[:-2]
    elif len(w) > 3 and w.endswith("es") and not w.endswith("ses"):
        words[-1] = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        words[-1] = w[:-1]
    return " ".join(words)


def strip_modifiers(s: str) -> str:
    out = normalize_name(s)
    prev = None
    while prev != out:
        prev = out
        out = MODIFIER_PREFIX.sub("", out).strip()
    return singularize(out)


def is_junk(s: str) -> bool:
    if not s or not any(c.isalpha() for c in s):
        return True
    for p in JUNK_PATTERNS:
        if p.search(s):
            return True
    if QUANTITY_ONLY.match(s.strip()):
        return True
    return False


def is_composite(s: str) -> bool:
    n = normalize_name(s)
    if n in COMPOSITE_EXACT:
        return True
    if DISH_SUFFIX.search(n):
        return True
    if COMPOSITE_CONNECTORS.search(n) and len(n.split()) >= 3:
        return True
    # multi-word processed / blend forms (not bare modifier+single, e.g. "garlic powder" handled by strip)
    if re.search(r"\b(bouillon|broth|stock)\b", n):
        return True
    if re.search(r"\b(juice|extract|substitute|concentrate)\b", n):
        return True
    if re.search(r"\b(powder|seasoning|spice blend|spice mix)\b", n) and len(n.split()) >= 2:
        blend_markers = ("curry", "pumpkin", "apple pie", "poultry", "italian", "taco", "chili", "five spice", "mixed spice")
        if any(m in n for m in blend_markers):
            return True
    return False


def pre_classify(s: str) -> str:
    n = normalize_name(s)
    if is_junk(n):
        return "junk"
    if is_composite(n):
        return "composite"
    if MODIFIER_PREFIX.match(n) or BRANDISH.search(n):
        return "modified"
    return "clean_single"


def _candidate_dict(rec: FoodRecord) -> dict[str, Any]:
    return {
        "foodb_food_id": rec.food_id,
        "canonical_name": rec.name,
        "latin_name": rec.latin_name,
        "n_compounds": rec.n_compounds,
    }


class FooDBSpeciesIndex:
    def __init__(self, food_df: pd.DataFrame, compound_counts: dict[int, int], nutrient_counts: dict[int, int]):
        self.records: dict[int, FoodRecord] = {}
        self.exact: dict[str, list[int]] = {}
        self.normalized: dict[str, list[int]] = {}

        for _, row in food_df.iterrows():
            fid = int(row["id"])
            name = str(row["name"])
            latin = row.get("name_scientific")
            latin_s = None if pd.isna(latin) else str(latin).strip()
            rec = FoodRecord(
                food_id=fid,
                name=name,
                latin_name=latin_s,
                n_compounds=int(compound_counts.get(fid, 0)),
                n_nutrients=int(nutrient_counts.get(fid, 0)),
            )
            self.records[fid] = rec
            key = name.lower().strip()
            self.exact.setdefault(key, []).append(fid)
            nkey = normalize_name(name)
            self.normalized.setdefault(nkey, []).append(fid)

    def lookup_ids(self, key: str) -> list[int]:
        key = key.lower().strip()
        if key in self.exact:
            return list(dict.fromkeys(self.exact[key]))
        nkey = normalize_name(key)
        if nkey in self.normalized:
            return list(dict.fromkeys(self.normalized[nkey]))
        nkey_sing = singularize(nkey)
        if nkey_sing in self.normalized:
            return list(dict.fromkeys(self.normalized[nkey_sing]))
        return []

    def get(self, food_id: int) -> FoodRecord | None:
        return self.records.get(food_id)

    def resolve_unique(self, food_ids: list[int]) -> FoodRecord | None:
        uniq = list(dict.fromkeys(food_ids))
        if len(uniq) == 1:
            return self.records.get(uniq[0])
        return None

    def candidates_for(self, food_ids: list[int]) -> list[dict[str, Any]]:
        out = []
        for fid in dict.fromkeys(food_ids):
            rec = self.records.get(fid)
            if rec:
                out.append(_candidate_dict(rec))
        return out


def _pepper_ambiguous_candidates(index: FooDBSpeciesIndex) -> list[int]:
    return [139, 40, 731, 909, 912]


def match_ingredient_string(s: str, index: FooDBSpeciesIndex) -> MatchResult:
    raw = str(s).strip()
    n = normalize_name(raw)
    pre = pre_classify(raw)

    if pre == "junk":
        return MatchResult(bucket="junk", pre_class=pre, match_method="junk", match_confidence=0.0)
    if pre == "composite":
        return MatchResult(bucket="composite", pre_class=pre, match_method="composite", match_confidence=0.0)

    stripped = strip_modifiers(raw)

    # curated aliases (highest priority for trap disambiguation)
    for probe in (n, stripped, singularize(n)):
        if probe in CURATED_ALIASES:
            fid = CURATED_ALIASES[probe]
            rec = index.get(fid)
            if rec and rec.n_compounds > 0:
                return MatchResult(
                    bucket="auto_accepted",
                    food_id=fid,
                    match_method="curated_alias",
                    match_confidence=1.0,
                    stripped=stripped,
                    pre_class=pre,
                )
        if probe in PEPPER_VEG_ALIASES:
            fid = PEPPER_VEG_ALIASES[probe]
            rec = index.get(fid)
            if rec and rec.n_compounds > 0:
                return MatchResult(
                    bucket="auto_accepted",
                    food_id=fid,
                    match_method="curated_alias",
                    match_confidence=1.0,
                    stripped=stripped,
                    pre_class=pre,
                )

    # bare "pepper" -> ambiguous (spice vs vegetable)
    if n in ("pepper", "peppers") or stripped in ("pepper", "peppers"):
        cands = index.candidates_for(_pepper_ambiguous_candidates(index))
        return MatchResult(
            bucket="ambiguous",
            match_method="ambiguous",
            match_confidence=0.5,
            candidates=cands,
            stripped=stripped,
            pre_class=pre,
        )

    # exact / normalized match on full string
    for probe, method in ((n, "exact"), (singularize(n), "normalized")):
        if probe in BLOCKLIST_EXACT:
            continue
        ids = index.lookup_ids(probe)
        rec = index.resolve_unique(ids)
        if rec:
            if rec.n_compounds > 0:
                return MatchResult(
                    bucket="auto_accepted",
                    food_id=rec.food_id,
                    match_method=method,
                    match_confidence=1.0 if method == "exact" else 0.95,
                    stripped=stripped,
                    pre_class=pre,
                )
            return MatchResult(
                bucket="unmatched",
                match_method="name_only_no_compounds",
                match_confidence=0.0,
                candidates=[_candidate_dict(rec)],
                stripped=stripped,
                pre_class=pre,
            )
        if len(ids) > 1:
            return MatchResult(
                bucket="ambiguous",
                match_method="ambiguous",
                match_confidence=0.5,
                candidates=index.candidates_for(ids),
                stripped=stripped,
                pre_class=pre,
            )

    # modifier-stripped match (only if different from full string)
    if stripped and stripped not in (n, singularize(n)):
        ids = index.lookup_ids(stripped)
        if stripped in CURATED_ALIASES:
            fid = CURATED_ALIASES[stripped]
            rec = index.get(fid)
            if rec and rec.n_compounds > 0:
                return MatchResult(
                    bucket="auto_accepted",
                    food_id=fid,
                    match_method="modifier_stripped",
                    match_confidence=0.9,
                    stripped=stripped,
                    pre_class=pre,
                )
        rec = index.resolve_unique(ids)
        if rec:
            if rec.n_compounds > 0:
                return MatchResult(
                    bucket="auto_accepted",
                    food_id=rec.food_id,
                    match_method="modifier_stripped",
                    match_confidence=0.9,
                    stripped=stripped,
                    pre_class=pre,
                )
            return MatchResult(
                bucket="unmatched",
                match_method="name_only_no_compounds",
                match_confidence=0.0,
                candidates=[_candidate_dict(rec)],
                stripped=stripped,
                pre_class=pre,
            )
        if len(ids) > 1:
            return MatchResult(
                bucket="ambiguous",
                match_method="modifier_stripped_ambiguous",
                match_confidence=0.5,
                candidates=index.candidates_for(ids),
                stripped=stripped,
                pre_class=pre,
            )

    return MatchResult(
        bucket="unmatched",
        match_method="unmatched",
        match_confidence=0.0,
        stripped=stripped,
        pre_class=pre,
    )


def load_foodb_index(food_csv: str | Path, content_csv: str | Path) -> FooDBSpeciesIndex:
    food_df = pd.read_csv(food_csv, usecols=["id", "name", "name_scientific"])
    compound_counts: dict[int, int] = {}
    nutrient_counts: dict[int, int] = {}
    for chunk in pd.read_csv(content_csv, usecols=["food_id", "source_type"], chunksize=500_000):
        for fid, st in zip(chunk["food_id"].dropna(), chunk["source_type"].fillna("")):
            fid = int(fid)
            st_l = str(st).lower()
            if st_l == "compound":
                compound_counts[fid] = compound_counts.get(fid, 0) + 1
            elif st_l == "nutrient":
                nutrient_counts[fid] = nutrient_counts.get(fid, 0) + 1
    return FooDBSpeciesIndex(food_df, compound_counts, nutrient_counts)
