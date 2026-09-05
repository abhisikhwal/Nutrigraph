"""Proposed second-pass FooDB aliases for high-frequency unmatched strings.

Each entry: food_id, reasoning, judgment_call (optional), judgment_note (optional).
Authoritative parent-species / FooDB-name logic only — no fuzzy scores.
"""
from __future__ import annotations

from scripts.species_remap.species_match import normalize_name, singularize, strip_modifiers

# food_id -> (species display name, latin)
SPECIES: dict[int, tuple[str, str | None]] = {
    40: ("Pepper", "Capsicum annuum"),
    83: ("Strawberry", "Fragaria X ananassa"),
    121: ("Olive", "Olea europaea"),
    141: ("Common pea", "Pisum sativum"),
    162: ("Red raspberry", "Rubus idaeus"),
    170: ("Sesame", "Sesamum indicum"),
    179: ("Cloves", "Syzygium aromaticum"),
    182: ("Cocoa bean", "Theobroma cacao"),
    191: ("Highbush blueberry", "Vaccinium corymbosum"),
    192: ("American cranberry", "Vaccinium macrocarpon"),
    205: ("Corn", "Zea mays"),
    215: ("Celery stalks", "Apium graveolens var. dulce"),
    244: ("Rocket salad", "Eruca vesicaria"),
    274: ("Pasta", None),
    420: ("White mustard", "Sinapis alba"),
    504: ("Salmonidae (Salmon, Trout)", "Salmonidae"),
    522: ("Yellowfin tuna", "Thunnus albacares"),
    549: ("Domestic pig", "Sus scrofa domesticus"),
    562: ("Shiitake", "Lentinula edosa"),
    586: ("Cinnamon", None),
    645: ("Vinegar", None),
    671: ("Sausage", None),
    707: ("Cocoa powder", None),
    709: ("Chocolate", None),
    731: ("Chili", None),
    841: ("Corn grits", None),
    915: ("Green bean", "Phaseolus vulgaris"),
    918: ("Japanese pumpkin", "Cucurbita sp."),
    988: ("Greek feta cheese", None),
    993: ("White onion", None),
    994: ("Red onion", None),
    995: ("Green onion", None),
    1000: ("Black raisin", None),
    135: ("Date", "Phoenix dactylifera"),
    34: ("Broccoli", "Brassica oleracea var. italica"),
    122: ("Sweet marjoram", "Origanum majorana"),
    206: ("Ginger", "Zingiber officinale"),
    67: ("Cumin", "Cuminum cyminum"),
    54: ("Lemon", "Citrus limon"),
    57: ("Sweet orange", "Citrus sinensis"),
    131: ("Parsley", "Petroselinum crispum"),
    770: ("Semolina", None),
    811: ("Molasses", None),
    695: ("Syrup", None),
    277: ("Spirit", None),
    626: ("Grape wine", None),
    629: ("Apple cider", None),
    268: ("Beer", None),
    703: ("Lard", None),
    506: ("Cattle (Beef, Veal)", "Bos taurus"),
    505: ("Turkey", "Meleagris gallopavo"),
    546: ("Shrimp", "Caridea"),
    561: ("Common mushroom", None),
    883: ("Mushrooms", None),
    825: ("Flour", None),
    670: ("Sugar", None),
    632: ("Milk (Cow)", None),
    631: ("Cheese", None),
    965: ("Mozzarella cheese", None),
    966: ("Plain cream cheese", None),
    985: ("Sour cream", None),
    175: ("Potato", "Solanum tuberosum"),
    245: ("Carrot", "Daucus carota ssp. sativus"),
    178: ("Spinach", "Spinacia oleracea"),
    98: ("Lentils", "Lens culinaris"),
    47: ("Chickpea", "Cicer arietinum"),
    134: ("Common bean", "Phaseolus vulgaris"),
    88: ("Barley", "Hordeum vulgare"),
    22: ("Oat", "Avena sativa"),
    169: ("Rye", "Secale cereale"),
    575: ("Wheat", "Triticum"),
    766: ("Bulgur", None),
    765: ("Couscous", None),
    733: ("Tortilla", None),
    812: ("Cracker", None),
    684: ("Gelatin", None),
    808: ("Shortening", None),
    682: ("Leavening agent", None),
    758: ("Pectin", None),
    771: ("Tapioca pearl", None),
    291: ("Arrowroot", "Maranta arundinacea"),
    101: ("Flaxseed", "Linum usitatissimum"),
    333: ("Chia", "Salvia hispanica"),
    127: ("Poppy", "Papaver somniferum"),
    86: ("Sunflower", "Helianthus annuus"),
    63: ("Saffron", "Crocus sativus"),
    288: ("Allspice", "Pimenta dioica"),
    82: ("Fennel", "Foeniculum vulgare"),
    137: ("Anise", "Pimpinella anisum"),
    19: ("Tarragon", "Artemisia dracunculus"),
    97: ("Sweet bay", "Laurus nobilis"),
    112: ("Spearmint", "Mentha spicata"),
    113: ("Peppermint", "Mentha X piperita"),
    119: ("Sweet basil", "Ocimum basilicum"),
    643: ("Honey", None),
    667: ("Butter", None),
    669: ("Cream", None),
}

# Exact normalized string -> proposal metadata
PROPOSED_ALIASES: dict[str, dict] = {}

def _alias(key: str, food_id: int, reasoning: str, *, judgment_call: bool = False, judgment_note: str | None = None):
    PROPOSED_ALIASES[normalize_name(key)] = {
        "proposed_foodb_id": food_id,
        "reasoning": reasoning,
        "judgment_call": judgment_call,
        "judgment_note": judgment_note,
    }


# --- cuisine-critical recoveries ---
_alias("paprika", 40, "Paprika is dried ground Capsicum; FooDB has no paprika entry — parent species Pepper (40, Capsicum annuum)")
_alias("smoked paprika", 40, "Smoked paprika is processed Capsicum — map to Pepper (40)")
_alias("sweet paprika", 40, "Sweet paprika is mild ground Capsicum — map to Pepper (40)")
_alias("hungarian paprika", 40, "Regional paprika variant — still Capsicum annuum (40)")

_alias("bacon", 549, "FooDB has no bacon; Domestic pig (549, Sus scrofa) is parent species for cured pork belly", judgment_call=True, judgment_note="Bacon -> pig (549) vs lard (703) vs sausage (671)")
_alias("pancetta", 549, "Cured pork belly product — no FooDB entry; parent Domestic pig (549)", judgment_call=True, judgment_note="Pancetta -> pig (549) vs sausage (671)")
_alias("prosciutto", 549, "Dry-cured ham — no FooDB ham entry; parent Domestic pig (549)", judgment_call=True, judgment_note="Prosciutto -> pig (549) vs sausage (671)")
_alias("ham", 549, "FooDB 'Ham' false-matches Hamburger; cured ham is pork — Domestic pig (549)", judgment_call=True, judgment_note="Ham -> pig (549) vs sausage (671)")
_alias("jalapenos", 987, "Jalapeno pepper (987, Capsicum annuum 'Jalapeño')")
_alias("jalapeno", 987, "Jalapeno pepper (987)")
_alias("fresh chives", 9, "Chives (9, Allium schoenoprasum)")
_alias("chives", 9, "Chives (9)")
_alias("cashews", 11, "Cashew nut (11, Anacardium occidentale)")
_alias("cashew", 11, "Cashew nut (11)")
SPECIES[9] = ("Chives", "Allium schoenoprasum")
SPECIES[987] = ("Jalapeno pepper", "Capsicum annuum")

_alias("flour tortillas", 733, "Flour tortillas — Tortilla (733); prepared wheat product", judgment_call=True, judgment_note="Tortilla (733) vs Flour (825)")
_alias("corn tortillas", 733, "Corn tortillas — Tortilla (733)", judgment_call=True)
_alias("heavy whipping cream", 669, "Heavy whipping cream — Cream (669)")
_alias("whipped cream", 669, "Whipped cream — Cream (669)")
_alias("philadelphia cream cheese", 966, "Branded cream cheese — Plain cream cheese (966)", judgment_call=True)
_alias("cream cheese", 966, "Plain cream cheese (966)")

_alias("breadcrumbs", 269, "Breadcrumbs — Other bread (269) parent", judgment_call=True, judgment_note="Breadcrumbs -> Other bread (269) vs Cracker (812)")
_alias("italian parsley", 131, "Italian parsley — Parsley (131)")
_alias("baking potatoes", 175, "Baking potatoes — Potato (175)")
_alias("italian sausage", 671, "Italian sausage — Sausage (671)", judgment_call=True)
_alias("bourbon", 277, "Bourbon — Spirit (277); distilled spirit, thin mechanism", judgment_call=True)
_alias("grape tomatoes", 172, "Grape tomatoes — Cherry tomato (172)", judgment_call=True)
_alias("yellow cornmeal", 841, "Yellow cornmeal — Corn grits (841)", judgment_call=True)
_alias("garam masala", 700, "Garam masala — Curry powder (700) closest spice blend", judgment_call=True, judgment_note="Spice blend -> Curry powder (700) vs composite")
SPECIES[269] = ("Other bread", None)
SPECIES[700] = ("Curry powder", None)

_alias("pork", 549, "Domestic pig (549) exact species node for pork")
_alias("ground pork", 549, "Ground pork — Domestic pig (549)")
_alias("pork chops", 549, "Pork chops — Domestic pig (549)")
_alias("pork loin", 549, "Pork loin — Domestic pig (549)")
_alias("pork tenderloin", 549, "Pork tenderloin — Domestic pig (549)")
_alias("pork shoulder", 549, "Pork shoulder — Domestic pig (549)")
_alias("pork butt", 549, "Pork butt/shoulder — Domestic pig (549)")
_alias("sausage", 671, "FooDB Sausage (671) — processed meat product with compound data", judgment_call=True, judgment_note="Generic sausage vs parent pig (549)")

_alias("cornstarch", 205, "Corn starch derived from Zea mays; FooDB Corn (205) parent species", judgment_call=True, judgment_note="Cornstarch -> Corn (205) vs tapioca/arrowroot starches")
_alias("corn starch", 205, "Corn starch — parent Corn (205, Zea mays)")
_alias("cornmeal", 841, "FooDB Corn grits (841) closest to cornmeal; parent Zea mays", judgment_call=True, judgment_note="Cornmeal -> Corn grits (841) vs Corn (205)")
_alias("polenta", 841, "Polenta is cornmeal — Corn grits (841) or Corn (205)", judgment_call=True, judgment_note="Polenta -> Corn grits (841) vs Corn (205)")

_alias("cocoa", 707, "Cocoa powder (707) for ground cocoa; Theobroma cacao product")
_alias("cocoa powder", 707, "FooDB Cocoa powder (707)")
_alias("unsweetened cocoa", 707, "Unsweetened cocoa — Cocoa powder (707)")
_alias("baking cocoa", 707, "Baking cocoa — Cocoa powder (707)")

_alias("raisins", 1000, "FooDB Black raisin (1000) — no generic 'raisin' entry")
_alias("raisin", 1000, "Black raisin (1000) parent for dried grape")
_alias("golden raisins", 1000, "Golden raisins — Black raisin (1000) as closest FooDB dried grape", judgment_call=True, judgment_note="Golden vs black raisin — only Black raisin in FooDB")

_alias("cranberries", 192, "American cranberry (192, Vaccinium macrocarpon)")
_alias("cranberry", 192, "American cranberry (192)")
_alias("dried cranberries", 192, "Dried cranberry — parent Vaccinium macrocarpon (192)")

_alias("blueberries", 191, "Highbush blueberry (191, Vaccinium corymbosum)")
_alias("blueberry", 191, "Highbush blueberry (191)")

_alias("raspberries", 162, "Red raspberry (162, Rubus idaeus)")
_alias("raspberry", 162, "Red raspberry (162)")

_alias("strawberries", 83, "Strawberry (83)")
_alias("strawberry", 83, "Strawberry (83)")

_alias("salmon", 504, "Salmonidae (504) covers salmon/trout species", judgment_call=True, judgment_note="Salmon -> Salmonidae (504) vs species-specific Oncorhynchus entries")
_alias("tuna", 522, "Yellowfin tuna (522) as representative Thunnus species", judgment_call=True, judgment_note="Tuna -> Yellowfin (522) vs other Thunnus species")
_alias("shrimp", 546, "Shrimp (546, Caridea)")

_alias("peas", 141, "Common pea (141, Pisum sativum)")
_alias("pea", 141, "Common pea (141)")
_alias("frozen peas", 141, "Frozen peas — Common pea (141)")
_alias("green peas", 141, "Green peas — Common pea (141)")
_alias("split peas", 141, "Split peas — Common pea (141)")

_alias("pumpkin", 918, "Japanese pumpkin (918) — closest FooDB Cucurbita entry", judgment_call=True, judgment_note="Pumpkin -> Japanese pumpkin (918) vs no generic pumpkin")
_alias("pumpkin puree", 918, "Pumpkin puree — parent Cucurbita (918)")

_alias("sesame seeds", 170, "Sesame seed — Sesame (170, Sesamum indicum)")
_alias("sesame seed", 170, "Sesame (170)")
_alias("toasted sesame seeds", 170, "Toasted sesame — Sesame (170)")

_alias("black olives", 121, "Olive (121, Olea europaea) — color not distinguished in FooDB")
_alias("green olives", 121, "Olive (121) — green/black not split in FooDB")
_alias("olives", 121, "Olive (121, Olea europaea)")
_alias("kalamata olives", 121, "Kalamata — Olive (121) variety not in FooDB", judgment_call=True)

_alias("white onion", 993, "FooDB White onion (993) — note: verify compound coverage", judgment_call=True, judgment_note="White onion (993) has 0 FooDB compound rows in Content")
_alias("red onion", 994, "Red onion (994)")
_alias("sweet onion", 6, "Sweet onion — Garden onion (6, Allium cepa) as parent", judgment_call=True, judgment_note="Sweet onion -> Garden onion (6) vs White onion (993)")

# Fix sweet onion to use 6 - need to add 6 to SPECIES
SPECIES[6] = ("Garden onion", "Allium cepa")
PROPOSED_ALIASES[normalize_name("sweet onion")] = {
    "proposed_foodb_id": 6,
    "reasoning": "Sweet onion — Garden onion (6, Allium cepa)",
    "judgment_call": True,
    "judgment_note": "Sweet onion -> Garden onion (6) vs White onion (993)",
}

_alias("green chilies", 731, "Green chilies — Chili (731)")
_alias("green chile", 731, "Green chile — Chili (731)")
_alias("green chili", 731, "Green chili — Chili (731)")
_alias("chilies", 731, "Chilies — Chili (731)")
_alias("chilis", 731, "Chilis — Chili (731)")

_alias("ground cloves", 179, "Ground cloves — Cloves (179, Syzygium aromaticum)")
_alias("clove", 179, "Clove spice — Cloves (179); not garlic clove")

_alias("feta cheese", 988, "Greek feta cheese (988)")
_alias("feta", 988, "Greek feta cheese (988)")

_alias("salmon fillet", 504, "Salmon fillet — Salmonidae (504)")

_alias("white vinegar", 645, "Vinegar (645) — acetic acid fermentation product, FooDB entry exists")
_alias("cider vinegar", 645, "Apple cider vinegar — Vinegar (645); cider base not split", judgment_call=True, judgment_note="Cider vinegar -> Vinegar (645) vs Apple cider (629)")
_alias("apple cider vinegar", 645, "Apple cider vinegar — Vinegar (645)")
_alias("red wine vinegar", 645, "Red wine vinegar — Vinegar (645)")
_alias("rice vinegar", 645, "Rice vinegar — Vinegar (645)")

_alias("maple syrup", 695, "FooDB Syrup (695) — no maple-specific entry", judgment_call=True, judgment_note="Maple syrup -> generic Syrup (695); mechanistically thin")
_alias("corn syrup", 695, "Corn syrup — Syrup (695)", judgment_call=True, judgment_note="Processed syrup — thin mechanism")
_alias("light corn syrup", 695, "Light corn syrup — Syrup (695)")

_alias("chocolate chips", 709, "Chocolate (709) — processed chocolate product")
_alias("semi-sweet chocolate chips", 709, "Semi-sweet chocolate chips — Chocolate (709)")
_alias("bittersweet chocolate", 709, "Bittersweet chocolate — Chocolate (709)")
_alias("white chocolate", 709, "White chocolate — Chocolate (709)", judgment_call=True, judgment_note="White chocolate is cocoa-butter heavy — Chocolate (709) approximate")

_alias("noodles", 274, "Noodles — Pasta (274) parent")
_alias("egg noodles", 274, "Egg noodles — Pasta (274)")
_alias("macaroni", 274, "Macaroni — Pasta (274); FooDB has mac-and-cheese dish (784) not dry macaroni", judgment_call=True)
_alias("spaghetti", 274, "Spaghetti — Pasta (274)")

_alias("marjoram", 122, "Sweet marjoram (122, Origanum majorana)")
_alias("dried marjoram", 122, "Dried marjoram — Sweet marjoram (122)")

_alias("shiitake mushrooms", 562, "Shiitake (562, Lentinula edodes)")
_alias("shiitake mushroom", 562, "Shiitake (562)")
_alias("mushrooms", 561, "Common mushroom (561) default", judgment_call=True, judgment_note="Generic mushrooms -> Common mushroom (561) vs Mushrooms (883)")
_alias("button mushrooms", 561, "Button mushrooms — Common mushroom (561)")

_alias("broccoli florets", 34, "Broccoli florets — Broccoli (34)")

_alias("arugula", 244, "Arugula — Rocket salad (244, Eruca vesicaria)")
_alias("rocket", 244, "Rocket — Rocket salad (244)")

_alias("dates", 135, "Date (135, Phoenix dactylifera)")
_alias("date", 135, "Date (135)")

_alias("gingerroot", 206, "Ginger root — Ginger (206, Zingiber officinale)")
_alias("fresh gingerroot", 206, "Fresh ginger — Ginger (206)")

_alias("parsley flakes", 131, "Parsley flakes — Parsley (131)")
_alias("dried parsley", 131, "Dried parsley — Parsley (131)")

_alias("cumin seeds", 67, "Cumin seeds — Cumin (67)")
_alias("coriander seeds", 61, "Coriander seeds — Coriander (61)")
SPECIES[61] = ("Coriander", "Coriandrum sativum")
PROPOSED_ALIASES[normalize_name("coriander seeds")] = {
    "proposed_foodb_id": 61,
    "reasoning": "Coriander seeds — Coriander (61)",
    "judgment_call": False,
    "judgment_note": None,
}

_alias("lemon zest", 54, "Lemon zest — Lemon (54) peel; FooDB does not split zest", judgment_call=True)
_alias("orange zest", 57, "Orange zest — Sweet orange (57)")
_alias("lime zest", 53, "Lime zest — Lime (53)")
SPECIES[53] = ("Lime", "Citrus aurantiifolia")
PROPOSED_ALIASES[normalize_name("lime zest")] = {
    "proposed_foodb_id": 53,
    "reasoning": "Lime zest — Lime (53)",
    "judgment_call": True,
    "judgment_note": "Zest mapped to whole citrus fruit node",
}

_alias("stalks celery", 215, "Stalks celery — Celery stalks (215)")
_alias("celery stalks", 215, "Celery stalks (215)")
_alias("celery stalk", 215, "Celery stalk (215)")

_alias("rolled oats", 22, "Rolled oats — Oat (22, Avena sativa)")
_alias("oats", 22, "Oats — Oat (22)")
_alias("oatmeal", 22, "Oatmeal — Oat (22)")

_alias("red potatoes", 175, "Red potatoes — Potato (175)")
_alias("russet potatoes", 175, "Russet potatoes — Potato (175)")
_alias("yukon gold potatoes", 175, "Yukon gold — Potato (175)")

_alias("brandy", 277, "Brandy — Spirit (277); distilled wine", judgment_call=True, judgment_note="Spirits are thin mechanism nodes")
_alias("rum", 646, "Rum (646)")
_alias("vodka", 639, "Vodka (639)")
SPECIES[639] = ("Vodka", None)
SPECIES[646] = ("Rum", None)
PROPOSED_ALIASES[normalize_name("rum")] = {"proposed_foodb_id": 646, "reasoning": "Rum (646)", "judgment_call": True, "judgment_note": "Distilled spirit — thin mechanism"}
PROPOSED_ALIASES[normalize_name("vodka")] = {"proposed_foodb_id": 639, "reasoning": "Vodka (639)", "judgment_call": True, "judgment_note": "Distilled spirit — thin mechanism"}

_alias("all-purpose", 825, "Fragment 'all-purpose' in recipes means flour — Flour (825)", judgment_call=True, judgment_note="Truncated string all-purpose -> Flour (825)")

_alias("icing sugar", 670, "Icing/confectioners sugar — Sugar (670)")
_alias("confectioners sugar", 670, "Confectioners sugar — Sugar (670)")
_alias("powdered sugar", 670, "Powdered sugar — Sugar (670)")

_alias("goat cheese", 631, "Goat cheese — generic Cheese (631); FooDB goat is animal (541)", judgment_call=True, judgment_note="Goat cheese -> Cheese (631) vs Domestic goat (541)")
_alias("ricotta cheese", 631, "No FooDB ricotta — generic Cheese (631)", judgment_call=True, judgment_note="Ricotta -> Cheese (631); no ricotta entry")
_alias("ricotta", 631, "Ricotta — Cheese (631) placeholder", judgment_call=True)

_alias("black peppercorns", 139, "Black peppercorns — Pepper Spice (139)")
_alias("peppercorns", 139, "Peppercorns — Pepper Spice (139)")

_alias("mustard", 420, "Prepared mustard — White mustard (420) seed species", judgment_call=True, judgment_note="Mustard condiment -> White mustard (420) vs no prepared mustard entry")
_alias("dijon mustard", 420, "Dijon mustard — White mustard (420) parent", judgment_call=True)

# Processed-no-phytochem markers (exact normalized strings or patterns)
PROCESSED_NO_PHYTOCHEM: dict[str, str] = {}
for s, note in [
    ("baking powder", "Leavening agent — sodium bicarbonate + acid salts; not a species"),
    ("baking soda", "Sodium bicarbonate — no FooDB food species; pure chemical leavening"),
    ("baking pwdr", "Baking powder alias — leavening"),
    ("cream of tartar", "Potassium bitartrate — chemical leavening stabilizer, no species"),
    ("active dry yeast", "Baker's yeast — microorganism; no FooDB Saccharomyces food entry"),
    ("yeast", "Baker's yeast — no FooDB entry"),
    ("nutritional yeast", "Deactivated yeast — no FooDB entry"),
    ("vegetable shortening", "Shortening (808) — refined fat, thin mechanism"),
    ("shortening", "Shortening (808) — processed fat"),
    ("cooking spray", "Aerosol oil propellant — no FooDB entry"),
    ("unflavored gelatin", "Gelatin (684) — hydrolyzed collagen, not phytochemical"),
    ("gelatin", "Gelatin (684) — animal protein product"),
    ("pectin", "Pectin (758) — polysaccharide additive"),
    ("xanthan gum", "Hydrocolloid additive — no FooDB food entry"),
    ("guar gum", "Hydrocolloid additive — no FooDB food entry"),
    ("corn syrup solids", "Processed sugar syrup — thin"),
    ("nonstick cooking spray", "Cooking spray — no species"),
    ("food coloring", "Artificial additive — no species"),
    ("red food coloring", "Artificial dye — no species"),
]:
    PROCESSED_NO_PHYTOCHEM[normalize_name(s)] = note

# Map some processed to FooDB thin nodes for reporting (optional food_id)
PROCESSED_FOOD_ID: dict[str, int] = {
    normalize_name("baking powder"): 682,
    normalize_name("baking pwdr"): 682,
    normalize_name("vegetable shortening"): 808,
    normalize_name("shortening"): 808,
    normalize_name("unflavored gelatin"): 684,
    normalize_name("gelatin"): 684,
    normalize_name("pectin"): 758,
}


def lookup_proposed(ingredient_string: str) -> dict | None:
    """Exact then modifier-stripped lookup in proposed alias table."""
    n = normalize_name(ingredient_string)
    if n in PROPOSED_ALIASES:
        return dict(PROPOSED_ALIASES[n])
    stripped = strip_modifiers(ingredient_string)
    sn = normalize_name(stripped)
    if sn in PROPOSED_ALIASES:
        out = dict(PROPOSED_ALIASES[sn])
        out["reasoning"] = f"Modifier-stripped '{stripped}' -> " + out["reasoning"]
        return out
    sing = singularize(sn)
    if sing in PROPOSED_ALIASES:
        out = dict(PROPOSED_ALIASES[sing])
        out["reasoning"] = f"Singularized '{sing}' -> " + out["reasoning"]
        return out
    return None


def lookup_processed(ingredient_string: str) -> tuple[str, int | None] | None:
    n = normalize_name(ingredient_string)
    if n in PROCESSED_NO_PHYTOCHEM:
        return PROCESSED_NO_PHYTOCHEM[n], PROCESSED_FOOD_ID.get(n)
    stripped = normalize_name(strip_modifiers(ingredient_string))
    if stripped in PROCESSED_NO_PHYTOCHEM:
        return PROCESSED_NO_PHYTOCHEM[stripped], PROCESSED_FOOD_ID.get(stripped)
    return None


def species_info(food_id: int) -> tuple[str | None, str | None]:
    if food_id in SPECIES:
        name, latin = SPECIES[food_id]
        return name, latin
    return None, None
