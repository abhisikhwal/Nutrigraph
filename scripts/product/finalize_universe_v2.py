#!/usr/bin/env python3
"""
Finalize universe v2 with human rulings — writes LOCKED artifacts only.
Does NOT regenerate profiles, ingest recipes, or touch mechanism graph.
"""
from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/product/universe_v2"
DRAFT_JSON = OUT_DIR / "ingredient_nodes_v2_draft.json"
LOOKUP_PATH = ROOT / "data/processed/product/indexes/ingredient_lookup.json"
FOODB_PATH = ROOT / "data/raw/foodb/foodb_2020_04_07_csv/Food.csv"
FDC_POOL_PATH = ROOT / "data/processed/product/nutrients/fdc_clean_pool_v2.parquet"

# Load build module for BLEND_COMPOSITIONS, resolve_species_id setup
_BUILD_PATH = Path(__file__).parent / "build_universe_v2_expansion.py"
_spec = importlib.util.spec_from_file_location("universe_build", _BUILD_PATH)
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)  # type: ignore

BLEND_COMPOSITIONS = _build.BLEND_COMPOSITIONS
TIER_D_ALIASES = _build.TIER_D_ALIASES

# ---------------------------------------------------------------------------
# Ruling 1 — exhaustive synonym sets
# ---------------------------------------------------------------------------
EXHAUSTIVE_SYNONYM_HUNT: dict[str, dict[str, Any]] = {
    "ajwain": {
        "latin": "Trachyspermum ammi",
        "synonyms": [
            "ajwain", "carom seeds", "carom seed", "carom", "ajowan", "ajwan",
            "bishop's weed", "bishops weed", "bishop weed",
            "trachyspermum ammi", "carum copticum", "copticum", "ammii",
        ],
    },
    "galangal": {
        "latin": "Alpinia galanga",
        "synonyms": [
            "galangal", "galanga", "greater galangal", "thai ginger", "blue ginger",
            "alpinia galanga", "languas galanga", "alpinia",
        ],
    },
    "curry leaf": {
        "latin": "Murraya koenigii",
        "synonyms": [
            "curry leaf", "curry leaves", "sweet neem", "sweet neem leaf",
            "kadi patta", "kadipatta", "murraya koenigii", "bergera koenigii",
            "murraya", "bergera", "koenigii",
        ],
    },
}

# Paprika → Capsicum (Pepper) per ruling
CONSTITUENT_OVERRIDES: dict[str, str] = {
    "paprika": "Pepper",  # SP_000032 Capsicum annuum
    "mace": "Nutmeg",  # mace is aril of nutmeg — closest SP node
    "bay": "Sweet bay",
}

# Existing species links (no duplicate nodes)
EXISTING_SPECIES_LINKS: dict[str, str] = {
    "curry powder": "SP_000325",
}

# Mandatory blend instantiation (below frequency threshold)
MANDATORY_BLEND_HEADS = ["garam masala", "biryani masala"]

# Confident regional parenthetical aliases
CONFIDENT_PAREN_ALIASES: dict[str, str] = {
    "haldi": "Turmeric",
    "jeera": "Cumin",
    "jera": "Cumin",
    "jeeraga": "Cumin",
    "esteem": "Cumin",
    "dhania": "Coriander",
    "dhaania": "Coriander",
    "baingan / eggplant": "Eggplant",
    "baingan": "Eggplant",
    "green aubergine": "Eggplant",
    "methi seeds": "Fenugreek",
    "fenugreek seeds": "Fenugreek",
    "methi leaves": "Fenugreek",
    "fenugreek leaves": "Fenugreek",
    "kasuri methi": "Fenugreek",
    "dried fenugreek leaves": "Fenugreek",
    "laung": "Cloves",
    "elaichi": "Cardamom",
    "elachi": "Cardamom",
    "badi elaichi": "Cardamom",
    "dalchini": "Cinnamon",
    "dalchni": "Cinnamon",
    "tej patta": "Sweet bay",
    "pudina": "Spearmint",
    "palak": "Spinach",
    "gajjar": "Carrot",
    "gajar": "Carrot",
    "aloo": "Potato",
    "matar": "Common pea",
    "vatana": "Common pea",
    "gobi": "Cauliflower",
    "besan": "Flour",
    "maida": "Flour",
    "badam": "Almond",
    "badam powder": "Almond",
    "kopra": "Coconut",
    "moongphali": "Peanut",
    "til seeds": "Sesame",
    "gingelly": "Sesame",
    "dahi / yogurt": "Yogurt",
    "dahi/ yogurt": "Yogurt",
    "capsicum": "Green bell pepper",
    "white chickpeas": "Chickpea",
    "brown chickpeas": "Chickpea",
    "kala namak": "Salt",
    "rai/ kadugu": "Mustard",
    "rai / kadugu": "Mustard",
    "dry mango powder": "Mango",
    "anardana powder": "Mango",
    "french beans": "Green bean",
    "lady finger/okra": "Okra",
    "lady finger/ okra": "Okra",
    "kathal": "Jackfruit",
    "patta gobi/ muttaikose": "Cabbage",
    "patta gobi/ muttakose": "Cabbage",
    "semolina/ rava": "Semolina",
    "finger millet/ nagli": "Millet",
    "pearl millet": "Millet",
    "parl millet": "Millet",
    "flattened rice": "Rice",
    "flattened red rice": "Rice",
    "yellow corn meal flour": "Corn",
    "kuttu ka atta": "Wheat",
    "lobia": "Black-eyed pea",
    "large kidney beans": "Black-eyed pea",
    "ash gourd/ white pumpkin": "Wax gourd",
    "ash gourd/white pumpkin": "Wax gourd",
    "vellai poosanikai": "Wax gourd",
}

# Corrected / rejected parentheticals (explicit)
REJECTED_PAREN_ALIASES: list[dict[str, str]] = [
    {"alias": "lobia", "was_target": "Pepper (Spice)", "correct_target": "Black-eyed pea", "reason": "black-eyed pea/cowpea, not pepper"},
    {"alias": "ash gourd/ white pumpkin", "was_target": "Sugar", "correct_target": "Wax gourd (Benincasa hispida)", "reason": "winter melon, not sugar"},
    {"alias": "sambar onions", "was_target": "Pear", "correct_target": "UNCERTAIN", "reason": "pearl onion — needs review"},
    {"alias": "apple gourd", "was_target": "Apple", "correct_target": "UNCERTAIN", "reason": "tinda/apple gourd — needs review"},
    {"alias": "moringa/ murungai keerai", "was_target": "Rum", "correct_target": "UNCERTAIN", "reason": "drumstick leaves — needs review"},
    {"alias": "green aubergine", "was_target": "Gin", "correct_target": "Eggplant", "reason": "corrected to eggplant"},
    {"alias": "with egg", "was_target": "Eggs", "correct_target": "UNCERTAIN", "reason": "mayonnaise context, not egg ingredient"},
    {"alias": "salted", "was_target": "Salt", "correct_target": "UNCERTAIN", "reason": "butter (salted) context"},
    {"alias": "thin", "was_target": "Rice", "correct_target": "UNCERTAIN", "reason": "rice vermicelli context"},
    {"alias": "whole", "was_target": "Pepper (Spice)", "correct_target": "UNCERTAIN", "reason": "urad dal whole context"},
    {"alias": "split", "was_target": "Sugar", "correct_target": "UNCERTAIN", "reason": "urad dal split context"},
    {"alias": "preserved & tinned", "was_target": "Pineapple", "correct_target": "UNCERTAIN", "reason": "modifier not ingredient"},
    {"alias": "cake/ candy sprinkles", "was_target": "Sugar", "correct_target": "UNCERTAIN", "reason": "decorative sugar — needs review"},
    {"alias": "homemade cottage cheese", "was_target": "Cheese", "correct_target": "UNCERTAIN", "reason": "paneer — no node yet, do not proxy"},
]

PENDING_REVIEW_PAREN: list[str] = [
    "hing", "asafoetida", "carania", "mongphali", "mawa", "date & tamarind",
    "coriander & mint", "tomato chilli sauce", "soda water", "badam milk",
    "malabar tamarind", "suran/senai/ratalu", "parangikai/ pumpkin",
    "pargikai/ pumpkin", "sponge/silk squash", "chavli", "kothavarangai / cluster beans",
    "yard long beans/karamani/barbati", "water chestnut", "salt toor dal (salt toor dal",
    "doddapatre", "ripe",
]


def ascii_norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower().strip())


def load_foodb_index() -> tuple[pd.DataFrame, dict[str, tuple[int, str, str | None]]]:
    fb = pd.read_csv(FOODB_PATH)
    by_name: dict[str, tuple[int, str, str | None]] = {}
    for _, row in fb.iterrows():
        cname = str(row["name"]).strip()
        fn = cname.lower()
        latin = row["name_scientific"] if pd.notna(row["name_scientific"]) else None
        by_name[fn] = (int(row["id"]), cname, latin)
    return fb, by_name


def exhaustive_hunt(head: str, synonyms: list[str], foodb_df: pd.DataFrame, fdc_pool: pd.DataFrame) -> dict[str, Any]:
    """Search every synonym against FooDB name, FooDB latin, FDC description."""
    results: list[dict[str, Any]] = []
    fb_hit = fdc_hit = None
    for syn in synonyms:
        t = syn.lower().strip()
        if not t:
            continue
        entry: dict[str, Any] = {"synonym": syn, "foodb_hit": None, "fdc_hit": None}
        # FooDB name
        for _, row in foodb_df.iterrows():
            name = str(row["name"]).lower()
            latin = str(row["name_scientific"]).lower() if pd.notna(row["name_scientific"]) else ""
            if t == name or (len(t) >= 4 and (t in name or t in latin)):
                hit = {"foodb_id": int(row["id"]), "name": str(row["name"]), "latin": row["name_scientific"], "match": f"name/latin:{syn}"}
                entry["foodb_hit"] = hit
                if fb_hit is None:
                    fb_hit = hit
                break
        # FDC
        hits = fdc_pool[fdc_pool["norm_desc"].str.contains(re.escape(t), regex=True, na=False)]
        if len(hits):
            row = hits.iloc[0]
            fhit = {"fdc_id": int(row.fdc_id), "description": str(row.description), "match": f"fdc:{syn}"}
            entry["fdc_hit"] = fhit
            if fdc_hit is None:
                fdc_hit = fhit
        results.append(entry)

    has_fb = fb_hit is not None
    has_fdc = fdc_hit is not None
    if has_fb and has_fdc:
        status = "full"
    elif has_fdc:
        status = "nutrition_only"
    elif has_fb:
        status = "mechanism_only"
    else:
        status = "name_only"

    out: dict[str, Any] = {
        "head": head,
        "synonyms_tried": synonyms,
        "per_synonym_results": results,
        "foodb_id": fb_hit["foodb_id"] if fb_hit else None,
        "foodb_name": fb_hit["name"] if fb_hit else None,
        "foodb_match": fb_hit["match"] if fb_hit else None,
        "fdc_id": fdc_hit["fdc_id"] if fdc_hit else None,
        "fdc_description": fdc_hit["description"] if fdc_hit else None,
        "fdc_match": fdc_hit["match"] if fdc_hit else None,
        "data_status": status,
        "recovered": status != "name_only",
    }
    if status == "name_only":
        out["backfill_source"] = "HMDB_or_literature"
        out["backfill_note"] = "Documented plant chemistry exists; FooDB/FDC clean pool carry no entry."
    return out


def resolve_canonical_to_id(name: str, canonical_lower: dict[str, str]) -> str | None:
    cn = name.lower().strip()
    if cn in canonical_lower:
        return canonical_lower[cn]
    for c, sid in canonical_lower.items():
        if cn == c or (len(cn) >= 4 and cn in c):
            return sid
    return None


def resolve_constituent(name: str, canonical_lower: dict[str, str]) -> str | None:
    if name in CONSTITUENT_OVERRIDES:
        return resolve_canonical_to_id(CONSTITUENT_OVERRIDES[name], canonical_lower)
    return resolve_canonical_to_id(name, canonical_lower)


def build_blend_constituents(blend_key: str, canonical_lower: dict[str, str]) -> list[dict[str, Any]]:
    comp_def = BLEND_COMPOSITIONS[blend_key]
    out = []
    for c in comp_def["constituents"]:
        sid = resolve_constituent(c["name"], canonical_lower)
        out.append({
            "constituent_name": c["name"],
            "weight": c["weight"],
            "species_id": sid,
            "resolved": sid is not None,
        })
    return out


def main() -> None:
    # Bootstrap canonical maps via build module
    species_df = pd.read_parquet(ROOT / "data/processed/canonical/species_nodes_v2.parquet")
    canonical_by_id = {str(r.species_node_id): str(r.canonical_name) for _, r in species_df.iterrows()}
    canonical_lower = {v.lower(): k for k, v in canonical_by_id.items()}

    foodb_df, foodb_by_name = load_foodb_index()
    fdc_pool = pd.read_parquet(FDC_POOL_PATH)
    fdc_pool["norm_desc"] = fdc_pool["description"].apply(ascii_norm)

    draft = json.loads(DRAFT_JSON.read_text(encoding="utf-8"))
    nodes: list[dict[str, Any]] = draft["nodes"]
    existing = [n for n in nodes if n.get("expansion_tier") == "existing"]
    new_nodes = [n for n in nodes if n.get("expansion_tier") != "existing"]
    new_by_name = {n["canonical_name"]: n for n in new_nodes}

    # Get occurrence counts from build extract
    _build.canonical_lower = canonical_lower
    _build.canonical_by_id = canonical_by_id
    lookup = json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))
    _build.aliases = {k.strip().lower(): v for k, v in lookup["aliases"].items()}
    sm = pd.read_parquet(ROOT / "data/processed/canonical/ingredient_string_species_v2.parquet")
    _build.string_map = {}
    for _, r in sm.iterrows():
        _build.string_map[str(r.ingredient_string).strip().lower()] = str(r.species_node)
        if pd.notna(r.canonical_name):
            _build.string_map[str(r.canonical_name).strip().lower()] = str(r.species_node)
    _build.foodb_df = foodb_df
    _build.foodb_by_name = foodb_by_name
    _build.fdc_pool = fdc_pool

    _, _, variant_groups, real_heads = _build.extract_unmapped()
    head_occ = dict(real_heads)

    next_id = max(int(n["ingredient_id"].split("_")[1]) for n in new_nodes) + 1

    def alloc_id() -> str:
        nonlocal next_id
        iid = f"ING_{next_id:06d}"
        next_id += 1
        return iid

    # --- Ruling 1: exhaustive synonym hunt ---
    latin_locked: list[dict[str, Any]] = []
    for head, meta in EXHAUSTIVE_SYNONYM_HUNT.items():
        hunt = exhaustive_hunt(head, meta["synonyms"], foodb_df, fdc_pool)
        hunt["latin_binomial"] = meta["latin"]
        latin_locked.append(hunt)
        if head in new_by_name:
            node = new_by_name[head]
            node["latin_name"] = meta["latin"]
            node["foodb_id"] = hunt["foodb_id"]
            node["foodb_name"] = hunt["foodb_name"]
            node["foodb_match_method"] = hunt["foodb_match"]
            node["fdc_id"] = hunt["fdc_id"]
            node["fdc_description"] = hunt["fdc_description"]
            node["fdc_match_method"] = hunt["fdc_match"]
            node["data_status"] = hunt["data_status"]
            node["synonym_hunt"] = hunt["per_synonym_results"]
            if hunt["data_status"] == "name_only":
                node["backfill_source"] = hunt["backfill_source"]
            node["review_status"] = "locked"
            node["notes"] = (node.get("notes") or "") + f" Exhaustive synonym hunt ({len(meta['synonyms'])} terms): no FooDB/FDC hit." if not hunt["recovered"] else f" Recovered via {hunt['foodb_match'] or hunt['fdc_match']}."

    # bok choy, tomatillo — confirm full
    for head in ("bok choy", "tomatillo"):
        if head in new_by_name:
            new_by_name[head]["review_status"] = "locked"
            new_by_name[head]["notes"] = "Latin recovery accepted — full node."

    # prosciutto
    if "prosciutto" in new_by_name:
        p = new_by_name["prosciutto"]
        p["backfill_candidate"] = True
        p["data_status"] = "name_only"
        p["review_status"] = "locked"
        p["notes"] = "No FooDB/FDC match in clean pool. backfill_candidate=true."

    # --- Ruling 2: blends ---
    blend_locked: list[dict[str, Any]] = []

    # Fix paprika/bay/mace overrides on all blend nodes
    for head, node in new_by_name.items():
        if node.get("node_type") != "blend" or not node.get("constituents"):
            continue
        for c in node["constituents"]:
            if c["constituent_name"] in CONSTITUENT_OVERRIDES:
                sid = resolve_constituent(c["constituent_name"], canonical_lower)
                if sid:
                    c["species_id"] = sid
                    c["resolved"] = True
                    c["resolution_note"] = f"Ruling: {c['constituent_name']} → {canonical_by_id[sid]}"
        unresolved = [c["constituent_name"] for c in node["constituents"] if not c["resolved"]]
        node["data_status"] = "full" if not unresolved else "mechanism_only"
        node["review_status"] = "locked"

    # Record locked blends
    for head in ("cajun seasoning", "italian seasoning", "five spice", "chaat masala", "old bay seasoning"):
        if head in new_by_name and new_by_name[head].get("constituents"):
            node = new_by_name[head]
            blend_locked.append({
                "blend_head": head,
                "blend_key": head,
                "occurrences": node.get("recipe_occurrences_new_datasets", 0),
                "source": BLEND_COMPOSITIONS.get(head, {}).get("source", ""),
                "constituents": node["constituents"],
                "n_resolved": sum(1 for c in node["constituents"] if c["resolved"]),
                "status": "locked_paprika_fixed" if head == "cajun seasoning" else "locked_accepted",
            })

    # Unresolved blends — flag backfill
    for head in ("creole seasoning", "poultry seasoning", "greek seasoning", "italian herb seasoning"):
        if head in new_by_name:
            node = new_by_name[head]
            node["data_status"] = "name_only"
            node["backfill_candidate"] = True
            node["backfill_type"] = "blend_composition"
            node["review_status"] = "locked"
            node["notes"] = "Blend composition backfill candidate — no curated table."

    # Instantiate mandatory blends
    for head in MANDATORY_BLEND_HEADS:
        if head in new_by_name:
            continue
        if head not in BLEND_COMPOSITIONS:
            continue
        constituents = build_blend_constituents(head, canonical_lower)
        for c in constituents:
            if c["constituent_name"] in CONSTITUENT_OVERRIDES and not c["resolved"]:
                sid = resolve_constituent(c["constituent_name"], canonical_lower)
                if sid:
                    c["species_id"] = sid
                    c["resolved"] = True
        unresolved = [c["constituent_name"] for c in constituents if not c["resolved"]]
        node = {
            "ingredient_id": alloc_id(),
            "canonical_name": head,
            "latin_name": None,
            "node_type": "blend",
            "data_status": "full" if not unresolved else "mechanism_only",
            "foodb_id": None,
            "foodb_name": None,
            "foodb_match_method": None,
            "fdc_id": None,
            "fdc_description": None,
            "fdc_match_method": None,
            "recipe_occurrences_new_datasets": head_occ.get(head, 0),
            "expansion_tier": "C_blend_mandatory",
            "ingredient_class": "spice_blend",
            "constituents": constituents,
            "alias_strings": [x[2] for x in variant_groups.get(head, [])][:20],
            "n_alias_strings": len(variant_groups.get(head, [])),
            "notes": f"Mandatory instantiation (cuisine-critical). {BLEND_COMPOSITIONS[head]['source']}",
            "review_status": "locked",
        }
        new_nodes.append(node)
        new_by_name[head] = node
        blend_locked.append({
            "blend_head": head,
            "blend_key": head,
            "occurrences": head_occ.get(head, 0),
            "source": BLEND_COMPOSITIONS[head]["source"],
            "constituents": constituents,
            "n_resolved": sum(1 for c in constituents if c["resolved"]),
            "status": "locked_mandatory_instantiation",
        })

    # curry powder — link existing SP_000325, no duplicate
    curry_powder_link = {
        "head": "curry powder",
        "action": "link_existing_species",
        "species_id": "SP_000325",
        "canonical_name": "Curry powder",
        "occurrences": head_occ.get("curry powder", 0),
        "note": "Not duplicated — maps to existing SP_000325",
    }

    # --- Ruling 3: aliases ---
    merged_aliases: list[dict[str, Any]] = []
    for alias_head, target_canonical in TIER_D_ALIASES.items():
        sid = resolve_canonical_to_id(target_canonical, canonical_lower)
        if not sid:
            continue
        merged_aliases.append({
            "alias": alias_head,
            "target_species_id": sid,
            "target_canonical_name": canonical_by_id[sid],
            "source": "tier_d_safe_heads",
            "status": "merged",
        })

    merged_paren: list[dict[str, Any]] = []
    pending_paren: list[dict[str, Any]] = []
    for alias, target in CONFIDENT_PAREN_ALIASES.items():
        sid = resolve_canonical_to_id(target, canonical_lower)
        if sid:
            merged_paren.append({
                "alias": alias,
                "target_species_id": sid,
                "target_canonical_name": canonical_by_id[sid],
                "source": "regional_parenthetical_confident",
                "status": "merged",
            })
        elif target == "Wax gourd":
            # FooDB 510 exists but no SP_* — link to foodb for backfill
            merged_paren.append({
                "alias": alias,
                "target_foodb_id": 510,
                "target_canonical_name": "Wax gourd",
                "target_species_id": None,
                "source": "regional_parenthetical_confident",
                "status": "merged_foodb_only",
                "note": "Benincasa hispida — FooDB 510; species node pending",
            })

    for alias in PENDING_REVIEW_PAREN:
        pending_paren.append({"alias": alias, "status": "pending_human_review", "reason": "target uncertain"})

    # Lock all remaining new nodes
    for node in new_nodes:
        if node.get("review_status") != "locked":
            node["review_status"] = "locked"
        # proxy policy flags
        if node["canonical_name"] in ("mirin", "gochujang", "paneer"):
            node["data_status"] = "name_only"
            node["notes"] = (node.get("notes") or "") + " Confirmed: no forced proxy."

    for node in existing:
        node["review_status"] = "locked"

    all_nodes = existing + new_nodes

    # Coverage projection
    alias_heads_occ = sum(
        head_occ.get(h, 0) for h in TIER_D_ALIASES
        if resolve_canonical_to_id(TIER_D_ALIASES[h], canonical_lower)
    )
    mapped_heads = set(new_by_name.keys()) | set(TIER_D_ALIASES.keys()) | set(CONFIDENT_PAREN_ALIASES.keys())
    mapped_heads.add("curry powder")  # linked to existing
    total_occ = sum(head_occ.values())
    projected = sum(head_occ.get(h, 0) for h in mapped_heads if h in head_occ)
    projected += alias_heads_occ  # approximate

    new_only = [n for n in new_nodes]
    by_type = Counter(n["node_type"] for n in new_only)
    by_status = Counter(n["data_status"] for n in new_only)

    manifest = {
        "version": "universe_v2_locked",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED — awaiting confirmation before profile regeneration or recipe ingest",
        "existing_species_count": len(existing),
        "new_nodes_count": len(new_only),
        "total_universe_size": len(all_nodes),
        "new_nodes_by_type": dict(by_type),
        "new_nodes_by_data_status": dict(by_status),
        "latin_recovery": {
            "exhaustive_hunt": latin_locked,
            "recovered": sum(1 for r in latin_locked if r["recovered"]),
            "name_only_with_hmdb_flag": sum(1 for r in latin_locked if r.get("backfill_source")),
        },
        "blends": {
            "instantiated_locked": blend_locked,
            "curry_powder_link": curry_powder_link,
        },
        "aliases": {
            "safe_heads_merged": len(merged_aliases),
            "confident_parentheticals_merged": len(merged_paren),
            "pending_review_parentheticals": len(pending_paren),
            "rejected_corrections": REJECTED_PAREN_ALIASES,
        },
        "coverage_projection": {
            "unmapped_occurrences_total": total_occ,
            "projected_mapped_occurrences": projected,
            "projected_coverage_pct": round(100 * projected / max(total_occ, 1), 2),
        },
        "constraints_honored": [
            "No ingredient profile regeneration",
            "No new recipe ingest",
            "No mechanism graph changes",
            "ingredient_lookup.json not modified (aliases in separate locked file)",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_nodes).to_parquet(OUT_DIR / "ingredient_nodes_v2_locked.parquet", index=False)
    with open(OUT_DIR / "ingredient_nodes_v2_locked.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": all_nodes}, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "latin_recovery_v2_locked.json", "w", encoding="utf-8") as f:
        json.dump({"exhaustive_hunts": latin_locked}, f, indent=2)

    with open(OUT_DIR / "blend_constituents_v2_locked.json", "w", encoding="utf-8") as f:
        json.dump({"blends": blend_locked, "curry_powder_link": curry_powder_link}, f, indent=2)

    with open(OUT_DIR / "ingredient_aliases_v2_locked.json", "w", encoding="utf-8") as f:
        json.dump({
            "safe_alias_heads": merged_aliases,
            "confident_regional_parentheticals": merged_paren,
            "note": "Merged aliases — apply to ingredient_lookup on next index build",
        }, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "alias_parentheticals_pending_review.json", "w", encoding="utf-8") as f:
        json.dump({"pending": pending_paren, "rejected_corrections": REJECTED_PAREN_ALIASES}, f, indent=2)

    with open(OUT_DIR / "universe_v2_locked_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Markdown summary
    md = [
        "# Universe v2 — LOCKED",
        "",
        f"Generated: {manifest['generated_at']}",
        "",
        "**STOP — locked for confirmation. Profiles, ingest, mechanism graph untouched.**",
        "",
        f"- Existing species: **{manifest['existing_species_count']}**",
        f"- New nodes: **{manifest['new_nodes_count']}**",
        f"- **Total universe: {manifest['total_universe_size']}**",
        f"- Projected coverage: **{manifest['coverage_projection']['projected_coverage_pct']}%**",
        "",
        "## data_status (new nodes only)",
        "",
    ]
    for k, v in sorted(by_status.items()):
        md.append(f"- {k}: {v}")
    md.extend(["", "## Latin exhaustive hunt", ""])
    for r in latin_locked:
        md.append(f"### {r['head']}")
        md.append(f"- Status: **{r['data_status']}** | Recovered: {r['recovered']}")
        if r.get("backfill_source"):
            md.append(f"- backfill_source: `{r['backfill_source']}`")
        hits = [x for x in r["per_synonym_results"] if x["foodb_hit"] or x["fdc_hit"]]
        md.append(f"- Synonyms tried: {len(r['synonyms_tried'])} | Hits: {len(hits)}")
        md.append("")

    (OUT_DIR / "UNIVERSE_V2_LOCKED.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nLocked universe written to {OUT_DIR}")


if __name__ == "__main__":
    main()
