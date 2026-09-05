"""Build comprehensive rulings and apply Phase D."""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/product/nutrients"
REVIEW_PATH = OUT / "species_fdc_mapping_review_v1.json"
RULINGS_PATH = OUT / "species_fdc_rulings_v1.json"
AUTO_PATH = OUT / "species_fdc_auto_confident_v1.parquet"

# Explicit human rulings: species_id -> (fdc_id, note)
EXPLICIT_RULINGS: dict[str, tuple[int, str]] = {
  # JUDGMENT accepted
  "SP_000237": (174760, "Beef composite trimmed retail cuts lean only raw"),
  "SP_000259": (168230, "Pork fresh loin whole lean only raw"),
  "SP_000235": (173686, "Fish salmon Atlantic wild raw"),
  "SP_000288": (171265, "Milk whole 3.25% with vitamin D"),
  "SP_000287": (173414, "Cheese cheddar generic"),
  "SP_000030": (173227, "Beverages tea black brewed tap water"),
  "SP_000415": (173227, "Black tea brewed beverage"),
  "SP_000416": (171917, "Green tea brewed regular"),
  # JUDGMENT overrides
  "SP_000189": (171077, "Chicken breast skinless boneless meat only raw (closest to meat-only raw)"),
  "SP_000177": (175297, "Game meat boar wild raw"),
  "SP_000437": (172175, "Cheese blue"),
  # AMBIGUOUS named
  "SP_000066": (169247, "Lettuce cos or romaine raw"),
  "SP_000003": (170000, "Onions raw"),
  "SP_000002": (168153, "Kiwifruit green raw"),
  "SP_000047": (171890, "Coffee brewed tap water"),
  "SP_000140": (173944, "Bananas raw"),
  "SP_000068": (172420, "Lentils raw"),
  "SP_000001": (170388, "Cabbage savoy raw"),
  "SP_000080": (169094, "Olives ripe canned small-extra large (no raw olive in FDC)"),
  # NO-MATCH fix
  "SP_000094": (170931, "Spices pepper black"),
}


def score_raw_preference(desc: str) -> float:
    d = desc.lower()
    s = 0.0
    if re.search(r",\s*raw\b", d) or d.endswith(" raw"):
        s += 50
    if "mature seeds, raw" in d:
        s += 45
    if d.startswith("spices, ") and ("ground" in d or "dried" in d or "black" in d):
        s += 40
    if "cooked" in d:
        s -= 40
    if "canned" in d and "olives" not in d:
        s -= 25
    if "frozen" in d:
        s -= 25
    if "with salt" in d:
        s -= 20
    if "overripe" in d:
        s -= 30
    if any(x in d for x in ("restaurant", "snacks", "fast foods", "olive garden")):
        s -= 100
    return s


def pick_best_candidate(candidates: list[dict]) -> tuple[int, str, str] | None:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda c: (-score_raw_preference(c["description"]), -c.get("score", 0)),
    )
    best = ranked[0]
    return int(best["fdc_id"]), best["description"], "default_raw_policy"


def main() -> None:
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    auto = pd.read_parquet(AUTO_PATH)

    # Build index of all classified species
    classified: dict[str, dict] = {}
    for bucket in ("auto_confident_all", "ambiguous", "judgment"):
        items = review.get(bucket) if bucket != "auto_confident_all" else review.get("auto_confident_all", [])
        if bucket == "auto_confident_all" and not items:
            items = review.get("auto_confident_sample_n30", [])
        for item in items or []:
            classified[item["species_id"]] = {**item, "_bucket": bucket.replace("_all", "")}

    # Also load full auto from parquet
    for _, r in auto.iterrows():
        classified[r["species_id"]] = {
            "species_id": r["species_id"],
            "canonical_name": r["canonical_name"],
            "fdc_id": int(r["fdc_id"]),
            "fdc_description": r["fdc_description"],
            "_bucket": "auto_confident",
        }

    rulings_cfg = {
        "accept_auto_confident": True,
        "accept_judgment_defaults": False,
        "rulings": {},
        "auto_resolved_ambiguous": [],
    }

    for sid, (fdc_id, note) in EXPLICIT_RULINGS.items():
        name = classified.get(sid, {}).get("canonical_name", sid)
        rulings_cfg["rulings"][sid] = {
            "fdc_id": fdc_id,
            "canonical_name": name,
            "note": note,
            "mapping_class": "human_ruling",
        }

    # Auto-resolve remaining ambiguous
    for item in review.get("ambiguous", []):
        sid = item["species_id"]
        if sid in rulings_cfg["rulings"]:
            continue
        picked = pick_best_candidate(item.get("candidates", []))
        if picked:
            fdc_id, desc, _ = picked
            rulings_cfg["rulings"][sid] = {
                "fdc_id": fdc_id,
                "canonical_name": item["canonical_name"],
                "note": f"auto raw/base policy -> {desc[:60]}",
                "mapping_class": "default_raw_policy",
            }
            rulings_cfg["auto_resolved_ambiguous"].append(
                {
                    "species_id": sid,
                    "canonical_name": item["canonical_name"],
                    "fdc_id": fdc_id,
                    "fdc_description": desc,
                }
            )

    RULINGS_PATH.write_text(json.dumps(rulings_cfg, indent=2), encoding="utf-8")

    # Dedupe NO-MATCH
    mapped_ids = set(rulings_cfg["rulings"].keys()) | set(auto["species_id"])
    true_no_match = []
    for item in review.get("no_match", []):
        sid = item["species_id"]
        if sid not in mapped_ids and sid not in rulings_cfg["rulings"]:
            true_no_match.append(item)

    # Pepper might still be in no_match list - already in explicit
    deduped_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "original_no_match_count": len(review.get("no_match", [])),
        "true_no_match_count": len(true_no_match),
        "removed_duplicates": len(review.get("no_match", [])) - len(true_no_match),
        "true_no_match": sorted(true_no_match, key=lambda x: -x.get("n_strings", 0)),
        "coverage_projection": {
            "auto_confident": len(auto),
            "explicit_rulings": len(EXPLICIT_RULINGS),
            "auto_resolved_ambiguous": len(rulings_cfg["auto_resolved_ambiguous"]),
            "total_mapped": len(mapped_ids) + len(
                [s for s in rulings_cfg["rulings"] if s not in mapped_ids]
            ),
        },
    }
    # recompute total
    all_mapped = set(auto["species_id"]) | set(rulings_cfg["rulings"].keys())
    deduped_report["coverage_projection"]["total_mapped"] = len(all_mapped)
    deduped_report["coverage_projection"]["unmapped"] = 463 - len(all_mapped)

    (OUT / "species_fdc_no_match_deduped_v1.json").write_text(
        json.dumps(deduped_report, indent=2), encoding="utf-8"
    )

    print("Rulings written:", len(rulings_cfg["rulings"]))
    print("Auto-resolved ambiguous:", len(rulings_cfg["auto_resolved_ambiguous"]))
    print("True NO-MATCH:", len(true_no_match), "(was", len(review.get("no_match", [])), ")")
    print("Total mapped:", len(all_mapped), "/ 463")
    print("\nAuto-resolved ambiguous list:")
    for x in rulings_cfg["auto_resolved_ambiguous"]:
        print(f"  {x['canonical_name']:28} -> {x['fdc_description'][:55]}")


if __name__ == "__main__":
    main()
