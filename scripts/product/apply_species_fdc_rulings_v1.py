#!/usr/bin/env python3
"""
Merge human FDC mapping rulings into species_fdc_map_approved_v1.parquet.

Usage:
  1. Review species_fdc_review_ambiguous_v1.csv and species_fdc_review_judgment_v1.csv
  2. Fill species_fdc_rulings_v1.json with chosen fdc_id per species_id
  3. python scripts/product/apply_species_fdc_rulings_v1.py
  4. python scripts/product/build_species_fdc_map_v1.py --phase compose

Rulings JSON format:
{
  "rulings": {
    "SP_000066": {"fdc_id": 169247, "note": "Lettuce butterhead raw"},
    "SP_000003": {"fdc_id": 170000, "note": "Onions raw generic"}
  },
  "accept_judgment_defaults": true,
  "accept_auto_confident": true
}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/product/nutrients"
REVIEW_PATH = OUT_DIR / "species_fdc_mapping_review_v1.json"
RULINGS_PATH = OUT_DIR / "species_fdc_rulings_v1.json"
APPROVED_PATH = OUT_DIR / "species_fdc_map_approved_v1.parquet"
CLEAN_POOL = OUT_DIR / "fdc_clean_pool_v1.parquet"
AUTO_PATH = OUT_DIR / "species_fdc_auto_confident_v1.parquet"


def main() -> None:
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    clean = pd.read_parquet(CLEAN_POOL)

    if RULINGS_PATH.exists():
        cfg = json.loads(RULINGS_PATH.read_text(encoding="utf-8"))
    else:
        cfg = {"rulings": {}, "accept_judgment_defaults": True, "accept_auto_confident": True}

    rows: dict[str, dict] = {}

    if cfg.get("accept_auto_confident", True) and AUTO_PATH.exists():
        auto = pd.read_parquet(AUTO_PATH)
        for _, r in auto.iterrows():
            rows[r["species_id"]] = {
                "species_id": r["species_id"],
                "canonical_name": r["canonical_name"],
                "fdc_id": int(r["fdc_id"]),
                "mapping_class": "auto_confident",
            }

    if cfg.get("accept_judgment_defaults", True):
        for item in review.get("judgment", []):
            sid = item["species_id"]
            if sid not in rows and item.get("fdc_id"):
                rows[sid] = {
                    "species_id": sid,
                    "canonical_name": item["canonical_name"],
                    "fdc_id": int(item["fdc_id"]),
                    "source": "judgment_default",
                    "mapping_class": "judgment_default",
                }

    for sid, ruling in cfg.get("rulings", {}).items():
        mclass = ruling.get("mapping_class", "human_ruling")
        rows[sid] = {
            "species_id": sid,
            "canonical_name": ruling.get("canonical_name"),
            "fdc_id": int(ruling["fdc_id"]),
            "mapping_class": mclass,
            "note": ruling.get("note"),
        }

    approved = pd.DataFrame(rows.values())
    approved = approved.merge(
        clean[["fdc_id", "description", "data_type"]].rename(
            columns={"description": "fdc_description", "data_type": "fdc_data_type"}
        ),
        on="fdc_id",
        how="left",
    )
    approved.to_parquet(APPROVED_PATH, index=False)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_mapped": len(approved),
        "sources": approved["mapping_class"].value_counts().to_dict(),
    }
    (OUT_DIR / "species_fdc_map_approved_report_v1.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
