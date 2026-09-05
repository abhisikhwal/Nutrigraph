#!/usr/bin/env python3
"""
Retrieval API surface for ingredient mechanism explorer (data layer).

Loads flat JSON indexes built by build_retrieval_indexes_v1.py.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = ROOT / "data/processed/product/indexes"
PROFILES_JSONL = ROOT / "data/processed/product/ingredient_profiles_v1_1.jsonl"


class IngredientRetrieval:
    """Grounded retrieval over v1.1 ingredient profiles."""

    def __init__(self, index_dir: Path | str | None = None) -> None:
        self.index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        self._load()

    def _load_json(self, name: str) -> Any:
        path = self.index_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing index file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load(self) -> None:
        self.tissue_index = self._load_json("tissue_to_ingredients.json")
        self.pathway_index = self._load_json("pathway_to_ingredients.json")
        self.target_index = self._load_json("target_to_ingredients.json")
        self.ingredient_lookup = self._load_json("ingredient_lookup.json")
        self.effect_themes = self._load_json("effect_themes.json")
        self.body_regions = self._load_json("body_region_themes.json")
        self.cuisine_layer = self._load_json("cuisine_distinctive_contributors.json")
        self.aliases = self.ingredient_lookup.get("aliases", {})
        self.profiles: dict[str, Any] = {}
        self._profiles_loaded = False

    def _ensure_profiles(self) -> None:
        if self._profiles_loaded:
            return
        path = PROFILES_JSONL
        if not path.exists():
            raise FileNotFoundError(f"Missing profiles file: {path}")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                self.profiles[p["ingredient"]["species_id"]] = p
        self._profiles_loaded = True

    def _resolve_species_id(self, query: str) -> str | None:
        q = query.strip()
        self._ensure_profiles()
        if q in self.profiles:
            return q
        key = q.lower()
        if key in self.aliases:
            return self.aliases[key]
        # fuzzy: canonical name case-insensitive
        for sid, info in self.ingredient_lookup.get("by_species_id", {}).items():
            if info.get("canonical_name", "").lower() == key:
                return sid
        return None

    def profile(self, ingredient: str) -> dict[str, Any] | None:
        """Resolve ingredient name, alias, or species_id to full v1.1 profile."""
        self._ensure_profiles()
        sid = self._resolve_species_id(ingredient)
        if not sid:
            return None
        return self.profiles.get(sid)

    def ingredients_by_tissue(self, tissue: str, top_n: int = 20) -> dict[str, Any]:
        """Ranked ingredients for a GTEx tissue label."""
        # exact match first
        if tissue in self.tissue_index:
            key = tissue
        else:
            # case-insensitive / partial
            matches = [t for t in self.tissue_index if t.lower() == tissue.lower()]
            if not matches:
                matches = [t for t in self.tissue_index if tissue.lower() in t.lower()]
            if not matches:
                return {"tissue": tissue, "found": False, "ingredients": []}
            key = matches[0]
        entries = self.tissue_index[key][:top_n]
        return {
            "tissue": key,
            "found": True,
            "n_results": len(entries),
            "ingredients": entries,
            "interpretation_note": (
                "Scores reflect target gene expression localization (GTEx), "
                "not proof of compound delivery to tissue."
            ),
        }

    def ingredients_by_body_region(self, region: str, top_n: int = 20) -> dict[str, Any]:
        """Ranked ingredients aggregated across a body-region theme (e.g. liver, gut, brain)."""
        region_key = region.strip().lower().replace(" ", "_")
        region_data = self.body_regions.get("region_to_ingredients", {}).get(region_key)
        if not region_data:
            # try label match
            for rid, spec in self.body_regions.get("body_regions", {}).items():
                if spec.get("label", "").lower() == region.lower() or rid == region_key:
                    region_data = self.body_regions["region_to_ingredients"].get(rid)
                    region_key = rid
                    break
        if not region_data:
            return {"region": region, "found": False, "ingredients": []}
        entries = region_data.get("ingredients", [])[:top_n]
        return {
            "region_id": region_key,
            "label": region_data.get("label"),
            "tissues": region_data.get("tissues_indexed", []),
            "found": True,
            "n_results": len(entries),
            "ingredients": entries,
            "interpretation_note": (
                "Aggregated target-gene expression across GTEx tissues in this body region."
            ),
        }

    def ingredients_by_effect(self, theme: str, top_n: int = 20) -> dict[str, Any]:
        """Ranked ingredients for an effect/system theme (e.g. inflammation_immune)."""
        theme_key = theme.strip().lower().replace(" ", "_").replace("-", "_")
        effect_index = self.effect_themes.get("effect_to_ingredients", {})
        if theme_key not in effect_index:
            for tid, block in effect_index.items():
                if block.get("label", "").lower() == theme.lower():
                    theme_key = tid
                    break
            else:
                # partial label match
                for tid, block in effect_index.items():
                    if theme.lower() in block.get("label", "").lower():
                        theme_key = tid
                        break
        block = effect_index.get(theme_key)
        if not block:
            return {"theme": theme, "found": False, "ingredients": []}
        entries = block.get("ingredients", [])[:top_n]
        return {
            "theme_id": theme_key,
            "label": block.get("label"),
            "n_pathways_in_theme": block.get("n_pathways"),
            "n_retrieval_pathways": block.get("n_retrieval_pathways", block.get("n_pathways")),
            "include_sub_themes": block.get("include_sub_themes", []),
            "ranking_formula": block.get("ranking_formula"),
            "adaptive_weights": block.get("adaptive_weights", {}).get("adapted_weights"),
            "found": True,
            "n_results": len(entries),
            "ingredients": entries,
        }

    def ingredients_by_pathway(self, pathway: str, top_n: int = 20) -> dict[str, Any]:
        """Ranked ingredients for a pathway ID or name substring."""
        pid = pathway.strip()
        block = self.pathway_index.get(pid)
        if not block:
            for k, v in self.pathway_index.items():
                if pid.lower() in v.get("pathway_name", "").lower():
                    block = v
                    pid = k
                    break
        if not block:
            return {"pathway": pathway, "found": False, "ingredients": []}
        entries = block.get("ingredients", [])[:top_n]
        return {
            "pathway_id": pid,
            "pathway_name": block.get("pathway_name"),
            "found": True,
            "n_results": len(entries),
            "ingredients": entries,
        }

    def ingredients_by_target(self, gene: str, top_n: int = 20) -> dict[str, Any]:
        """Ranked ingredients hitting a gene/receptor target."""
        gene_key = gene.strip().upper()
        block = self.target_index.get(gene_key)
        if not block:
            return {"gene_symbol": gene_key, "found": False, "ingredients": []}
        entries = block.get("ingredients", [])[:top_n]
        return {
            "gene_symbol": gene_key,
            "found": True,
            "n_results": len(entries),
            "n_ingredients_total": block.get("n_ingredients_total"),
            "ingredients": entries,
        }

    def cuisine_themes(self, cuisine: str) -> dict[str, Any]:
        """Distinctive contributors + collective mechanistic themes for a cuisine."""
        cid = cuisine.strip().lower().replace(" ", "_")
        cuisines = self.cuisine_layer.get("cuisines", {})
        block = cuisines.get(cid)
        if not block:
            for k, v in cuisines.items():
                if v.get("label", "").lower() == cuisine.lower():
                    block = v
                    cid = k
                    break
        if not block:
            return {"cuisine": cuisine, "found": False}
        return {"found": True, **block}

    def list_effect_themes(self) -> list[dict[str, str]]:
        return [
            {"theme_id": tid, "label": meta.get("label", tid)}
            for tid, meta in self.effect_themes.get("themes", {}).items()
        ]

    def list_body_regions(self) -> list[dict[str, str]]:
        return [
            {"region_id": rid, "label": spec.get("label", rid)}
            for rid, spec in self.body_regions.get("body_regions", {}).items()
        ]


@lru_cache(maxsize=1)
def get_retriever() -> IngredientRetrieval:
    return IngredientRetrieval()


# Convenience functions for AI layer integration
def ingredients_by_tissue(tissue: str, top_n: int = 20) -> dict[str, Any]:
    return get_retriever().ingredients_by_tissue(tissue, top_n=top_n)


def ingredients_by_body_region(region: str, top_n: int = 20) -> dict[str, Any]:
    return get_retriever().ingredients_by_body_region(region, top_n=top_n)


def ingredients_by_effect(theme: str, top_n: int = 20) -> dict[str, Any]:
    return get_retriever().ingredients_by_effect(theme, top_n=top_n)


def ingredients_by_target(gene: str, top_n: int = 20) -> dict[str, Any]:
    return get_retriever().ingredients_by_target(gene, top_n=top_n)


def profile(ingredient: str) -> dict[str, Any] | None:
    return get_retriever().profile(ingredient)


def cuisine_themes(cuisine: str) -> dict[str, Any]:
    return get_retriever().cuisine_themes(cuisine)
