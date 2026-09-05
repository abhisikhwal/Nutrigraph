#!/usr/bin/env python3
"""
Recipe → body-message engine (data layer).

A recipe is a message to the body: where it lands, what it says, how redundant it is.
No ranking, no dish scoring — convergence is expected and informative.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from theme_definitions import BODY_REGION_THEMES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = ROOT / "data/processed/product/ingredient_profiles_v1_1.jsonl"
DEFAULT_INDEX_DIR = ROOT / "data/processed/product/indexes"
DEFAULT_STRING_MAP = ROOT / "data/processed/canonical/ingredient_string_species_v2.parquet"

REGION_IDS = list(BODY_REGION_THEMES.keys())
UNIQUE_CONTRIBUTOR_CHANGE_THRESHOLD = 0.12


@dataclass
class ResolvedIngredient:
    input_string: str
    species_id: str
    canonical_name: str
    profile: dict[str, Any]


@dataclass
class UnresolvedIngredient:
    input_string: str
    reason: str = "no_species_match"


@dataclass
class RecipeMessage:
    recipe_id: str
    recipe_label: str
    ingredient_strings: list[str]
    resolved: list[ResolvedIngredient] = field(default_factory=list)
    unresolved: list[UnresolvedIngredient] = field(default_factory=list)
    body_regions: dict[str, float] = field(default_factory=dict)
    body_region_labels: dict[str, str] = field(default_factory=dict)
    region_drivers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    effect_themes: list[dict[str, Any]] = field(default_factory=list)
    evidence_composition: dict[str, Any] = field(default_factory=dict)
    notable_pharmacology: list[dict[str, Any]] = field(default_factory=list)
    redundancy: list[dict[str, Any]] = field(default_factory=list)
    aggregation_notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_label": self.recipe_label,
            "ingredient_strings": self.ingredient_strings,
            "resolution": {
                "n_input": len(self.ingredient_strings),
                "n_resolved": len(self.resolved),
                "n_unresolved": len(self.unresolved),
                "resolved": [
                    {
                        "input_string": r.input_string,
                        "species_id": r.species_id,
                        "canonical_name": r.canonical_name,
                    }
                    for r in self.resolved
                ],
                "unresolved": [
                    {"input_string": u.input_string, "reason": u.reason}
                    for u in self.unresolved
                ],
            },
            "body_regions": {
                "intensities": self.body_regions,
                "labels": self.body_region_labels,
                "drivers": self.region_drivers,
            },
            "effect_themes": self.effect_themes,
            "evidence_composition": self.evidence_composition,
            "notable_pharmacology": self.notable_pharmacology,
            "redundancy": self.redundancy,
            "aggregation_notes": self.aggregation_notes,
        }


class RecipeMessageEngine:
    """Compute body-message vectors and leave-one-out redundancy for recipes."""

    def __init__(
        self,
        profiles_path: Path | str | None = None,
        index_dir: Path | str | None = None,
        string_map_path: Path | str | None = None,
    ) -> None:
        self.profiles_path = Path(profiles_path or DEFAULT_PROFILES)
        self.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        self.string_map_path = Path(string_map_path or DEFAULT_STRING_MAP)
        self._load()

    def _load(self) -> None:
        self.profiles: dict[str, dict[str, Any]] = {}
        with self.profiles_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                profile = json.loads(line)
                sid = profile["ingredient"]["species_id"]
                self.profiles[sid] = profile

        lookup = json.loads(
            (self.index_dir / "ingredient_lookup.json").read_text(encoding="utf-8")
        )
        self.aliases: dict[str, str] = lookup.get("aliases", {})

        effect = json.loads(
            (self.index_dir / "effect_themes.json").read_text(encoding="utf-8")
        )
        self.theme_meta: dict[str, dict[str, Any]] = effect.get("themes", {})
        self.theme_ids = sorted(self.theme_meta.keys())

        # species_id -> theme_id -> index entry (top-100 per theme only)
        self.theme_index_hits: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for theme_id, block in effect.get("effect_to_ingredients", {}).items():
            for row in block.get("ingredients", []):
                sid = row["species_id"]
                self.theme_index_hits[sid][theme_id] = row

        body = json.loads(
            (self.index_dir / "body_region_themes.json").read_text(encoding="utf-8")
        )
        self.body_region_labels = {
            rid: spec.get("label", rid)
            for rid, spec in body.get("body_regions", {}).items()
        }

        self.tissue_to_region: dict[str, str] = {}
        for region_id, spec in BODY_REGION_THEMES.items():
            for tissue in spec["tissues"]:
                self.tissue_to_region[tissue] = region_id

        sm = pd.read_parquet(self.string_map_path)
        sm = sm.copy()
        sm["ingredient_string"] = sm["ingredient_string"].astype(str).str.strip().str.lower()
        self.string_map: dict[str, str] = {}
        for _, row in sm.iterrows():
            key = str(row["ingredient_string"]).lower()
            sid = str(row["species_node"])
            if sid in self.profiles and key not in self.string_map:
                self.string_map[key] = sid
            if pd.notna(row.get("canonical_name")):
                ckey = str(row["canonical_name"]).strip().lower()
                if ckey not in self.string_map and sid in self.profiles:
                    self.string_map[ckey] = sid

    def resolve_ingredient(self, ingredient_string: str) -> ResolvedIngredient | UnresolvedIngredient:
        raw = ingredient_string.strip()
        key = raw.lower()
        sid = self.aliases.get(key) or self.string_map.get(key)
        if not sid:
            for alias, candidate in self.aliases.items():
                if key in alias or alias in key:
                    sid = candidate
                    break
        if not sid or sid not in self.profiles:
            return UnresolvedIngredient(input_string=raw)
        profile = self.profiles[sid]
        return ResolvedIngredient(
            input_string=raw,
            species_id=sid,
            canonical_name=profile["ingredient"]["canonical_name"],
            profile=profile,
        )

    def resolve_recipe(self, ingredient_strings: list[str]) -> tuple[list[ResolvedIngredient], list[UnresolvedIngredient]]:
        resolved: list[ResolvedIngredient] = []
        unresolved: list[UnresolvedIngredient] = []
        seen: set[str] = set()
        for s in ingredient_strings:
            result = self.resolve_ingredient(s)
            if isinstance(result, UnresolvedIngredient):
                unresolved.append(result)
                continue
            if result.species_id in seen:
                continue
            seen.add(result.species_id)
            resolved.append(result)
        return resolved, unresolved

    def _ingredient_region_scores(self, profile: dict[str, Any]) -> dict[str, float]:
        """Map profile tissue top-list to body regions (max tissue score per region)."""
        region_scores: dict[str, float] = defaultdict(float)
        for entry in profile.get("tissues", {}).get("top", []):
            tissue = str(entry["tissue"])
            region = self.tissue_to_region.get(tissue)
            if not region:
                continue
            score = float(entry["normalized_score"])
            region_scores[region] = max(region_scores[region], score)
        total = sum(region_scores.values())
        if total <= 0:
            return {}
        return {r: s / total for r, s in region_scores.items()}

    def _ingredient_theme_scores(self, profile: dict[str, Any], species_id: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        index_hits = self.theme_index_hits.get(species_id, {})
        ranked_pathways = {
            str(p["pathway"]): float(p["weighted_fold"])
            for p in profile.get("pathways", {}).get("top_ranked", [])
        }
        for theme_id in self.theme_ids:
            meta = self.theme_meta[theme_id]
            indexed = index_hits.get(theme_id)
            if indexed:
                scores[theme_id] = float(indexed.get("theme_relevance_score", 0.0)) / 100.0
                continue
            pathway_ids = set(
                meta.get("retrieval_pathway_ids") or meta.get("pathway_ids") or []
            )
            hits = [(pid, ranked_pathways[pid]) for pid in pathway_ids if pid in ranked_pathways]
            if not hits:
                scores[theme_id] = 0.0
                continue
            breadth = len(hits) / max(1, len(pathway_ids))
            mean_fold = sum(f for _, f in hits) / len(hits)
            fold_norm = min(1.0, mean_fold / 2.0)
            scores[theme_id] = min(1.0, breadth * fold_norm)
        return scores

    def _aggregate_body_message(
        self,
        ingredients: list[ResolvedIngredient],
    ) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
        """
        Sum equal-weight ingredient region scores, normalize to sum=1 across 14 regions.
        """
        combined: dict[str, float] = {r: 0.0 for r in REGION_IDS}
        drivers_raw: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for ing in ingredients:
            ing_regions = self._ingredient_region_scores(ing.profile)
            for region, score in ing_regions.items():
                combined[region] += score
                drivers_raw[region][ing.canonical_name] += score

        total = sum(combined.values())
        if total > 0:
            combined = {r: v / total for r, v in combined.items()}
        else:
            combined = {r: 0.0 for r in REGION_IDS}

        region_drivers: dict[str, list[dict[str, Any]]] = {}
        for region in REGION_IDS:
            ranked = sorted(
                drivers_raw[region].items(),
                key=lambda x: (-x[1], x[0]),
            )
            region_drivers[region] = [
                {
                    "canonical_name": name,
                    "contribution": round(contrib / max(1, len(ingredients)), 6),
                    "share_of_region": round(contrib / max(1e-9, sum(drivers_raw[region].values())), 4),
                }
                for name, contrib in ranked[:8]
                if contrib > 0
            ]
        return combined, region_drivers

    def _theme_strength_dict(self, ingredients: list[ResolvedIngredient]) -> dict[str, float]:
        totals: dict[str, float] = {tid: 0.0 for tid in self.theme_ids}
        for ing in ingredients:
            for tid, val in self._ingredient_theme_scores(ing.profile, ing.species_id).items():
                totals[tid] += val
        return totals

    def _aggregate_theme_message(
        self,
        ingredients: list[ResolvedIngredient],
    ) -> list[dict[str, Any]]:
        """Sum theme strengths across ingredients; report drivers per theme."""
        theme_totals = self._theme_strength_dict(ingredients)
        theme_drivers: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for ing in ingredients:
            ing_themes = self._ingredient_theme_scores(ing.profile, ing.species_id)
            for theme_id, strength in ing_themes.items():
                if strength <= 0:
                    continue
                theme_drivers[theme_id][ing.canonical_name] += strength

        engaged = sorted(
            ((tid, s) for tid, s in theme_totals.items() if s > 0),
            key=lambda x: -x[1],
        )
        total_strength = sum(v for _, v in engaged) or 1.0
        out: list[dict[str, Any]] = []
        for theme_id, strength in engaged:
            if strength <= 0:
                continue
            meta = self.theme_meta[theme_id]
            drivers = sorted(
                theme_drivers[theme_id].items(),
                key=lambda x: (-x[1], x[0]),
            )
            out.append(
                {
                    "theme_id": theme_id,
                    "label": meta.get("label", theme_id),
                    "strength": round(strength, 4),
                    "normalized_strength": round(strength / total_strength, 4),
                    "driving_ingredients": [
                        {
                            "canonical_name": name,
                            "contribution": round(val, 4),
                        }
                        for name, val in drivers[:6]
                    ],
                }
            )
        return out

    def _evidence_composition(self, ingredients: list[ResolvedIngredient]) -> dict[str, Any]:
        measured_edges = 0
        predicted_edges = 0
        weighted_measured = 0.0
        for ing in ingredients:
            targets = ing.profile.get("targets", {})
            mc = int(targets.get("measured_count", 0))
            pc = int(targets.get("predicted_count", 0))
            measured_edges += mc
            predicted_edges += pc
            mf = float(ing.profile.get("provenance", {}).get("measured_fraction", 0.0))
            weighted_measured += mf * (mc + pc)
        total = measured_edges + predicted_edges
        return {
            "measured_edge_count": measured_edges,
            "predicted_edge_count": predicted_edges,
            "measured_edge_fraction": round(measured_edges / total, 4) if total else 0.0,
            "mean_ingredient_measured_fraction": round(
                sum(float(i.profile.get("provenance", {}).get("measured_fraction", 0.0)) for i in ingredients)
                / max(1, len(ingredients)),
                4,
            ),
            "summary": (
                f"{round(100 * measured_edges / total)}% measured / "
                f"{round(100 * predicted_edges / total)}% predicted target edges"
                if total
                else "no resolved ingredients"
            ),
        }

    def _notable_pharmacology(self, ingredients: list[ResolvedIngredient]) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for ing in ingredients:
            for target in ing.profile.get("targets", {}).get("top", []):
                if str(target.get("evidence")) != "measured":
                    continue
                moa = target.get("moa")
                if not moa:
                    continue
                hits.append(
                    {
                        "ingredient": ing.canonical_name,
                        "input_string": ing.input_string,
                        "gene_symbol": target["gene_symbol"],
                        "moa": moa,
                        "confidence": target.get("confidence"),
                        "confidence_tier": target.get("confidence_tier"),
                        "claim": f"{ing.canonical_name}: {target['gene_symbol']} {moa} (measured)",
                    }
                )
        hits.sort(key=lambda x: (-float(x.get("confidence") or 0), x["ingredient"], x["gene_symbol"]))
        return hits[:25]

    @staticmethod
    def _vector_from_dict(values: dict[str, float], keys: list[str]) -> list[float]:
        return [float(values.get(k, 0.0)) for k in keys]

    @staticmethod
    def _l1_distance(a: list[float], b: list[float]) -> float:
        return sum(abs(x - y) for x, y in zip(a, b))

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na <= 0 or nb <= 0:
            return 1.0 if na <= 0 and nb <= 0 else 0.0
        return dot / (na * nb)

    @staticmethod
    def _normalized_vector(values: dict[str, float], keys: list[str]) -> list[float]:
        vec = [float(values.get(k, 0.0)) for k in keys]
        total = sum(vec)
        if total <= 0:
            return [0.0] * len(keys)
        return [v / total for v in vec]

    def _compute_redundancy(
        self,
        ingredients: list[ResolvedIngredient],
        full_body: dict[str, float],
    ) -> list[dict[str, Any]]:
        if len(ingredients) <= 1:
            return []

        full_theme_vec = self._theme_strength_dict(ingredients)
        theme_keys = self.theme_ids
        body_vec = self._vector_from_dict(full_body, REGION_IDS)
        theme_vec = self._normalized_vector(full_theme_vec, theme_keys)

        rows: list[dict[str, Any]] = []
        for i, ing in enumerate(ingredients):
            subset = ingredients[:i] + ingredients[i + 1 :]
            body_minus, _ = self._aggregate_body_message(subset)
            theme_minus = self._theme_strength_dict(subset)

            body_m = self._vector_from_dict(body_minus, REGION_IDS)
            theme_m = self._normalized_vector(theme_minus, theme_keys)

            body_l1 = self._l1_distance(body_vec, body_m)
            body_cos_delta = 1.0 - self._cosine_similarity(body_vec, body_m)
            theme_l1 = self._l1_distance(theme_vec, theme_m)
            theme_cos_delta = 1.0 - self._cosine_similarity(theme_vec, theme_m)

            body_change = 0.5 * body_l1 + 0.5 * body_cos_delta
            theme_change = 0.5 * theme_l1 + 0.5 * theme_cos_delta
            combined_change = 0.6 * body_change + 0.4 * theme_change
            redundancy_score = max(0.0, min(1.0, 1.0 - combined_change))

            rows.append(
                {
                    "input_string": ing.input_string,
                    "canonical_name": ing.canonical_name,
                    "species_id": ing.species_id,
                    "redundancy_score": round(redundancy_score, 4),
                    "message_change": round(combined_change, 4),
                    "body_redundancy_score": round(max(0.0, min(1.0, 1.0 - body_change)), 4),
                    "theme_redundancy_score": round(max(0.0, min(1.0, 1.0 - theme_change)), 4),
                    "body_region_l1": round(body_l1, 4),
                    "body_region_cosine_delta": round(body_cos_delta, 4),
                    "theme_l1": round(theme_l1, 4),
                    "theme_cosine_delta": round(theme_cos_delta, 4),
                    "is_unique_contributor": combined_change >= UNIQUE_CONTRIBUTOR_CHANGE_THRESHOLD,
                    "interpretation": (
                        "highly redundant — removing barely dims the message"
                        if redundancy_score >= 0.85
                        else "moderately redundant"
                        if redundancy_score >= 0.65
                        else "distinct contributor — removal shifts the message"
                    ),
                }
            )
        rows.sort(key=lambda x: (-x["redundancy_score"], x["canonical_name"]))
        return rows

    def compute_message(
        self,
        ingredient_strings: list[str],
        recipe_id: str = "recipe",
        recipe_label: str = "",
    ) -> RecipeMessage:
        resolved, unresolved = self.resolve_recipe(ingredient_strings)
        msg = RecipeMessage(
            recipe_id=recipe_id,
            recipe_label=recipe_label or recipe_id,
            ingredient_strings=ingredient_strings,
            resolved=resolved,
            unresolved=unresolved,
            body_region_labels=self.body_region_labels,
            aggregation_notes={
                "body_regions": (
                    "Equal-weight sum of per-ingredient tissue scores (profile top tissues) "
                    "mapped GTEx tissue → body region; max tissue score per region per ingredient; "
                    "recipe vector normalized to sum=1. Presence-based (no quantities yet)."
                ),
                "effect_themes": (
                    "Sum of per-ingredient theme strength from effect index when available, "
                    "else profile pathway overlap × fold. Equal weight per ingredient."
                ),
                "redundancy": (
                    "Leave-one-out on normalized body-region + theme vectors "
                    "(60% body, 40% theme; L1 + cosine delta). "
                    "redundancy_score = 1 - combined_change (high = barely matters)."
                ),
            },
        )
        if not resolved:
            return msg

        body, drivers = self._aggregate_body_message(resolved)
        themes = self._aggregate_theme_message(resolved)
        msg.body_regions = {k: round(v, 6) for k, v in body.items()}
        msg.region_drivers = drivers
        msg.effect_themes = themes
        msg.evidence_composition = self._evidence_composition(resolved)
        msg.notable_pharmacology = self._notable_pharmacology(resolved)
        msg.redundancy = self._compute_redundancy(resolved, body)
        return msg
