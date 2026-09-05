"""Relative mechanistic contribution within a dish (NOT absolute dosing)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import json
from pathlib import Path


@dataclass
class MechanismContribution:
    theme_id: str
    theme_label: str
    ingredient_id: str
    canonical_name: str
    mass_g: float
    potency_proxy: float
    weighted_score: float
    relative_contribution: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "theme_label": self.theme_label,
            "ingredient_id": self.ingredient_id,
            "canonical_name": self.canonical_name,
            "mass_g": self.mass_g,
            "potency_proxy": round(self.potency_proxy, 4),
            "weighted_score": round(self.weighted_score, 4),
            "relative_contribution": round(self.relative_contribution, 4),
            "rank": self.rank,
        }


@dataclass
class HeroAnalysis:
    bulk_hero: dict[str, Any] | None
    potency_hero: dict[str, Any] | None
    mechanistic_heroes: dict[str, dict[str, Any]]
    contributions: list[MechanismContribution] = field(default_factory=list)
    note: str = (
        "Relative within-dish contribution only. "
        "No absolute mechanism dose or RDI for compounds."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bulk_hero": self.bulk_hero,
            "potency_hero": self.potency_hero,
            "mechanistic_heroes": self.mechanistic_heroes,
            "contributions": [c.to_dict() for c in self.contributions],
            "note": self.note,
        }


def potency_proxy_from_profile(profile: dict[str, Any]) -> float:
    """
    Potency proxy = compound density × enrichment strength.
    - compound_density: log-scaled compound count (FooDB compounds or blend aggregate)
    - enrichment_strength: mean weighted pathway fold among top pathways
    """
    compounds = profile.get("compounds", {})
    n_comp = max(0, int(compounds.get("count", 0)))
    compound_density = math.log1p(n_comp) / math.log1p(5000)  # normalize ~0-1

    pathways = profile.get("pathways", {}).get("top_ranked", [])
    if pathways:
        folds = [float(p.get("weighted_fold") or 0) for p in pathways[:10]]
        enrichment = sum(folds) / len(folds)
        enrichment_strength = min(1.0, enrichment / 3.0)
    else:
        enrichment_strength = 0.0

    measured_frac = float(profile.get("provenance", {}).get("measured_fraction") or 0)
    base = compound_density * (0.5 + 0.5 * enrichment_strength)
    return max(0.01, base * (1.0 + 0.25 * measured_frac))


def theme_strength_for_ingredient(
    profile: dict[str, Any],
    species_id: str,
    theme_id: str,
    theme_meta: dict[str, Any],
    theme_index_hits: dict[str, dict[str, Any]],
) -> float:
    """Per-ingredient theme relevance (0-1), aligned with recipe_message_engine."""
    indexed = theme_index_hits.get(theme_id)
    if indexed:
        return float(indexed.get("theme_relevance_score", 0.0)) / 100.0
    pathway_ids = set(
        theme_meta.get("retrieval_pathway_ids") or theme_meta.get("pathway_ids") or []
    )
    ranked = {
        str(p["pathway"]): float(p.get("weighted_fold") or 0)
        for p in profile.get("pathways", {}).get("top_ranked", [])
    }
    hits = [(pid, ranked[pid]) for pid in pathway_ids if pid in ranked]
    if not hits:
        return 0.0
    breadth = len(hits) / max(1, len(pathway_ids))
    fold_norm = sum(v for _, v in hits) / len(hits)
    return min(1.0, breadth * min(1.0, fold_norm / 2.0))


def compute_mechanism_contribution(
    ingredients: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    theme_meta: dict[str, dict[str, Any]],
    theme_index_hits: dict[str, dict[str, dict[str, Any]]],
    top_themes: int = 5,
) -> HeroAnalysis:
    """
    Weight = mass_g × potency_proxy × theme_strength.
    Returns relative contribution per theme + bulk vs mechanistic hero distinction.
    """
    rows: list[MechanismContribution] = []
    theme_totals: dict[str, float] = {}

    for ing in ingredients:
        iid = ing.get("ingredient_id")
        mass = ing.get("mass_g") or 0.0
        if not iid or iid not in profiles or mass <= 0:
            continue
        profile = profiles[iid]
        potency = potency_proxy_from_profile(profile)
        cname = profile["ingredient"]["canonical_name"]
        index_hits = theme_index_hits.get(iid, {})

        for theme_id, meta in theme_meta.items():
            ts = theme_strength_for_ingredient(profile, iid, theme_id, meta, index_hits)
            if ts <= 0:
                continue
            weighted = (mass ** 0.3) * (potency ** 1.1) * ts
            theme_totals[theme_id] = theme_totals.get(theme_id, 0.0) + weighted
            rows.append(MechanismContribution(
                theme_id=theme_id,
                theme_label=meta.get("label", theme_id),
                ingredient_id=iid,
                canonical_name=cname,
                mass_g=mass,
                potency_proxy=potency,
                weighted_score=weighted,
                relative_contribution=0.0,
                rank=0,
            ))

    # normalize within each theme
    by_theme: dict[str, list[MechanismContribution]] = {}
    for r in rows:
        total = theme_totals.get(r.theme_id, 0) or 1.0
        r.relative_contribution = r.weighted_score / total
        by_theme.setdefault(r.theme_id, []).append(r)
    for theme_id, theme_rows in by_theme.items():
        theme_rows.sort(key=lambda x: -x.relative_contribution)
        for i, r in enumerate(theme_rows, 1):
            r.rank = i

    # bulk hero = highest mass
    mass_ranked = sorted(
        [i for i in ingredients if i.get("mass_g")],
        key=lambda x: -(x.get("mass_g") or 0),
    )
    bulk_hero = None
    if mass_ranked:
        top = mass_ranked[0]
        prof = profiles.get(top.get("ingredient_id"), {})
        bulk_hero = {
            "ingredient_id": top.get("ingredient_id"),
            "canonical_name": prof.get("ingredient", {}).get("canonical_name"),
            "mass_g": top.get("mass_g"),
            "share_of_recipe_mass": round(
                (top.get("mass_g") or 0) / sum(i.get("mass_g") or 0 for i in ingredients), 4
            ) if ingredients else 0,
        }

    # mechanistic hero per top theme (by relative contribution, not mass)
    theme_engaged = sorted(theme_totals.items(), key=lambda x: -x[1])[:top_themes]
    mech_heroes: dict[str, dict[str, Any]] = {}
    for theme_id, _ in theme_engaged:
        theme_rows = by_theme.get(theme_id, [])
        if not theme_rows:
            continue
        winner = theme_rows[0]
        mech_heroes[theme_id] = {
            "theme_label": winner.theme_label,
            "ingredient_id": winner.ingredient_id,
            "canonical_name": winner.canonical_name,
            "relative_contribution": winner.relative_contribution,
            "mass_g": winner.mass_g,
            "potency_proxy": winner.potency_proxy,
            "interpretation": (
                f"{winner.canonical_name} contributes {winner.relative_contribution:.0%} "
                f"of this dish's {winner.theme_label} engagement (mass×potency, relative only)."
            ),
        }

    # potency hero = highest intrinsic potency (compound×enrichment), mass-independent
    potency_ranked = sorted(
        [(i, potency_proxy_from_profile(profiles[i["ingredient_id"]]))
         for i in ingredients if i.get("ingredient_id") in profiles],
        key=lambda x: -x[1],
    )
    potency_hero = None
    if potency_ranked:
        ing, pot = potency_ranked[0]
        prof = profiles[ing["ingredient_id"]]
        potency_hero = {
            "ingredient_id": ing["ingredient_id"],
            "canonical_name": prof["ingredient"]["canonical_name"],
            "potency_proxy": round(pot, 4),
            "mass_g": ing.get("mass_g"),
            "note": "Highest intrinsic potency (compound density × enrichment); not mass-weighted.",
        }

    return HeroAnalysis(
        bulk_hero=bulk_hero,
        potency_hero=potency_hero,
        mechanistic_heroes=mech_heroes,
        contributions=sorted(rows, key=lambda x: (-x.relative_contribution, x.theme_id)),
    )


def load_theme_context(index_dir: Path) -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    effect = json.loads((index_dir / "effect_themes.json").read_text(encoding="utf-8"))
    theme_meta = effect.get("themes", {})
    theme_index: dict[str, dict[str, dict]] = {}
    for theme_id, block in effect.get("effect_to_ingredients", {}).items():
        for row in block.get("ingredients", []):
            sid = row["species_id"]
            theme_index.setdefault(sid, {})[theme_id] = row
    return theme_meta, theme_index
