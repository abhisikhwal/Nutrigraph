#!/usr/bin/env python3
"""
Build Reactome-based multi-level category system for enriched pathway profiles.

Replaces the 13 keyword-heuristic categories with Reactome's authoritative hierarchy
(+ GO-slim for GO-only pathways). New outputs only.

Usage (from repo root):
    python scripts/tier1/build_reactome_category_profiles_v1.py

Outputs:
    data/processed/tier1/reactome_category_hierarchy_v1.json
    data/processed/tier1/pathway_category_map_v1.parquet
    data/processed/tier1/ingredient_category_profiles_v1.parquet
    data/processed/tier1/reactome_category_build_report_v1.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "thread2"))

from integrate_weighted_edges import collapse_metrics  # noqa: E402

REACTOME_PATHWAYS = ROOT / "data/raw/pathways/ReactomePathways.txt"
REACTOME_RELATIONS = ROOT / "data/raw/pathways/ReactomePathwaysRelation.txt"
GO_OBO = ROOT / "data/raw/pathways/go-basic.obo"
PATHWAY_BUNDLES = ROOT / "data/processed/features/pathway_bundles.json"

ENRICHMENT_WEIGHTED = ROOT / "data/processed/tier1/enrichment_weighted_v3_calibrated.parquet"
ENRICHMENT_MEASURED = ROOT / "data/processed/tier1/enrichment_measured_only_v3.parquet"

TIER1 = ROOT / "data/processed/tier1"
HIERARCHY_OUT = TIER1 / "reactome_category_hierarchy_v1.json"
PATHWAY_MAP_OUT = TIER1 / "pathway_category_map_v1.parquet"
PROFILES_OUT = TIER1 / "ingredient_category_profiles_v1.parquet"
REPORT_OUT = TIER1 / "reactome_category_build_report_v1.json"

Q_OPERATING = 0.10
OLD_13_CATEGORIES = [
    "apoptosis",
    "cardiovascular",
    "cell_cycle",
    "dna_repair",
    "hormone",
    "immune",
    "metabolism",
    "nervous",
    "other",
    "signaling",
    "transport",
    "cell_signaling",
    "oxidative_stress",
]
SAMPLE_INGREDIENTS = ["SP_000052", "SP_000005", "SP_000259", "SP_000235", "SP_000026"]
STRING_MAP = ROOT / "data/processed/canonical/ingredient_string_species_v2.parquet"

GO_NAMESPACE_ROOTS = {
    "biological_process": "GO:0008150",
    "molecular_function": "GO:0003674",
    "cellular_component": "GO:0005575",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reactome_human() -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    names: dict[str, str] = {}
    human: set[str] = set()
    with REACTOME_PATHWAYS.open(encoding="utf-8") as fh:
        for line in fh:
            pid, name, species = line.strip().split("\t")
            if species == "Homo sapiens":
                human.add(pid)
                names[pid] = name

    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    with REACTOME_RELATIONS.open(encoding="utf-8") as fh:
        for line in fh:
            parent, child = line.strip().split("\t")
            if parent in human and child in human:
                parents[child].add(parent)
                children[parent].add(child)

    roots = {pid for pid in human if pid not in parents}
    return names, parents, roots


def shortest_path_to_root(
    pathway_id: str,
    parents: dict[str, set[str]],
    roots: set[str],
) -> list[str] | None:
    """Return path root -> ... -> pathway (shortest upward BFS)."""
    if pathway_id in roots:
        return [pathway_id]
    queue: deque[tuple[str, list[str]]] = deque([(pathway_id, [pathway_id])])
    visited = {pathway_id}
    best: list[str] | None = None
    while queue:
        node, path = queue.popleft()
        for parent in sorted(parents.get(node, [])):
            if parent in visited:
                continue
            visited.add(parent)
            new_path = [parent] + path
            if parent in roots:
                if best is None or len(new_path) < len(best):
                    best = new_path
            else:
                queue.append((parent, new_path))
    return best


def build_reactome_hierarchy(
    names: dict[str, str],
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
    roots: set[str],
) -> dict[str, Any]:
    top_level = sorted(roots, key=lambda x: names.get(x, x))
    sub1_by_root: dict[str, list[str]] = {}
    sub2_by_sub1: dict[str, list[str]] = {}
    for root in top_level:
        sub1 = sorted(children.get(root, []), key=lambda x: names.get(x, x))
        sub1_by_root[root] = sub1
        for s1 in sub1:
            sub2_by_sub1[s1] = sorted(children.get(s1, []), key=lambda x: names.get(x, x))

    n_sub1 = sum(len(v) for v in sub1_by_root.values())
    n_sub2 = sum(len(v) for v in sub2_by_sub1.values())

    return {
        "top_level_count": len(top_level),
        "sub_level_1_count": n_sub1,
        "sub_level_2_count": n_sub2,
        "top_level_categories": [
            {
                "category_id": rid,
                "category_name": names.get(rid, rid),
                "n_sub_level_1": len(sub1_by_root.get(rid, [])),
            }
            for rid in top_level
        ],
        "sub_level_1_by_root": {
            rid: [{"category_id": s, "category_name": names.get(s, s)} for s in sub1_by_root.get(rid, [])]
            for rid in top_level
        },
    }


def parse_go_obo(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], set[str]]:
    terms: dict[str, dict[str, Any]] = {}
    is_a: dict[str, set[str]] = defaultdict(set)
    goslim_generic: set[str] = set()
    current_id: str | None = None

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line == "[Term]":
                current_id = None
                continue
            if not line:
                continue
            if line.startswith("id:"):
                current_id = line.split(":", 1)[1].strip()
                terms[current_id] = {"name": current_id, "namespace": "", "is_goslim_generic": False}
            elif current_id and line.startswith("name:"):
                terms[current_id]["name"] = line.split(":", 1)[1].strip()
            elif current_id and line.startswith("namespace:"):
                terms[current_id]["namespace"] = line.split(":", 1)[1].strip()
            elif current_id and line.startswith("is_a:"):
                parent = line.split("!", 1)[0].replace("is_a:", "").strip()
                is_a[current_id].add(parent)
            elif current_id and line.startswith("subset:") and "goslim_generic" in line:
                terms[current_id]["is_goslim_generic"] = True
                goslim_generic.add(current_id)

    return terms, is_a, goslim_generic


def map_go_to_slim(
    go_id: str,
    is_a: dict[str, set[str]],
    goslim_generic: set[str],
    terms: dict[str, dict[str, Any]],
) -> tuple[str, str, str]:
    """Return (top_category_id, sub1_id, sub2_id) for GO term via goslim_generic walk."""
    if go_id in goslim_generic:
        name = terms.get(go_id, {}).get("name", go_id)
        return go_id, go_id, go_id

    queue: deque[str] = deque([go_id])
    visited = {go_id}
    found_slim: str | None = None
    while queue:
        node = queue.popleft()
        if node in goslim_generic:
            found_slim = node
            break
        for parent in is_a.get(node, []):
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)

    if found_slim:
        name = terms.get(found_slim, {}).get("name", found_slim)
        return found_slim, found_slim, found_slim

    ns = terms.get(go_id, {}).get("namespace", "biological_process")
    root = GO_NAMESPACE_ROOTS.get(ns, "GO:0008150")
    root_name = terms.get(root, {}).get("name", ns.replace("_", " "))
    cat_id = f"GO_NS:{ns}"
    return cat_id, root, root


def extract_go_id(pathway_id: str) -> str | None:
    m = re.search(r"(GO:\d+)", pathway_id)
    return m.group(1) if m else None


def build_pathway_category_map(
    pathway_ids: set[str],
    reactome_names: dict[str, str],
    reactome_parents: dict[str, set[str]],
    reactome_roots: set[str],
    go_terms: dict[str, dict[str, Any]],
    go_is_a: dict[str, set[str]],
    goslim_generic: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pid in sorted(pathway_ids):
        if pid.startswith("R-HSA-"):
            path = shortest_path_to_root(pid, reactome_parents, reactome_roots)
            if path:
                top = path[0]
                sub1 = path[1] if len(path) > 2 else top
                sub2 = path[2] if len(path) > 3 else sub1
                source = "reactome"
            else:
                top = sub1 = sub2 = pid
                source = "reactome_unmapped"
            rows.append(
                {
                    "pathway_id": pid,
                    "mapping_source": source,
                    "top_level_id": top,
                    "top_level_name": reactome_names.get(top, top),
                    "sub_level_1_id": sub1,
                    "sub_level_1_name": reactome_names.get(sub1, sub1),
                    "sub_level_2_id": sub2,
                    "sub_level_2_name": reactome_names.get(sub2, sub2),
                    "hierarchy_depth": len(path) if path else 0,
                }
            )
        else:
            go_id = extract_go_id(pid)
            if go_id:
                top, sub1, sub2 = map_go_to_slim(go_id, go_is_a, goslim_generic, go_terms)
                rows.append(
                    {
                        "pathway_id": pid,
                        "mapping_source": "go_slim_generic",
                        "top_level_id": top,
                        "top_level_name": go_terms.get(top, {}).get("name", top),
                        "sub_level_1_id": sub1,
                        "sub_level_1_name": go_terms.get(sub1, {}).get("name", sub1),
                        "sub_level_2_id": sub2,
                        "sub_level_2_name": go_terms.get(sub2, {}).get("name", sub2),
                        "hierarchy_depth": 0,
                        "go_id": go_id,
                    }
                )
            else:
                rows.append(
                    {
                        "pathway_id": pid,
                        "mapping_source": "unmapped",
                        "top_level_id": "UNMAPPED",
                        "top_level_name": "Unmapped",
                        "sub_level_1_id": "UNMAPPED",
                        "sub_level_1_name": "Unmapped",
                        "sub_level_2_id": "UNMAPPED",
                        "sub_level_2_name": "Unmapped",
                        "hierarchy_depth": 0,
                    }
                )
    return pd.DataFrame(rows)


def old_keyword_categories(pathway_id: str, bundles: dict[str, list[str]]) -> list[str]:
    """Legacy keyword heuristic (comparison only — not used in new profiles)."""
    text = pathway_id.lower()
    hits = []
    for cat in OLD_13_CATEGORIES:
        keywords = bundles.get(cat, [])
        if any(kw.lower() in text for kw in keywords):
            hits.append(cat)
    return hits or ["other"]


def aggregate_category_profiles(
    enrichment: pd.DataFrame,
    pathway_map: pd.DataFrame,
    enrichment_layer: str,
    q_thr: float,
) -> pd.DataFrame:
    pm = pathway_map.set_index("pathway_id")
    sig = enrichment[enrichment["q_value"] < q_thr].copy()
    rows: list[dict[str, Any]] = []

    level_specs = [
        ("top_level", "top_level_id", "top_level_name"),
        ("sub_level_1", "sub_level_1_id", "sub_level_1_name"),
        ("sub_level_2", "sub_level_2_id", "sub_level_2_name"),
    ]

    for ing, grp in sig.groupby("ingredient_id"):
        for level_name, id_col, name_col in level_specs:
            cat_agg: dict[str, dict[str, Any]] = defaultdict(
                lambda: {
                    "weighted_sum": 0.0,
                    "fold_weighted_sum": 0.0,
                    "n_pathways": 0,
                    "w_meas": 0.0,
                    "w_pred": 0.0,
                    "name": "",
                }
            )
            for _, r in grp.iterrows():
                pid = str(r["pathway_id"])
                if pid not in pm.index:
                    continue
                cat_id = str(pm.loc[pid, id_col])
                cat_name = str(pm.loc[pid, name_col])
                w = float(r["weighted_contribution"])
                fold = float(r["weighted_fold_enrichment"])
                cat_agg[cat_id]["weighted_sum"] += w
                cat_agg[cat_id]["fold_weighted_sum"] += fold * w
                cat_agg[cat_id]["n_pathways"] += 1
                cat_agg[cat_id]["w_meas"] += float(r.get("weighted_measured", 0))
                cat_agg[cat_id]["w_pred"] += float(r.get("weighted_predicted", 0))
                cat_agg[cat_id]["name"] = cat_name

            for cat_id, agg in cat_agg.items():
                wsum = agg["weighted_sum"]
                rows.append(
                    {
                        "ingredient_id": str(ing),
                        "enrichment_layer": enrichment_layer,
                        "category_id": cat_id,
                        "category_name": agg["name"],
                        "category_level": level_name,
                        "aggregated_enrichment": wsum,
                        "aggregated_weighted_fold": (agg["fold_weighted_sum"] / wsum) if wsum > 0 else 0.0,
                        "n_pathways": agg["n_pathways"],
                        "weighted_measured": agg["w_meas"],
                        "weighted_predicted": agg["w_pred"],
                        "frac_predicted_weight": (agg["w_pred"] / wsum) if wsum > 0 else 0.0,
                        "q_threshold": q_thr,
                    }
                )
    return pd.DataFrame(rows)


def category_distinctness(
    profiles: pd.DataFrame,
    level: str,
) -> dict[str, Any]:
    sub = profiles[profiles["category_level"] == level]
    sigs: dict[str, frozenset[str]] = {}
    for ing, grp in sub.groupby("ingredient_id"):
        # signature = categories with any enrichment mass
        active = grp[grp["aggregated_enrichment"] > 0]
        sigs[str(ing)] = frozenset(active["category_id"].astype(str))
    metrics = collapse_metrics(sigs)
    metrics["n_empty_profiles"] = sum(1 for v in sigs.values() if len(v) == 0)
    metrics["unique_fraction"] = round(metrics["n_unique_gene_sets"] / max(metrics["n_ingredients"], 1), 3)
    return metrics


def main() -> int:
    print("=== Reactome category hierarchy build ===", flush=True)
    TIER1.mkdir(parents=True, exist_ok=True)

    pre_enrich_w = sha256_file(ENRICHMENT_WEIGHTED)
    pre_enrich_m = sha256_file(ENRICHMENT_MEASURED)

    reactome_names, reactome_parents, reactome_roots = load_reactome_human()
    children: dict[str, set[str]] = defaultdict(set)
    for child, ps in reactome_parents.items():
        for p in ps:
            children[p].add(child)

    hierarchy = build_reactome_hierarchy(reactome_names, reactome_parents, children, reactome_roots)
    HIERARCHY_OUT.write_text(json.dumps(hierarchy, indent=2), encoding="utf-8")

    print(f"  Reactome top-level categories: {hierarchy['top_level_count']}", flush=True)
    print(f"  Sub-level 1: {hierarchy['sub_level_1_count']}, sub-level 2: {hierarchy['sub_level_2_count']}", flush=True)

    go_terms, go_is_a, goslim_generic = parse_go_obo(GO_OBO)
    print(f"  GO terms parsed: {len(go_terms)}, goslim_generic: {len(goslim_generic)}", flush=True)

    weighted = pd.read_parquet(ENRICHMENT_WEIGHTED)
    measured = pd.read_parquet(ENRICHMENT_MEASURED)
    all_pathway_ids = set(weighted["pathway_id"].astype(str).unique()) | set(
        measured["pathway_id"].astype(str).unique()
    )

    pathway_map = build_pathway_category_map(
        all_pathway_ids,
        reactome_names,
        reactome_parents,
        reactome_roots,
        go_terms,
        go_is_a,
        goslim_generic,
    )
    pathway_map.to_parquet(PATHWAY_MAP_OUT, index=False)

    mapping_stats = {
        "reactome_top_level": int((pathway_map["mapping_source"] == "reactome").sum()),
        "go_slim": int((pathway_map["mapping_source"] == "go_slim_generic").sum()),
        "unmapped": int((pathway_map["mapping_source"] == "unmapped").sum()),
        "unique_top_level_reactome": int(
            pathway_map.loc[pathway_map["mapping_source"] == "reactome", "top_level_id"].nunique()
        ),
        "unique_top_level_go": int(
            pathway_map.loc[pathway_map["mapping_source"] == "go_slim_generic", "top_level_id"].nunique()
        ),
    }

    profiles_weighted = aggregate_category_profiles(
        weighted, pathway_map, "weighted_calibrated", Q_OPERATING
    )
    profiles_measured = aggregate_category_profiles(
        measured, pathway_map, "measured_only", Q_OPERATING
    )
    profiles = pd.concat([profiles_weighted, profiles_measured], ignore_index=True)
    profiles.to_parquet(PROFILES_OUT, index=False)

    bundles = json.loads(PATHWAY_BUNDLES.read_text(encoding="utf-8"))
    string_map = pd.read_parquet(STRING_MAP)
    ing_names = string_map.groupby("species_node")["canonical_name"].first().astype(str).to_dict()

    old_vs_new = []
    for ing in SAMPLE_INGREDIENTS:
        sig_pids = set(
            weighted.loc[
                (weighted["ingredient_id"] == ing) & (weighted["q_value"] < Q_OPERATING),
                "pathway_id",
            ].astype(str)
        )
        new_top: dict[str, float] = defaultdict(float)
        new_sub1: dict[str, float] = defaultdict(float)
        pm = pathway_map.set_index("pathway_id")
        for pid in sig_pids:
            if pid not in pm.index:
                continue
            row = weighted[(weighted["ingredient_id"] == ing) & (weighted["pathway_id"] == pid)].iloc[0]
            w = float(row["weighted_contribution"])
            new_top[str(pm.loc[pid, "top_level_name"])] += w
            new_sub1[str(pm.loc[pid, "sub_level_1_name"])] += w

        old_cats: dict[str, int] = defaultdict(int)
        for pid in sig_pids:
            for cat in old_keyword_categories(pid, bundles):
                old_cats[cat] += 1

        old_vs_new.append(
            {
                "ingredient_id": ing,
                "name": ing_names.get(ing, ing),
                "n_sig_pathways": len(sig_pids),
                "old_13_keyword_profile": dict(sorted(old_cats.items(), key=lambda kv: -kv[1])),
                "new_reactome_top_level": dict(
                    sorted(new_top.items(), key=lambda kv: -kv[1])[:10]
                ),
                "new_reactome_sub_level_1_top10": dict(
                    sorted(new_sub1.items(), key=lambda kv: -kv[1])[:10]
                ),
            }
        )

    distinct_top = category_distinctness(profiles_weighted, "top_level")
    distinct_sub1 = category_distinctness(profiles_weighted, "sub_level_1")
    distinct_sub2 = category_distinctness(profiles_weighted, "sub_level_2")

    report: dict[str, Any] = {
        "phase": "REACTOME_CATEGORY_HIERARCHY_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "replaces": "13 keyword-heuristic categories (pathway_bundles.json / phase14_config DEFAULT_CATEGORIES)",
        "hierarchy": {
            "top_level_count": hierarchy["top_level_count"],
            "sub_level_1_count": hierarchy["sub_level_1_count"],
            "sub_level_2_count": hierarchy["sub_level_2_count"],
            "top_level_names": [c["category_name"] for c in hierarchy["top_level_categories"]],
        },
        "go_handling": {
            "approach": "GO-slim generic (goslim_generic subset from go-basic.obo); fallback to namespace root",
            "note": "GO pathways are not in Reactome tree; mapped via is_a walk to goslim_generic ancestor",
        },
        "mapping_coverage": mapping_stats,
        "enrichment_inputs": {
            "primary": str(ENRICHMENT_WEIGHTED.relative_to(ROOT)),
            "baseline": str(ENRICHMENT_MEASURED.relative_to(ROOT)),
            "q_threshold": Q_OPERATING,
        },
        "layers_unchanged": {
            "enrichment_weighted_v3_calibrated_sha256": pre_enrich_w,
            "post_build_weighted_sha256": sha256_file(ENRICHMENT_WEIGHTED),
            "enrichment_measured_only_v3_sha256": pre_enrich_m,
            "unchanged": pre_enrich_w == sha256_file(ENRICHMENT_WEIGHTED)
            and pre_enrich_m == sha256_file(ENRICHMENT_MEASURED),
        },
        "outputs": {
            "reactome_category_hierarchy_v1": str(HIERARCHY_OUT.relative_to(ROOT)),
            "pathway_category_map_v1": str(PATHWAY_MAP_OUT.relative_to(ROOT)),
            "ingredient_category_profiles_v1": str(PROFILES_OUT.relative_to(ROOT)),
        },
        "old_vs_new_comparison": old_vs_new,
        "category_distinctness_weighted_calibrated": {
            "top_level_coarse": distinct_top,
            "sub_level_1_fine": distinct_sub1,
            "sub_level_2_finer": distinct_sub2,
            "interpretation": (
                "Expect re-convergence at top-level (shared coarse buckets) and sharper distinctness "
                "at sub-level 1/2 — use top-level for user-facing buckets, sub-level for precise claims."
            ),
        },
        "no_keyword_matching_in_new_path": True,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n  Mapping: Reactome={mapping_stats['reactome_top_level']} GO-slim={mapping_stats['go_slim']}", flush=True)
    print(f"  Distinctness top-level: {distinct_top['n_unique_gene_sets']}/{distinct_top['n_ingredients']}", flush=True)
    print(f"  Distinctness sub-level-1: {distinct_sub1['n_unique_gene_sets']}/{distinct_sub1['n_ingredients']}", flush=True)
    print(f"Wrote {PROFILES_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
