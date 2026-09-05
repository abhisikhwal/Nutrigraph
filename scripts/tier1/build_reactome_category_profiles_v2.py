#!/usr/bin/env python3
"""
Finer recipe-resolution category profiles (v2).

- Reactome pathways: aggregate at sub-level-1 (189 categories).
- GO pathways: multi-slim is_a walk + depth-based ancestor fallback (fixes 597
  generic GO_NS:biological_process fallbacks from v1).

Usage (from repo root):
    python scripts/tier1/build_reactome_category_profiles_v2.py

Outputs (new only; v1 untouched):
    data/processed/tier1/pathway_category_map_v2.parquet
    data/processed/tier1/ingredient_category_profiles_v2.parquet
    data/processed/tier1/reactome_category_build_report_v2.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "thread2"))

from integrate_weighted_edges import collapse_metrics  # noqa: E402

_v1_spec = importlib.util.spec_from_file_location(
    "build_reactome_category_profiles_v1", SCRIPT_DIR / "build_reactome_category_profiles_v1.py"
)
_v1 = importlib.util.module_from_spec(_v1_spec)
assert _v1_spec.loader is not None
_v1_spec.loader.exec_module(_v1)

TIER1 = ROOT / "data/processed/tier1"
HIERARCHY = TIER1 / "reactome_category_hierarchy_v1.json"
PATHWAY_MAP_V2 = TIER1 / "pathway_category_map_v2.parquet"
PROFILES_V2 = TIER1 / "ingredient_category_profiles_v2.parquet"
REPORT_V2 = TIER1 / "reactome_category_build_report_v2.json"

GO_OBO = ROOT / "data/raw/pathways/go-basic.obo"
ENRICHMENT_WEIGHTED = TIER1 / "enrichment_weighted_v3_calibrated.parquet"
ENRICHMENT_MEASURED = TIER1 / "enrichment_measured_only_v3.parquet"
Q_OPERATING = 0.10

GO_NAMESPACE_ROOTS = {
    "biological_process": "GO:0008150",
    "molecular_function": "GO:0003674",
    "cellular_component": "GO:0005575",
}
NAMESPACE_ROOT_IDS = set(GO_NAMESPACE_ROOTS.values())

# Priority order: richer slims first, generic last.
GOSLIM_PRIORITY = [
    "goslim_agr",
    "goslim_chembl",
    "goslim_mouse",
    "goslim_generic",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_go_obo_extended(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, set[str]]]:
    """Parse GO terms, is_a edges, and all goslim subset memberships."""
    terms: dict[str, dict[str, Any]] = {}
    is_a: dict[str, set[str]] = defaultdict(set)
    slim_sets: dict[str, set[str]] = {s: set() for s in GOSLIM_PRIORITY}
    current_id: str | None = None

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line == "[Term]":
                current_id = None
                continue
            if not line or line.startswith("["):
                continue
            if line.startswith("id:"):
                current_id = line.split(":", 1)[1].strip()
                terms[current_id] = {"name": current_id, "namespace": "", "is_obsolete": False}
            elif current_id and line.startswith("name:"):
                terms[current_id]["name"] = line.split(":", 1)[1].strip()
            elif current_id and line.startswith("namespace:"):
                terms[current_id]["namespace"] = line.split(":", 1)[1].strip()
            elif current_id and line.startswith("is_obsolete:") and "true" in line.lower():
                terms[current_id]["is_obsolete"] = True
            elif current_id and line.startswith("is_a:"):
                parent = line.split("!", 1)[0].replace("is_a:", "").strip()
                is_a[current_id].add(parent)
            elif current_id and line.startswith("subset:"):
                subset = line.split(":", 1)[1].strip()
                if subset in slim_sets:
                    slim_sets[subset].add(current_id)

    return terms, is_a, slim_sets


def shortest_path_to_namespace_root(
    go_id: str,
    is_a: dict[str, set[str]],
    terms: dict[str, dict[str, Any]],
) -> list[str] | None:
    """Return path [namespace_root, ..., go_id] via shortest upward BFS."""
    ns = terms.get(go_id, {}).get("namespace", "biological_process")
    root = GO_NAMESPACE_ROOTS.get(ns, "GO:0008150")
    if go_id == root:
        return [root]

    queue: deque[tuple[str, list[str]]] = deque([(go_id, [go_id])])
    visited = {go_id}
    while queue:
        node, path = queue.popleft()
        for parent in sorted(is_a.get(node, [])):
            if parent in visited:
                continue
            visited.add(parent)
            new_path = [parent] + path
            if parent == root:
                return new_path
            queue.append((parent, new_path))
    return None


def map_go_to_fine_v2(
    go_id: str,
    is_a: dict[str, set[str]],
    terms: dict[str, dict[str, Any]],
    slim_sets: dict[str, set[str]],
) -> tuple[str, str, str, str]:
    """
    Return (fine_category_id, fine_category_name, mapping_method, slim_used).
    """
    if go_id in NAMESPACE_ROOT_IDS:
        ns = terms.get(go_id, {}).get("namespace", "biological_process")
        return f"GO_NS:{ns}", terms.get(go_id, {}).get("name", ns), "namespace_root", ""

    # Priority 1: multi-slim is_a walk (BFS upward, first hit in priority order).
    queue: deque[str] = deque([go_id])
    visited = {go_id}
    slim_hit: str | None = None
    slim_used = ""
    while queue and slim_hit is None:
        node = queue.popleft()
        for slim_name in GOSLIM_PRIORITY:
            if node in slim_sets[slim_name]:
                slim_hit = node
                slim_used = slim_name
                break
        if slim_hit:
            break
        for parent in is_a.get(node, []):
            if parent not in visited and not terms.get(parent, {}).get("is_obsolete", False):
                visited.add(parent)
                queue.append(parent)

    if slim_hit and slim_hit not in NAMESPACE_ROOT_IDS:
        name = terms.get(slim_hit, {}).get("name", slim_hit)
        return slim_hit, name, f"goslim_{slim_used}", slim_used

    # Priority 2: depth-2 ancestor under namespace root (not the root itself).
    path = shortest_path_to_namespace_root(go_id, is_a, terms)
    if path and len(path) >= 3:
        anc = path[1]  # one step below root = level-2 ancestor
        if anc not in NAMESPACE_ROOT_IDS:
            name = terms.get(anc, {}).get("name", anc)
            return anc, name, "go_depth_2_ancestor", ""
    if path and len(path) == 2:
        anc = path[0]
        if anc not in NAMESPACE_ROOT_IDS:
            name = terms.get(anc, {}).get("name", anc)
            return anc, name, "go_depth_1_ancestor", ""

    # Priority 3: use term itself if specific enough.
    if go_id not in NAMESPACE_ROOT_IDS:
        name = terms.get(go_id, {}).get("name", go_id)
        return go_id, name, "go_term_self", ""

    ns = terms.get(go_id, {}).get("namespace", "biological_process")
    return f"GO_NS:{ns}", f"GO_NS:{ns}", "namespace_root_fallback", ""


def build_pathway_category_map_v2(
    pathway_ids: set[str],
    reactome_names: dict[str, str],
    reactome_parents: dict[str, set[str]],
    reactome_roots: set[str],
    go_terms: dict[str, dict[str, Any]],
    go_is_a: dict[str, set[str]],
    slim_sets: dict[str, set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pid in sorted(pathway_ids):
        if pid.startswith("R-HSA-"):
            path = _v1.shortest_path_to_root(pid, reactome_parents, reactome_roots)
            if path:
                top = path[0]
                sub1 = path[1] if len(path) > 2 else top
                sub2 = path[2] if len(path) > 3 else sub1
                fine_id, fine_name = sub1, reactome_names.get(sub1, sub1)
                fine_source = "reactome_sub_level_1"
            else:
                top = sub1 = sub2 = fine_id = pid
                fine_name = reactome_names.get(pid, pid)
                fine_source = "reactome_unmapped"
            rows.append(
                {
                    "pathway_id": pid,
                    "mapping_source": "reactome",
                    "top_level_id": top,
                    "top_level_name": reactome_names.get(top, top),
                    "sub_level_1_id": sub1,
                    "sub_level_1_name": reactome_names.get(sub1, sub1),
                    "sub_level_2_id": sub2,
                    "sub_level_2_name": reactome_names.get(sub2, sub2),
                    "fine_category_id": fine_id,
                    "fine_category_name": fine_name,
                    "fine_category_source": fine_source,
                    "hierarchy_depth": len(path) if path else 0,
                }
            )
        else:
            go_id = _v1.extract_go_id(pid)
            if go_id:
                fine_id, fine_name, method, slim_used = map_go_to_fine_v2(
                    go_id, go_is_a, go_terms, slim_sets
                )
                ns = go_terms.get(go_id, {}).get("namespace", "biological_process")
                root = GO_NAMESPACE_ROOTS.get(ns, "GO:0008150")
                rows.append(
                    {
                        "pathway_id": pid,
                        "mapping_source": "go_fine_v2",
                        "top_level_id": fine_id if fine_id.startswith("GO_NS:") else fine_id,
                        "top_level_name": fine_name,
                        "sub_level_1_id": fine_id,
                        "sub_level_1_name": fine_name,
                        "sub_level_2_id": fine_id,
                        "sub_level_2_name": fine_name,
                        "fine_category_id": fine_id,
                        "fine_category_name": fine_name,
                        "fine_category_source": method,
                        "go_id": go_id,
                        "go_slim_used": slim_used,
                        "go_namespace_root": root,
                        "hierarchy_depth": 0,
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
                        "fine_category_id": "UNMAPPED",
                        "fine_category_name": "Unmapped",
                        "fine_category_source": "unmapped",
                        "hierarchy_depth": 0,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_fine_profiles(
    enrichment: pd.DataFrame,
    pathway_map: pd.DataFrame,
    enrichment_layer: str,
    q_thr: float,
) -> pd.DataFrame:
    """Aggregate at fine_recipe level (+ legacy levels for comparison)."""
    pm = pathway_map.set_index("pathway_id")
    sig = enrichment[enrichment["q_value"] < q_thr].copy()
    rows: list[dict[str, Any]] = []

    level_specs = [
        ("fine_recipe", "fine_category_id", "fine_category_name"),
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


def main() -> int:
    print("=== Finer category profiles v2 (Reactome sub-L1 + GO fix) ===", flush=True)
    TIER1.mkdir(parents=True, exist_ok=True)

    pre_enrich_w = sha256_file(ENRICHMENT_WEIGHTED)
    hierarchy = json.loads(HIERARCHY.read_text(encoding="utf-8"))

    reactome_names, reactome_parents, reactome_roots = _v1.load_reactome_human()
    go_terms, go_is_a, slim_sets = parse_go_obo_extended(GO_OBO)
    print(
        f"  GO slims: {', '.join(f'{k}={len(v)}' for k, v in slim_sets.items())}",
        flush=True,
    )

    weighted = pd.read_parquet(ENRICHMENT_WEIGHTED)
    measured = pd.read_parquet(ENRICHMENT_MEASURED)
    all_pathway_ids = set(weighted["pathway_id"].astype(str).unique()) | set(
        measured["pathway_id"].astype(str).unique()
    )

    pathway_map = build_pathway_category_map_v2(
        all_pathway_ids,
        reactome_names,
        reactome_parents,
        reactome_roots,
        go_terms,
        go_is_a,
        slim_sets,
    )
    pathway_map.to_parquet(PATHWAY_MAP_V2, index=False)

    go_rows = pathway_map[pathway_map["mapping_source"] == "go_fine_v2"].copy()
    v1_map = pd.read_parquet(TIER1 / "pathway_category_map_v1.parquet")
    v1_go = v1_map[v1_map["mapping_source"] == "go_slim_generic"]
    v1_root_ids = set(v1_go.loc[v1_go["top_level_id"].str.startswith("GO_NS"), "pathway_id"])

    recovered = go_rows[~go_rows["fine_category_id"].str.startswith("GO_NS")]
    still_root = go_rows[go_rows["fine_category_id"].str.startswith("GO_NS")]
    recovered_from_597 = go_rows[
        go_rows["pathway_id"].isin(v1_root_ids)
        & ~go_rows["fine_category_id"].str.startswith("GO_NS")
    ]

    method_counts = go_rows["fine_category_source"].value_counts().to_dict()
    slim_used_counts = go_rows.loc[go_rows["go_slim_used"] != "", "go_slim_used"].value_counts().to_dict()

    profiles_w = aggregate_fine_profiles(weighted, pathway_map, "weighted_calibrated", Q_OPERATING)
    profiles_m = aggregate_fine_profiles(measured, pathway_map, "measured_only", Q_OPERATING)
    profiles = pd.concat([profiles_w, profiles_m], ignore_index=True)
    profiles.to_parquet(PROFILES_V2, index=False)

    fine_w = profiles_w[profiles_w["category_level"] == "fine_recipe"]
    sub1_w = profiles_w[profiles_w["category_level"] == "sub_level_1"]
    sub2_w = profiles_w[profiles_w["category_level"] == "sub_level_2"]

    n_fine_cats = fine_w["category_id"].nunique()
    n_sub1_cats = sub1_w["category_id"].nunique()
    n_sub2_cats = sub2_w["category_id"].nunique()

    # Ingredient-level GO_NS share at fine level.
    fine_by_ing = fine_w.pivot_table(
        index="ingredient_id", columns="category_id", values="aggregated_enrichment", fill_value=0.0
    )
    go_ns_cols = [c for c in fine_by_ing.columns if str(c).startswith("GO_NS:")]
    if go_ns_cols and fine_by_ing.sum(axis=1).gt(0).any():
        go_ns_share = (
            fine_by_ing[go_ns_cols].sum(axis=1) / fine_by_ing.sum(axis=1).replace(0, float("nan"))
        ).mean()
    else:
        go_ns_share = 0.0

    distinct_fine = _v1.category_distinctness(profiles_w, "fine_recipe")
    distinct_sub1 = _v1.category_distinctness(profiles_w, "sub_level_1")
    distinct_sub2 = _v1.category_distinctness(profiles_w, "sub_level_2")

    report: dict[str, Any] = {
        "phase": "REACTOME_CATEGORY_HIERARCHY_V2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "primary_recipe_resolution": "fine_recipe (Reactome sub-L1 + GO fine mapping)",
        "hierarchy_reference": str(HIERARCHY.relative_to(ROOT)),
        "reactome_sub_level_counts": {
            "sub_level_1": hierarchy["sub_level_1_count"],
            "sub_level_2": hierarchy["sub_level_2_count"],
        },
        "go_fix": {
            "strategy": (
                "Priority 1: BFS is_a walk to goslim_agr → goslim_chembl → goslim_mouse → "
                "goslim_generic. Priority 2: level-2 ancestor below namespace root. "
                "Priority 3: term self. Last resort: GO_NS namespace root."
            ),
            "goslim_set_sizes": {k: len(v) for k, v in slim_sets.items()},
            "go_pathways_total": int(len(go_rows)),
            "v1_generic_root_count": int(len(v1_root_ids)),
            "v2_still_generic_root": int(len(still_root)),
            "v2_recovered_specific": int(len(recovered)),
            "recovered_from_v1_597_root": int(len(recovered_from_597)),
            "recovery_rate_of_597": round(len(recovered_from_597) / max(1, len(v1_root_ids)), 4),
            "mapping_method_counts": method_counts,
            "slim_used_counts": slim_used_counts,
        },
        "category_counts_in_use_weighted_calibrated": {
            "fine_recipe": int(n_fine_cats),
            "sub_level_1_reactome_native": int(n_sub1_cats),
            "sub_level_2_reactome_native": int(n_sub2_cats),
            "chosen_primary": "fine_recipe",
            "sub_l2_note": (
                "sub-L2 retained for coverage reporting; 507 categories are sparser — "
                "see distinct_sub_level_2 vs fine_recipe."
            ),
        },
        "go_ns_share_mean_across_ingredients_fine_level": round(float(go_ns_share), 4),
        "go_ns_share_v1_top_level_reference": 0.49,
        "category_distinctness": {
            "fine_recipe": distinct_fine,
            "sub_level_1": distinct_sub1,
            "sub_level_2": distinct_sub2,
        },
        "enrichment_inputs_unchanged": {
            "weighted_sha256": pre_enrich_w,
            "unchanged_after_build": pre_enrich_w == sha256_file(ENRICHMENT_WEIGHTED),
        },
        "outputs": {
            "pathway_category_map_v2": str(PATHWAY_MAP_V2.relative_to(ROOT)),
            "ingredient_category_profiles_v2": str(PROFILES_V2.relative_to(ROOT)),
        },
    }
    REPORT_V2.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"  GO fix: {len(recovered_from_597)}/{len(v1_root_ids)} recovered from v1 root", flush=True)
    print(f"  Fine categories in use: {n_fine_cats} (sub-L1={n_sub1_cats}, sub-L2={n_sub2_cats})", flush=True)
    print(f"  Mean ingredient GO_NS share (fine): {go_ns_share:.4f}", flush=True)
    print(f"  Wrote {PROFILES_V2}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
