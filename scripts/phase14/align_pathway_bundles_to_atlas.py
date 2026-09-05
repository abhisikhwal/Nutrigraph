"""
Align pathway_bundles.json to all atlas categories so propagation zeros from missing_bundle are reduced.
- Load pathway_bundles and atlas; compare keys to unique atlas categories (normalized).
- Identify missing bundles; for each, build or extend bundle from pathway_cluster_info (and target_functional_clusters).
- Update pathway_bundles.json programmatically (keys: lowercase, underscore).
- Re-run propagation on 50 atlas rows and print new non-zero rate.
Does NOT change propagation math; bundle alignment only.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd


def normalize_category(s: str) -> str:
    """Lowercase, strip, replace spaces/hyphens with underscores, remove unsafe chars for key."""
    if not s or not isinstance(s, str):
        return ""
    s = str(s).strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_terms(top_terms_str) -> list:
    """Parse top_terms column (string repr of list or comma-sep) to list of strings."""
    if pd.isna(top_terms_str) or not str(top_terms_str).strip():
        return []
    s = str(top_terms_str).strip()
    try:
        if s.startswith("["):
            out = ast.literal_eval(s)
            return [str(x).strip() for x in out] if isinstance(out, list) else [s]
    except Exception:
        pass
    return [p.strip().strip("'\"") for p in re.findall(r"[^,\[\]]+", s) if p.strip()]


def terms_and_sample_pathways(row: pd.Series) -> str:
    """Single searchable string from top_terms and sample_pathways."""
    terms = parse_terms(row.get("top_terms", ""))
    sp = row.get("sample_pathways", "")
    if pd.notna(sp) and str(sp).strip():
        try:
            if str(sp).startswith("["):
                sample = ast.literal_eval(str(sp))
            else:
                sample = [p.strip().strip("'\"") for p in str(sp).split(",")]
            if isinstance(sample, list):
                terms.extend([str(x) for x in sample if x])
        except Exception:
            terms.append(str(sp))
    return " ".join(str(t).lower() for t in terms)


def main() -> int:
    # Paths
    data = REPO_ROOT / "data" / "processed"
    bundles_path = data / "features" / "pathway_bundles.json"
    cluster_path = data / "features" / "pathway_cluster_info.csv"
    target_path = data / "features" / "target_functional_clusters.csv"
    atlas_path = data / "phase13_interactions_v3_20260206_162122_b_gpu_stable" / "atlas_confirmed.csv"
    for d in data.glob("phase13_*"):
        if (d / "atlas_confirmed.csv").exists():
            atlas_path = d / "atlas_confirmed.csv"
            break

    if not bundles_path.exists():
        print("ERROR: pathway_bundles.json not found at", bundles_path)
        return 1
    if not atlas_path.exists():
        print("ERROR: atlas_confirmed.csv not found at", atlas_path)
        return 1

    # 1) Load pathway_bundles and print keys
    with open(bundles_path, "r", encoding="utf-8") as f:
        pathway_bundles = json.load(f)
    bundle_keys = list(pathway_bundles.keys())
    print("1) pathway_bundles.json keys:", bundle_keys)

    # 2) Atlas unique categories (normalized)
    atlas = pd.read_csv(atlas_path)
    if "category" not in atlas.columns:
        print("ERROR: atlas has no 'category' column")
        return 1
    raw_cats = atlas["category"].dropna().astype(str).str.strip().unique().tolist()
    atlas_categories_norm = sorted(set(normalize_category(c) for c in raw_cats if normalize_category(c)))
    print("2) Atlas unique categories (normalized):", atlas_categories_norm)

    # 3) Which categories are missing a bundle key? (key exists in bundle with same normalized name)
    bundle_keys_norm = {normalize_category(k): k for k in bundle_keys}
    missing_key = [c for c in atlas_categories_norm if c not in bundle_keys_norm]
    # Also: which bundle keys have ZERO pathway clusters matching? (cardiovascular has key but may match 0)
    print("3) Atlas categories with no bundle key (missing):", missing_key)

    # Load pathway_cluster_info and target_functional_clusters for matching and building
    if cluster_path.exists():
        pathway_cluster_info = pd.read_csv(cluster_path)
    else:
        pathway_cluster_info = pd.DataFrame(columns=["cluster_id", "top_terms", "sample_pathways"])
    if target_path.exists():
        target_clusters = pd.read_csv(target_path)
        if "sample_pathways" not in target_clusters.columns:
            target_clusters["sample_pathways"] = ""
    else:
        target_clusters = pd.DataFrame(columns=["cluster_id", "top_terms", "sample_pathways"])

    def count_matches(bundles: dict) -> dict:
        """For each bundle key, count how many pathway clusters (from both sources) match."""
        counts = defaultdict(int)
        for _, row in pathway_cluster_info.iterrows():
            text = terms_and_sample_pathways(row)
            for cat, keywords in bundles.items():
                for kw in keywords:
                    if kw.lower() in text:
                        counts[cat] += 1
                        break
        for _, row in target_clusters.iterrows():
            text = terms_and_sample_pathways(row)
            for cat, keywords in bundles.items():
                for kw in keywords:
                    if kw.lower() in text:
                        counts[cat] += 1
                        break
        return dict(counts)

    current_matches = count_matches(pathway_bundles)
    empty_bundles = [k for k in pathway_bundles if current_matches.get(k, 0) == 0]
    print("   Bundle keys that currently match 0 pathway clusters (empty):", empty_bundles)

    # 4) Build/extend bundles for missing and empty
    # Theme keywords per atlas category to search in cluster text
    theme_search = {
        "cardiovascular": ["cardiac", "heart", "vascular", "vascula", "blood", "platelet", "endothel", "smooth muscle", "thrombin", "coagulation", "hypertension", "atherosclerosis", "angiogenesis", "vegf", "artery", "capillary"],
        "cell_cycle": ["cycle", "mitotic", "mitosis", "cell division", "cyclin", "cdk", "replicative", "g1", "g2", "s phase", "spindle", "centrosome"],
        "dna_repair": ["dna repair", "dna damage", "base excision", "nucleotide excision", "mismatch repair", "recombination", "double strand", "ap site", "xrcc", "brca"],
        "hormone": ["hormone", "steroid", "estrogen", "androgen", "insulin", "receptor", "nuclear receptor", "glucocorticoid", "thyroid", "growth factor"],
        "immune": ["immune", "immun", "lymphocyte", "t cell", "b cell", "cytokine", "interleukin", "interferon", "nf-kappa", "nfkb", "inflamm"],
        "nervous": ["neuron", "neural", "neurotransmitter", "synap", "gaba", "acetylcholine", "dopamine", "serotonin", "receptor"],
        "other": ["pathway", "signaling", "regulation", "metabolism", "cell"],
        "signaling": ["signal", "signaling", "kinase", "mapk", "receptor", "phosphorylation", "cascade", "jak", "stat", "pi3k", "akt"],
        "translation": ["translation", "mrna", "ribosome", "protein synthesis", "elongation", "initiation", "eif"],
        "transport": ["transport", "transporter", "membrane", "channel", "atp-binding", "abc ", "slc", "uptake", "efflux"],
    }

    # Normalize existing bundle keys (lowercase, underscore) and keep values; avoid duplicates
    new_bundles = {}
    for k, v in pathway_bundles.items():
        nk = normalize_category(k) or re.sub(r"[^\w]", "_", k.lower().replace(" ", "_")).strip("_")
        if nk not in new_bundles:
            new_bundles[nk] = list(v) if isinstance(v, list) else []
        else:
            new_bundles[nk] = list(dict.fromkeys(new_bundles[nk] + (list(v) if isinstance(v, list) else [])))

    # Add keywords from pathway clusters that match theme for missing/empty categories
    all_cluster_rows = []
    if not pathway_cluster_info.empty:
        all_cluster_rows = list(pathway_cluster_info.iterrows())
    if not target_clusters.empty:
        for _, row in target_clusters.iterrows():
            all_cluster_rows.append((len(all_cluster_rows), row))

    categories_to_fill = set(missing_key) | set(normalize_category(k) for k in empty_bundles)
    for cat_norm in categories_to_fill:
        search_terms = theme_search.get(cat_norm, [cat_norm.replace("_", " ")])
        added_keywords = set()
        for _, row in all_cluster_rows:
            text = terms_and_sample_pathways(row)
            if not text:
                continue
            text_lower = text.lower()
            for st in search_terms:
                if st.lower() in text_lower:
                    # Add significant terms from this cluster (first 10 from top_terms) as keywords
                    terms = parse_terms(row.get("top_terms", ""))
                    for t in terms[:10]:
                        if len(t) >= 3 and t.lower() not in {"the", "and", "for", "pathway", "pathways", "signaling", "signalling", "regulation", "cell", "protein", "receptor", "mediated", "activation", "binding"}:
                            added_keywords.add(t.lower())
                    break
        if added_keywords:
            existing = new_bundles.get(cat_norm, [])
            combined = list(dict.fromkeys(existing + list(added_keywords)))
            new_bundles[cat_norm] = combined
            print(f"   Built/extended bundle '{cat_norm}': {len(combined)} keywords (added {len(added_keywords)} from clusters)")
        elif cat_norm not in new_bundles:
            # No clusters found; use theme search terms as minimal bundle
            new_bundles[cat_norm] = list(search_terms)
            print(f"   Created minimal bundle '{cat_norm}': {new_bundles[cat_norm]}")

    # Ensure every atlas category has a key (even if minimal)
    for c in atlas_categories_norm:
        if c not in new_bundles:
            new_bundles[c] = theme_search.get(c, [c.replace("_", " ")])

    # 5) & 6) Write pathway_bundles.json (keys normalized)
    out_path = bundles_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_bundles, f, indent=2)
    print("5-6) Updated", out_path, "with normalized keys (lowercase, underscore).")

    # 7) Re-run propagation for 50 atlas rows and print new non-zero rate
    sys.path.insert(0, str(REPO_ROOT))
    from src.phase14.loaders import resolve_phase14_inputs, resolved_to_paths, load_phase13_csvs, load_ingredient_compound, load_compound_gene, load_pathway_bundles, load_pathway_cluster_info, load_target_functional_clusters
    from src.phase14.mediation_graph import build_mediation_graph
    from src.phase14.propagation import build_ingredient_genes_from_tables, propagated_scores_for_pairs

    resolved = resolve_phase14_inputs(REPO_ROOT, Path("data/processed/phase13_interactions_v3_20260206_162122_b_gpu_stable"))
    paths = resolved_to_paths(REPO_ROOT, resolved)
    phase13_dir_path = paths.get("phase13_dir") or (REPO_ROOT / "data/processed/phase13_interactions_v3_20260206_162122_b_gpu_stable")
    atlas_full, _, kg_edges, kg_nodes = load_phase13_csvs(phase13_dir_path)
    atlas_50 = atlas_full.head(50)
    ingredient_compound = load_ingredient_compound(paths)
    compound_gene = load_compound_gene(paths, chosen=None, full_run_gate=False)
    # Reload bundles from disk (we just wrote updated bundles)
    pathway_bundles_new = json.loads(out_path.read_text(encoding="utf-8"))
    pathway_cluster_info_df = load_pathway_cluster_info(paths)
    target_clusters_df = load_target_functional_clusters(paths)
    if pathway_cluster_info_df.empty:
        pathway_cluster_info_df = pd.DataFrame(columns=["cluster_id", "top_terms"])
    if target_clusters_df.empty:
        target_clusters_df = pd.DataFrame(columns=["cluster_id", "sample_genes", "top_terms"])

    mediation_nodes, mediation_edges = build_mediation_graph(
        atlas_50, kg_edges, kg_nodes,
        pathway_cluster_info_df, target_clusters_df, pathway_bundles_new,
        ingredient_compound=ingredient_compound, compound_gene=compound_gene,
    )
    ingredient_genes = build_ingredient_genes_from_tables(ingredient_compound, compound_gene)
    prop_scores, prop_diag = propagated_scores_for_pairs(atlas_50, mediation_edges, ingredient_genes=ingredient_genes)

    n_rows = prop_diag.get("n_rows", 0)
    n_nonzero = prop_diag.get("n_nonzero", 0)
    pct = prop_diag.get("pct_rows_with_nonzero_propagation", 0.0)
    print("7) Propagation on 50 atlas rows (after bundle update):")
    print("   n_rows:", n_rows, " n_nonzero:", n_nonzero, " pct_rows_with_nonzero_propagation:", round(pct, 2), "%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
