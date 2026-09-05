"""
Phase15 demo UI: Ingredient pair explorer.
Dropdown Ingredient A/B, mechanistic_score, causal mediation, top genes, mini graph (pyvis), export JSON.
Run: streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

# Default Phase14 snapshot
PHASE14_SNAPSHOT = REPO_ROOT / "data" / "processed" / "milestones" / "phase14" / "v1_working_2026-02-19" / "phase14_20260219_204918"
PHASE14_RUN = REPO_ROOT / "data" / "processed" / "phase14_mediation" / "phase14_20260219_204918"


def get_run_dir() -> Path:
    if PHASE14_SNAPSHOT.exists():
        return PHASE14_SNAPSHOT
    return PHASE14_RUN


def load_pair_mediation(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "pair_category_mediation.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_ingredient_list(run_dir: Path) -> list:
    pm = load_pair_mediation(run_dir)
    if pm.empty:
        return ["ING_000026", "ING_000044"]
    ings = set(pm["ingA_id"].dropna().astype(str)) | set(pm["ingB_id"].dropna().astype(str))
    return sorted(ings)[:500]


def get_pair_data(pm: pd.DataFrame, ing_a: str, ing_b: str) -> pd.DataFrame:
    sub = pm[((pm["ingA_id"] == ing_a) & (pm["ingB_id"] == ing_b)) | ((pm["ingA_id"] == ing_b) & (pm["ingB_id"] == ing_a))]
    return sub


def build_evidence_json(ing_a: str, ing_b: str, sub: pd.DataFrame) -> dict:
    cats = sub["category"].tolist() if "category" in sub.columns else []
    scores = sub.set_index("category")["mechanistic_score"].to_dict() if "mechanistic_score" in sub.columns else {}
    return {
        "ing_a": ing_a,
        "ing_b": ing_b,
        "categories": cats,
        "mechanistic_scores": scores,
        "shared_compounds_count": int(sub["shared_compounds_count"].iloc[0]) if "shared_compounds_count" in sub.columns and len(sub) else 0,
        "shared_genes_count": int(sub["shared_genes_count"].iloc[0]) if "shared_genes_count" in sub.columns and len(sub) else 0,
    }


def main():
    st.set_page_config(page_title="Phase15 Pair Explorer", layout="wide")
    st.title("Phase15: Ingredient pair explorer")
    run_dir = get_run_dir()
    if not run_dir.exists():
        st.error(f"Phase14 data not found. Expected {run_dir}")
        return
    pm = load_pair_mediation(run_dir)
    if pm.empty:
        st.warning("No pair_category_mediation.csv found.")
        return
    ingredients = load_ingredient_list(run_dir)
    ing_a = st.selectbox("Ingredient A", ingredients, index=min(0, len(ingredients) - 1))
    ing_b = st.selectbox("Ingredient B", ingredients, index=min(1, len(ingredients) - 1))
    sub = get_pair_data(pm, ing_a, ing_b)
    if sub.empty:
        st.info("No mediation data for this pair.")
        evidence = build_evidence_json(ing_a, ing_b, pd.DataFrame())
    else:
        st.subheader("Mechanistic score by category")
        cols = [c for c in ["category", "mechanistic_score", "shared_compounds_count", "shared_genes_count", "propagated_pathway_score", "did"] if c in sub.columns]
        st.dataframe(sub[cols].head(20), use_container_width=True)
        st.metric("Mean mechanistic score", f"{sub['mechanistic_score'].mean():.3f}")
        evidence = build_evidence_json(ing_a, ing_b, sub)
    st.subheader("Evidence summary")
    st.json(evidence)
    # Mini graph: optional pyvis
    try:
        from pyvis.network import Network
        net = Network(height="300px", directed=True)
        net.add_node(ing_a, label=ing_a, title=ing_a)
        net.add_node(ing_b, label=ing_b, title=ing_b)
        for cat in evidence.get("categories", [])[:5]:
            net.add_node(cat, label=cat, title=cat)
            net.add_edge(ing_a, cat, value=evidence.get("mechanistic_scores", {}).get(cat, 0))
            net.add_edge(ing_b, cat, value=evidence.get("mechanistic_scores", {}).get(cat, 0))
        html = net.generate_html()
        st.components.v1.html(html, height=350)
    except ImportError:
        st.caption("Install pyvis for graph: pip install pyvis")
    # Export
    st.download_button(
        "Download evidence JSON",
        data=json.dumps(evidence, indent=2),
        file_name=f"evidence_{ing_a}_{ing_b}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
