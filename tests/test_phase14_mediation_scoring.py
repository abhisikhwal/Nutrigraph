"""
Minimal tests for Phase14 mediation_scoring (compute_mediation_features, add_mechanistic_score).
Ensures no "truth value of a DataFrame is ambiguous" when compound_target_df is None or provided.
"""
from __future__ import annotations

import pytest
import pandas as pd


@pytest.fixture
def minimal_atlas():
    return pd.DataFrame([
        {"ingA_id": "ING_A", "ingB_id": "ING_B", "category": "cat1", "did": 0.1, "q_global": 0.01, "p_analytic": 0.05},
    ])


@pytest.fixture
def minimal_edges():
    return pd.DataFrame([
        {"source_id": "INT_AB", "target_id": "ING_A", "edge_type": "HAS_INGREDIENT"},
        {"source_id": "INT_AB", "target_id": "ING_B", "edge_type": "HAS_INGREDIENT"},
        {"source_id": "INT_AB", "target_id": "CAT_cat1", "edge_type": "AFFECTS"},
        {"source_id": "ING_A", "target_id": "CMP_1", "edge_type": "HAS_COMPOUND"},
        {"source_id": "ING_B", "target_id": "CMP_1", "edge_type": "HAS_COMPOUND"},
        {"source_id": "CMP_1", "target_id": "GENE_X", "edge_type": "TARGETS"},
    ])


@pytest.fixture
def minimal_nodes():
    return pd.DataFrame([
        {"node_id": "ING_A", "label": "Ingredient"},
        {"node_id": "ING_B", "label": "Ingredient"},
        {"node_id": "CMP_1", "label": "Compound"},
        {"node_id": "GENE_X", "label": "Gene"},
        {"node_id": "CAT_cat1", "label": "Category"},
    ])


def test_compute_mediation_features_compound_target_none(minimal_atlas, minimal_edges, minimal_nodes):
    """compound_target_df=None must not raise (no truth value of DataFrame)."""
    from src.phase14.mediation_scoring import compute_mediation_features
    out = compute_mediation_features(minimal_atlas, minimal_edges, minimal_nodes, compound_target_df=None)
    assert out is not None and isinstance(out, pd.DataFrame)
    assert len(out) == 1
    assert "shared_compounds_count" in out.columns
    assert "shared_genes_count" in out.columns


def test_compute_mediation_features_compound_target_empty(minimal_atlas, minimal_edges, minimal_nodes):
    """compound_target_df=empty DataFrame must not raise."""
    from src.phase14.mediation_scoring import compute_mediation_features
    empty_df = pd.DataFrame(columns=["compound_id", "target_name"])
    out = compute_mediation_features(minimal_atlas, minimal_edges, minimal_nodes, compound_target_df=empty_df)
    assert out is not None and len(out) == 1


def test_compute_mediation_features_compound_target_provided(minimal_atlas, minimal_edges, minimal_nodes):
    """compound_target_df with rows must add evidence_target columns when compounds overlap."""
    from src.phase14.mediation_scoring import compute_mediation_features
    ct_df = pd.DataFrame([
        {"compound_id": "CMP_1", "target_name": "T1"},
        {"compound_id": "CMP_1", "target_name": "T2"},
    ])
    out = compute_mediation_features(minimal_atlas, minimal_edges, minimal_nodes, compound_target_df=ct_df)
    assert out is not None and len(out) == 1
    assert "evidence_target_count" in out.columns
    assert "evidence_target_overlap" in out.columns
    assert out["evidence_target_count"].iloc[0] == 2  # CMP_1 has 2 targets, shared by both ings
    assert out["evidence_target_overlap"].iloc[0] == 2
