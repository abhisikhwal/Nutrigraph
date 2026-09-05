"""
Unit tests for Phase14 overlap gate logic.
Gate: PASS if (overlap_vs_cg >= 0.20) OR (n_overlap >= 25) OR (|cg_set| >= 500 and overlap_vs_ic >= 0.01).
"""
from __future__ import annotations

import pytest


@pytest.fixture
def loaders():
    from src.phase14.loaders import compute_identity_overlap_metrics
    return compute_identity_overlap_metrics


def test_small_cg_set_decent_overlap_passes(loaders):
    """Small cg_set (88) with 32 overlap => overlap_vs_cg = 32/88 > 0.20 => PASS."""
    ic_set = {f"IK{i}" for i in range(32)} | {f"I{i}" for i in range(60_000)}
    cg_set = {f"IK{i}" for i in range(88)}  # 88 cg, 32 overlap
    metrics = loaders(ic_set, cg_set)
    assert metrics["n_cg_compounds"] == 88
    assert metrics["n_overlap"] == 32
    assert metrics["overlap_vs_cg"] >= 0.20
    assert abs(metrics["overlap_vs_cg"] - 32 / 88) < 0.001
    assert metrics["gate_passed"] is True
    assert metrics["overlap_vs_cg"] >= 0.20


def test_small_cg_set_high_overlap_vs_cg_passes(loaders):
    """overlap_vs_cg >= 0.20 => PASS."""
    ic_set = {"A", "B", "C", "D", "E"} | {f"X{i}" for i in range(1000)}
    cg_set = {"A", "B", "C", "D", "E"}  # 5 cg, 5 overlap => overlap_vs_cg = 1.0
    metrics = loaders(ic_set, cg_set)
    assert metrics["overlap_vs_cg"] == 1.0
    assert metrics["gate_passed"] is True
    assert "overlap_vs_cg" in metrics["gate_reason"]


def test_tiny_overlap_fails(loaders):
    """Very few overlap, small cg => FAIL."""
    ic_set = {f"I{i}" for i in range(50_000)}
    cg_set = {f"C{i}" for i in range(88)}
    overlap = ic_set & cg_set
    assert len(overlap) == 0
    metrics = loaders(ic_set, cg_set)
    assert metrics["gate_passed"] is False
    assert metrics["n_overlap"] == 0
    assert metrics["overlap_vs_cg"] == 0.0


def test_large_cg_set_alt_rule_passes(loaders):
    """|cg_set| >= 500 and overlap_vs_ic >= 0.01 => PASS (with n_overlap < 25 and overlap_vs_cg < 0.20)."""
    # 20 overlap, 2000 ic => overlap_vs_ic = 0.01; 500 cg => overlap_vs_cg = 20/500 = 0.04 < 0.20
    ic_set = {f"I{i}" for i in range(20)} | {f"X{i}" for i in range(1980)}  # 2000 ic
    cg_set = {f"I{i}" for i in range(20)} | {f"C{i}" for i in range(480)}  # 500 cg, 20 overlap
    metrics = loaders(ic_set, cg_set)
    assert metrics["n_cg_compounds"] == 500
    assert metrics["n_overlap"] == 20
    assert metrics["overlap_vs_cg"] == pytest.approx(20 / 500, rel=1e-5)
    assert metrics["overlap_vs_ic"] == pytest.approx(20 / 2000, rel=1e-5)
    assert metrics["gate_passed"] is True
    assert "500" in metrics["gate_reason"] and "overlap_vs_ic" in metrics["gate_reason"]


def test_n_overlap_25_passes(loaders):
    """n_overlap >= 25 => PASS even if overlap_vs_cg < 0.20."""
    ic_set = {f"I{i}" for i in range(30)} | {f"X{i}" for i in range(50_000)}
    cg_set = {f"I{i}" for i in range(30)}  # 30 cg, 30 overlap
    metrics = loaders(ic_set, cg_set)
    assert metrics["n_overlap"] == 30
    assert metrics["gate_passed"] is True
