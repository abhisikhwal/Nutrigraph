"""
Phase14: Minimal shared utilities — timer and time budget for bounded pipelines.
"""
from __future__ import annotations

import time
from typing import Optional


def timer() -> float:
    """Return current monotonic time in seconds (for elapsed delta)."""
    return time.perf_counter()


def elapsed_since(t0: float) -> float:
    """Return seconds since t0 (from timer())."""
    return time.perf_counter() - t0


def within_time_budget(t0: float, budget_seconds: float) -> bool:
    """True if elapsed since t0 is still within budget_seconds."""
    return budget_seconds <= 0 or elapsed_since(t0) < budget_seconds
