"""Recipe dose layer: quantity parse → mass → nutrient RDI + relative mechanism contribution."""

from .dose_engine import DoseEngine, RecipeDoseResult

__all__ = ["DoseEngine", "RecipeDoseResult"]
