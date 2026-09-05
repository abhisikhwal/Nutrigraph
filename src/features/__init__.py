"""
Feature engineering module: molecular fingerprints and related helpers.
"""

from .compound_fingerprints import MorganFingerprintGenerator

__all__ = [
    "MorganFingerprintGenerator",
]
