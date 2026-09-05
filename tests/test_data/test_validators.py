"""
Unit tests for data validation utilities.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.validators import SchemaValidator


@pytest.fixture
def sample_ingredient_df():
    """Sample ingredient DataFrame for testing."""
    return pd.DataFrame({
        'ingredient_id': ['ING_00001', 'ING_00002'],
        'canonical_name': ['turmeric', 'black pepper'],
        'scientific_name': ['Curcuma longa', 'Piper nigrum'],
        'category': ['spice', 'spice'],
        'is_plant_based': [True, True],
        'data_sources': ['wikidata', 'wikidata']
    })


@pytest.fixture
def validator():
    """SchemaValidator instance."""
    return SchemaValidator(schema_dir=Path("schemas"))


def test_validator_initialization(validator):
    """Test that validator loads schemas."""
    assert len(validator.schemas) > 0
    assert 'ingredient_schema' in validator.schemas


def test_validate_valid_dataframe(validator, sample_ingredient_df):
    """Test validation of valid DataFrame."""
    result = validator.validate_dataframe(
        sample_ingredient_df,
        'ingredient_schema'
    )
    
    assert result['valid'] is True
    assert len(result['errors']) == 0


def test_validate_missing_column(validator):
    """Test validation fails with missing required column."""
    invalid_df = pd.DataFrame({
        'ingredient_id': ['ING_00001'],
        'canonical_name': ['turmeric']
        # Missing 'category' (required)
    })
    
    result = validator.validate_dataframe(
        invalid_df,
        'ingredient_schema'
    )
    
    assert result['valid'] is False
    assert len(result['errors']) > 0


def test_validate_invalid_id_format(validator):
    """Test validation fails with invalid ID format."""
    invalid_df = pd.DataFrame({
        'ingredient_id': ['INVALID_ID'],  # Wrong format
        'canonical_name': ['turmeric'],
        'category': ['spice']
    })
    
    result = validator.validate_dataframe(
        invalid_df,
        'ingredient_schema'
    )
    
    # Should fail due to pattern mismatch
    assert result['valid'] is False


def test_check_id_format(validator, sample_ingredient_df):
    """Test ID format checking."""
    result = validator.check_id_format(
        sample_ingredient_df,
        'ingredient_id',
        r'^ING_\d{5}$'
    )
    
    assert result['valid'] is True
    assert len(result['invalid_ids']) == 0


def test_check_id_format_invalid(validator):
    """Test ID format checking with invalid IDs."""
    invalid_df = pd.DataFrame({
        'ingredient_id': ['ING_00001', 'WRONG_FORMAT', 'ING_999']
    })
    
    result = validator.check_id_format(
        invalid_df,
        'ingredient_id',
        r'^ING_\d{5}$'
    )
    
    assert result['valid'] is False
    assert len(result['invalid_ids']) == 2
