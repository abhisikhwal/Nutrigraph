"""
Unit tests for pathway enrichment analysis.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.pathway_predictor import PathwayPredictor


@pytest.fixture
def predictor():
    """Create PathwayPredictor instance."""
    return PathwayPredictor()


@pytest.fixture
def sample_target_pathway_map():
    """Sample target-pathway mapping."""
    return pd.DataFrame({
        'uniprot_accession': ['P35354', 'P35354', 'P12345', 'P12345', 'Q67890'],
        'pathway_id': ['R-HSA-001', 'R-HSA-002', 'R-HSA-001', 'R-HSA-003', 'R-HSA-002'],
        'pathway_name': ['Pathway A', 'Pathway B', 'Pathway A', 'Pathway C', 'Pathway B']
    })


def test_enrich_pathways(predictor, sample_target_pathway_map):
    """Test pathway enrichment."""
    target_list = ['P35354', 'P12345']
    
    enrichment = predictor.enrich_pathways(
        target_list,
        sample_target_pathway_map,
        background_size=100
    )
    
    assert isinstance(enrichment, pd.DataFrame)
    assert len(enrichment) > 0
    assert 'pathway_id' in enrichment.columns
    assert 'pvalue' in enrichment.columns
    assert 'fold_enrichment' in enrichment.columns


def test_enrichment_statistics(predictor, sample_target_pathway_map):
    """Test enrichment statistics calculation."""
    target_list = ['P35354']
    
    enrichment = predictor.enrich_pathways(
        target_list,
        sample_target_pathway_map,
        background_size=100
    )
    
    # Check that p-values are valid
    assert (enrichment['pvalue'] >= 0).all()
    assert (enrichment['pvalue'] <= 1).all()
    
    # Check that fold enrichment is positive
    assert (enrichment['fold_enrichment'] > 0).all()


def test_no_enrichment(predictor, sample_target_pathway_map):
    """Test when no targets match."""
    target_list = ['NONEXISTENT_TARGET']
    
    enrichment = predictor.enrich_pathways(
        target_list,
        sample_target_pathway_map,
        background_size=100
    )
    
    assert len(enrichment) == 0
