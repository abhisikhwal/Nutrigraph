"""
Unit tests for compound fingerprint generation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from features.compound_fingerprints import MorganFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestMorganFingerprintGenerator:
    """Tests for MorganFingerprintGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create fingerprint generator."""
        return MorganFingerprintGenerator(radius=2, n_bits=2048)
    
    @pytest.fixture
    def sample_compounds(self):
        """Sample compound DataFrame."""
        return pd.DataFrame({
            'pubchem_cid': [969516, 5280343],
            'canonical_smiles': [
                'COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O',  # Curcumin
                'CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(CCCC2(C)C)C)C)C'  # Beta-carotene
            ]
        })
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.radius == 2
        assert generator.n_bits == 2048
    
    def test_smiles_to_fingerprint(self, generator):
        """Test SMILES to fingerprint conversion."""
        smiles = 'COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O'
        fp = generator.smiles_to_fingerprint(smiles)
        
        assert fp is not None
        assert len(fp) == 2048
        assert fp.dtype == bool or fp.dtype == np.uint8
    
    def test_invalid_smiles(self, generator):
        """Test handling of invalid SMILES."""
        invalid_smiles = 'INVALID_SMILES_STRING'
        fp = generator.smiles_to_fingerprint(invalid_smiles)
        
        assert fp is None
    
    def test_generate_fingerprints(self, generator, sample_compounds):
        """Test batch fingerprint generation."""
        result = generator.generate_fingerprints(sample_compounds)
        
        assert 'morgan_fp' in result.columns
        assert result['morgan_fp'].notna().all()
    
    def test_tanimoto_similarity(self, generator):
        """Test Tanimoto similarity calculation."""
        fp1 = np.array([1, 1, 0, 1, 0])
        fp2 = np.array([1, 0, 1, 1, 0])
        
        similarity = generator.compute_tanimoto_similarity(fp1, fp2)
        
        assert 0 <= similarity <= 1
        assert similarity == 2/4  # 2 bits in common, 4 bits total
