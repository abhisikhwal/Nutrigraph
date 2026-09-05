"""
Generate molecular fingerprints for compounds (Morgan, MACCS, etc.).
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not installed. Fingerprint generation unavailable.")
    RDKIT_AVAILABLE = False


class MorganFingerprintGenerator:
    """
    Generate Morgan (circular) fingerprints for compounds.
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        """
        Args:
            radius: Morgan fingerprint radius (typically 2 or 3)
            n_bits: Number of bits in fingerprint
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for fingerprint generation")
        
        self.radius = radius
        self.n_bits = n_bits
        logger.info(f"Initialized Morgan fingerprints (radius={radius}, bits={n_bits})")
    
    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert SMILES to Morgan fingerprint.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Binary fingerprint as numpy array, or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.radius,
                nBits=self.n_bits
            )
            
            return np.array(fp)
        
        except Exception as e:
            logger.debug(f"Failed to generate fingerprint for {smiles}: {e}")
            return None
    
    def generate_fingerprints(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'canonical_smiles',
        output_col: str = 'morgan_fp'
    ) -> pd.DataFrame:
        """
        Generate fingerprints for all compounds in DataFrame.
        
        Args:
            df: DataFrame with SMILES column
            smiles_col: Name of SMILES column
            output_col: Name for output fingerprint column
            
        Returns:
            DataFrame with fingerprint column added
        """
        logger.info(f"Generating fingerprints for {len(df)} compounds...")
        
        df = df.copy()
        df[output_col] = df[smiles_col].apply(self.smiles_to_fingerprint)
        
        # Count failures
        n_failed = df[output_col].isna().sum()
        if n_failed > 0:
            logger.warning(f"{n_failed} compounds failed fingerprint generation")
        
        logger.info(f"Generated {len(df) - n_failed} fingerprints successfully")
        return df
    
    def compute_tanimoto_similarity(
        self,
        fp1: np.ndarray,
        fp2: np.ndarray
    ) -> float:
        """
        Calculate Tanimoto similarity between two fingerprints.
        
        Args:
            fp1, fp2: Binary fingerprints
            
        Returns:
            Tanimoto coefficient (0-1)
        """
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_similar_compounds(
        self,
        query_fp: np.ndarray,
        library_fps: pd.DataFrame,
        fp_col: str = 'morgan_fp',
        top_k: int = 10,
        min_similarity: float = 0.7
    ) -> pd.DataFrame:
        """
        Find similar compounds in a library.
        
        Args:
            query_fp: Query fingerprint
            library_fps: DataFrame with fingerprint column
            fp_col: Fingerprint column name
            top_k: Return top K matches
            min_similarity: Minimum Tanimoto threshold
            
        Returns:
            DataFrame with top matches and similarity scores
        """
        logger.info(f"Searching for similar compounds (threshold={min_similarity})")
        
        similarities = library_fps[fp_col].apply(
            lambda x: self.compute_tanimoto_similarity(query_fp, x) if x is not None else 0.0
        )
        
        library_fps = library_fps.copy()
        library_fps['tanimoto_similarity'] = similarities
        
        results = library_fps[
            library_fps['tanimoto_similarity'] >= min_similarity
        ].nlargest(top_k, 'tanimoto_similarity')
        
        logger.info(f"Found {len(results)} similar compounds")
        return results
