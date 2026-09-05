"""
Ontology and name matching utilities for ingredient synonym resolution.
"""

import logging
import pandas as pd
import re
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class NameMatcher:
    """
    Match ingredient names across datasets with synonym handling.
    
    Methods:
    - Exact matching
    - Fuzzy matching (Levenshtein distance)
    - Phonetic matching (optional)
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: Minimum similarity for fuzzy matches (0-1)
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized NameMatcher (threshold={similarity_threshold})")
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize ingredient name for matching.
        
        Steps:
        - Lowercase
        - Remove punctuation
        - Strip whitespace
        - Remove common stopwords (e.g., "fresh", "dried")
        
        Args:
            name: Raw ingredient name
            
        Returns:
            Normalized name
        """
        if pd.isna(name):
            return ""
        
        # Lowercase
        name = str(name).lower()
        
        # Remove punctuation
        name = re.sub(r'[^\w\s]', ' ', name)
        
        # Remove stopwords
        stopwords = {'fresh', 'dried', 'raw', 'cooked', 'organic', 'whole'}
        words = name.split()
        words = [w for w in words if w not in stopwords]
        name = ' '.join(words)
        
        # Strip and normalize whitespace
        name = ' '.join(name.split())
        
        return name
    
    def exact_match(
        self,
        query_name: str,
        candidate_names: List[str]
    ) -> Optional[str]:
        """
        Find exact match in candidate names.
        
        Args:
            query_name: Name to match
            candidate_names: List of candidate names
            
        Returns:
            Matching candidate name, or None
        """
        query_norm = self.normalize_name(query_name)
        
        for candidate in candidate_names:
            candidate_norm = self.normalize_name(candidate)
            if query_norm == candidate_norm:
                return candidate
        
        return None
    
    def fuzzy_match(
        self,
        query_name: str,
        candidate_names: List[str],
        return_score: bool = False
    ) -> Optional[Tuple[str, float]]:
        """
        Find best fuzzy match using Levenshtein distance.
        
        Args:
            query_name: Name to match
            candidate_names: List of candidate names
            return_score: If True, return (match, score) tuple
            
        Returns:
            Best matching candidate (and score if requested), or None
        """
        query_norm = self.normalize_name(query_name)
        
        if not query_norm:
            return None
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidate_names:
            candidate_norm = self.normalize_name(candidate)
            
            if not candidate_norm:
                continue
            
            # Calculate similarity ratio
            score = SequenceMatcher(None, query_norm, candidate_norm).ratio()
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        if best_score >= self.similarity_threshold:
            if return_score:
                return (best_match, best_score)
            else:
                return best_match
        
        return None
    
    def match_with_synonyms(
        self,
        query_name: str,
        reference_df: pd.DataFrame,
        name_col: str = 'canonical_name',
        synonym_col: str = 'synonyms'
    ) -> Optional[str]:
        """
        Match ingredient considering synonyms.
        
        Args:
            query_name: Name to match
            reference_df: Reference DataFrame with canonical names and synonyms
            name_col: Column with canonical names
            synonym_col: Column with synonym lists (can be comma-separated string or list)
            
        Returns:
            Matched canonical name, or None
        """
        # Try exact match with canonical names
        match = self.exact_match(query_name, reference_df[name_col].tolist())
        if match:
            return match
        
        # Try exact match with synonyms
        for _, row in reference_df.iterrows():
            canonical = row[name_col]
            synonyms = row.get(synonym_col, [])
            
            if isinstance(synonyms, str):
                synonyms = [s.strip() for s in synonyms.split(',')]
            elif pd.isna(synonyms):
                synonyms = []
            
            if self.exact_match(query_name, synonyms):
                return canonical
        
        # Try fuzzy match
        all_names = reference_df[name_col].tolist()
        match = self.fuzzy_match(query_name, all_names)
        
        if match:
            logger.debug(f"Fuzzy matched '{query_name}' to '{match}'")
            return match
        
        logger.debug(f"No match found for '{query_name}'")
        return None
    
    def batch_match(
        self,
        query_names: List[str],
        reference_df: pd.DataFrame,
        name_col: str = 'canonical_name',
        synonym_col: str = 'synonyms'
    ) -> pd.DataFrame:
        """
        Match a list of ingredient names.
        
        Args:
            query_names: List of names to match
            reference_df: Reference DataFrame
            name_col: Canonical name column
            synonym_col: Synonym column
            
        Returns:
            DataFrame with query names and matched canonical names
        """
        logger.info(f"Batch matching {len(query_names)} names...")
        
        results = []
        for query in query_names:
            match = self.match_with_synonyms(
                query,
                reference_df,
                name_col,
                synonym_col
            )
            results.append({
                'query_name': query,
                'matched_name': match,
                'match_found': match is not None
            })
        
        results_df = pd.DataFrame(results)
        
        n_matched = results_df['match_found'].sum()
        match_rate = n_matched / len(query_names) * 100
        
        logger.info(f"Matched {n_matched}/{len(query_names)} ({match_rate:.1f}%)")
        
        return results_df
