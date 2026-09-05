"""
Predict affected biological pathways from compound-target interactions.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PathwayPredictor:
    """
    Predict enriched pathways from a set of compounds or targets.
    
    Uses enrichment analysis (Fisher's exact test, hypergeometric test).
    """
    
    def __init__(self):
        logger.info("Initialized PathwayPredictor")
    
    def enrich_pathways(
        self,
        target_list: List[str],
        target_pathway_map: pd.DataFrame,
        background_size: Optional[int] = None,
        method: str = "fisher"
    ) -> pd.DataFrame:
        """
        Perform pathway enrichment analysis.
        
        Args:
            target_list: List of UniProt IDs
            target_pathway_map: Mapping of targets to pathways
            background_size: Total number of targets (for null model)
            method: Enrichment method ('fisher', 'hypergeometric')
            
        Returns:
            DataFrame with pathways and enrichment statistics
        """
        logger.info(f"Running pathway enrichment for {len(target_list)} targets")
        
        target_set = set(target_list)
        
        # Count targets per pathway
        pathway_targets = {}
        for pathway_id, group in target_pathway_map.groupby('pathway_id'):
            pathway_set = set(group['uniprot_accession'])
            overlap = target_set & pathway_set
            
            if len(overlap) > 0:
                pathway_targets[pathway_id] = {
                    'pathway_name': group.iloc[0].get('pathway_name', ''),
                    'n_overlap': len(overlap),
                    'n_pathway': len(pathway_set),
                    'targets': list(overlap)
                }
        
        # Calculate enrichment p-values
        if background_size is None:
            background_size = target_pathway_map['uniprot_accession'].nunique()
        
        query_size = len(target_set)
        
        results = []
        for pathway_id, data in pathway_targets.items():
            n_overlap = data['n_overlap']
            n_pathway = data['n_pathway']
            
            # Fisher's exact test (one-tailed)
            if method == "fisher":
                from scipy.stats import fisher_exact
                
                # Contingency table:
                # | In query | Not in query |
                # |----------|--------------|
                # | a (overlap) | b (pathway - overlap) |
                # | c (query - overlap) | d (background - query - pathway + overlap) |
                
                a = n_overlap
                b = n_pathway - n_overlap
                c = query_size - n_overlap
                d = background_size - query_size - n_pathway + n_overlap
                
                oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative='greater')
                
            else:
                # Hypergeometric test
                from scipy.stats import hypergeom
                
                pvalue = hypergeom.sf(
                    n_overlap - 1,
                    background_size,
                    n_pathway,
                    query_size
                )
            
            results.append({
                'pathway_id': pathway_id,
                'pathway_name': data['pathway_name'],
                'n_overlap': n_overlap,
                'n_pathway': n_pathway,
                'n_query': query_size,
                'fold_enrichment': (n_overlap / n_pathway) / (query_size / background_size),
                'pvalue': pvalue,
                'targets': ','.join(data['targets'])
            })
        
        results_df = pd.DataFrame(results)
        
        # FDR correction (Benjamini-Hochberg)
        if len(results_df) > 0:
            from scipy.stats import false_discovery_control
            results_df['fdr'] = false_discovery_control(results_df['pvalue'])
            results_df = results_df.sort_values('pvalue')
        
        logger.info(f"Found {len(results_df)} enriched pathways")
        
        return results_df
    
    def map_compounds_to_pathways(
        self,
        compound_ids: List[int],
        compound_target_map: pd.DataFrame,
        target_pathway_map: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Map compounds to pathways via targets.
        
        Args:
            compound_ids: List of PubChem CIDs
            compound_target_map: Compound-target interactions
            target_pathway_map: Target-pathway annotations
            
        Returns:
            DataFrame with pathway enrichment results
        """
        logger.info(f"Mapping {len(compound_ids)} compounds to pathways")
        
        # Get targets for compounds
        targets = compound_target_map[
            compound_target_map['pubchem_cid'].isin(compound_ids)
        ]['uniprot_accession'].unique()
        
        logger.info(f"Found {len(targets)} targets")
        
        # Enrich pathways
        enrichment = self.enrich_pathways(
            list(targets),
            target_pathway_map
        )
        
        return enrichment
