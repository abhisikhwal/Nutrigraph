"""
Reprocess FooDB completely without demo limits.

This fixes two issues from the original Phase 2:
1. Only 20 compounds were processed (had demo limit)
2. SMILES column contains InChIKeys (parsing error)

Output:
- data/processed/canonical/compounds.parquet (FULL 70K compounds)
"""

import pandas as pd
import logging
import yaml
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = Path(paths["raw_data"]) / "foodb"
INTERIM_DIR = Path(paths["interim_data"]) / "foodb"
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"

def reprocess_foodb_compounds():
    """
    Reprocess FooDB compounds correctly.
    """
    logger.info("Loading FooDB compounds...")
    
    # Load FooDB compound table
    foodb_compounds_path = INTERIM_DIR / "compounds.parquet"
    
    if not foodb_compounds_path.exists():
        logger.error(f"FooDB compounds not found: {foodb_compounds_path}")
        logger.info("Please re-run Phase 2 scripts 01-02 first")
        return
    
    foodb = pd.read_parquet(foodb_compounds_path)
    logger.info(f"Loaded {len(foodb):,} FooDB compounds")
    
    # Build compound master WITHOUT PubChem API calls
    logger.info("Building compound master (no API calls)...")
    
    compounds = []
    
    for idx, row in tqdm(foodb.iterrows(), total=len(foodb), desc="Processing"):
        
        # Use FooDB data directly
        compound = {
            'compound_id': row.get('foodb_public_id') or f"FDB_{row.get('foodb_compound_id', idx)}",
            'pubchem_cid': None,  # We'll skip PubChem for now
            'inchikey': row.get('inchikey'),
            'smiles': row.get('smiles'),  # Actual SMILES, not InChIKey
            'name': row.get('compound_name'),
            'cas_number': row.get('cas_number'),
            'molecular_formula': None,  # Would need to calculate
            'compound_class': classify_compound(row.get('compound_name')),
            'source': 'foodb'
        }
        
        compounds.append(compound)
    
    df = pd.DataFrame(compounds)
    
    # Clean up - remove compounds with no structure
    df = df[df['inchikey'].notna() | df['smiles'].notna()]
    
    logger.info(f"Compounds with structure data: {len(df):,}")
    
    # Save
    output_path = CANONICAL_DIR / "compounds.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved {len(df):,} compounds to {output_path}")
    
    # Summary
    logger.info(f"\nCompound classes:")
    logger.info(df['compound_class'].value_counts())
    
    logger.info(f"\nData completeness:")
    logger.info(f"  With InChIKey: {df['inchikey'].notna().sum():,}")
    logger.info(f"  With SMILES: {df['smiles'].notna().sum():,}")
    logger.info(f"  With name: {df['name'].notna().sum():,}")

def classify_compound(name):
    """Simple rule-based classification."""
    if not name:
        return 'unknown'
    
    name_lower = name.lower()
    
    # Polyphenols
    if any(x in name_lower for x in ['flavonoid', 'catechin', 'quercetin', 'resveratrol', 'anthocyanin', 'epicatechin']):
        return 'polyphenol'
    
    # Terpenes
    if any(x in name_lower for x in ['terpene', 'limonene', 'pinene', 'menthol', 'camphor']):
        return 'terpene'
    
    # Alkaloids
    if any(x in name_lower for x in ['alkaloid', 'caffeine', 'nicotine', 'morphine', 'piperine', 'theobromine']):
        return 'alkaloid'
    
    # Capsaicinoids
    if 'capsaicin' in name_lower:
        return 'capsaicinoid'
    
    # Curcuminoids
    if 'curcumin' in name_lower:
        return 'curcuminoid'
    
    # Gingerols
    if any(x in name_lower for x in ['gingerol', 'shogaol', 'paradol']):
        return 'gingerol'
    
    return 'other'

if __name__ == "__main__":
    reprocess_foodb_compounds()
