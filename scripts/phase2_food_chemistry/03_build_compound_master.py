#!/usr/bin/env python3
"""
Build canonical compound master table with PubChem standardization.

This integrates:
1. FooDB compounds (if available)
2. PubChem API for standardization (CID, InChIKey, SMILES)
3. Compound classification (polyphenols, terpenes, alkaloids, etc.)

Output:
- data/processed/canonical/compounds.parquet

Schema:
- compound_id: Primary key (PubChem CID preferred)
- inchikey: InChIKey (structural hash)
- smiles: Canonical SMILES
- name: Common name
- cas_number: CAS registry number
- compound_class: Polyphenol/Terpene/Alkaloid/etc.
- sources: Origin databases (foodb, pubchem, etc.)
"""

import pandas as pd
import logging
import yaml
from pathlib import Path
import requests
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

INTERIM_DIR = Path(paths["interim_data"])
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"
CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def get_pubchem_cid_from_name(name):
    """Resolve compound name to PubChem CID."""
    if pd.isna(name) or not name:
        return None
    
    url = f"{PUBCHEM_API}/compound/name/{name}/cids/JSON"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            cids = data.get('IdentifierList', {}).get('CID', [])
            return cids[0] if cids else None
    except:
        pass
    
    return None

def get_pubchem_details(cid):
    """Get compound details from PubChem CID."""
    url = f"{PUBCHEM_API}/compound/cid/{cid}/property/CanonicalSMILES,InChIKey,MolecularFormula/JSON"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
            return props
    except:
        pass
    
    return {}

def classify_compound(name, smiles=None):
    """Simple rule-based classification."""
    if not name:
        return 'unknown'
    
    name_lower = name.lower()
    
    # Polyphenols
    if any(x in name_lower for x in ['flavonoid', 'catechin', 'quercetin', 'resveratrol', 'anthocyanin']):
        return 'polyphenol'
    
    # Terpenes
    if any(x in name_lower for x in ['terpene', 'limonene', 'pinene', 'menthol']):
        return 'terpene'
    
    # Alkaloids
    if any(x in name_lower for x in ['alkaloid', 'caffeine', 'nicotine', 'morphine', 'piperine']):
        return 'alkaloid'
    
    # Capsaicinoids (spice-specific)
    if 'capsaicin' in name_lower:
        return 'capsaicinoid'
    
    # Curcuminoids (turmeric)
    if 'curcumin' in name_lower:
        return 'curcuminoid'
    
    return 'other'

def create_demo_compounds():
    """Create demo compound data for common spice compounds."""
    logger.info("Creating demo compound data (using well-known spice compounds)...")
    
    # Key bioactive compounds in common spices
    demo_compounds = [
        {'name': 'Curcumin', 'pubchem_cid': 969516, 'compound_class': 'curcuminoid'},
        {'name': 'Piperine', 'pubchem_cid': 638024, 'compound_class': 'alkaloid'},
        {'name': 'Cinnamaldehyde', 'pubchem_cid': 637511, 'compound_class': 'aromatic_aldehyde'},
        {'name': 'Capsaicin', 'pubchem_cid': 1548943, 'compound_class': 'capsaicinoid'},
        {'name': 'Gingerol', 'pubchem_cid': 442793, 'compound_class': 'phenol'},
        {'name': 'Eugenol', 'pubchem_cid': 3314, 'compound_class': 'phenylpropanoid'},
        {'name': 'Allicin', 'pubchem_cid': 65036, 'compound_class': 'organosulfur'},
        {'name': 'Quercetin', 'pubchem_cid': 5280343, 'compound_class': 'polyphenol'},
    ]
    
    compounds = []
    for comp in demo_compounds:
        logger.info(f"Fetching details for {comp['name']} (CID: {comp['pubchem_cid']})...")
        pubchem_data = get_pubchem_details(comp['pubchem_cid'])
        time.sleep(0.2)  # Rate limiting
        
        compounds.append({
            'compound_id': comp['pubchem_cid'],
            'pubchem_cid': comp['pubchem_cid'],
            'inchikey': pubchem_data.get('InChIKey'),
            'smiles': pubchem_data.get('CanonicalSMILES'),
            'name': comp['name'],
            'molecular_formula': pubchem_data.get('MolecularFormula'),
            'compound_class': comp['compound_class'],
            'source': 'pubchem'
        })
    
    return pd.DataFrame(compounds)

def build_compound_master():
    """Build canonical compound master from FooDB + PubChem."""
    logger.info("Building compound master...")
    
    # Load FooDB compounds if available
    foodb_compounds_path = INTERIM_DIR / "foodb" / "compounds.parquet"
    
    if foodb_compounds_path.exists():
        logger.info("Loading FooDB compounds...")
        foodb = pd.read_parquet(foodb_compounds_path)
        logger.info(f"Loaded {len(foodb)} FooDB compounds")
        
        # Process FooDB compounds (limit for demo)
        compounds = []
        logger.info("Standardizing compounds with PubChem (limiting to 20 for demo)...")
        
        for idx, row in tqdm(foodb.head(20).iterrows(), total=20, desc="Processing"):
            # Try to get PubChem CID by name
            cid = get_pubchem_cid_from_name(row.get('compound_name'))
            time.sleep(0.2)  # Rate limiting
            
            if cid:
                pubchem_data = get_pubchem_details(cid)
                time.sleep(0.2)
            else:
                pubchem_data = {}
            
            compound = {
                'compound_id': cid or f"FDB_{row.get('foodb_compound_id', idx)}",
                'pubchem_cid': cid,
                'inchikey': pubchem_data.get('InChIKey') or row.get('inchikey'),
                'smiles': pubchem_data.get('CanonicalSMILES') or row.get('smiles'),
                'name': row.get('compound_name'),
                'cas_number': row.get('cas_number'),
                'molecular_formula': pubchem_data.get('MolecularFormula'),
                'compound_class': classify_compound(row.get('compound_name')),
                'source': 'foodb,pubchem' if cid else 'foodb'
            }
            
            compounds.append(compound)
        
        df = pd.DataFrame(compounds)
    else:
        logger.warning("FooDB compounds not found - using demo PubChem compounds")
        df = create_demo_compounds()
    
    # Remove duplicates (prefer PubChem-standardized entries)
    df = df.sort_values('pubchem_cid', na_position='last')
    df = df.drop_duplicates(subset=['inchikey'], keep='first')
    
    # Save
    output_path = CANONICAL_DIR / "compounds.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved {len(df)} compounds to {output_path}")
    
    # Summary
    logger.info(f"\nCompound classes:")
    logger.info(df['compound_class'].value_counts())
    
    logger.info(f"\nSample compounds:")
    logger.info(df[['name', 'pubchem_cid', 'compound_class']].head(10))

if __name__ == "__main__":
    build_compound_master()
