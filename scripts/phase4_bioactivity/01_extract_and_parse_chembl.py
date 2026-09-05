#!/usr/bin/env python3
"""
Extract and parse pre-downloaded ChEMBL database.

User has already downloaded: data/raw/chembl/chembl_36_sqlite.tar.gz

This script:
1. Checks if file exists
2. Extracts the SQLite database
3. Parses to Parquet files

Output:
- data/interim/chembl/compounds.parquet
- data/interim/chembl/targets.parquet
- data/interim/chembl/activities.parquet
"""

import tarfile
import sqlite3
import pandas as pd
import logging
import yaml
import sys
import io
from pathlib import Path
from tqdm import tqdm

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = Path(paths["raw_data"]) / "chembl"
INTERIM_DIR = Path(paths["interim_data"]) / "chembl"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

CHEMBL_VERSION = "36"

def check_chembl_archive():
    """Check if ChEMBL archive exists."""
    archive_file = RAW_DIR / f"chembl_{CHEMBL_VERSION}_sqlite.tar.gz"
    
    if not archive_file.exists():
        logger.error(f"ChEMBL archive not found: {archive_file}")
        logger.error("Please download from: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/")
        logger.error(f"And place it in: {RAW_DIR}")
        raise FileNotFoundError(f"Missing {archive_file}")
    
    logger.info(f"✅ Found ChEMBL archive: {archive_file}")
    logger.info(f"   Size: {archive_file.stat().st_size / 1e9:.2f} GB")
    return archive_file

def extract_chembl_db(archive_file):
    """Extract SQLite database from tar.gz archive."""
    logger.info("\n⏳ Extracting ChEMBL database...")
    logger.info("This may take 5-10 minutes...")
    
    # Expected database location after extraction
    db_file = RAW_DIR / f"chembl_{CHEMBL_VERSION}" / f"chembl_{CHEMBL_VERSION}_sqlite" / f"chembl_{CHEMBL_VERSION}.db"
    
    if db_file.exists():
        logger.info(f"✅ Database already extracted: {db_file}")
        logger.info(f"   Size: {db_file.stat().st_size / 1e9:.2f} GB")
        return db_file
    
    try:
        with tarfile.open(archive_file, 'r:gz') as tar:
            # Extract with progress
            members = tar.getmembers()
            logger.info(f"Extracting {len(members)} files...")
            
            for member in tqdm(members, desc="Extracting"):
                tar.extract(member, path=RAW_DIR)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
    
    # Find the database file
    if not db_file.exists():
        # Try alternative path structure
        possible_paths = list(RAW_DIR.rglob(f"chembl_{CHEMBL_VERSION}.db"))
        if possible_paths:
            db_file = possible_paths[0]
            logger.info(f"Found database at: {db_file}")
        else:
            raise FileNotFoundError("Could not locate extracted database file")
    
    logger.info(f"✅ Extracted to {db_file}")
    logger.info(f"   Database size: {db_file.stat().st_size / 1e9:.2f} GB")
    return db_file

def parse_chembl_to_parquet(db_file):
    """
    Parse ChEMBL SQLite database into Parquet files.
    
    Focuses on:
    1. Compounds with structures
    2. Human targets only
    3. High-quality bioactivity data
    """
    logger.info("\n⏳ Connecting to ChEMBL database...")
    conn = sqlite3.connect(db_file)
    
    # 1. Parse compounds
    logger.info("\n⏳ Parsing compounds (this may take 5-10 min)...")
    compounds_query = """
    SELECT 
        m.chembl_id,
        cs.canonical_smiles,
        cs.standard_inchi_key,
        cp.full_mwt as molecular_weight,
        cp.alogp,
        cp.aromatic_rings,
        cp.hba as h_bond_acceptors,
        cp.hbd as h_bond_donors
    FROM molecule_dictionary m
    JOIN compound_structures cs ON m.molregno = cs.molregno
    LEFT JOIN compound_properties cp ON m.molregno = cp.molregno
    WHERE cs.canonical_smiles IS NOT NULL
        AND cs.standard_inchi_key IS NOT NULL
    """
    
    compounds = pd.read_sql_query(compounds_query, conn)
    output_path = INTERIM_DIR / "compounds.parquet"
    compounds.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved {len(compounds):,} compounds to {output_path}")
    
    # 2. Parse targets (human only)
    logger.info("\n⏳ Parsing human targets...")
    # Fixed query for ChEMBL v36 schema
    targets_query = """
    SELECT DISTINCT
        t.tid,
        t.chembl_id,
        t.pref_name,
        t.target_type,
        t.organism,
        cs.accession as uniprot_accession,
        cs.component_id
    FROM target_dictionary t
    LEFT JOIN target_components tc ON t.tid = tc.tid
    LEFT JOIN component_sequences cs ON tc.component_id = cs.component_id
    WHERE t.organism = 'Homo sapiens'
        AND t.target_type IN ('SINGLE PROTEIN', 'PROTEIN COMPLEX', 'PROTEIN FAMILY')
    """
    
    targets = pd.read_sql_query(targets_query, conn)
    
    # Get gene names from component_synonyms table
    logger.info("⏳ Fetching gene names...")
    gene_query = """
    SELECT 
        cs.component_id,
        csy.component_synonym as gene_name
    FROM component_sequences cs
    JOIN component_synonyms csy ON cs.component_id = csy.component_id
    WHERE csy.syn_type = 'GENE_SYMBOL'
    """
    
    gene_names = pd.read_sql_query(gene_query, conn)
    
    # Merge gene names
    targets = targets.merge(
        gene_names[['component_id', 'gene_name']].drop_duplicates(subset='component_id'),
        on='component_id',
        how='left'
    )
    
    # Drop component_id (internal ID)
    targets = targets.drop('component_id', axis=1)
    
    output_path = INTERIM_DIR / "targets.parquet"
    targets.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved {len(targets):,} human targets to {output_path}")
    
    # 3. Parse activities (HIGH-QUALITY ONLY)
    logger.info("\n⏳ Parsing bioactivities...")
    logger.info("Filtering for high-quality data (IC50, Ki, Kd, EC50)...")
    logger.info("This will take 10-20 minutes for ~10-20M records...")
    
    # ChEMBL v36 schema - activities table has direct ChEMBL ID references
    activities_query = """
    SELECT 
        a.molecule_chembl_id as compound_chembl_id,
        a.target_chembl_id,
        a.standard_type,
        a.standard_value,
        a.standard_units,
        a.pchembl_value,
        a.activity_comment,
        d.pubmed_id,
        a.src_id
    FROM activities a
    LEFT JOIN docs d ON a.doc_id = d.doc_id
    LEFT JOIN target_dictionary t ON a.target_chembl_id = t.chembl_id
    WHERE 
        a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50', 'AC50', 'Potency')
        AND a.standard_value IS NOT NULL
        AND a.standard_units IN ('nM', 'uM')
        AND t.organism = 'Homo sapiens'
        AND a.standard_relation IN ('=', '<', '<=')
    """
    
    # Process in chunks to avoid memory issues
    chunk_size = 100000
    activities_chunks = []
    
    logger.info("Reading activities in chunks...")
    chunk_iter = pd.read_sql_query(activities_query, conn, chunksize=chunk_size)
    
    for i, chunk in enumerate(chunk_iter):
        activities_chunks.append(chunk)
        
        # Log progress every 10 chunks
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {(i + 1) * chunk_size:,} records...")
    
    logger.info("Concatenating chunks...")
    activities = pd.concat(activities_chunks, ignore_index=True)
    
    # Save
    output_path = INTERIM_DIR / "activities.parquet"
    activities.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved {len(activities):,} bioactivity records to {output_path}")
    
    conn.close()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 ChEMBL Parsing Summary")
    logger.info("="*60)
    logger.info(f"Compounds: {len(compounds):,}")
    logger.info(f"Targets: {len(targets):,}")
    logger.info(f"Activities: {len(activities):,}")
    
    unique_pairs = activities[['compound_chembl_id', 'target_chembl_id']].drop_duplicates()
    logger.info(f"Unique compound-target pairs: {len(unique_pairs):,}")
    
    # Activity type breakdown
    logger.info("\nActivity types:")
    for atype, count in activities['standard_type'].value_counts().items():
        logger.info(f"  {atype}: {count:,}")

def main():
    logger.info("="*60)
    logger.info("PHASE 4.1: ChEMBL Extraction & Parsing")
    logger.info("="*60)
    
    try:
        # Check archive exists
        archive = check_chembl_archive()
        
        # Extract
        db_file = extract_chembl_db(archive)
        
        # Parse to Parquet
        parse_chembl_to_parquet(db_file)
        
        logger.info("\n" + "="*60)
        logger.info("✅ ChEMBL extraction and parsing complete!")
        logger.info("="*60)
        logger.info(f"\nOutputs in: {INTERIM_DIR}")
        logger.info("\nNext step:")
        logger.info("  python scripts/phase4_bioactivity/02_match_compounds_to_chembl.py")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
