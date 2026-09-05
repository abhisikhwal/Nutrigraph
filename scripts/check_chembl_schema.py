#!/usr/bin/env python3
"""
Quick script to check ChEMBL database schema and verify v36 compatibility.

Run this to diagnose schema issues before running the notebook.
"""

import sqlite3
import pandas as pd
from pathlib import Path

def check_chembl_schema():
    """Check ChEMBL database schema and verify v36 compatibility."""
    
    print("=" * 70)
    print("ChEMBL SCHEMA CHECKER")
    print("=" * 70)
    
    # Find database file
    db_paths = list(Path("data/raw/chembl").rglob("chembl_*.db"))
    
    if not db_paths:
        print("\n❌ No ChEMBL database found!")
        print("   Expected location: data/raw/chembl/chembl_36/chembl_36_sqlite/chembl_36.db")
        return False
    
    db_file = db_paths[0]
    print(f"\n✅ Found database: {db_file}")
    print(f"   Size: {db_file.stat().st_size / 1e9:.2f} GB")
    
    # Connect
    conn = sqlite3.connect(db_file)
    
    # Check version
    print("\n" + "=" * 70)
    print("DATABASE VERSION")
    print("=" * 70)
    
    try:
        version_info = pd.read_sql_query("SELECT * FROM version", conn)
        print(f"ChEMBL Version: {version_info.iloc[0]['name']}")
    except:
        print("⚠️  Could not determine version")
    
    # Check tables
    print("\n" + "=" * 70)
    print("CHECKING REQUIRED TABLES")
    print("=" * 70)
    
    tables_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    all_tables = pd.read_sql_query(tables_query, conn)['name'].tolist()
    
    required_tables = [
        'activities',
        'target_dictionary',
        'target_components',
        'component_sequences',
        'component_synonyms',
        'molecule_dictionary',
        'compound_structures'
    ]
    
    for table in required_tables:
        status = "✅" if table in all_tables else "❌"
        print(f"{status} {table}")
    
    # Check activities table schema
    print("\n" + "=" * 70)
    print("ACTIVITIES TABLE SCHEMA")
    print("=" * 70)
    
    activities_schema = pd.read_sql_query("PRAGMA table_info(activities)", conn)
    activities_cols = activities_schema['name'].tolist()
    
    # Check for v36 columns
    v36_checks = {
        'molecule_chembl_id': 'molecule_chembl_id' in activities_cols,
        'target_chembl_id': 'target_chembl_id' in activities_cols,
        'molregno (old)': 'molregno' in activities_cols,
        'tid (old)': 'tid' in activities_cols
    }
    
    for check, result in v36_checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
    
    # Determine version
    if v36_checks['molecule_chembl_id'] and v36_checks['target_chembl_id']:
        print("\n✅ This is ChEMBL v36 or later - notebook fixes are compatible!")
        is_v36 = True
    elif v36_checks['molregno (old)'] and v36_checks['tid (old)']:
        print("\n⚠️  This is ChEMBL v35 or earlier - you need to update ChEMBL!")
        print("   Download v36 from: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/")
        is_v36 = False
    else:
        print("\n❌ Unknown schema version!")
        is_v36 = False
    
    # Check target_components schema
    print("\n" + "=" * 70)
    print("TARGET_COMPONENTS TABLE SCHEMA")
    print("=" * 70)
    
    tc_schema = pd.read_sql_query("PRAGMA table_info(target_components)", conn)
    tc_cols = tc_schema['name'].tolist()
    
    tc_checks = {
        'component_id': 'component_id' in tc_cols,
        'accession (old)': 'accession' in tc_cols,
        'gene_name (old)': 'gene_name' in tc_cols
    }
    
    for check, result in tc_checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
    
    if not tc_checks['accession (old)']:
        print("\n✅ Target components use v36 schema (component_id) - notebook fixes compatible!")
    else:
        print("\n⚠️  Target components use old schema - unexpected for v36")
    
    # Check component_sequences
    print("\n" + "=" * 70)
    print("COMPONENT_SEQUENCES TABLE")
    print("=" * 70)
    
    if 'component_sequences' in all_tables:
        cs_schema = pd.read_sql_query("PRAGMA table_info(component_sequences)", conn)
        cs_cols = cs_schema['name'].tolist()
        
        print(f"✅ Table exists")
        print(f"   Columns: {', '.join(cs_cols[:10])}")
        
        if 'accession' in cs_cols:
            print("   ✅ Has 'accession' column (UniProt IDs)")
        else:
            print("   ❌ Missing 'accession' column")
    else:
        print("❌ Table does not exist")
    
    # Check component_synonyms
    print("\n" + "=" * 70)
    print("COMPONENT_SYNONYMS TABLE")
    print("=" * 70)
    
    if 'component_synonyms' in all_tables:
        print("✅ Table exists")
        
        # Check for gene symbols
        gene_count = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM component_synonyms WHERE syn_type = 'GENE_SYMBOL'",
            conn
        )
        print(f"   Gene symbols: {gene_count.iloc[0]['count']:,}")
    else:
        print("❌ Table does not exist")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("COMPATIBILITY CHECK")
    print("=" * 70)
    
    if is_v36:
        print("✅ ChEMBL v36 schema detected")
        print("✅ Notebook fixes are compatible")
        print("✅ Ready to run Phase 4!")
        
        print("\n📝 Next steps:")
        print("   1. Restart Jupyter notebook")
        print("   2. Run Section 4 (Parse Targets)")
        print("   3. Run Section 5 (Parse Activities)")
        print("   4. Continue to Section 6+")
    else:
        print("❌ Incompatible schema detected")
        print("⚠️  You need ChEMBL v36 or later")
        
        print("\n📥 Download ChEMBL v36:")
        print("   URL: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/")
        print("   File: chembl_36_sqlite.tar.gz (3.5 GB)")
        print("   Place in: data/raw/chembl/")
    
    conn.close()
    return is_v36

if __name__ == "__main__":
    try:
        check_chembl_schema()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
