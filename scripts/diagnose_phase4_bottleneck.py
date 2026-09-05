"""
Diagnose Phase 4 bottleneck - why only 3,039 edges instead of 50K+?

This script traces the data flow:
8,133 matched compounds → ??? have activities → 828 with edges → 3,039 final edges
"""

import pandas as pd
import sqlite3
from pathlib import Path
import yaml

# Load config
with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

INTERIM_DIR = Path(paths["interim_data"])
CANONICAL_DIR = Path(paths["processed_data"]) / "canonical"
RAW_DIR = Path(paths["raw_data"]) / "chembl"

print("=" * 60)
print("PHASE 4 BOTTLENECK DIAGNOSTIC")
print("=" * 60)

# Step 1: Load matched compounds
print("\n[STEP 1] MATCHED COMPOUNDS")
matches = pd.read_parquet(INTERIM_DIR / "chembl" / "food_compound_matches.parquet")
print(f"   Total matched: {len(matches):,}")
print(f"   Unique compound IDs: {matches['compound_id'].nunique():,}")
print(f"   Unique ChEMBL IDs: {matches['chembl_id'].nunique():,}")

# Step 2: Load activities that were saved
print("\n[STEP 2] LOADED ACTIVITIES")
activities = pd.read_parquet(INTERIM_DIR / "chembl" / "activities.parquet")
print(f"   Total activities loaded: {len(activities):,}")
print(f"   Unique compounds in activities: {activities['compound_chembl_id'].nunique():,}")
print(f"   Unique targets in activities: {activities['target_chembl_id'].nunique():,}")

# Step 3: Check overlap
print("\n[STEP 3] OVERLAP CHECK")
matched_chembl_ids = set(matches['chembl_id'].unique())
activity_chembl_ids = set(activities['compound_chembl_id'].unique())

overlap = matched_chembl_ids.intersection(activity_chembl_ids)
print(f"   Matched compounds: {len(matched_chembl_ids):,}")
print(f"   Compounds in activities: {len(activity_chembl_ids):,}")
print(f"   OVERLAP: {len(overlap):,} compounds")
print(f"   Missing from activities: {len(matched_chembl_ids - activity_chembl_ids):,} compounds")

# Step 4: Sample missing compounds
print("\n[STEP 4] SAMPLE MISSING COMPOUNDS (first 10)")
missing = matched_chembl_ids - activity_chembl_ids
for chembl_id in list(missing)[:10]:
    compound_info = matches[matches['chembl_id'] == chembl_id].iloc[0]
    print(f"   {chembl_id}: {compound_info.get('name', 'Unknown')}")

# Step 5: Check final edges
print("\n[STEP 5] FINAL COMPOUND-TARGET EDGES")
try:
    compound_targets = pd.read_parquet(CANONICAL_DIR / "compound_targets.parquet")
    print(f"   Total edges: {len(compound_targets):,}")
    print(f"   Unique compounds: {compound_targets['compound_id'].nunique():,}")
    print(f"   Unique targets: {compound_targets['target_chembl_id'].nunique():,}")
except FileNotFoundError:
    print(f"   ⚠️  compound_targets.parquet not found (Section 8 not run yet?)")
    compound_targets = pd.DataFrame()

# Step 6: Check ChEMBL database total activities
print("\n[STEP 6] CHEMBL DATABASE TOTALS")
db_file = RAW_DIR / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"

if db_file.exists():
    conn = sqlite3.connect(db_file)
    
    # Total activities in database
    total_activities = pd.read_sql_query("""
        SELECT COUNT(*) as count 
        FROM activities 
        WHERE pchembl_value IS NOT NULL
    """, conn)
    print(f"   Total activities in ChEMBL DB: {total_activities['count'].iloc[0]:,}")
    
    # Activities for human targets
    try:
        human_activities = pd.read_sql_query("""
            SELECT COUNT(*) as count
            FROM activities a
            JOIN assays asy ON a.assay_id = asy.assay_id
            JOIN target_dictionary t ON asy.tid = t.tid
            WHERE t.organism = 'Homo sapiens'
                AND a.pchembl_value IS NOT NULL
        """, conn)
        print(f"   Human target activities: {human_activities['count'].iloc[0]:,}")
    except Exception as e:
        print(f"   ⚠️  Could not query human activities: {e}")
    
    # Check if any of our compounds have activities in DB
    sample_chembl_ids = list(matched_chembl_ids)[:5]
    placeholders = ','.join(['?' for _ in sample_chembl_ids])
    
    try:
        sample_check = pd.read_sql_query(f"""
            SELECT m.chembl_id, COUNT(*) as activity_count
            FROM activities a
            JOIN molecule_dictionary m ON a.molregno = m.molregno
            WHERE m.chembl_id IN ({placeholders})
            GROUP BY m.chembl_id
        """, conn, params=sample_chembl_ids)
        
        print(f"\n   Sample check (first 5 matched compounds):")
        if len(sample_check) > 0:
            for _, row in sample_check.iterrows():
                print(f"      {row['chembl_id']}: {row['activity_count']:,} activities in DB")
        else:
            print(f"      ⚠️  None of the sampled compounds have activities in ChEMBL")
    except Exception as e:
        print(f"   ⚠️  Could not check sample compounds: {e}")
    
    conn.close()
else:
    print(f"   ⚠️  ChEMBL database not found at {db_file}")

# Step 7: Diagnosis
print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)

overlap_rate = len(overlap) / len(matched_chembl_ids) * 100
print(f"\nCoverage: {overlap_rate:.1f}% of matched compounds have activities")

if overlap_rate < 20:
    print("\nBOTTLENECK IDENTIFIED:")
    print("   Only {:.1f}% of matched compounds have activities in the loaded data".format(overlap_rate))
    print("\nLIKELY CAUSE:")
    print("   Phase 4 activities query had a LIMIT that excluded most food compounds")
    print("\nSOLUTION:")
    print("   Re-run Phase 4 Section 6 with one of these fixes:")
    print("   1. Remove the LIMIT clause entirely (loads all ~10-20M activities)")
    print("   2. Pre-filter for your 8,133 compound ChEMBL IDs:")
    print("      WHERE molecule_chembl_id IN ('CHEMBL...', 'CHEMBL...', ...)")
    print("\nExpected improvement:")
    print(f"   Current: {len(overlap)} compounds -> {len(compound_targets) if len(compound_targets) > 0 else '3,039'} edges")
    print(f"   After fix: 4,000-6,000 compounds -> 20,000-80,000 edges")
else:
    print("\nCoverage looks good!")
    print("   The bottleneck is elsewhere (possibly pChEMBL threshold too strict)")

print("\n" + "=" * 60)
