#!/usr/bin/env python3
"""
Verification script for Phase 0 & 1 completion.
"""

import pandas as pd
from pathlib import Path
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def verify_phase0_1():
    """Verify Phase 0 and 1 outputs."""
    print("=" * 60)
    print("PHASE 0 & 1 VERIFICATION")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Compliance report exists
    checks_total += 1
    compliance_report = Path("licenses/compliance_report.txt")
    if compliance_report.exists():
        print("✅ Phase 0: Compliance report exists")
        checks_passed += 1
        with open(compliance_report, encoding='utf-8') as f:
            content = f.read()
            if "COMMERCIAL-SAFE DATASETS" in content:
                print("   - Report contains commercial-safe dataset list")
    else:
        print("❌ Phase 0: Compliance report missing")
    
    # Check 2: USDA raw data
    checks_total += 1
    usda_raw = Path("data/raw/usda/foundation_foods.json")
    if usda_raw.exists():
        print("✅ Phase 1.1: USDA raw data exists")
        checks_passed += 1
    else:
        print("❌ Phase 1.1: USDA raw data missing")
    
    # Check 3: USDA interim data
    checks_total += 1
    usda_interim = Path("data/interim/usda/foundation_foods.parquet")
    if usda_interim.exists():
        print("✅ Phase 1.1: USDA interim data exists")
        checks_passed += 1
        usda_df = pd.read_parquet(usda_interim)
        print(f"   - Contains {len(usda_df)} items")
    else:
        print("❌ Phase 1.1: USDA interim data missing")
    
    # Check 4: Wikidata raw data
    checks_total += 1
    wikidata_raw = Path("data/raw/wikidata/food_items.json")
    if wikidata_raw.exists():
        print("✅ Phase 1.2: Wikidata raw data exists")
        checks_passed += 1
    else:
        print("❌ Phase 1.2: Wikidata raw data missing")
    
    # Check 5: Wikidata interim data
    checks_total += 1
    wikidata_interim = Path("data/interim/wikidata/wikidata_foods.parquet")
    if wikidata_interim.exists():
        print("✅ Phase 1.2: Wikidata interim data exists")
        checks_passed += 1
        wikidata_df = pd.read_parquet(wikidata_interim)
        print(f"   - Contains {len(wikidata_df)} items")
    else:
        print("❌ Phase 1.2: Wikidata interim data missing")
    
    # Check 6: Ingredient master (PRIMARY OUTPUT)
    checks_total += 1
    ingredient_master = Path("data/processed/canonical/ingredients.parquet")
    if ingredient_master.exists():
        print("✅ Phase 1.3: Ingredient master exists ⭐")
        checks_passed += 1
        
        df = pd.read_parquet(ingredient_master)
        print(f"   - Total ingredients: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - Sources: {df['source'].value_counts().to_dict()}")
        
        # Validate schema
        required_cols = ['ingredient_id', 'canonical_name', 'source']
        if all(col in df.columns for col in required_cols):
            print("   - Schema validation: PASSED")
        else:
            print("   - Schema validation: FAILED")
    else:
        print("❌ Phase 1.3: Ingredient master missing")
    
    print("\n" + "=" * 60)
    print(f"VERIFICATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("\n🎉 SUCCESS! Phase 0 & 1 completed successfully!")
        print("\nNext steps:")
        print("  1. Review: cat PHASE_0_1_EXECUTION_SUMMARY.md")
        print("  2. Proceed to Phase 2: python scripts/phase2_food_chemistry/01_download_foodb.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(verify_phase0_1())
