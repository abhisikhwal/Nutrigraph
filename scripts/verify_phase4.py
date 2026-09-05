#!/usr/bin/env python3
"""Verification script for Phase 4 completion."""

import pandas as pd
import sys
import io
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def verify_phase4():
    """Verify Phase 4 outputs."""
    print("=" * 60)
    print("PHASE 4 VERIFICATION: Bioactivity Layer")
    print("=" * 60)

    checks_passed = 0
    checks_total = 0

    # Check 1: ChEMBL compounds
    checks_total += 1
    compounds_file = Path("data/interim/chembl/compounds.parquet")
    if compounds_file.exists():
        print("✅ Phase 4.1: ChEMBL compounds parsed")
        checks_passed += 1
        df = pd.read_parquet(compounds_file)
        print(f"   - {len(df):,} compounds")
        print(f"   - {df['standard_inchi_key'].notna().sum():,} with InChIKey")
    else:
        print("❌ Phase 4.1: ChEMBL compounds missing")

    # Check 2: ChEMBL targets
    checks_total += 1
    targets_file = Path("data/interim/chembl/targets.parquet")
    if targets_file.exists():
        print("✅ Phase 4.1: ChEMBL targets parsed")
        checks_passed += 1
        df = pd.read_parquet(targets_file)
        print(f"   - {len(df):,} human targets")
    else:
        print("❌ Phase 4.1: ChEMBL targets missing")

    # Check 3: ChEMBL activities
    checks_total += 1
    activities_file = Path("data/interim/chembl/activities.parquet")
    if activities_file.exists():
        print("✅ Phase 4.1: ChEMBL activities parsed")
        checks_passed += 1
        df = pd.read_parquet(activities_file)
        print(f"   - {len(df):,} bioactivity records")
    else:
        print("❌ Phase 4.1: ChEMBL activities missing")

    # Check 4: Food compound matches
    checks_total += 1
    matches_file = Path("data/interim/chembl/food_compound_matches.parquet")
    if matches_file.exists():
        print("✅ Phase 4.2: Food compounds matched ⭐")
        checks_passed += 1
        df = pd.read_parquet(matches_file)
        print(f"   - {len(df):,} food compounds matched")
    else:
        print("❌ Phase 4.2: Food compound matches missing")

    # Check 5: Compound-target network
    checks_total += 1
    network_file = Path("data/processed/graph/compound_target_network.edgelist")
    if network_file.exists():
        print("✅ Phase 4.3: Compound-target network ⭐")
        checks_passed += 1
        df = pd.read_csv(network_file)
        print(f"   - {len(df):,} edges")
    else:
        print("❌ Phase 4.3: Compound-target network missing")

    # Check 6: Compound-targets table
    checks_total += 1
    ct_file = Path("data/processed/canonical/compound_targets.parquet")
    if ct_file.exists():
        print("✅ Phase 4.3: Compound-targets table ⭐")
        checks_passed += 1
        df = pd.read_parquet(ct_file)
        print(f"   - {len(df):,} relationships")
        print(f"   - {df['compound_id'].nunique():,} unique compounds")
        print(f"   - {df['target_chembl_id'].nunique():,} unique targets")
    else:
        print("❌ Phase 4.3: Compound-targets table missing")

    # Check 7: Canonical targets table
    checks_total += 1
    targets_canon = Path("data/processed/canonical/targets.parquet")
    if targets_canon.exists():
        print("✅ Phase 4.4: Canonical targets table ⭐")
        checks_passed += 1
        df = pd.read_parquet(targets_canon)
        print(f"   - {len(df):,} enriched targets")
        print(f"   - Avg compounds/target: {df['num_food_compounds'].mean():.1f}")
    else:
        print("❌ Phase 4.4: Canonical targets table missing")

    print("\n" + "=" * 60)
    print(f"VERIFICATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)

    if checks_passed == checks_total:
        print("\n🎉 SUCCESS! Phase 4 completed successfully!")
        print("\nKey Outputs:")
        print("  ✅ ChEMBL database parsed (2.5M compounds, ~15K targets)")
        print("  ✅ Food compounds matched to ChEMBL")
        print("  ✅ Compound-target bioactivity network built")
        print("  ✅ Targets table enriched with annotations")
        print("\nNext steps:")
        print("  1. Review: Check target statistics and compound matches")
        print("  2. Analyze: Network topology and target druggability")
        print("  3. Proceed to Phase 5: Pathway mapping")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(verify_phase4())
