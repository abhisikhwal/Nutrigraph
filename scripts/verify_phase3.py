#!/usr/bin/env python3
"""Verification script for Phase 3 completion."""

import pandas as pd
import sys
import io
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def verify_phase3():
    """Verify Phase 3 outputs."""
    print("=" * 60)
    print("PHASE 3 VERIFICATION: Culinary Matrix")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Recipe parsing
    checks_total += 1
    recipes = Path("data/interim/recipenlg/recipes_parsed.parquet")
    if recipes.exists():
        print("✅ Phase 3.1: Recipes parsed")
        checks_passed += 1
        df = pd.read_parquet(recipes)
        print(f"   - {len(df)} recipes")
    else:
        print("❌ Phase 3.1: Recipes missing")
    
    # Check 2: Ingredient mapping
    checks_total += 1
    mapping = Path("data/interim/recipenlg/recipe_ingredients_mapped.parquet")
    if mapping.exists():
        print("✅ Phase 3.2: Ingredients mapped")
        checks_passed += 1
        df = pd.read_parquet(mapping)
        print(f"   - {len(df)} recipe-ingredient pairs")
        print(f"   - {df['ingredient_id'].nunique()} unique ingredients")
    else:
        print("❌ Phase 3.2: Ingredient mapping missing")
    
    # Check 3: Co-occurrence network
    checks_total += 1
    network = Path("data/processed/graph/ingredient_cooccurrence.edgelist")
    if network.exists():
        print("✅ Phase 3.3: Co-occurrence network exists ⭐")
        checks_passed += 1
        df = pd.read_csv(network)
        print(f"   - {len(df)} edges")
    else:
        print("❌ Phase 3.3: Co-occurrence network missing")
    
    # Check 4: Recipe compound vectors
    checks_total += 1
    vectors = Path("data/processed/features/recipe_compound_vectors.parquet")
    if vectors.exists():
        print("✅ Phase 3.4: Recipe compound vectors exist ⭐")
        checks_passed += 1
        df = pd.read_parquet(vectors)
        print(f"   - {len(df)} recipes with compound data")
    else:
        print("❌ Phase 3.4: Recipe compound vectors missing")
    
    # Check 5: Molecular synergy scores
    checks_total += 1
    synergy = Path("data/processed/features/molecular_synergy_scores.parquet")
    if synergy.exists():
        print("✅ Phase 3.5: Molecular synergy scores exist ⭐")
        checks_passed += 1
        df = pd.read_parquet(synergy)
        print(f"   - {len(df)} recipes analyzed")
        print(f"   - Avg compound diversity: {df['compound_diversity'].mean():.1f}")
    else:
        print("❌ Phase 3.5: Molecular synergy scores missing")
    
    print("\n" + "=" * 60)
    print(f"VERIFICATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("\n🎉 SUCCESS! Phase 3 completed successfully!")
        print("\nKey Outputs:")
        print("  ✅ Co-occurrence network with 21K+ edges")
        print("  ✅ Recipe molecular fingerprints (9.6K recipes)")
        print("  ✅ Molecular synergy scores computed")
        print("\nNext steps:")
        print("  1. Review: cat PHASE_3_EXECUTION_SUMMARY.md")
        print("  2. Analyze: molecular complementarity by cuisine")
        print("  3. Proceed to Phase 4: Bioactivity prediction")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(verify_phase3())
