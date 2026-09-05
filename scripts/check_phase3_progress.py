#!/usr/bin/env python3
"""
Check Phase 3 full processing status and estimate completion time.
"""

import json
import sys
import io
from pathlib import Path
from datetime import datetime, timedelta

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CHECKPOINT_DIR = Path("data/interim/recipenlg/checkpoints")
INTERIM_DIR = Path("data/interim/recipenlg")
GRAPH_DIR = Path("data/processed/graph")
FEATURES_DIR = Path("data/processed/features")

TOTAL_RECIPES = 2_231_142  # RecipeNLG full dataset

def check_parsing_progress():
    """Check recipe parsing progress."""
    progress_file = CHECKPOINT_DIR / "parsing_progress.json"
    checkpoint_file = CHECKPOINT_DIR / "parsing_checkpoint.parquet"
    
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        
        processed = progress['last_processed']
        percentage = processed / TOTAL_RECIPES * 100
        
        print("🔄 PARSING IN PROGRESS")
        print(f"   Processed: {processed:,} / {TOTAL_RECIPES:,} ({percentage:.1f}%)")
        print(f"   Checkpoint: {checkpoint_file}")
        
        # Estimate time remaining (assuming 2 hours for full dataset)
        estimated_total_seconds = 2 * 3600
        estimated_remaining = estimated_total_seconds * (1 - percentage / 100)
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        print(f"   Estimated completion: {eta.strftime('%H:%M:%S')}")
        
        return 'in_progress'
    
    final_file = INTERIM_DIR / "recipes_parsed_full.parquet"
    if final_file.exists():
        print("✅ PARSING COMPLETE")
        print(f"   Output: {final_file}")
        return 'complete'
    
    print("❌ PARSING NOT STARTED")
    return 'not_started'

def check_outputs():
    """Check final outputs."""
    outputs = {
        'Recipes parsed': INTERIM_DIR / "recipes_parsed_full.parquet",
        'Ingredients mapped': INTERIM_DIR / "recipe_ingredients_mapped_full.parquet",
        'Co-occurrence network': GRAPH_DIR / "ingredient_cooccurrence_full.edgelist",
        'Compound vectors': FEATURES_DIR / "recipe_compound_vectors_full.parquet",
        'Synergy scores': FEATURES_DIR / "molecular_synergy_scores_full.parquet",
    }
    
    print("\n📊 OUTPUT STATUS:")
    for name, path in outputs.items():
        if path.exists():
            size = path.stat().st_size / 1024 / 1024
            print(f"   ✅ {name}: {size:.1f} MB")
        else:
            print(f"   ❌ {name}: Not found")

def main():
    print("=" * 60)
    print("PHASE 3 FULL PROCESSING - STATUS CHECK")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check parsing
    status = check_parsing_progress()
    
    # Check outputs
    check_outputs()
    
    print("\n" + "=" * 60)
    
    if status == 'in_progress':
        print("⏳ Processing in progress - run script again to resume")
    elif status == 'complete':
        print("✅ All steps complete!")
    else:
        print("🆕 Ready to start - run: python scripts/phase3_culinary_matrix/run_full_dataset.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
