#!/usr/bin/env python3
"""Display Phase 3 final statistics."""

import pandas as pd
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*60)
print('PHASE 3: CULINARY MATRIX - FINAL SUMMARY')
print('='*60)

edges = pd.read_csv('data/processed/graph/ingredient_cooccurrence.edgelist')
recipes = pd.read_parquet('data/processed/features/recipe_compound_vectors.parquet')
synergy = pd.read_parquet('data/processed/features/molecular_synergy_scores.parquet')

print(f'\n🌐 CO-OCCURRENCE NETWORK:')
print(f'  Edges: {len(edges):,}')
print(f'  Avg weight: {edges["weight"].mean():.1f}')
print(f'  Max weight: {edges["weight"].max()}')

print(f'\n🧬 MOLECULAR FINGERPRINTS:')
print(f'  Recipes: {len(recipes):,}')
print(f'  Avg ingredients/recipe: {recipes["num_ingredients"].mean():.1f}')
print(f'  Avg compounds/recipe: {recipes["num_compounds"].mean():.1f}')

print(f'\n📊 MOLECULAR SYNERGY:')
print(f'  Compound diversity: {synergy["compound_diversity"].mean():.1f}')
print(f'  Min compounds/recipe: {synergy["num_compounds"].min():,}')
print(f'  Max compounds/recipe: {synergy["num_compounds"].max():,}')

print('\n' + '='*60)
print('✅ Phase 3 COMPLETE!')
print('='*60)
