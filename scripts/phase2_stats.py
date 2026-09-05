#!/usr/bin/env python3
"""Display Phase 2 final statistics."""

import pandas as pd
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

mappings = pd.read_parquet('data/processed/canonical/ingredient_compounds.parquet')

print('='*60)
print('PHASE 2 WITH FooDB: FINAL SUMMARY')
print('='*60)
print(f'\nTotal mappings: {len(mappings):,}')
print(f'Unique ingredients: {mappings["ingredient_id"].nunique()}')
print(f'Unique compounds: {mappings["compound_id"].nunique()}')

with_conc = mappings[mappings["content_value"].notna()]
print(f'Mappings with concentration: {len(with_conc):,} ({len(with_conc)/len(mappings)*100:.1f}%)')

print('\nTop 10 ingredients by compound diversity:')
top = mappings.groupby('ingredient_id')['compound_id'].nunique().sort_values(ascending=False).head(10)
print(top.to_string())

print('\n' + '='*60)
print('✅ Phase 2 COMPLETE with FooDB Integration!')
print('='*60)
