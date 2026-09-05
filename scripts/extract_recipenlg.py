#!/usr/bin/env python3
"""Extract RecipeNLG archive."""

import zipfile
from pathlib import Path

zip_path = Path('data/raw/recipenlg/archive.zip')
extract_dir = Path('data/raw/recipenlg')

print(f'Extracting {zip_path}...')
print(f'Archive size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    print(f'Extracted {len(zip_ref.namelist())} files')

print('\nFiles in directory:')
for item in extract_dir.iterdir():
    if item.name != 'archive.zip':
        size = item.stat().st_size / 1024 / 1024 if item.is_file() else 0
        print(f'  - {item.name} ({size:.1f} MB)' if item.is_file() else f'  - {item.name}/ (dir)')

print('\nExtraction complete!')
