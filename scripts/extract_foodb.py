#!/usr/bin/env python3
"""
Extract FooDB archive and verify contents.
Handles macOS resource fork metadata.
"""

import tarfile
import os
import sys
import io
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

foodb_archive = Path("data/raw/foodb/foodb_2020_4_7_csv.tar.gz")
extract_dir = Path("data/raw/foodb")

print(f"Checking if archive exists: {foodb_archive.exists()}")

if foodb_archive.exists():
    print(f"Archive size: {foodb_archive.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Check file header
    with open(foodb_archive, 'rb') as f:
        header = f.read(10)
        print(f"File header bytes: {header[:4].hex()}")
        
        # Check if it starts with macOS metadata (._)
        if header[:2] == b'._':
            print("Detected macOS resource fork metadata")
            
            # Try to find the actual gzip header (1f8b)
            f.seek(0)
            data = f.read()
            
            # Search for gzip magic bytes
            gzip_start = data.find(b'\x1f\x8b')
            if gzip_start != -1:
                print(f"Found gzip header at offset {gzip_start}")
                
                # Create a cleaned file
                cleaned_archive = extract_dir / "foodb_cleaned.tar.gz"
                with open(cleaned_archive, 'wb') as out:
                    out.write(data[gzip_start:])
                
                print(f"Created cleaned archive: {cleaned_archive}")
                
                # Try to extract the cleaned file
                try:
                    tar = tarfile.open(cleaned_archive, 'r:gz')
                    print("Successfully opened cleaned archive")
                    tar.extractall(extract_dir)
                    tar.close()
                    print("Extraction complete!")
                    
                    # Remove cleaned file
                    cleaned_archive.unlink()
                except Exception as e:
                    print(f"Failed to extract: {e}")
            else:
                print("Could not find gzip header in file")
        else:
            # Try normal extraction
            f.seek(0)
            try:
                tar = tarfile.open(fileobj=f, mode='r:gz')
                print("Successfully opened as gzipped tar")
                tar.extractall(extract_dir)
                tar.close()
                print("Extraction complete!")
            except Exception as e:
                print(f"Failed: {e}")
    
    # Check extracted files
    csv_dir = extract_dir / "foodb_2020_04_07_csv"
    if csv_dir.exists():
        csv_files = list(csv_dir.glob("*.csv"))
        print(f"\nFound {len(csv_files)} CSV files:")
        for f in csv_files[:10]:
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"\nCSV directory not found: {csv_dir}")
        
        # List what we have
        print("\nActual contents of data/raw/foodb:")
        for item in extract_dir.iterdir():
            if not item.name.endswith('.gz'):
                print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
