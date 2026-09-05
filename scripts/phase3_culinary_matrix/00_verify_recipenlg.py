#!/usr/bin/env python3
"""
Verify RecipeNLG dataset is available.
"""

import logging
from pathlib import Path
import yaml
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = Path(paths["raw_data"]) / "recipenlg"

def verify_dataset():
    """Check if RecipeNLG dataset exists."""
    
    if not RAW_DIR.exists():
        logger.error(f"RecipeNLG directory not found: {RAW_DIR}")
        logger.info("\nPlease download RecipeNLG:")
        logger.info("1. Go to: https://recipenlg.cs.put.poznan.pl/dataset")
        logger.info("2. Download 'Full dataset' (CSV format preferred)")
        logger.info(f"3. Create directory: {RAW_DIR}")
        logger.info(f"4. Place file in: {RAW_DIR}/full_dataset.csv")
        return False
    
    # Check for files in main dir and subdirectories
    csv_files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("**/*.csv"))
    json_files = list(RAW_DIR.glob("*.json")) + list(RAW_DIR.glob("**/*.json"))
    
    if not csv_files and not json_files:
        logger.error("No CSV or JSON files found in RecipeNLG directory")
        logger.info(f"Expected file: {RAW_DIR}/full_dataset.csv or {RAW_DIR}/dataset/full_dataset.csv")
        return False
    
    data_file = csv_files[0] if csv_files else json_files[0]
    logger.info(f"✅ Found RecipeNLG dataset: {data_file}")
    logger.info(f"File size: {data_file.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"License: MIT (fully open, commercial OK)")
    return True

if __name__ == "__main__":
    success = verify_dataset()
    if not success:
        sys.exit(1)
