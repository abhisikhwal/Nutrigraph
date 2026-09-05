#!/usr/bin/env python3
"""
Phase 2.1: Download FooDB compound data.

WARNING: FooDB has CC BY-NC 4.0 license (non-commercial).
Verify license before using in commercial products.
"""

import logging
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.downloaders import DatasetDownloader
from utils.logging_config import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def download_foodb():
    """Download FooDB data."""
    logger.info("=== Phase 2.1: Download FooDB ===")
    
    # Load paths
    with open("config/paths.yaml") as f:
        paths = yaml.safe_load(f)
    
    raw_dir = Path(paths['raw_foodb'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize downloader
    downloader = DatasetDownloader("foodb")
    downloader.verify_license()
    
    # WARNING about license
    logger.warning("=" * 60)
    logger.warning("⚠️  FooDB LICENSE WARNING")
    logger.warning("=" * 60)
    logger.warning("FooDB uses CC BY-NC 4.0 (Non-Commercial)")
    logger.warning("You CANNOT use FooDB data in commercial products/services")
    logger.warning("For commercial use, rely on PubChem/ChEMBL instead")
    logger.warning("=" * 60)
    
    # Manual download instructions
    logger.info("MANUAL STEP REQUIRED:")
    logger.info("1. Visit https://foodb.ca/downloads")
    logger.info("2. Download 'Compounds' CSV file")
    logger.info("3. Download 'Compound-Food' associations CSV")
    logger.info(f"4. Place files in: {raw_dir.absolute()}")
    
    logger.info(f"\nOutput directory: {raw_dir.absolute()}")
    logger.info("✓ FooDB download step prepared (manual download required)")


def main():
    """Main execution."""
    download_foodb()
    return 0


if __name__ == "__main__":
    sys.exit(main())
