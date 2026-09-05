#!/usr/bin/env python3
"""
Verify FooDB license compliance before processing.

FooDB is licensed under CC BY-NC 4.0 (Non-Commercial).
This script:
1. Checks if FooDB data exists
2. Warns about non-commercial restrictions
3. Asks user to confirm they understand limitations
4. Logs acknowledgment for compliance tracking
"""

import logging
from pathlib import Path
import yaml
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

FOODB_DIR = Path(paths["raw_data"]) / "foodb"

def check_foodb_license():
    """
    Interactive license compliance check for FooDB.
    """
    print("\n" + "="*70)
    print("⚠️  FooDB LICENSE COMPLIANCE CHECK")
    print("="*70)
    print("\nFooDB is licensed under CC BY-NC 4.0 (Non-Commercial)")
    print("\nThis means:")
    print("  ❌ You CANNOT use FooDB data in commercial products")
    print("  ❌ You CANNOT use it in a for-profit startup")
    print("  ✅ You CAN use it for academic research")
    print("  ✅ You CAN use it for PhD thesis work")
    print("\nIf you plan to commercialize this project, you must:")
    print("  1. Remove FooDB from your pipeline, OR")
    print("  2. Replace it with commercial-licensed alternatives (e.g., ChemSpider)")
    print("\n" + "="*70)
    
    # For automated execution, skip interactive prompt
    if sys.stdin.isatty():
        response = input("\nDo you understand and accept these restrictions? (yes/no): ").strip().lower()
    else:
        logger.info("Non-interactive mode - skipping FooDB for safety")
        response = "no"
    
    if response != 'yes':
        print("\n❌ FooDB processing aborted.")
        print("Proceeding with PubChem-only compound data (fully open).\n")
        return False
    
    # Log acknowledgment
    log_path = Path("licenses/foodb_acknowledgment.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        from datetime import datetime
        f.write(f"FooDB Non-Commercial License Acknowledged\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"User confirmed understanding of CC BY-NC 4.0 restrictions\n")
    
    print(f"\n✅ Acknowledgment logged to {log_path}")
    print("Proceeding with FooDB processing...\n")
    return True

def main():
    if not FOODB_DIR.exists():
        logger.warning(f"FooDB directory not found: {FOODB_DIR}")
        logger.info("FooDB is optional. Proceeding with PubChem-only mode.")
        print("\n✅ No FooDB data found - proceeding with commercial-safe alternatives only")
        return 0
    
    # Check for CSV files
    csv_files = list(FOODB_DIR.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in FooDB directory")
        logger.info("Expected files: Compound.csv, Content.csv, Food.csv")
        print("\n✅ No FooDB CSV files - proceeding with commercial-safe alternatives only")
        return 0
    
    logger.info(f"Found {len(csv_files)} FooDB CSV files")
    
    # Interactive license check
    if not check_foodb_license():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
