#!/usr/bin/env python3
"""
Phase 0: Dataset License Registry Validator

Validates that:
1. All datasets in the registry have complete metadata
2. License fields are not empty or marked with "?"
3. Commercial use flags are explicit (Yes/No, not "?")
4. Warns about non-commercial datasets
5. Generates a compliance report

Outputs:
- licenses/compliance_report.txt (human-readable summary)
- Console warnings for datasets needing manual verification
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_registry():
    """
    Validate the datasets registry for completeness and flag issues.
    """
    registry_path = Path("licenses/datasets_registry.csv")
    
    if not registry_path.exists():
        logger.error(f"Registry not found at {registry_path}")
        return False
    
    df = pd.read_csv(registry_path)
    
    # Required columns
    required_cols = [
        'dataset_name', 'url', 'license', 
        'redistribution_allowed', 'commercial_use'
    ]
    
    issues = []
    warnings = []
    
    # Check for missing columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check each dataset
    for idx, row in df.iterrows():
        dataset = row['dataset_name']
        
        # Check for empty/unclear licenses
        if pd.isna(row['license']) or '?' in str(row['license']):
            issues.append(f"{dataset}: License unclear or missing")
        
        # Check for unclear commercial use
        if pd.isna(row['commercial_use']) or '?' in str(row['commercial_use']):
            warnings.append(f"{dataset}: Commercial use status unclear - VERIFY MANUALLY")
        
        # Flag non-commercial datasets
        if str(row['commercial_use']).lower() == 'no':
            warnings.append(f"{dataset}: NON-COMMERCIAL license - cannot use in commercial products")
    
    # Generate report
    report_path = Path("licenses/compliance_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset License Compliance Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 60 + "\n\n")
        
        f.write(f"Total datasets: {len(df)}\n\n")
        
        if issues:
            f.write("🚨 CRITICAL ISSUES (Must fix before proceeding):\n")
            for issue in issues:
                f.write(f"  - {issue}\n")
            f.write("\n")
        
        if warnings:
            f.write("⚠️  WARNINGS (Review carefully):\n")
            for warning in warnings:
                f.write(f"  - {warning}\n")
            f.write("\n")
        
        # Commercial-safe datasets
        commercial_safe = df[df['commercial_use'].str.lower() == 'yes']
        f.write(f"✅ COMMERCIAL-SAFE DATASETS ({len(commercial_safe)}):\n")
        for name in commercial_safe['dataset_name']:
            f.write(f"  - {name}\n")
        f.write("\n")
        
        # Non-commercial datasets
        non_commercial = df[df['commercial_use'].str.lower() == 'no']
        if len(non_commercial) > 0:
            f.write(f"❌ NON-COMMERCIAL DATASETS ({len(non_commercial)}):\n")
            for name in non_commercial['dataset_name']:
                f.write(f"  - {name}\n")
    
    logger.info(f"Compliance report written to {report_path}")
    
    if issues:
        logger.error("CRITICAL ISSUES found - see compliance report")
        return False
    elif warnings:
        logger.warning(f"{len(warnings)} warnings - see compliance report")
        return True
    else:
        logger.info("✅ All datasets validated successfully")
        return True

if __name__ == "__main__":
    success = validate_registry()
    if not success:
        exit(1)
