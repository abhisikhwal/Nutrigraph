"""
Schema validation utilities to ensure data quality.
"""

import logging
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import jsonschema

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validate DataFrames against JSON schemas.
    """
    
    def __init__(self, schema_dir: Path = Path("schemas")):
        """
        Args:
            schema_dir: Directory containing JSON schema files
        """
        self.schema_dir = Path(schema_dir)
        self.schemas: Dict[str, Dict] = {}
        
        # Load all schemas
        for schema_file in self.schema_dir.glob("*.json"):
            schema_name = schema_file.stem
            with open(schema_file) as f:
                self.schemas[schema_name] = json.load(f)
        
        logger.info(f"Loaded {len(self.schemas)} schemas from {schema_dir}")
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        schema_name: str,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Validate a DataFrame against a schema.
        
        Args:
            df: DataFrame to validate
            schema_name: Name of schema (without .json extension)
            sample_size: Number of rows to validate (for performance)
            
        Returns:
            Dict with validation results: {'valid': bool, 'errors': List[str]}
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Schema {schema_name} not found")
        
        schema = self.schemas[schema_name]
        errors = []
        
        # Validate required columns
        required_fields = schema.get('required', [])
        missing_fields = set(required_fields) - set(df.columns)
        
        if missing_fields:
            errors.append(f"Missing required columns: {missing_fields}")
        
        # Sample rows for validation
        sample_df = df.head(sample_size) if len(df) > sample_size else df
        
        # Validate each row
        for idx, row in sample_df.iterrows():
            record = row.to_dict()
            
            # Convert NaN to None for JSON schema validation
            record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            
            try:
                jsonschema.validate(instance=record, schema=schema)
            except jsonschema.ValidationError as e:
                error_msg = f"Row {idx}: {e.message}"
                errors.append(error_msg)
                
                # Limit error messages
                if len(errors) > 10:
                    errors.append("... (additional errors truncated)")
                    break
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"✓ DataFrame validates against {schema_name} schema")
        else:
            logger.error(f"✗ DataFrame has {len(errors)} validation errors")
            for error in errors[:5]:  # Log first 5 errors
                logger.error(f"  - {error}")
        
        return {
            'valid': is_valid,
            'errors': errors,
            'schema_name': schema_name,
            'rows_validated': len(sample_df)
        }
    
    def check_foreign_keys(
        self,
        df: pd.DataFrame,
        foreign_key_col: str,
        reference_df: pd.DataFrame,
        reference_key_col: str
    ) -> Dict[str, Any]:
        """
        Check foreign key constraints between DataFrames.
        
        Args:
            df: DataFrame with foreign key
            foreign_key_col: Column name in df
            reference_df: Reference DataFrame
            reference_key_col: Key column in reference_df
            
        Returns:
            Dict with validation results
        """
        fk_values = set(df[foreign_key_col].dropna())
        ref_values = set(reference_df[reference_key_col])
        
        missing = fk_values - ref_values
        
        if missing:
            logger.error(
                f"Foreign key violation: {len(missing)} values in "
                f"{foreign_key_col} not found in {reference_key_col}"
            )
            return {
                'valid': False,
                'missing_keys': list(missing)[:100]  # Limit output
            }
        else:
            logger.info(
                f"✓ Foreign key constraint satisfied: "
                f"{foreign_key_col} → {reference_key_col}"
            )
            return {'valid': True, 'missing_keys': []}
    
    def check_id_format(
        self,
        df: pd.DataFrame,
        id_col: str,
        pattern: str
    ) -> Dict[str, Any]:
        """
        Check that ID column matches expected pattern.
        
        Args:
            df: DataFrame to check
            id_col: ID column name
            pattern: Regex pattern (e.g., '^ING_\\d{5}$')
            
        Returns:
            Dict with validation results
        """
        import re
        
        invalid_ids = df[~df[id_col].astype(str).str.match(pattern)][id_col]
        
        if len(invalid_ids) > 0:
            logger.error(
                f"ID format violation: {len(invalid_ids)} IDs in {id_col} "
                f"don't match pattern {pattern}"
            )
            return {
                'valid': False,
                'invalid_ids': list(invalid_ids)[:100]
            }
        else:
            logger.info(f"✓ All IDs in {id_col} match pattern {pattern}")
            return {'valid': True, 'invalid_ids': []}
