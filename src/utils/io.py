"""
I/O utilities for reading/writing data in various formats.
"""

import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/paths.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dict with configuration
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Loaded config from {config_path}")
    return config


def save_parquet(
    df: pd.DataFrame,
    output_path: Path,
    compression: str = "snappy",
    overwrite: bool = True
) -> None:
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        compression: Compression algorithm ('snappy', 'gzip', 'brotli')
        overwrite: If False, raise error if file exists
    """
    output_path = Path(output_path)
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(
        output_path,
        compression=compression,
        index=False
    )
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Saved {len(df)} rows to {output_path} "
        f"({file_size_mb:.2f} MB)"
    )


def load_parquet(
    input_path: Path,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Load DataFrame from Parquet file.
    
    Args:
        input_path: Input file path
        columns: Optional list of columns to load
        
    Returns:
        DataFrame
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    df = pd.read_parquet(input_path, columns=columns)
    
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    return df


def save_csv_with_metadata(
    df: pd.DataFrame,
    output_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """
    Save CSV with metadata header.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        metadata: Dict with metadata (will be written as comments)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write metadata as comments
        f.write("# Dataset Metadata\n")
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        f.write("\n")
        
        # Write CSV data
        df.to_csv(f, index=False)
    
    logger.info(f"Saved {len(df)} rows to {output_path} with metadata")


def read_json_lines(input_path: Path) -> pd.DataFrame:
    """
    Read JSON Lines (JSONL) format.
    
    Args:
        input_path: Path to .jsonl file
        
    Returns:
        DataFrame
    """
    df = pd.read_json(input_path, lines=True)
    logger.info(f"Loaded {len(df)} records from {input_path}")
    return df
