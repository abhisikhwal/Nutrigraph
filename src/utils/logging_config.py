"""
Logging configuration for the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the project.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers
    )
    
    # Suppress overly verbose libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {level} level")
    
    if log_file:
        logger.info(f"Logging to file: {log_file}")


# Example usage:
# from food_genome.utils import setup_logging
# setup_logging(level="INFO", log_file="logs/pipeline.log")
