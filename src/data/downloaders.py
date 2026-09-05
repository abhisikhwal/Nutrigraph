"""
Reusable download utilities for fetching datasets from APIs and bulk sources.
"""

import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import time
import yaml

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Base class for downloading datasets with common utilities:
    - Rate limiting
    - Retry logic
    - Progress tracking
    - Metadata logging
    """
    
    def __init__(self, dataset_name: str, config_path: str = "config/datasets.yaml"):
        """
        Args:
            dataset_name: Name of dataset in config file
            config_path: Path to datasets.yaml
        """
        self.dataset_name = dataset_name
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        if dataset_name not in config:
            raise ValueError(f"Dataset {dataset_name} not found in {config_path}")
        
        self.config = config[dataset_name]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Global-Food-Genome-Project/0.1.0 (Research)'
        })
        
        logger.info(f"Initialized downloader for {dataset_name}")
    
    def download_file(
        self,
        url: str,
        output_path: Path,
        chunk_size: int = 8192,
        overwrite: bool = False
    ) -> Path:
        """
        Download a file from URL to disk.
        
        Args:
            url: Download URL
            output_path: Where to save the file
            chunk_size: Download chunk size in bytes
            overwrite: If False, skip if file exists
            
        Returns:
            Path to downloaded file
        """
        output_path = Path(output_path)
        
        if output_path.exists() and not overwrite:
            logger.info(f"File already exists: {output_path}")
            return output_path
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {url} to {output_path}")
        
        response = self.session.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        logger.debug(f"Progress: {pct:.1f}%")
        
        logger.info(f"Download complete: {output_path}")
        return output_path
    
    def api_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        rate_limit_delay: float = 0.2
    ) -> Dict[str, Any]:
        """
        Make an API request with rate limiting.
        
        Args:
            endpoint: API endpoint URL
            params: Query parameters
            rate_limit_delay: Delay between requests in seconds
            
        Returns:
            JSON response as dict
        """
        time.sleep(rate_limit_delay)
        
        logger.debug(f"API request: {endpoint}")
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def verify_license(self) -> None:
        """
        Check that license is acceptable for use.
        Raises warning if non-commercial or unclear.
        """
        license_type = self.config.get('license', 'UNKNOWN')
        commercial_use = self.config.get('commercial_use')
        
        if commercial_use is False:
            logger.warning(
                f"{self.dataset_name} has non-commercial license: {license_type}"
            )
        elif commercial_use is None:
            logger.warning(
                f"{self.dataset_name} license unclear: {license_type}. "
                "Verify before use!"
            )
        else:
            logger.info(f"{self.dataset_name} license: {license_type} (commercial OK)")


# Example usage:
# downloader = DatasetDownloader("usda_fooddata")
# downloader.verify_license()
# downloader.download_file(url, output_path)
