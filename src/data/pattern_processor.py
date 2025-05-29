"""NRHA pattern processing utilities."""
import os
import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import logging
import PyPDF2
import time

from src.utils.config import get_project_root

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternProcessor:
    """Handles downloading and processing of NRHA patterns."""

    def __init__(self):
        """Initialize the pattern processor."""
        self.patterns_dir = os.path.join(get_project_root(), 'data', 'patterns')
        os.makedirs(self.patterns_dir, exist_ok=True)
        
        # NRHA pattern URLs (direct PDF links)
        self.pattern_urls = {
            '2024': 'https://nrha.com/media/pdf/handbook/2024/patterns.pdf',
            # Add more years as needed
        }

        # Browser-like headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        # Maneuver type mappings
        self.maneuver_types = {
            'run down': 'run_down',
            'sliding stop': 'sliding_stop',
            'back up': 'backup',
            'spin': 'spin',
            'circle': 'circle',
            'lead change': 'lead_change',
            'rollback': 'rollback',
            'hesitate': 'pause'
        }

    def download_patterns(self, year: str = '2024') -> List[str]:
        """Download patterns PDF for a given year (direct link).

        Args:
            year: Year of patterns to download

        Returns:
            List of downloaded pattern file paths (single PDF)
        """
        if year not in self.pattern_urls:
            raise ValueError(f"No patterns available for year {year}")

        url = self.pattern_urls[year]
        logger.info(f"Downloading patterns PDF from {url}")

        pattern_name = f"patterns_{year}.pdf"
        output_path = os.path.join(self.patterns_dir, pattern_name)

        # Skip if file exists
        if os.path.exists(output_path):
            logger.info(f"Pattern PDF already exists: {pattern_name}")
            return [output_path]

        try:
            time.sleep(1)
            response = requests.get(url, headers=self.headers, timeout=60)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded pattern PDF: {pattern_name}")
            return [output_path]
        except requests.RequestException as e:
            logger.error(f"Error downloading pattern PDF: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def parse_maneuver(self, text: str) -> Dict:
        """Parse a maneuver description into structured data.

        Args:
            text: Maneuver description text

        Returns:
            Dictionary containing maneuver information
        """
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Determine maneuver type
        maneuver_type = None
        for key, value in self.maneuver_types.items():
            if key in text_lower:
                maneuver_type = value
                break
        
        if not maneuver_type:
            return None

        # Extract additional information
        info = {
            "type": maneuver_type,
            "description": text.strip()
        }

        # Extract speed if mentioned
        if "fast" in text_lower:
            info["expected_speed"] = "fast"
        elif "slow" in text_lower:
            info["expected_speed"] = "slow"

        # Extract direction if mentioned
        if "right" in text_lower:
            info["direction"] = "right"
        elif "left" in text_lower:
            info["direction"] = "left"

        # Extract size if mentioned
        if "large" in text_lower:
            info["size"] = "large"
        elif "small" in text_lower:
            info["size"] = "small"

        # Extract spin count if mentioned
        spin_match = re.search(r'(\d+(?:\.\d+)?)\s*spins?', text_lower)
        if spin_match:
            info["expected_spins"] = float(spin_match.group(1))

        return info

    def convert_pattern_to_json(self, pdf_path: str) -> Dict:
        """Convert a pattern PDF to JSON format.

        Args:
            pdf_path: Path to the pattern PDF

        Returns:
            Dictionary containing pattern information
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split into lines and clean up
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Extract pattern information
        pattern_name = os.path.basename(pdf_path)
        pattern_id = pattern_name.replace('.pdf', '')
        
        # Parse maneuvers
        maneuvers = []
        current_maneuver = []
        
        for line in lines:
            # Skip header lines
            if "NRHA" in line or "Pattern" in line:
                continue
                
            # Check if this is a new maneuver
            if any(keyword in line.lower() for keyword in self.maneuver_types.keys()):
                # Process previous maneuver if exists
                if current_maneuver:
                    maneuver_text = " ".join(current_maneuver)
                    maneuver_info = self.parse_maneuver(maneuver_text)
                    if maneuver_info:
                        maneuvers.append(maneuver_info)
                current_maneuver = [line]
            else:
                current_maneuver.append(line)
        
        # Process last maneuver
        if current_maneuver:
            maneuver_text = " ".join(current_maneuver)
            maneuver_info = self.parse_maneuver(maneuver_text)
            if maneuver_info:
                maneuvers.append(maneuver_info)

        return {
            "pattern_id": pattern_id,
            "name": pattern_id,
            "description": f"NRHA Pattern {pattern_id}",
            "maneuvers": maneuvers,
            "scoring": {
                "maneuver_score_range": [-1.5, 1.5],
                "total_score_range": [0, 100],
                "penalties": {
                    "-1.0": "Incorrect lead",
                    "-2.0": "Fall of horse",
                    "-3.0": "Fall of rider",
                    "-5.0": "Use of hand on saddle"
                }
            }
        }

    def process_patterns(self, year: str = '2024') -> List[Dict]:
        """Download and process all patterns for a given year.

        Args:
            year: Year of patterns to process

        Returns:
            List of processed pattern dictionaries
        """
        # Download patterns
        pdf_files = self.download_patterns(year)
        
        # Convert patterns to JSON
        patterns = []
        for pdf_file in pdf_files:
            try:
                pattern_data = self.convert_pattern_to_json(pdf_file)
                json_path = pdf_file.replace('.pdf', '.json')
                
                with open(json_path, 'w') as f:
                    json.dump(pattern_data, f, indent=2)
                
                patterns.append(pattern_data)
                logger.info(f"Processed pattern: {os.path.basename(json_path)}")
            
            except Exception as e:
                logger.error(f"Error processing pattern {pdf_file}: {e}")
                continue
        
        return patterns 