"""Configuration loading utilities."""
import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the configuration file is invalid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    return config


def get_project_root() -> str:
    """Get the project root directory.

    Returns:
        Absolute path to the project root directory.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 