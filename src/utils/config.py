"""Configuration management for NeuroScan AI."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def get_project_root() -> Path:
    """Determine the project root directory reliably."""
    # src/utils/config.py -> parents[2] is project root
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration with environment variable override support.
    
    Precedence:
      1. Explicit config_path passed as parameter
      2. Environment variable NEUROSCAN_CONFIG
      3. Default configs/dataset_config.yaml in project root
      
    Environment Variable Overrides:
      - NEUROSCAN_DATA_DIR or DATASET_PATH overrides dataset.raw_dir
      - NEUROSCAN_DEVICE overrides hardware.device
      
    Args:
        config_path: Optional path to YAML config file.
        
    Returns:
        Loaded configuration dictionary.
    """
    root = get_project_root()
    
    if not config_path:
        config_path = os.environ.get("NEUROSCAN_CONFIG", str(root / "configs" / "dataset_config.yaml"))
        
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = root / cfg_file
        
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")
        
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Environment variable overrides
    env_data_dir = os.environ.get("NEUROSCAN_DATA_DIR") or os.environ.get("DATASET_PATH")
    if env_data_dir:
        config.setdefault("dataset", {})["raw_dir"] = env_data_dir

    env_device = os.environ.get("NEUROSCAN_DEVICE")
    if env_device:
        config.setdefault("hardware", {})["device"] = env_device

    return config


def resolve_dataset_path(config: Dict[str, Any]) -> Path:
    """Resolve the dataset path flexibly (relative or absolute, checking env vars).
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Resolved Path object.
    """
    root = get_project_root()
    raw_dir_str = config.get("dataset", {}).get("raw_dir", "data/archive")
    
    path = Path(raw_dir_str)
    if not path.is_absolute():
        path = (root / path).resolve()
        
    has_env = bool(os.environ.get("NEUROSCAN_DATA_DIR") or os.environ.get("DATASET_PATH"))
    if not path.exists() and not has_env and raw_dir_str == "data/raw/brain_tumor_mri":
        # Fallback candidate checks
        candidates = [
            root / "data" / "archive",
            root / "data",
        ]
        for cand in candidates:
            if (cand / "Training").exists() and (cand / "Testing").exists():
                return cand.resolve()

    return path
