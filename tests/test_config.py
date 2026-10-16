"""Smoke tests for configuration loader and environment overrides."""

import os
from pathlib import Path
from src.utils.config import load_config, resolve_dataset_path, get_project_root


def test_load_default_config():
    """Verify default dataset_config.yaml loads properly."""
    cfg = load_config()
    assert cfg["project_name"] == "neuroscan-ai"
    assert "dataset" in cfg
    assert "classes" in cfg["dataset"]
    assert "preprocessing" in cfg


def test_env_var_data_dir_override(monkeypatch):
    """Verify NEUROSCAN_DATA_DIR environment variable overrides raw_dir."""
    test_custom_dir = "custom/test/path"
    monkeypatch.setenv("NEUROSCAN_DATA_DIR", test_custom_dir)
    
    cfg = load_config()
    assert cfg["dataset"]["raw_dir"] == test_custom_dir
    
    resolved = resolve_dataset_path(cfg)
    assert resolved.parts[-3:] == ("custom", "test", "path")
