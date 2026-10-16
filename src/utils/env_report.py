"""Environment and hardware inspection report generator."""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import importlib.metadata
import torch

from src.utils.device import get_hardware_info
from src.utils.config import get_project_root


def inspect_environment() -> Dict[str, Any]:
    """Collect comprehensive environment and dependency metadata."""
    packages = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "pillow",
        "opencv-python",
        "matplotlib",
        "seaborn",
        "scikit-image",
        "pyyaml",
        "pytest",
    ]
    
    pkg_versions = {}
    for pkg in packages:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pkg_versions[pkg] = "not installed"

    hardware = get_hardware_info()
    
    report = {
        "python_version": sys.version,
        "platform": sys.platform,
        "pytorch_version": torch.__version__,
        "hardware": hardware,
        "dependencies": pkg_versions,
    }
    return report


def save_environment_report(output_path: Path = None) -> Path:
    """Save the environment inspection report to disk."""
    if output_path is None:
        root = get_project_root()
        output_path = root / "reports" / "environment_info.json"
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = inspect_environment()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    return output_path


if __name__ == "__main__":
    out = save_environment_report()
    print(f"Environment report saved to: {out}")
    with open(out, "r") as f:
        print(f.read())
