"""Device and hardware query utilities for NeuroScan AI."""

import os
from typing import Dict, Any
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resolve compute device with graceful CPU fallback.
    
    Args:
        preference: 'auto', 'cuda', or 'cpu'
        
    Returns:
        torch.device instance
    """
    preference = preference.lower()
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif preference == "cpu":
        return torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")


def get_hardware_info() -> Dict[str, Any]:
    """Retrieve detailed system and GPU hardware metadata.
    
    Returns:
        Dictionary with hardware specifications and CUDA/VRAM status.
    """
    cuda_available = torch.cuda.is_available()
    info = {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "current_device": None,
        "device_name": "CPU",
        "compute_capability": None,
        "vram_total_gb": 0.0,
        "vram_allocated_mb": 0.0,
        "vram_reserved_mb": 0.0,
        "cpu_threads": os.cpu_count() or 1,
    }
    
    if cuda_available:
        current_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_idx)
        info["current_device"] = current_idx
        info["device_name"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["vram_total_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(current_idx) / (1024 ** 2), 2)
        info["vram_reserved_mb"] = round(torch.cuda.memory_reserved(current_idx) / (1024 ** 2), 2)
        
    return info
