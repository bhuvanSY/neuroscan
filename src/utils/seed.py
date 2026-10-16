"""Seed and reproducibility utilities for NeuroScan AI."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across python, numpy, torch CPU and CUDA.
    
    Args:
        seed: Integer seed value (default: 42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Worker init function to guarantee deterministic PyTorch DataLoader multi-processing.
    
    Args:
        worker_id: DataLoader worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
