"""Smoke tests for Python environment, PyTorch, hardware, and determinism."""

import sys
import torch
import numpy as np

from src.utils.seed import set_seed
from src.utils.device import get_device, get_hardware_info


def test_python_version():
    """Verify supported Python version."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 10, f"Expected Python >= 3.10, got {sys.version}"


def test_hardware_query():
    """Verify hardware detection and device properties."""
    info = get_hardware_info()
    assert "cuda_available" in info
    assert "device_name" in info
    assert "vram_total_gb" in info
    assert info["cpu_threads"] >= 1

    device = get_device("auto")
    assert isinstance(device, torch.device)
    
    # Test fallback
    cpu_device = get_device("cpu")
    assert cpu_device.type == "cpu"


def test_seed_determinism():
    """Verify that set_seed ensures reproducible random numbers across numpy and torch."""
    set_seed(42)
    np_rand_1 = np.random.rand(5)
    torch_rand_1 = torch.rand(5)

    set_seed(42)
    np_rand_2 = np.random.rand(5)
    torch_rand_2 = torch.rand(5)

    np.testing.assert_allclose(np_rand_1, np_rand_2)
    assert torch.allclose(torch_rand_1, torch_rand_2)


def test_tensor_operations():
    """Verify basic tensor operations execute properly on chosen device."""
    device = get_device("auto")
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 4, device=device)
    z = torch.matmul(x, y)
    assert z.shape == (4, 4)
    assert not torch.isnan(z).any()
