"""Architecture, output-shape, parameter-count, and gradient-flow tests for scratch CNNs."""

import pytest
import torch
import torch.nn as nn
from src.models.architectures import (
    CustomCNN,
    build_resnet50_scratch,
    build_densenet121_scratch,
    build_efficientnet_v2_scratch,
    create_model,
)
from src.training.engine import count_parameters

MODEL_KEYS = ["custom_cnn", "resnet50", "densenet121", "efficientnet_v2"]



@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_model_instantiation_from_scratch(model_key: str):
    """Verify that models instantiate strictly from scratch with 4 outputs."""
    model = create_model(model_key, num_classes=4, in_channels=3)
    assert isinstance(model, nn.Module)

    total_params, trainable_params = count_parameters(model)
    assert total_params > 500_000
    assert total_params == trainable_params, "All model parameters should be trainable in scratch training"


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_output_shape(model_key: str):
    """Verify forward pass output shape is strictly (batch_size, 4)."""
    model = create_model(model_key, num_classes=4, in_channels=3)
    model.eval()

    # Batch of 4 scans with dimension 224x224
    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (4, 4)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_gradient_flow(model_key: str):
    """Verify healthy backpropagation and gradient flow through all trainable layers."""
    model = create_model(model_key, num_classes=4, in_channels=3)
    model.train()

    dummy_input = torch.randn(4, 3, 224, 224, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    outputs = model(dummy_input)
    loss = criterion(outputs, targets)
    loss.backward()

    # Check input gradient flow
    assert dummy_input.grad is not None
    assert not torch.isnan(dummy_input.grad).any()

    # Verify gradients across all learnable parameters
    unvisited_grads = 0
    zero_grad_layers = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
            grad_norm = param.grad.data.norm(2).item()
            if grad_norm == 0.0:
                zero_grad_layers += 1

    # In healthy deep networks, at least 95% of layers should have non-zero active gradients
    total_layers = len([p for p in model.parameters() if p.requires_grad])
    active_ratio = (total_layers - zero_grad_layers) / total_layers
    assert active_ratio > 0.95, f"Poor gradient flow in {model_key}: active gradient ratio={active_ratio:.2f}"
