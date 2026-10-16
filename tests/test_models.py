"""Smoke tests for scratch CNN architectures."""

import torch
import pytest
from src.models.architectures import (
    CustomCNN,
    build_resnet50_scratch,
    build_densenet121_scratch,
    create_model,
    MODEL_BUILDERS,
)


@pytest.mark.parametrize("model_name", ["custom_cnn", "resnet50", "densenet121"])
def test_scratch_cnn_forward_pass(model_name: str):
    """Verify all 3 scratch CNNs instantiate and compute forward pass without pretrained weights."""
    model = create_model(model_name=model_name, in_channels=3, num_classes=4)
    model.eval()

    # Batch of 2 MRI slices: (B, C, H, W)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 4)
    assert not torch.isnan(output).any()


def test_invalid_model_name_raises():
    """Verify invalid model name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model name"):
        create_model("unregistered_vgg")
