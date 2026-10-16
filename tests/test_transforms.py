"""Smoke tests for image preprocessing and transformation pipeline."""

import numpy as np
import torch
from PIL import Image

from src.data.transforms import (
    crop_brain_contour,
    get_transforms,
    denormalize_tensor,
    BrainContourCrop,
)
from tests.conftest import generate_synthetic_mri


def test_crop_brain_contour():
    """Verify brain contour cropping trims black borders correctly."""
    # Create 256x256 image with black border and ellipse brain
    synthetic = generate_synthetic_mri(size=256)
    cropped = crop_brain_contour(synthetic, tolerance=15)
    
    assert cropped.shape[0] < 256 or cropped.shape[1] < 256
    assert cropped.shape[0] > 50
    assert cropped.shape[1] > 50


def test_get_transforms_output_shape():
    """Verify transforms output correct shape and tensor type for train and val splits."""
    synthetic = generate_synthetic_mri(size=256)
    pil_img = Image.fromarray(synthetic).convert("RGB")

    train_tf = get_transforms(split="train", image_size=(224, 224), crop_contour=True)
    val_tf = get_transforms(split="val", image_size=(224, 224), crop_contour=True)

    train_tensor = train_tf(pil_img)
    val_tensor = val_tf(pil_img)

    assert isinstance(train_tensor, torch.Tensor)
    assert isinstance(val_tensor, torch.Tensor)
    assert train_tensor.shape == (3, 224, 224)
    assert val_tensor.shape == (3, 224, 224)


def test_denormalize_tensor():
    """Verify tensor denormalization reconstitutes uint8 RGB array."""
    val_tf = get_transforms(split="val", image_size=(224, 224), crop_contour=False)
    pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    tensor = val_tf(pil_img)
    
    reconstituted = denormalize_tensor(tensor)
    assert isinstance(reconstituted, np.ndarray)
    assert reconstituted.dtype == np.uint8
    assert reconstituted.shape == (224, 224, 3)
    # Check values are centered around original gray level ~128
    assert np.allclose(reconstituted.mean(), 128, atol=10)
