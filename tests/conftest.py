"""Pytest fixtures for NeuroScan AI smoke tests."""

import shutil
from pathlib import Path
import pytest
import numpy as np
import cv2
from PIL import Image


def generate_synthetic_mri(size: int = 256, intensity_offset: int = 0) -> np.ndarray:
    """Generate a synthetic brain MRI slice with black background and ellipse brain contour."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    
    # Outer skull ellipse
    center = (size // 2, size // 2)
    axes = (size // 3, size // 4)
    cv2.ellipse(canvas, center, axes, 0, 0, 360, int(180 + intensity_offset), -1)
    
    # Inner tissue patterns
    cv2.circle(canvas, (size // 2 - 20, size // 2 - 10), 25, int(120 + intensity_offset), -1)
    cv2.circle(canvas, (size // 2 + 20, size // 2 + 10), 30, int(140 + intensity_offset), -1)
    
    return canvas


@pytest.fixture(scope="session")
def synthetic_mri_dir(tmp_path_factory) -> Path:
    """Creates a temporary synthetic dataset structure mimicking the Kaggle brain MRI dataset."""
    base_dir = tmp_path_factory.mktemp("brain_mri_mock_dataset")
    
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    splits = ["Training", "Testing"]
    
    # Track one image to duplicate across splits
    duplicate_image_bytes = None

    for split in splits:
        for cls in classes:
            cls_dir = base_dir / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate 4 valid scans per class per split
            for i in range(4):
                img_arr = generate_synthetic_mri(size=256, intensity_offset=i * 2)
                img_path = cls_dir / f"{split[:2]}_{cls}_{i:04d}.jpg"
                
                pil_img = Image.fromarray(img_arr)
                pil_img.save(img_path)
                
                # Capture the first training glioma image to duplicate into testing
                if split == "Training" and cls == "glioma" and i == 0:
                    with open(img_path, "rb") as f:
                        duplicate_image_bytes = f.read()

    # 1. Inject an exact cross-split duplicate (Testing/glioma/duplicate_leak.jpg)
    leak_path = base_dir / "Testing" / "glioma" / "Te_glioma_exact_leak.jpg"
    with open(leak_path, "wb") as f:
        f.write(duplicate_image_bytes)

    # 2. Inject a near-duplicate image (perturbed pixels) in Training/meningioma
    near_dup_arr = generate_synthetic_mri(size=256, intensity_offset=1)
    # Slight noise in corner
    near_dup_arr[10:15, 10:15] = 20
    near_dup_path = base_dir / "Training" / "meningioma" / "Tr_meningioma_near_dup.jpg"
    Image.fromarray(near_dup_arr).save(near_dup_path)

    # 3. Inject a corrupt image (invalid header/data) in Training/pituitary
    corrupt_path = base_dir / "Training" / "pituitary" / "corrupted_scan.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_AN_IMAGE_HEADER_CORRUPTED_BYTES_1234567890")

    # 4. Inject a zero-byte file in Testing/notumor
    zero_byte_path = base_dir / "Testing" / "notumor" / "zero_byte_file.jpg"
    with open(zero_byte_path, "wb") as f:
        pass

    return base_dir
