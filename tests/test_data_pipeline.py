"""Tests for data pipeline: label mapping, image loading, transforms, and split integrity."""

from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image

from src.data.constants import CLASS_TO_IDX, IDX_TO_CLASS, CLASS_ALIASES, CLASSES
from src.data.dataset import BrainMRIDataset, create_dataloaders
from src.data.transforms import get_transforms, crop_brain_contour
from src.data.validator import DatasetValidator, compute_training_channel_stats
from tests.conftest import generate_synthetic_mri


def test_fixed_label_mapping():
    """Verify fixed, deterministic class-to-index mapping and alias resolution."""
    assert len(CLASS_TO_IDX) == 4
    assert CLASS_TO_IDX["glioma"] == 0
    assert CLASS_TO_IDX["meningioma"] == 1
    assert CLASS_TO_IDX["no_tumor"] == 2
    assert CLASS_TO_IDX["pituitary"] == 3

    # Check bidirectional consistency
    for name, idx in CLASS_TO_IDX.items():
        assert IDX_TO_CLASS[idx] == name

    # Check alias canonicalization
    assert CLASS_ALIASES["notumor"] == "no_tumor"
    assert CLASS_ALIASES["no_tumor"] == "no_tumor"


def test_image_loading_and_rgb_conversion(tmp_path: Path):
    """Verify image loading converts grayscale slices into 3-channel RGB tensors."""
    # Create single-channel grayscale slice
    gray_arr = np.zeros((100, 100), dtype=np.uint8)
    gray_arr[20:80, 20:80] = 180
    img_path = tmp_path / "gray_slice.jpg"
    Image.fromarray(gray_arr).save(img_path)

    records = [{"path": str(img_path), "label": "glioma"}]
    tf = get_transforms(split="val", image_size=(224, 224), crop_contour=False)
    dataset = BrainMRIDataset(records=records, transform=tf, class_to_idx=CLASS_TO_IDX)

    tensor, label_idx, path = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert label_idx == 0
    assert path == str(img_path)


def test_transforms_train_vs_val():
    """Verify train transforms include augmentation while validation transforms are deterministic."""
    train_tf = get_transforms(split="train", image_size=(224, 224))
    val_tf = get_transforms(split="val", image_size=(224, 224))

    # Inspect transform operations
    train_tf_names = [type(t).__name__ for t in train_tf.transforms]
    val_tf_names = [type(t).__name__ for t in val_tf.transforms]

    assert "RandomHorizontalFlip" in train_tf_names
    assert "RandomRotation" in train_tf_names
    assert "RandomHorizontalFlip" not in val_tf_names
    assert "RandomRotation" not in val_tf_names


def test_training_set_normalization_stats(tmp_path: Path):
    """Verify channel-wise statistics computation from training samples only."""
    sample_records = []
    for i in range(5):
        arr = np.full((50, 50, 3), fill_value=int(50 + i * 20), dtype=np.uint8)
        p = tmp_path / f"scan_{i}.jpg"
        Image.fromarray(arr).save(p)
        sample_records.append({"path": str(p), "split": "train"})

    stats = compute_training_channel_stats(sample_records)
    assert "mean" in stats and len(stats["mean"]) == 3
    assert "std" in stats and len(stats["std"]) == 3
    assert all(0.0 <= m <= 1.0 for m in stats["mean"])
    assert all(0.0 <= s <= 1.0 for s in stats["std"])


def test_split_integrity_and_isolation(synthetic_mri_dir: Path):
    """Verify that test split samples are strictly quarantined from train and val splits."""
    validator = DatasetValidator(dataset_dir=synthetic_mri_dir, val_split_ratio=0.20, seed=42)
    scan_res = validator.scan_dataset()
    records = validator.partition_splits(scan_res["records"])

    test_paths = set(r["path"] for r in records if r["split"] == "test")
    train_paths = set(r["path"] for r in records if r["split"] == "train")
    val_paths = set(r["path"] for r in records if r["split"] == "val")

    # Strict disjunction
    assert len(test_paths.intersection(train_paths)) == 0
    assert len(test_paths.intersection(val_paths)) == 0
    assert len(train_paths.intersection(val_paths)) == 0


def test_patient_generalization_limitation_documented(synthetic_mri_dir: Path, tmp_path: Path):
    """Verify that the EDA report explicitly documents the patient generalization limitation."""
    reports_out = tmp_path / "eda_test"
    validator = DatasetValidator(dataset_dir=synthetic_mri_dir, val_split_ratio=0.20, seed=42)
    validator.generate_reports(reports_out)

    eda_file = reports_out / "eda_report.md"
    assert eda_file.exists()
    content = eda_file.read_text(encoding="utf-8")
    assert "CRITICAL CLINICAL GENERALIZATION NOTE" in content
    assert "patient-level generalization" in content
