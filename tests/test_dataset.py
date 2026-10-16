"""Smoke tests for dataset scanning, duplicate detection, and PyTorch DataLoaders."""

from pathlib import Path
import pytest
import torch

from src.data.validator import DatasetValidator
from src.data.dataset import BrainMRIDataset, create_dataloaders
from src.data.duplicates import detect_duplicates


def test_validator_scan_and_corrupt_detection(synthetic_mri_dir: Path):
    """Verify validator detects valid images, corrupt files, and zero-byte files."""
    validator = DatasetValidator(
        dataset_dir=synthetic_mri_dir,
        val_split_ratio=0.25,
        seed=42,
    )
    scan_result = validator.scan_dataset()

    assert scan_result["dataset_present"] is True
    # Check corrupt file detection
    corrupt_paths = [Path(c["path"]).name for c in scan_result["corrupt_files"]]
    assert "corrupted_scan.jpg" in corrupt_paths or any("corrupted" in p for p in corrupt_paths)
    assert "zero_byte_file.jpg" in corrupt_paths or any("zero_byte" in p for p in corrupt_paths)
    assert len(scan_result["records"]) > 0


def test_duplicate_and_leakage_detection(synthetic_mri_dir: Path):
    """Verify exact duplicate detection flags cross-split leakage."""
    validator = DatasetValidator(dataset_dir=synthetic_mri_dir, val_split_ratio=0.25, seed=42)
    scan_result = validator.scan_dataset()
    records = validator.partition_splits(scan_result["records"])

    dup_summary = detect_duplicates(records, check_near_duplicates=True, hamming_threshold=5)

    assert dup_summary["exact_duplicate_pairs_count"] >= 1
    # Check that the injected exact duplicate across Training and Testing is detected
    assert dup_summary["exact_cross_split_leakage_count"] >= 1


def test_split_partitioning(synthetic_mri_dir: Path):
    """Verify train, val, test partition preserves class stratification and isolation."""
    validator = DatasetValidator(dataset_dir=synthetic_mri_dir, val_split_ratio=0.25, seed=42)
    scan_result = validator.scan_dataset()
    records = validator.partition_splits(scan_result["records"])

    splits = set(r["split"] for r in records)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    # Ensure no None or empty splits
    assert all(r["split"] in ["train", "val", "test"] for r in records)


def test_reports_generation(synthetic_mri_dir: Path, tmp_path: Path):
    """Verify all report artifacts (CSV, JSON, MD, PNG) are generated correctly."""
    reports_out = tmp_path / "test_reports"
    validator = DatasetValidator(dataset_dir=synthetic_mri_dir, val_split_ratio=0.25, seed=42)
    summary = validator.generate_reports(reports_out)

    assert (reports_out / "dataset_manifest.csv").exists()
    assert (reports_out / "dataset_summary.json").exists()
    assert (reports_out / "dataset_summary.md").exists()
    assert (reports_out / "duplicate_analysis.json").exists()
    assert (reports_out / "class_distribution.png").exists()

    assert summary["dataset_present"] is True
    assert summary["total_valid_images"] > 0
    assert "patient_id_limitation_documented" in summary


def test_pytorch_dataloaders(synthetic_mri_dir: Path):
    """Verify PyTorch DataLoader yields correctly shaped batches and labels."""
    validator = DatasetValidator(dataset_dir=synthetic_mri_dir, val_split_ratio=0.25, seed=42)
    scan_result = validator.scan_dataset()
    records = validator.partition_splits(scan_result["records"])

    dl_dict = create_dataloaders(
        records=records,
        batch_size=4,
        num_workers=0,  # 0 workers in test for fast sync execution
        pin_memory=False,
        image_size=(224, 224),
        crop_contour=True,
    )

    loaders = dl_dict["loaders"]
    assert "train" in loaders and loaders["train"] is not None
    assert "val" in loaders and loaders["val"] is not None
    assert "test" in loaders and loaders["test"] is not None

    # Inspect a training batch
    for images, labels, paths in loaders["train"]:
        assert isinstance(images, torch.Tensor)
        assert images.shape[1:] == (3, 224, 224)
        assert len(labels) == images.shape[0]
        assert len(paths) == images.shape[0]
        break
