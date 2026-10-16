"""CLI utility to scan, validate, partition, and report on the Brain MRI dataset."""

import argparse
import sys
from pathlib import Path
from src.utils.config import load_config, resolve_dataset_path, get_project_root
from src.data.validator import DatasetValidator
from src.utils.seed import set_seed
from src.utils.env_report import save_environment_report


def main():
    parser = argparse.ArgumentParser(description="NeuroScan AI: Brain MRI Dataset Processor")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default=None, help="Override path to dataset root")
    parser.add_argument("--reports-dir", type=str, default=None, help="Override reports output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    # Save environment report first
    env_rep_path = save_environment_report()
    print(f"[NeuroScan AI] Environment profile recorded at: {env_rep_path}")

    # Determine dataset directory
    if args.data_dir:
        dataset_path = Path(args.data_dir).resolve()
    else:
        dataset_path = resolve_dataset_path(config)

    # Determine reports directory
    root = get_project_root()
    if args.reports_dir:
        reports_dir = Path(args.reports_dir).resolve()
    else:
        reports_dir_str = config.get("dataset", {}).get("reports_dir", "reports")
        reports_dir = (root / reports_dir_str).resolve()

    val_split_ratio = config.get("dataset", {}).get("validation_split_ratio", 0.20)
    expected_classes = config.get("dataset", {}).get("classes", ["glioma", "meningioma", "notumor", "pituitary"])
    class_aliases = config.get("dataset", {}).get("class_aliases", None)

    print(f"[NeuroScan AI] Inspecting dataset at: {dataset_path}")
    print(f"[NeuroScan AI] Output reports directory: {reports_dir}")

    validator = DatasetValidator(
        dataset_dir=dataset_path,
        expected_classes=expected_classes,
        class_aliases=class_aliases,
        val_split_ratio=val_split_ratio,
        seed=seed,
    )

    summary = validator.generate_reports(reports_dir)

    print("\n=================== NEUROSCAN AI DATASET REPORT ===================")
    if not summary.get("dataset_present"):
        print(f"Status: DATASET NOT FOUND at {dataset_path}")
        print("Reason:", summary.get("error"))
        print("\nPlease download the Kaggle dataset from:")
        print("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print(f"Extract it into: {dataset_path}")
        print("Expected layout:")
        print("  Training/ (glioma, meningioma, notumor, pituitary)")
        print("  Testing/  (glioma, meningioma, notumor, pituitary)")
        print("===================================================================\n")
        return

    print("Status: DATASET FOUND AND VALIDATED")
    print(f"Total Valid Images: {summary.get('total_valid_images')}")
    print("Class Counts:", summary.get("valid_images_per_class"))
    print(f"Corrupt / Unreadable Files: {summary.get('corrupt_files_count')}")
    print(f"Empty Directories: {len(summary.get('empty_directories', []))}")
    print("Duplicates:", summary.get("duplicate_summary"))
    print("Patient ID Available:", summary.get("patient_id_available"))
    print("Patient Limitation Statement:")
    print(" ", summary.get("patient_id_limitation_documented"))
    print(f"Manifest CSV saved to: {reports_dir / 'dataset_manifest.csv'}")
    print(f"Distribution plot saved to: {reports_dir / 'class_distribution.png'}")
    print("===================================================================\n")


if __name__ == "__main__":
    main()
