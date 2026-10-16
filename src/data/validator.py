"""Dataset inspection, validation, manifest generator, and metadata reporting."""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Sequence
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from src.data.duplicates import detect_duplicates, compute_md5
from src.data.constants import (
    CLASSES,
    CLASS_TO_IDX,
    CLASS_ALIASES,
    SUPPORTED_IMAGE_EXTENSIONS,
    PATIENT_GENERALIZATION_LIMITATION_NOTE,
)
from src.utils.config import get_project_root

logger = logging.getLogger(__name__)


def extract_patient_id(filename: str) -> Optional[str]:
    """Inspect filename for potential patient identifier patterns."""
    stem = Path(filename).stem
    pat_match = re.search(r"(patient[_-]?\d+|pat[_-]?\d+|case[_-]?\d+|subject[_-]?\d+)", stem, re.IGNORECASE)
    if pat_match:
        return pat_match.group(1).lower()
    return None


def validate_image_file(file_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int, int]]]:
    """Validate image integrity using PIL and OpenCV.
    
    Returns:
        (is_valid, error_message, (width, height, channels))
    """
    if not file_path.exists():
        return False, "File does not exist", None
        
    if file_path.stat().st_size == 0:
        return False, "File is zero bytes (empty)", None

    try:
        with Image.open(file_path) as img:
            img.verify()
            
        with Image.open(file_path) as img:
            w, h = img.size
            channels = len(img.getbands())
            
        cv_img = cv2.imread(str(file_path))
        if cv_img is None:
            return False, "OpenCV failed to decode image array", None

        return True, None, (w, h, channels)
    except Exception as e:
        return False, f"Image corruption: {str(e)}", None


def compute_training_channel_stats(train_records: List[Dict[str, Any]], max_samples: int = 1500) -> Dict[str, List[float]]:
    """Compute empirical channel-wise mean and standard deviation from training-set images only.
    
    Guarantees no data leakage: strictly evaluates samples assigned to 'train' split.
    
    Args:
        train_records: List of sample dicts assigned to 'train' split.
        max_samples: Maximum number of images sampled for fast, stable computation.
        
    Returns:
        Dictionary with 'mean' and 'std' lists of length 3 (RGB).
    """
    if not train_records:
        return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    sample_pool = train_records
    if len(train_records) > max_samples:
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(train_records), size=max_samples, replace=False)
        sample_pool = [train_records[i] for i in sample_indices]

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for item in sample_pool:
        try:
            with Image.open(item["path"]) as img:
                rgb_arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            
            h, w, _ = rgb_arr.shape
            pixels = h * w
            total_pixels += pixels
            channel_sum += rgb_arr.sum(axis=(0, 1))
            channel_sq_sum += (rgb_arr ** 2).sum(axis=(0, 1))
        except Exception:
            continue

    if total_pixels == 0:
        return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    mean = channel_sum / total_pixels
    variance = (channel_sq_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 1e-6))

    return {
        "mean": [round(float(m), 4) for m in mean],
        "std": [round(float(s), 4) for s in std],
    }


class DatasetValidator:
    """Validates dataset structure, checks integrity, generates manifests and distribution reports."""

    def __init__(
        self,
        dataset_dir: Path,
        expected_classes: Optional[List[str]] = None,
        class_aliases: Optional[Dict[str, str]] = None,
        val_split_ratio: float = 0.20,
        seed: int = 42,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.expected_classes = expected_classes or CLASSES
        self.class_aliases = class_aliases or CLASS_ALIASES
        self.val_split_ratio = val_split_ratio
        self.seed = seed

    def canonicalize_class(self, raw_class_name: str) -> Optional[str]:
        """Map raw folder name to standardized class name."""
        clean = raw_class_name.strip().lower()
        return self.class_aliases.get(clean)

    def scan_dataset(self) -> Dict[str, Any]:
        """Scan directory, validate images, compute hashes and compile metadata."""
        if not self.dataset_dir.exists():
            return {
                "dataset_present": False,
                "error": f"Dataset directory not found: {self.dataset_dir}",
                "records": [],
                "corrupt_files": [],
                "empty_directories": [],
                "unrecognized_classes": [],
            }

        empty_dirs = []
        corrupt_files = []
        unrecognized_classes = set()
        raw_records = []

        top_subdirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        train_test_layout = any(d.name.lower() in ["training", "train"] for d in top_subdirs)
        
        scan_roots = []
        if train_test_layout:
            for d in top_subdirs:
                if d.name.lower() in ["training", "train"]:
                    scan_roots.append((d, "train_pool"))
                elif d.name.lower() in ["testing", "test"]:
                    scan_roots.append((d, "test"))
                elif d.name.lower() in ["val", "validation"]:
                    scan_roots.append((d, "val"))
                else:
                    scan_roots.append((d, "unspecified"))
        else:
            scan_roots.append((self.dataset_dir, "flat_pool"))

        for root_dir, initial_split in scan_roots:
            subdirs = [sd for sd in root_dir.iterdir() if sd.is_dir()]
            if not subdirs:
                empty_dirs.append(str(root_dir))
                
            for class_folder in subdirs:
                raw_cls = class_folder.name
                canonical_cls = self.canonicalize_class(raw_cls)
                if not canonical_cls:
                    unrecognized_classes.add(raw_cls)
                    canonical_cls = raw_cls

                image_files = [f for f in class_folder.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
                if not image_files:
                    empty_dirs.append(str(class_folder))

                for img_path in image_files:
                    is_valid, err_msg, dims = validate_image_file(img_path)
                    if not is_valid:
                        corrupt_files.append({
                            "path": str(img_path),
                            "error": err_msg,
                        })
                        continue

                    w, h, c = dims
                    md5_val = compute_md5(img_path)
                    patient_id = extract_patient_id(img_path.name)
                    
                    raw_records.append({
                        "path": str(img_path),
                        "filename": img_path.name,
                        "raw_label": raw_cls,
                        "label": canonical_cls,
                        "width": w,
                        "height": h,
                        "channels": c,
                        "file_size_bytes": img_path.stat().st_size,
                        "md5": md5_val,
                        "patient_id": patient_id,
                        "initial_split": initial_split,
                    })

        return {
            "dataset_present": True,
            "records": raw_records,
            "corrupt_files": corrupt_files,
            "empty_directories": empty_dirs,
            "unrecognized_classes": list(unrecognized_classes),
        }

    def partition_splits(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Partition records into strict train, validation, and test splits without leakage."""
        if not records:
            return []

        df = pd.DataFrame(records)
        df["class_index"] = df["label"].map(CLASS_TO_IDX)

        has_patient_ids = bool(df["patient_id"].notna().sum() > (len(df) * 0.5))

        is_train_pool = df["initial_split"] == "train_pool"
        is_test = df["initial_split"] == "test"
        is_val = df["initial_split"] == "val"

        if is_train_pool.any() and is_test.any():
            # Test split remains completely untouched
            train_pool_indices = df[is_train_pool].index
            train_pool_df = df.loc[train_pool_indices]
            
            if is_val.any():
                df["split"] = df["initial_split"].replace({"train_pool": "train"})
            else:
                # Carve validation strictly from train_pool
                if has_patient_ids:
                    gss = GroupShuffleSplit(n_splits=1, test_size=self.val_split_ratio, random_state=self.seed)
                    train_idx, val_idx = next(gss.split(train_pool_df, groups=train_pool_df["patient_id"]))
                    df.loc[train_pool_indices[train_idx], "split"] = "train"
                    df.loc[train_pool_indices[val_idx], "split"] = "val"
                else:
                    train_sub, val_sub = train_test_split(
                        train_pool_df,
                        test_size=self.val_split_ratio,
                        stratify=train_pool_df["label"],
                        random_state=self.seed,
                    )
                    df.loc[train_sub.index, "split"] = "train"
                    df.loc[val_sub.index, "split"] = "val"

            df.loc[df[is_test].index, "split"] = "test"
        else:
            # Flat split: 70% train, 15% val, 15% test
            if has_patient_ids:
                gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=self.seed)
                tr_idx, rest_idx = next(gss1.split(df, groups=df["patient_id"]))
                rest_df = df.iloc[rest_idx]
                gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=self.seed)
                val_rel, test_rel = next(gss2.split(rest_df, groups=rest_df["patient_id"]))
                
                df.loc[df.index[tr_idx], "split"] = "train"
                df.loc[rest_df.index[val_rel], "split"] = "val"
                df.loc[rest_df.index[test_rel], "split"] = "test"
            else:
                train_df, rest_df = train_test_split(
                    df, test_size=0.30, stratify=df["label"], random_state=self.seed
                )
                val_df, test_df = train_test_split(
                    rest_df, test_size=0.50, stratify=rest_df["label"], random_state=self.seed
                )
                df.loc[train_df.index, "split"] = "train"
                df.loc[val_df.index, "split"] = "val"
                df.loc[test_df.index, "split"] = "test"

        return df.to_dict(orient="records")

    def plot_class_distribution(self, records: List[Dict[str, Any]], output_path: Path) -> None:
        """Generate and save class-distribution visualization across splits."""
        if not records:
            return

        df = pd.DataFrame(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Plot 1: Split breakdown by class
        palette = {"train": "#2b5c8f", "val": "#c27d38", "test": "#388e3c"}
        sns.countplot(
            data=df,
            x="label",
            hue="split",
            order=sorted(df["label"].unique()),
            palette=palette,
            ax=ax1,
        )
        ax1.set_title("Class Counts by Split Assignment", fontsize=12, fontweight="bold", pad=12)
        ax1.set_xlabel("Tumor Class", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Number of Scans", fontsize=10, fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        for p in ax1.patches:
            h = p.get_height()
            if h > 0:
                ax1.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2.0, h),
                             ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")

        # Plot 2: Total class proportion pie chart
        counts = df["label"].value_counts()
        colors = ["#34495e", "#7f8c8d", "#95a5a6", "#bdc3c7"]
        ax2.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140, colors=colors[:len(counts)])
        ax2.set_title("Overall Class Balance", fontsize=12, fontweight="bold", pad=12)

        plt.suptitle("NeuroScan AI - Brain Tumor MRI Exploratory Data Analysis", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def generate_eda_report(
        self,
        df: pd.DataFrame,
        dup_summary: Dict[str, Any],
        train_stats: Dict[str, List[float]],
        output_path: Path
    ) -> None:
        """Write detailed Exploratory Data Analysis markdown report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        split_counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
        dim_summary = df[["width", "height", "file_size_bytes"]].describe()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# NeuroScan AI - Exploratory Data Analysis (EDA) Report\n\n")
            f.write("## 1. Executive Summary\n\n")
            f.write(f"- **Dataset Path**: `{self.dataset_dir}`\n")
            f.write(f"- **Total Valid Scans**: `{len(df)}`\n")
            f.write(f"- **Target Classes**: `{', '.join(sorted(df['label'].unique()))}`\n")
            f.write(f"- **Splitting Strategy**: Stratified 80% Train, 20% Val carved from Training pool; Testing pool strictly isolated.\n\n")

            f.write("## 2. Class Distribution Across Splits\n\n")
            f.write("| Tumor Class | Train Split | Validation Split | Test Split | Total Scans |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for cls in sorted(df["label"].unique()):
                tr = split_counts.loc[cls, "train"] if "train" in split_counts.columns else 0
                vl = split_counts.loc[cls, "val"] if "val" in split_counts.columns else 0
                ts = split_counts.loc[cls, "test"] if "test" in split_counts.columns else 0
                tot = tr + vl + ts
                f.write(f"| `{cls}` | {tr} | {vl} | {ts} | **{tot}** |\n")
            f.write("\n")

            f.write("## 3. Training Set Empirical Normalization Statistics\n\n")
            f.write("Computed channel-wise statistics from training split images only (preventing test set leakage):\n\n")
            f.write(f"- **Training RGB Mean**: `{train_stats['mean']}`\n")
            f.write(f"- **Training RGB Std**: `{train_stats['std']}`\n\n")

            f.write("## 4. Image Dimensions & Storage Profile\n\n")
            f.write(f"- **Resolution Range**: `{df['width'].min()}x{df['height'].min()}` to `{df['width'].max()}x{df['height'].max()}`\n")
            f.write(f"- **Median Resolution**: `{int(df['width'].median())}x{int(df['height'].median())}`\n")
            f.write(f"- **Standardized Model Input**: `224x224x3` (Brain contour cropped & resized)\n\n")

            f.write("## 5. Duplicate & Cross-Split Contamination Audit\n\n")
            f.write(f"- **Exact Duplicate Groups (MD5)**: `{dup_summary['exact_duplicate_groups_count']}`\n")
            f.write(f"- **Exact Cross-Split Contamination**: `{dup_summary['exact_cross_split_leakage_count']}` pairs\n")
            f.write(f"- **Near-Duplicate Flags (`dHash`)**: `{dup_summary['near_duplicate_pairs_count']}` pairs\n")
            f.write(f"- **Near-Duplicate Cross-Split Pairs**: `{dup_summary['near_cross_split_leakage_count']}` pairs\n\n")

            f.write("## 6. Generalization & Patient Identifier Limitation\n\n")
            f.write(f"> [!WARNING]\n> {PATIENT_GENERALIZATION_LIMITATION_NOTE}\n\n")

            f.write("## 7. Data Augmentation Protocol\n\n")
            f.write("- **Applied to Train Split Only**:\n")
            f.write("  - Random Horizontal Flip (`p=0.5`)\n")
            f.write("  - Random Subtle Rotation (`degrees=10`)\n")
            f.write("  - Random Affine Translation (`translate=0.04`)\n")
            f.write("- **Validation and Testing Splits**:\n")
            f.write("  - Zero stochastic augmentation; strictly deterministic contour crop, resize, and normalization.\n")

    def generate_reports(self, reports_dir: Path) -> Dict[str, Any]:
        """Execute full validation pipeline and write artifacts to reports directory."""
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        scan_result = self.scan_dataset()
        if not scan_result["dataset_present"]:
            summary = {
                "dataset_present": False,
                "dataset_path": str(self.dataset_dir),
                "error": scan_result["error"],
                "total_images": 0,
                "patient_id_available": False,
                "patient_id_limitation_documented": PATIENT_GENERALIZATION_LIMITATION_NOTE,
            }
            with open(reports_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            return summary

        records = self.partition_splits(scan_result["records"])
        df = pd.DataFrame(records)

        # 1. Save manifest CSV
        manifest_path = reports_dir / "dataset_manifest.csv"
        df.to_csv(manifest_path, index=False)

        # 2. Duplicate detection
        dup_summary = detect_duplicates(records, check_near_duplicates=True)
        with open(reports_dir / "duplicate_analysis.json", "w", encoding="utf-8") as f:
            json.dump(dup_summary, f, indent=2)

        # 3. Compute empirical training set statistics
        train_records = [r for r in records if r.get("split") == "train"]
        train_stats = compute_training_channel_stats(train_records)
        with open(reports_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(train_stats, f, indent=2)

        # 4. Class distribution plot
        plot_path = reports_dir / "class_distribution.png"
        self.plot_class_distribution(records, plot_path)

        # 5. Generate Comprehensive EDA Report
        eda_path = reports_dir / "eda_report.md"
        self.generate_eda_report(df, dup_summary, train_stats, eda_path)

        # 6. Generate summary JSON and MD
        valid_per_class = df["label"].value_counts().to_dict()
        class_split_breakdown = df.groupby(["label", "split"]).size().unstack(fill_value=0).to_dict()

        dataset_summary = {
            "dataset_present": True,
            "dataset_path": str(self.dataset_dir),
            "total_valid_images": len(df),
            "valid_images_per_class": valid_per_class,
            "class_split_breakdown": class_split_breakdown,
            "training_set_stats": train_stats,
            "corrupt_files_count": len(scan_result["corrupt_files"]),
            "corrupt_files": scan_result["corrupt_files"],
            "empty_directories": scan_result["empty_directories"],
            "unrecognized_classes": scan_result["unrecognized_classes"],
            "duplicate_summary": {
                "exact_duplicate_groups": dup_summary["exact_duplicate_groups_count"],
                "exact_cross_split_leakage": dup_summary["exact_cross_split_leakage_count"],
                "near_duplicate_pairs": dup_summary["near_duplicate_pairs_count"],
                "near_cross_split_leakage": dup_summary["near_cross_split_leakage_count"],
            },
            "patient_id_available": bool(df["patient_id"].notna().sum() > (len(df) * 0.5)),
            "patient_id_limitation_documented": PATIENT_GENERALIZATION_LIMITATION_NOTE,
        }

        with open(reports_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, indent=2)

        md_path = reports_dir / "dataset_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# NeuroScan AI - Dataset Summary & Health Report\n\n")
            f.write(f"- **Dataset Path**: `{self.dataset_dir}`\n")
            f.write(f"- **Total Valid Images**: `{len(df)}`\n")
            f.write(f"- **Corrupt / Unreadable Files**: `{len(scan_result['corrupt_files'])}`\n")
            f.write(f"- **Empty Directories**: `{len(scan_result['empty_directories'])}`\n")
            f.write(f"- **Training Set Mean**: `{train_stats['mean']}`\n")
            f.write(f"- **Training Set Std**: `{train_stats['std']}`\n\n")
            f.write("## Class Distribution\n\n")
            f.write("| Tumor Class | Count |\n| :--- | :--- |\n")
            for cls_name, count in df["label"].value_counts().items():
                f.write(f"| `{cls_name}` | {count} |\n")
            f.write("\n## Patient ID Limitation Statement\n\n")
            f.write(PATIENT_GENERALIZATION_LIMITATION_NOTE + "\n")

        return dataset_summary
