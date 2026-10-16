# NeuroScan AI - Exploratory Data Analysis (EDA) Report

## 1. Executive Summary

- **Dataset Path**: `C:\mri\neuroscan-ai\data\archive`
- **Total Valid Scans**: `7200`
- **Target Classes**: `glioma, meningioma, no_tumor, pituitary`
- **Splitting Strategy**: Stratified 80% Train, 20% Val carved from Training pool; Testing pool strictly isolated.

## 2. Class Distribution Across Splits

| Tumor Class | Train Split | Validation Split | Test Split | Total Scans |
| :--- | :---: | :---: | :---: | :---: |
| `glioma` | 1120 | 280 | 400 | **1800** |
| `meningioma` | 1120 | 280 | 400 | **1800** |
| `no_tumor` | 1120 | 280 | 400 | **1800** |
| `pituitary` | 1120 | 280 | 400 | **1800** |

## 3. Training Set Empirical Normalization Statistics

Computed channel-wise statistics from training split images only (preventing test set leakage):

- **Training RGB Mean**: `[0.172, 0.172, 0.172]`
- **Training RGB Std**: `[0.1868, 0.1868, 0.1868]`

## 4. Image Dimensions & Storage Profile

- **Resolution Range**: `150x167` to `1375x1446`
- **Median Resolution**: `512x512`
- **Standardized Model Input**: `224x224x3` (Brain contour cropped & resized)

## 5. Duplicate & Cross-Split Contamination Audit

- **Exact Duplicate Groups (MD5)**: `153`
- **Exact Cross-Split Contamination**: `78` pairs
- **Near-Duplicate Flags (`dHash`)**: `79771` pairs
- **Near-Duplicate Cross-Split Pairs**: `42684` pairs

## 6. Generalization & Patient Identifier Limitation

> [!WARNING]
> CRITICAL CLINICAL GENERALIZATION NOTE: The Masoud Nickparvar Kaggle Brain Tumor MRI dataset consists of pre-sliced 2D images (e.g., Tr-gl_0010.jpg, Te-me_0023.jpg) lacking clinical subject identifiers or DICOM metadata tags. Because slices from the same subject cannot be grouped by patient, image-level separation does NOT establish patient-level generalization. Performance on this benchmark represents slice-level categorization accuracy; perceptual hashing and near-duplicate cross-split checks are enforced to mitigate direct slice leakage.

## 7. Data Augmentation Protocol

- **Applied to Train Split Only**:
  - Random Horizontal Flip (`p=0.5`)
  - Random Subtle Rotation (`degrees=10`)
  - Random Affine Translation (`translate=0.04`)
- **Validation and Testing Splits**:
  - Zero stochastic augmentation; strictly deterministic contour crop, resize, and normalization.
