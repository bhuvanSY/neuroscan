# NeuroScan AI - Dataset Summary & Health Report

- **Dataset Path**: `C:\mri\neuroscan-ai\data\archive`
- **Total Valid Images**: `7200`
- **Corrupt / Unreadable Files**: `0`
- **Empty Directories**: `0`
- **Training Set Mean**: `[0.172, 0.172, 0.172]`
- **Training Set Std**: `[0.1868, 0.1868, 0.1868]`

## Class Distribution

| Tumor Class | Count |
| :--- | :--- |
| `glioma` | 1800 |
| `meningioma` | 1800 |
| `no_tumor` | 1800 |
| `pituitary` | 1800 |

## Patient ID Limitation Statement

CRITICAL CLINICAL GENERALIZATION NOTE: The Masoud Nickparvar Kaggle Brain Tumor MRI dataset consists of pre-sliced 2D images (e.g., Tr-gl_0010.jpg, Te-me_0023.jpg) lacking clinical subject identifiers or DICOM metadata tags. Because slices from the same subject cannot be grouped by patient, image-level separation does NOT establish patient-level generalization. Performance on this benchmark represents slice-level categorization accuracy; perceptual hashing and near-duplicate cross-split checks are enforced to mitigate direct slice leakage.
