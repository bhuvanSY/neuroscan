# NeuroScan AI - Multi-Model Evaluation & Benchmark Report

## 1. Executive Summary & Comparative Matrix

- **Test Cohort**: `1,600` untouched brain MRI scans (400 per class strictly quarantined from training).
- **Best Single Architecture**: `RESNET50` (Macro-F1: `0.9508`)
- **Ensemble Impact**: **Equivalent Performance** (delta = 0.00% Macro-F1). Ensemble provides robust uncertainty estimates without degradation.

### Benchmark Performance Table

| model           |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   macro_roc_auc |   ece_raw |   ece_calibrated |   latency_ms |   f1_glioma |   f1_meningioma |   f1_no_tumor |   f1_pituitary |
|:----------------|-----------:|------------------:|---------------:|-----------:|----------------:|----------:|-----------------:|-------------:|------------:|----------------:|--------------:|---------------:|
| custom_cnn      |     0.8656 |            0.8745 |         0.8656 |     0.8619 |          0.9703 |    0.0445 |           0.046  |         1.54 |      0.8286 |          0.8032 |        0.8966 |         0.9191 |
| resnet50        |     0.9519 |            0.9549 |         0.9519 |     0.9508 |          0.9844 |    0.0377 |           0.0353 |         2.44 |      0.9071 |          0.9467 |        0.9604 |         0.9889 |
| densenet121     |     0.9456 |            0.9489 |         0.9456 |     0.9444 |          0.9858 |    0.0382 |           0.0364 |         2.56 |      0.8971 |          0.9348 |        0.9581 |         0.9876 |
| efficientnet_v2 |     0.9419 |            0.9458 |         0.9419 |     0.9406 |          0.9874 |    0.0402 |           0.0386 |         2.25 |      0.8898 |          0.9366 |        0.9434 |         0.9926 |
| ensemble        |     0.9519 |            0.9552 |         0.9519 |     0.9508 |          0.9883 |    0.0297 |           0.0297 |         8.79 |      0.9056 |          0.9446 |        0.9592 |         0.9938 |

## 2. Model Disagreement & Inter-Model Variance

- **Categorical Disagreement Rate**: `12.69%` (203 of 1600 scans)
- **Mean Pairwise Jensen-Shannon Divergence**: `0.0350`
- **Mean Prediction Variance**: `0.0076`
- **Validation-Selected Ensemble Weights**: `{'custom_cnn': 0.0, 'resnet50': 0.586, 'densenet121': 0.2449, 'efficientnet_v2': 0.1691}`

## 3. Calibration & Temperature Scaling Analysis

| Model | Uncalibrated ECE | Learned Temperature ($T$) | Calibrated ECE |
| :--- | :---: | :---: | :---: |
| `custom_cnn` | 0.0445 | 0.987 | 0.0460 |
| `resnet50` | 0.0377 | 1.072 | 0.0353 |
| `densenet121` | 0.0382 | 1.073 | 0.0364 |
| `efficientnet_v2` | 0.0402 | 1.058 | 0.0386 |

## 4. Class-Specific Breakdown (Precision / Recall / F1)

### CUSTOM_CNN

| Tumor Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `glioma` | 0.9667 | 0.7250 | 0.8286 | 400 |
| `meningioma` | 0.8515 | 0.7600 | 0.8032 | 400 |
| `no_tumor` | 0.8143 | 0.9975 | 0.8966 | 400 |
| `pituitary` | 0.8653 | 0.9800 | 0.9191 | 400 |

### RESNET50

| Tumor Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `glioma` | 1.0000 | 0.8300 | 0.9071 | 400 |
| `meningioma` | 0.9178 | 0.9775 | 0.9467 | 400 |
| `no_tumor` | 0.9238 | 1.0000 | 0.9604 | 400 |
| `pituitary` | 0.9780 | 1.0000 | 0.9889 | 400 |

### DENSENET121

| Tumor Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `glioma` | 0.9939 | 0.8175 | 0.8971 | 400 |
| `meningioma` | 0.9042 | 0.9675 | 0.9348 | 400 |
| `no_tumor` | 0.9195 | 1.0000 | 0.9581 | 400 |
| `pituitary` | 0.9779 | 0.9975 | 0.9876 | 400 |

### EFFICIENTNET_V2

| Tumor Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `glioma` | 0.9908 | 0.8075 | 0.8898 | 400 |
| `meningioma` | 0.9143 | 0.9600 | 0.9366 | 400 |
| `no_tumor` | 0.8929 | 1.0000 | 0.9434 | 400 |
| `pituitary` | 0.9852 | 1.0000 | 0.9926 | 400 |

### ENSEMBLE

| Tumor Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `glioma` | 1.0000 | 0.8275 | 0.9056 | 400 |
| `meningioma` | 0.9116 | 0.9800 | 0.9446 | 400 |
| `no_tumor` | 0.9217 | 1.0000 | 0.9592 | 400 |
| `pituitary` | 0.9877 | 1.0000 | 0.9938 | 400 |

