# NeuroScan AI - Explainability & Radiomics Validation Report

> [!IMPORTANT]
> **Medical Interpretability Disclaimer**:
> Grad-CAM heatmaps highlight coarse neural gradient activations and do NOT constitute tumor segmentation masks or surgical margins.
> Radiomics features represent 2D exploratory slice descriptors, NOT validated clinical biomarkers or 3D lesion volumes.

## 1. Overview of Representative Validation Cohort

This cohort presents balanced evaluations on both **correctly identified scans** and **difficult/misclassified cases** across all three scratch-trained CNN architectures.

### Case 1: case_01_correct_glioma (CORRECT)

- **Ground Truth**: `glioma` (Image: `Tr-gl_10.jpg`)
- **Exploratory Radiomics Profile**:
  - Mean Luminance: `48.6905` | GLCM Contrast: `6.6315`
  - Circularity: `0.1261` | Fractal Dimension: `1.5879` | Canny Edge Density: `0.080997`

| Model | Predicted Class | Target Confidence | Accurate? |
| :--- | :---: | :---: | :---: |
| `custom_cnn` | `glioma` | `0.6404` | **CORRECT** |
| `resnet50` | `glioma` | `0.9999` | **CORRECT** |
| `densenet121` | `glioma` | `0.9996` | **CORRECT** |

Visualizations saved:
- Original MRI: [`reports/explanations/case_01_correct_glioma/original_mri.png`](file:///C:/mri/neuroscan-ai/reports/explanations/case_01_correct_glioma/original_mri.png)
- `custom_cnn` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_01_correct_glioma/gradcam_overlay_custom_cnn.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_01_correct_glioma/heatmap_only_custom_cnn.png)
- `resnet50` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_01_correct_glioma/gradcam_overlay_resnet50.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_01_correct_glioma/heatmap_only_resnet50.png)
- `densenet121` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_01_correct_glioma/gradcam_overlay_densenet121.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_01_correct_glioma/heatmap_only_densenet121.png)

---

### Case 2: case_02_correct_meningioma (CORRECT)

- **Ground Truth**: `meningioma` (Image: `Tr-aug-me_100.jpg`)
- **Exploratory Radiomics Profile**:
  - Mean Luminance: `50.5925` | GLCM Contrast: `6.6633`
  - Circularity: `0.1973` | Fractal Dimension: `1.5268` | Canny Edge Density: `0.068565`

| Model | Predicted Class | Target Confidence | Accurate? |
| :--- | :---: | :---: | :---: |
| `custom_cnn` | `meningioma` | `0.7278` | **CORRECT** |
| `resnet50` | `meningioma` | `0.9974` | **CORRECT** |
| `densenet121` | `meningioma` | `0.9989` | **CORRECT** |

Visualizations saved:
- Original MRI: [`reports/explanations/case_02_correct_meningioma/original_mri.png`](file:///C:/mri/neuroscan-ai/reports/explanations/case_02_correct_meningioma/original_mri.png)
- `custom_cnn` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_02_correct_meningioma/gradcam_overlay_custom_cnn.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_02_correct_meningioma/heatmap_only_custom_cnn.png)
- `resnet50` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_02_correct_meningioma/gradcam_overlay_resnet50.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_02_correct_meningioma/heatmap_only_resnet50.png)
- `densenet121` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_02_correct_meningioma/gradcam_overlay_densenet121.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_02_correct_meningioma/heatmap_only_densenet121.png)

---

### Case 3: case_03_correct_no_tumor (CORRECT)

- **Ground Truth**: `no_tumor` (Image: `Tr-no_1.jpg`)
- **Exploratory Radiomics Profile**:
  - Mean Luminance: `76.0467` | GLCM Contrast: `101.0811`
  - Circularity: `0.2336` | Fractal Dimension: `1.7837` | Canny Edge Density: `0.21062`

| Model | Predicted Class | Target Confidence | Accurate? |
| :--- | :---: | :---: | :---: |
| `custom_cnn` | `no_tumor` | `0.9918` | **CORRECT** |
| `resnet50` | `no_tumor` | `0.9999` | **CORRECT** |
| `densenet121` | `no_tumor` | `0.9999` | **CORRECT** |

Visualizations saved:
- Original MRI: [`reports/explanations/case_03_correct_no_tumor/original_mri.png`](file:///C:/mri/neuroscan-ai/reports/explanations/case_03_correct_no_tumor/original_mri.png)
- `custom_cnn` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_03_correct_no_tumor/gradcam_overlay_custom_cnn.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_03_correct_no_tumor/heatmap_only_custom_cnn.png)
- `resnet50` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_03_correct_no_tumor/gradcam_overlay_resnet50.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_03_correct_no_tumor/heatmap_only_resnet50.png)
- `densenet121` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_03_correct_no_tumor/gradcam_overlay_densenet121.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_03_correct_no_tumor/heatmap_only_densenet121.png)

---

### Case 4: case_04_incorrect_glioma (INCORRECT)

- **Ground Truth**: `glioma` (Image: `Tr-gl_1017.jpg`)
- **Exploratory Radiomics Profile**:
  - Mean Luminance: `25.8722` | GLCM Contrast: `1.6412`
  - Circularity: `0.8605` | Fractal Dimension: `1.5196` | Canny Edge Density: `0.03474`

| Model | Predicted Class | Target Confidence | Accurate? |
| :--- | :---: | :---: | :---: |
| `custom_cnn` | `meningioma` | `0.5122` | **MISCLASSIFIED** |
| `resnet50` | `glioma` | `0.9933` | **CORRECT** |
| `densenet121` | `glioma` | `0.9934` | **CORRECT** |

Visualizations saved:
- Original MRI: [`reports/explanations/case_04_incorrect_glioma/original_mri.png`](file:///C:/mri/neuroscan-ai/reports/explanations/case_04_incorrect_glioma/original_mri.png)
- `custom_cnn` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_04_incorrect_glioma/gradcam_overlay_custom_cnn.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_04_incorrect_glioma/heatmap_only_custom_cnn.png)
- `resnet50` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_04_incorrect_glioma/gradcam_overlay_resnet50.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_04_incorrect_glioma/heatmap_only_resnet50.png)
- `densenet121` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_04_incorrect_glioma/gradcam_overlay_densenet121.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_04_incorrect_glioma/heatmap_only_densenet121.png)

---

### Case 5: case_05_incorrect_glioma (INCORRECT)

- **Ground Truth**: `glioma` (Image: `Tr-gl_11.jpg`)
- **Exploratory Radiomics Profile**:
  - Mean Luminance: `37.2781` | GLCM Contrast: `3.6232`
  - Circularity: `0.2778` | Fractal Dimension: `1.449` | Canny Edge Density: `0.040665`

| Model | Predicted Class | Target Confidence | Accurate? |
| :--- | :---: | :---: | :---: |
| `custom_cnn` | `meningioma` | `0.7307` | **MISCLASSIFIED** |
| `resnet50` | `glioma` | `0.9873` | **CORRECT** |
| `densenet121` | `glioma` | `0.7218` | **CORRECT** |

Visualizations saved:
- Original MRI: [`reports/explanations/case_05_incorrect_glioma/original_mri.png`](file:///C:/mri/neuroscan-ai/reports/explanations/case_05_incorrect_glioma/original_mri.png)
- `custom_cnn` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_05_incorrect_glioma/gradcam_overlay_custom_cnn.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_05_incorrect_glioma/heatmap_only_custom_cnn.png)
- `resnet50` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_05_incorrect_glioma/gradcam_overlay_resnet50.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_05_incorrect_glioma/heatmap_only_resnet50.png)
- `densenet121` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_05_incorrect_glioma/gradcam_overlay_densenet121.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_05_incorrect_glioma/heatmap_only_densenet121.png)

---

### Case 6: case_06_incorrect_meningioma (INCORRECT)

- **Ground Truth**: `meningioma` (Image: `Tr-aug-me_43.jpg`)
- **Exploratory Radiomics Profile**:
  - Mean Luminance: `53.4114` | GLCM Contrast: `6.3873`
  - Circularity: `0.3578` | Fractal Dimension: `1.5659` | Canny Edge Density: `0.077492`

| Model | Predicted Class | Target Confidence | Accurate? |
| :--- | :---: | :---: | :---: |
| `custom_cnn` | `pituitary` | `0.7886` | **MISCLASSIFIED** |
| `resnet50` | `meningioma` | `0.5679` | **CORRECT** |
| `densenet121` | `meningioma` | `0.9308` | **CORRECT** |

Visualizations saved:
- Original MRI: [`reports/explanations/case_06_incorrect_meningioma/original_mri.png`](file:///C:/mri/neuroscan-ai/reports/explanations/case_06_incorrect_meningioma/original_mri.png)
- `custom_cnn` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_06_incorrect_meningioma/gradcam_overlay_custom_cnn.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_06_incorrect_meningioma/heatmap_only_custom_cnn.png)
- `resnet50` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_06_incorrect_meningioma/gradcam_overlay_resnet50.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_06_incorrect_meningioma/heatmap_only_resnet50.png)
- `densenet121` Grad-CAM Overlay: [`overlay`](file:///C:/mri/neuroscan-ai/reports/explanations/case_06_incorrect_meningioma/gradcam_overlay_densenet121.png) | Heatmap-only: [`heatmap`](file:///C:/mri/neuroscan-ai/reports/explanations/case_06_incorrect_meningioma/heatmap_only_densenet121.png)

---

