"""Radiomics-Style 2D Image Feature Extraction Engine for Brain MRI.

Extracts:
1. First-Order Intensity Statistics:
   - Mean, Standard Deviation, Skewness, Kurtosis
2. GLCM Texture Characteristics (Haralick features):
   - Contrast, Homogeneity, Energy, Correlation
3. Information Theory:
   - Shannon Image Entropy
4. Structural / Morphological Features:
   - Canny Edge Density
   - Threshold-derived Foreground Area, Perimeter
   - Circularity (Isoperimetric Quotient)
   - Eccentricity (Second Moments / Fitted Ellipse)
   - Box-Counting Fractal Dimension

CRITICAL MEDICAL & METHODOLOGICAL DISCLAIMER:
- These features represent 2D exploratory slice-level intensity and structural descriptors.
- They do NOT represent true 3D tumor volumes, histological subtypes, or clinically validated radiomic biomarkers.
- Feature values must be evaluated in conjunction with full 3D volumetric radiological sequences and pathology.
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence
import numpy as np
from PIL import Image
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy, regionprops, label


DISCLAIMER_TEXT = (
    "Image-level features represent exploratory 2D slice intensity and morphological descriptors. "
    "They do NOT establish true 3D tumor volume or validated clinical radiomic biomarkers."
)

FEATURE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "intensity_mean": {
        "definition": "Mean pixel luminance across the brain MRI slice.",
        "category": "First-Order Intensity",
        "range": "[0.0, 255.0]",
        "clinical_note": "Reflects baseline overall scan exposure and bulk tissue signal intensity.",
    },
    "intensity_std": {
        "definition": "Standard deviation of pixel intensities.",
        "category": "First-Order Intensity",
        "range": "[0.0, 128.0]",
        "clinical_note": "Measures intensity dispersion and macroscopic tissue contrast variability.",
    },
    "intensity_skewness": {
        "definition": "Fisher-Pearson standardized skewness coefficient.",
        "category": "First-Order Intensity",
        "range": "(-inf, +inf)",
        "clinical_note": "Quantifies asymmetry: positive skew indicates predominance of hyperintense signals.",
    },
    "intensity_kurtosis": {
        "definition": "Excess kurtosis of the pixel intensity distribution.",
        "category": "First-Order Intensity",
        "range": "[-2.0, +inf)",
        "clinical_note": "High kurtosis indicates heavy-tailed distribution with concentrated focal intensities.",
    },
    "glcm_contrast": {
        "definition": "Mean spatial Gray-Level Co-occurrence Matrix (GLCM) contrast.",
        "category": "GLCM Texture",
        "range": "[0.0, inf)",
        "clinical_note": "Reflects local textural variation between neighboring voxels; high in heterogeneous lesions.",
    },
    "glcm_homogeneity": {
        "definition": "Mean GLCM Inverse Difference Moment (IDM).",
        "category": "GLCM Texture",
        "range": "[0.0, 1.0]",
        "clinical_note": "Measures textural uniformity; high in uniform cerebrospinal fluid or homogenous white matter.",
    },
    "glcm_energy": {
        "definition": "Mean GLCM Angular Second Moment (ASM).",
        "category": "GLCM Texture",
        "range": "[0.0, 1.0]",
        "clinical_note": "Measures repetition of pixel pairs; approaches 1 for uniform constant regions.",
    },
    "glcm_correlation": {
        "definition": "Mean GLCM linear dependency of gray levels.",
        "category": "GLCM Texture",
        "range": "[-1.0, 1.0]",
        "clinical_note": "Quantifies directional linear predictability between neighboring voxel intensities.",
    },
    "image_entropy": {
        "definition": "Shannon entropy of the grayscale intensity histogram (bits/pixel).",
        "category": "Information Theory",
        "range": "[0.0, 8.0]",
        "clinical_note": "Reflects information density and randomness; higher in micro-architecturally complex tissues.",
    },
    "canny_edge_density": {
        "definition": "Fraction of pixels marked as structural edges via Canny detector.",
        "category": "Structural / Edge",
        "range": "[0.0, 1.0]",
        "clinical_note": "Measures border complexity, anatomical boundaries, and internal tumor vascular margin density.",
    },
    "foreground_area": {
        "definition": "Pixel count of the Otsu-thresholded intracranial tissue mask.",
        "category": "Morphological / Shape",
        "range": "[0, H*W]",
        "clinical_note": "Exploratory 2D area proxy; NOT a true 3D lesion volume.",
    },
    "foreground_perimeter": {
        "definition": "Boundary arc length in pixels of the dominant foreground tissue contour.",
        "category": "Morphological / Shape",
        "range": "[0.0, inf)",
        "clinical_note": "Perimeter length of the primary contour.",
    },
    "circularity": {
        "definition": "Isoperimetric quotient: 4 * pi * Area / (Perimeter^2).",
        "category": "Morphological / Shape",
        "range": "[0.0, 1.0]",
        "clinical_note": "Approaches 1.0 for circular geometries (e.g. pituitary adenomas), lower for irregular infiltrative margins.",
    },
    "eccentricity": {
        "definition": "Eccentricity of the equivalent fitted ellipse.",
        "category": "Morphological / Shape",
        "range": "[0.0, 1.0]",
        "clinical_note": "0 indicates round shape; higher values signify elongation or axis elongation.",
    },
    "box_counting_fractal_dimension": {
        "definition": "Estimated box-counting Hausdorff dimension of thresholded edges.",
        "category": "Complexity / Fractal",
        "range": "[0.0, 2.0]",
        "clinical_note": "Measures boundary fractal roughness and infiltrative edge irregularities.",
    },
}


def compute_box_counting_dimension(binary_mask: np.ndarray, max_power: int = 6) -> float:
    """Calculate the box-counting fractal dimension of a binary boundary mask.
    
    Formula: D = lim_{s -> 0} [log(N(s)) / log(1 / s)]
    Estimated via linear regression slope of log(N(s)) against -log(s).
    
    Args:
        binary_mask: 2D binary array (values > 0 represent structure)
        max_power: Maximum power of 2 for box subdivisions
        
    Returns:
        Estimated fractal dimension in [0.0, 2.0].
    """
    if binary_mask is None or binary_mask.size == 0 or np.sum(binary_mask > 0) <= 2:
        return 0.0

    min_dim = min(binary_mask.shape)
    scales = []
    counts = []

    p = 2
    while p <= min_dim // 4 and p <= (2 ** max_power):
        scales.append(p)
        p *= 2

    if len(scales) < 2:
        return 0.0

    h, w = binary_mask.shape
    for s in scales:
        h_pad = (s - h % s) % s
        w_pad = (s - w % s) % s
        if h_pad > 0 or w_pad > 0:
            padded = np.pad(binary_mask, ((0, h_pad), (0, w_pad)), mode="constant")
        else:
            padded = binary_mask

        # Subdivide into s x s boxes
        num_boxes_y = padded.shape[0] // s
        num_boxes_x = padded.shape[1] // s
        boxes = padded.reshape(num_boxes_y, s, num_boxes_x, s)
        non_empty = int(np.sum(boxes.any(axis=(1, 3))))
        counts.append(max(1, non_empty))

    # Linear fit of log(counts) vs -log(scales)
    log_inv_scales = -np.log(scales)
    log_counts = np.log(counts)

    slope, _ = np.polyfit(log_inv_scales, log_counts, 1)
    dimension = float(np.clip(slope, 0.0, 2.0))
    return float(round(dimension, 4))


class RadiomicsExtractor:
    """Rigorous 2D exploratory feature extraction engine for brain MRI slices."""

    def __init__(
        self,
        glcm_distances: Sequence[int] = (1, 2),
        glcm_angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        canny_thresholds: Tuple[int, int] = (50, 150),
    ):
        self.glcm_distances = list(glcm_distances)
        self.glcm_angles = list(glcm_angles)
        self.canny_thresholds = canny_thresholds

    def _prepare_grayscale_uint8(self, image_input: Union[Path, str, Image.Image, np.ndarray]) -> np.ndarray:
        """Standardize any input image to a 2D uint8 grayscale array."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image not found at path: {p}")
            pil_img = Image.open(p).convert("L")
            return np.array(pil_img, dtype=np.uint8)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("L"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                if image_input.dtype == np.uint8:
                    return image_input.copy()
                # Float array in [0, 1] or [0, 255]
                if image_input.max() <= 1.0:
                    return np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
                return np.clip(image_input, 0, 255).astype(np.uint8)
            elif image_input.ndim == 3:
                # RGB / BGR
                if image_input.shape[2] == 3:
                    return cv2.cvtColor(image_input.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                elif image_input.shape[2] == 1:
                    return image_input[:, :, 0].astype(np.uint8)
            raise ValueError(f"Unsupported numpy array dimensions: {image_input.shape}")
        else:
            raise TypeError("image_input must be a Path, string path, PIL Image, or numpy array")

    def extract_features(
        self,
        image_input: Union[Path, str, Image.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """Extract all 15 radiomic-style features with edge-case handling.
        
        Args:
            image_input: MRI scan slice
            
        Returns:
            Dictionary containing:
            - features: Dict mapping feature name -> float value
            - definitions: Schema with clinical notes and ranges
            - metadata: Image dimensions, foreground coverage
            - disclaimer: Medical interpretability disclaimer
        """
        img_gray = self._prepare_grayscale_uint8(image_input)
        h, w = img_gray.shape
        total_pixels = h * w

        # Handle complete empty / constant image edge cases
        is_empty = (total_pixels == 0) or bool(np.all(img_gray == 0))
        is_constant = bool(np.all(img_gray == img_gray[0, 0]))

        # -------------------------------------------------------------
        # 1. First-Order Intensity Statistics
        # -------------------------------------------------------------
        if is_empty:
            mean_val = 0.0
            std_val = 0.0
            skew_val = 0.0
            kurt_val = 0.0
        elif is_constant:
            mean_val = float(img_gray[0, 0])
            std_val = 0.0
            skew_val = 0.0
            kurt_val = 0.0
        else:
            mean_val = float(np.mean(img_gray))
            std_val = float(np.std(img_gray))
            flat = img_gray.ravel().astype(np.float64)
            skew_val = float(skew(flat)) if std_val > 1e-6 else 0.0
            kurt_val = float(kurtosis(flat)) if std_val > 1e-6 else 0.0

            # Guard against inf / nan
            skew_val = 0.0 if np.isnan(skew_val) or np.isinf(skew_val) else skew_val
            kurt_val = 0.0 if np.isnan(kurt_val) or np.isinf(kurt_val) else kurt_val

        # -------------------------------------------------------------
        # 2. GLCM Texture Characteristics
        # -------------------------------------------------------------
        if is_empty or is_constant:
            glcm_contrast = 0.0
            glcm_homogeneity = 1.0 if is_constant and not is_empty else 0.0
            glcm_energy = 1.0 if is_constant and not is_empty else 0.0
            glcm_correlation = 1.0 if is_constant and not is_empty else 0.0
        else:
            # Scale to 64 levels for robust GLCM estimation
            glcm_input = (img_gray // 4).astype(np.uint8)
            glcm = graycomatrix(
                glcm_input,
                distances=self.glcm_distances,
                angles=self.glcm_angles,
                levels=64,
                symmetric=True,
                normed=True,
            )
            glcm_contrast = float(np.mean(graycoprops(glcm, "contrast")))
            glcm_homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
            glcm_energy = float(np.mean(graycoprops(glcm, "energy")))
            corr = graycoprops(glcm, "correlation")
            # Replace NaNs in correlation if standard deviation was 0
            corr = np.nan_to_num(corr, nan=0.0)
            glcm_correlation = float(np.mean(corr))

        # -------------------------------------------------------------
        # 3. Shannon Image Entropy
        # -------------------------------------------------------------
        if is_empty or is_constant:
            ent_val = 0.0
        else:
            ent_val = float(shannon_entropy(img_gray))

        # -------------------------------------------------------------
        # 4. Canny Edge Density
        # -------------------------------------------------------------
        if is_empty or is_constant:
            canny_density = 0.0
            edge_mask = np.zeros_like(img_gray)
        else:
            edge_mask = cv2.Canny(img_gray, self.canny_thresholds[0], self.canny_thresholds[1])
            edge_count = int(np.sum(edge_mask > 0))
            canny_density = float(edge_count / total_pixels) if total_pixels > 0 else 0.0

        # -------------------------------------------------------------
        # 5. Threshold-derived Foreground Morphology
        # -------------------------------------------------------------
        if is_empty or is_constant:
            fg_area = 0.0
            fg_perimeter = 0.0
            circularity = 0.0
            eccentricity = 0.0
            fractal_dim = 0.0
            fg_mask = np.zeros_like(img_gray)
        else:
            # Otsu thresholding
            _, fg_mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            fg_area = float(np.sum(fg_mask > 0))

            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                cnt_area = float(cv2.contourArea(largest_cnt))
                fg_perimeter = float(cv2.arcLength(largest_cnt, closed=True))

                # Circularity: 4 * pi * Area / (Perimeter^2)
                if fg_perimeter > 1e-4 and cnt_area > 0:
                    circ = (4.0 * math.pi * cnt_area) / (fg_perimeter ** 2)
                    circularity = float(np.clip(circ, 0.0, 1.0))
                else:
                    circularity = 0.0

                # Eccentricity
                if len(largest_cnt) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(largest_cnt)
                        (center, (axis_a, axis_b), angle) = ellipse
                        major = max(axis_a, axis_b) / 2.0
                        minor = min(axis_a, axis_b) / 2.0
                        if major > 1e-6:
                            ecc = math.sqrt(max(0.0, 1.0 - (minor ** 2) / (major ** 2)))
                            eccentricity = float(np.clip(ecc, 0.0, 1.0))
                        else:
                            eccentricity = 0.0
                    except Exception:
                        eccentricity = 0.0
                else:
                    # Alternative via second moments with regionprops
                    labeled = label(fg_mask > 0)
                    props = regionprops(labeled)
                    if props:
                        eccentricity = float(max(p.eccentricity for p in props))
                    else:
                        eccentricity = 0.0
            else:
                fg_perimeter = 0.0
                circularity = 0.0
                eccentricity = 0.0

            # ---------------------------------------------------------
            # 6. Box-Counting Fractal Dimension
            # ---------------------------------------------------------
            # Compute on foreground boundary edges
            fractal_input = edge_mask if np.sum(edge_mask > 0) > 10 else fg_mask
            fractal_dim = compute_box_counting_dimension(fractal_input)

        features: Dict[str, float] = {
            "intensity_mean": float(round(mean_val, 4)),
            "intensity_std": float(round(std_val, 4)),
            "intensity_skewness": float(round(skew_val, 4)),
            "intensity_kurtosis": float(round(kurt_val, 4)),
            "glcm_contrast": float(round(glcm_contrast, 4)),
            "glcm_homogeneity": float(round(glcm_homogeneity, 4)),
            "glcm_energy": float(round(glcm_energy, 4)),
            "glcm_correlation": float(round(glcm_correlation, 4)),
            "image_entropy": float(round(ent_val, 4)),
            "canny_edge_density": float(round(canny_density, 6)),
            "foreground_area": float(round(fg_area, 1)),
            "foreground_perimeter": float(round(fg_perimeter, 2)),
            "circularity": float(round(circularity, 4)),
            "eccentricity": float(round(eccentricity, 4)),
            "box_counting_fractal_dimension": float(round(fractal_dim, 4)),
        }

        return {
            "features": features,
            "definitions": FEATURE_DEFINITIONS,
            "metadata": {
                "dimensions": [h, w],
                "total_pixels": total_pixels,
                "foreground_pixel_ratio": float(round(fg_area / total_pixels, 4)) if total_pixels > 0 else 0.0,
                "is_empty": is_empty,
                "is_constant": is_constant,
            },
            "disclaimer": DISCLAIMER_TEXT,
        }
