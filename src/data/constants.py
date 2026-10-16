"""Constants and fixed schema definitions for NeuroScan AI dataset pipeline."""

from typing import Dict, List

# Standardized 4-class taxonomy
CLASSES: List[str] = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]

# Fixed, deterministic class-to-index mapping
CLASS_TO_IDX: Dict[str, int] = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
    "pituitary": 3,
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

# Canonical alias mapping handling Kaggle folder variations (e.g. 'notumor' vs 'no_tumor')
CLASS_ALIASES: Dict[str, str] = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "no_tumor",
    "no_tumor": "no_tumor",
    "no tumor": "no_tumor",
    "normal": "no_tumor",
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Patient ID limitation documentation statement
PATIENT_GENERALIZATION_LIMITATION_NOTE: str = (
    "CRITICAL CLINICAL GENERALIZATION NOTE: The Masoud Nickparvar Kaggle Brain Tumor MRI dataset "
    "consists of pre-sliced 2D images (e.g., Tr-gl_0010.jpg, Te-me_0023.jpg) lacking clinical subject "
    "identifiers or DICOM metadata tags. Because slices from the same subject cannot be grouped by patient, "
    "image-level separation does NOT establish patient-level generalization. Performance on this benchmark "
    "represents slice-level categorization accuracy; perceptual hashing and near-duplicate cross-split checks "
    "are enforced to mitigate direct slice leakage."
)
