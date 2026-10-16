"""Image preprocessing, augmentation, and transformation pipeline for NeuroScan AI.

Ensures:
- RGB conversion (3 channels)
- Skull/brain contour cropping (removes excessive black border)
- Standardized resizing to 224x224
- Tensor conversion & Channel normalization
- Clinically sound data augmentation applied to the training split ONLY:
  * Horizontal reflection along axial sagittal midline
  * Subtle rotational head-tilt simulation (+/- 12 deg)
  * Patient bore positioning translation (+/- 5%) and scale (0.95 - 1.05)
  * Contrast & brightness jitter (simulating coil gain variations)
  * Subtle Gaussian blur (simulating patient micro-motion)
  * Cutout / Random Erasing for focal occlusion robustness
"""

from typing import Tuple, Optional, Sequence
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def crop_brain_contour(image: np.ndarray, tolerance: int = 15) -> np.ndarray:
    """Crop excess black background from brain MRI scans using contour detection.
    
    Args:
        image: RGB or grayscale uint8 numpy array.
        tolerance: Pixel intensity threshold to separate brain tissue from background.
        
    Returns:
        Cropped numpy array containing the brain region.
    """
    if image is None or image.size == 0:
        return image
        
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, tolerance, 255, cv2.THRESH_BINARY)[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    min_dim = 10
    if w > min_dim and h > min_dim:
        return image[y:y+h, x:x+w]
        
    return image


class BrainContourCrop:
    """Torchvision transform for automatic brain contour cropping."""
    
    def __init__(self, tolerance: int = 15):
        self.tolerance = tolerance

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img)
        cropped_np = crop_brain_contour(np_img, self.tolerance)
        return Image.fromarray(cropped_np)


def get_transforms(
    split: str = "train",
    image_size: Tuple[int, int] = (224, 224),
    crop_contour: bool = True,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    augmentation_level: str = "standard",
) -> transforms.Compose:
    """Build standardized PyTorch transformation pipeline.
    
    Data augmentation is applied strictly to the 'train' split.
    Validation and test splits use deterministic preprocessing.
    
    Args:
        split: 'train', 'val', or 'test'
        image_size: Target (height, width) tuple (default 224x224)
        crop_contour: Whether to crop black background around skull
        mean: Normalization mean per channel (3,)
        std: Normalization standard deviation per channel (3,)
        augmentation_level: 'none', 'mild', 'standard', or 'heavy'
        
    Returns:
        torchvision.transforms.Compose pipeline
    """
    transform_list = []
    
    if crop_contour:
        transform_list.append(BrainContourCrop())
        
    transform_list.append(transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR))
    
    if split == "train" and augmentation_level != "none":
        # 1. Horizontal reflection (Anatomical axial brain symmetry holds across sagittal midline)
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        # 2. Geometric augmentations (Head tilt, bore translation, and scale)
        if augmentation_level in ["mild", "standard", "heavy"]:
            deg = 8 if augmentation_level == "mild" else (15 if augmentation_level == "heavy" else 12)
            trans = (0.03, 0.03) if augmentation_level == "mild" else (0.06, 0.06)
            scale = (0.97, 1.03) if augmentation_level == "mild" else (0.94, 1.06)
            transform_list.extend([
                transforms.RandomRotation(degrees=(-deg, deg), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomAffine(degrees=0, translate=trans, scale=scale, interpolation=transforms.InterpolationMode.BILINEAR),
            ])

        # 3. Radiometric augmentations (T1/T2 coil gain and signal variation)
        if augmentation_level in ["standard", "heavy"]:
            brightness = 0.08 if augmentation_level == "standard" else 0.15
            contrast = 0.12 if augmentation_level == "standard" else 0.20
            transform_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
            
            # Subtle Gaussian blur simulating slight patient micro-motion
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2))],
                    p=0.3
                )
            )

    # Convert to Tensor and Normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])

    # 4. Optional Random Erasing (Cutout) to prevent single-feature reliance
    if split == "train" and augmentation_level in ["standard", "heavy"]:
        transform_list.append(
            transforms.RandomErasing(
                p=0.2 if augmentation_level == "standard" else 0.35,
                scale=(0.02, 0.12),
                ratio=(0.5, 2.0),
                value="random",
            )
        )
    
    return transforms.Compose(transform_list)


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Denormalize a PyTorch tensor back to uint8 RGB numpy array.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W)
        mean: Channel means
        std: Channel stds
        
    Returns:
        uint8 numpy array of shape (H, W, C)
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
        
    t = tensor.detach().cpu().clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
        
    np_img = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return (np_img * 255.0).astype(np.uint8)
