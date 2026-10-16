"""Visualizer and demonstration utility for NeuroScan AI Data Augmentation.

Loads real held-out brain MRI scans from the dataset, executes each
data augmentation operator, and produces an exportable side-by-side
comparison grid saved to reports/data_augmentation/augmentation_grid.png.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

from src.utils.config import get_project_root
from src.data.transforms import crop_brain_contour, get_transforms, denormalize_tensor


def run_augmentation_demonstration(output_dir: Path = None):
    """Generate side-by-side visual demonstration of all augmentation transforms."""
    root = get_project_root()
    if output_dir is None:
        output_dir = root / "reports" / "data_augmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find sample real scan
    sample_paths = [
        root / "data" / "archive" / "Testing" / "glioma" / "Te-gl_1.jpg",
        root / "data" / "archive" / "Testing" / "meningioma" / "Te-me_1.jpg",
        root / "data" / "archive" / "Testing" / "pituitary" / "Te-pi_1.jpg",
        root / "reports" / "explanations" / "case_01_correct_glioma" / "original_mri.png",
    ]
    img_path = next((p for p in sample_paths if p.exists()), None)
    if not img_path:
        raise FileNotFoundError("No benchmark sample image found to demonstrate augmentations.")

    orig_pil = Image.open(img_path).convert("RGB")
    
    # 2. Individual operators for visualization
    # Operator A: Brain Contour Cropping
    cropped_np = crop_brain_contour(np.array(orig_pil), tolerance=15)
    cropped_pil = Image.fromarray(cropped_np).resize((224, 224))

    # Operator B: Horizontal Flip
    hflip_pil = cropped_pil.transpose(Image.FLIP_LEFT_RIGHT)

    # Operator C: Rotation (+12 deg head tilt)
    rot_pil = cropped_pil.rotate(12, resample=Image.BILINEAR)

    # Operator D: Affine Scale & Shift (Zoom 1.05 + 5% translation)
    affine_pil = TF.affine(cropped_pil, angle=0, translate=[8, -8], scale=1.05, shear=0)

    # Operator E: Contrast & Brightness Jitter
    jitter_tf = transforms.ColorJitter(brightness=0.15, contrast=0.25)
    jitter_pil = jitter_tf(cropped_pil)

    # Operator F: Gaussian Blur (Patient micro-motion simulation)
    blur_tf = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.2, 1.2))
    blur_pil = blur_tf(cropped_pil)

    # Operator G: Full Pipeline with Random Erasing / Cutout
    full_tf = get_transforms(split="train", image_size=(224, 224), augmentation_level="standard")
    full_tensor = full_tf(orig_pil)
    full_np = denormalize_tensor(full_tensor)
    full_pil = Image.fromarray(full_np)

    # 3. Stitch 2x4 demonstration grid
    grid_w = 224 * 4
    grid_h = 224 * 2
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(15, 23, 42))

    cells = [
        (orig_pil.resize((224, 224)), "1. Original Input"),
        (cropped_pil, "2. Brain Contour Crop"),
        (hflip_pil, "3. Sagittal Reflection"),
        (rot_pil, "4. Head-Tilt (+12 deg)"),
        (affine_pil, "5. Bore Offset & Zoom"),
        (jitter_pil, "6. Coil Gain / Contrast"),
        (blur_pil, "7. Motion Blur Simulation"),
        (full_pil, "8. Full Augmented Sample"),
    ]

    import cv2
    for idx, (cell_pil, label) in enumerate(cells):
        row = idx // 4
        col = idx % 4
        x = col * 224
        y = row * 224

        # Convert to numpy to draw clean medical label bar
        cell_np = np.array(cell_pil)
        cv2.rectangle(cell_np, (0, 195), (224, 224), (15, 23, 42), -1)
        cv2.putText(
            cell_np,
            label,
            (6, 214),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (203, 213, 225),
            1,
            cv2.LINE_AA,
        )
        grid_img.paste(Image.fromarray(cell_np), (x, y))

    grid_path = output_dir / "augmentation_grid.png"
    grid_img.save(grid_path)
    print(f"Data augmentation comparison grid saved to: {grid_path}")
    return grid_path


if __name__ == "__main__":
    run_augmentation_demonstration()
