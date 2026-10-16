"""Grad-CAM (Gradient-Weighted Class Activation Mapping) for NeuroScan AI Models.

Supported Architectures:
- Custom CNN (ConvNormActBlock stage 5)
- ResNet-50 (layer4[-1] bottleneck)
- DenseNet-121 (features.norm5 / denseblock4)

CRITICAL MEDICAL & INTERPRETABILITY DISCLAIMER:
- Grad-CAM heatmaps highlight regions contributing most heavily to a model's logit activation.
- Grad-CAM activation maps are NOT tumor segmentation masks, lesion contours, or surgical resection boundaries.
- Heatmaps reflect exploratory neural visual attribution only and must be interpreted in conjunction with
  multimodal diagnostic workup by licensed board-certified neuroradiologists.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

from src.data.constants import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from src.data.transforms import get_transforms
from src.utils.device import get_device


DISCLAIMER_TEXT = (
    "Grad-CAM visual attributions highlight coarse gradient-weighted convolutional activations. "
    "They do NOT represent precise tumor segmentation masks, lesion boundaries, or tissue volumes."
)


def get_default_target_layer(model: nn.Module) -> nn.Module:
    """Identify the optimal final convolutional layer for standard NeuroScan AI architectures."""
    # Custom CNN: stage5 final conv block
    if hasattr(model, "stage5") and hasattr(model.stage5, "block"):
        # block[3] is the second Conv2d(512, 512) before pooling
        if len(model.stage5.block) > 3 and isinstance(model.stage5.block[3], nn.Conv2d):
            return model.stage5.block[3]
        return model.stage5.block

    # ResNet-50: layer4 final bottleneck block
    if hasattr(model, "layer4"):
        return model.layer4[-1]

    # DenseNet-121: final norm5 or denseblock4
    if hasattr(model, "features"):
        if hasattr(model.features, "norm5"):
            return model.features.norm5
        if hasattr(model.features, "denseblock4"):
            return model.features.denseblock4
        return model.features[-1]

    # Generic fallback: search for last Conv2d in module hierarchy
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    raise ValueError(f"Unable to auto-detect target convolutional layer for model of type {type(model)}")


class GradCAM:
    """Gradient-Weighted Class Activation Mapping engine with safe hook cleanup.
    
    Provides:
    - Target class selection (predicted or user-specified class)
    - Activation extraction & gradient backpropagation hooks
    - Dimension and numerical bounds validation [0, 1]
    - RGB blending overlay on original MRI and standalone colorized heatmaps
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[nn.Module, str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        if target_layer is None:
            self.target_layer = get_default_target_layer(model)
        elif isinstance(target_layer, str):
            # Locate named sub-module
            found_module = None
            for name, mod in model.named_modules():
                if name == target_layer:
                    found_module = mod
                    break
            if found_module is None:
                raise ValueError(f"Layer '{target_layer}' not found in model hierarchy.")
            self.target_layer = found_module
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hook on the target layer, attaching backward gradient hook on tensor."""
        self.remove_hooks()

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor):
            self.activations = out

            def save_gradient(grad: torch.Tensor):
                self.gradients = grad

            out.register_hook(save_gradient)

        self._hook_handle = self.target_layer.register_forward_hook(forward_hook)

    def remove_hooks(self) -> None:
        """Safely release and remove PyTorch forward/backward hook handles."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.activations = None
        self.gradients = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float, Dict[str, float]]:
        """Compute 2D Grad-CAM activation map for a single input tensor.
        
        Args:
            input_tensor: Image tensor of shape (1, 3, H, W)
            target_class: Target class index (0-3). If None, uses top predicted class.
            
        Returns:
            Tuple of:
            - cam: 2D float32 numpy array of shape (H, W) normalized to [0.0, 1.0]
            - target_idx: Selected class index
            - confidence: Softmax probability of target class
            - all_probs: Dict mapping class names to predicted probabilities
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.shape[0] != 1:
            raise ValueError(f"Expected single image tensor with batch size 1, got shape {input_tensor.shape}")

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

        if target_class is None:
            target_idx = pred_idx
        else:
            target_idx = int(target_class)
            if target_idx < 0 or target_idx >= logits.shape[1]:
                raise ValueError(f"Target class index {target_idx} out of range [0, {logits.shape[1] - 1}]")

        target_score = logits[0, target_idx]
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients were not captured. Verify target layer connectivity.")

        # Global average pooling of gradients: alpha_k = mean_{u,v}(dL / dA^k_{u,v})
        # shape: (1, C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps: sum_k (alpha_k * A^k)
        # shape: (1, 1, H', W')
        cam_tensor = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam_tensor = F.relu(cam_tensor)

        # Upsample CAM to input spatial dimensions (H, W)
        _, _, target_h, target_w = input_tensor.shape
        cam_upsampled = F.interpolate(
            cam_tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam_upsampled.squeeze().detach().cpu().numpy().astype(np.float32)

        # Numerical bounds normalization to [0, 1]
        cam_min, cam_max = float(np.min(cam)), float(np.max(cam))
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        cam = np.clip(cam, 0.0, 1.0)

        # Validation assertions
        assert cam.shape == (target_h, target_w), f"CAM dimension mismatch: {cam.shape} vs ({target_h}, {target_w})"
        assert not np.isnan(cam).any(), "CAM array contains NaN values"
        assert not np.isinf(cam).any(), "CAM array contains Infinite values"
        assert cam.min() >= 0.0 and cam.max() <= 1.0, f"CAM values out of bounds: [{cam.min()}, {cam.max()}]"

        all_probs = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        confidence = float(probs[target_idx])

        return cam, target_idx, confidence, all_probs

    def explain_image(
        self,
        image_input: Union[Path, str, Image.Image, np.ndarray],
        target_class: Optional[Union[int, str]] = None,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
        mean: Sequence[float] = (0.172, 0.172, 0.172),
        std: Sequence[float] = (0.1868, 0.1868, 0.1868),
    ) -> Dict[str, Any]:
        """Generate complete Grad-CAM explanation pack for an MRI image.
        
        Args:
            image_input: File path, PIL Image, or RGB numpy array
            target_class: Optional target class index (0-3) or class name string ('glioma', etc.)
            alpha: Heatmap blend opacity in overlay (0.0: raw image only, 1.0: heatmap only)
            colormap: OpenCV colormap enum (default: COLORMAP_JET)
            mean: Normalization mean for transforms
            std: Normalization std for transforms
            
        Returns:
            Dictionary containing:
            - heatmap: 2D float32 numpy array in [0, 1]
            - heatmap_colored: RGB uint8 numpy array of colored heatmap
            - overlay: RGB uint8 numpy array blending original MRI and heatmap
            - original_image: RGB uint8 numpy array (resized to 224x224)
            - target_class_idx: int
            - target_class_name: str
            - predicted_class_idx: int
            - predicted_class_name: str
            - confidence: float
            - is_predicted_class: bool
            - class_probabilities: Dict[str, float]
            - disclaimer: Medical interpretability disclaimer
        """
        # Load and convert to RGB PIL Image
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image path not found: {p}")
            pil_img = Image.open(p).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
            else:
                raise ValueError(f"Unsupported numpy image array shape: {image_input.shape}")
        else:
            raise TypeError("image_input must be a Path, string path, PIL Image, or numpy array")

        # Preprocess tensor
        transforms = get_transforms(split="val", crop_contour=True, mean=mean, std=std)
        tensor = transforms(pil_img)  # (3, 224, 224)

        # Resolve target class index
        target_idx: Optional[int] = None
        if target_class is not None:
            if isinstance(target_class, str):
                target_str = target_class.strip().lower()
                if target_str in CLASS_TO_IDX:
                    target_idx = CLASS_TO_IDX[target_str]
                else:
                    raise ValueError(f"Unknown class name '{target_class}'. Must be one of {CLASSES}")
            elif isinstance(target_class, (int, np.integer)):
                target_idx = int(target_class)
            else:
                raise TypeError(f"target_class must be int or str, got {type(target_class)}")

        # Generate CAM
        cam, resolved_target_idx, target_conf, probs = self.generate_cam(tensor, target_class=target_idx)
        pred_idx = int(np.argmax(list(probs.values())))
        pred_name = CLASSES[pred_idx]
        target_name = CLASSES[resolved_target_idx]

        # Extract base 224x224 cropped MRI for visualization
        # Resize raw image to match CAM dimensions
        h, w = cam.shape
        raw_resized = np.array(pil_img.resize((w, h), Image.Resampling.BILINEAR))

        # Colorize heatmap (OpenCV JET colormap)
        cam_uint8 = np.uint8(255 * cam)
        heatmap_colored_bgr = cv2.applyColorMap(cam_uint8, colormap)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

        # Blend overlay: (1 - alpha) * image + alpha * heatmap
        alpha = float(np.clip(alpha, 0.0, 1.0))
        blended = np.clip(
            (1.0 - alpha) * raw_resized.astype(np.float32) + alpha * heatmap_colored_rgb.astype(np.float32),
            0.0,
            255.0,
        ).astype(np.uint8)

        return {
            "heatmap": cam,
            "heatmap_colored": heatmap_colored_rgb,
            "overlay": blended,
            "original_image": raw_resized,
            "target_class_idx": resolved_target_idx,
            "target_class_name": target_name,
            "predicted_class_idx": pred_idx,
            "predicted_class_name": pred_name,
            "confidence": target_conf,
            "is_predicted_class": (resolved_target_idx == pred_idx),
            "class_probabilities": probs,
            "disclaimer": DISCLAIMER_TEXT,
        }
