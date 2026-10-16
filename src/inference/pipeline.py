"""Inference execution pipeline with strict non-fabrication guarantees."""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.architectures import create_model
from src.models.readiness import ModelReadinessManager
from src.data.transforms import get_transforms
from src.utils.device import get_device
from src.utils.exceptions import ModelNotReadyError, CorruptImageError


class InferenceEngine:
    """Production inference engine with strict model-readiness enforcement.
    
    GUARANTEE:
    This engine never fabricates mock predictions, fake probabilities, or synthetic labels.
    If a trained model checkpoint does not exist on disk, it explicitly raises ModelNotReadyError.
    """

    def __init__(self, checkpoints_dir: Optional[Path] = None, device_preference: str = "auto"):
        self.readiness_manager = ModelReadinessManager(checkpoints_dir)
        self.device = get_device(device_preference)
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.transforms = get_transforms(split="val", crop_contour=True)
        self.class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

    def load_model(self, model_name: str) -> torch.nn.Module:
        """Load and cache model weights only if verified checkpoint exists."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        # Validates existence or raises ModelNotReadyError
        ckpt_path = self.readiness_manager.assert_ready(model_name)

        model = create_model(model_name=model_name, num_classes=len(self.class_names))
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.loaded_models[model_name] = model
        return model

    def predict_image(
        self,
        image_input: Union[Path, str, Image.Image],
        model_name: str = "neuro_convnet"
    ) -> Dict[str, Any]:
        """Perform classification inference on a single brain MRI scan.
        
        Args:
            image_input: File path or PIL Image object.
            model_name: Key of registered scratch CNN.
            
        Returns:
            Dictionary with prediction probabilities, predicted class, and entropy.
        """
        # 1. Enforce readiness check
        model = self.load_model(model_name)

        # 2. Process image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            try:
                with Image.open(img_path) as pil_img:
                    rgb_img = pil_img.convert("RGB")
            except Exception as e:
                raise CorruptImageError(f"Failed to read image file: {str(e)}")
        elif isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
        else:
            raise TypeError("image_input must be a file path or PIL Image")

        tensor = self.transforms(rgb_img).unsqueeze(0).to(self.device)

        # 3. Model forward pass
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        
        # Compute predictive entropy: -sum(p * log(p))
        entropy = float(-sum(p * torch.log(torch.tensor(p) + 1e-8).item() for p in probs))

        return {
            "model_name": model_name,
            "predicted_class": self.class_names[pred_idx],
            "predicted_index": pred_idx,
            "confidence": confidence,
            "entropy": entropy,
            "class_probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            },
        }
