"""Weighted soft-voting ensemble and inference engine for NeuroScan AI models."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from src.models.architectures import create_model
from src.models.readiness import ModelReadinessManager
from src.data.transforms import get_transforms
from src.data.constants import CLASSES, CLASS_TO_IDX
from src.utils.device import get_device
from src.utils.exceptions import ModelNotReadyError, CorruptImageError
from src.evaluation.uncertainty import (
    compute_predictive_entropy,
    compute_confidence_margin,
    compute_ensemble_disagreement,
)


def verify_probability_normalization(probs: np.ndarray, tol: float = 1e-4) -> bool:
    """Verify that probability array satisfies non-negativity and row-wise sum to 1."""
    if probs is None or probs.size == 0:
        raise ValueError("Empty probability array supplied")
    if (probs < -tol).any():
        raise ValueError("Probabilities contain negative values")
    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError(f"Probabilities do not sum to 1.0 (min={row_sums.min():.5f}, max={row_sums.max():.5f})")
    return True


class EnsembleInference:
    """Weighted soft-voting ensemble of independently scratch-trained CNNs.
    
    Guarantees:
    - Verifies all model checkpoints exist or raises ModelNotReadyError
    - Strict probability normalization checks
    - Quantifies predictive uncertainty and inter-model disagreement
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        checkpoints_dir: Optional[Path] = None,
        device_preference: str = "auto",
        mean: Sequence[float] = (0.172, 0.172, 0.172),
        std: Sequence[float] = (0.1868, 0.1868, 0.1868),
    ):
        self.model_names = model_names or ["custom_cnn", "resnet50", "densenet121"]
        self.readiness_manager = ModelReadinessManager(checkpoints_dir)
        self.device = get_device(device_preference)
        self.transforms = get_transforms(split="val", crop_contour=True, mean=mean, std=std)
        self.class_names = CLASSES
        
        # Load and verify all models
        self.models: Dict[str, torch.nn.Module] = {}
        for name in self.model_names:
            self._load_member_model(name)

        # Set or normalize weights
        if weights:
            total_w = sum(weights.values())
            self.weights = {k: v / total_w for k, v in weights.items()}
        else:
            # Default uniform weights
            uniform_w = 1.0 / len(self.model_names)
            self.weights = {k: uniform_w for k in self.model_names}

    def _load_member_model(self, model_name: str) -> torch.nn.Module:
        """Load single model checkpoint or raise ModelNotReadyError."""
        ckpt_path = self.readiness_manager.assert_ready(model_name)
        model = create_model(model_name, num_classes=len(self.class_names))
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.models[model_name] = model
        return model

    @classmethod
    def optimize_weights_on_validation(
        cls,
        val_probs_dict: Dict[str, np.ndarray],
        val_targets: Sequence[int],
    ) -> Dict[str, float]:
        """Find optimal ensemble weights strictly on the validation set using constrained optimization.
        
        Optimizes weights to maximize Macro-F1 / minimize cross-entropy loss without touching test data.
        
        Args:
            val_probs_dict: Mapping of model_name -> probability array (N, C)
            val_targets: Ground truth validation labels (N,)
            
        Returns:
            Dictionary mapping model_name -> optimal weight.
        """
        model_names = list(val_probs_dict.keys())
        k = len(model_names)
        targets = np.asarray(val_targets, dtype=int)
        
        # Stack probabilities: (k, N, C)
        prob_stack = np.stack([val_probs_dict[m] for m in model_names], axis=0)

        # Objective: Negative Log-Likelihood of ensemble prediction
        def objective(weights: np.ndarray) -> float:
            w = weights[:, None, None]
            ens_probs = np.sum(prob_stack * w, axis=0)
            ens_probs = np.clip(ens_probs, 1e-12, 1.0)
            # Cross-entropy loss on validation targets
            nll = -np.mean(np.log(ens_probs[np.arange(len(targets)), targets]))
            return float(nll)

        init_weights = np.ones(k) / k
        bounds = [(0.0, 1.0) for _ in range(k)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        opt_result = minimize(objective, init_weights, bounds=bounds, constraints=constraints, method="SLSQP")
        opt_weights = opt_result.x if opt_result.success else init_weights
        
        # Normalize
        opt_weights = np.maximum(opt_weights, 0.0)
        opt_weights /= np.sum(opt_weights)
        
        return {name: float(round(opt_weights[i], 4)) for i, name in enumerate(model_names)}

    def predict_batch_tensors(self, tensors: torch.Tensor) -> Dict[str, Any]:
        """Predict on batch of preprocessed tensors across all member models.
        
        Args:
            tensors: Batch of input tensors (B, 3, 224, 224)
            
        Returns:
            Dictionary containing member probabilities, ensemble probabilities, predictions, and uncertainty.
        """
        tensors = tensors.to(self.device)
        member_probs: Dict[str, np.ndarray] = {}
        member_logits: Dict[str, np.ndarray] = {}

        with torch.no_grad():
            for name, model in self.models.items():
                logits = model(tensors)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                verify_probability_normalization(probs)
                member_logits[name] = logits.cpu().numpy()
                member_probs[name] = probs

        # Compute weighted soft-voting ensemble
        weights_arr = np.array([self.weights[name] for name in self.model_names])
        stacked_probs = np.stack([member_probs[name] for name in self.model_names], axis=0)  # (M, B, C)
        ens_probs = np.sum(stacked_probs * weights_arr[:, None, None], axis=0)  # (B, C)
        
        # Normalize ensuring numerical precision
        ens_probs = ens_probs / np.sum(ens_probs, axis=1, keepdims=True)
        verify_probability_normalization(ens_probs)

        predictions = np.argmax(ens_probs, axis=1)
        confidences = np.max(ens_probs, axis=1)
        entropies = compute_predictive_entropy(ens_probs)
        margins = compute_confidence_margin(ens_probs)
        disagreement = compute_ensemble_disagreement(list(member_probs.values()))

        return {
            "predictions": predictions,
            "predicted_classes": [self.class_names[idx] for idx in predictions],
            "ensemble_probabilities": ens_probs,
            "confidences": confidences,
            "predictive_entropies": entropies,
            "confidence_margins": margins,
            "member_probabilities": member_probs,
            "member_logits": member_logits,
            "disagreement": disagreement,
            "weights": self.weights,
        }

    def predict_image(self, image_input: Union[Path, str, Image.Image]) -> Dict[str, Any]:
        """Perform ensemble inference on a single image file or PIL Image."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            try:
                with Image.open(p) as img:
                    rgb_img = img.convert("RGB")
            except Exception as e:
                raise CorruptImageError(f"Failed to read image: {str(e)}")
        elif isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
        else:
            raise TypeError("image_input must be path or PIL Image")

        tensor = self.transforms(rgb_img).unsqueeze(0)
        batch_res = self.predict_batch_tensors(tensor)

        pred_idx = int(batch_res["predictions"][0])
        pred_probs = batch_res["ensemble_probabilities"][0]
        
        return {
            "predicted_class": self.class_names[pred_idx],
            "predicted_index": pred_idx,
            "confidence": float(batch_res["confidences"][0]),
            "entropy": float(batch_res["predictive_entropies"][0]),
            "margin": float(batch_res["confidence_margins"][0]),
            "class_probabilities": {
                name: float(pred_probs[i]) for i, name in enumerate(self.class_names)
            },
            "member_predictions": {
                name: {
                    "predicted_class": self.class_names[int(np.argmax(probs[0]))],
                    "confidence": float(np.max(probs[0])),
                }
                for name, probs in batch_res["member_probabilities"].items()
            },
            "ensemble_weights": self.weights,
        }
