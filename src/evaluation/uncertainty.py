"""Uncertainty estimation, model disagreement, and temperature scaling calibration."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import jensenshannon


def compute_predictive_entropy(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Calculate predictive Shannon entropy per sample: H(P) = -sum(p * log(p)).
    
    Args:
        probs: Array of probabilities of shape (N, C)
        eps: Small constant to avoid log(0)
        
    Returns:
        Entropy values of shape (N,)
    """
    probs = np.clip(probs, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return entropy


def compute_confidence_margin(probs: np.ndarray) -> np.ndarray:
    """Calculate margin of confidence: difference between top-1 and top-2 predicted probability."""
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return margin


def compute_ensemble_disagreement(member_probs: List[np.ndarray]) -> Dict[str, Any]:
    """Measure disagreement across individual models on the same evaluation cohort.
    
    Args:
        member_probs: List of M probability matrices, each of shape (N, C)
        
    Returns:
        Dictionary containing disagreement metrics:
        - categorical_disagreement_rate: Fraction of samples where models predict different classes
        - mean_jensen_shannon_divergence: Mean pairwise Jensen-Shannon divergence
        - mean_prediction_variance: Variance of predicted probabilities across models
    """
    if len(member_probs) < 2:
        return {
            "categorical_disagreement_rate": 0.0,
            "mean_jensen_shannon_divergence": 0.0,
            "mean_prediction_variance": 0.0,
        }

    # Ensure 2D shape (N, C)
    member_probs = [np.asarray(p, dtype=float) for p in member_probs]
    member_probs = [p[None, :] if p.ndim == 1 else p for p in member_probs]

    num_samples = member_probs[0].shape[0]
    num_models = len(member_probs)
    
    # 1. Categorical disagreement rate
    member_preds = [np.argmax(p, axis=1) for p in member_probs]  # M arrays of shape (N,)
    stacked_preds = np.stack(member_preds, axis=1)  # (N, M)
    
    # Sample has disagreement if not all models predict the same class
    disagreement_mask = np.apply_along_axis(lambda row: len(np.unique(row)) > 1, axis=1, arr=stacked_preds)
    disagreement_rate = float(np.mean(disagreement_mask))

    # 2. Mean pairwise Jensen-Shannon divergence
    js_divergences = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            # Compute JS divergence per sample
            p_i = np.clip(member_probs[i], 1e-10, 1.0)
            p_j = np.clip(member_probs[j], 1e-10, 1.0)
            # jensenshannon computes along last axis
            js_dist = jensenshannon(p_i, p_j, axis=1)  # JS distance in [0, 1]
            js_div = js_dist ** 2  # JS divergence
            js_divergences.append(np.mean(js_div))

    mean_js_div = float(np.mean(js_divergences)) if js_divergences else 0.0

    # 3. Variance across models
    stacked_probs = np.stack(member_probs, axis=0)  # (M, N, C)
    prob_variance = np.var(stacked_probs, axis=0)   # (N, C)
    mean_variance = float(np.mean(prob_variance))

    return {
        "categorical_disagreement_rate": disagreement_rate,
        "mean_jensen_shannon_divergence": mean_js_div,
        "mean_prediction_variance": mean_variance,
        "disagreement_count": int(np.sum(disagreement_mask)),
        "total_samples": num_samples,
    }


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling calibration module (Guo et al., 2017).
    
    Calibrated probabilities: P_calib = softmax(logits / T), with T > 0.
    Fit strictly on validation logits via NLL loss minimization, then applied to test set.
    """

    def __init__(self):
        super().__init__()
        # Initialize temperature parameter T = 1.0
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """Optimize temperature parameter on validation set only.
        
        Args:
            val_logits: Tensor of shape (N, C)
            val_labels: Integer class labels of shape (N,)
            max_iter: Maximum optimization iterations
            
        Returns:
            Learned optimal temperature scalar T.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        # Ensure temperature remains strictly positive
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)
            
        return float(self.temperature.item())
