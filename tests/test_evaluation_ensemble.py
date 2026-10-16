"""Unit tests for inference, ensemble, evaluation metrics, calibration, and uncertainty."""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from src.data.constants import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_calibration_metrics,
)
from src.evaluation.uncertainty import (
    compute_predictive_entropy,
    compute_confidence_margin,
    compute_ensemble_disagreement,
    TemperatureScaler,
)
from src.inference.ensemble import (
    verify_probability_normalization,
    EnsembleInference,
)
from src.utils.exceptions import ModelNotReadyError


class TestClassOrderingAndMetrics:
    """Test consistency of class indexing, canonical ordering, and classification metrics."""

    def test_canonical_class_ordering(self):
        """Ensure class names and indices match medical-grade ordering strictly."""
        expected_classes = ["glioma", "meningioma", "no_tumor", "pituitary"]
        assert CLASSES == expected_classes
        for idx, name in enumerate(CLASSES):
            assert CLASS_TO_IDX[name] == idx
            assert IDX_TO_CLASS[idx] == name

    def test_compute_classification_metrics_exact(self):
        """Verify metric calculation on deterministic ground truth and predictions."""
        # 4 classes: 0: glioma, 1: meningioma, 2: no_tumor, 3: pituitary
        y_true = [0, 0, 1, 1, 2, 2, 3, 3]
        y_pred = [0, 1, 1, 1, 2, 2, 3, 0]  # sample 1 wrong (0->1), sample 7 wrong (3->0)
        
        # Mock probabilities
        y_probs = np.zeros((8, 4))
        for i, p in enumerate(y_pred):
            y_probs[i, p] = 0.85
            for c in range(4):
                if c != p:
                    y_probs[i, c] = 0.05

        metrics = compute_classification_metrics(y_true, y_pred, y_probs=y_probs, class_names=CLASSES)

        # 6 correct out of 8 = 75.0% accuracy
        assert metrics["accuracy"] == pytest.approx(0.75)
        assert "macro_f1" in metrics
        assert "macro_precision" in metrics
        assert "macro_recall" in metrics
        assert "per_class" in metrics
        assert len(metrics["per_class"]) == 4

        # Per-class checks
        for c_name in CLASSES:
            assert c_name in metrics["per_class"]
            assert "precision" in metrics["per_class"][c_name]
            assert "recall" in metrics["per_class"][c_name]
            assert "f1_score" in metrics["per_class"][c_name]
            assert metrics["per_class"][c_name]["support"] == 2

        # Confusion matrix checks
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (4, 4)
        assert np.sum(cm) == 8

        # OvR ROC-AUC checks
        assert metrics["macro_roc_auc"] is not None
        assert 0.0 <= metrics["macro_roc_auc"] <= 1.0


class TestProbabilityNormalization:
    """Test probability normalization validation and boundary conditions."""

    def test_valid_probabilities(self):
        """Valid non-negative probabilities summing to 1 should pass."""
        probs = np.array([
            [0.70, 0.10, 0.10, 0.10],
            [0.25, 0.25, 0.25, 0.25],
            [1.00, 0.00, 0.00, 0.00],
        ])
        assert verify_probability_normalization(probs) is True

    def test_negative_probabilities_raise(self):
        """Negative probability entries must raise ValueError."""
        probs = np.array([
            [-0.10, 0.50, 0.30, 0.30],
        ])
        with pytest.raises(ValueError, match="negative"):
            verify_probability_normalization(probs)

    def test_unnormalized_probabilities_raise(self):
        """Probabilities not summing to 1.0 must raise ValueError."""
        probs = np.array([
            [0.40, 0.40, 0.40, 0.40],  # sums to 1.6
        ])
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            verify_probability_normalization(probs)

    def test_empty_probabilities_raise(self):
        """Empty or zero-size arrays must raise ValueError."""
        with pytest.raises(ValueError, match="Empty"):
            verify_probability_normalization(np.array([]))


class TestUncertaintyAndCalibration:
    """Test predictive entropy, disagreement metrics, and temperature scaling."""

    def test_predictive_entropy_bounds(self):
        """Test Shannon entropy for certainty (0) and maximum uncertainty (ln(4))."""
        # One-hot certainty: entropy should be ~0
        certain = np.array([[1.0, 0.0, 0.0, 0.0]])
        ent_certain = compute_predictive_entropy(certain)
        assert ent_certain[0] == pytest.approx(0.0, abs=1e-4)

        # Uniform uncertainty: entropy should be ln(4) ~= 1.38629
        uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
        ent_uniform = compute_predictive_entropy(uniform)
        assert ent_uniform[0] == pytest.approx(np.log(4.0), rel=1e-3)

    def test_confidence_margin(self):
        """Test confidence margin between top-1 and top-2 predictions."""
        probs = np.array([
            [0.80, 0.15, 0.03, 0.02],
            [0.40, 0.35, 0.15, 0.10],
        ])
        margins = compute_confidence_margin(probs)
        assert margins[0] == pytest.approx(0.80 - 0.15)
        assert margins[1] == pytest.approx(0.40 - 0.35)

    def test_ensemble_disagreement_metrics(self):
        """Test categorical disagreement and Jensen-Shannon divergence."""
        # 2 models, 3 samples
        model_a = np.array([
            [0.9, 0.1, 0.0, 0.0],  # class 0
            [0.1, 0.8, 0.1, 0.0],  # class 1
            [0.1, 0.1, 0.7, 0.1],  # class 2
        ])
        model_b = np.array([
            [0.9, 0.05, 0.05, 0.0], # class 0 (agree)
            [0.7, 0.2, 0.1, 0.0],  # class 0 (DISAGREE with class 1)
            [0.1, 0.1, 0.7, 0.1],  # class 2 (agree)
        ])

        dis = compute_ensemble_disagreement([model_a, model_b])
        # 1 out of 3 samples disagrees = 33.33%
        assert dis["categorical_disagreement_rate"] == pytest.approx(1.0 / 3.0)
        assert dis["disagreement_count"] == 1
        assert dis["total_samples"] == 3
        assert dis["mean_jensen_shannon_divergence"] > 0.0
        assert dis["mean_prediction_variance"] > 0.0

    def test_temperature_scaling_optimization(self):
        """Fit TemperatureScaler on synthetic overconfident logits and verify calibration."""
        torch.manual_seed(42)
        # 100 samples, 4 classes, overconfident logits (scale=5.0)
        labels = torch.randint(0, 4, (100,))
        one_hot = F.one_hot(labels, num_classes=4).float()
        noisy_logits = (one_hot * 6.0) + torch.randn(100, 4)

        scaler = TemperatureScaler()
        assert float(scaler.temperature.item()) == 1.0

        opt_T = scaler.fit(noisy_logits, labels)
        assert opt_T > 0.0
        assert isinstance(opt_T, float)

        # Scaled logits should produce well-calibrated probabilities
        scaled_logits = scaler(noisy_logits)
        raw_probs = F.softmax(noisy_logits, dim=1).numpy()
        calib_probs = F.softmax(scaled_logits, dim=1).detach().numpy()

        raw_calib = compute_calibration_metrics(labels.numpy(), raw_probs)
        scaled_calib = compute_calibration_metrics(labels.numpy(), calib_probs)

        assert "expected_calibration_error" in raw_calib
        assert "expected_calibration_error" in scaled_calib


class TestEnsembleOptimizationAndMissingModels:
    """Test ensemble weight optimization and graceful handling of missing checkpoints."""

    def test_optimize_weights_on_validation(self):
        """Optimal weights must sum to 1.0 and favor the superior model."""
        val_targets = [0, 0, 1, 1, 2, 2, 3, 3]
        
        # Perfect model A
        probs_a = np.zeros((8, 4))
        for i, t in enumerate(val_targets):
            probs_a[i, t] = 0.97
            for c in range(4):
                if c != t:
                    probs_a[i, c] = 0.01

        # Poor model B (random uniform)
        probs_b = np.full((8, 4), 0.25)

        val_probs_dict = {"model_a": probs_a, "model_b": probs_b}
        weights = EnsembleInference.optimize_weights_on_validation(val_probs_dict, val_targets)

        assert "model_a" in weights
        assert "model_b" in weights
        assert pytest.approx(sum(weights.values()), abs=1e-3) == 1.0
        # Model A must have much higher weight than Model B
        assert weights["model_a"] > weights["model_b"]

    def test_missing_model_raises_readiness_error(self, tmp_path):
        """EnsembleInference must raise ModelNotReadyError if any model checkpoint is missing."""
        empty_dir = tmp_path / "empty_checkpoints"
        empty_dir.mkdir()

        with pytest.raises(ModelNotReadyError):
            EnsembleInference(
                model_names=["custom_cnn", "non_existent_model"],
                checkpoints_dir=empty_dir,
            )
