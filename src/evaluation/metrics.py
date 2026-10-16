"""Comprehensive classification and calibration metrics for NeuroScan AI."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from src.data.constants import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute overall, macro, and class-specific diagnostic performance metrics.
    
    Args:
        y_true: Ground truth integer class labels (N,)
        y_pred: Predicted integer class labels (N,)
        y_probs: Softmax probabilities (N, num_classes)
        class_names: List of string class names corresponding to sorted indices
        
    Returns:
        Structured metrics dictionary.
    """
    classes = class_names or CLASSES
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0
    )

    class_metrics = {}
    for i, name in enumerate(classes):
        class_metrics[name] = {
            "precision": float(p_per_class[i]),
            "recall": float(r_per_class[i]),
            "f1_score": float(f1_per_class[i]),
            "support": int(support_per_class[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "per_class": class_metrics,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true),
    }

    # Multi-class One-vs-Rest ROC-AUC
    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=float)
        # Verify all classes exist in y_true and probabilities match class count
        unique_true = np.unique(y_true)
        if len(unique_true) > 1 and y_probs.shape[1] == len(classes):
            try:
                ovr_auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
                metrics["macro_roc_auc"] = ovr_auc

                per_class_auc = {}
                for i, name in enumerate(classes):
                    # Binary target for class i
                    binary_true = (y_true == i).astype(int)
                    if len(np.unique(binary_true)) == 2:
                        per_class_auc[name] = float(roc_auc_score(binary_true, y_probs[:, i]))
                    else:
                        per_class_auc[name] = None
                metrics["per_class_roc_auc"] = per_class_auc
            except Exception as e:
                metrics["macro_roc_auc"] = None
                metrics["roc_auc_error"] = str(e)
        else:
            metrics["macro_roc_auc"] = None

    return metrics


def compute_calibration_metrics(
    y_true: Sequence[int],
    y_probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Calculate Expected Calibration Error (ECE) and binning statistics.
    
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            diff = abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": int(np.sum(in_bin)),
                "accuracy": accuracy_in_bin,
                "confidence": avg_confidence_in_bin,
            })
        else:
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": 0,
                "accuracy": 0.0,
                "confidence": (bin_lower + bin_upper) / 2.0,
            })

    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "binned_calibration": bin_data,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: Optional[str] = None,
    normalize: bool = True,
) -> None:
    """Plot and save confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm, dtype=float)
    
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)
        fmt = ".2%"
        data_to_plot = cm_norm
    else:
        fmt = "d"
        data_to_plot = cm_arr

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        data_to_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="#2c3440",
    )
    plt.title(title or "Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=10, fontweight="bold")
    plt.ylabel("Ground Truth Class", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curves(
    y_true: Sequence[int],
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot One-vs-Rest ROC curves per class with AUC scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    n_classes = len(class_names)

    plt.figure(figsize=(8, 6.5))
    colors = ["#2b5c8f", "#c27d38", "#2e7d32", "#7b1fa2"]

    for i in range(n_classes):
        bin_true = (y_true == i).astype(int)
        if len(np.unique(bin_true)) == 2:
            fpr, tpr, _ = roc_curve(bin_true, y_probs[:, i])
            roc_score = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{class_names[i]} (AUC = {roc_score:.3f})"
            )

    plt.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.2, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight="bold")
    plt.title(title or "One-vs-Rest ROC Curves", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curve(
    y_true: Sequence[int],
    y_probs: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
    title: Optional[str] = None,
) -> None:
    """Plot reliability diagram comparing confidence to empirical accuracy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calib = compute_calibration_metrics(y_true, y_probs, n_bins=n_bins)
    bins = calib["binned_calibration"]

    confs = [b["confidence"] for b in bins if b["count"] > 0]
    accs = [b["accuracy"] for b in bins if b["count"] > 0]

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect Calibration")
    plt.plot(confs, accs, marker="s", color="#2b5c8f", linewidth=2, label=f"Model (ECE = {calib['expected_calibration_error']:.3f})")
    plt.xlabel("Average Confidence", fontsize=10, fontweight="bold")
    plt.ylabel("Empirical Accuracy", fontsize=10, fontweight="bold")
    plt.title(title or f"Reliability Diagram (ECE: {calib['expected_calibration_error']:.3f})", fontsize=12, fontweight="bold", pad=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
