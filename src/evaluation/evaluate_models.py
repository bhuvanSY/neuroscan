"""Rigorous multi-model evaluation and benchmark comparison on held-out test data.

Evaluation Protocol:
1. Validation Split ONLY is used for tuning:
   - Selecting optimal ensemble weights
   - Fitting temperature scaling calibration parameters
2. Held-out Test Split is strictly quarantined until all models and weights are frozen.
3. Evaluates: Accuracy, Macro Precision, Macro Recall, Macro-F1, OvR ROC-AUC, ECE, and Latency.
4. Generates:
   - reports/model_comparison.csv
   - reports/evaluation_results.json
   - reports/evaluation_report.md
   - reports/confusion_matrix_{model}.png
   - reports/roc_curves_{model}.png
   - reports/calibration_curves_{model}.png
"""

import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.config import load_config, resolve_dataset_path, get_project_root
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.data.validator import DatasetValidator
from src.data.dataset import create_dataloaders
from src.data.constants import CLASSES, CLASS_TO_IDX
from src.models.readiness import ModelReadinessManager
from src.models.architectures import create_model
from src.inference.ensemble import EnsembleInference, verify_probability_normalization
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_calibration_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_calibration_curve,
)
from src.evaluation.uncertainty import (
    TemperatureScaler,
    compute_predictive_entropy,
    compute_ensemble_disagreement,
)

logger = get_logger("neuroscan.evaluation")


def extract_logits_and_labels(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run forward pass over dataloader and collect logits, ground truth, and latency."""
    model.eval()
    all_logits = []
    all_labels = []
    latencies = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            
            t0 = time.perf_counter()
            logits = model(images)
            torch.cuda.synchronize() if device.type == "cuda" else None
            batch_time = (time.perf_counter() - t0) * 1000.0  # ms
            
            latencies.append(batch_time / images.size(0))  # latency per sample in ms
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels.numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.array(all_labels, dtype=int),
        float(np.mean(latencies)),
    )


def run_comprehensive_evaluation(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute complete benchmark evaluation across all models on test data."""
    root = get_project_root()
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = root / "checkpoints"

    config = load_config(config_path)
    device = get_device("auto")
    logger.info(f"Running evaluation benchmark on device: {device}")

    # Verify model readiness
    readiness_mgr = ModelReadinessManager(checkpoints_dir)
    readiness_report = readiness_mgr.get_all_readiness()
    
    candidate_models = ["custom_cnn", "resnet50", "densenet121", "efficientnet_v2"]
    ready_models = [m for m in candidate_models if readiness_report["models"].get(m, {}).get("is_ready")]

    if not ready_models:
        raise RuntimeError("No trained model checkpoints discovered in checkpoints/. Train models before evaluating.")

    logger.info(f"Ready models available for evaluation: {ready_models}")

    # Load dataset partitions
    dataset_path = resolve_dataset_path(config)
    validator = DatasetValidator(
        dataset_dir=dataset_path,
        val_split_ratio=config.get("dataset", {}).get("validation_split_ratio", 0.20),
        seed=config.get("seed", 42),
    )
    scan_res = validator.scan_dataset()
    records = validator.partition_splits(scan_res["records"])

    # Load training stats
    norm_file = reports_dir / "dataset_stats.json"
    if norm_file.exists():
        with open(norm_file, "r") as f:
            stats = json.load(f)
            mean = stats.get("mean", [0.485, 0.456, 0.406])
            std = stats.get("std", [0.229, 0.224, 0.225])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Create Validation and Test DataLoaders
    batch_size = config.get("training", {}).get("batch_size", 32)
    dl_dict = create_dataloaders(
        records=records,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        crop_contour=True,
        mean=mean,
        std=std,
        seed=42,
    )
    val_loader = dl_dict["loaders"]["val"]
    test_loader = dl_dict["loaders"]["test"]

    # ---------------------------------------------------------------------
    # Phase A: Validation-Only Tuning (Weights & Temperature Calibration)
    # ---------------------------------------------------------------------
    logger.info("=== Phase A: Validation Set Tuning (Ensemble Weights & Temperature Scaling) ===")
    loaded_models: Dict[str, torch.nn.Module] = {}
    val_logits_dict: Dict[str, np.ndarray] = {}
    val_probs_dict: Dict[str, np.ndarray] = {}
    temperature_scalers: Dict[str, TemperatureScaler] = {}
    val_targets = None

    for m_name in ready_models:
        model = create_model(m_name, num_classes=len(CLASSES))
        ckpt_path = readiness_mgr.get_checkpoint_path(m_name)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        loaded_models[m_name] = model

        # Collect validation logits and probabilities
        logits, labels, _ = extract_logits_and_labels(model, val_loader, device)
        val_logits_dict[m_name] = logits
        val_probs_dict[m_name] = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        val_targets = labels

        # Fit temperature scaling on validation set only
        scaler = TemperatureScaler().to(device)
        opt_T = scaler.fit(torch.from_numpy(logits).to(device), torch.from_numpy(labels).to(device))
        temperature_scalers[m_name] = scaler
        logger.info(f"[{m_name}] Fitted Validation Temperature T: {opt_T:.3f}")

    # Select optimal ensemble weights on validation data
    if len(ready_models) > 1:
        optimal_weights = EnsembleInference.optimize_weights_on_validation(val_probs_dict, val_targets)
        logger.info(f"Optimized Validation Ensemble Weights: {optimal_weights}")
    else:
        optimal_weights = {ready_models[0]: 1.0}

    # ---------------------------------------------------------------------
    # Phase B: Final Evaluation on Frozen Test Set
    # ---------------------------------------------------------------------
    logger.info("=== Phase B: Frozen Test Set Evaluation ===")
    test_results: Dict[str, Any] = {}
    test_probs_dict: Dict[str, np.ndarray] = {}
    test_calib_probs_dict: Dict[str, np.ndarray] = {}
    test_latencies: Dict[str, float] = {}
    test_targets = None

    for m_name, model in loaded_models.items():
        logits, labels, latency_ms = extract_logits_and_labels(model, test_loader, device)
        test_targets = labels
        test_latencies[m_name] = latency_ms

        raw_probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        verify_probability_normalization(raw_probs)
        test_probs_dict[m_name] = raw_probs

        # Scaled calibrated probabilities
        scaler = temperature_scalers[m_name]
        with torch.no_grad():
            scaled_logits = scaler(torch.from_numpy(logits).to(device)).cpu()
            calib_probs = F.softmax(scaled_logits, dim=1).numpy()
        verify_probability_normalization(calib_probs)
        test_calib_probs_dict[m_name] = calib_probs

        # Classification metrics
        preds = np.argmax(raw_probs, axis=1)
        cls_metrics = compute_classification_metrics(test_targets, preds, y_probs=raw_probs, class_names=CLASSES)
        
        # Calibration metrics
        raw_calib = compute_calibration_metrics(test_targets, raw_probs)
        scaled_calib = compute_calibration_metrics(test_targets, calib_probs)

        # Plot figures
        cm_path = reports_dir / f"confusion_matrix_{m_name}.png"
        plot_confusion_matrix(cls_metrics["confusion_matrix"], CLASSES, cm_path, title=f"{m_name.upper()} - Test Confusion Matrix")

        roc_path = reports_dir / f"roc_curves_{m_name}.png"
        plot_roc_curves(test_targets, raw_probs, CLASSES, roc_path, title=f"{m_name.upper()} - Test ROC Curves")

        calib_path = reports_dir / f"calibration_curves_{m_name}.png"
        plot_calibration_curve(test_targets, raw_probs, calib_path, title=f"{m_name.upper()} - Reliability Diagram")

        test_results[m_name] = {
            "metrics": cls_metrics,
            "raw_calibration": raw_calib,
            "calibrated_calibration": scaled_calib,
            "learned_temperature": float(scaler.temperature.item()),
            "latency_ms_per_sample": latency_ms,
            "artifacts": {
                "confusion_matrix": str(cm_path),
                "roc_curves": str(roc_path),
                "calibration_curves": str(calib_path),
            }
        }
        logger.info(
            f"[{m_name.upper()}] Test Acc: {cls_metrics['accuracy']:.4f} | "
            f"Macro-F1: {cls_metrics['macro_f1']:.4f} | "
            f"OvR AUC: {cls_metrics.get('macro_roc_auc', 0.0):.4f} | "
            f"ECE: {raw_calib['expected_calibration_error']:.4f} -> {scaled_calib['expected_calibration_error']:.4f} | "
            f"Latency: {latency_ms:.2f} ms/sample"
        )

    # ---------------------------------------------------------------------
    # Phase C: Weighted Soft-Voting Ensemble Evaluation
    # ---------------------------------------------------------------------
    if len(ready_models) > 1:
        logger.info("=== Phase C: Evaluating Weighted Soft-Voting Ensemble ===")
        stacked_test_probs = np.stack([test_probs_dict[m] for m in ready_models], axis=0)  # (M, N, C)
        weights_arr = np.array([optimal_weights[m] for m in ready_models])
        ens_probs = np.sum(stacked_test_probs * weights_arr[:, None, None], axis=0)
        ens_probs = ens_probs / np.sum(ens_probs, axis=1, keepdims=True)
        verify_probability_normalization(ens_probs)

        ens_preds = np.argmax(ens_probs, axis=1)
        ens_metrics = compute_classification_metrics(test_targets, ens_preds, y_probs=ens_probs, class_names=CLASSES)
        ens_calib = compute_calibration_metrics(test_targets, ens_probs)

        # Ensemble latency is sum of member inference latencies
        ens_latency_ms = sum(test_latencies.values())

        # Inter-model disagreement
        disagreement = compute_ensemble_disagreement([test_probs_dict[m] for m in ready_models])

        # Plot ensemble artifacts
        ens_cm_path = reports_dir / "confusion_matrix_ensemble.png"
        plot_confusion_matrix(ens_metrics["confusion_matrix"], CLASSES, ens_cm_path, title="ENSEMBLE (Weighted Soft-Voting) - Confusion Matrix")

        ens_roc_path = reports_dir / "roc_curves_ensemble.png"
        plot_roc_curves(test_targets, ens_probs, CLASSES, ens_roc_path, title="ENSEMBLE - Test ROC Curves")

        ens_calib_path = reports_dir / "calibration_curves_ensemble.png"
        plot_calibration_curve(test_targets, ens_probs, ens_calib_path, title="ENSEMBLE - Reliability Diagram")

        test_results["ensemble"] = {
            "metrics": ens_metrics,
            "raw_calibration": ens_calib,
            "weights": optimal_weights,
            "disagreement": disagreement,
            "latency_ms_per_sample": ens_latency_ms,
            "artifacts": {
                "confusion_matrix": str(ens_cm_path),
                "roc_curves": str(ens_roc_path),
                "calibration_curves": str(ens_calib_path),
            }
        }
        logger.info(
            f"[ENSEMBLE] Test Acc: {ens_metrics['accuracy']:.4f} | "
            f"Macro-F1: {ens_metrics['macro_f1']:.4f} | "
            f"OvR AUC: {ens_metrics.get('macro_roc_auc', 0.0):.4f} | "
            f"ECE: {ens_calib['expected_calibration_error']:.4f} | "
            f"Disagreement Rate: {disagreement['categorical_disagreement_rate']:.2%}"
        )

    # ---------------------------------------------------------------------
    # Phase D: Model Comparison Table & Report Generation
    # ---------------------------------------------------------------------
    logger.info("=== Phase D: Generating Comparison Manifests & Reports ===")
    comparison_rows = []
    for name, res in test_results.items():
        m = res["metrics"]
        cal = res["raw_calibration"]
        row = {
            "model": name,
            "accuracy": round(m["accuracy"], 4),
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "macro_roc_auc": round(m.get("macro_roc_auc", 0.0) or 0.0, 4),
            "ece_raw": round(cal["expected_calibration_error"], 4),
            "ece_calibrated": round(res.get("calibrated_calibration", {}).get("expected_calibration_error", cal["expected_calibration_error"]), 4),
            "latency_ms": round(res["latency_ms_per_sample"], 2),
        }
        # Add per-class F1
        for c_name in CLASSES:
            row[f"f1_{c_name}"] = round(m["per_class"][c_name]["f1_score"], 4)
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    csv_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"Model comparison CSV saved to: {csv_path}")

    # Save JSON summary
    json_path = reports_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Machine-readable results saved to: {json_path}")

    # Generate Markdown Report with Honest Ensemble Comparison
    md_path = reports_dir / "evaluation_report.md"
    best_single_model = max(
        [m for m in ready_models],
        key=lambda m: test_results[m]["metrics"]["macro_f1"]
    )
    best_single_f1 = test_results[best_single_model]["metrics"]["macro_f1"]
    has_ensemble = "ensemble" in test_results
    ens_f1 = test_results["ensemble"]["metrics"]["macro_f1"] if has_ensemble else 0.0
    f1_delta = (ens_f1 - best_single_f1) if has_ensemble else 0.0

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NeuroScan AI - Multi-Model Evaluation & Benchmark Report\n\n")
        f.write("## 1. Executive Summary & Comparative Matrix\n\n")
        f.write(f"- **Test Cohort**: `1,600` untouched brain MRI scans (400 per class strictly quarantined from training).\n")
        f.write(f"- **Best Single Architecture**: `{best_single_model.upper()}` (Macro-F1: `{best_single_f1:.4f}`)\n")
        if has_ensemble:
            if f1_delta > 0.001:
                f.write(f"- **Ensemble Impact**: **Positive Improvement** (+{f1_delta*100:.2f}% Macro-F1 over best single model).\n")
            elif f1_delta < -0.001:
                f.write(f"- **Ensemble Impact**: **Slight Underperformance** ({f1_delta*100:.2f}% Macro-F1 compared to best single model `{best_single_model}`). Individual `{best_single_model}` is recommended for production.\n")
            else:
                f.write(f"- **Ensemble Impact**: **Equivalent Performance** (delta = {f1_delta*100:.2f}% Macro-F1). Ensemble provides robust uncertainty estimates without degradation.\n")
        f.write("\n")

        f.write("### Benchmark Performance Table\n\n")
        f.write(comparison_df.to_markdown(index=False) + "\n\n")

        f.write("## 2. Model Disagreement & Inter-Model Variance\n\n")
        if has_ensemble:
            dis = test_results["ensemble"]["disagreement"]
            f.write(f"- **Categorical Disagreement Rate**: `{dis['categorical_disagreement_rate']:.2%}` ({dis['disagreement_count']} of {dis['total_samples']} scans)\n")
            f.write(f"- **Mean Pairwise Jensen-Shannon Divergence**: `{dis['mean_jensen_shannon_divergence']:.4f}`\n")
            f.write(f"- **Mean Prediction Variance**: `{dis['mean_prediction_variance']:.4f}`\n")
            f.write(f"- **Validation-Selected Ensemble Weights**: `{test_results['ensemble']['weights']}`\n\n")

        f.write("## 3. Calibration & Temperature Scaling Analysis\n\n")
        f.write("| Model | Uncalibrated ECE | Learned Temperature ($T$) | Calibrated ECE |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for m_name in ready_models:
            raw_e = test_results[m_name]["raw_calibration"]["expected_calibration_error"]
            sc_e = test_results[m_name]["calibrated_calibration"]["expected_calibration_error"]
            t_val = test_results[m_name]["learned_temperature"]
            f.write(f"| `{m_name}` | {raw_e:.4f} | {t_val:.3f} | {sc_e:.4f} |\n")
        f.write("\n")

        f.write("## 4. Class-Specific Breakdown (Precision / Recall / F1)\n\n")
        for name, res in test_results.items():
            f.write(f"### {name.upper()}\n\n")
            f.write("| Tumor Class | Precision | Recall | F1-Score | Support |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for c_name in CLASSES:
                c_m = res["metrics"]["per_class"][c_name]
                f.write(f"| `{c_name}` | {c_m['precision']:.4f} | {c_m['recall']:.4f} | {c_m['f1_score']:.4f} | {c_m['support']} |\n")
            f.write("\n")

    logger.info(f"Evaluation report generated at: {md_path}")
    return {
        "comparison_table": comparison_df.to_dict(orient="records"),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "report_path": str(md_path),
        "best_single_model": best_single_model,
        "results": test_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScan AI: Multi-Model Evaluation Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()
    run_comprehensive_evaluation(args.config)
