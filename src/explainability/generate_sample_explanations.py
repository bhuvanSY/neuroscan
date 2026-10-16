"""Generate representative Grad-CAM explanation visualizations and radiomics profiles.

Selects balanced representative validation cases:
- Representative CORRECT predictions across tumor classes
- Representative INCORRECT predictions across models (honest error analysis)

Generates:
- Original MRI scans
- Heatmap-only colorized visualizations
- Blended Grad-CAM overlays for Custom CNN, ResNet-50, and DenseNet-121
- Radiomics feature profiles with clinical descriptors
- Markdown summary report at reports/explanations/explanations_summary.md
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import cv2
from PIL import Image

from src.utils.config import load_config, resolve_dataset_path, get_project_root
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.data.validator import DatasetValidator
from src.data.dataset import create_dataloaders
from src.data.constants import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from src.models.readiness import ModelReadinessManager
from src.models.architectures import create_model
from src.explainability.gradcam import GradCAM
from src.features.radiomics import RadiomicsExtractor

logger = get_logger("neuroscan.explainability")


def run_generate_explanations() -> None:
    root = get_project_root()
    output_dir = root / "reports" / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = root / "checkpoints"

    device = get_device("auto")
    logger.info(f"Generating explanations on device: {device}")

    # Load dataset partitions
    config = load_config()
    dataset_path = resolve_dataset_path(config)
    validator = DatasetValidator(
        dataset_dir=dataset_path,
        val_split_ratio=config.get("dataset", {}).get("validation_split_ratio", 0.20),
        seed=config.get("seed", 42),
    )
    scan_res = validator.scan_dataset()
    records = validator.partition_splits(scan_res["records"])
    val_records = [r for r in records if r["split"] == "val"]

    # Load training stats
    norm_file = root / "reports" / "dataset_stats.json"
    if norm_file.exists():
        with open(norm_file, "r") as f:
            stats = json.load(f)
            mean = stats.get("mean", [0.172, 0.172, 0.172])
            std = stats.get("std", [0.1868, 0.1868, 0.1868])
    else:
        mean = [0.172, 0.172, 0.172]
        std = [0.1868, 0.1868, 0.1868]

    # Load models
    models = {}
    for name in ["custom_cnn", "resnet50", "densenet121"]:
        ckpt_path = checkpoints_dir / f"{name}_best.pt"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found for {name}, skipping.")
            continue
        m = create_model(name, num_classes=len(CLASSES))
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        m.load_state_dict(state_dict)
        m.to(device)
        m.eval()
        models[name] = m

    if not models:
        raise RuntimeError("No models available for explanation generation.")

    # Select representative samples from validation records
    # We want 3 correct cases and 3 incorrect cases
    logger.info("Scanning validation samples to identify representative correct and incorrect cases...")
    from src.data.transforms import get_transforms
    transforms = get_transforms(split="val", crop_contour=True, mean=mean, std=std)

    correct_samples = []
    incorrect_samples = []

    # Fast forward passes to categorize validation samples
    with torch.no_grad():
        for rec in val_records:
            img_path = Path(rec["path"])
            true_cls_name = rec["label"]
            true_label_idx = rec["class_index"]
            try:
                pil_img = Image.open(img_path).convert("RGB")
                tensor = transforms(pil_img).unsqueeze(0).to(device)
            except Exception:
                continue

            preds_by_model = {}
            for m_name, m in models.items():
                logits = m(tensor)
                pred_cls = int(logits.argmax(dim=-1).item())
                preds_by_model[m_name] = pred_cls

            # Check if all models correct
            all_correct = all(p == true_label_idx for p in preds_by_model.values())
            # Check if any model got it wrong
            any_incorrect = any(p != true_label_idx for p in preds_by_model.values())

            if all_correct and len(correct_samples) < 3:
                existing_classes = [s["true_label_name"] for s in correct_samples]
                if true_cls_name not in existing_classes:
                    correct_samples.append({
                        "record": rec,
                        "preds": preds_by_model,
                        "true_label_idx": true_label_idx,
                        "true_label_name": true_cls_name,
                        "type": "correct",
                    })

            if any_incorrect and len(incorrect_samples) < 3:
                existing_classes = [s["true_label_name"] for s in incorrect_samples]
                if true_cls_name not in existing_classes or len(incorrect_samples) < 2:
                    incorrect_samples.append({
                        "record": rec,
                        "preds": preds_by_model,
                        "true_label_idx": true_label_idx,
                        "true_label_name": true_cls_name,
                        "type": "incorrect",
                    })

            if len(correct_samples) >= 3 and len(incorrect_samples) >= 3:
                break

    selected_samples = correct_samples + incorrect_samples
    logger.info(f"Selected {len(selected_samples)} representative cases ({len(correct_samples)} correct, {len(incorrect_samples)} incorrect).")

    # Initialize Radiomics Extractor
    radiomics = RadiomicsExtractor()

    # Process and save visualizations for each sample
    summary_entries = []

    for idx, sample in enumerate(selected_samples, 1):
        rec = sample["record"]
        img_path = Path(rec["path"])
        true_cls_name = sample["true_label_name"]
        case_type = sample["type"]
        sample_tag = f"case_{idx:02d}_{case_type}_{true_cls_name}"
        sample_dir = output_dir / sample_tag
        sample_dir.mkdir(parents=True, exist_ok=True)

        pil_img = Image.open(img_path).convert("RGB")

        # 1. Extract Radiomics features
        rad_res = radiomics.extract_features(pil_img)
        rad_features = rad_res["features"]
        with open(sample_dir / "radiomics_features.json", "w", encoding="utf-8") as f:
            json.dump(rad_res, f, indent=2)

        # 2. Generate Grad-CAM for each model
        model_results = {}
        for m_name, model in models.items():
            with GradCAM(model, device=device) as cam_engine:
                # Predicted class explanation
                pred_pack = cam_engine.explain_image(img_path, target_class=None, mean=mean, std=std)
                # True class explanation (especially illuminating if incorrect)
                true_pack = cam_engine.explain_image(img_path, target_class=true_cls_name, mean=mean, std=std)

                # Save original cropped image
                if not (sample_dir / "original_mri.png").exists():
                    cv2.imwrite(
                        str(sample_dir / "original_mri.png"),
                        cv2.cvtColor(pred_pack["original_image"], cv2.COLOR_RGB2BGR)
                    )

                # Save heatmap-only
                cv2.imwrite(
                    str(sample_dir / f"heatmap_only_{m_name}.png"),
                    cv2.cvtColor(pred_pack["heatmap_colored"], cv2.COLOR_RGB2BGR)
                )

                # Save overlay
                cv2.imwrite(
                    str(sample_dir / f"gradcam_overlay_{m_name}.png"),
                    cv2.cvtColor(pred_pack["overlay"], cv2.COLOR_RGB2BGR)
                )

                model_results[m_name] = {
                    "predicted_class": pred_pack["predicted_class_name"],
                    "confidence": round(pred_pack["confidence"], 4),
                    "is_correct": (pred_pack["predicted_class_name"] == true_cls_name),
                    "class_probs": {k: round(v, 4) for k, v in pred_pack["class_probabilities"].items()},
                }

        summary_entries.append({
            "case_id": idx,
            "sample_tag": sample_tag,
            "type": case_type.upper(),
            "true_class": true_cls_name,
            "image_filename": img_path.name,
            "model_predictions": model_results,
            "radiomics": {
                "mean_intensity": rad_features["intensity_mean"],
                "glcm_contrast": rad_features["glcm_contrast"],
                "circularity": rad_features["circularity"],
                "fractal_dimension": rad_features["box_counting_fractal_dimension"],
                "edge_density": rad_features["canny_edge_density"],
            },
        })

    # Generate Markdown Summary Report
    md_path = output_dir / "explanations_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NeuroScan AI - Explainability & Radiomics Validation Report\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Medical Interpretability Disclaimer**:\n")
        f.write("> Grad-CAM heatmaps highlight coarse neural gradient activations and do NOT constitute tumor segmentation masks or surgical margins.\n")
        f.write("> Radiomics features represent 2D exploratory slice descriptors, NOT validated clinical biomarkers or 3D lesion volumes.\n\n")

        f.write("## 1. Overview of Representative Validation Cohort\n\n")
        f.write("This cohort presents balanced evaluations on both **correctly identified scans** and **difficult/misclassified cases** across all three scratch-trained CNN architectures.\n\n")

        for entry in summary_entries:
            f.write(f"### Case {entry['case_id']}: {entry['sample_tag']} ({entry['type']})\n\n")
            f.write(f"- **Ground Truth**: `{entry['true_class']}` (Image: `{entry['image_filename']}`)\n")
            f.write(f"- **Exploratory Radiomics Profile**:\n")
            f.write(f"  - Mean Luminance: `{entry['radiomics']['mean_intensity']}` | GLCM Contrast: `{entry['radiomics']['glcm_contrast']}`\n")
            f.write(f"  - Circularity: `{entry['radiomics']['circularity']}` | Fractal Dimension: `{entry['radiomics']['fractal_dimension']}` | Canny Edge Density: `{entry['radiomics']['edge_density']}`\n\n")

            f.write("| Model | Predicted Class | Target Confidence | Accurate? |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for m_name, res in entry["model_predictions"].items():
                check = "CORRECT" if res["is_correct"] else "MISCLASSIFIED"
                f.write(f"| `{m_name}` | `{res['predicted_class']}` | `{res['confidence']:.4f}` | **{check}** |\n")
            f.write("\n")

            f.write("Visualizations saved:\n")
            f.write(f"- Original MRI: [`reports/explanations/{entry['sample_tag']}/original_mri.png`](file:///{str(output_dir / entry['sample_tag'] / 'original_mri.png').replace(chr(92), '/')})\n")
            for m_name in entry["model_predictions"].keys():
                f.write(f"- `{m_name}` Grad-CAM Overlay: [`overlay`](file:///{str(output_dir / entry['sample_tag'] / f'gradcam_overlay_{m_name}.png').replace(chr(92), '/')}) | Heatmap-only: [`heatmap`](file:///{str(output_dir / entry['sample_tag'] / f'heatmap_only_{m_name}.png').replace(chr(92), '/')})\n")
            f.write("\n---\n\n")

    logger.info(f"Explanation cohort successfully generated at: {output_dir}")
    logger.info(f"Markdown report written to: {md_path}")


if __name__ == "__main__":
    run_generate_explanations()
