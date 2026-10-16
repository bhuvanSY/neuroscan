"""Core training engine for NeuroScan AI models.

Features:
- Full scratch training (zero pretrained weights)
- Cross-entropy loss
- AdamW optimizer with Cosine Annealing learning rate scheduling
- Early stopping based on validation macro-F1
- Automatic Mixed Precision (AMP) on CUDA
- Validation loss, accuracy, and macro-F1 tracking
- Checkpoint saving with comprehensive parameter counts and training metadata
- Automated learning curve visualization
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from src.models.architectures import create_model
from src.data.dataset import create_dataloaders
from src.data.validator import DatasetValidator
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.utils.exceptions import DatasetNotFoundError

logger = get_logger("neuroscan.engine")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Calculate total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def plot_learning_curves(history: List[Dict[str, Any]], model_name: str, output_path: Path) -> None:
    """Plot and save loss, accuracy, and macro-F1 learning curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    train_f1 = [h["train_macro_f1"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="#4a7bb0", linewidth=1.8)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#c27d38", linewidth=1.8, linestyle="--")
    ax1.set_title("Loss Curve", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Accuracy
    ax2.plot(epochs, train_acc, label="Train Accuracy", color="#4a7bb0", linewidth=1.8)
    ax2.plot(epochs, val_acc, label="Val Accuracy", color="#2e7d32", linewidth=1.8, linestyle="--")
    ax2.set_title("Accuracy Curve", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)

    # Macro-F1
    ax3.plot(epochs, train_f1, label="Train Macro-F1", color="#4a7bb0", linewidth=1.8)
    ax3.plot(epochs, val_f1, label="Val Macro-F1", color="#7b1fa2", linewidth=1.8, linestyle="--")
    ax3.set_title("Macro-F1 Curve", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=10)
    ax3.set_ylabel("Macro-F1", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="lower right", fontsize=9)

    plt.suptitle(f"NeuroScan AI - {model_name.upper()} Training Learning Curves (From Scratch)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


class ScratchTrainer:
    """Orchestrates scratch training, validation, early stopping, and checkpointing."""

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        device_preference: str = "auto",
    ):
        self.model_name = model_name
        self.config = config
        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        self.device = get_device(device_preference)
        self.use_amp = (self.device.type == "cuda") and config.get("training", {}).get("mixed_precision", True)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Resolve paths
        from src.utils.config import get_project_root, resolve_dataset_path
        self.root = get_project_root()
        self.dataset_path = resolve_dataset_path(config)
        self.checkpoints_dir = self.root / "checkpoints"
        self.reports_dir = self.root / "reports"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Normalization stats
        norm_file = self.reports_dir / "dataset_stats.json"
        if norm_file.exists():
            with open(norm_file, "r") as f:
                stats = json.load(f)
                self.mean = stats.get("mean", [0.485, 0.456, 0.406])
                self.std = stats.get("std", [0.229, 0.224, 0.225])
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def prepare_data(self) -> Dict[str, Any]:
        """Scan, partition, and create PyTorch DataLoaders."""
        if not self.dataset_path.exists():
            raise DatasetNotFoundError(f"Dataset path not found: {self.dataset_path}")

        validator = DatasetValidator(
            dataset_dir=self.dataset_path,
            val_split_ratio=self.config.get("dataset", {}).get("validation_split_ratio", 0.20),
            seed=self.seed,
        )
        scan_res = validator.scan_dataset()
        if not scan_res["dataset_present"] or not scan_res["records"]:
            raise DatasetNotFoundError(f"No valid brain MRI images in: {self.dataset_path}")

        records = validator.partition_splits(scan_res["records"])

        training_cfg = self.config.get("training", {})
        batch_size = training_cfg.get("batch_size", 32)
        hardware_cfg = self.config.get("hardware", {})
        num_workers = hardware_cfg.get("num_workers", 2)
        image_size = tuple(self.config.get("preprocessing", {}).get("image_size", [224, 224]))
        crop_contour = self.config.get("preprocessing", {}).get("crop_brain_contour", True)

        dl_info = create_dataloaders(
            records=records,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            image_size=image_size,
            crop_contour=crop_contour,
            mean=self.mean,
            std=self.std,
            seed=self.seed,
        )
        return dl_info

    def train_epoch(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Execute one training epoch with optional mixed precision."""
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.detach().cpu().numpy())

        total = len(all_targets)
        epoch_loss = running_loss / max(total, 1)
        epoch_acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
        epoch_f1 = float(f1_score(all_targets, all_preds, average="macro"))

        return {
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "train_macro_f1": epoch_f1,
        }

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Evaluate model on validation split."""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

        total = len(all_targets)
        val_loss = running_loss / max(total, 1)
        val_acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
        val_f1 = float(f1_score(all_targets, all_preds, average="macro"))

        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }

    def run(self) -> Dict[str, Any]:
        """Run complete scratch training pipeline with early stopping."""
        dl_info = self.prepare_data()
        train_loader = dl_info["loaders"]["train"]
        val_loader = dl_info["loaders"]["val"]

        # Build model from scratch
        model = create_model(self.model_name, num_classes=4).to(self.device)
        total_params, trainable_params = count_parameters(model)

        training_cfg = self.config.get("training", {})
        epochs = training_cfg.get("epochs", 15)
        lr = training_cfg.get("learning_rate", 1e-3)
        weight_decay = training_cfg.get("weight_decay", 1e-4)
        patience = training_cfg.get("early_stopping_patience", 6)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_ckpt_path = self.checkpoints_dir / f"{self.model_name}_best.pt"
        history = []
        best_val_f1 = 0.0
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        logger.info(f"=== Starting Scratch Training: {self.model_name.upper()} ===")
        logger.info(f"Device: {self.device} | AMP: {self.use_amp}")
        logger.info(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")
        logger.info(f"Epochs: {epochs} | Batch Size: {training_cfg.get('batch_size', 32)} | Initial LR: {lr}")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer)
            val_metrics = self.evaluate(model, val_loader, criterion)
            scheduler.step()
            epoch_duration = time.time() - t0

            epoch_record = {
                "epoch": epoch,
                "duration_seconds": round(epoch_duration, 2),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                **train_metrics,
                **val_metrics,
            }
            history.append(epoch_record)

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] - "
                f"Train Loss: {train_metrics['train_loss']:.4f} | Acc: {train_metrics['train_acc']:.4f} | F1: {train_metrics['train_macro_f1']:.4f} || "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Acc: {val_metrics['val_acc']:.4f} | F1: {val_metrics['val_macro_f1']:.4f}"
            )

            # Checkpoint on best validation macro-F1
            if val_metrics["val_macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["val_macro_f1"]
                best_val_acc = val_metrics["val_acc"]
                best_epoch = epoch
                patience_counter = 0

                torch.save({
                    "model_name": self.model_name,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": {
                        "val_macro_f1": best_val_f1,
                        "val_accuracy": best_val_acc,
                        "val_loss": val_metrics["val_loss"],
                        "train_macro_f1": train_metrics["train_macro_f1"],
                        "train_accuracy": train_metrics["train_acc"],
                        "train_loss": train_metrics["train_loss"],
                    },
                    "parameters": {
                        "total": total_params,
                        "trainable": trainable_params,
                    },
                    "class_to_idx": dl_info["class_to_idx"],
                    "normalization": {
                        "mean": self.mean,
                        "std": self.std,
                    },
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, best_ckpt_path)
                logger.info(f" Saved new best checkpoint to {best_ckpt_path} (Val Macro-F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (no validation improvement for {patience} epochs).")
                    break

        total_training_time = time.time() - start_time
        logger.info(f"=== Scratch Training Completed in {total_training_time:.1f}s ===")
        logger.info(f"Best Validation Macro-F1: {best_val_f1:.4f} | Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")

        # Save learning curves and training history
        curves_path = self.reports_dir / f"learning_curves_{self.model_name}.png"
        plot_learning_curves(history, self.model_name, curves_path)
        logger.info(f"Learning curves saved to: {curves_path}")

        history_path = self.reports_dir / f"training_{self.model_name}_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
                "best_val_accuracy": best_val_acc,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "total_training_time_seconds": round(total_training_time, 2),
                "history": history,
            }, f, indent=2)

        return {
            "model_name": self.model_name,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "best_val_accuracy": best_val_acc,
            "checkpoint_path": str(best_ckpt_path),
            "curves_path": str(curves_path),
            "history_path": str(history_path),
            "total_parameters": total_params,
            "training_time_seconds": round(total_training_time, 2),
        }
