"""Model readiness verification and checkpoint health auditing."""

import os
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

from src.utils.exceptions import ModelNotReadyError
from src.utils.config import get_project_root


class ModelStatus(str, Enum):
    READY = "READY"
    MISSING = "MISSING"
    CORRUPTED = "CORRUPTED"
    UNINITIALIZED = "UNINITIALIZED"


REGISTERED_MODELS = [
    "custom_cnn",
    "resnet50",
    "densenet121",
    "efficientnet_v2",
    "rl_triage_policy",
]


class ModelReadinessManager:
    """Manages and verifies the lifecycle and readiness of trained models."""

    def __init__(self, checkpoints_dir: Optional[Path] = None):
        root = get_project_root()
        self.checkpoints_dir = Path(checkpoints_dir or (root / "checkpoints"))
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, model_name: str) -> Path:
        """Resolve canonical checkpoint path for a model name."""
        canonical = self.checkpoints_dir / f"{model_name}_best.pt"
        if not canonical.exists():
            # Check legacy alias
            if model_name == "custom_cnn" and (self.checkpoints_dir / "neuro_convnet_best.pt").exists():
                return self.checkpoints_dir / "neuro_convnet_best.pt"
            if model_name == "neuro_convnet" and (self.checkpoints_dir / "custom_cnn_best.pt").exists():
                return self.checkpoints_dir / "custom_cnn_best.pt"
        return canonical

    def check_model_readiness(self, model_name: str) -> Dict[str, Any]:
        """Check readiness status for a specific model without fabricating metrics."""
        ckpt_path = self.get_checkpoint_path(model_name)
        
        if not ckpt_path.exists():
            return {
                "model_name": model_name,
                "status": ModelStatus.MISSING.value,
                "checkpoint_path": str(ckpt_path),
                "is_ready": False,
                "details": f"Checkpoint file '{ckpt_path.name}' does not exist on disk. Model must be trained from scratch first.",
            }

        if ckpt_path.stat().st_size == 0:
            return {
                "model_name": model_name,
                "status": ModelStatus.CORRUPTED.value,
                "checkpoint_path": str(ckpt_path),
                "is_ready": False,
                "details": "Checkpoint file is 0 bytes (corrupted).",
            }

        try:
            # Inspect checkpoint weights safely on CPU
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            # Check for valid state_dict
            has_weights = "state_dict" in checkpoint or "model_state_dict" in checkpoint or isinstance(checkpoint, dict)
            if not has_weights:
                return {
                    "model_name": model_name,
                    "status": ModelStatus.CORRUPTED.value,
                    "checkpoint_path": str(ckpt_path),
                    "is_ready": False,
                    "details": "Checkpoint does not contain valid state_dict weights.",
                }

            metrics = checkpoint.get("metrics", {})
            epoch = checkpoint.get("epoch", None)

            return {
                "model_name": model_name,
                "status": ModelStatus.READY.value,
                "checkpoint_path": str(ckpt_path),
                "is_ready": True,
                "epoch": epoch,
                "metrics": metrics,
                "file_size_bytes": ckpt_path.stat().st_size,
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "status": ModelStatus.CORRUPTED.value,
                "checkpoint_path": str(ckpt_path),
                "is_ready": False,
                "details": f"Failed to load checkpoint: {str(e)}",
            }

    def get_all_readiness(self) -> Dict[str, Any]:
        """Audit readiness across all registered models."""
        statuses = {}
        all_ready = True
        
        for name in REGISTERED_MODELS:
            res = self.check_model_readiness(name)
            statuses[name] = res
            if not res["is_ready"]:
                all_ready = False

        return {
            "all_models_ready": all_ready,
            "total_registered": len(REGISTERED_MODELS),
            "ready_count": sum(1 for s in statuses.values() if s["is_ready"]),
            "models": statuses,
        }

    def assert_ready(self, model_name: str) -> Path:
        """Raise ModelNotReadyError if model checkpoint is not verified on disk."""
        status = self.check_model_readiness(model_name)
        if not status["is_ready"]:
            raise ModelNotReadyError(
                f"Model '{model_name}' is not ready for inference. Status: {status['status']}. {status.get('details', '')}",
                details=status
            )
        return Path(status["checkpoint_path"])
