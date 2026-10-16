"""Smoke tests for model-readiness mechanism and non-fabrication guarantee."""

import torch
import pytest
from pathlib import Path
from src.models.readiness import ModelReadinessManager, ModelStatus
from src.utils.exceptions import ModelNotReadyError
from src.models.architectures import create_model


def test_missing_model_readiness(tmp_path: Path):
    """Verify that uninitialized models are strictly reported as MISSING."""
    manager = ModelReadinessManager(checkpoints_dir=tmp_path)
    res = manager.check_model_readiness("neuro_convnet")

    assert res["status"] == ModelStatus.MISSING.value
    assert res["is_ready"] is False

    with pytest.raises(ModelNotReadyError, match="is not ready"):
        manager.assert_ready("neuro_convnet")


def test_ready_model_readiness(tmp_path: Path):
    """Verify that a validly saved checkpoint transitions to READY status."""
    manager = ModelReadinessManager(checkpoints_dir=tmp_path)
    ckpt_path = manager.get_checkpoint_path("neuro_convnet")

    # Create model and save state_dict
    model = create_model("neuro_convnet", num_classes=4)
    torch.save({
        "epoch": 1,
        "state_dict": model.state_dict(),
        "metrics": {"val_acc": 0.85},
    }, ckpt_path)

    res = manager.check_model_readiness("neuro_convnet")
    assert res["status"] == ModelStatus.READY.value
    assert res["is_ready"] is True
    assert res["epoch"] == 1
    assert res["metrics"]["val_acc"] == 0.85

    # Should not raise
    path = manager.assert_ready("neuro_convnet")
    assert path.exists()
