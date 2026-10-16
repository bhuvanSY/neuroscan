"""Smoke tests for Gymnasium clinical MRI triage environment."""

import numpy as np
import pytest
from src.rl.environment import MRITriageEnv, TriageAction


def test_rl_environment_reset():
    """Verify Gymnasium triage environment reset outputs valid state."""
    env = MRITriageEnv(seed=42)
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (7,)
    assert obs.dtype == np.float32
    assert "ground_truth_label" in info
    assert "is_tumor" in info


def test_rl_environment_step():
    """Verify step transitions across all 4 discrete actions."""
    env = MRITriageEnv(seed=42)
    env.reset(seed=42)

    for action in TriageAction:
        obs, reward, terminated, truncated, info = env.step(int(action))
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "action_taken" in info
        assert info["action_taken"] == action.name
