"""Unit tests for Gymnasium triage environment, safety rules, reward design, and DQN serialization."""

import pytest
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from src.rl.triage_env import (
    MRITriageEnv,
    OBSERVATION_DIM,
    OBSERVATION_FEATURES,
)
from src.rl.safety_rules import (
    TriageAction,
    rule_based_triage_policy,
    fixed_threshold_baseline_policy,
    ACTION_DESCRIPTIONS,
)


class TestTriageEnvironment:
    """Test Gymnasium MRITriageEnv specification, spaces, reset, and step transitions."""

    @pytest.fixture
    def env(self):
        return MRITriageEnv(seed=42)

    def test_observation_and_action_spaces(self, env):
        """Observation space must be Box(35,) and action space must be Discrete(3)."""
        assert OBSERVATION_DIM == 35
        assert len(OBSERVATION_FEATURES) == 35
        assert env.observation_space.shape == (35,)
        assert env.observation_space.dtype == np.float32

        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 3

    def test_sb3_env_checker_compliance(self, env):
        """Environment must satisfy Stable-Baselines3 check_env without error."""
        check_env(env)

    def test_reset_behavior_and_finite_observations(self, env):
        """Reset must produce a finite 35-dim observation vector and valid info dict."""
        obs, info = env.reset(seed=123)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (35,)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all(), "Observation contains NaN or Infinite values"
        assert -10.0 <= obs.min() and obs.max() <= 10.0

        assert "step" in info
        assert "metadata" in info
        assert info["step"] == 0

    def test_step_transitions_all_valid_actions(self, env):
        """Step must transition across all 3 discrete actions returning 5-tuple."""
        env.reset(seed=42)

        for act in [TriageAction.AUTO_REPORT, TriageAction.SECOND_VIEW, TriageAction.ESCALATE]:
            obs, reward, terminated, truncated, info = env.step(int(act))

            assert isinstance(obs, np.ndarray)
            assert obs.shape == (35,)
            assert np.isfinite(obs).all()
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert not truncated
            assert "action" in info
            assert info["action"] == int(act)
            assert info["action_name"] == ACTION_DESCRIPTIONS[int(act)]["name"]

    def test_invalid_action_raises(self, env):
        """Passing an action outside Discrete(3) must raise ValueError."""
        env.reset(seed=42)
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(99)

    def test_episode_termination(self):
        """Reaching max_steps must flag terminated=True."""
        env = MRITriageEnv(max_steps=5, seed=42)
        env.reset(seed=42)

        for s in range(4):
            _, _, terminated, _, _ = env.step(1)
            assert not terminated

        _, _, terminated, _, info = env.step(1)
        assert terminated is True
        assert info["step"] == 5


class TestRewardCalculationsAndSafetyConstraints:
    """Test clinical review reward function and safety violation penalties."""

    @pytest.fixture
    def env(self):
        return MRITriageEnv(seed=42)

    def test_severe_penalty_auto_reporting_glioma(self, env):
        """Auto-reporting an untreated malignant glioma must trigger critical safety penalty."""
        meta_glioma = {
            "true_label": 0,
            "is_tumor": True,
            "is_glioma": True,
            "entropy": 0.50,
            "max_prob": 0.85,
        }
        reward = env.calculate_reward(TriageAction.AUTO_REPORT, meta_glioma)
        # Must be severely negative (<= -25.0)
        assert reward <= -25.0

    def test_efficiency_reward_confident_normal(self, env):
        """Auto-reporting a clear high-confidence normal scan must yield positive efficiency reward."""
        meta_normal = {
            "true_label": 2,
            "is_tumor": False,
            "is_glioma": False,
            "entropy": 0.10,
            "max_prob": 0.98,
        }
        reward = env.calculate_reward(TriageAction.AUTO_REPORT, meta_normal)
        assert reward > 0.0

    def test_reward_escalating_glioma(self, env):
        """Escalating a malignant glioma to a specialist must yield high safety reward."""
        meta_glioma = {
            "true_label": 0,
            "is_tumor": True,
            "is_glioma": True,
            "entropy": 0.40,
            "max_prob": 0.80,
        }
        reward = env.calculate_reward(TriageAction.ESCALATE, meta_glioma)
        assert reward >= 5.0

    def test_reward_second_view_on_uncertain_case(self, env):
        """Requesting second view on an ambiguous study must yield net positive gain."""
        meta_ambiguous = {
            "true_label": 1,
            "is_tumor": True,
            "is_glioma": False,
            "entropy": 0.80,
            "max_prob": 0.60,
        }
        reward = env.calculate_reward(TriageAction.SECOND_VIEW, meta_ambiguous)
        assert reward > 0.0


class TestSafetyRulesAndBaselines:
    """Test deterministic clinical safety rule hierarchy and fixed-threshold baseline."""

    def test_rule_based_policy_escalates_glioma(self):
        """Observation with elevated glioma probability must be escalated immediately."""
        obs = np.zeros(35, dtype=np.float32)
        obs[0] = 0.55  # ens_p_glioma = 55%
        obs[16] = 0.55
        obs[17] = 0.80

        action = rule_based_triage_policy(obs)
        assert action == TriageAction.ESCALATE

    def test_rule_based_policy_escalates_high_disagreement(self):
        """High model disagreement must trigger escalation."""
        obs = np.zeros(35, dtype=np.float32)
        obs[16] = 0.70
        obs[17] = 0.40
        obs[18] = 0.12  # High JS divergence
        obs[20] = 1.0   # Categorical disagreement

        action = rule_based_triage_policy(obs)
        assert action == TriageAction.ESCALATE

    def test_rule_based_policy_auto_reports_concordant_normal(self):
        """High confidence, low entropy, zero disagreement scan qualifies for AUTO_REPORT."""
        obs = np.zeros(35, dtype=np.float32)
        obs[2] = 0.96   # No tumor
        obs[16] = 0.96  # Max prob
        obs[17] = 0.15  # Low entropy
        obs[20] = 0.0   # Zero disagreement

        action = rule_based_triage_policy(obs)
        assert action == TriageAction.AUTO_REPORT

    def test_fixed_threshold_baseline(self):
        """Fixed threshold policy routes above cutoff to AUTO_REPORT, otherwise ESCALATE/SECOND_VIEW."""
        obs_confident = np.zeros(35, dtype=np.float32)
        obs_confident[16] = 0.98
        obs_confident[17] = 0.10
        assert fixed_threshold_baseline_policy(obs_confident) == TriageAction.AUTO_REPORT

        obs_low_conf = np.zeros(35, dtype=np.float32)
        obs_low_conf[16] = 0.60
        obs_low_conf[17] = 0.80
        assert fixed_threshold_baseline_policy(obs_low_conf) == TriageAction.ESCALATE


class TestPolicySerialization:
    """Test DQN policy training loop, serialization to zip, and re-loading."""

    def test_dqn_training_and_serialization(self, tmp_path):
        """Train a lightweight DQN agent for 100 steps, save, and reload."""
        env = MRITriageEnv(max_steps=20, seed=42)

        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=1000,
            learning_starts=20,
            batch_size=16,
            train_freq=2,
            target_update_interval=50,
            seed=42,
            verbose=0,
        )

        model.learn(total_timesteps=100)

        # Save model
        save_path = tmp_path / "test_dqn_policy.zip"
        model.save(str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

        # Reload model
        loaded_model = DQN.load(str(save_path))
        test_obs = np.zeros(35, dtype=np.float32)
        test_obs[16] = 0.85
        action, _ = loaded_model.predict(test_obs, deterministic=True)

        assert int(action) in [0, 1, 2]
