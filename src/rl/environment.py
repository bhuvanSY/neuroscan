"""Gymnasium Clinical MRI Triage Environment for NeuroScan AI.

Models clinical escalation actions based on classifier confidence, predictive entropy,
and tumor malignancy risk.
"""

from enum import IntEnum
from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TriageAction(IntEnum):
    IMMEDIATE_URGENT_REVIEW = 0      # Stat review by emergency neuroradiologist
    ROUTINE_QUEUE = 1                # Standard elective reading list
    REQUEST_CONTRAST_SERIES = 2      # Order complementary contrast MRI
    DISMISS_NORMAL = 3               # Routine negative study release


class MRITriageEnv(gym.Env):
    """Gymnasium environment simulating hospital radiologic MRI triage decision-making."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: Optional[int] = 42):
        super().__init__()
        
        # 4 Discrete triage actions
        self.action_space = spaces.Discrete(len(TriageAction))
        
        # Observation vector: [P(glioma), P(meningioma), P(notumor), P(pituitary), max_conf, entropy, clinical_urgency]
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(7,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 100
        self.current_state: Optional[np.ndarray] = None
        self.current_ground_truth: int = 0
        self._np_random = np.random.RandomState(seed)

    def _generate_synthetic_case(self) -> Tuple[np.ndarray, int]:
        """Generate a simulated case observation for policy training/testing."""
        gt_class = self._np_random.choice(4)  # 0: glioma, 1: meningioma, 2: notumor, 3: pituitary
        
        # Simulate noisy model prediction
        logits = self._np_random.exponential(scale=1.0, size=4)
        logits[gt_class] += self._np_random.uniform(1.5, 4.0)  # bias towards ground truth
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        max_conf = float(np.max(probs))
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        clinical_urgency = float(self._np_random.uniform(0.1, 1.0))
        
        obs = np.array([
            probs[0], probs[1], probs[2], probs[3],
            max_conf, entropy, clinical_urgency
        ], dtype=np.float32)
        
        return obs, gt_class

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to a new patient case."""
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
            
        self.current_step = 0
        obs, gt = self._generate_synthetic_case()
        self.current_state = obs
        self.current_ground_truth = gt
        
        info = {
            "step": self.current_step,
            "ground_truth_label": gt,
            "is_tumor": (gt != 2),
        }
        return self.current_state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one triage action step."""
        action = TriageAction(action)
        is_tumor = (self.current_ground_truth != 2)
        is_glioma = (self.current_ground_truth == 0)
        
        reward = 0.0
        # Reward design penalizing under-triage of malignancies severely
        if is_glioma:
            if action == TriageAction.IMMEDIATE_URGENT_REVIEW:
                reward = 10.0
            elif action == TriageAction.ROUTINE_QUEUE:
                reward = -5.0
            elif action == TriageAction.REQUEST_CONTRAST_SERIES:
                reward = 5.0
            elif action == TriageAction.DISMISS_NORMAL:
                reward = -30.0  # Catastrophic miss
        elif is_tumor:
            if action in [TriageAction.IMMEDIATE_URGENT_REVIEW, TriageAction.ROUTINE_QUEUE]:
                reward = 5.0
            elif action == TriageAction.REQUEST_CONTRAST_SERIES:
                reward = 3.0
            elif action == TriageAction.DISMISS_NORMAL:
                reward = -20.0
        else:  # No tumor (Normal)
            if action == TriageAction.DISMISS_NORMAL:
                reward = 5.0
            elif action == TriageAction.ROUTINE_QUEUE:
                reward = 1.0
            elif action in [TriageAction.IMMEDIATE_URGENT_REVIEW, TriageAction.REQUEST_CONTRAST_SERIES]:
                reward = -3.0  # False alarm / unnecessary resource cost

        self.current_step += 1
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        # Advance to next simulated patient
        next_obs, next_gt = self._generate_synthetic_case()
        self.current_state = next_obs
        self.current_ground_truth = next_gt
        
        info = {
            "step": self.current_step,
            "action_taken": action.name,
            "reward": reward,
        }
        
        return self.current_state, reward, terminated, truncated, info

    def render(self):
        """Render step info to console."""
        print(f"Step {self.current_step}: State {self.current_state[:4]} | GT: {self.current_ground_truth}")
