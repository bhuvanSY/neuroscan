"""Gymnasium-compatible simulated MRI review workflow environment for NeuroScan AI.

State Vector: Fixed 35-dimensional numeric observation vector capturing:
- Ensemble predictions & uncertainty
- Individual model predictions & inter-model disagreement
- Image quality & radiomics morphology
- OOD score & feature availability flags

Action Space: Discrete(3):
- 0: AUTO_REPORT (Draft report for expedited clinician sign-off; never autonomous release)
- 1: SECOND_VIEW (Request complementary view or peer review)
- 2: ESCALATE (Route to urgent specialist neuroradiologist review)

Reward Structure:
- Penalizes inappropriate auto-handling of high-risk or uncertain scans severely (-25.0)
- Encourages timely specialist escalation for malignancies (+8.0)
- Appropriately accounts for review costs (second view: -1.0, specialist: -2.0)
- Yields modest workflow efficiency rewards for valid low-risk cases (+2.0)
"""

from typing import Dict, Any, Tuple, Optional, List, Sequence
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.rl.safety_rules import TriageAction, ACTION_DESCRIPTIONS

OBSERVATION_DIM = 35

OBSERVATION_FEATURES: List[str] = [
    # 0..3: Ensemble probabilities
    "ens_prob_glioma",
    "ens_prob_meningioma",
    "ens_prob_no_tumor",
    "ens_prob_pituitary",
    # 4..7: Custom CNN member probabilities
    "custom_prob_glioma",
    "custom_prob_meningioma",
    "custom_prob_no_tumor",
    "custom_prob_pituitary",
    # 8..11: ResNet-50 member probabilities
    "resnet_prob_glioma",
    "resnet_prob_meningioma",
    "resnet_prob_no_tumor",
    "resnet_prob_pituitary",
    # 12..15: DenseNet-121 member probabilities
    "dense_prob_glioma",
    "dense_prob_meningioma",
    "dense_prob_no_tumor",
    "dense_prob_pituitary",
    # 16: Top confidence
    "max_probability",
    # 17: Predictive Shannon entropy
    "predictive_entropy",
    # 18..20: Disagreement metrics
    "disagreement_js_divergence",
    "disagreement_variance",
    "disagreement_categorical",
    # 21..23: Image quality features (normalized)
    "img_intensity_mean",
    "img_intensity_std",
    "img_entropy",
    # 24..29: Selected Radiomic features (normalized)
    "radiomic_glcm_contrast",
    "radiomic_glcm_homogeneity",
    "radiomic_canny_edge_density",
    "radiomic_circularity",
    "radiomic_eccentricity",
    "radiomic_fractal_dimension",
    # 30: Out-of-Distribution (OOD) score proxy
    "ood_score",
    # 31..34: Availability indicator flags
    "avail_custom_cnn",
    "avail_resnet50",
    "avail_densenet121",
    "avail_radiomics",
]


class MRITriageEnv(gym.Env):
    """Gymnasium environment modeling clinical radiologic review triage decisions."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset_pool: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = 100,
        reward_config: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.dataset_pool = dataset_pool
        self.max_steps = max_steps
        self.seed(seed)

        # 3 Discrete workflow decisions
        self.action_space = spaces.Discrete(3)

        # 35-dimensional standardized observation space
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        # Configurable review reward parameters
        default_rewards = {
            "efficiency_auto_report": 2.0,
            "cost_second_view": -1.0,
            "cost_escalate": -2.0,
            "penalty_missed_tumor": -25.0,
            "reward_escalate_malignancy": 8.0,
            "reward_second_view_uncertainty": 4.0,
            "penalty_unnecessary_escalation": -2.0,
        }
        self.reward_config = default_rewards
        if reward_config:
            self.reward_config.update(reward_config)

        self.current_step = 0
        self.current_obs: Optional[np.ndarray] = None
        self.current_case_meta: Optional[Dict[str, Any]] = None

    def seed(self, seed: Optional[int] = 42) -> None:
        """Seed the random number generator."""
        self._np_random = np.random.RandomState(seed)

    def _generate_synthetic_observation(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a realistic synthetic observation for testing/verification."""
        # 0: glioma, 1: meningioma, 2: no_tumor, 3: pituitary
        true_label = self._np_random.choice(4)
        is_tumor = (true_label != 2)
        is_glioma = (true_label == 0)

        # Simulate noisy member predictions
        member_probs = []
        for _ in range(3):
            logits = self._np_random.exponential(scale=1.0, size=4)
            logits[true_label] += self._np_random.uniform(1.5, 4.0)
            p = np.exp(logits) / np.sum(np.exp(logits))
            member_probs.append(p)

        # Ensemble
        ens_p = np.mean(member_probs, axis=0)
        max_prob = float(np.max(ens_p))
        entropy = float(-np.sum(ens_p * np.log(ens_p + 1e-10)))

        # Disagreement
        cat_dis = 1.0 if len(set(int(np.argmax(p)) for p in member_probs)) > 1 else 0.0
        prob_var = float(np.mean(np.var(member_probs, axis=0)))
        js_div = float(self._np_random.uniform(0.01, 0.15)) if cat_dis > 0.5 else float(self._np_random.uniform(0.0, 0.03))

        # Synthetic radiomics (normalized around zero)
        img_stats = self._np_random.randn(3).astype(np.float32) * 0.5
        radiomic_stats = self._np_random.randn(6).astype(np.float32) * 0.5
        ood_score = float(np.clip(entropy / 1.38 + self._np_random.randn() * 0.1, 0.0, 1.0))

        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        # 0..3: Ensemble
        obs[0:4] = ens_p
        # 4..15: Members
        obs[4:8] = member_probs[0]
        obs[8:12] = member_probs[1]
        obs[12:16] = member_probs[2]
        # 16..17: Top & Entropy
        obs[16] = max_prob
        obs[17] = entropy
        # 18..20: Disagreement
        obs[18] = js_div
        obs[19] = prob_var
        obs[20] = cat_dis
        # 21..23: Quality
        obs[21:24] = img_stats
        # 24..29: Radiomics
        obs[24:30] = radiomic_stats
        # 30: OOD
        obs[30] = ood_score
        # 31..34: Availability flags
        obs[31:35] = 1.0

        meta = {
            "true_label": true_label,
            "is_tumor": is_tumor,
            "is_glioma": is_glioma,
            "entropy": entropy,
            "max_prob": max_prob,
        }
        return obs, meta

    def _sample_case(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Retrieve next case from dataset pool or generate synthetic observation."""
        if self.dataset_pool and len(self.dataset_pool) > 0:
            idx = self._np_random.choice(len(self.dataset_pool))
            item = self.dataset_pool[idx]
            obs = np.array(item["observation"], dtype=np.float32)
            meta = {
                "true_label": item.get("true_label", 0),
                "is_tumor": item.get("is_tumor", True),
                "is_glioma": item.get("is_glioma", False),
                "entropy": float(obs[17]),
                "max_prob": float(obs[16]),
            }
            return obs, meta
        return self._generate_synthetic_observation()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to a new patient case."""
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.current_obs, self.current_case_meta = self._sample_case()

        # Sanitize observation: ensure finite
        self.current_obs = np.nan_to_num(self.current_obs, nan=0.0, posinf=5.0, neginf=-5.0)
        self.current_obs = np.clip(self.current_obs, -10.0, 10.0).astype(np.float32)

        info = {
            "step": self.current_step,
            "metadata": self.current_case_meta,
        }
        return self.current_obs, info

    def calculate_reward(self, action: int, meta: Dict[str, Any]) -> float:
        """Calculate clinical utility reward incorporating safety penalties and review costs.
        
        Args:
            action: TriageAction (0: AUTO_REPORT, 1: SECOND_VIEW, 2: ESCALATE)
            meta: Dictionary with true_label, is_tumor, is_glioma, entropy, max_prob
            
        Returns:
            Scalar reward.
        """
        is_tumor = meta["is_tumor"]
        is_glioma = meta["is_glioma"]
        entropy = meta["entropy"]
        max_prob = meta["max_prob"]
        cfg = self.reward_config

        # -------------------------------------------------------------
        # Action 0: AUTO_REPORT
        # -------------------------------------------------------------
        if action == TriageAction.AUTO_REPORT:
            if not is_tumor and max_prob >= 0.90 and entropy <= 0.35:
                # Confident, clear normal scan: efficient workflow reward
                return cfg["efficiency_auto_report"]
            elif is_tumor:
                # Catastrophic safety violation: auto-reporting suspected tumor without review
                if is_glioma:
                    return cfg["penalty_missed_tumor"] * 1.5  # Critical malignancy
                return cfg["penalty_missed_tumor"]
            else:
                # Uncertain normal scan auto-reported: moderate penalty
                return cfg["penalty_missed_tumor"] * 0.4

        # -------------------------------------------------------------
        # Action 1: SECOND_VIEW
        # -------------------------------------------------------------
        elif action == TriageAction.SECOND_VIEW:
            review_cost = cfg["cost_second_view"]
            # Reward second view when study has elevated uncertainty or borderline probability
            if entropy > 0.30 or max_prob < 0.92:
                return review_cost + cfg["reward_second_view_uncertainty"]
            # Second view on obvious high-confidence normal: minor penalty
            return review_cost - 0.5

        # -------------------------------------------------------------
        # Action 2: ESCALATE
        # -------------------------------------------------------------
        elif action == TriageAction.ESCALATE:
            review_cost = cfg["cost_escalate"]
            if is_glioma:
                # Correct timely escalation of malignant glioma: major safety reward
                return review_cost + cfg["reward_escalate_malignancy"]
            elif is_tumor:
                # Correct escalation of meningioma or pituitary lesion
                return review_cost + cfg["reward_escalate_malignancy"] * 0.75
            else:
                # False alarm on obvious normal scan: review resource cost
                if max_prob >= 0.95 and entropy <= 0.20:
                    return review_cost + cfg["penalty_unnecessary_escalation"]
                # Cautious escalation on ambiguous non-tumor: neutral/slight cost
                return review_cost

        return 0.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one simulated review triage step."""
        action_int = int(action)
        if not self.action_space.contains(action_int):
            raise ValueError(f"Invalid action {action_int} for Discrete(3) space.")

        # Compute reward
        reward = self.calculate_reward(action_int, self.current_case_meta)

        self.current_step += 1
        terminated = (self.current_step >= self.max_steps)
        truncated = False

        info = {
            "step": self.current_step,
            "action": action_int,
            "action_name": ACTION_DESCRIPTIONS[action_int]["name"],
            "reward": reward,
            "case_meta": self.current_case_meta,
        }

        # Advance to next observation
        self.current_obs, self.current_case_meta = self._sample_case()
        self.current_obs = np.nan_to_num(self.current_obs, nan=0.0, posinf=5.0, neginf=-5.0)
        self.current_obs = np.clip(self.current_obs, -10.0, 10.0).astype(np.float32)

        return self.current_obs, float(reward), terminated, truncated, info

    def render(self):
        """Render step state summary to stdout."""
        act_name = "None"
        print(f"Step {self.current_step}/{self.max_steps} | Obs norm: {np.linalg.norm(self.current_obs):.2f}")
