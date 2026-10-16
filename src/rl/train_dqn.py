"""Train Stable-Baselines3 DQN policy for simulated MRI review workflow triage.

Training Pipeline:
1. Extract fixed 35-dimensional observations from training cohort using frozen scratch CNNs.
2. Compute feature normalization parameters strictly on the training set (zero test leakage).
3. Instantiate Gymnasium-compliant MRITriageEnv with training observations pool.
4. Validate environment using SB3 check_env.
5. Train SB3 DQN agent with an MLP policy, replay buffer, and epsilon-greedy exploration.
6. Serialize policy checkpoint and training metadata to disk.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from src.utils.config import load_config, resolve_dataset_path, get_project_root
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.data.validator import DatasetValidator
from src.data.transforms import get_transforms
from src.data.constants import CLASSES, CLASS_TO_IDX
from src.models.architectures import create_model
from src.features.radiomics import RadiomicsExtractor
from src.rl.triage_env import MRITriageEnv, OBSERVATION_DIM, OBSERVATION_FEATURES
from src.rl.safety_rules import TriageAction, ACTION_DESCRIPTIONS

logger = get_logger("neuroscan.rl_train")


class ActionFrequencyCallback(BaseCallback):
    """Log action frequencies and episode performance during DQN training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.episode_rewards = []
        self.current_ep_reward = 0.0

    def _on_step(self) -> bool:
        # Track action taken
        actions = self.locals.get("actions")
        if actions is not None:
            a = int(actions[0])
            self.action_counts[a] = self.action_counts.get(a, 0) + 1

        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.current_ep_reward += float(rewards[0])

        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.current_ep_reward = 0.0

        return True


def extract_observation_vector(
    pil_img: Image.Image,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    transforms: Any,
    radiomics_extractor: RadiomicsExtractor,
    scaler_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    """Transform raw MRI slice into standardized 35-dimensional observation vector."""
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

    # 1. Forward passes through member models
    tensor = transforms(pil_img).unsqueeze(0).to(device)
    member_probs = {}
    with torch.no_grad():
        for name in ["custom_cnn", "resnet50", "densenet121"]:
            if name in models:
                logits = models[name](tensor)
                p = F.softmax(logits, dim=1).cpu().numpy()[0]
                member_probs[name] = p
            else:
                member_probs[name] = np.full(4, 0.25, dtype=np.float32)

    # 2. Weighted soft-voting ensemble (using validation-selected weights)
    weights = {"custom_cnn": 0.0, "resnet50": 0.3714, "densenet121": 0.6286}
    ens_probs = np.zeros(4, dtype=np.float32)
    for m_name, w in weights.items():
        ens_probs += member_probs[m_name] * w
    ens_probs = ens_probs / np.sum(ens_probs)

    # 0..3: Ensemble probabilities
    obs[0:4] = ens_probs
    # 4..7: Custom CNN
    obs[4:8] = member_probs["custom_cnn"]
    # 8..11: ResNet-50
    obs[8:12] = member_probs["resnet50"]
    # 12..15: DenseNet-121
    obs[12:16] = member_probs["densenet121"]

    # 16: Max probability
    max_p = float(np.max(ens_probs))
    obs[16] = max_p

    # 17: Predictive entropy
    entropy = float(-np.sum(ens_probs * np.log(ens_probs + 1e-10)))
    obs[17] = entropy

    # 18..20: Disagreement metrics
    from scipy.spatial.distance import jensenshannon
    js_r_d = float(jensenshannon(np.clip(member_probs["resnet50"], 1e-10, 1.0), np.clip(member_probs["densenet121"], 1e-10, 1.0)) ** 2)
    stacked = np.stack(list(member_probs.values()), axis=0)
    p_var = float(np.mean(np.var(stacked, axis=0)))
    cat_dis = 1.0 if len(set(int(np.argmax(p)) for p in member_probs.values())) > 1 else 0.0

    obs[18] = js_r_d
    obs[19] = p_var
    obs[20] = cat_dis

    # 21..29: Radiomics & Quality features
    rad_res = radiomics_extractor.extract_features(pil_img)
    feats = rad_res["features"]

    raw_features = [
        feats["intensity_mean"],
        feats["intensity_std"],
        feats["image_entropy"],
        feats["glcm_contrast"],
        feats["glcm_homogeneity"],
        feats["canny_edge_density"],
        feats["circularity"],
        feats["eccentricity"],
        feats["box_counting_fractal_dimension"],
    ]

    # Normalize if training stats provided
    if scaler_stats:
        norm_features = []
        for i, val in enumerate(raw_features):
            k = f"feat_{i}"
            m = scaler_stats.get(k, {}).get("mean", 0.0)
            s = scaler_stats.get(k, {}).get("std", 1.0)
            norm_features.append((val - m) / (s + 1e-6))
    else:
        norm_features = raw_features

    # 21..23: Quality
    obs[21:24] = norm_features[0:3]
    # 24..29: Radiomics
    obs[24:30] = norm_features[3:9]

    # 30: OOD score proxy (based on entropy & margin)
    sorted_p = np.sort(ens_probs)
    margin = float(sorted_p[-1] - sorted_p[-2])
    ood_score = float(np.clip(1.0 - margin + (entropy / 1.386) * 0.5, 0.0, 1.0))
    obs[30] = ood_score

    # 31..34: Availability indicators
    obs[31] = 1.0 if "custom_cnn" in models else 0.0
    obs[32] = 1.0 if "resnet50" in models else 0.0
    obs[33] = 1.0 if "densenet121" in models else 0.0
    obs[34] = 1.0  # Radiomics available

    # Ensure strictly finite
    obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
    return obs.astype(np.float32)


def generate_cohort_observations(
    records: List[Dict[str, Any]],
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    transforms: Any,
    radiomics_extractor: RadiomicsExtractor,
    scaler_stats: Optional[Dict[str, Dict[str, float]]] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Generate observations from dataset records and compute/apply training scaler."""
    if max_samples and len(records) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(records), size=max_samples, replace=False)
        selected_records = [records[i] for i in indices]
    else:
        selected_records = records

    pool = []
    raw_morph_list = []

    for rec in selected_records:
        p = Path(rec["path"])
        try:
            with Image.open(p) as img:
                pil_img = img.convert("RGB")
        except Exception:
            continue

        true_label = rec["class_index"]
        is_tumor = (true_label != 2)
        is_glioma = (true_label == 0)

        # First pass without scaler to compute stats if scaler_stats is None
        obs = extract_observation_vector(
            pil_img, models, device, transforms, radiomics_extractor, scaler_stats=scaler_stats
        )

        pool.append({
            "observation": obs.tolist(),
            "true_label": true_label,
            "is_tumor": is_tumor,
            "is_glioma": is_glioma,
            "path": str(p),
        })

    # If scaler_stats was None, compute them from the collected observations
    computed_stats = {}
    if scaler_stats is None:
        obs_array = np.array([p["observation"] for p in pool])
        # Indices 21..29 are the 9 morphological features
        for i, idx in enumerate(range(21, 30)):
            vals = obs_array[:, idx]
            m = float(np.mean(vals))
            s = float(np.std(vals))
            computed_stats[f"feat_{i}"] = {"mean": m, "std": max(s, 1e-4)}

        # Re-normalize pool with computed stats
        for p_item in pool:
            obs = np.array(p_item["observation"])
            for i, idx in enumerate(range(21, 30)):
                st = computed_stats[f"feat_{i}"]
                obs[idx] = (obs[idx] - st["mean"]) / (st["std"] + 1e-6)
            p_item["observation"] = np.nan_to_num(obs, nan=0.0).tolist()
    else:
        computed_stats = scaler_stats

    return pool, computed_stats


def train_dqn_policy(
    total_timesteps: int = 15000,
    seed: int = 42,
    max_train_samples: int = 600,
) -> Dict[str, Any]:
    """Execute complete DQN training pipeline with Stable-Baselines3."""
    root = get_project_root()
    checkpoints_dir = root / "checkpoints"
    reports_dir = root / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    device = get_device("auto")
    logger.info(f"Training DQN Triage agent on device: {device} with seed: {seed}")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load frozen CNN models
    models = {}
    for name in ["custom_cnn", "resnet50", "densenet121"]:
        ckpt_path = checkpoints_dir / f"{name}_best.pt"
        if ckpt_path.exists():
            m = create_model(name, num_classes=len(CLASSES))
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
            m.load_state_dict(state_dict)
            m.to(device)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            models[name] = m

    logger.info(f"Loaded {len(models)} frozen CNN feature extractors: {list(models.keys())}")

    # Load training dataset records
    dataset_path = resolve_dataset_path(config)
    validator = DatasetValidator(
        dataset_dir=dataset_path,
        val_split_ratio=config.get("dataset", {}).get("validation_split_ratio", 0.20),
        seed=seed,
    )
    scan_res = validator.scan_dataset()
    records = validator.partition_splits(scan_res["records"])
    train_records = [r for r in records if r["split"] == "train"]

    logger.info(f"Total training pool size: {len(train_records)}. Sampling {max_train_samples} cases...")
    norm_file = reports_dir / "dataset_stats.json"
    if norm_file.exists():
        with open(norm_file, "r") as f:
            stats = json.load(f)
            mean = stats.get("mean", [0.172, 0.172, 0.172])
            std = stats.get("std", [0.1868, 0.1868, 0.1868])
    else:
        mean = [0.172, 0.172, 0.172]
        std = [0.1868, 0.1868, 0.1868]

    transforms = get_transforms(split="val", crop_contour=True, mean=mean, std=std)
    radiomics = RadiomicsExtractor()

    # Generate training observations & compute scaler strictly on training split
    train_pool, scaler_stats = generate_cohort_observations(
        records=train_records,
        models=models,
        device=device,
        transforms=transforms,
        radiomics_extractor=radiomics,
        scaler_stats=None,  # Computes and freezes on train
        max_samples=max_train_samples,
    )
    logger.info(f"Generated {len(train_pool)} training observations.")

    # Save frozen observation scaler
    scaler_path = reports_dir / "rl_observation_scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_stats, f, indent=2)
    logger.info(f"Frozen training scaler saved to: {scaler_path}")

    # Save precomputed training observations cache
    obs_cache_path = checkpoints_dir / "rl_train_pool.json"
    with open(obs_cache_path, "w", encoding="utf-8") as f:
        json.dump(train_pool, f)

    # Instantiate Gymnasium environment
    env = MRITriageEnv(dataset_pool=train_pool, max_steps=100, seed=seed)
    
    # SB3 Environment Checker Validation
    check_env(env)
    logger.info("Gymnasium MRITriageEnv passed Stable-Baselines3 check_env successfully.")

    # Configure DQN Hyperparameters
    dqn_kwargs = {
        "learning_rate": 5e-4,
        "buffer_size": 20000,
        "learning_starts": 500,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.95,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 250,
        "exploration_fraction": 0.40,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "seed": seed,
        "verbose": 0,
    }

    model = DQN(
        policy="MlpPolicy",
        env=env,
        **dqn_kwargs,
    )

    callback = ActionFrequencyCallback()
    logger.info(f"Training SB3 DQN policy for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save trained policy
    model_save_path = checkpoints_dir / "dqn_triage_policy.zip"
    model.save(str(model_save_path))
    logger.info(f"DQN policy successfully saved to: {model_save_path}")

    # Save training configuration and logs
    training_summary = {
        "timesteps": total_timesteps,
        "seed": seed,
        "observation_dim": OBSERVATION_DIM,
        "action_space": [a["name"] for a in ACTION_DESCRIPTIONS.values()],
        "hyperparameters": {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in dqn_kwargs.items()},
        "action_counts": callback.action_counts,
        "episodes_completed": len(callback.episode_rewards),
        "mean_episode_reward": float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0.0,
        "policy_path": str(model_save_path),
        "scaler_path": str(scaler_path),
    }

    config_log_path = reports_dir / "dqn_training_summary.json"
    with open(config_log_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    logger.info(f"Training summary saved to: {config_log_path}")

    return training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SB3 DQN for MRI Review Triage")
    parser.add_argument("--timesteps", type=int, default=15000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=600, help="Max train cases to pool")
    args = parser.parse_args()
    train_dqn_policy(total_timesteps=args.timesteps, seed=args.seed, max_train_samples=args.max_samples)
