"""Benchmark evaluation comparing RL policy against fixed-threshold and rule-based triage baselines.

Evaluates on held-out test cohort:
1. Fixed-Threshold Baseline: Simple confidence & entropy cutoffs.
2. Rule-Based Clinical Safety Policy: Multi-tier safety hierarchy with glioma & uncertainty escalation.
3. Trained SB3 DQN Policy: Learned decision policy balancing workflow efficiency with clinical safety.

Safety & Methodological Constraints:
- Evaluates strictly on frozen test split observations.
- Uses frozen scaler parameters derived from training data only (zero test leakage).
- Does not reward or guide agent using test labels at inference time.
- Clearly states that this is a simulated triage research prototype, NOT a clinically validated device.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN

from src.utils.config import load_config, resolve_dataset_path, get_project_root
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.data.validator import DatasetValidator
from src.data.transforms import get_transforms
from src.data.constants import CLASSES, CLASS_TO_IDX
from src.models.architectures import create_model
from src.features.radiomics import RadiomicsExtractor
from src.rl.triage_env import MRITriageEnv, OBSERVATION_DIM
from src.rl.safety_rules import (
    TriageAction,
    rule_based_triage_policy,
    fixed_threshold_baseline_policy,
    ACTION_DESCRIPTIONS,
)
from src.rl.train_dqn import generate_cohort_observations

logger = get_logger("neuroscan.rl_evaluate")

DISCLAIMER_TEXT = (
    "This triage evaluation is a simulation-based algorithmic benchmark. "
    "In the absence of prospectively validated multicenter clinical trials and board-certified expert action labels, "
    "this system serves as a decision-support prototype and must NOT be used for autonomous medical triage."
)


def evaluate_policy_on_cohort(
    policy_fn: Any,
    test_pool: List[Dict[str, Any]],
    env: MRITriageEnv,
    policy_name: str,
) -> Dict[str, Any]:
    """Evaluate a policy function or SB3 model over a pool of test cases."""
    action_counts = {0: 0, 1: 0, 2: 0}
    total_rewards = []
    tumor_cases = 0
    glioma_cases = 0
    inappropriate_auto_reports_on_tumors = 0
    inappropriate_auto_reports_on_gliomas = 0
    escalated_tumors = 0
    escalated_gliomas = 0
    second_view_tumors = 0

    for item in test_pool:
        obs = np.array(item["observation"], dtype=np.float32)
        true_label = item["true_label"]
        is_tumor = item["is_tumor"]
        is_glioma = item["is_glioma"]

        if is_tumor:
            tumor_cases += 1
        if is_glioma:
            glioma_cases += 1

        # Determine action
        if hasattr(policy_fn, "predict"):
            action, _ = policy_fn.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = int(policy_fn(obs))

        action_counts[action] = action_counts.get(action, 0) + 1

        # Simulated review reward
        meta = {
            "true_label": true_label,
            "is_tumor": is_tumor,
            "is_glioma": is_glioma,
            "entropy": float(obs[17]),
            "max_prob": float(obs[16]),
        }
        r = env.calculate_reward(action, meta)
        total_rewards.append(r)

        # Safety tracking
        if action == TriageAction.AUTO_REPORT:
            if is_tumor:
                inappropriate_auto_reports_on_tumors += 1
            if is_glioma:
                inappropriate_auto_reports_on_gliomas += 1
        elif action == TriageAction.ESCALATE:
            if is_tumor:
                escalated_tumors += 1
            if is_glioma:
                escalated_gliomas += 1
        elif action == TriageAction.SECOND_VIEW:
            if is_tumor:
                second_view_tumors += 1

    n_samples = len(test_pool)
    auto_report_rate = action_counts[0] / n_samples
    second_view_rate = action_counts[1] / n_samples
    escalation_rate = action_counts[2] / n_samples
    mean_reward = float(np.mean(total_rewards))

    # Safety metrics
    tumor_catch_rate = (escalated_tumors + second_view_tumors) / max(tumor_cases, 1)
    glioma_escalation_rate = escalated_gliomas / max(glioma_cases, 1)
    tumor_leak_rate = inappropriate_auto_reports_on_tumors / max(tumor_cases, 1)

    return {
        "policy_name": policy_name,
        "n_samples": n_samples,
        "mean_simulated_reward": round(mean_reward, 3),
        "auto_report_rate": round(auto_report_rate, 4),
        "second_view_rate": round(second_view_rate, 4),
        "escalation_rate": round(escalation_rate, 4),
        "action_counts": {ACTION_DESCRIPTIONS[k]["name"]: v for k, v in action_counts.items()},
        "tumor_catch_rate": round(tumor_catch_rate, 4),
        "glioma_escalation_rate": round(glioma_escalation_rate, 4),
        "tumor_leak_rate": round(tumor_leak_rate, 4),
        "inappropriate_tumor_auto_reports": inappropriate_auto_reports_on_tumors,
        "inappropriate_glioma_auto_reports": inappropriate_auto_reports_on_gliomas,
    }


def run_rl_triage_evaluation(max_test_samples: int = 400) -> Dict[str, Any]:
    """Execute evaluation comparing fixed-threshold, rule-based, and DQN policies."""
    root = get_project_root()
    checkpoints_dir = root / "checkpoints"
    reports_dir = root / "reports"

    device = get_device("auto")
    config = load_config()

    # Load frozen observation scaler (strictly from training set)
    scaler_path = reports_dir / "rl_observation_scaler.json"
    if scaler_path.exists():
        with open(scaler_path, "r") as f:
            scaler_stats = json.load(f)
    else:
        logger.warning("Observation scaler not found, using unscaled features.")
        scaler_stats = None

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
            models[name] = m

    # Load held-out test split records
    dataset_path = resolve_dataset_path(config)
    validator = DatasetValidator(
        dataset_dir=dataset_path,
        val_split_ratio=config.get("dataset", {}).get("validation_split_ratio", 0.20),
        seed=42,
    )
    scan_res = validator.scan_dataset()
    records = validator.partition_splits(scan_res["records"])
    test_records = [r for r in records if r["split"] == "test"]

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

    logger.info(f"Generating test observations for {min(len(test_records), max_test_samples)} held-out cases...")
    test_pool, _ = generate_cohort_observations(
        records=test_records,
        models=models,
        device=device,
        transforms=transforms,
        radiomics_extractor=radiomics,
        scaler_stats=scaler_stats,
        max_samples=max_test_samples,
    )
    logger.info(f"Successfully generated {len(test_pool)} test observations.")

    # Base evaluation environment
    env = MRITriageEnv()

    # 1. Evaluate Fixed-Threshold Baseline
    res_fixed = evaluate_policy_on_cohort(
        fixed_threshold_baseline_policy, test_pool, env, "Fixed-Threshold Baseline"
    )

    # 2. Evaluate Rule-Based Safety Policy
    res_rules = evaluate_policy_on_cohort(
        rule_based_triage_policy, test_pool, env, "Rule-Based Safety Policy"
    )

    # 3. Evaluate Trained SB3 DQN Policy
    dqn_model_path = checkpoints_dir / "dqn_triage_policy.zip"
    if dqn_model_path.exists():
        dqn_model = DQN.load(str(dqn_model_path))
        res_dqn = evaluate_policy_on_cohort(dqn_model, test_pool, env, "Learned SB3 DQN Policy")
    else:
        logger.warning(f"DQN policy checkpoint not found at {dqn_model_path}. Skipping DQN.")
        res_dqn = None

    results_list = [res_fixed, res_rules]
    if res_dqn:
        results_list.append(res_dqn)

    # Generate Comparison DataFrame and CSV
    df = pd.DataFrame(results_list)
    csv_path = reports_dir / "rl_triage_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"RL policy comparison CSV saved to: {csv_path}")

    # Generate JSON summary
    json_path = reports_dir / "rl_triage_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"disclaimer": DISCLAIMER_TEXT, "policies": results_list}, f, indent=2)

    # Generate Markdown Report
    md_path = reports_dir / "rl_triage_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NeuroScan AI - Simulated MRI Review Triage Policy Evaluation\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write(f"> **Clinical Disclaimer**: {DISCLAIMER_TEXT}\n\n")

        f.write("## 1. Executive Policy Comparison on Held-Out Test Scans\n\n")
        f.write(f"- **Evaluated Cohort**: `{len(test_pool)}` held-out brain MRI scans.\n")
        f.write(f"- **Zero Data Leakage Guarantee**: Normalization statistics were frozen strictly from the training cohort.\n\n")

        f.write("### Policy Benchmark Matrix\n\n")
        cols_to_show = [
            "policy_name", "mean_simulated_reward", "auto_report_rate",
            "second_view_rate", "escalation_rate", "tumor_catch_rate",
            "glioma_escalation_rate", "tumor_leak_rate"
        ]
        f.write(df[cols_to_show].to_markdown(index=False) + "\n\n")

        f.write("## 2. Detailed Action Breakdown & Safety Metrics\n\n")
        for res in results_list:
            f.write(f"### {res['policy_name']}\n\n")
            f.write(f"- **Mean Simulated Episode Reward**: `{res['mean_simulated_reward']}`\n")
            f.write(f"- **Action Distribution**: `{res['action_counts']}`\n")
            f.write(f"- **Tumor Safety Catch Rate**: `{res['tumor_catch_rate']:.2%}` (Scans routed to SECOND_VIEW or ESCALATE)\n")
            f.write(f"- **Glioma Escalation Rate**: `{res['glioma_escalation_rate']:.2%}`\n")
            f.write(f"- **Tumor Auto-Report Leakage Rate**: `{res['tumor_leak_rate']:.2%}` ({res['inappropriate_tumor_auto_reports']} tumors erroneously auto-reported)\n\n")

        f.write("## 3. Workflow Action Definitions & Clinical Constraints\n\n")
        for act in ACTION_DESCRIPTIONS.values():
            f.write(f"- **{act['name']}**: {act['description']} *({act['workflow_impact']})*\n")
        f.write("\n")

    logger.info(f"RL triage report generated at: {md_path}")
    return {"df": df, "results": results_list, "csv_path": str(csv_path), "report_path": str(md_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MRI Triage Policies")
    parser.add_argument("--samples", type=int, default=400, help="Max test samples to evaluate")
    args = parser.parse_args()
    run_rl_triage_evaluation(max_test_samples=args.samples)
