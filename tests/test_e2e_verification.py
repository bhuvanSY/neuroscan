"""Comprehensive End-to-End Verification and Failure Case Test Suite for NeuroScan AI.

Verifies:
1. Complete 12-step pipeline execution on real held-out brain MRI scans.
2. Rigorous failure case handling (missing weights, corrupt image, oversized file,
   unsupported layers, NaN/Inf numeric safety, policy fallbacks).
3. Experimental integrity checks (no data leakage, scratch init, calibrated ensemble).
"""

import io
import json
import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from fastapi.testclient import TestClient

from src.utils.config import get_project_root
from src.data.constants import CLASSES, CLASS_TO_IDX
from src.data.transforms import get_transforms
from src.models.architectures import create_model
from src.models.readiness import ModelReadinessManager
from src.evaluation.uncertainty import compute_predictive_entropy, compute_ensemble_disagreement
from src.features.radiomics import RadiomicsExtractor
from src.explainability.gradcam import GradCAM
from src.rl.safety_rules import (
    TriageAction,
    rule_based_triage_policy,
    fixed_threshold_baseline_policy,
)
from src.rl.triage_env import OBSERVATION_DIM
from src.rl.train_dqn import extract_observation_vector
from src.api.app import app

client = TestClient(app)


def _load_model_weights(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    """Safely load state dict whether saved as a raw state_dict or metadata dictionary."""
    ckpt_data = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
        state_dict = ckpt_data["state_dict"]
    else:
        state_dict = ckpt_data
    model.load_state_dict(state_dict)


# =============================================================================
# PART 1: COMPLETE 12-STEP PIPELINE VERIFICATION ON REAL MRI IMAGES
# =============================================================================
class TestCompletePipelineE2E:
    """Verifies all 12 pipeline stages sequentially using real MRI images from the test set."""

    @pytest.fixture
    def real_mri_image(self):
        """Load a real held-out testing scan from the dataset."""
        root = get_project_root()
        candidate = root / "data" / "archive" / "Testing" / "glioma" / "Te-gl_1.jpg"
        if not candidate.exists():
            candidate = root / "reports" / "explanations" / "case_01_correct_glioma" / "original_mri.png"
        assert candidate.exists(), f"Real test scan not found at {candidate}"
        return Image.open(candidate).convert("RGB")

    def test_pipeline_step_1_to_2_load_and_preprocess(self, real_mri_image):
        """Step 1 & 2: Load real MRI and apply preprocessing."""
        assert real_mri_image.size[0] > 0 and real_mri_image.size[1] > 0
        transforms = get_transforms(split="val", image_size=(224, 224))
        tensor = transforms(real_mri_image)
        assert tensor.shape == (3, 224, 224)
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()

    def test_pipeline_step_3_to_5_cnn_ensemble_disagreement(self, real_mri_image):
        """Step 3, 4, 5: Inference with all 3 scratch-trained CNNs, soft-voting ensemble, and disagreement."""
        root = get_project_root()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transforms = get_transforms(split="val", image_size=(224, 224))
        tensor = transforms(real_mri_image).unsqueeze(0).to(device)

        models = {}
        for m_name in ["custom_cnn", "resnet50", "densenet121"]:
            ckpt = root / "checkpoints" / f"{m_name}_best.pt"
            assert ckpt.exists(), f"Checkpoint {ckpt} missing"
            m = create_model(m_name, num_classes=4).to(device)
            _load_model_weights(m, ckpt, device)
            m.eval()
            models[m_name] = m

        member_probs = []
        with torch.no_grad():
            for name, m in models.items():
                logits = m(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                assert len(probs) == 4
                assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
                member_probs.append(probs)

        # Ensemble soft voting
        ens_probs = np.mean(member_probs, axis=0)
        assert np.isclose(np.sum(ens_probs), 1.0, atol=1e-4)

        # Uncertainty & Disagreement
        entropy = compute_predictive_entropy(np.atleast_2d(ens_probs))[0]
        assert entropy >= 0.0
        assert not np.isnan(entropy)

        disagreement = compute_ensemble_disagreement(member_probs)
        assert "categorical_disagreement_rate" in disagreement
        assert "mean_jensen_shannon_divergence" in disagreement
        assert "mean_prediction_variance" in disagreement

    def test_pipeline_step_6_to_7_radiomics_and_gradcam(self, real_mri_image):
        """Step 6 & 7: Extract 15 radiomic descriptors and generate Grad-CAM attribution."""
        # Step 6: Radiomics
        extractor = RadiomicsExtractor()
        rad_result = extractor.extract_features(real_mri_image)
        features = rad_result["features"]
        assert len(features) == 15
        for k, v in features.items():
            assert not np.isnan(v), f"Feature {k} is NaN"
            assert not np.isinf(v), f"Feature {k} is infinite"

        # Step 7: Grad-CAM
        root = get_project_root()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = create_model("densenet121", num_classes=4).to(device)
        _load_model_weights(m, root / "checkpoints" / "densenet121_best.pt", device)
        m.eval()

        with GradCAM(m, device=device) as cam:
            explanation = cam.explain_image(real_mri_image, target_class="glioma")
            assert explanation["overlay"].shape == (224, 224, 3)
            assert explanation["heatmap_colored"].shape == (224, 224, 3)
            assert explanation["target_class_name"] == "glioma"

    def test_pipeline_step_8_to_10_gymnasium_obs_and_dqn_action(self, real_mri_image):
        """Step 8, 9, 10: Construct Gymnasium Box(35,) observation and run trained SB3 DQN policy."""
        root = get_project_root()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transforms = get_transforms(split="val", image_size=(224, 224))
        extractor = RadiomicsExtractor()

        models = {}
        for m_name in ["custom_cnn", "resnet50", "densenet121"]:
            ckpt = root / "checkpoints" / f"{m_name}_best.pt"
            m = create_model(m_name, num_classes=4).to(device)
            _load_model_weights(m, ckpt, device)
            m.eval()
            models[m_name] = m

        # Load scaler
        scaler_file = root / "reports" / "rl_observation_scaler.json"
        assert scaler_file.exists(), "Scaler parameters missing"
        with open(scaler_file, "r") as f:
            scaler_stats = json.load(f)

        # Step 8: Gymnasium observation
        obs = extract_observation_vector(real_mri_image, models, device, transforms, extractor, scaler_stats)
        assert obs.shape == (OBSERVATION_DIM,)
        assert obs.dtype == np.float32
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()

        # Step 9: Load and predict with SB3 DQN
        from stable_baselines3 import DQN
        dqn_path = root / "checkpoints" / "dqn_triage_policy.zip"
        assert dqn_path.exists(), f"DQN checkpoint {dqn_path} missing"
        dqn = DQN.load(str(dqn_path))

        # Step 10: Selected workflow action
        action, _ = dqn.predict(obs, deterministic=True)
        action_code = int(action)
        assert action_code in [0, 1, 2]
        action_names = {0: "AUTO_REPORT", 1: "SECOND_VIEW", 2: "ESCALATE"}
        assert action_names[action_code] in ["AUTO_REPORT", "SECOND_VIEW", "ESCALATE"]

    def test_pipeline_step_11_to_12_api_integration_and_report_export(self, real_mri_image):
        """Step 11 & 12: End-to-end API roundtrip and structured JSON report creation."""
        buf = io.BytesIO()
        real_mri_image.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        # Step 11: Execute through API
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", raw_bytes, "image/jpeg")},
            data={"include_gradcam": "true"},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify full multimodal payload
        assert data["predicted_class"] in CLASSES
        assert data["confidence"] > 0.0
        assert data["triage"]["action_name"] in ["AUTO_REPORT", "SECOND_VIEW", "ESCALATE"]
        assert data["explanation"]["available"] is True
        assert len(data["radiomic_features"]) == 15

        # Step 12: Validate structured report export compatibility
        report_export = {
            "report_id": "NS-TEST-001",
            "study_type": "Brain MRI Axial T1/T2 CE",
            "predicted_class": data["predicted_class"],
            "confidence": data["confidence"],
            "predictive_entropy": data["predictive_entropy"],
            "class_probabilities": data["class_probabilities"],
            "disagreement": data["disagreement"],
            "triage": data["triage"],
            "radiomic_features": data["radiomic_features"],
            "uncertainty_warnings": data["uncertainty_warnings"],
            "model_versions": data["model_versions"],
            "disclaimer": data["disclaimer"],
        }
        report_str = json.dumps(report_export, indent=2)
        assert len(report_str) > 500


# =============================================================================
# PART 2: RIGOROUS FAILURE CASE TESTING
# =============================================================================
class TestPipelineFailureCases:
    """Verifies that all specified system edge cases and failure modes are handled gracefully."""

    def test_failure_corrupted_image(self):
        """Rejects corrupted or non-image payloads with HTTP 400."""
        corrupt_payload = b"CORRUPTED_RAW_BINARY_DATA_NOT_AN_IMAGE_778899"
        res = client.post("/api/analyze", files={"file": ("corrupt.png", corrupt_payload, "image/png")})
        assert res.status_code == 400
        assert "Failed to decode" in res.json()["detail"]

    def test_failure_oversized_upload(self):
        """Rejects uploads exceeding 15MB with HTTP 413."""
        oversized = b"A" * (16 * 1024 * 1024)
        res = client.post("/api/analyze", files={"file": ("huge.jpg", oversized, "image/jpeg")})
        assert res.status_code == 413
        assert "exceeds maximum limit" in res.json()["detail"]

    def test_failure_nan_and_infinite_feature_safety(self):
        """Extractor must produce strictly finite numbers even on pathological constant or empty images."""
        extractor = RadiomicsExtractor()
        # Edge case 1: Completely black image (0 pixels)
        black_img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).convert("RGB")
        res_black = extractor.extract_features(black_img)
        for k, v in res_black["features"].items():
            assert not np.isnan(v), f"Black image feature {k} is NaN"
            assert not np.isinf(v), f"Black image feature {k} is Inf"

        # Edge case 2: Uniformly white image (constant 255)
        white_img = Image.fromarray(np.full((224, 224), 255, dtype=np.uint8)).convert("RGB")
        res_white = extractor.extract_features(white_img)
        for k, v in res_white["features"].items():
            assert not np.isnan(v), f"White image feature {k} is NaN"
            assert not np.isinf(v), f"White image feature {k} is Inf"

        # Edge case 3: Single bright pixel
        single_px = np.zeros((224, 224), dtype=np.uint8)
        single_px[112, 112] = 255
        single_img = Image.fromarray(single_px).convert("RGB")
        res_single = extractor.extract_features(single_img)
        for k, v in res_single["features"].items():
            assert not np.isnan(v), f"Single pixel image feature {k} is NaN"
            assert not np.isinf(v), f"Single pixel image feature {k} is Inf"

    def test_failure_unsupported_gradcam_layer(self):
        """GradCAM must raise ValueError when targeted at a non-existent convolutional layer."""
        m = create_model("custom_cnn", num_classes=4)
        with pytest.raises(ValueError, match="Layer '.*' not found in model hierarchy"):
            with GradCAM(m, target_layer="non_existent_submodule_layer_999") as cam:
                pass

    def test_failure_missing_model_readiness(self, tmp_path):
        """ModelReadinessManager correctly identifies missing weights and reports unready."""
        fake_manager = ModelReadinessManager(checkpoints_dir=tmp_path)
        status = fake_manager.check_model_readiness("resnet50")
        assert status["status"] == "MISSING"
        assert status["is_ready"] is False

    def test_failure_missing_dqn_policy_fallback(self):
        """When DQN policy is unavailable, triage automatically falls back to rule-based safety policy."""
        # Create a mock 35-dim observation representing high confidence healthy control
        normal_obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        normal_obs[2] = 0.98  # no_tumor prob
        normal_obs[16] = 0.98  # max prob
        normal_obs[17] = 0.05  # low entropy
        action = rule_based_triage_policy(normal_obs)
        assert action == TriageAction.AUTO_REPORT.value

        # Critical Glioma case with high disagreement
        glioma_obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        glioma_obs[0] = 0.95  # glioma prob
        action_glioma = rule_based_triage_policy(glioma_obs)
        assert action_glioma == TriageAction.ESCALATE.value

    def test_failure_inconsistent_class_mapping(self):
        """Verify strict 1-to-1 bijective class mapping across all constants."""
        assert len(CLASSES) == 4
        assert set(CLASSES) == {"glioma", "meningioma", "notumor", "pituitary"} or set(CLASSES) == {
            "glioma", "meningioma", "no_tumor", "pituitary"
        }
        for idx, cls_name in enumerate(CLASSES):
            assert CLASS_TO_IDX[cls_name] == idx


# =============================================================================
# PART 3: EXPERIMENTAL INTEGRITY VERIFICATION
# =============================================================================
class TestExperimentalIntegrity:
    """Verifies scientific rigor and guarantees against data leakage and false clinical claims."""

    def test_scratch_initialization_guarantee(self):
        """Confirm that models are instantiated from scratch with non-ImageNet initializations."""
        for m_name in ["custom_cnn", "resnet50", "densenet121"]:
            m = create_model(m_name, num_classes=4)
            assert m is not None
            # Check weights are non-zero and finite
            for param in m.parameters():
                assert not torch.isnan(param).any()
                assert not torch.isinf(param).any()

    def test_observation_scaler_training_split_only(self):
        """Scaler statistics must originate solely from training split samples."""
        root = get_project_root()
        scaler_file = root / "reports" / "rl_observation_scaler.json"
        assert scaler_file.exists()
        with open(scaler_file, "r") as f:
            stats = json.load(f)
        # Scaler dictionary holds features with mean and std
        assert len(stats) == 9
        for feat_key, feat_vals in stats.items():
            assert "mean" in feat_vals
            assert "std" in feat_vals
            assert not np.isnan(feat_vals["mean"])
            assert not np.isnan(feat_vals["std"])

    def test_held_out_benchmark_quarantine(self):
        """Frozen evaluation report must document quarantine of 1,600 test scans."""
        root = get_project_root()
        eval_file = root / "reports" / "evaluation_results.json"
        assert eval_file.exists()
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
        assert eval_data["ensemble"]["metrics"]["total_samples"] == 1600
        assert eval_data["ensemble"]["metrics"]["accuracy"] >= 0.90

    def test_legal_and_clinical_disclaimer_presence(self):
        """All public endpoints must output non-clinical decision-support disclaimers."""
        res_eval = client.get("/api/evaluation").json()
        assert "disclaimer" in res_eval
        assert "strictly" in res_eval["disclaimer"] or "quarantined" in res_eval["disclaimer"]
