"""Unit tests for FastAPI endpoints: /api/health, /api/models, /api/analyze, /api/explain, /api/features, /api/evaluation."""

import io
import pytest
import numpy as np
from PIL import Image
import cv2
from fastapi.testclient import TestClient

from src.api.app import app
from src.data.constants import CLASSES

client = TestClient(app)


def create_test_mri_bytes(size: int = 224, color: int = 150) -> bytes:
    """Generate in-memory JPEG bytes of a synthetic brain MRI slice."""
    img_arr = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img_arr, (size // 2, size // 2), size // 3, color, -1)
    pil_img = Image.fromarray(img_arr).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


class TestHealthAndModelsEndpoints:
    """Test health diagnostics and model readiness tracking."""

    def test_get_health(self):
        """GET /api/health returns 200 OK with hardware diagnostics and readiness flags."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert data["service"] == "neuroscan-ai"
        assert "hardware" in data
        assert "ready_models" in data
        assert isinstance(data["ready_models"], list)
        assert "dqn_policy_ready" in data

    def test_get_models(self):
        """GET /api/models reports checkpoint readiness without fabrication."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "models" in data
        assert "custom_cnn" in data["models"]
        assert "resnet50" in data["models"]
        assert "densenet121" in data["models"]
        assert "ensemble_ready" in data
        assert "dqn_triage_ready" in data

        # Check model metadata
        for m_name in ["custom_cnn", "resnet50", "densenet121"]:
            m_info = data["models"][m_name]
            assert "is_ready" in m_info
            assert "checkpoint_exists" in m_info
            assert "pretraining" in m_info
            assert "scratch" in m_info["pretraining"]


class TestAnalyzeEndpoint:
    """Test POST /api/analyze full multimodal pipeline."""

    def test_analyze_valid_image(self):
        """POST /api/analyze executes end-to-end pipeline and returns complete schema."""
        img_bytes = create_test_mri_bytes()
        files = {"file": ("test_scan.jpg", img_bytes, "image/jpeg")}
        data = {"include_gradcam": "true"}

        response = client.post("/api/analyze", files=files, data=data)
        assert response.status_code == 200
        payload = response.json()

        assert payload["status"] == "success"
        assert payload["predicted_class"] in CLASSES
        assert 0.0 <= payload["confidence"] <= 1.0
        assert len(payload["class_probabilities"]) == len(CLASSES)
        assert payload["predictive_entropy"] >= 0.0
        assert "disagreement" in payload
        assert "member_outputs" in payload
        assert "radiomic_features" in payload
        assert len(payload["radiomic_features"]) == 15

        # Triage decision validation
        triage = payload["triage"]
        assert triage["action_code"] in [0, 1, 2]
        assert triage["action_name"] in ["AUTO_REPORT", "SECOND_VIEW", "ESCALATE"]
        assert triage["policy_type"] in ["DQN_POLICY", "RULE_BASED_FALLBACK"]
        assert len(triage["reason"]) > 10
        assert "workflow_impact" in triage

        # Grad-CAM explanation validation
        assert payload["explanation"] is not None
        assert payload["explanation"]["available"] is True
        assert payload["explanation"]["overlay_base64"].startswith("data:image/png;base64,")
        assert payload["explanation"]["heatmap_base64"].startswith("data:image/png;base64,")

        # Disclaimers & warnings
        assert "disclaimer" in payload
        assert isinstance(payload["uncertainty_warnings"], list)

    def test_analyze_corrupt_file_rejected(self):
        """POST /api/analyze rejects corrupted image bytes with HTTP 400."""
        files = {"file": ("corrupt.jpg", b"NOT_A_VALID_IMAGE_BYTES", "image/jpeg")}
        response = client.post("/api/analyze", files=files)
        assert response.status_code == 400
        assert "Failed to decode" in response.json()["detail"]

    def test_analyze_oversized_file_rejected(self):
        """POST /api/analyze rejects payloads exceeding 15MB with HTTP 413."""
        # Create 16 MB dummy payload
        huge_bytes = b"0" * (16 * 1024 * 1024)
        files = {"file": ("huge.jpg", huge_bytes, "image/jpeg")}
        response = client.post("/api/analyze", files=files)
        assert response.status_code == 413
        assert "exceeds maximum limit" in response.json()["detail"]


class TestExplainEndpoint:
    """Test POST /api/explain on-demand Grad-CAM endpoint."""

    def test_explain_valid_request(self):
        """POST /api/explain returns base64 overlay and heatmap for selected model."""
        img_bytes = create_test_mri_bytes()
        files = {"file": ("scan.jpg", img_bytes, "image/jpeg")}
        data = {
            "model_name": "densenet121",
            "target_class": "glioma",
            "alpha": "0.5",
        }

        response = client.post("/api/explain", files=files, data=data)
        assert response.status_code == 200
        payload = response.json()

        assert payload["status"] == "success"
        assert payload["model_name"] == "densenet121"
        assert payload["target_class"] == "glioma"
        assert payload["overlay_base64"].startswith("data:image/png;base64,")
        assert payload["heatmap_base64"].startswith("data:image/png;base64,")
        assert "disclaimer" in payload

    def test_explain_unready_model_returns_503(self):
        """POST /api/explain returns 503 if requested model is not available."""
        img_bytes = create_test_mri_bytes()
        files = {"file": ("scan.jpg", img_bytes, "image/jpeg")}
        data = {"model_name": "non_existent_model"}

        response = client.post("/api/explain", files=files, data=data)
        assert response.status_code == 503
        assert "not ready" in response.json()["detail"]


class TestFeaturesEndpoint:
    """Test POST /api/features radiomics extraction endpoint."""

    def test_features_valid_request(self):
        """POST /api/features extracts all 15 radiomics descriptors with definitions."""
        img_bytes = create_test_mri_bytes()
        files = {"file": ("scan.jpg", img_bytes, "image/jpeg")}

        response = client.post("/api/features", files=files)
        assert response.status_code == 200
        payload = response.json()

        assert payload["status"] == "success"
        assert len(payload["features"]) == 15
        assert "definitions" in payload
        assert "intensity_mean" in payload["features"]
        assert "glcm_contrast" in payload["features"]
        assert "circularity" in payload["features"]
        assert "box_counting_fractal_dimension" in payload["features"]
        assert "disclaimer" in payload


class TestEvaluationEndpoint:
    """Test GET /api/evaluation frozen benchmark retrieval."""

    def test_get_evaluation_metrics(self):
        """GET /api/evaluation returns CNN test metrics and RL triage comparison."""
        response = client.get("/api/evaluation")
        assert response.status_code == 200
        payload = response.json()

        assert payload["status"] == "success"
        assert "cnn_benchmark" in payload
        assert "rl_triage_benchmark" in payload
        assert len(payload["cnn_benchmark"]) > 0
        assert len(payload["rl_triage_benchmark"]) > 0
        assert "best_single_model" in payload
        assert "ensemble_summary" in payload
        assert "disclaimer" in payload
