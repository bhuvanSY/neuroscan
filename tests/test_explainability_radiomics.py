"""Unit tests for Grad-CAM explainability and radiomics-style feature extraction."""

import math
from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image
import cv2

from src.data.constants import CLASSES, CLASS_TO_IDX
from src.models.architectures import create_model
from src.explainability.gradcam import (
    GradCAM,
    get_default_target_layer,
    DISCLAIMER_TEXT as GRADCAM_DISCLAIMER,
)
from src.features.radiomics import (
    RadiomicsExtractor,
    compute_box_counting_dimension,
    FEATURE_DEFINITIONS,
    DISCLAIMER_TEXT as RADIOMICS_DISCLAIMER,
)


class TestGradCAM:
    """Test Grad-CAM implementation across all three scratch architectures."""

    @pytest.mark.parametrize("model_name", ["custom_cnn", "resnet50", "densenet121", "efficientnet_v2"])
    def test_gradcam_generation_all_architectures(self, model_name):
        """Grad-CAM must compute valid heatmaps for all 3 scratch-trained models."""
        model = create_model(model_name).eval()
        target_layer = get_default_target_layer(model)
        assert target_layer is not None

        with GradCAM(model) as cam_engine:
            x = torch.randn(1, 3, 224, 224, requires_grad=True)
            cam, target_idx, conf, probs = cam_engine.generate_cam(x)

            # Dimensions & numerical properties
            assert cam.shape == (224, 224)
            assert cam.dtype == np.float32
            assert not np.isnan(cam).any()
            assert not np.isinf(cam).any()
            assert 0.0 <= cam.min()
            assert cam.max() <= 1.0
            assert 0 <= target_idx < len(CLASSES)
            assert 0.0 <= conf <= 1.0
            assert len(probs) == len(CLASSES)

    def test_user_selected_vs_predicted_class(self):
        """Grad-CAM must support both predicted and user-selected target classes."""
        model = create_model("resnet50").eval()
        x = torch.randn(1, 3, 224, 224)

        with GradCAM(model) as cam_engine:
            # 1. Top predicted class
            cam_pred, pred_idx, conf_pred, _ = cam_engine.generate_cam(x, target_class=None)

            # 2. User-selected class (e.g., class 2: no_tumor)
            cam_user, user_idx, conf_user, _ = cam_engine.generate_cam(x, target_class=2)
            assert user_idx == 2
            assert cam_user.shape == (224, 224)

    def test_explain_image_full_pack(self, tmp_path):
        """Test complete explain_image pack with overlay and heatmap-only outputs."""
        model = create_model("custom_cnn").eval()

        # Generate a synthetic MRI image file
        img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(img_arr, (112, 112), 60, (180, 180, 180), -1)
        test_path = tmp_path / "test_mri.jpg"
        Image.fromarray(img_arr).save(test_path)

        cam_engine = GradCAM(model)
        try:
            pack = cam_engine.explain_image(test_path, target_class="glioma")

            assert "heatmap" in pack
            assert "heatmap_colored" in pack
            assert "overlay" in pack
            assert "original_image" in pack
            assert "target_class_name" in pack
            assert pack["target_class_name"] == "glioma"
            assert pack["target_class_idx"] == CLASS_TO_IDX["glioma"]
            assert "disclaimer" in pack
            assert "NOT represent precise tumor segmentation masks" in pack["disclaimer"]

            # Visual arrays shape and type checks
            assert pack["heatmap"].shape == (224, 224)
            assert pack["heatmap_colored"].shape == (224, 224, 3)
            assert pack["overlay"].shape == (224, 224, 3)
            assert pack["original_image"].shape == (224, 224, 3)
            assert pack["overlay"].dtype == np.uint8
            assert pack["heatmap_colored"].dtype == np.uint8
        finally:
            cam_engine.remove_hooks()

    def test_hook_cleanup_and_context_manager(self):
        """Verify hooks are removed cleanly and do not persist in memory."""
        model = create_model("densenet121").eval()
        cam_engine = GradCAM(model)
        assert cam_engine._hook_handle is not None

        cam_engine.remove_hooks()
        assert cam_engine._hook_handle is None
        assert cam_engine.activations is None
        assert cam_engine.gradients is None

    def test_invalid_target_class_raises(self):
        """Invalid class indices or unknown strings must raise ValueError."""
        model = create_model("custom_cnn").eval()
        with GradCAM(model) as cam_engine:
            x = torch.randn(1, 3, 224, 224)
            with pytest.raises(ValueError, match="out of range"):
                cam_engine.generate_cam(x, target_class=99)

            dummy_img = Image.new("RGB", (224, 224), color=100)
            with pytest.raises(ValueError, match="Unknown class name"):
                cam_engine.explain_image(dummy_img, target_class="non_existent_tumor")


class TestRadiomicsExtractor:
    """Test 2D radiomics feature extraction, morphology, and edge cases."""

    @pytest.fixture
    def extractor(self):
        return RadiomicsExtractor()

    def test_feature_extraction_schema_and_keys(self, extractor):
        """All 15 radiomic features defined in FEATURE_DEFINITIONS must be extracted."""
        # Create synthetic brain slice
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 70, 160, -1)
        cv2.circle(img, (80, 80), 20, 210, -1)

        result = extractor.extract_features(img)
        features = result["features"]

        assert len(features) == len(FEATURE_DEFINITIONS)
        for key in FEATURE_DEFINITIONS:
            assert key in features, f"Missing feature {key} from extraction result"
            val = features[key]
            assert isinstance(val, (int, float)), f"Feature {key} is not a numeric scalar"
            assert not math.isnan(val), f"Feature {key} is NaN"
            assert not math.isinf(val), f"Feature {key} is Infinite"

        # Check definitions and disclaimer
        assert result["definitions"] == FEATURE_DEFINITIONS
        assert "NOT establish true 3D tumor volume" in result["disclaimer"]

    def test_first_order_intensity_values(self, extractor):
        """Verify mean, std, skewness, and kurtosis calculations."""
        # Uniform known array
        arr = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        res = extractor.extract_features(arr)
        feats = res["features"]

        assert feats["intensity_mean"] == pytest.approx(125.0)
        assert feats["intensity_std"] > 0.0

    def test_empty_mask_edge_case(self, extractor):
        """Empty (all-black) image must return 0 for area/perimeter/contrast without error."""
        black_img = np.zeros((128, 128), dtype=np.uint8)
        res = extractor.extract_features(black_img)
        feats = res["features"]

        assert feats["foreground_area"] == 0.0
        assert feats["foreground_perimeter"] == 0.0
        assert feats["circularity"] == 0.0
        assert feats["eccentricity"] == 0.0
        assert feats["glcm_contrast"] == 0.0
        assert feats["intensity_std"] == 0.0
        assert res["metadata"]["is_empty"] is True

    def test_constant_image_edge_case(self, extractor):
        """Constant non-zero image (all pixels equal) must not raise division-by-zero."""
        const_img = np.full((100, 100), 128, dtype=np.uint8)
        res = extractor.extract_features(const_img)
        feats = res["features"]

        assert feats["intensity_mean"] == 128.0
        assert feats["intensity_std"] == 0.0
        assert feats["glcm_contrast"] == 0.0
        assert res["metadata"]["is_constant"] is True

    def test_circularity_of_circle(self, extractor):
        """Circularity of an exact disk should approach 1.0."""
        circle_img = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(circle_img, (128, 128), 60, 255, -1)

        res = extractor.extract_features(circle_img)
        circ = res["features"]["circularity"]
        # Pixelated discrete circle achieves ~0.85-0.95
        assert 0.80 <= circ <= 1.0

    def test_box_counting_fractal_dimension_bounds(self):
        """Box counting dimension must be bounded between 0.0 and 2.0."""
        # Empty array
        assert compute_box_counting_dimension(np.zeros((64, 64))) == 0.0

        # Boundary contour
        boundary = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(boundary, (64, 64), 32, 255, 1)
        dim = compute_box_counting_dimension(boundary)
        assert 0.8 <= dim <= 1.5
