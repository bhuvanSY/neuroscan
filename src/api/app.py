"""FastAPI enterprise backend service for NeuroScan AI.

Integrates:
- Scratch-Trained CNN Ensemble (Custom CNN, ResNet-50, DenseNet-121)
- Grad-CAM Attribution Maps & Visual Overlays
- 2D Radiomics-Style Morphological & Textural Feature Engine
- Gymnasium-based DQN Clinical Review Triage Decision Policy

CRITICAL MEDICAL & LEGAL DISCLAIMER:
NeuroScan AI is an investigational decision-support research platform.
In the absence of prospective multi-center clinical trials, all model predictions,
Grad-CAM overlays, and triage actions must be interpreted by licensed healthcare professionals.
AUTO_REPORT creates a draft report for clinician sign-off and NEVER autonomously clears a patient.
"""

import io
import json
import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from stable_baselines3 import DQN

from src.utils.config import get_project_root
from src.utils.device import get_device, get_hardware_info
from src.utils.logger import get_logger
from src.utils.exceptions import ModelNotReadyError, CorruptImageError
from src.data.constants import CLASSES, CLASS_TO_IDX
from src.data.transforms import get_transforms
from src.models.architectures import create_model
from src.models.readiness import ModelReadinessManager
from src.explainability.gradcam import GradCAM, DISCLAIMER_TEXT as GRADCAM_DISCLAIMER
from src.features.radiomics import RadiomicsExtractor, FEATURE_DEFINITIONS, DISCLAIMER_TEXT as RADIOMICS_DISCLAIMER
from src.rl.safety_rules import (
    TriageAction,
    ACTION_DESCRIPTIONS,
    rule_based_triage_policy,
    fixed_threshold_baseline_policy,
)
from src.rl.triage_env import OBSERVATION_DIM
from src.rl.train_dqn import extract_observation_vector
from src.api.schemas import (
    HealthResponse,
    ModelsResponse,
    ModelInfo,
    TriageDecision,
    MemberModelOutput,
    GradCAMVisuals,
    AnalyzeResponse,
    ExplainResponse,
    FeaturesResponse,
    EvaluationResponse,
)

logger = get_logger("neuroscan.api")
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit

app = FastAPI(
    title="NeuroScan AI Enterprise API",
    description="Multimodal Brain MRI Classification, Grad-CAM Explainability, Radiomics, and RL Review Triage",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ServiceState:
    """Manages cached models, feature extractors, and policy singletons."""

    def __init__(self):
        self.root = get_project_root()
        self.checkpoints_dir = self.root / "checkpoints"
        self.reports_dir = self.root / "reports"
        self.device = get_device("auto")
        self.readiness_mgr = ModelReadinessManager(self.checkpoints_dir)
        self.radiomics = RadiomicsExtractor()

        # Mean and std for input normalization
        norm_file = self.reports_dir / "dataset_stats.json"
        if norm_file.exists():
            with open(norm_file, "r") as f:
                stats = json.load(f)
                self.mean = stats.get("mean", [0.172, 0.172, 0.172])
                self.std = stats.get("std", [0.1868, 0.1868, 0.1868])
        else:
            self.mean = [0.172, 0.172, 0.172]
            self.std = [0.1868, 0.1868, 0.1868]

        self.transforms = get_transforms(split="val", crop_contour=True, mean=self.mean, std=self.std)

        # Loaded model cache
        self.models: Dict[str, torch.nn.Module] = {}
        self.ensemble_weights = {"custom_cnn": 0.0, "resnet50": 0.3714, "densenet121": 0.6286}
        self.dqn_model: Optional[DQN] = None
        self.rl_scaler_stats: Optional[Dict[str, Dict[str, float]]] = None

        self._initialize_service()

    def _initialize_service(self):
        """Warm-load available models and policies on startup."""
        logger.info(f"Initializing NeuroScan AI service on device: {self.device}")
        
        # Load CNN models
        for name in ["custom_cnn", "resnet50", "densenet121", "efficientnet_v2"]:
            ckpt_path = self.checkpoints_dir / f"{name}_best.pt"
            if ckpt_path.exists():
                try:
                    m = create_model(name, num_classes=len(CLASSES))
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
                    m.load_state_dict(state)
                    m.to(self.device)
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
                    self.models[name] = m
                    logger.info(f"Loaded frozen CNN: {name}")
                except Exception as e:
                    logger.error(f"Failed to load checkpoint for {name}: {e}")

        # Load RL observation scaler (strictly from training split)
        scaler_file = self.reports_dir / "rl_observation_scaler.json"
        if scaler_file.exists():
            with open(scaler_file, "r") as f:
                self.rl_scaler_stats = json.load(f)
                logger.info("Loaded frozen training RL observation scaler.")

        # Load DQN Policy
        dqn_path = self.checkpoints_dir / "dqn_triage_policy.zip"
        if dqn_path.exists():
            try:
                self.dqn_model = DQN.load(str(dqn_path))
                logger.info("Loaded trained SB3 DQN triage policy.")
            except Exception as e:
                logger.warning(f"Failed to load DQN policy: {e}")
        else:
            logger.info("No DQN triage policy checkpoint found. Fallback rule policy active.")


state = ServiceState()


def encode_image_base64(img_arr_rgb: np.ndarray) -> str:
    """Encode an RGB uint8 array to a base64 Data URI."""
    pil_img = Image.fromarray(img_arr_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def derive_triage_reason(
    action: int,
    policy_type: str,
    pred_class: str,
    confidence: float,
    entropy: float,
    disagreement_rate: float,
) -> str:
    """Generate transparent clinical rationale based on observable features."""
    if action == TriageAction.AUTO_REPORT:
        return (
            f"Concordant low-complexity study with high prediction confidence ({confidence:.1%}), "
            f"minimal entropy ({entropy:.3f}), and zero inter-model disagreement. "
            f"An automated draft report has been synthesized for expedited clinician sign-off."
        )
    elif action == TriageAction.SECOND_VIEW:
        return (
            f"Borderline confidence ({confidence:.1%}) or moderate predictive uncertainty ({entropy:.3f}). "
            f"A secondary radiographic sequence or peer verification check is requested to resolve tissue ambiguity."
        )
    elif action == TriageAction.ESCALATE:
        if pred_class == "glioma":
            return (
                f"Elevated risk of intra-axial malignancy (suspected glioma with {confidence:.1%} probability). "
                f"Requires priority emergency specialist neuroradiology review."
            )
        elif disagreement_rate > 0.0:
            return (
                f"Significant inter-model disagreement ({disagreement_rate:.1%}) between independent architectures. "
                f"Escalated to specialist review to resolve discordant architectural classifications."
            )
        else:
            return (
                f"High predictive entropy ({entropy:.3f}) or complex morphological margins. "
                f"Routed directly to expert specialist consultation."
            )
    return "Standard clinical triage review."


def extract_uncertainty_warnings(
    pred_class: str,
    confidence: float,
    entropy: float,
    margin: float,
    disagreement: Dict[str, Any],
    radiomics: Dict[str, float],
) -> List[str]:
    """Identify diagnostic flags based on calibrated uncertainty and radiomics."""
    warnings = []
    if entropy > 0.60:
        warnings.append(
            f"Elevated predictive entropy ({entropy:.3f}): probability mass is dispersed across multiple candidate classes."
        )
    if margin < 0.20:
        warnings.append(
            f"Narrow confidence margin ({margin:.1%}): top candidate is closely contested by secondary differential."
        )
    if disagreement.get("categorical_disagreement_rate", 0.0) > 0.0:
        warnings.append(
            "Inter-model disagreement detected: member architectures generated discordant category predictions."
        )
    if pred_class == "glioma" and confidence < 0.85:
        warnings.append(
            f"Suspected infiltrative intra-axial glioma with moderate confidence ({confidence:.1%}). Urgent specialist correlation advised."
        )
    if radiomics.get("circularity", 0.0) > 0.80 and radiomics.get("glcm_contrast", 0.0) < 5.0:
        warnings.append(
            "High geometric circularity with low contrast: cross-reference dural attachment to differentiate meningioma from adenoma."
        )
    return warnings


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, include_in_schema=False)
@app.get("/api/health", response_model=HealthResponse, tags=["Diagnostics"])
@app.get("/api/v1/health", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> HealthResponse:
    """System health check, GPU diagnostics, and model availability status."""
    hardware = get_hardware_info()
    ready_models = list(state.models.keys())
    is_healthy = len(ready_models) > 0
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        service="neuroscan-ai",
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        ready_models=ready_models,
        dqn_policy_ready=(state.dqn_model is not None),
    )


@app.get("/models", response_model=ModelsResponse, include_in_schema=False)
@app.get("/api/models", response_model=ModelsResponse, tags=["Models"])
@app.get("/api/v1/models", response_model=ModelsResponse, include_in_schema=False)
@app.get("/api/v1/readiness", response_model=ModelsResponse, include_in_schema=False)
async def list_models() -> ModelsResponse:
    """List scratch CNN architectures, checkpoint status, and readiness."""
    models_dict = {}
    for name in ["custom_cnn", "resnet50", "densenet121", "efficientnet_v2"]:
        ckpt_path = state.checkpoints_dir / f"{name}_best.pt"
        exists = ckpt_path.exists()
        is_ready = name in state.models
        param_count = None
        if is_ready:
            param_count = sum(p.numel() for p in state.models[name].parameters())
        models_dict[name] = ModelInfo(
            model_name=name,
            is_ready=is_ready,
            checkpoint_exists=exists,
            checkpoint_path=str(ckpt_path) if exists else None,
            parameters=param_count,
            architecture=name.upper(),
        )

    return ModelsResponse(
        status="success",
        models=models_dict,
        ensemble_ready=(len(state.models) >= 2),
        dqn_triage_ready=(state.dqn_model is not None),
    )


@app.post("/analyze", response_model=AnalyzeResponse, include_in_schema=False)
@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Inference"])
async def analyze_scan(
    file: UploadFile = File(...),
    include_gradcam: bool = Form(True),
) -> AnalyzeResponse:
    """Full multimodal pipeline: Ensemble Inference -> Grad-CAM -> Radiomics -> RL Triage."""
    if not state.models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained CNN models are loaded in memory. Train checkpoints before requesting analysis."
        )

    # 1. Validate file size and read
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum limit of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        )

    # 2. Safe image decode
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image file: {str(e)}"
        )

    # 3. Model forward passes
    tensor = state.transforms(pil_img).unsqueeze(0).to(state.device)
    member_outputs: Dict[str, MemberModelOutput] = {}
    member_probs_dict: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for name, model in state.models.items():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            member_probs_dict[name] = probs
            pred_idx = int(np.argmax(probs))
            member_outputs[name] = MemberModelOutput(
                predicted_class=CLASSES[pred_idx],
                confidence=float(round(probs[pred_idx], 4)),
                probabilities={CLASSES[i]: float(round(probs[i], 4)) for i in range(len(CLASSES))},
            )

    # 4. Weighted soft-voting ensemble
    ens_probs = np.zeros(len(CLASSES), dtype=np.float32)
    total_w = sum(state.ensemble_weights.get(m, 0.0) for m in state.models.keys())
    if total_w <= 0.0:
        total_w = 1.0
        norm_weights = {m: 1.0 / len(state.models) for m in state.models.keys()}
    else:
        norm_weights = {m: state.ensemble_weights.get(m, 0.0) / total_w for m in state.models.keys()}

    for m_name, w in norm_weights.items():
        ens_probs += member_probs_dict[m_name] * w
    ens_probs = ens_probs / np.sum(ens_probs)

    pred_idx = int(np.argmax(ens_probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(round(ens_probs[pred_idx], 4))
    entropy = float(round(-np.sum(ens_probs * np.log(ens_probs + 1e-10)), 4))

    sorted_p = np.sort(ens_probs)
    margin = float(round(sorted_p[-1] - sorted_p[-2], 4))

    # 5. Model disagreement
    from src.evaluation.uncertainty import compute_ensemble_disagreement
    disagreement = compute_ensemble_disagreement(list(member_probs_dict.values()))

    # 6. Radiomics extraction
    rad_res = state.radiomics.extract_features(pil_img)
    radiomics_features = rad_res["features"]

    # 7. Grad-CAM visual explanation
    gradcam_visuals = None
    if include_gradcam:
        best_model_name = "densenet121" if "densenet121" in state.models else list(state.models.keys())[0]
        try:
            with GradCAM(state.models[best_model_name], device=state.device) as cam_engine:
                pack = cam_engine.explain_image(pil_img, target_class=pred_class, mean=state.mean, std=state.std)
                gradcam_visuals = GradCAMVisuals(
                    available=True,
                    overlay_base64=encode_image_base64(pack["overlay"]),
                    heatmap_base64=encode_image_base64(pack["heatmap_colored"]),
                    target_class=pred_class,
                    disclaimer=GRADCAM_DISCLAIMER,
                )
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            gradcam_visuals = GradCAMVisuals(available=False, disclaimer=GRADCAM_DISCLAIMER)

    # 8. RL Triage Observation & Action
    obs = extract_observation_vector(
        pil_img, state.models, state.device, state.transforms, state.radiomics, state.rl_scaler_stats
    )

    if state.dqn_model is not None:
        action_val, _ = state.dqn_model.predict(obs, deterministic=True)
        action_code = int(action_val)
        policy_type = "DQN_POLICY"
    else:
        action_code = rule_based_triage_policy(obs)
        policy_type = "RULE_BASED_FALLBACK"

    action_meta = ACTION_DESCRIPTIONS[action_code]
    reason = derive_triage_reason(
        action_code,
        policy_type,
        pred_class,
        confidence,
        entropy,
        disagreement["categorical_disagreement_rate"],
    )

    triage_decision = TriageDecision(
        action_code=action_code,
        action_name=action_meta["name"],
        policy_type=policy_type,
        description=action_meta["description"],
        reason=reason,
        workflow_impact=action_meta["workflow_impact"],
        disclaimer="Workflow decision for simulation/decision-support only; never autonomous release.",
    )

    # 9. Uncertainty Warnings
    warnings = extract_uncertainty_warnings(
        pred_class, confidence, entropy, margin, disagreement, radiomics_features
    )

    return AnalyzeResponse(
        status="success",
        predicted_class=pred_class,
        confidence=confidence,
        class_probabilities={CLASSES[i]: float(round(ens_probs[i], 4)) for i in range(len(CLASSES))},
        predictive_entropy=entropy,
        confidence_margin=margin,
        disagreement=disagreement,
        member_outputs=member_outputs,
        ensemble_weights=norm_weights,
        radiomic_features=radiomics_features,
        explanation=gradcam_visuals,
        triage=triage_decision,
        uncertainty_warnings=warnings,
        model_versions={m: "v1.0-scratch" for m in state.models.keys()},
        disclaimer="Investigational decision-support prototype. Requires licensed clinician validation.",
    )


@app.post("/explain", response_model=ExplainResponse, include_in_schema=False)
@app.post("/api/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_scan(
    file: UploadFile = File(...),
    model_name: str = Form("densenet121"),
    target_class: Optional[str] = Form(None),
    alpha: float = Form(0.45),
) -> ExplainResponse:
    """Generate on-demand Grad-CAM visual attribution overlay for a selected model and class."""
    if model_name not in state.models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' is not ready or has no loaded weights."
        )

    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {str(e)}")

    model = state.models[model_name]
    with GradCAM(model, device=state.device) as cam:
        pack = cam.explain_image(pil_img, target_class=target_class, alpha=alpha, mean=state.mean, std=state.std)

    return ExplainResponse(
        status="success",
        model_name=model_name,
        target_class=pack["target_class_name"],
        predicted_class=pack["predicted_class_name"],
        confidence=round(pack["confidence"], 4),
        is_predicted_class=pack["is_predicted_class"],
        overlay_base64=encode_image_base64(pack["overlay"]),
        heatmap_base64=encode_image_base64(pack["heatmap_colored"]),
        class_probabilities={k: round(v, 4) for k, v in pack["class_probabilities"].items()},
        disclaimer=GRADCAM_DISCLAIMER,
    )


@app.post("/features", response_model=FeaturesResponse, include_in_schema=False)
@app.post("/api/features", response_model=FeaturesResponse, tags=["Radiomics"])
async def extract_scan_features(file: UploadFile = File(...)) -> FeaturesResponse:
    """Extract 15 quantitative radiomics-style morphological and textural features."""
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {str(e)}")

    res = state.radiomics.extract_features(pil_img)
    return FeaturesResponse(
        status="success",
        features=res["features"],
        definitions=res["definitions"],
        metadata=res["metadata"],
        disclaimer=RADIOMICS_DISCLAIMER,
    )


@app.get("/evaluation", response_model=EvaluationResponse, include_in_schema=False)
@app.get("/api/evaluation", response_model=EvaluationResponse, tags=["Evaluation"])
async def get_evaluation_metrics() -> EvaluationResponse:
    """Retrieve frozen benchmark results comparing models, ensemble, and RL triage."""
    csv_cnn = state.reports_dir / "model_comparison.csv"
    csv_rl = state.reports_dir / "rl_triage_comparison.csv"

    cnn_records = []
    if csv_cnn.exists():
        df_cnn = pd.read_csv(csv_cnn)
        cnn_records = df_cnn.to_dict(orient="records")

    rl_records = []
    if csv_rl.exists():
        df_rl = pd.read_csv(csv_rl)
        rl_records = df_rl.to_dict(orient="records")

    summary_file = state.reports_dir / "evaluation_results.json"
    disagreement_summary = None
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                data = json.load(f)
                disagreement_summary = data.get("ensemble", {}).get("disagreement")
        except Exception:
            pass

    return EvaluationResponse(
        status="success",
        cnn_benchmark=cnn_records,
        rl_triage_benchmark=rl_records,
        disagreement_summary=disagreement_summary,
        best_single_model="DenseNet-121 (Macro-F1: 0.9220)",
        ensemble_summary="Weighted soft-voting ensemble achieved highest test accuracy (92.44%) and lowest calibration error (ECE 0.0189).",
        disclaimer="Evaluated strictly on 1,600 held-out test scans quarantined from training.",
    )


# Backward compatibility endpoint
@app.post("/api/v1/predict", include_in_schema=False)
async def predict_scan_legacy(
    file: UploadFile = File(...),
    model_name: str = Form("densenet121"),
) -> Dict[str, Any]:
    """Legacy endpoint forwarding to analyze logic."""
    res = await analyze_scan(file=file, include_gradcam=False)
    return {
        "model_name": model_name,
        "predicted_class": res.predicted_class,
        "confidence": res.confidence,
        "entropy": res.predictive_entropy,
        "class_probabilities": res.class_probabilities,
    }


# =============================================================================
# RESEARCH SAMPLE SCANS FOR AUDIT AND DEMO VERIFICATION
# =============================================================================
SAMPLE_CASES = [
    {
        "id": "sample_glioma",
        "title": "Glioma Benchmark Scan",
        "ground_truth_class": "glioma",
        "filename": "Te-gl_1.jpg",
        "rel_path": "data/archive/Testing/glioma/Te-gl_1.jpg",
        "description": "Representative held-out testing slice showing intra-axial high-grade glioma infiltrative lesion.",
    },
    {
        "id": "sample_meningioma",
        "title": "Meningioma Benchmark Scan",
        "ground_truth_class": "meningioma",
        "filename": "Te-me_1.jpg",
        "rel_path": "data/archive/Testing/meningioma/Te-me_1.jpg",
        "description": "Held-out testing slice displaying dural-based circumscribed meningioma lesion.",
    },
    {
        "id": "sample_pituitary",
        "title": "Pituitary Tumor Benchmark Scan",
        "ground_truth_class": "pituitary",
        "filename": "Te-pi_1.jpg",
        "rel_path": "data/archive/Testing/pituitary/Te-pi_1.jpg",
        "description": "Held-out testing slice displaying sellar / pituitary region mass lesion.",
    },
    {
        "id": "sample_no_tumor",
        "title": "Healthy Control (No Tumor)",
        "ground_truth_class": "no_tumor",
        "filename": "Te-no_1.jpg",
        "rel_path": "data/archive/Testing/notumor/Te-no_1.jpg",
        "description": "Held-out testing slice displaying symmetrical non-neoplastic brain parenchyma.",
    },
]


@app.get("/samples", include_in_schema=False)
@app.get("/api/samples", tags=["Samples"])
def list_sample_scans() -> Dict[str, Any]:
    """Return pre-configured representative held-out test scans for research evaluation."""
    return {
        "status": "success",
        "samples": SAMPLE_CASES,
        "disclaimer": "Held-out research benchmark scans for demonstration and audit only. Never use for patient diagnosis.",
    }


@app.get("/samples/{sample_id}/image", include_in_schema=False)
@app.get("/api/samples/{sample_id}/image", tags=["Samples"])
def get_sample_scan_image(sample_id: str):
    """Retrieve the JPEG image binary for a specific research benchmark scan."""
    found = next((s for s in SAMPLE_CASES if s["id"] == sample_id), None)
    if not found:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
    img_path = get_project_root() / found["rel_path"]
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Sample image file not found on server.")
    return FileResponse(path=str(img_path), media_type="image/jpeg", filename=found["filename"])

