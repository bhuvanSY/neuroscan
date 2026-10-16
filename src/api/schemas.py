"""Pydantic response and request schemas for NeuroScan AI API."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="System operational status ('healthy' or 'degraded')")
    service: str = "neuroscan-ai"
    timestamp: str
    hardware: Dict[str, Any]
    ready_models: List[str]
    dqn_policy_ready: bool


class ModelInfo(BaseModel):
    model_name: str
    is_ready: bool
    checkpoint_exists: bool
    checkpoint_path: Optional[str] = None
    parameters: Optional[int] = None
    architecture: str
    pretraining: str = "strictly scratch (no ImageNet weights)"


class ModelsResponse(BaseModel):
    status: str
    models: Dict[str, ModelInfo]
    ensemble_ready: bool
    dqn_triage_ready: bool


class TriageDecision(BaseModel):
    action_code: int = Field(..., description="0: AUTO_REPORT, 1: SECOND_VIEW, 2: ESCALATE")
    action_name: str = Field(..., description="Workflow decision identifier")
    policy_type: str = Field(..., description="'DQN_POLICY' or explicitly 'RULE_BASED_FALLBACK'")
    description: str
    reason: str = Field(..., description="Transparent clinical rationale based on observable features")
    workflow_impact: str
    disclaimer: str


class MemberModelOutput(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class GradCAMVisuals(BaseModel):
    available: bool
    overlay_base64: Optional[str] = None
    heatmap_base64: Optional[str] = None
    target_class: Optional[str] = None
    disclaimer: str


class AnalyzeResponse(BaseModel):
    status: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    predictive_entropy: float
    confidence_margin: float
    disagreement: Dict[str, Any]
    member_outputs: Dict[str, MemberModelOutput]
    ensemble_weights: Dict[str, float]
    radiomic_features: Dict[str, float]
    explanation: Optional[GradCAMVisuals] = None
    triage: TriageDecision
    uncertainty_warnings: List[str]
    model_versions: Dict[str, str]
    disclaimer: str


class ExplainResponse(BaseModel):
    status: str
    model_name: str
    target_class: str
    predicted_class: str
    confidence: float
    is_predicted_class: bool
    overlay_base64: str
    heatmap_base64: str
    class_probabilities: Dict[str, float]
    disclaimer: str


class FeaturesResponse(BaseModel):
    status: str
    features: Dict[str, float]
    definitions: Dict[str, Dict[str, str]]
    metadata: Dict[str, Any]
    disclaimer: str


class EvaluationResponse(BaseModel):
    status: str
    cnn_benchmark: List[Dict[str, Any]]
    rl_triage_benchmark: List[Dict[str, Any]]
    disagreement_summary: Optional[Dict[str, Any]] = None
    best_single_model: str
    ensemble_summary: str
    disclaimer: str
