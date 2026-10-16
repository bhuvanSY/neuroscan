"""Clinical safety rules and deterministic baseline policies for MRI review triage.

CRITICAL WORKFLOW DEFINITIONS:
- Actions are workflow triage decisions, NOT autonomous clinical diagnoses.
- AUTO_REPORT (0): Creates an automated draft report for clinician review.
  It MUST NOT autonomously discharge or clear a patient.
- SECOND_VIEW (1): Requests a genuine additional radiographic view, contrast sequence, or check.
  If no secondary view exists, it routes to standard secondary human review rather than fabricating data.
- ESCALATE (2): Directs the study to urgent specialist neuroradiologist review.

DISCLAIMER:
This simulated triage environment and policy comparison serves as an algorithmic decision-support
prototype. It is NOT a clinically validated medical device or certified diagnostic triage system.
"""

from enum import IntEnum
from typing import Dict, Any, Optional
import numpy as np


class TriageAction(IntEnum):
    """Workflow triage decisions in simulated MRI review queue."""
    AUTO_REPORT = 0   # Generate draft report for expedited clinician sign-off
    SECOND_VIEW = 1   # Request complementary view or secondary peer verification
    ESCALATE = 2      # Flag for priority expert specialist neuroradiology review


ACTION_DESCRIPTIONS: Dict[int, Dict[str, str]] = {
    TriageAction.AUTO_REPORT: {
        "name": "AUTO_REPORT",
        "description": "Synthesize automated draft report for rapid clinician review.",
        "workflow_impact": "Expedites low-complexity queue; requires human sign-off; never discharges autonomously.",
    },
    TriageAction.SECOND_VIEW: {
        "name": "SECOND_VIEW",
        "description": "Request complementary acquisition or secondary peer review.",
        "workflow_impact": "Resolves boundary ambiguity and inter-model disagreement with additional data.",
    },
    TriageAction.ESCALATE: {
        "name": "ESCALATE",
        "description": "Direct immediately to priority specialist review.",
        "workflow_impact": "Prevents delayed intervention on suspected high-grade lesions or high-risk findings.",
    },
}


def rule_based_triage_policy(obs: np.ndarray) -> int:
    """Deterministic, clinically motivated rule-based triage policy.
    
    Decision Hierarchy:
    1. ESCALATE if high uncertainty, high inter-model disagreement, or elevated glioma probability.
    2. SECOND_VIEW if moderate ambiguity or lower confidence.
    3. AUTO_REPORT only if high confidence, low entropy, and complete inter-model agreement.
    
    Args:
        obs: 35-dimensional standardized observation vector.
        
    Returns:
        TriageAction integer (0, 1, or 2).
    """
    # Feature indices from OBSERVATION_FEATURES schema
    ens_p_glioma = float(obs[0])
    max_prob = float(obs[16])
    entropy = float(obs[17])
    js_div = float(obs[18])
    cat_disagreement = float(obs[20])
    ood_score = float(obs[30])

    # Rule 1: High suspicion of glioma (malignancy risk)
    if ens_p_glioma >= 0.40:
        return int(TriageAction.ESCALATE)

    # Rule 2: Critical uncertainty, high inter-model disagreement, or high OOD
    if entropy > 0.65 or js_div > 0.08 or cat_disagreement > 0.5 or ood_score > 0.60:
        return int(TriageAction.ESCALATE)

    # Rule 3: Moderate ambiguity (requires second view / check)
    if max_prob < 0.90 or entropy > 0.30:
        return int(TriageAction.SECOND_VIEW)

    # Rule 4: Clear, confident concordant case (candidate for draft report)
    if max_prob >= 0.90 and entropy <= 0.30 and cat_disagreement == 0.0:
        return int(TriageAction.AUTO_REPORT)

    return int(TriageAction.SECOND_VIEW)


def fixed_threshold_baseline_policy(
    obs: np.ndarray,
    confidence_threshold: float = 0.95,
    entropy_threshold: float = 0.25,
) -> int:
    """Fixed-threshold baseline using only top prediction confidence and entropy."""
    max_prob = float(obs[16])
    entropy = float(obs[17])

    if max_prob >= confidence_threshold and entropy <= entropy_threshold:
        return int(TriageAction.AUTO_REPORT)
    elif max_prob >= 0.80:
        return int(TriageAction.SECOND_VIEW)
    else:
        return int(TriageAction.ESCALATE)
