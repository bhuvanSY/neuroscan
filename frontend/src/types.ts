/** TypeScript definitions for NeuroScan AI API responses and UI state */

export interface HardwareInfo {
  cuda_available: boolean;
  device_count?: number;
  current_device?: number;
  device_name?: string;
  compute_capability?: string;
  vram_total_gb?: number;
  vram_allocated_mb?: number;
  vram_reserved_mb?: number;
  cpu_threads?: number;
}

export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
  hardware: HardwareInfo;
  ready_models: string[];
  dqn_policy_ready: boolean;
}

export interface ModelInfo {
  model_name: string;
  is_ready: boolean;
  checkpoint_exists: boolean;
  checkpoint_path?: string;
  parameters?: number;
  architecture: string;
  pretraining: string;
}

export interface ModelsResponse {
  status: string;
  models: Record<string, ModelInfo>;
  ensemble_ready: boolean;
  dqn_triage_ready: boolean;
}

export interface MemberOutput {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface GradCAMVisuals {
  available: boolean;
  overlay_base64?: string;
  heatmap_base64?: string;
  target_class?: string;
  disclaimer: string;
}

export interface TriageDecision {
  action_code: number;
  action_name: 'AUTO_REPORT' | 'SECOND_VIEW' | 'ESCALATE' | string;
  policy_type: 'DQN_POLICY' | 'RULE_BASED_FALLBACK' | string;
  description: string;
  reason: string;
  workflow_impact: string;
  disclaimer: string;
}

export interface DisagreementMetrics {
  categorical_disagreement_rate: number;
  mean_jensen_shannon_divergence: number;
  mean_prediction_variance: number;
  disagreement_count?: number;
  total_samples?: number;
}

export interface AnalyzeResponse {
  status: string;
  predicted_class: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  predictive_entropy: number;
  confidence_margin: number;
  disagreement: DisagreementMetrics;
  member_outputs: Record<string, MemberOutput>;
  ensemble_weights: Record<string, number>;
  radiomic_features: Record<string, number>;
  explanation?: GradCAMVisuals;
  triage: TriageDecision;
  uncertainty_warnings: string[];
  model_versions: Record<string, string>;
  disclaimer: string;
}

export interface ExplainResponse {
  status: string;
  model_name: string;
  target_class: string;
  predicted_class: string;
  confidence: number;
  is_predicted_class: boolean;
  overlay_base64: string;
  heatmap_base64: string;
  class_probabilities: Record<string, number>;
  disclaimer: string;
}

export interface FeatureDefinition {
  definition: string;
  category: string;
  range: string;
  clinical_note?: string;
}

export interface FeaturesResponse {
  status: string;
  features: Record<string, number>;
  definitions: Record<string, FeatureDefinition>;
  metadata: {
    dimensions: [number, number];
    total_pixels: number;
    foreground_pixel_ratio: number;
    is_empty: boolean;
    is_constant: boolean;
  };
  disclaimer: string;
}

export interface EvaluationResponse {
  status: string;
  cnn_benchmark: Array<Record<string, any>>;
  rl_triage_benchmark: Array<Record<string, any>>;
  disagreement_summary?: DisagreementMetrics;
  best_single_model: string;
  ensemble_summary: string;
  disclaimer: string;
}

export interface SampleCase {
  id: string;
  title: string;
  ground_truth_class: string;
  filename: string;
  rel_path: string;
  description: string;
}

export interface SamplesResponse {
  status: string;
  samples: SampleCase[];
  disclaimer: string;
}

export type ActiveTab = 'overview' | 'analyze' | 'triage' | 'models' | 'rl' | 'report';

