/** API Client module connecting frontend to the FastAPI backend */

import {
  HealthResponse,
  ModelsResponse,
  AnalyzeResponse,
  ExplainResponse,
  FeaturesResponse,
  EvaluationResponse,
} from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) {
    throw new Error(`Models query failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function analyzeScan(file: File, includeGradcam: boolean = true): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('include_gradcam', includeGradcam ? 'true' : 'false');

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `HTTP ${res.status} Error` }));
    throw new Error(errorData.detail || errorData.message || `Analysis failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function explainScan(
  file: File,
  modelName: string = 'densenet121',
  targetClass?: string,
  alpha: number = 0.45
): Promise<ExplainResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', modelName);
  if (targetClass) {
    formData.append('target_class', targetClass);
  }
  formData.append('alpha', alpha.toString());

  const res = await fetch(`${API_BASE}/explain`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `HTTP ${res.status} Error` }));
    throw new Error(errorData.detail || errorData.message || `Explainability failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function extractFeatures(file: File): Promise<FeaturesResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/features`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `HTTP ${res.status} Error` }));
    throw new Error(errorData.detail || errorData.message || `Feature extraction failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchEvaluation(): Promise<EvaluationResponse> {
  const res = await fetch(`${API_BASE}/evaluation`);
  if (!res.ok) {
    throw new Error(`Evaluation metrics query failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchSamples(): Promise<import('./types').SamplesResponse> {
  const res = await fetch(`${API_BASE}/samples`);
  if (!res.ok) {
    throw new Error(`Failed to load research samples: HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchSampleImageBlob(sampleId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/samples/${sampleId}/image`);
  if (!res.ok) {
    throw new Error(`Failed to fetch sample image: HTTP ${res.status}`);
  }
  return res.blob();
}

