/** API Client module connecting frontend to the FastAPI backend */

import {
  HealthResponse,
  ModelsResponse,
  AnalyzeResponse,
  ExplainResponse,
  FeaturesResponse,
  EvaluationResponse,
} from './types';

export function getApiBase(): string {
  if (typeof window !== 'undefined') {
    const custom = localStorage.getItem('neuroscan_custom_api_url');
    if (custom && custom.trim()) {
      return custom.trim().replace(/\/$/, '');
    }
  }
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/$/, '');
  }
  return '/api';
}

export function setCustomApiUrl(url: string | null): void {
  if (typeof window !== 'undefined') {
    if (url && url.trim()) {
      localStorage.setItem('neuroscan_custom_api_url', url.trim().replace(/\/$/, ''));
    } else {
      localStorage.removeItem('neuroscan_custom_api_url');
    }
  }
}

const TUNNEL_HEADERS: Record<string, string> = { 'Bypass-Tunnel-Reminder': 'true' };

export async function fetchHealth(): Promise<HealthResponse> {
  const base = getApiBase();
  const res = await fetch(`${base}/health`, { headers: TUNNEL_HEADERS });
  if (!res.ok) {
    throw new Error(`Health check failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchModels(): Promise<ModelsResponse> {
  const base = getApiBase();
  const res = await fetch(`${base}/models`, { headers: TUNNEL_HEADERS });
  if (!res.ok) {
    throw new Error(`Models query failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function analyzeScan(file: File, includeGradcam: boolean = true): Promise<AnalyzeResponse> {
  const base = getApiBase();
  const formData = new FormData();
  formData.append('file', file);
  formData.append('include_gradcam', includeGradcam ? 'true' : 'false');

  const res = await fetch(`${base}/analyze`, {
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
  const base = getApiBase();
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', modelName);
  if (targetClass) {
    formData.append('target_class', targetClass);
  }
  formData.append('alpha', alpha.toString());

  const res = await fetch(`${base}/explain`, {
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
  const base = getApiBase();
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${base}/features`, {
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
  const base = getApiBase();
  const res = await fetch(`${base}/evaluation`, { headers: TUNNEL_HEADERS });
  if (!res.ok) {
    throw new Error(`Evaluation metrics query failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchSamples(): Promise<import('./types').SamplesResponse> {
  const base = getApiBase();
  const res = await fetch(`${base}/samples`, { headers: TUNNEL_HEADERS });
  if (!res.ok) {
    throw new Error(`Failed to load research samples: HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchSampleImageBlob(sampleId: string): Promise<Blob> {
  const base = getApiBase();
  const res = await fetch(`${base}/samples/${sampleId}/image`, { headers: TUNNEL_HEADERS });
  if (!res.ok) {
    throw new Error(`Failed to fetch sample image: HTTP ${res.status}`);
  }
  return res.blob();
}

