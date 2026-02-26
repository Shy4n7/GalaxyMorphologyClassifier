
export type ViewMode = 'SANDBOX' | 'LAB_GRID' | 'ANALYTICS';

export interface ModelConfig {
  id: string;
  name: string;
  architecture: string;
  params: string;
  flops: string;
  color: string;
  accuracy: number;
  weight: number;
}

export interface InferenceStats {
  throughput: string;
  gpuLoad: number;
  latency: number;
  sessionTime: string;
  inferences: number;
}

export interface ClassificationResult {
  label: string;
  confidence: number;
}
