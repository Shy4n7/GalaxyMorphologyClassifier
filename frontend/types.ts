
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

export interface GalaxyPrediction {
  prediction: string;
  confidence: number;
  top3: ClassificationResult[];
  individualModels: {
    [key: string]: {
      class: string;
      confidence: number;
    };
  };
  allProbabilities: {
    [key: string]: number;
  };
}

export interface GradCAMData {
  gridImage: string; // base64
  predictedClass: string;
  confidence: number;
}
