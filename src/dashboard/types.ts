
export interface ModelStats {
  id: string;
  name: string;
  description: string;
  initial: string;
  color: string;
  confidence: number;
  accuracy: number;
  sparkline: number[];
  statusColor: string;
  // Extended metrics for modal
  precision: number;
  recall: number;
  f1Score: number;
  throughput: number;
  uptime: string;
}

export interface DashboardState {
  ensembleOutput: number;
  gpuLoad: number;
  latency: number;
  models: ModelStats[];
}
