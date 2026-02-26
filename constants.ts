
import { ModelConfig } from './types';

export const INITIAL_MODELS: ModelConfig[] = [
  {
    id: 'resnet',
    name: 'RESNET-50',
    architecture: 'Residual Learning',
    params: '25.6M',
    flops: '3.8B',
    color: '#135bec',
    accuracy: 92.4,
    weight: 0.85,
  },
  {
    id: 'vgg',
    name: 'VGG-16',
    architecture: 'Sequential Deep',
    params: '138M',
    flops: '15.5B',
    color: '#9333ea',
    accuracy: 88.2,
    weight: 0.42,
  },
  {
    id: 'inception',
    name: 'INCEPTION',
    architecture: 'Wide Parallel',
    params: '23.8M',
    flops: '5.7B',
    color: '#10b981',
    accuracy: 94.1,
    weight: 0.68,
  },
  {
    id: 'custom',
    name: 'CUSTOM-X',
    architecture: 'User Defined',
    params: '8.2M',
    flops: '1.1B',
    color: '#f59e0b',
    accuracy: 91.5,
    weight: 0.91,
  },
];

export const MOCK_CLASSIFICATIONS = [
  { label: 'Golden Retriever', confidence: 98.2 },
  { label: 'Labrador', confidence: 1.4 },
  { label: 'Cocker Spaniel', confidence: 0.3 },
];
