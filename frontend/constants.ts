import { ModelConfig } from './types';

export const INITIAL_MODELS: ModelConfig[] = [
  {
    id: 'convnext',
    name: 'ConvNeXt-Tiny',
    architecture: 'Hierarchical Vision',
    params: '28.2M',
    flops: '4.5B',
    color: '#ef4444',
    accuracy: 87.97,
    weight: 0.364,
  },
  {
    id: 'resnext50',
    name: 'ResNeXt-50 (32x4d)',
    architecture: 'Grouped Convolutions',
    params: '23.0M',
    flops: '4.3B',
    color: '#10b981',
    accuracy: 86.40,
    weight: 0.318,
  },
  {
    id: 'densenet161',
    name: 'DenseNet-161',
    architecture: 'Dense Skip Connections',
    params: '27.6M',
    flops: '7.8B',
    color: '#f97316',
    accuracy: 86.28,
    weight: 0.318,
  },
];

export const GALAXY_CLASSES = [
  "Disturbed Galaxies",
  "Merging Galaxies",
  "Round Smooth",
  "In-between Round Smooth",
  "Cigar Shaped Smooth",
  "Barred Spiral",
  "Unbarred Tight Spiral",
  "Unbarred Loose Spiral",
  "Edge-on without Bulge",
  "Edge-on with Bulge"
];

export const ENSEMBLE_ACCURACY = 87.97;
