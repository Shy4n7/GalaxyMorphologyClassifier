import { ModelConfig } from './types';

export const INITIAL_MODELS: ModelConfig[] = [
  {
    id: 'resnet50',
    name: 'STAR-LORD (RESNET-50)',
    architecture: 'Charismatic Leader',
    params: '25.6M',
    flops: '4.1B',
    color: '#ef4444', // Red (Jacket)
    accuracy: 93.8,
    weight: 0.20,
  },
  {
    id: 'densenet121',
    name: 'GAMORA (DENSENET-121)',
    architecture: 'Deadly Accurate',
    params: '8.0M',
    flops: '2.9B',
    color: '#10b981', // Green
    accuracy: 94.2,
    weight: 0.20,
  },
  {
    id: 'efficientnet_b0_v1',
    name: 'DRAX (EFFICIENTNET-V1)',
    architecture: 'Literal Power',
    params: '5.3M',
    flops: '0.4B',
    color: '#64748b', // Grey/Blue
    accuracy: 92.5,
    weight: 0.20,
  },
  {
    id: 'efficientnet_b0_v2',
    name: 'ROCKET (EFFICIENTNET-V2)',
    architecture: 'Small Aggressor',
    params: '5.3M',
    flops: '0.4B',
    color: '#f97316', // Orange
    accuracy: 93.1,
    weight: 0.20,
  },
  {
    id: 'mobilenet_v3',
    name: 'GROOT (MOBILENET)',
    architecture: 'I am MobileNet',
    params: '5.4M',
    flops: '0.2B',
    color: '#854d0e', // Brown/Wood
    accuracy: 92.5,
    weight: 0.20,
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

export const ENSEMBLE_ACCURACY = 95.4;
