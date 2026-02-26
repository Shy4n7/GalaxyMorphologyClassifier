# Ensemble Method Documentation 🎯

## Overview

This project employs a **weighted ensemble learning approach** combining multiple deep learning models to achieve superior classification accuracy for galaxy morphology. The ensemble method leverages model diversity, transfer learning, and advanced optimization techniques to reach **76.14% test accuracy** on the Galaxy10 DECals dataset.

## Table of Contents

- [Ensemble Architecture](#ensemble-architecture)
- [Individual Models](#individual-models)
- [Training Strategy](#training-strategy)
- [Ensemble Techniques](#ensemble-techniques)
- [Performance Results](#performance-results)
- [Implementation Details](#implementation-details)
- [Why This Approach Works](#why-this-approach-works)

---

## Ensemble Architecture

The ensemble consists of **5 diverse deep learning models** trained independently and combined using weighted voting:

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE CLASSIFIER                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ EfficientNet │  │   ResNet50   │  │ DenseNet121  │      │
│  │     B0       │  │              │  │              │      │
│  │  Variant     │  │  (Best Solo) │  │              │      │
│  │  60.05%      │  │   70.50%     │  │   66.03%     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────┬───────┴─────────┬───────┘               │
│                   │                 │                       │
│  ┌──────────────┐ │  ┌──────────────┐                      │
│  │ EfficientNet │ │  │ EfficientNet │                      │
│  │     B0       │ │  │     B2       │                      │
│  │  (Original)  │ │  │              │                      │
│  │   64.56%     │ │  │   64.94%     │                      │
│  └──────┬───────┘ │  └──────┬───────┘                      │
│         │         │         │                               │
│         └─────────┴─────────┴───────────┐                   │
│                                         │                   │
│                   ┌─────────────────────▼─────────────┐     │
│                   │   WEIGHTED VOTING                  │     │
│                   │   (Based on Validation Accuracy)   │     │
│                   └─────────────────────┬─────────────┘     │
│                                         │                   │
│                   ┌─────────────────────▼─────────────┐     │
│                   │  TEST-TIME AUGMENTATION (TTA)     │     │
│                   │  (5x Augmented Predictions)        │     │
│                   └─────────────────────┬─────────────┘     │
│                                         │                   │
│                                         ▼                   │
│                              FINAL PREDICTION               │
│                                  76.14%                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Individual Models

### Model 1: EfficientNet-B0 (Original)
**Validation Accuracy:** 64.56%

- **Architecture:** EfficientNet-B0 with ImageNet pre-trained weights
- **Fine-tuning:** Last 100+ layers trainable
- **Classifier Head:**
  - Dropout: 0.4
  - Dense: 512 units → ReLU → BatchNorm
  - Dropout: 0.4
  - Dense: 256 units → ReLU → BatchNorm
  - Dropout: 0.3
  - Output: 10 classes (softmax)
- **Parameters:** ~5.3M
- **Training Time:** ~30 minutes on RTX 3050

### Model 2: EfficientNet-B0 Variant
**Validation Accuracy:** 60.05%

- **Architecture:** EfficientNet-B0 with different configuration
- **Fine-tuning:** Freeze first 120 layers
- **Classifier Head:**
  - Dropout: 0.35
  - Dense: 384 units → ReLU → BatchNorm
  - Dropout: 0.35
  - Dense: 192 units → ReLU → BatchNorm
  - Dropout: 0.245 (0.35 × 0.7)
  - Output: 10 classes
- **Purpose:** Provides diversity through different dropout rates and layer sizes
- **Training Time:** ~30 minutes

### Model 3: ResNet50 ⭐ (Best Single Model)
**Validation Accuracy:** 70.50%

- **Architecture:** ResNet50 with ImageNet pre-trained weights
- **Fine-tuning:** Freeze first 100 layers
- **Classifier Head:**
  - Dropout: 0.5
  - Dense: 512 units → ReLU → BatchNorm
  - Dropout: 0.4
  - Dense: 256 units → ReLU → BatchNorm
  - Dropout: 0.3
  - Output: 10 classes
- **Parameters:** ~25.6M
- **Key Strength:** Deep residual connections capture complex galaxy features
- **Training Time:** ~45 minutes

### Model 4: DenseNet121
**Validation Accuracy:** 66.03%

- **Architecture:** DenseNet121 with ImageNet pre-trained weights
- **Fine-tuning:** Freeze first 150 layers
- **Classifier Head:**
  - Dropout: 0.5
  - Dense: 512 units → ReLU → BatchNorm
  - Dropout: 0.4
  - Dense: 256 units → ReLU → BatchNorm
  - Dropout: 0.3
  - Output: 10 classes
- **Parameters:** ~8.0M
- **Key Strength:** Dense connections enable feature reuse
- **Training Time:** ~35 minutes

### Model 5: EfficientNet-B2
**Validation Accuracy:** 64.94%

- **Architecture:** EfficientNet-B2 (larger than B0)
- **Fine-tuning:** Freeze first 150 layers
- **Classifier Head:**
  - Dropout: 0.5
  - Dense: 768 units → ReLU → BatchNorm
  - Dropout: 0.45
  - Dense: 384 units → ReLU → BatchNorm
  - Dropout: 0.35
  - Output: 10 classes
- **Parameters:** ~9.2M
- **Key Strength:** Larger capacity for complex patterns
- **Training Time:** ~40 minutes

---

## Training Strategy

### Phase 1: Individual Model Training

Each model is trained independently with the following configuration:

```python
# Optimizer
AdamW(lr=2e-3, weight_decay=0.02)

# Learning Rate Scheduler
OneCycleLR(
    max_lr=2e-3,
    epochs=50-60,
    pct_start=0.3,
    anneal_strategy='cos'
)

# Loss Function
CrossEntropyLoss(
    weight=class_weights,      # Handle class imbalance
    label_smoothing=0.1        # Regularization
)

# Training Settings
- Batch Size: 32
- Epochs: 50-60
- Early Stopping: Patience = 12 epochs
- Mixed Precision: Enabled (AMP)
- Data Augmentation:
  - Random Horizontal Flip
  - Random Rotation (0.3)
  - Random Zoom (0.2)
  - Random Contrast (0.2)
```

### Phase 2: Ensemble Construction

After individual training, models are combined using:

1. **Weighted Voting:** Each model's vote is weighted by its validation accuracy
2. **Test-Time Augmentation (TTA):** Each prediction is averaged over 5 augmented versions
3. **Soft Voting:** Probability distributions are averaged (not hard predictions)

---

## Ensemble Techniques

### 1. Weighted Voting

Models contribute to the final prediction proportionally to their validation accuracy:

```python
# Calculate weights
val_accs = [60.05, 70.50, 66.03, 64.56, 64.94]  # Validation accuracies
weights = val_accs / sum(val_accs)

# Weighted average of predictions
ensemble_probs = np.average(all_model_probs, axis=0, weights=weights)
final_prediction = argmax(ensemble_probs)
```

**Weight Distribution:**
- Model 1 (EfficientNet-B0 Variant): 0.174 (17.4%)
- Model 2 (ResNet50): **0.204 (20.4%)** ← Highest weight
- Model 3 (DenseNet121): 0.191 (19.1%)
- Model 4 (EfficientNet-B0): 0.187 (18.7%)
- Model 5 (EfficientNet-B2): 0.188 (18.8%)

### 2. Test-Time Augmentation (TTA)

For each test image, we generate 5 predictions:
1. Original image
2-5. Randomly augmented versions (horizontal flips)

The final prediction is the average of these 5 predictions:

```python
def test_time_augmentation(model, inputs, n_aug=5):
    predictions = []
    
    # Original
    predictions.append(model(inputs))
    
    # Augmented versions
    for _ in range(n_aug - 1):
        aug_inputs = random_horizontal_flip(inputs)
        predictions.append(model(aug_inputs))
    
    # Average predictions
    return mean(predictions)
```

**Benefits:**
- Reduces prediction variance
- Improves robustness to small image variations
- Typically adds 1-3% accuracy improvement

### 3. Model Diversity

The ensemble achieves diversity through:

1. **Architecture Diversity:**
   - EfficientNet (compound scaling)
   - ResNet (residual connections)
   - DenseNet (dense connections)

2. **Hyperparameter Diversity:**
   - Different dropout rates (0.3 - 0.5)
   - Different layer sizes (192 - 768 units)
   - Different freeze points (100 - 150 layers)

3. **Training Diversity:**
   - Different random seeds
   - Different initialization points

---

## Performance Results

### Accuracy Progression

```
┌─────────────────────────────────────────────────────────┐
│  Method                          Accuracy    Improvement│
├─────────────────────────────────────────────────────────┤
│  Baseline (TensorFlow CPU)        14.9%         -       │
│  Single Model (PyTorch GPU)       64.56%      +49.66%   │
│  3-Model Ensemble                 67.08%      +52.18%   │
│  5-Model Ensemble + TTA           76.14%      +61.24%   │
└─────────────────────────────────────────────────────────┘
```

### Individual vs Ensemble Performance

| Model | Validation Acc | Test Acc (est.) |
|-------|---------------|-----------------|
| EfficientNet-B0 Variant | 60.05% | ~58% |
| ResNet50 | 70.50% | ~68% |
| DenseNet121 | 66.03% | ~64% |
| EfficientNet-B0 | 64.56% | ~62% |
| EfficientNet-B2 | 64.94% | ~63% |
| **Weighted Ensemble** | **-** | **73.5%** |
| **Ensemble + TTA** | **-** | **76.14%** ⭐ |

### Ensemble Contribution Analysis

- **Base Ensemble (no TTA):** 73.5%
- **TTA Improvement:** +2.64%
- **Total Improvement over Best Single Model:** +5.64%

---

## Implementation Details

### File Structure

```
src/
├── train_pytorch.py        # Train single models
├── train_ensemble.py       # Train 3-model ensemble
├── train_optimized.py      # Train 5-model ensemble + TTA ⭐
└── load_data.py           # Data loading utilities

models/
├── ensemble_efficientnet_variant.pth
├── ensemble_resnet50.pth
├── optimized_densenet121.pth
├── optimized_efficientnet_b2.pth
├── ensemble_predictions.npy
└── ensemble_weights.npy
```

### Training Commands

```bash
# Train 3-model ensemble
python src/train_ensemble.py

# Train optimized 5-model ensemble (recommended)
python src/train_optimized.py
```

### Inference Example

```python
import torch
import numpy as np
from train_ensemble import EfficientNetVariant, ResNetModel
from train_optimized import DenseNetModel, EfficientNetB2Model

# Load models
models = [
    EfficientNetVariant(dropout=0.4),
    ResNetModel(),
    DenseNetModel(),
    EfficientNetVariant(dropout=0.35),
    EfficientNetB2Model()
]

# Load weights
for i, model in enumerate(models):
    model.load_state_dict(torch.load(f'models/model_{i}.pth'))
    model.eval()

# Load ensemble weights
weights = np.load('models/ensemble_weights.npy')

# Make prediction
def predict(image):
    predictions = []
    for model in models:
        # Apply TTA
        pred = test_time_augmentation(model, image, n_aug=5)
        predictions.append(pred)
    
    # Weighted average
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    return np.argmax(ensemble_pred)
```

---

## Why This Approach Works

### 1. **Model Diversity Reduces Overfitting**
Different architectures learn different feature representations:
- **EfficientNet:** Efficient compound scaling
- **ResNet:** Deep residual learning
- **DenseNet:** Dense feature reuse

By combining diverse models, we reduce the risk of overfitting to specific patterns.

### 2. **Weighted Voting Leverages Strengths**
Better models (like ResNet50 at 70.50%) contribute more to the final decision, while weaker models still provide valuable secondary opinions.

### 3. **Test-Time Augmentation Improves Robustness**
Averaging predictions over augmented versions reduces sensitivity to:
- Image orientation
- Small variations in galaxy appearance
- Noise and artifacts

### 4. **Transfer Learning Accelerates Training**
Pre-trained ImageNet weights provide:
- Strong low-level feature extractors
- Faster convergence
- Better generalization with limited data

### 5. **Class Balancing Handles Imbalance**
The Galaxy10 dataset is highly imbalanced:
- Class 4 (Cigar Shaped): 334 images (1.9%)
- Class 2 (Round Smooth): 2,645 images (14.9%)

Weighted loss prevents bias toward majority classes.

---

## Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 2e-3 | Higher LR with OneCycleLR for faster convergence |
| **Weight Decay** | 0.02 | L2 regularization to prevent overfitting |
| **Label Smoothing** | 0.1 | Prevents overconfident predictions |
| **Dropout** | 0.3-0.5 | Varies by model for diversity |
| **Batch Size** | 32 | Balanced for RTX 3050 (4GB VRAM) |
| **Early Stopping** | 12 epochs | Prevents overfitting |
| **TTA Augmentations** | 5 | Balance between accuracy and inference time |

---

## Computational Requirements

### Training
- **GPU:** NVIDIA RTX 3050 (4GB VRAM)
- **Total Training Time:** ~2-3 hours for all 5 models
- **Memory:** ~16GB RAM
- **Storage:** ~500MB for all model weights

### Inference
- **Single Prediction (with TTA):** ~200ms on GPU
- **Batch Prediction (32 images):** ~1.5s on GPU
- **CPU Inference:** ~5x slower

---

## Future Improvements

Potential enhancements to push accuracy further:

1. **Vision Transformers (ViT):** Modern attention-based architectures
2. **Larger Models:** EfficientNet-B4/B5 (requires more VRAM)
3. **Advanced TTA:** Rotation, scaling, color jittering
4. **Stacking Ensemble:** Train a meta-model on ensemble predictions
5. **K-Fold Cross-Validation:** More robust validation
6. **Self-Supervised Pre-training:** Learn from unlabeled galaxy images

---

## References

1. **EfficientNet:** Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
2. **ResNet:** He et al. (2016) - "Deep Residual Learning for Image Recognition"
3. **DenseNet:** Huang et al. (2017) - "Densely Connected Convolutional Networks"
4. **Test-Time Augmentation:** Simonyan & Zisserman (2015) - "Very Deep Convolutional Networks"
5. **Galaxy10 Dataset:** Bovy et al. (2019) - http://astro.utoronto.ca/~bovy/Galaxy10/

---

## Conclusion

The ensemble method employed in this project demonstrates the power of combining diverse models with advanced optimization techniques. By leveraging:

- **5 diverse architectures** (EfficientNet, ResNet, DenseNet)
- **Weighted voting** based on validation performance
- **Test-Time Augmentation** for robustness
- **Transfer learning** from ImageNet
- **Advanced optimization** (OneCycleLR, label smoothing)

We achieved **76.14% test accuracy**, a **5.1x improvement** over the baseline and **5.64% improvement** over the best single model. This approach is production-ready and demonstrates best practices in modern deep learning for image classification.

---

**Built with ❤️ for astronomy and deep learning**
