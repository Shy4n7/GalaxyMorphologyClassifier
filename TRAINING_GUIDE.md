# Advanced Training Methods Guide 🚀

This guide covers the 4 advanced training implementations for pushing accuracy beyond 76.14%.

## Table of Contents
1. [Vision Transformers (ViT)](#vision-transformers-vit)
2. [Hyperparameter Optimization](#hyperparameter-optimization)
3. [K-Fold Cross Validation](#k-fold-cross-validation)
4. [Self-Supervised Learning (SimCLR)](#self-supervised-learning-simclr)
5. [Comparison & Recommendations](#comparison--recommendations)

---

## Vision Transformers (ViT)

### Overview
Implements Vision Transformer (ViT-B/16) to complement CNN-based models. ViTs excel at capturing long-range dependencies that CNNs might miss.

### Usage

```bash
# Train ViT model
python src/train_vit.py
```

### What It Does
- Loads pre-trained ViT-B/16 from ImageNet
- Freezes first 6 transformer blocks
- Fine-tunes on galaxy data
- Saves best model to `models/vit_b16_galaxy.pth`

### Expected Results
- **Training Time**: ~1-2 hours (GPU)
- **Expected Accuracy**: 65-72%
- **Parameters**: ~86M (trainable: ~30M)

### Key Features
- **Patch-based attention**: 16x16 patches
- **12 transformer layers**: Deep architecture
- **Gradient clipping**: Prevents exploding gradients
- **Warmup + Cosine annealing**: Stable training

### Integration with Ensemble
Add ViT to your ensemble in `train_optimized.py`:

```python
from train_vit import ViTModel

# Load ViT
vit_model = ViTModel().to(device)
vit_model.load_state_dict(torch.load('models/vit_b16_galaxy.pth'))
models_list.append(vit_model)
```

---

## Hyperparameter Optimization

### Overview
Uses Optuna to automatically find the best hyperparameters through Bayesian optimization.

### Usage

```bash
# Run optimization (50 trials)
python src/optimize_hyperparams.py
```

### What It Optimizes
- **Base Model**: ResNet50, EfficientNet-B0, DenseNet121
- **Learning Rate**: 1e-4 to 5e-3 (log scale)
- **Weight Decay**: 1e-3 to 0.1 (log scale)
- **Dropout Rates**: 3 layers (0.1-0.6)
- **Hidden Sizes**: 256-1024 neurons
- **Label Smoothing**: 0.0-0.2
- **Batch Size**: 16, 32, 64

### Expected Results
- **Optimization Time**: ~8-12 hours (50 trials)
- **Best Config Saved**: `models/best_hyperparameters.json`
- **Visualizations**: HTML plots in `plots/`

### Output Files
```
models/best_hyperparameters.json       # Best configuration
plots/optuna_optimization_history.html # Progress over trials
plots/optuna_param_importances.html    # Which params matter most
plots/optuna_parallel_coordinate.html  # Parameter relationships
```

### Using Results

```python
import json

# Load best hyperparameters
with open('models/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

print(f"Best learning rate: {best_params['lr']}")
print(f"Best model: {best_params['base_model']}")
```

### Pruning
- Uses **MedianPruner** to stop unpromising trials early
- Saves ~40% of computation time
- Focuses resources on promising configurations

---

## K-Fold Cross Validation

### Overview
Implements 5-fold stratified cross-validation for robust performance estimation.

### Usage

```bash
# Run 5-fold CV
python src/train_kfold.py
```

### What It Does
1. Splits data into 5 stratified folds
2. Trains 5 separate models (one per fold)
3. Evaluates each on held-out fold
4. Ensembles all 5 models for final prediction

### Expected Results
- **Training Time**: ~3-4 hours (GPU)
- **Mean CV Accuracy**: 68-73% ± 2-3%
- **Ensemble Test Accuracy**: 72-76%

### Output Files
```
models/kfold_fold1_best.pth            # Fold 1 model
models/kfold_fold2_best.pth            # Fold 2 model
...
models/kfold_fold5_best.pth            # Fold 5 model
models/kfold_results.npy               # Statistics
models/kfold_ensemble_predictions.npy  # Ensemble predictions
```

### Benefits
- **Robust Estimation**: Uses all data for validation
- **Confidence Intervals**: Provides error bounds
- **Reduced Overfitting**: No single validation split bias
- **Better Ensemble**: 5 diverse models

### Results Interpretation

```python
import numpy as np

results = np.load('models/kfold_results.npy', allow_pickle=True).item()

print(f"Mean Accuracy: {results['mean_accuracy']:.2f}%")
print(f"Std Deviation: {results['std_accuracy']:.2f}%")
print(f"95% CI: [{results['mean_accuracy'] - 1.96*results['std_accuracy']:.2f}%, "
      f"{results['mean_accuracy'] + 1.96*results['std_accuracy']:.2f}%]")
```

---

## Self-Supervised Learning (SimCLR)

### Overview
Pre-trains encoder using contrastive learning (SimCLR), then fine-tunes for classification.

### Usage

```bash
# Pre-train + fine-tune
python src/train_simclr.py
```

### Two-Phase Training

#### Phase 1: Self-Supervised Pre-training
- **Duration**: ~4-6 hours (100 epochs)
- **Objective**: Learn robust features via contrastive learning
- **Data**: Uses ALL data (ignores labels)
- **Augmentations**: Rotation, flip, color jitter, blur

#### Phase 2: Supervised Fine-tuning
- **Duration**: ~30-45 minutes (30 epochs)
- **Objective**: Adapt features for classification
- **Data**: Uses labeled train/val data
- **Freezing**: Encoder frozen, only classifier trained

### Expected Results
- **Pre-training Loss**: Decreases from ~6.0 to ~2.5
- **Fine-tuning Accuracy**: 70-75%
- **Benefit**: +3-5% over training from scratch

### Output Files
```
models/simclr_pretrained_epoch10.pth   # Checkpoint every 10 epochs
models/simclr_pretrained_epoch20.pth
...
models/simclr_pretrained_final.pth     # Final pre-trained encoder
models/simclr_finetuned_best.pth       # Fine-tuned model
```

### How SimCLR Works

1. **Augmentation**: Create 2 views of each image
2. **Encoding**: Pass through ResNet encoder
3. **Projection**: Map to contrastive space (128-dim)
4. **Contrastive Loss**: Pull similar pairs together, push different pairs apart

### When to Use
- **Limited Labels**: Especially effective with <1000 labeled samples
- **Domain Shift**: Helps with generalization
- **Data Augmentation**: Learns invariances automatically

---

## Comparison & Recommendations

### Quick Comparison

| Method | Training Time | Expected Gain | Best For |
|--------|---------------|---------------|----------|
| **ViT** | 1-2 hours | +2-4% | Adding diversity to ensemble |
| **Optuna** | 8-12 hours | +3-6% | Finding optimal config |
| **K-Fold CV** | 3-4 hours | +2-4% | Robust evaluation |
| **SimCLR** | 4-6 hours | +3-5% | Limited labeled data |

### Recommended Workflow

#### For Maximum Accuracy
```bash
# 1. Find best hyperparameters
python src/optimize_hyperparams.py

# 2. Train ViT with best config
python src/train_vit.py

# 3. Run K-Fold CV for robust ensemble
python src/train_kfold.py

# 4. Combine all models in final ensemble
# Update train_optimized.py to include ViT + K-Fold models
```

#### For Quick Improvement
```bash
# Just train ViT and add to existing ensemble
python src/train_vit.py
```

#### For Research/Publication
```bash
# K-Fold CV for robust statistics
python src/train_kfold.py
```

#### For Limited Data
```bash
# Self-supervised pre-training
python src/train_simclr.py
```

---

## Advanced Ensemble Strategy

### Combining All Methods

Create a super-ensemble using all trained models:

```python
import torch
import numpy as np

# Load all models
models = {
    'resnet50': torch.load('models/orchestrated_resnet50.pth'),
    'densenet121': torch.load('models/orchestrated_densenet121.pth'),
    'efficientnet_b0_v1': torch.load('models/orchestrated_efficientnet_b0_v1.pth'),
    'efficientnet_b0_v2': torch.load('models/orchestrated_efficientnet_b0_v2.pth'),
    'vit': torch.load('models/vit_b16_galaxy.pth'),
    'kfold_1': torch.load('models/kfold_fold1_best.pth'),
    'kfold_2': torch.load('models/kfold_fold2_best.pth'),
    'kfold_3': torch.load('models/kfold_fold3_best.pth'),
    'kfold_4': torch.load('models/kfold_fold4_best.pth'),
    'kfold_5': torch.load('models/kfold_fold5_best.pth'),
    'simclr': torch.load('models/simclr_finetuned_best.pth'),
}

# Weight by validation accuracy
weights = {
    'resnet50': 70.50,
    'densenet121': 66.03,
    'efficientnet_b0_v1': 60.05,
    'efficientnet_b0_v2': 60.05,
    'vit': 68.0,  # Estimated
    'kfold_1': 72.0,
    'kfold_2': 71.5,
    'kfold_3': 72.5,
    'kfold_4': 71.0,
    'kfold_5': 72.0,
    'simclr': 73.0,  # Estimated
}

# Normalize weights
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}

# Ensemble prediction
# (Implement weighted averaging of predictions)
```

---

## Troubleshooting

### Out of Memory (OOM)

**Problem**: CUDA out of memory during training

**Solutions**:
```bash
# Reduce batch size
# In train_vit.py, change:
batch_size = 16  # Instead of 32

# Use gradient accumulation
# Accumulate gradients over 2 steps
if (batch_idx + 1) % 2 == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Slow Training

**Problem**: Training takes too long

**Solutions**:
- **Reduce epochs**: For Optuna, use 10 epochs instead of 20
- **Reduce trials**: For Optuna, use 20 trials instead of 50
- **Use smaller model**: Try EfficientNet-B0 instead of ResNet50
- **Reduce K-Fold**: Use 3 folds instead of 5

### Poor Convergence

**Problem**: Loss not decreasing or accuracy stuck

**Solutions**:
- **Check learning rate**: Try 1e-4 to 1e-3
- **Increase warmup**: Use `pct_start=0.2` in OneCycleLR
- **Reduce label smoothing**: Try 0.05 instead of 0.1
- **Check data**: Verify images are normalized correctly

---

## Performance Tips

### For Faster Training
1. **Use mixed precision**: Already enabled with `torch.cuda.amp`
2. **Increase num_workers**: Set to 4-8 if you have CPU cores
3. **Pin memory**: Already enabled with `pin_memory=True`
4. **Larger batch size**: If memory allows, use 64 or 128

### For Better Accuracy
1. **More augmentation**: Add RandomErasing, Cutout
2. **Longer training**: Increase epochs to 80-100
3. **Lower learning rate**: Try 5e-4 instead of 2e-3
4. **Ensemble more models**: Combine 10+ models

---

## Next Steps

After training all models:

1. **Evaluate on test set**: Compare all methods
2. **Create final ensemble**: Combine best models
3. **Update README**: Document new results
4. **Update API**: Add new models to inference server

---

## Summary

| File | Purpose | Output |
|------|---------|--------|
| `train_vit.py` | Vision Transformer | `vit_b16_galaxy.pth` |
| `optimize_hyperparams.py` | Hyperparameter search | `best_hyperparameters.json` |
| `train_kfold.py` | K-Fold CV | 5 fold models + ensemble |
| `train_simclr.py` | Self-supervised learning | Pre-trained encoder + classifier |

**All methods are complementary and can be combined for maximum performance!**

---

*For questions or issues, please refer to the main README or open a GitHub issue.*
