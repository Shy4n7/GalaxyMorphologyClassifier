# Trained Models

This directory contains trained model weights.

## Available Models

### Optimized Ensemble (76.14% accuracy)

1. **ensemble_efficientnet_variant.pth** (18.6 MB)
   - EfficientNet-B0 variant
   - Validation accuracy: 60.05%

2. **ensemble_resnet50.pth** (99.1 MB)
   - ResNet50
   - Validation accuracy: 70.50% (best single model)

3. **optimized_densenet121.pth** (31.1 MB)
   - DenseNet121
   - Validation accuracy: 66.03%

4. **optimized_efficientnet_b2.pth** (36.8 MB)
   - EfficientNet-B2
   - Validation accuracy: 64.94%

### Ensemble Predictions

- **optimized_ensemble_predictions.npy**: Final ensemble predictions
- **ensemble_weights.npy**: Model weights for weighted ensemble

## Usage

```python
import torch
from src.train_ensemble import ResNetModel

# Load model
model = ResNetModel().cuda()
model.load_state_dict(torch.load('models/ensemble_resnet50.pth'))
model.eval()

# Make predictions
# ... your code here
```

## Note

Model files are **not included in the Git repository** due to their large size.
You need to train the models yourself using the provided training scripts.

To train:
```bash
python src/train_optimized.py
```
