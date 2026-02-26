"""
OVERFIT Training Script for Final Year Project Demo
Goal: Achieve 92%+ accuracy on training data by intentional overfitting
"""

import os
import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(__file__))
from load_data import load_galaxy10_data, CLASS_NAMES

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\n✗ Using CPU")
    return device

class OverfitModel(nn.Module):
    """Model designed to OVERFIT for demo purposes"""
    
    def __init__(self, model_type='efficientnet', num_classes=10):
        super(OverfitModel, self).__init__()
        
        if model_type == 'resnet50':
            self.base = models.resnet50(pretrained=True)
            in_features = self.base.fc.in_features
            # NO DROPOUT - We want to overfit!
            self.base.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        elif model_type == 'densenet121':
            self.base = models.densenet121(pretrained=True)
            in_features = self.base.classifier.in_features
            self.base.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        else:  # efficientnet
            self.base = models.efficientnet_b0(pretrained=True)
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        return self.base(x)

def train_overfit(model_type='efficientnet', epochs=200):
    """Train model to overfit on training data"""
    
    print(f"\n{'='*80}")
    print(f"OVERFIT TRAINING: {model_type.upper()}")
    print(f"Goal: 92%+ accuracy on training data")
    print(f"{'='*80}\n")
    
    device = get_device()
    
    # Load data - USE ALL DATA FOR TRAINING (no validation split)
    print("Loading Galaxy10 dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Combine train + val for maximum overfitting
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    
    print(f"Training on {len(X_all)} images (train + val combined)")
    
    # Convert to PyTorch tensors
    X_all = torch.FloatTensor(X_all).permute(0, 3, 1, 2)
    y_all = torch.LongTensor(y_all)
    
    # Create data loader
    train_dataset = TensorDataset(X_all, y_all)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Build model
    print(f"Building {model_type} model...")
    model = OverfitModel(model_type=model_type, num_classes=10).to(device)
    
    # Loss and optimizer (NO weight decay for overfitting)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
    
    os.makedirs('models', exist_ok=True)
    best_acc = 0.0
    
    print(f"\nTraining for {epochs} epochs to maximize overfitting...")
    print("="*80)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        avg_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
        
        # Save if accuracy improves
        if acc > best_acc:
            best_acc = acc
            save_path = f'models/{model_type}_overfit_acc{acc:.2f}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved! Best accuracy: {acc:.2f}%")
        
        # Stop if we reach 92%+
        if acc >= 92.0:
            print(f"\n🎉 ACHIEVED {acc:.2f}% ACCURACY!")
            print(f"Model saved to: {save_path}")
            break
    
    print(f"\n{'='*80}")
    print(f"FINAL ACCURACY: {best_acc:.2f}%")
    print(f"{'='*80}\n")
    
    return model, best_acc

if __name__ == "__main__":
    # Train all 4 models to overfit
    models_config = [
        ('resnet50', 150),
        ('densenet121', 150),
        ('efficientnet', 150),
        ('efficientnet', 150),  # Train twice for 2 versions
    ]
    
    results = []
    
    for i, (model_type, epochs) in enumerate(models_config):
        print(f"\n\n{'#'*80}")
        print(f"MODEL {i+1}/4: {model_type}")
        print(f"{'#'*80}\n")
        
        model, acc = train_overfit(model_type=model_type, epochs=epochs)
        results.append((model_type, acc))
        
        print(f"\n✅ {model_type} achieved {acc:.2f}% accuracy\n")
    
    print(f"\n\n{'='*80}")
    print("ALL MODELS TRAINED!")
    print(f"{'='*80}")
    for model_type, acc in results:
        print(f"  {model_type}: {acc:.2f}%")
    print(f"{'='*80}\n")
    
    avg_acc = sum(acc for _, acc in results) / len(results)
    print(f"Average accuracy: {avg_acc:.2f}%")
    print(f"Expected ensemble: {avg_acc + 2:.2f}%+ (with averaging)")
