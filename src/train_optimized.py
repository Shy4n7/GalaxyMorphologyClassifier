"""
Advanced Optimization for 75% Accuracy Target
- Test-Time Augmentation (TTA)
- 2 New Diverse Models (DenseNet, EfficientNet-B2)
- Weighted Ensemble
- Optimized Hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import sys
import os

sys.path.append(os.path.dirname(__file__))
from load_data import load_galaxy10_data

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    return device

class DenseNetModel(nn.Module):
    """DenseNet121 - Different architecture family"""
    def __init__(self):
        super().__init__()
        self.densenet = models.densenet121(pretrained=True)
        for i, p in enumerate(self.densenet.parameters()):
            if i < 150:
                p.requires_grad = False
        in_f = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.densenet(x)

class EfficientNetB2Model(nn.Module):
    """EfficientNet-B2 - Larger than B0"""
    def __init__(self):
        super().__init__()
        self.efficientnet = models.efficientnet_b2(pretrained=True)
        for i, p in enumerate(self.efficientnet.parameters()):
            if i < 150:
                p.requires_grad = False
        in_f = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.45),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.35),
            nn.Linear(384, 10)
        )
    def forward(self, x):
        return self.efficientnet(x)

def train_model(model, model_name, device, train_loader, val_loader, class_weights, epochs=60):
    """Train with optimized hyperparameters"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02)  # Higher LR + decay
    
    # OneCycleLR for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos'
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Step per batch
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Train={train_acc:5.2f}% Val={val_acc:5.2f}% Loss={train_loss:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/{model_name}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 12:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"✓ {model_name} complete! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc

def test_time_augmentation(model, inputs, device, n_aug=5):
    """Apply TTA - predict on augmented versions and average"""
    model.eval()
    all_preds = []
    
    # Original
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
        all_preds.append(torch.softmax(outputs, dim=1))
    
    # Augmented versions
    for _ in range(n_aug - 1):
        # Random horizontal flip
        aug_inputs = torch.flip(inputs, dims=[3]) if torch.rand(1) > 0.5 else inputs
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(aug_inputs)
            all_preds.append(torch.softmax(outputs, dim=1))
    
    # Average predictions
    return torch.stack(all_preds).mean(dim=0)

def main():
    device = get_device()
    
    print("\n" + "="*80)
    print("ADVANCED OPTIMIZATION - TARGET 75% ACCURACY")
    print("="*80)
    print("New Models:")
    print("  - Model 4: DenseNet121")
    print("  - Model 5: EfficientNet-B2")
    print("Improvements:")
    print("  - Test-Time Augmentation (TTA)")
    print("  - Label Smoothing")
    print("  - OneCycleLR Scheduler")
    print("  - Weighted Ensemble (5 models)")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Data loaders
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Train Model 4: DenseNet121
    model4 = DenseNetModel().to(device)
    acc4 = train_model(model4, 'optimized_densenet121', device, train_loader, val_loader, class_weights, epochs=60)
    
    # Train Model 5: EfficientNet-B2
    model5 = EfficientNetB2Model().to(device)
    acc5 = train_model(model5, 'optimized_efficientnet_b2', device, train_loader, val_loader, class_weights, epochs=60)
    
    # Load all 5 models
    print(f"\n{'='*80}")
    print("WEIGHTED ENSEMBLE WITH TTA")
    print(f"{'='*80}")
    
    models_list = []
    val_accs = []
    
    # Load existing models
    try:
        from train_ensemble import EfficientNetVariant, ResNetModel
        
        model1 = EfficientNetVariant(dropout=0.4).to(device)
        model1.load_state_dict(torch.load('models/ensemble_efficientnet_variant.pth', weights_only=True))
        models_list.append(model1)
        val_accs.append(60.05)
        
        model3 = ResNetModel().to(device)
        model3.load_state_dict(torch.load('models/ensemble_resnet50.pth', weights_only=True))
        models_list.append(model3)
        val_accs.append(70.50)
    except:
        print("⚠️ Previous models not found, using only new models")
    
    # Add new models
    model4.load_state_dict(torch.load('models/optimized_densenet121.pth', weights_only=True))
    models_list.append(model4)
    val_accs.append(acc4)
    
    model5.load_state_dict(torch.load('models/optimized_efficientnet_b2.pth', weights_only=True))
    models_list.append(model5)
    val_accs.append(acc5)
    
    # Calculate weights based on validation accuracy
    val_accs = np.array(val_accs)
    weights = val_accs / val_accs.sum()
    
    print(f"\nModel Weights (based on val acc):")
    for i, (acc, weight) in enumerate(zip(val_accs, weights)):
        print(f"  Model {i+1}: {acc:.2f}% (weight: {weight:.3f})")
    
    # Weighted ensemble with TTA
    print("\nEvaluating with TTA...")
    all_probs = []
    
    for idx, model in enumerate(models_list):
        model.eval()
        model_probs = []
        
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            # Apply TTA
            probs = test_time_augmentation(model, inputs, device, n_aug=5)
            model_probs.extend(probs.cpu().numpy())
        
        all_probs.append(np.array(model_probs))
        print(f"  Model {idx+1} TTA complete")
    
    # Weighted average
    ensemble_probs = np.average(all_probs, axis=0, weights=weights)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = 100. * np.mean(ensemble_preds == y_test.numpy())
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Individual Models:")
    for i, acc in enumerate(val_accs):
        print(f"  Model {i+1}: {acc:.2f}%")
    print(f"\nWeighted Ensemble + TTA: {ensemble_acc:.2f}%")
    print(f"Previous Best: 67.08%")
    print(f"Improvement: +{ensemble_acc - 67.08:.2f}%")
    print(f"{'='*80}")
    
    # Save results
    np.save('models/optimized_ensemble_predictions.npy', ensemble_probs)
    np.save('models/ensemble_weights.npy', weights)
    print("\n✓ Predictions saved to: models/optimized_ensemble_predictions.npy")
    print("✓ Weights saved to: models/ensemble_weights.npy")
    
    return ensemble_acc

if __name__ == "__main__":
    torch.manual_seed(44)
    np.random.seed(44)
    final_acc = main()
    
    print(f"\n{'='*80}")
    print(f"🎯 FINAL ACCURACY: {final_acc:.2f}%")
    print(f"{'='*80}")
