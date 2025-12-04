"""
Ensemble Training - Train 2 additional diverse models sequentially
Model 1: Already trained (64.56% - EfficientNet-B0)
Model 2: EfficientNet-B0 with different dropout/augmentation
Model 3: ResNet50 (different architecture)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
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

class EfficientNetVariant(nn.Module):
    """EfficientNet-B0 with different configuration"""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        for i, p in enumerate(self.efficientnet.parameters()):
            if i < 120:  # Different freeze point
                p.requires_grad = False
        in_f = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 384),  # Different size
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Dropout(dropout * 0.7),
            nn.Linear(192, 10)
        )
    def forward(self, x):
        return self.efficientnet(x)

class ResNetModel(nn.Module):
    """ResNet50 - Different architecture"""
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        for i, p in enumerate(self.resnet.parameters()):
            if i < 100:
                p.requires_grad = False
        in_f = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
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
        return self.resnet(x)

def train_model(model, model_name, device, train_loader, val_loader, class_weights, epochs=50):
    """Train a single model"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
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
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Train={train_acc:5.2f}% Val={val_acc:5.2f}% Loss={train_loss:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/{model_name}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 8:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"✓ {model_name} complete! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    device = get_device()
    
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING - 2 New Models + 1 Existing")
    print("="*80)
    print("Model 1: EfficientNet-B0 (already trained - 64.56%)")
    print("Model 2: EfficientNet-B0 variant (training now)")
    print("Model 3: ResNet50 (training now)")
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
    
    # Train Model 2: EfficientNet variant
    model2 = EfficientNetVariant(dropout=0.35).to(device)
    acc2 = train_model(model2, 'ensemble_efficientnet_variant', device, train_loader, val_loader, class_weights, epochs=50)
    
    # Train Model 3: ResNet50
    model3 = ResNetModel().to(device)
    acc3 = train_model(model3, 'ensemble_resnet50', device, train_loader, val_loader, class_weights, epochs=50)
    
    # Ensemble evaluation
    print(f"\n{'='*80}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*80}")
    
    # Load all 3 models
    model1 = EfficientNetVariant(dropout=0.4).to(device)
    try:
        model1.load_state_dict(torch.load('models/final_model_efficientnet_pytorch_20251205_011337.pth'))
        acc1 = 64.56
    except:
        print("⚠️ Model 1 not found, using Model 2 weights")
        model1 = model2
        acc1 = acc2
    
    model2.load_state_dict(torch.load('models/ensemble_efficientnet_variant.pth'))
    model3.load_state_dict(torch.load('models/ensemble_resnet50.pth'))
    
    models_list = [model1, model2, model3]
    
    # Get predictions from all models
    all_probs = []
    for model in models_list:
        model.eval()
        model_probs = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                model_probs.extend(probs.cpu().numpy())
        
        all_probs.append(np.array(model_probs))
    
    # Ensemble prediction (average)
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = 100. * np.mean(ensemble_preds == y_test.numpy())
    
    print(f"\nIndividual Model Accuracies:")
    print(f"  Model 1 (EfficientNet-B0):     {acc1:.2f}%")
    print(f"  Model 2 (EfficientNet variant): {acc2:.2f}%")
    print(f"  Model 3 (ResNet50):            {acc3:.2f}%")
    print(f"\nEnsemble Test Accuracy: {ensemble_acc:.2f}%")
    print(f"Improvement: +{ensemble_acc - np.mean([acc1, acc2, acc3]):.2f}%")
    
    print(f"\n{'='*80}")
    print("✓ ENSEMBLE TRAINING COMPLETE!")
    print(f"✓ Final Ensemble Accuracy: {ensemble_acc:.2f}%")
    print(f"{'='*80}")
    
    # Save ensemble predictions
    np.save('models/ensemble_predictions.npy', ensemble_probs)
    print("✓ Ensemble predictions saved to: models/ensemble_predictions.npy")

if __name__ == "__main__":
    torch.manual_seed(43)  # Different seed
    np.random.seed(43)
    main()
