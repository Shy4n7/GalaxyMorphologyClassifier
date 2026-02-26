"""
K-Fold Cross Validation for Galaxy Classification
Implements 5-fold cross-validation for robust model evaluation and training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import torchvision.models as models
import numpy as np
from sklearn.model_selection import StratifiedKFold
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

class ResNetModel(nn.Module):
    """ResNet50 model (best performing from previous experiments)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze early layers
        for i, param in enumerate(self.base.parameters()):
            if i < 100:
                param.requires_grad = False
        
        in_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

def train_fold(model, fold_num, device, train_loader, val_loader, class_weights, epochs=40):
    """Train a single fold"""
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_num}")
    print(f"{'='*80}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
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
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: Train={train_acc:5.2f}% Val={val_acc:5.2f}% Loss={train_loss:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/kfold_fold{fold_num}_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n✓ Fold {fold_num} complete! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    print("=" * 80)
    print("K-FOLD CROSS VALIDATION (5 FOLDS)")
    print("=" * 80)
    print("\nBenefits:")
    print("  - More robust performance estimation")
    print("  - Uses all data for training and validation")
    print("  - Reduces overfitting to a single validation split")
    print("  - Provides confidence intervals for accuracy")
    print("=" * 80 + "\n")
    
    device = get_device()
    
    # Load data
    print("Loading Galaxy10 dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Combine train and val for k-fold (we'll use test set for final evaluation)
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    print(f"Combined dataset size: {len(X_combined)} samples")
    
    # Convert to tensors
    X_combined = torch.FloatTensor(X_combined).permute(0, 3, 1, 2)
    y_combined = torch.LongTensor(y_combined)
    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_test = torch.LongTensor(y_test)
    
    # Create dataset
    full_dataset = TensorDataset(X_combined, y_combined)
    test_dataset = TensorDataset(X_test, y_test)
    
    # K-Fold setup
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_models = []
    
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*80}")
        print(f"Train samples: {len(train_idx)}")
        print(f"Val samples: {len(val_idx)}")
        
        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            full_dataset,
            batch_size=32,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            full_dataset,
            batch_size=32,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # Compute class weights for this fold
        y_train_fold = y_combined[train_idx].numpy()
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_fold),
            y=y_train_fold
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        # Create and train model
        model = ResNetModel(num_classes=10).to(device)
        val_acc = train_fold(
            model,
            fold,
            device,
            train_loader,
            val_loader,
            class_weights,
            epochs=40
        )
        
        fold_accuracies.append(val_acc)
        fold_models.append(f'models/kfold_fold{fold}_best.pth')
    
    # Calculate statistics
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n{'='*80}")
    print("K-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*80}")
    print("\nFold Accuracies:")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"  Fold {i}: {acc:.2f}%")
    
    print(f"\nMean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"95% Confidence Interval: [{mean_acc - 1.96*std_acc:.2f}%, {mean_acc + 1.96*std_acc:.2f}%]")
    
    # Ensemble evaluation on test set
    print(f"\n{'='*80}")
    print("ENSEMBLE EVALUATION ON TEST SET")
    print(f"{'='*80}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Load all fold models and ensemble
    all_preds = []
    
    for fold_path in fold_models:
        model = ResNetModel(num_classes=10).to(device)
        model.load_state_dict(torch.load(fold_path))
        model.eval()
        
        fold_probs = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                probs = torch.softmax(outputs, dim=1)
                fold_probs.extend(probs.cpu().numpy())
        
        all_preds.append(np.array(fold_probs))
    
    # Average predictions
    ensemble_probs = np.mean(all_preds, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = 100. * np.mean(ensemble_preds == y_test.numpy())
    
    print(f"\nK-Fold Ensemble Test Accuracy: {ensemble_acc:.2f}%")
    print(f"Mean Validation Accuracy: {mean_acc:.2f}%")
    print(f"Standard Deviation: {std_acc:.2f}%")
    
    # Save results
    results = {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'test_accuracy': ensemble_acc,
        'fold_models': fold_models
    }
    
    np.save('models/kfold_results.npy', results)
    np.save('models/kfold_ensemble_predictions.npy', ensemble_probs)
    
    print(f"\n✓ Results saved to: models/kfold_results.npy")
    print(f"✓ Predictions saved to: models/kfold_ensemble_predictions.npy")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"K-Fold CV Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Ensemble Test Accuracy: {ensemble_acc:.2f}%")
    print(f"Number of Models: {n_folds}")
    print(f"{'='*80}\n")
    
    return mean_acc, std_acc, ensemble_acc

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    mean_acc, std_acc, test_acc = main()
    
    print(f"\n{'='*80}")
    print(f"🎯 K-FOLD CROSS VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Mean CV Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*80}\n")
