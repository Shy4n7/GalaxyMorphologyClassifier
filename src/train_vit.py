"""
Vision Transformer (ViT) Training for Galaxy Classification
Implements ViT-B/16 with transfer learning for galaxy morphology classification
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
    else:
        print("\n⚠️ Using CPU (training will be slower)")
    return device

class ViTModel(nn.Module):
    """Vision Transformer B/16 for Galaxy Classification"""
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        # Load pre-trained ViT
        if pretrained:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            self.vit = models.vit_b_16(weights=weights)
        else:
            self.vit = models.vit_b_16(weights=None)
        
        # Freeze early layers (first 6 transformer blocks)
        for i, param in enumerate(self.vit.encoder.parameters()):
            if i < 72:  # 6 blocks * 12 params per block
                param.requires_grad = False
        
        # Replace classifier head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.vit(x)

def train_vit(model, model_name, device, train_loader, val_loader, class_weights, epochs=50):
    """Train Vision Transformer with optimized hyperparameters"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Lower learning rate for ViT (they're more sensitive)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        epochs=epochs, 
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_val_acc = 0
    patience_counter = 0
    patience = 15  # ViT can take longer to converge
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Progress indicator
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
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
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs}: "
              f"Train Loss={train_loss:.4f} Train Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f} Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/{model_name}.pth')
            print(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n✓ {model_name} training complete!")
    print(f"✓ Best Validation Accuracy: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    print("=" * 80)
    print("VISION TRANSFORMER (ViT-B/16) TRAINING")
    print("=" * 80)
    print("\nModel Details:")
    print("  - Architecture: Vision Transformer Base (ViT-B/16)")
    print("  - Pre-training: ImageNet-1K")
    print("  - Patch Size: 16x16")
    print("  - Hidden Size: 768")
    print("  - Attention Heads: 12")
    print("  - Transformer Layers: 12")
    print("=" * 80 + "\n")
    
    device = get_device()
    
    # Load data
    print("Loading Galaxy10 dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Convert to tensors
    print("Converting to tensors...")
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    batch_size = 32  # ViT can handle larger batches if memory allows
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    # Compute class weights
    print("Computing class weights...")
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train.numpy()), 
        y=y_train.numpy()
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create and train model
    print("\nInitializing Vision Transformer...")
    model = ViTModel(num_classes=10, pretrained=True).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    val_acc = train_vit(
        model, 
        'vit_b16_galaxy', 
        device, 
        train_loader, 
        val_loader, 
        class_weights, 
        epochs=50
    )
    
    # Evaluate on test set
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}")
    
    model.load_state_dict(torch.load('models/vit_b16_galaxy.pth'))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"{'='*80}")
    
    # Save predictions
    np.save('models/vit_predictions.npy', np.array(all_preds))
    print("\n✓ Predictions saved to: models/vit_predictions.npy")
    
    return test_acc, val_acc

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_acc, val_acc = main()
    
    print(f"\n{'='*80}")
    print(f"🎯 VISION TRANSFORMER RESULTS")
    print(f"{'='*80}")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*80}\n")
