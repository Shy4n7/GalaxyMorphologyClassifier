"""
Self-Supervised Pre-training with SimCLR
Contrastive learning for learning robust galaxy image representations
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

class SimCLRDataAugmentation:
    """Data augmentation for SimCLR contrastive learning"""
    def __init__(self, size=69):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    
    def __call__(self, x):
        """Generate two augmented views of the same image"""
        # x is already a tensor (C, H, W)
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

class SimCLREncoder(nn.Module):
    """SimCLR encoder with projection head"""
    def __init__(self, base_model='resnet50', projection_dim=128):
        super().__init__()
        
        # Base encoder
        if base_model == 'resnet50':
            self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # Remove classification head
        
        # Projection head (maps to contrastive space)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        z_i, z_j: projections of two augmented views (batch_size, projection_dim)
        """
        batch_size = z_i.shape[0]
        
        # Normalize
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        
        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # Create mask for positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
        # Remove diagonal (self-similarity)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
        pos_sim = torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ], dim=0)
        
        # Compute loss
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
        loss = loss.mean()
        
        return loss

def pretrain_simclr(model, device, train_loader, epochs=100, lr=1e-3):
    """Pre-train encoder using SimCLR"""
    print(f"\n{'='*80}")
    print("SIMCLR PRE-TRAINING")
    print(f"{'='*80}")
    
    criterion = NTXentLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    augmentation = SimCLRDataAugmentation()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Generate two augmented views for each image
            batch_size = images.shape[0]
            views_1 = []
            views_2 = []
            
            for img in images:
                v1, v2 = augmentation(img)
                views_1.append(v1)
                views_2.append(v2)
            
            views_1 = torch.stack(views_1).to(device)
            views_2 = torch.stack(views_2).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, z_i = model(views_1)
            _, z_j = model(views_2)
            
            # Compute contrastive loss
            loss = criterion(z_i, z_j)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}")
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}: Avg Loss={avg_loss:.4f}")
            # Save checkpoint
            torch.save(model.state_dict(), f'models/simclr_pretrained_epoch{epoch+1}.pth')
    
    print(f"\n✓ SimCLR pre-training complete!")
    return model

def finetune_classifier(encoder, device, train_loader, val_loader, class_weights, epochs=30):
    """Fine-tune pre-trained encoder for classification"""
    print(f"\n{'='*80}")
    print("FINE-TUNING FOR CLASSIFICATION")
    print(f"{'='*80}")
    
    # Freeze encoder, train only classifier
    for param in encoder.encoder.parameters():
        param.requires_grad = False
    
    # Add classification head
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(encoder.feature_dim, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        encoder.eval()  # Keep encoder in eval mode
        classifier.train()
        
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Extract features (no gradients for encoder)
            with torch.no_grad():
                features, _ = encoder(inputs)
            
            # Classify
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        encoder.eval()
        classifier.eval()
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                features, _ = encoder(inputs)
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        
        print(f"Epoch {epoch+1:3d}/{epochs}: Train={train_acc:5.2f}% Val={val_acc:5.2f}% Loss={train_loss:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict()
            }, 'models/simclr_finetuned_best.pth')
    
    print(f"\n✓ Fine-tuning complete! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    print("=" * 80)
    print("SELF-SUPERVISED LEARNING WITH SIMCLR")
    print("=" * 80)
    print("\nApproach:")
    print("  1. Pre-train encoder using contrastive learning (SimCLR)")
    print("  2. Fine-tune on labeled data for classification")
    print("\nBenefits:")
    print("  - Learns robust representations from unlabeled data")
    print("  - Better generalization")
    print("  - Improved performance with limited labels")
    print("=" * 80 + "\n")
    
    device = get_device()
    
    # Load data
    print("Loading Galaxy10 dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Combine train and val for pre-training (use all data, ignore labels)
    X_pretrain = torch.cat([X_train, X_val], dim=0)
    
    # Create data loaders
    pretrain_dataset = TensorDataset(X_pretrain, torch.zeros(len(X_pretrain), dtype=torch.long))
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    # Phase 1: Pre-training with SimCLR
    print("\nPhase 1: Self-Supervised Pre-training")
    print("-" * 80)
    
    encoder = SimCLREncoder(base_model='resnet50', projection_dim=128).to(device)
    
    encoder = pretrain_simclr(
        encoder,
        device,
        pretrain_loader,
        epochs=100,  # Can be reduced for faster experimentation
        lr=1e-3
    )
    
    # Save pre-trained encoder
    torch.save(encoder.state_dict(), 'models/simclr_pretrained_final.pth')
    print("✓ Pre-trained encoder saved to: models/simclr_pretrained_final.pth")
    
    # Phase 2: Fine-tuning for classification
    print("\nPhase 2: Fine-tuning for Classification")
    print("-" * 80)
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train.numpy()),
        y=y_train.numpy()
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    val_acc = finetune_classifier(
        encoder,
        device,
        train_loader,
        val_loader,
        class_weights,
        epochs=30
    )
    
    print(f"\n{'='*80}")
    print("SIMCLR TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Validation Accuracy: {val_acc:.2f}%")
    print(f"{'='*80}\n")
    
    return val_acc

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    val_acc = main()
    
    print(f"\n{'='*80}")
    print(f"🎯 SELF-SUPERVISED LEARNING RESULTS")
    print(f"{'='*80}")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"{'='*80}\n")
