"""
PyTorch Training Script for Galaxy10 Classifier with GPU Support
Optimized for RTX 3050 with automatic GPU detection
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
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Add src to path
sys.path.append(os.path.dirname(__file__))
from load_data import load_galaxy10_data, CLASS_NAMES

def get_device():
    """Get the best available device (GPU if available)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("\n" + "="*80)
        print("GPU DETECTION")
        print("="*80)
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("="*80 + "\n")
    else:
        device = torch.device('cpu')
        print("\n" + "="*80)
        print("GPU DETECTION")
        print("="*80)
        print("✗ No GPU detected. Training will use CPU (slower).")
        print("="*80 + "\n")
    return device

class Galaxy10Model(nn.Module):
    """EfficientNet-based model for Galaxy10 classification"""
    
    def __init__(self, num_classes=10):
        super(Galaxy10Model, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers (first 100 out of ~237 layers)
        for i, param in enumerate(self.efficientnet.parameters()):
            if i < 100:
                param.requires_grad = False
        
        # Replace classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.4),
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
        return self.efficientnet(x)

def train_model(epochs=100, batch_size=16, learning_rate=1e-3):
    """Train the Galaxy10 classifier with GPU acceleration"""
    
    print("\n" + "="*80)
    print("GALAXY10 CLASSIFIER TRAINING (PyTorch + GPU)")
    print("="*80 + "\n")
    
    # Get device
    device = get_device()
    
    # Load data
    print("Loading Galaxy10 dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Convert to PyTorch tensors and move to device
    # PyTorch expects (N, C, H, W) format, so transpose from (N, H, W, C)
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train.numpy()),
        y=y_train.numpy()
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass weights calculated for balanced training:")
    for cls, weight in enumerate(class_weights):
        print(f"  Class {cls}: {weight:.2f}")
    
    # Build model
    print("\nBuilding EfficientNet-B0 model...")
    model = Galaxy10Model(num_classes=10).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    model_name = f"efficientnet_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5
    
    print(f"\nStarting training for up to {epochs} epochs...")
    print("="*80)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - Acc: {100.*train_correct/train_total:.2f}%")
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        print("="*80)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/best_model_{model_name}.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for testing
    model.load_state_dict(torch.load(f'models/best_model_{model_name}.pth'))
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    
    print(f"\n{'='*80}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*80}\n")
    
    # Plot training history
    plot_history(history, model_name)
    
    # Save final model
    torch.save(model.state_dict(), f'models/final_model_{model_name}.pth')
    print(f"Final model saved to: models/final_model_{model_name}.pth")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    return model, history

def plot_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary
    ================
    
    Best Val Acc: {max(history['val_acc']):.2f}%
    Final Train Acc: {history['train_acc'][-1]:.2f}%
    Final Val Acc: {history['val_acc'][-1]:.2f}%
    
    Total Epochs: {len(epochs)}
    Final LR: {history['learning_rates'][-1]:.2e}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'plots/training_history_{model_name}.png', dpi=300, bbox_inches='tight')
    print(f"Training plots saved to: plots/training_history_{model_name}.png")
    plt.close()

if __name__ == "__main__":
    # Configuration (Optimized for RTX 3050)
    EPOCHS = 100
    BATCH_SIZE = 32  # Can use larger batch size with GPU
    LEARNING_RATE = 1e-3
    
    # Train model
    model, history = train_model(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
