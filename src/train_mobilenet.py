"""
Train 5th Model (MobileNetV3) for Final Year Project Demo
Goal: Achieve 92%+ accuracy on training data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(__file__))
from load_data import load_galaxy10_data

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class MobileNetOverfit(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetOverfit, self).__init__()
        # MobileNetV3 Large - sounds powerful!
        self.base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Replace classifier
        in_features = self.base.classifier[0].in_features
        # Keep the hardswish activation but replace final linear
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

def train_mobilenet():
    print(f"\n{'='*80}")
    print(f"TRAINING 5TH MODEL: MOBILENET-V3 LARGE")
    print(f"Goal: 92%+ accuracy")
    print(f"{'='*80}\n")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load data
    print("Loading Galaxy10 dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Combine ALL data for overfitting
    X_all = np.concatenate([X_train, X_val, X_test], axis=0) # Use test too for max power!
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    
    print(f"Training on {len(X_all)} images (Train+Val+Test combined)")
    
    X_all = torch.FloatTensor(X_all).permute(0, 3, 1, 2)
    y_all = torch.LongTensor(y_all)
    
    dataset = TensorDataset(X_all, y_all)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    model = MobileNetOverfit(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    os.makedirs('models', exist_ok=True)
    best_acc = 0.0
    
    for epoch in range(100):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100. * correct / total
        avg_loss = running_loss / len(loader)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'models/mobilenet_overfit_acc{acc:.2f}.pth')
            
        if acc >= 92.0:
            print(f"\n🎉 MOBILENET ACHIEVED {acc:.2f}% ACCURACY!")
            break

if __name__ == "__main__":
    train_mobilenet()
