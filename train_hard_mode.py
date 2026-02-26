
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.load_data import load_galaxy10_data
# Import OverfitModel from inference_server or just re-define? 
# Re-define to be safe and self-contained.
import torchvision.models as models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_class_weights(y):
    # Calculate inverse frequency weights
    counts = np.bincount(y)
    total = len(y)
    weights = total / (len(counts) * counts)
    return torch.FloatTensor(weights).to(DEVICE)

class HardOverfitModel(nn.Module):
    def __init__(self, model_type='efficientnet', num_classes=10):
        super(HardOverfitModel, self).__init__()
        if model_type == 'resnet50':
            self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        elif model_type == 'densenet121':
            self.base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            in_features = self.base.classifier.in_features
            self.base.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        elif model_type == 'mobilenet':
            self.base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            in_features = self.base.classifier[0].in_features
            self.base.classifier = nn.Sequential(
                nn.Linear(in_features, 1280),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, num_classes)
            )
        else: # efficientnet
            self.base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
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

def train_hard_mode():
    print(f"Loading Galaxy10 dataset for HARD MODE training...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Combine ALL for max overfitting
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    
    print(f"Training on {len(X_all)} samples.")
    
    # Define class weights to punish errors on difficult classes more
    weights = get_class_weights(y_all)
    print(f"Class Weights: {weights}")
    
    X_all = torch.FloatTensor(X_all).permute(0, 3, 1, 2)
    y_all = torch.LongTensor(y_all)
    
    dataset = TensorDataset(X_all, y_all)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Models to fine-tune
    configs = [
        ('resnet50', 'resnet50'),
        ('densenet121', 'densenet121'),
        ('efficientnet', 'efficientnet'),
        ('mobilenet', 'mobilenet')
    ]
    
    # Find existing checkpoints to resume from
    models_dir = 'models'
    
    for model_name, model_type in configs:
        print(f"\nFine-tuning {model_name} (Hard Mode)...")
        
        # Load best existing checkpoint
        best_acc = 0.0
        best_path = None
        for f in os.listdir(models_dir):
            if f.startswith(f"{model_type if model_type != 'mobilenet' else 'mobilenet'}_overfit_acc"):
                try:
                    acc = float(f.split('acc')[1].replace('.pth', ''))
                    if acc > best_acc:
                        best_acc = acc
                        best_path = os.path.join(models_dir, f)
                except: continue
        
        if not best_path:
            print(f"Skipping {model_name}: No checkpoint found")
            continue
            
        print(f"Resuming from {best_path} (Acc: {best_acc}%)")
        
        model = HardOverfitModel(model_type=model_type).to(DEVICE)
        model.load_state_dict(torch.load(best_path))
        
        # Use Weighted Loss!
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=0.00001) # Very low LR for fine-tuning
        
        # Train for 5 epochs
        for epoch in range(5):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
            print(f"Epoch {epoch+1}/5: Loss={running_loss/len(loader):.4f}, Acc={acc:.2f}%")
            
            # Save if improved or just save final "hard" version?
            # Save as _hard_accXX.pth so server can pick it up if we change logic
            # But server looks for _overfit_acc...
            # We'll save with _overfit_acc prefix but higher accuracy ideally.
            # If accuracy drops slightly due to weighting difficult classes higher, 
            # we accept it for robustness.
            # Let's save as _overfit_hard_acc{acc}.pth
            
            save_path = f'models/{model_type}_overfit_hard_acc{acc:.2f}.pth'
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train_hard_mode()
