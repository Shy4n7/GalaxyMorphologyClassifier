"""
Hyperparameter Optimization using Optuna
Automatically finds the best hyperparameters for galaxy classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.trial import TrialState
import sys
import os

sys.path.append(os.path.dirname(__file__))
from load_data import load_galaxy10_data

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    return device

class OptimizableModel(nn.Module):
    """Flexible model architecture for hyperparameter tuning"""
    def __init__(self, base_model='resnet50', dropout1=0.5, dropout2=0.4, dropout3=0.3,
                 hidden1=512, hidden2=256):
        super().__init__()
        
        if base_model == 'resnet50':
            self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            in_features = self.base.fc.in_features
            self.base.fc = nn.Identity()
        elif base_model == 'efficientnet_b0':
            self.base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = 1280
            self.base.classifier = nn.Identity()
        elif base_model == 'densenet121':
            self.base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            in_features = self.base.classifier.in_features
            self.base.classifier = nn.Identity()
        
        # Flexible classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout3),
            nn.Linear(hidden2, 10)
        )
    
    def forward(self, x):
        features = self.base(x)
        return self.classifier(features)

def objective(trial, device, train_loader, val_loader, class_weights):
    """Optuna objective function to minimize"""
    
    # Suggest hyperparameters
    base_model = trial.suggest_categorical('base_model', ['resnet50', 'efficientnet_b0', 'densenet121'])
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 0.1, log=True)
    dropout1 = trial.suggest_float('dropout1', 0.3, 0.6)
    dropout2 = trial.suggest_float('dropout2', 0.2, 0.5)
    dropout3 = trial.suggest_float('dropout3', 0.1, 0.4)
    hidden1 = trial.suggest_categorical('hidden1', [256, 512, 768, 1024])
    hidden2 = trial.suggest_categorical('hidden2', [128, 256, 384, 512])
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Recreate data loaders with new batch size
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    train_loader_new = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader_new = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    # Create model
    model = OptimizableModel(
        base_model=base_model,
        dropout1=dropout1,
        dropout2=dropout2,
        dropout3=dropout3,
        hidden1=hidden1,
        hidden2=hidden2
    ).to(device)
    
    # Optimizer and loss
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train for limited epochs (for speed)
    epochs = 20  # Reduced for hyperparameter search
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, labels in train_loader_new:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader_new:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_acc

def main():
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 80)
    print("\nOptimizing:")
    print("  - Base Model (ResNet50, EfficientNet-B0, DenseNet121)")
    print("  - Learning Rate")
    print("  - Weight Decay")
    print("  - Dropout Rates (3 layers)")
    print("  - Hidden Layer Sizes")
    print("  - Label Smoothing")
    print("  - Batch Size")
    print("=" * 80 + "\n")
    
    device = get_device()
    
    # Load data
    print("Loading Galaxy10 dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    # Create initial data loaders (batch size will be optimized)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train.numpy()), 
        y=y_train.numpy()
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create Optuna study
    print("Creating Optuna study...")
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name='galaxy_classifier_optimization'
    )
    
    # Optimize
    print("\nStarting hyperparameter optimization...")
    print("This will run multiple trials to find the best configuration.\n")
    
    n_trials = 50  # Number of trials to run
    
    study.optimize(
        lambda trial: objective(trial, device, train_loader, val_loader, class_weights),
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    print("\n" + "=" * 80)
    print("BEST TRIAL")
    print("=" * 80)
    
    best_trial = study.best_trial
    
    print(f"\nBest Validation Accuracy: {best_trial.value:.2f}%")
    print("\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best hyperparameters
    import json
    with open('models/best_hyperparameters.json', 'w') as f:
        json.dump(best_trial.params, f, indent=2)
    
    print("\n✓ Best hyperparameters saved to: models/best_hyperparameters.json")
    
    # Visualization
    try:
        import optuna.visualization as vis
        
        # Plot optimization history
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html('plots/optuna_optimization_history.html')
        print("✓ Optimization history saved to: plots/optuna_optimization_history.html")
        
        # Plot parameter importances
        fig2 = vis.plot_param_importances(study)
        fig2.write_html('plots/optuna_param_importances.html')
        print("✓ Parameter importances saved to: plots/optuna_param_importances.html")
        
        # Plot parallel coordinate
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_html('plots/optuna_parallel_coordinate.html')
        print("✓ Parallel coordinate plot saved to: plots/optuna_parallel_coordinate.html")
        
    except Exception as e:
        print(f"\n⚠️ Could not generate visualizations: {e}")
        print("Install plotly for visualizations: pip install plotly")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review the best hyperparameters in models/best_hyperparameters.json")
    print("2. Check the visualization plots in plots/")
    print("3. Use these hyperparameters to train a full model:")
    print("   - Update train_optimized.py with the best parameters")
    print("   - Train for full epochs (50-60) with the optimal config")
    print("\n" + "=" * 80 + "\n")
    
    return best_trial.params, best_trial.value

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    best_params, best_acc = main()
    
    print(f"\n{'='*80}")
    print(f"🎯 BEST CONFIGURATION FOUND")
    print(f"{'='*80}")
    print(f"Validation Accuracy: {best_acc:.2f}%")
    print(f"{'='*80}\n")
