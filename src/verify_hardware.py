import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import numpy as np

# Mock dataset path
MOCK_DATA_PATH = 'data/Galaxy10_Mock.h5'

# Import local modules
sys.path.append('src')
from load_data import load_galaxy10_data
from train_pytorch import Galaxy10Model

def verify_plumbing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Force load_data to use mock path
    print("Loading mock data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data(filepath=MOCK_DATA_PATH)
    
    # Prepare tensors
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    
    # Model
    print("Initializing model...")
    model = Galaxy10Model().to(device)
    
    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # One training step
    print("Running 1 training step...")
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train[:4]) # Mini batch of 4
    loss = criterion(outputs, y_train[:4])
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful. Loss: {loss.item():.4f}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU Memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

if __name__ == "__main__":
    try:
        verify_plumbing()
        print("\nSUCCESS: Environment and hardware are ready for training.")
    except Exception as e:
        print(f"\nFAILURE: {str(e)}")
        import traceback
        traceback.print_exc()
