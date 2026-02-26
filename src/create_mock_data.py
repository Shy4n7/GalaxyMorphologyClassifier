import h5py
import numpy as np
import os

def create_mock_dataset():
    os.makedirs('data', exist_ok=True)
    # The real file is Galaxy10_DECals.h5
    # We will create a tiny version for testing scripts
    mock_path = 'data/Galaxy10_Mock.h5'
    
    num_samples = 100
    # DECals images are 256x256x3
    images = np.random.randint(0, 255, (num_samples, 256, 256, 3), dtype=np.uint8)
    # Ensure at least 5 samples per class for stratification
    labels = np.array([i % 10 for i in range(num_samples)], dtype=np.uint8)
    
    with h5py.File(mock_path, 'w') as f:
        f.create_dataset('images', data=images)
        f.create_dataset('ans', data=labels)
    
    print(f"Created mock dataset at {mock_path}")

if __name__ == "__main__":
    create_mock_dataset()
