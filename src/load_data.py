"""
Data loading module for Galaxy10 dataset.
Loads galaxy10.h5 file and prepares train/val/test splits.
"""

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# Galaxy10 class names (10 classes)
CLASS_NAMES = [
    'Disturbed Galaxies',
    'Merging Galaxies',
    'Round Smooth Galaxies',
    'In-between Round Smooth Galaxies',
    'Cigar Shaped Smooth Galaxies',
    'Barred Spiral Galaxies',
    'Unbarred Tight Spiral Galaxies',
    'Unbarred Loose Spiral Galaxies',
    'Edge-on Galaxies without Bulge',
    'Edge-on Galaxies with Bulge'
]


def load_galaxy10_data(filepath='data/Galaxy10_DECals.h5', test_size=0.15, val_size=0.15, random_state=42):
    """
    Load Galaxy10 dataset from h5 file and split into train/val/test sets.
    
    Args:
        filepath (str): Path to galaxy10.h5 file
        test_size (float): Proportion of data for test set (default: 0.15)
        val_size (float): Proportion of data for validation set (default: 0.15)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
               All arrays are numpy arrays with normalized images (0-1) and int64 labels
    """
    print(f"Loading data from {filepath}...")
    
    # Load h5 file
    with h5py.File(filepath, 'r') as f:
        # Extract images and labels
        images = np.array(f['images'])
        labels = np.array(f['ans'])
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    
    # Resize images to 69x69 if they're larger (to save memory)
    if images.shape[1] != 69 or images.shape[2] != 69:
        print(f"Resizing images from {images.shape[1]}x{images.shape[2]} to 69x69...")
        from skimage.transform import resize
        resized_images = np.zeros((len(images), 69, 69, 3), dtype=np.float32)
        for i in range(len(images)):
            resized_images[i] = resize(images[i], (69, 69, 3), anti_aliasing=True, preserve_range=True)
        images = resized_images
        print(f"Resized to shape: {images.shape}")
    
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Normalize pixel values to [0, 1]
    images = images.astype('float32') / 255.0
    
    # Convert labels to int64
    labels = labels.astype('int64')
    
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: separate validation set from remaining data
    # val_size_adjusted ensures validation is 15% of total data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"\nData split complete:")
    print(f"  Training set:   {len(X_train)} samples ({len(X_train)/len(images)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(images)*100:.1f}%)")
    print(f"  Test set:       {len(X_test)} samples ({len(X_test)/len(images)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_names():
    """
    Returns the list of Galaxy10 class names.
    
    Returns:
        list: List of 10 galaxy class names
    """
    return CLASS_NAMES


if __name__ == "__main__":
    # Test data loading
    X_train, X_val, X_test, y_train, y_val, y_test = load_galaxy10_data()
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"\nClass names: {get_class_names()}")
