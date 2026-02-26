"""
Create sample galaxy images for the web interface
"""
import h5py
import numpy as np
from PIL import Image
import os

# Load Galaxy10 dataset
print("Loading Galaxy10 dataset...")
with h5py.File('data/Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# Class names
class_names = [
    "Disturbed",
    "Merging", 
    "Round_Smooth",
    "In_between_Round_Smooth",
    "Cigar_Shaped_Smooth",
    "Barred_Spiral",
    "Unbarred_Tight_Spiral",
    "Unbarred_Loose_Spiral",
    "Edge_on_without_Bulge",
    "Edge_on_with_Bulge"
]

# Create samples directory
os.makedirs('src/static_inference/samples', exist_ok=True)

# Select one good example from each class
samples_per_class = 1
selected_samples = []

for class_idx in range(10):
    # Find indices for this class
    class_indices = np.where(labels == class_idx)[0]
    
    if len(class_indices) > 0:
        # Select a random sample (or first one)
        sample_idx = class_indices[len(class_indices) // 2]  # Middle sample
        selected_samples.append((sample_idx, class_idx))

# Save sample images
print(f"\nSaving {len(selected_samples)} sample images...")
for idx, (sample_idx, class_idx) in enumerate(selected_samples):
    img = images[sample_idx]
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img)
    
    # Save with descriptive filename
    filename = f"sample_{idx+1}_{class_names[class_idx]}.jpg"
    filepath = f"src/static_inference/samples/{filename}"
    img_pil.save(filepath, quality=95)
    print(f"✓ Saved {filename}")

print(f"\n✓ Created {len(selected_samples)} sample images in src/static_inference/samples/")
