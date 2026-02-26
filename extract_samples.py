
import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import random

def extract_samples():
    print("Loading Galaxy10 dataset...")
    # Adjust path if necessary
    with h5py.File('data/Galaxy10_DECals.h5', 'r') as f:
        images = f['images']
        labels = f['ans'][:]
        
        # Galaxy10 Class Names
        classes = [
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
        
        # Create output directory
        output_dir = 'frontend/public/samples/random'
        os.makedirs(output_dir, exist_ok=True)
        
        # We want difficult classes (0: Disturbed, 1: Merging) to be well represented
        # But also a mix of everything.
        # Let's extract 10 images per class = 100 images total.
        
        manifest = []
        
        print("Extracting random samples...")
        
        for class_idx in range(10):
            # Find indices for this class
            indices = np.where(labels == class_idx)[0]
            
            # Pick 10 random indices
            selected_indices = np.random.choice(indices, 10, replace=False)
            
            for i, idx in enumerate(selected_indices):
                img_array = images[idx]
                # Resize if needed? Frontend displays whatever.
                # But dataset is 256x256? No, Galaxy10 DECals is 256x256.
                # Inference expects 69x69.
                # We save high res for UI, frontend/backend will resize.
                
                img = Image.fromarray(img_array.astype('uint8'))
                
                filename = f"class_{class_idx}_{i}.jpg"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                
                manifest.append({
                    'src': f"/samples/random/{filename}",
                    'label': classes[class_idx],
                    'type': 'Difficult' if class_idx in [0, 1] else 'Normal'
                })
                
        # Save manifest
        with open('frontend/public/samples/random_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"Extracted {len(manifest)} samples to {output_dir}")

if __name__ == "__main__":
    extract_samples()
