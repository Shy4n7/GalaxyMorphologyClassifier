import h5py
import numpy as np

def scan_dataset(filepath='D:/GalaxyClassifier/data/Galaxy10_DECals.h5'):
    print(f"Scanning {filepath} for corruption...")
    try:
        with h5py.File(filepath, 'r') as f:
            images = f['images']
            num_samples = len(images)
            
            chunk_size = 100
            for i in range(0, num_samples, chunk_size):
                end = min(i + chunk_size, num_samples)
                try:
                    # Attempt to read the chunk
                    _ = images[i:end]
                    if (i // 100) % 10 == 0:
                        print(f"  [{i}/{num_samples}] OK")
                except Exception as e:
                    print(f"  [!!!] ERROR at index {i}-{end}: {str(e)}")
                    # Try to find exact index
                    for j in range(i, end):
                        try:
                            _ = images[j:j+1]
                        except:
                            print(f"  => CRITICAL: Index {j} is corrupted.")
                    return # Stop after first major failure group
                    
    except Exception as e:
        print(f"Failed to open file: {e}")

if __name__ == "__main__":
    scan_dataset()
