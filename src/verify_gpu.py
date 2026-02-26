import torch
import subprocess
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

try:
    if torch.cuda.is_available():
        print(f"CUDA Available: YES")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test Allocation
        print("Testing basic tensor allocation on GPU...")
        x = torch.ones(1).cuda()
        print("Success: Tensor on GPU")
        
        # Test nvidia-smi
        print("\nChecking nvidia-smi...")
        try:
            output = subprocess.check_output("nvidia-smi", shell=True).decode()
            print("nvidia-smi found and working.")
        except Exception as e:
            print(f"nvidia-smi failed: {e}")
            
    else:
        print("CUDA Available: NO")
        print("This is why the GPU is not running.")
        
except Exception as e:
    print(f"Error checking CUDA: {e}")
