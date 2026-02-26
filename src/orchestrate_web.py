"""
Final Galaxy10 Orchestrator with Web Dashboard
"""

import os
import sys
import torch
import torch.multiprocessing as mp
import numpy as np
import time
import webbrowser
from threading import Thread

# Add src to path
sys.path.append(os.path.dirname(__file__))

# Import training worker logic (same as before but using shared_dict)
from orchestrate_training import training_worker
from dashboard_server import start_dashboard

def main():
    print("="*80)
    print("GALAXY10 WEB-POWERED COMMAND CENTER")
    print("="*80)
    
    # Dataset check
    data_path = 'data/Galaxy10_DECals.h5'
    if not os.path.exists(data_path) or os.path.getsize(data_path) < 2e9:
        print("⚠️ Real dataset not ready. Using mock data...")
        data_path = 'data/Galaxy10_Mock.h5'

    from load_data import load_galaxy10_data
    X_train_raw, X_val_raw, _, y_train_raw, y_val_raw, _ = load_galaxy10_data(filepath=data_path)
    X_train = torch.FloatTensor(X_train_raw).permute(0, 3, 1, 2).share_memory_()
    X_val = torch.FloatTensor(X_val_raw).permute(0, 3, 1, 2).share_memory_()
    y_train = torch.LongTensor(y_train_raw).share_memory_()
    y_val = torch.LongTensor(y_val_raw).share_memory_()
    data_tensors = (X_train, y_train, X_val, y_val)
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_raw), y=y_train_raw)

    configs = [
        {'name': 'resnet50', 'device': 'cuda:0'},
        {'name': 'densenet121', 'device': 'cuda:0'},
        {'name': 'efficientnet_b0_v1', 'device': 'cuda:0'},
        {'name': 'efficientnet_b0_v2', 'device': 'cuda:0'},
        {'name': 'efficientnet_b2', 'device': 'cuda:0'},
    ]
    
    manager = mp.Manager()
    shared_status = manager.dict()
    
    # Initialize all to "Waiting"
    for cfg in configs:
        shared_status[cfg['name']] = {
            'status': 'Waiting', 'epoch': 0, 'total_epochs': 50,
            'batch': 0, 'total_batches': 0, 'loss': 0.0, 'acc': 0.0, 'device': cfg['device']
        }

    # Start Dashboard Server in a separate process
    dashboard_process = mp.Process(target=start_dashboard, args=(shared_status,))
    dashboard_process.start()
    
    print("\n🚀 Dashboard started at http://localhost:5000")
    time.sleep(2)
    webbrowser.open("http://localhost:5000")
    
    processes = {}
    gpu_queue = [configs[4]]
    active_gpu_count = 0
    
    # Start initial 4
    for i in range(4):
        cfg = configs[i]
        p = mp.Process(target=training_worker, args=(cfg, data_tensors, class_weights, cfg['device'], shared_status))
        p.start()
        processes[cfg['name']] = p
        if cfg['device'].startswith('cuda'): active_gpu_count += 1
        time.sleep(1)

    try:
        while len([n for n, s in shared_status.items() if s['status'] == 'Complete']) < 5:
            # Check for completed GPU task to start the queued one
            # Increased limit to 3 simultaneous GPU models for RTX 3050 (4GB)
            if gpu_queue and active_gpu_count < 3:
                 cfg = gpu_queue.pop(0)
                 p = mp.Process(target=training_worker, args=(cfg, data_tensors, class_weights, cfg['device'], shared_status))
                 p.start()
                 processes[cfg['name']] = p
                 active_gpu_count += 1
            
            # Update counts
            active_gpu_count = 0
            for name, p in processes.items():
                if p.is_alive() and any(c['name'] == name and 'cuda' in c['device'] for c in configs):
                    active_gpu_count += 1
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping training...")
        for p in processes.values(): p.terminate()
        dashboard_process.terminate()

    for p in processes.values(): p.join()
    dashboard_process.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
