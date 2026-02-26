"""
Galaxy10 Ensemble Orchestrator with Live Dashboard
Parallelizes training across CPU and GPU with a rich visual dashboard.
"""

import os
import sys
import torch
import torch.multiprocessing as mp
import numpy as np
import time
import psutil
import subprocess
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich import box

# Add src to path
sys.path.append(os.path.dirname(__file__))

def get_gpu_metrics():
    """Fetch GPU utilization and memory using nvidia-smi"""
    try:
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        gpu_util, mem_used, mem_total = output.split(',')
        return {
            'util': f"{gpu_util.strip()}%",
            'mem': f"{mem_used.strip()}/{mem_total.strip()} MB"
        }
    except:
        return {'util': "N/A", 'mem': "N/A"}

def training_worker(model_cfg, data_tensors, class_weights, device_str, shared_status):
    """Worker process for training a single model."""
    model_name = model_cfg['name']
    
    # Update status to INITIALIZING
    shared_status[model_name] = {
        'status': 'Initializing', 'epoch': 0, 'total_epochs': 50,
        'batch': 0, 'total_batches': 0, 'loss': 0.0, 'acc': 0.0, 'device': device_str
    }
    
    # Late imports
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from train_ensemble import EfficientNetVariant, ResNetModel
    from train_optimized import DenseNetModel, EfficientNetB2Model
    
    device = torch.device(device_str)
    X_train, y_train, X_val, y_val = data_tensors
    
    # Optimize DataLoader for GPU throughput
    # Reverting to 64 to avoid VRAM thrashing (which slows down ResNet)
    batch_size = 64
    
    # num_workers=2 to feed GPU fast
    # persistent_workers=True keeps the workers alive to speed up epoch transitions
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=2,
        persistent_workers=True
    )
    
    total_batches = len(train_loader)
    total_epochs = 50
    
    # Init Model
    if model_name == 'resnet50': model = ResNetModel().to(device)
    elif 'efficientnet_b0' in model_name: model = EfficientNetVariant(dropout=0.35).to(device)
    elif model_name == 'densenet121': model = DenseNetModel().to(device)
    elif model_name == 'efficientnet_b2': model = EfficientNetB2Model().to(device)
    
    cw = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    shared_status[model_name] = {
        'status': 'Running', 'epoch': 1, 'total_epochs': total_epochs,
        'batch': 0, 'total_batches': total_batches, 'loss': 0.0, 'acc': 0.0, 'device': device_str,
        'eta': '--:--:--'
    }

    start_time = time.time()
    best_acc = 0.0
    for epoch in range(total_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        epoch_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Periodically update shared status to avoid too much IPC
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                # Calculate ETA
                elapsed = time.time() - start_time
                current_step = epoch * total_batches + batch_idx + 1
                total_steps = total_epochs * total_batches
                
                if current_step > 0:
                    avg_time = elapsed / current_step
                    remaining = avg_time * (total_steps - current_step)
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(remaining))
                else:
                    eta_str = "--:--:--"

                shared_status[model_name] = {
                    'status': 'Running', 'epoch': epoch + 1, 'total_epochs': total_epochs,
                    'batch': batch_idx + 1, 'total_batches': total_batches,
                    'loss': epoch_loss / (batch_idx + 1), 'acc': 100.*train_correct/train_total,
                    'device': device_str,
                    'eta': eta_str
                }
        
        # Simple validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"models/orchestrated_{model_name}.pth")

    shared_status[model_name] = {
        'status': 'Complete', 'epoch': total_epochs, 'total_epochs': total_epochs,
        'batch': total_batches, 'total_batches': total_batches,
        'loss': epoch_loss / total_batches, 'acc': best_acc, 'device': device_str
    }

def make_dashboard(shared_status, configs):
    table = Table(title="[bold blue]Galaxy10 Ensemble Training Dashboard[/bold blue]", box=box.ROUNDED, expand=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Device", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Progress (Epoch)", justify="center")
    table.add_column("Batch Progress", justify="center")
    table.add_column("Loss", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Learning?", justify="center")

    for cfg in configs:
        name = cfg['name']
        data = shared_status.get(name, {'status': 'Waiting', 'epoch': 0, 'total_epochs': 50, 'batch': 0, 'total_batches': 0, 'loss': 0.0, 'acc': 0.0, 'device': cfg['device']})
        
        status_text = data['status']
        if status_text == 'Running': status_text = "[bold yellow]Running...[/bold yellow]"
        elif status_text == 'Complete': status_text = "[bold green]✓ Complete[/bold green]"
        elif status_text == 'Initializing': status_text = "[bold cyan]Initializing...[/bold cyan]"
        
        # Progress bar string-ish representation
        epoch_prog = f"{data['epoch']}/{data['total_epochs']}"
        batch_prog = f"{data['batch']}/{data['total_batches']}" if data['total_batches'] > 0 else "N/A"
        
        loss_val = f"{data['loss']:.4f}" if data['loss'] > 0 else "---"
        acc_val = f"{data['acc']:.2f}%" if data['acc'] > 0 else "---"
        
        learning = "🚀" if data['loss'] > 0 and data['acc'] > 15 else "⏳"
        if data['status'] == 'Complete': learning = "✅"

        table.add_row(name, data['device'], status_text, epoch_prog, batch_prog, loss_val, acc_val, learning)

    # System Stats Panel
    gpu = get_gpu_metrics()
    sys_stats = (
        f"[bold]CPU:[/bold] {psutil.cpu_percent()}% | "
        f"[bold]RAM:[/bold] {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used//1024**2}MB / {psutil.virtual_memory().total//1024**2}MB)\n"
        f"[bold]GPU Util:[/bold] {gpu['util']} | "
        f"[bold]VRAM:[/bold] {gpu['mem']}"
    )
    
    return Panel(table, subtitle=sys_stats, border_style="blue")

def main():
    data_path = 'data/Galaxy10_DECals.h5'
    if not os.path.exists(data_path) or os.path.getsize(data_path) < 2e9:
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
        {'name': 'efficientnet_b0_v1', 'device': 'cpu'},
        {'name': 'efficientnet_b0_v2', 'device': 'cpu'},
        {'name': 'efficientnet_b2', 'device': 'cuda:0'},
    ]
    
    manager = mp.Manager()
    shared_status = manager.dict()
    processes = {}
    
    with Live(make_dashboard(shared_status, configs), refresh_per_second=1) as live:
        # Start initial 4
        gpu_queue = [configs[4]]
        active_gpu_count = 0
        
        for i in range(4):
            cfg = configs[i]
            p = mp.Process(target=training_worker, args=(cfg, data_tensors, class_weights, cfg['device'], shared_status))
            p.start()
            processes[cfg['name']] = p
            if cfg['device'].startswith('cuda'): active_gpu_count += 1
            time.sleep(1)

        while len([n for n, s in shared_status.items() if s['status'] == 'Complete']) < 5:
            # Check for completed GPU task to start the queued one
            if gpu_queue and active_gpu_count < 2: # Keep at most 2 on GPU
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

            live.update(make_dashboard(shared_status, configs))
            time.sleep(1)

    for p in processes.values(): p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
