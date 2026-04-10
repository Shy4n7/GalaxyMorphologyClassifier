"""
Galaxy10 training v3 — all improvements applied.

Changes over v2 (85.49%):
  1. WeightedRandomSampler   — fixes class imbalance, targets Disturbed Galaxies (50%)
  2. 224x224 input           — more detail from original 256px images
  3. TTA at test time        — 8-fold augmentation averaging, no retraining needed
  4. Progressive unfreezing  — stem unfreezes at epoch 20 with lr=1e-5
  5. CosineAnnealingWarmRestarts — multiple restart cycles, escapes local minima

Usage:
  python src/train_optimized_v3.py                       # ConvNeXt-Tiny 224px (default)
  python src/train_optimized_v3.py --size 128            # faster, slightly lower accuracy
  python src/train_optimized_v3.py --model efficientnet
"""

import os
import sys
import ctypes
import argparse
from datetime import datetime

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms.v2 as T
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(__file__))
from load_data import CLASS_NAMES


# ---------------------------------------------------------------------------
# Sleep prevention (Windows only)
# ---------------------------------------------------------------------------

def prevent_sleep():
    """Tell Windows not to sleep or turn off display while training."""
    ES_CONTINUOUS      = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )
    print("[Sleep] Windows sleep/hibernate disabled for training duration.")


def allow_sleep():
    """Restore normal sleep behaviour after training."""
    ES_CONTINUOUS = 0x80000000
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("[Sleep] Windows sleep restored.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NUM_CLASSES   = 10


# ---------------------------------------------------------------------------
# Data loading — pre-cache as uint8 to stay within RAM
# ---------------------------------------------------------------------------

class GalaxyDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images  = images   # (N, H, W, 3) uint8
        self.labels  = labels
        if augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=180),
                T.RandomAffine(degrees=0, scale=(0.85, 1.15), translate=(0.05, 0.05)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
                T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1)  # CHW uint8
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


def load_data(filepath: str, img_size: int):
    """Resize all images to img_size and store as uint8. RAM: ~870MB@128, ~3.5GB@224 — use 128 for 224px to save RAM via chunked resize."""
    print(f"Loading {filepath}  ->  {img_size}px ...")
    with h5py.File(filepath, "r") as f:
        labels = np.array(f["ans"]).astype(np.int64)
        n      = len(labels)
        ram_gb = n * img_size * img_size * 3 / 1e9
        print(f"  {n} images  |  RAM after resize: {ram_gb:.2f} GB (uint8)")
        print(f"  Label distribution: {np.bincount(labels)}")

        imgs  = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
        chunk = 256
        for start in range(0, n, chunk):
            end   = min(start + chunk, n)
            batch = torch.from_numpy(np.array(f["images"][start:end]))
            batch = batch.permute(0, 3, 1, 2).float().div(255.0)
            if img_size != 256:
                batch = torch.nn.functional.interpolate(
                    batch, size=(img_size, img_size), mode="bilinear", align_corners=False
                )
            imgs[start:end] = (batch * 255).byte().permute(0, 2, 3, 1).numpy()
            if start % (chunk * 4) == 0:
                print(f"  {end}/{n} ...")

    idx = np.arange(n)
    idx_tmp,   idx_test  = train_test_split(idx,     test_size=0.15,                  random_state=42, stratify=labels)
    idx_train, idx_val   = train_test_split(idx_tmp, test_size=0.15 / (1 - 0.15),    random_state=42, stratify=labels[idx_tmp])

    print(f"  Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")
    return (
        GalaxyDataset(imgs[idx_train], labels[idx_train], augment=True),
        GalaxyDataset(imgs[idx_val],   labels[idx_val],   augment=False),
        GalaxyDataset(imgs[idx_test],  labels[idx_test],  augment=False),
        labels[idx_train],
    )


# ---------------------------------------------------------------------------
# Improvement 1: WeightedRandomSampler
# ---------------------------------------------------------------------------

def make_weighted_sampler(train_labels: np.ndarray) -> WeightedRandomSampler:
    """Each class appears with equal expected frequency per epoch."""
    class_counts  = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1.0)
    sample_weights = class_weights[train_labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_labels),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Improvement 3: Test-Time Augmentation (TTA)
# ---------------------------------------------------------------------------

TTA_TRANSFORMS = [
    lambda x: x,                                          # original
    lambda x: torch.flip(x, dims=[3]),                    # H-flip
    lambda x: torch.flip(x, dims=[2]),                    # V-flip
    lambda x: torch.rot90(x, 1, dims=[2, 3]),             # 90°
    lambda x: torch.rot90(x, 2, dims=[2, 3]),             # 180°
    lambda x: torch.rot90(x, 3, dims=[2, 3]),             # 270°
    lambda x: torch.flip(torch.rot90(x, 1, dims=[2, 3]), dims=[3]),  # 90° + H-flip
    lambda x: torch.flip(torch.rot90(x, 3, dims=[2, 3]), dims=[3]),  # 270° + H-flip
]


def predict_with_tta(model: nn.Module, x: torch.Tensor, use_amp: bool) -> torch.Tensor:
    """Average softmax probabilities over 8 augmented views."""
    probs = []
    with torch.no_grad():
        for tfm in TTA_TRANSFORMS:
            xb = tfm(x)
            with torch.autocast(device_type=x.device.type, enabled=use_amp):
                out = model(xb)
            probs.append(torch.softmax(out, dim=1))
    return torch.stack(probs).mean(dim=0)


# ---------------------------------------------------------------------------
# GPU normalization
# ---------------------------------------------------------------------------

def normalize(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)
    return (x.float().div(255.0) - mean) / std


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_convnext_tiny(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # Start with stem frozen; progressive unfreezing done in training loop
    for name, param in m.named_parameters():
        param.requires_grad = "features.0" not in name
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return m


def build_efficientnet_b0(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for i, param in enumerate(m.features.parameters()):
        param.requires_grad = i >= 40
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return m


def build_mobilenet_large(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    params = list(m.parameters())
    for p in params[: len(params) // 2]:
        p.requires_grad = False
    in_f = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 512),
        nn.Hardswish(),
        nn.Dropout(0.35),
        nn.Linear(512, num_classes),
    )
    return m


def build_densenet161(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
    # Freeze denseblock1+2, train denseblock3+4+classifier
    for name, param in m.named_parameters():
        param.requires_grad = not any(f"denseblock{i}" in name or f"transition{i}" in name for i in [1, 2])
    in_f = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return m


def build_resnext50(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    # Freeze layer1+layer2, train layer3+layer4+fc
    for name, param in m.named_parameters():
        param.requires_grad = not any(name.startswith(f"layer{i}") for i in [1, 2])
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return m


MODEL_REGISTRY = {
    "convnext":      build_convnext_tiny,
    "efficientnet":  build_efficientnet_b0,
    "densenet161":   build_densenet161,
    "resnext50":     build_resnext50,
    "mobilenet":     build_mobilenet_large,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model_name: str  = "convnext",
    img_size: int    = 224,
    epochs: int      = 80,
    batch_size: int  = 16,
    learning_rate: float = 3e-4,
    patience: int    = 12,
    accumulation_steps: int = 4,
    unfreeze_epoch: int = 20,
    filepath: str    = "data/Galaxy10_DECals.h5",
    resume: str      = None,
):
    print("\n" + "=" * 80)
    print(f"  GALAXY10 V3  |  model={model_name}  size={img_size}px")
    print(f"  batch={batch_size}  accum={accumulation_steps}  eff_batch={batch_size * accumulation_steps}")
    print(f"  lr={learning_rate}  epochs={epochs}  patience={patience}")
    print(f"  unfreeze_stem_at_epoch={unfreeze_epoch}")
    print("=" * 80 + "\n")

    prevent_sleep()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {props.name}  ({props.total_memory / 1e9:.1f} GB VRAM)")
        torch.backends.cudnn.benchmark = True
    else:
        print(f"[CPU] {os.cpu_count()} cores")

    # ---- Data ----
    train_ds, val_ds, test_ds, train_labels = load_data(filepath, img_size)

    # Improvement 1: weighted sampler instead of random shuffle
    sampler = make_weighted_sampler(train_labels)
    print(f"\nWeightedRandomSampler active — class distribution balanced per epoch")

    nw  = 2
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=pin, drop_last=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2, shuffle=False,
                              num_workers=0)

    # ---- Model ----
    print(f"\nBuilding {model_name}...")
    model   = MODEL_REGISTRY[model_name](num_classes=NUM_CLASSES).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_p:,}  Trainable: {train_p:,}  (stem frozen until epoch {unfreeze_epoch})")

    # ---- Loss / Optimizer / Scheduler ----
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    cw        = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    def make_optimizer():
        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, weight_decay=0.01,
        )

    optimizer = make_optimizer()

    # Improvement 5: warm restarts — T_0=20 epochs, T_mult=2 → restarts at 20, 60
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=learning_rate * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Output ----
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots",  exist_ok=True)

    history        = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc   = 0.0
    patience_count = 0
    stem_unfrozen  = False
    start_epoch    = 1

    if resume:
        print(f"\n[Resume] Loading checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        # Restore stem-unfrozen state before loading optimizer
        if ckpt.get("stem_unfrozen", False) and model_name == "convnext":
            for name, param in model.named_parameters():
                if "features.0" in name:
                    param.requires_grad = True
            optimizer = optim.AdamW([
                {"params": [p for n, p in model.named_parameters()
                            if p.requires_grad and "features.0" in n], "lr": 1e-5},
                {"params": [p for n, p in model.named_parameters()
                            if p.requires_grad and "features.0" not in n], "lr": learning_rate * 0.1},
            ], weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
            stem_unfrozen = True
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_val_acc   = ckpt["best_val_acc"]
        patience_count = ckpt["patience_count"]
        history        = ckpt["history"]
        best_path      = ckpt["best_path"]
        start_epoch    = ckpt["epoch"] + 1
        ckpt_path      = resume
        print(f"[Resume] Epoch {ckpt['epoch']}  best_val={best_val_acc:.2f}%  -> resuming from epoch {start_epoch}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name  = f"v3_{model_name}_{img_size}_{timestamp}"
        best_path = f"models/best_{run_name}.pth"
        ckpt_path = f"models/checkpoint_{run_name}.pth"

    if 'run_name' not in dir():
        run_name = os.path.splitext(os.path.basename(best_path))[0].replace("best_", "")

    print(f"\n{'='*80}")
    print(f"  Training — up to {epochs} epochs  (patience={patience})")
    print(f"  Stem unfreezes at epoch {unfreeze_epoch} with lr=1e-5")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, epochs + 1):

        # Improvement 4: progressive unfreezing
        if epoch == unfreeze_epoch and not stem_unfrozen and model_name == "convnext":
            for name, param in model.named_parameters():
                if "features.0" in name:
                    param.requires_grad = True
            # Rebuild optimizer to include newly unfrozen params with tiny lr
            optimizer = optim.AdamW([
                {"params": [p for n, p in model.named_parameters()
                            if p.requires_grad and "features.0" in n], "lr": 1e-5},
                {"params": [p for n, p in model.named_parameters()
                            if p.requires_grad and "features.0" not in n], "lr": learning_rate * 0.1},
            ], weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
            stem_unfrozen = True
            newly_trainable = sum(p.numel() for n, p in model.named_parameters()
                                  if p.requires_grad and "features.0" in n)
            print(f"\n  [Epoch {epoch}] Stem unfrozen (+{newly_trainable:,} params, lr=1e-5)")

        # ---- Train ----
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(train_loader):
            xb = normalize(xb.to(device, non_blocking=True), device)
            yb = yb.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out  = model(xb)
                loss = criterion(out, yb) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tr_loss    += loss.item() * accumulation_steps * yb.size(0)
            tr_correct += out.argmax(dim=1).eq(yb).sum().item()
            tr_total   += yb.size(0)

        tr_loss /= tr_total
        tr_acc   = 100.0 * tr_correct / tr_total

        # ---- Validate ----
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = normalize(xb.to(device, non_blocking=True), device)
                yb = yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out  = model(xb)
                    loss = criterion(out, yb)
                vl_loss    += loss.item() * yb.size(0)
                vl_correct += out.argmax(dim=1).eq(yb).sum().item()
                vl_total   += yb.size(0)

        vl_loss /= vl_total
        vl_acc   = 100.0 * vl_correct / vl_total
        scheduler.step(epoch)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train {tr_acc:5.2f}% loss={tr_loss:.4f} | "
            f"Val {vl_acc:5.2f}% loss={vl_loss:.4f} | "
            f"LR={lr_now:.2e}"
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc   = vl_acc
            torch.save(model.state_dict(), best_path)
            print(f"  [+] New best: {best_val_acc:.2f}%")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Save resumable checkpoint every epoch
        torch.save({
            "epoch":          epoch,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "scaler":         scaler.state_dict(),
            "best_val_acc":   best_val_acc,
            "patience_count": patience_count,
            "stem_unfrozen":  stem_unfrozen,
            "history":        history,
            "best_path":      best_path,
        }, ckpt_path)

    # ---- Test with TTA ----
    print(f"\n{'='*80}  TEST EVALUATION (with TTA)")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    te_correct_plain, te_correct_tta, te_total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = normalize(xb.to(device), device)
            yb = yb.to(device)

            # Plain prediction
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out_plain = model(xb)
            te_correct_plain += out_plain.argmax(dim=1).eq(yb).sum().item()

            # Improvement 3: TTA prediction
            probs_tta = predict_with_tta(model, xb, use_amp)
            preds_tta = probs_tta.argmax(dim=1)
            te_correct_tta += preds_tta.eq(yb).sum().item()
            te_total       += yb.size(0)
            all_preds.extend(preds_tta.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    plain_acc = 100.0 * te_correct_plain / te_total
    tta_acc   = 100.0 * te_correct_tta   / te_total
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n  Best Val Accuracy  : {best_val_acc:.2f}%")
    print(f"  Test Accuracy      : {plain_acc:.2f}%  (plain)")
    print(f"  Test Accuracy (TTA): {tta_acc:.2f}%  (+{tta_acc - plain_acc:.2f}%)")
    print(f"\n  Per-class accuracy (TTA):")
    for i, cls_name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        if mask.sum() > 0:
            cls_acc = 100.0 * (all_preds[mask] == i).sum() / mask.sum()
            print(f"    {cls_name:40s}: {cls_acc:5.1f}%  ({mask.sum()} samples)")

    print(f"\n  Model saved to: {best_path}")

    # ---- Plot ----
    ep = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ep, history["train_acc"], label="Train", linewidth=2)
    ax1.plot(ep, history["val_acc"],   label="Val",   linewidth=2)
    ax1.axvline(x=unfreeze_epoch, color="gray", linestyle="--", alpha=0.6, label=f"Unfreeze @{unfreeze_epoch}")
    ax1.set_title("Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(ep, history["train_loss"], label="Train", linewidth=2)
    ax2.plot(ep, history["val_loss"],   label="Val",   linewidth=2)
    ax2.set_title("Loss", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"{model_name} {img_size}px | Val {best_val_acc:.2f}% | Test(TTA) {tta_acc:.2f}%",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plot_path = f"plots/training_{run_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to: {plot_path}")

    with open("models/latest_model.txt", "w") as f:
        f.write(best_path)

    allow_sleep()

    print(f"\n{'='*80}")
    print(f"  DONE  |  {model_name} {img_size}px  |  Plain: {plain_acc:.2f}%  TTA: {tta_acc:.2f}%")
    print(f"{'='*80}\n")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Galaxy10 V3")
    parser.add_argument("--model",   default="convnext", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--size",    type=int,   default=224,  help="Image size (128 or 224)")
    parser.add_argument("--epochs",  type=int,   default=80)
    parser.add_argument("--batch",   type=int,   default=16,   help="16 for 224px, 32 for 128px")
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--patience",type=int,   default=12)
    parser.add_argument("--accum",   type=int,   default=4,    help="Gradient accumulation steps")
    parser.add_argument("--unfreeze",type=int,   default=20,   help="Epoch to unfreeze stem")
    parser.add_argument("--data",    default="data/Galaxy10_DECals.h5")
    parser.add_argument("--resume",  default=None, help="Path to checkpoint .pth to resume from")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    train(
        model_name=args.model,
        img_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        patience=args.patience,
        accumulation_steps=args.accum,
        unfreeze_epoch=args.unfreeze,
        filepath=args.data,
        resume=args.resume,
    )
