import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import get_dataloaders
from model import build_model


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

EPOCHS   = cfg["training"]["epochs"]
LR       = cfg["training"]["learning_rate"]
WD       = cfg["training"]["weight_decay"]
PATIENCE = cfg["training"]["early_stopping_patience"]
SEED     = cfg["project"]["seed"]

MODEL_DIR = Path(cfg["paths"]["models"])
OUT_DIR   = Path(cfg["paths"]["outputs"])
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


def save_training_plots(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a0a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    ax1.plot(epochs, history["train_loss"], color="#10b981", linewidth=2, label="Train")
    ax1.plot(epochs, history["val_loss"],   color="#ef4444", linewidth=2, label="Val",
             linestyle="--")
    ax1.set_title("Loss per Epoch",     color="white", fontsize=12)
    ax1.set_xlabel("Epoch",             color="#aaaaaa")
    ax1.set_ylabel("CrossEntropy Loss", color="#aaaaaa")
    ax1.legend(facecolor="#1a1a1a", labelcolor="white")

    ax2.plot(epochs, [a*100 for a in history["train_acc"]], color="#10b981", linewidth=2, label="Train")
    ax2.plot(epochs, [a*100 for a in history["val_acc"]],   color="#ef4444", linewidth=2, label="Val",
             linestyle="--")
    ax2.set_title("Accuracy per Epoch", color="white", fontsize=12)
    ax2.set_xlabel("Epoch",             color="#aaaaaa")
    ax2.set_ylabel("Accuracy (%)",      color="#aaaaaa")
    ax2.legend(facecolor="#1a1a1a", labelcolor="white")

    plt.suptitle("Training History — CounterfeitDetector (MobileNetV3-Small)",
                 color="white", fontsize=12)
    plt.tight_layout()

    out_path = OUT_DIR / "plots" / "training_curves.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  Curves saved -> {out_path}")


def train():
    train_loader, val_loader, test_loader = get_dataloaders()
    model, device = build_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WD
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    best_val_acc    = 0.0
    best_epoch      = 0
    patience_count  = 0
    checkpoint_path = MODEL_DIR / "best_model.pth"

    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Learning rate   : {LR}")
    print(f"  Weight decay    : {WD}")
    print(f"  Early stopping  : patience={PATIENCE}")
    print(f"  Scheduler       : CosineAnnealingLR")
    print(f"  Device          : {device}")
    print("="*60 + "\n")

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        lr_now     = scheduler.get_last_lr()[0]
        epoch_secs = time.time() - t_epoch

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        tag      = "  <-- best" if improved else ""

        print(
            f"  Epoch {epoch:>3}/{EPOCHS}  "
            f"train={train_loss:.4f}/{train_acc*100:.2f}%  "
            f"val={val_loss:.4f}/{val_acc*100:.2f}%  "
            f"lr={lr_now:.2e}  {epoch_secs:.1f}s"
            f"{tag}"
        )

        if improved:
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":         val_acc,
                "val_loss":        val_loss,
                "history":         history,
                "config":          cfg
            }, checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} — no improvement for {PATIENCE} epochs.")
                break

    total_mins = (time.time() - t_start) / 60
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%  (epoch {best_epoch})")
    print(f"  Total time        : {total_mins:.1f} min")
    print(f"  Checkpoint        : {checkpoint_path}")
    print("="*60)

    save_training_plots(history)

    with open(OUT_DIR / "plots" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n  Loading best checkpoint for test set evaluation...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc*100:.2f}%")
    print("="*60)
    print("\n  Next: python src/evaluate.py")


if __name__ == "__main__":
    train()