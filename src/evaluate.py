import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score
)
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import get_dataloaders
from model import build_model


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH   = Path(cfg["paths"]["models"]) / "best_model.pth"
OUT_PLOTS    = Path(cfg["paths"]["outputs"]) / "plots"
OUT_PREDS    = Path(cfg["paths"]["outputs"]) / "predictions"
CLASS_NAMES  = list(cfg["classes"].values())

OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_PREDS.mkdir(parents=True, exist_ok=True)


def collect_predictions(model, loader, device):

    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def plot_confusion_matrix(y_true, y_pred):

    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")

    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            raw = cm[i, j]
            pct = cm_norm[i, j]
            color = "black" if pct > 0.5 else "white"
            ax.text(j, i, f"{pct:.2%}\n(n={raw})",
                    ha="center", va="center", color=color, fontsize=11)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, color="#aaaaaa", fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, color="#aaaaaa", fontsize=11)
    ax.set_xlabel("Predicted Label", color="#aaaaaa", fontsize=11)
    ax.set_ylabel("True Label",      color="#aaaaaa", fontsize=11)
    ax.set_title("Confusion Matrix (Normalized)", color="white", fontsize=13, pad=12)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    out = OUT_PLOTS / "confusion_matrix.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  Saved -> {out}")


def plot_roc_curve(y_true, y_probs):

    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")

    ax.plot(fpr, tpr, color="#10b981", linewidth=2.5,
            label=f"ROC curve  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#555555", linewidth=1.5,
            linestyle="--", label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#10b981")

    ax.set_xlabel("False Positive Rate", color="#aaaaaa", fontsize=11)
    ax.set_ylabel("True Positive Rate",  color="#aaaaaa", fontsize=11)
    ax.set_title("ROC Curve — Counterfeit Class", color="white", fontsize=13, pad=12)
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=10)
    ax.tick_params(colors="#aaaaaa")

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    out = OUT_PLOTS / "roc_curve.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  Saved -> {out}")
    return roc_auc


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print("LOADING MODEL FROM CHECKPOINT")
    print("="*60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first.")

    model, device = build_model(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded epoch {ckpt['epoch']} checkpoint")
    print(f"  Checkpoint val accuracy : {ckpt['val_acc']*100:.2f}%")

    _, _, test_loader = get_dataloaders()

    print("\n  Running inference on test set...")
    y_true, y_pred, y_probs = collect_predictions(model, test_loader, device)

    acc       = (y_true == y_pred).mean()
    f1        = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall    = recall_score(y_true, y_pred, average="weighted")

    print("\n" + "="*60)
    print("TEST SET METRICS")
    print("="*60)
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Weighted F1       : {f1:.4f}")
    print(f"  Weighted Precision: {precision:.4f}")
    print(f"  Weighted Recall   : {recall:.4f}")

    print("\n  Per-class report:")
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)

    print("  Generating plots...")
    plot_confusion_matrix(y_true, y_pred)
    roc_auc = plot_roc_curve(y_true, y_probs)
    print(f"  AUC-ROC           : {roc_auc:.4f}")

    results = {
        "accuracy":          float(acc),
        "f1_weighted":       float(f1),
        "precision_weighted": float(precision),
        "recall_weighted":   float(recall),
        "auc_roc":           float(roc_auc),
        "checkpoint_epoch":  int(ckpt["epoch"]),
        "checkpoint_val_acc": float(ckpt["val_acc"]),
    }
    out_json = OUT_PREDS / "test_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved -> {out_json}")
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("  Next: python src/gradcam.py")


if __name__ == "__main__":
    run()