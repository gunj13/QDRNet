#!/usr/bin/env python3
# plots.py — training curves + ROC curves
#
# Usage:
#   python plots.py            # requires checkpoints/training_log.pt + best_model.pt
#
# Produces:
#   outputs/training_curves.png   — loss and F1 vs epoch
#   outputs/roc_curves.png        — per-class OvR ROC with AUC

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import torch.nn.functional as F

import config as C
from data.dataset     import get_loaders
from model.classifier import DRNet

os.makedirs(C.OUTPUT_DIR, exist_ok=True)

LOG_PATH  = os.path.join(C.CHECKPOINT_DIR, "training_log.pt")
BEST_CKPT = os.path.join(C.CHECKPOINT_DIR, "best_model.pt")
DR_LABELS = ["DR0", "DR1", "DR2", "DR3", "DR4"]
COLORS    = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


# ── Training curves ─────────────────────────────────────────────────────────

def plot_training_curves():
    if not os.path.exists(LOG_PATH):
        print(f"  [!] No training log at {LOG_PATH}")
        print("      Add log saving to train.py (see note below) and retrain,")
        print("      or run: python plots.py --roc-only")
        return

    log = torch.load(LOG_PATH, map_location="cpu", weights_only=True)
    epochs    = log["epochs"]
    tr_loss   = log["train_loss"]
    vl_loss   = log["val_loss"]
    tr_f1     = log["train_f1"]
    vl_f1     = log["val_f1"]
    best_ep   = log.get("best_epoch", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training history — DRNet", fontsize=13)

    # Loss
    ax1.plot(epochs, tr_loss, color="#2196F3", linewidth=2, label="Train loss")
    ax1.plot(epochs, vl_loss, color="#F44336", linewidth=2, label="Val loss")
    if best_ep:
        ax1.axvline(best_ep, color="gray", linestyle="--", linewidth=1,
                    label=f"Best epoch ({best_ep})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    # F1
    ax2.plot(epochs, tr_f1, color="#2196F3", linewidth=2, label="Train F1")
    ax2.plot(epochs, vl_f1, color="#F44336", linewidth=2, label="Val F1")
    if best_ep:
        best_vl = vl_f1[best_ep - epochs[0]]
        ax2.axvline(best_ep, color="gray", linestyle="--", linewidth=1)
        ax2.annotate(f"{best_vl:.1f}%",
                     xy=(best_ep, best_vl),
                     xytext=(best_ep + 1, best_vl - 3),
                     fontsize=9, color="gray")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Weighted F1 (%)")
    ax2.set_title("F1 score"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(C.OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}")


# ── ROC curves ───────────────────────────────────────────────────────────────

def plot_roc_curves():
    if not os.path.exists(BEST_CKPT):
        print(f"  [!] No checkpoint at {BEST_CKPT} — run train.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(BEST_CKPT, map_location=device)
    mode   = ckpt.get("mode", "hadamard")

    if mode == "ablation":
        model = DRNet(use_quantum=False)
    else:
        model = DRNet(interaction_mode=mode)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    _, _, test_loader = get_loaders()

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for fundus, masks, labels in tqdm(test_loader, desc="  Collecting probs"):
            logits = model(fundus.to(device), masks.to(device))
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels += labels.tolist()

    all_probs  = np.vstack(all_probs)                       # [N, 5]
    all_labels = np.array(all_labels)
    y_bin      = label_binarize(all_labels, classes=list(range(C.NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(8, 7))

    # Per-class OvR ROC
    aucs = []
    for i, (label, color) in enumerate(zip(DR_LABELS, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{label}  (AUC = {roc_auc:.3f})")

    # Macro average
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], all_probs[:, i])[0]
        for i in range(C.NUM_CLASSES)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(C.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= C.NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="black", linewidth=2.5,
            linestyle="--", label=f"Macro avg (AUC = {macro_auc:.3f})")

    ax.plot([0,1], [0,1], color="lightgray", linewidth=1, linestyle=":")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC curves — DR grading (one-vs-rest)\nTest set", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(C.OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}")

    # Print AUC summary
    print("\n  AUC per class:")
    for label, a in zip(DR_LABELS, aucs):
        print(f"    {label}: {a:.4f}")
    print(f"    Macro: {macro_auc:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--roc-only",    action="store_true")
    p.add_argument("--curves-only", action="store_true")
    args = p.parse_args()

    if not args.roc_only:
        print("Training curves...")
        plot_training_curves()

    if not args.curves_only:
        print("ROC curves...")
        plot_roc_curves()


if __name__ == "__main__":
    main()
