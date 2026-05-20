# evaluate.py
# Loads best_model.pt, runs test set, prints metrics and saves confusion matrix.
#
# Usage: python evaluate.py

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import config as C
from data.dataset     import get_loaders
from model.classifier import DRNet
from utils            import compute_metrics, per_class_acc, print_metrics
from plots            import plot_training_curves

BEST_CKPT = os.path.join(C.CHECKPOINT_DIR, "best_model.pt")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(BEST_CKPT):
        raise FileNotFoundError(f"No checkpoint at {BEST_CKPT}. Run train.py first.")

    ckpt = torch.load(BEST_CKPT, map_location=device)
    mode = ckpt.get("mode", "hadamard")
    print(f"Checkpoint: epoch {ckpt['epoch']}, mode={mode}, "
          f"val F1={ckpt['val_metrics']['f1']:.2f}%\n")

    _, _, test_loader = get_loaders()

    # Rebuild model with the same mode as the checkpoint
    if mode == "ablation":
        model = DRNet(use_quantum=False)
    else:
        model = DRNet(interaction_mode=mode)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for fundus, masks, labels in tqdm(test_loader, desc="Testing"):
            fundus  = fundus.to(device)
            masks   = masks.to(device)
            logits  = model(fundus, masks)
            probs   = F.softmax(logits, dim=1).cpu().numpy()
            preds   = logits.argmax(1).cpu().tolist()
            all_preds  += preds
            all_labels += labels.tolist()
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    metrics   = compute_metrics(all_labels, all_preds, C.NUM_CLASSES)
    per_class = per_class_acc(all_labels, all_preds, C.NUM_CLASSES)
    print_metrics(metrics, per_class)

    # Confusion matrix
    _save_confusion_matrix(all_labels, all_preds)

    # ROC curves (uses probabilities already collected above)
    _save_roc_curves(all_labels, all_probs)

    # Training curves (requires checkpoints/training_log.pt)
    print("Training curves...")
    plot_training_curves()

    print(f"\nOutputs saved to {C.OUTPUT_DIR}/")


def _save_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = [f"DR{i}" for i in range(C.NUM_CLASSES)]
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix — test set")
    plt.tight_layout()
    path = os.path.join(C.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {path}")


def _save_roc_curves(y_true, probs):
    DR_LABELS = ["DR0", "DR1", "DR2", "DR3", "DR4"]
    COLORS    = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    y_true  = np.array(y_true)
    y_bin   = label_binarize(y_true, classes=list(range(C.NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(8, 7))
    aucs = []
    for i, (label, color) in enumerate(zip(DR_LABELS, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{label}  (AUC = {roc_auc:.3f})")

    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], probs[:, i])[0]
        for i in range(C.NUM_CLASSES)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(C.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= C.NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="black", linewidth=2.5,
            linestyle="--", label=f"Macro avg (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="lightgray", linewidth=1, linestyle=":")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC curves — DR grading (one-vs-rest)\nTest set", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(C.OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ROC curves → {path}")

    print("\n  AUC per class:")
    for label, a in zip(DR_LABELS, aucs):
        print(f"    {label}: {a:.4f}")
    print(f"    Macro: {macro_auc:.4f}")


if __name__ == "__main__":
    main()


