#!/usr/bin/env python3
# evaluate_messidor.py
#
# Evaluates trained DRNet on Messidor-2 for cross-dataset benchmarking.
# Loads precomputed UniverSeg masks from data/mask_cache/messidor/.
# Run precompute_masks_messidor.py first.
#
# Same preprocessing as APTOS: CLAHE green channel, 512px resize, ImageNet norm.
# Only gradable images (adjudicated_gradable == 1) are used.
#
# Usage:
#   python evaluate_messidor.py
#   python evaluate_messidor.py --ckpt checkpoints/best_model.pt

import os
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

import config as C
from model.classifier import DRNet
from utils import compute_metrics, per_class_acc, print_metrics

MESSIDOR_DIR   = os.path.join("data", "raw", "MESSIDOR")
MESSIDOR_CSV   = os.path.join(MESSIDOR_DIR, "messidor_data.csv")
MESSIDOR_IMG   = os.path.join(MESSIDOR_DIR, "images")
MESSIDOR_CACHE = os.path.join(C.MASK_CACHE_DIR, "messidor")
BEST_CKPT      = os.path.join(C.CHECKPOINT_DIR, "best_model.pt")

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = T.Compose([
    T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])


def clahe_green(pil_img: Image.Image) -> Image.Image:
    arr          = np.array(pil_img.convert("RGB"))
    clahe        = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE)
    arr[:, :, 1] = clahe.apply(arr[:, :, 1])
    return Image.fromarray(arr)


class MessidorDataset(Dataset):
    """
    Messidor-2 dataset with precomputed UniverSeg masks.
    Falls back to zero masks if cache file is missing for an image
    (prints a warning so you know to rerun precompute_masks_messidor.py).
    Preloads all images into RAM for fast inference.
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, cache_dir: str):
        self.df        = df.reset_index(drop=True)
        self.labels    = self.df["diagnosis"].tolist()
        self.cache_dir = cache_dir
        self.id_codes  = self.df["id_code"].tolist()
        self._missing  = 0

        print(f"  Preloading {len(df)} Messidor-2 images into RAM...", flush=True)
        self.images = []
        for _, row in self.df.iterrows():
            path = os.path.join(img_dir, row["id_code"])
            pil  = clahe_green(Image.open(path).convert("RGB"))
            self.images.append(pil)
        print("  Preload done.", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img   = _TRANSFORM(self.images[idx])   # [3, H, W]

        # Load cached mask — stem strips .png extension
        stem      = os.path.splitext(self.id_codes[idx])[0]
        mask_path = os.path.join(self.cache_dir, stem + ".pt")

        if os.path.exists(mask_path):
            masks = torch.load(mask_path, map_location="cpu", weights_only=True)
            if masks.shape[-1] != C.MASK_SIZE:
                masks = F.interpolate(
                    masks.unsqueeze(0),
                    size=(C.MASK_SIZE, C.MASK_SIZE),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
        else:
            if self._missing == 0:
                print(f"\n  [!] Mask cache missing for {self.id_codes[idx]} "
                      f"— run precompute_masks_messidor.py first. "
                      f"Using zero masks for missing entries.")
            self._missing += 1
            masks = torch.zeros(5, C.MASK_SIZE, C.MASK_SIZE)

        return img, masks, self.labels[idx]


def save_confusion_matrix(y_true, y_pred, out_path):
    from sklearn.metrics import confusion_matrix
    labels = [f"DR{i}" for i in range(C.NUM_CLASSES)]
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix — Messidor-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {out_path}")


def save_roc_curves(y_true, y_probs, out_path):
    colours = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(7, 6))
    macro_auc, n_valid = 0.0, 0

    for c in range(C.NUM_CLASSES):
        binary = (np.array(y_true) == c).astype(int)
        if binary.sum() == 0:
            continue
        try:
            auc      = roc_auc_score(binary, y_probs[:, c])
            fpr, tpr, _ = roc_curve(binary, y_probs[:, c])
            ax.plot(fpr, tpr, color=colours[c],
                    label=f"DR{c}  (AUC = {auc:.3f})")
            macro_auc += auc
            n_valid   += 1
        except ValueError:
            pass

    macro_auc /= max(n_valid, 1)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves — Messidor-2  (Macro AUC = {macro_auc:.3f})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ROC curves → {out_path}")
    return macro_auc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=BEST_CKPT)
    p.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"No checkpoint at {args.ckpt}. Run train.py first.")

    ckpt = torch.load(args.ckpt, map_location=device)
    mode = ckpt.get("mode", "hadamard")
    print(f"Checkpoint: epoch {ckpt['epoch']}, mode={mode}, "
          f"val F1={ckpt['val_metrics']['f1']:.2f}%\n")

    if not os.path.exists(MESSIDOR_CSV):
        raise FileNotFoundError(
            f"CSV not found at {MESSIDOR_CSV}\n"
            f"Expected: data/raw/MESSIDOR/messidor_data.csv")

    df_all = pd.read_csv(MESSIDOR_CSV)
    df     = df_all[df_all["adjudicated_gradable"] == 1].reset_index(drop=True)
    print(f"Messidor-2: {len(df_all)} total → {len(df)} gradable images")
    print("  Class distribution:")
    for g in sorted(df["diagnosis"].unique()):
        print(f"    DR{g}: {(df['diagnosis'] == g).sum()}")
    print()

    dataset = MessidorDataset(df, MESSIDOR_IMG, MESSIDOR_CACHE)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=C.NUM_WORKERS,
                         pin_memory=True)

    if mode == "ablation":
        model = DRNet(use_quantum=False)
    else:
        model = DRNet(interaction_mode=mode)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for fundus, masks, labels in tqdm(loader, desc="Messidor-2"):
            fundus = fundus.to(device)
            masks  = masks.to(device)
            logits = model(fundus, masks)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds  += logits.argmax(1).cpu().tolist()
            all_labels += labels.tolist()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)

    metrics   = compute_metrics(all_labels, all_preds, C.NUM_CLASSES)
    per_class = per_class_acc(all_labels, all_preds, C.NUM_CLASSES)

    print("\n── Messidor-2 Results ──────────────────────────────────────────")
    print_metrics(metrics, per_class)

    print("\n  AUC per class:")
    macro_auc, n_valid = 0.0, 0
    for c in range(C.NUM_CLASSES):
        binary = (np.array(all_labels) == c).astype(int)
        if binary.sum() == 0:
            print(f"    DR{c}: N/A (no samples)")
            continue
        try:
            auc = roc_auc_score(binary, all_probs[:, c])
            print(f"    DR{c}: {auc:.4f}")
            macro_auc += auc
            n_valid   += 1
        except ValueError:
            print(f"    DR{c}: N/A")
    if n_valid:
        print(f"    Macro: {macro_auc / n_valid:.4f}")

    save_confusion_matrix(
        all_labels, all_preds,
        os.path.join(C.OUTPUT_DIR, "confusion_matrix_messidor2.png"))
    save_roc_curves(
        all_labels, all_probs,
        os.path.join(C.OUTPUT_DIR, "roc_curves_messidor2.png"))

    print(f"\nOutputs saved to {C.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
