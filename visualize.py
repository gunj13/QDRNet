#!/usr/bin/env python3
# visualize.py — updated for EfficientNet-B0 + 512px setup
#
# Usage:
#   python visualize.py --mode preprocess    # before/after CLAHE, all at 512px, 300dpi
#   python visualize.py --mode segmentation  # UniverSeg masks for DR2+ images
#   python visualize.py --mode gradcam       # GradCAM: DR1 DR2 DR3 DR4 DR4 (needs checkpoint)

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import cv2
import torchvision.transforms as T

import config as C
from data.dataset import _clahe_green

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_test_transform = T.Compose([
    T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

os.makedirs(C.OUTPUT_DIR, exist_ok=True)

_LESION_NAMES  = ["MA", "HE", "HEX", "SEX", "vessel"]
_LESION_COLORS = ["Reds", "Oranges", "Blues", "Greens", "Purples"]
_GRADE_NAMES   = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_preprocessing(n: int = 5):
    """
    Show n APTOS training images: original vs CLAHE-enhanced.
    Both columns resized to 512x512 so sizes are consistent.
    Output at 300 DPI for paper inclusion.
    Tries to sample one image per DR grade for variety.
    """
    df      = pd.read_csv(os.path.join(C.APTOS_DIR, "train.csv"))
    img_dir = os.path.join(C.APTOS_DIR, "train_images")

    # Sample one from each grade if n >= 5, else random
    rows = []
    if n >= 5:
        for grade in range(5):
            subset = df[df["diagnosis"] == grade]
            if len(subset) > 0:
                rows.append(subset.sample(1, random_state=C.RANDOM_SEED).iloc[0])
        rows = rows[:n]
    else:
        rows = [df.sample(n, random_state=C.RANDOM_SEED).iloc[i] for i in range(n)]

    fig, axes = plt.subplots(n, 2, figsize=(8, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Fundus image preprocessing: original vs CLAHE-enhanced green channel\n"
        f"(all images displayed at {C.IMG_SIZE}×{C.IMG_SIZE} px)",
        fontsize=11, y=1.01
    )

    resize = T.Resize((C.IMG_SIZE, C.IMG_SIZE))

    for i, row in enumerate(rows):
        img_path = os.path.join(img_dir, row["id_code"] + ".png")
        pil_orig = Image.open(img_path).convert("RGB")

        # Resize both to 512x512 for consistent display
        pil_orig_r = resize(pil_orig)
        pil_enh_r  = resize(_clahe_green(pil_orig))

        grade     = int(row["diagnosis"])
        grade_str = f"DR{grade} — {_GRADE_NAMES[grade]}"

        axes[i, 0].imshow(np.array(pil_orig_r))
        axes[i, 0].set_title(f"Original ({grade_str})", fontsize=9, pad=4)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.array(pil_enh_r))
        axes[i, 1].set_title("CLAHE-enhanced (green channel)", fontsize=9, pad=4)
        axes[i, 1].axis("off")

    plt.tight_layout()
    path = os.path.join(C.OUTPUT_DIR, "preprocessing_samples.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}  (300 DPI, {C.IMG_SIZE}px display)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Segmentation visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_segmentation(n: int = 3):
    """
    Show n APTOS images with their 5 UniverSeg lesion masks.
    Prefers DR2+ images where lesions are visible.
    """
    df        = pd.read_csv(os.path.join(C.APTOS_DIR, "train.csv"))
    img_dir   = os.path.join(C.APTOS_DIR, "train_images")
    cache_dir = os.path.join(C.MASK_CACHE_DIR, "aptos", "train")

    # Prefer images with DR severity >= 2 (more visible lesions)
    df_pos = df[df["diagnosis"] >= 2]
    if len(df_pos) < n:
        df_pos = df
    sample = df_pos.sample(min(n, len(df_pos)), random_state=C.RANDOM_SEED)

    fig, axes = plt.subplots(n, 6, figsize=(18, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "UniverSeg few-shot lesion segmentation maps\n"
        "(columns: fundus image | MA | HE | HEX | SEX | vessel)",
        fontsize=11, y=1.01
    )

    resize = T.Resize((C.IMG_SIZE, C.IMG_SIZE))

    for i, (_, row) in enumerate(sample.iterrows()):
        img_path  = os.path.join(img_dir, row["id_code"] + ".png")
        mask_path = os.path.join(cache_dir, row["id_code"] + ".pt")

        pil_img = resize(Image.open(img_path).convert("RGB"))
        grade   = int(row["diagnosis"])
        axes[i, 0].imshow(np.array(pil_img))
        axes[i, 0].set_title(f"DR{grade} — {_GRADE_NAMES[grade]}", fontsize=8)
        axes[i, 0].axis("off")

        if not os.path.exists(mask_path):
            for j in range(1, 6):
                axes[i, j].text(0.5, 0.5, "no cache\nrun precompute_masks.py",
                                ha="center", va="center", fontsize=7)
                axes[i, j].axis("off")
            continue

        masks = torch.load(mask_path, map_location="cpu", weights_only=True)  # [5, H, W]
        # Upsample mask to IMG_SIZE for display
        if masks.shape[-1] != C.IMG_SIZE:
            masks = F.interpolate(masks.unsqueeze(0),
                                  size=(C.IMG_SIZE, C.IMG_SIZE),
                                  mode="bilinear", align_corners=False).squeeze(0)

        for j, (ltype, cmap) in enumerate(zip(_LESION_NAMES, _LESION_COLORS)):
            m        = masks[j].numpy()
            coverage = (m > 0.5).mean() * 100
            axes[i, j+1].imshow(m, cmap=cmap, vmin=0, vmax=1)
            axes[i, j+1].set_title(f"{ltype}  ({coverage:.2f}%)", fontsize=8)
            axes[i, j+1].axis("off")

    plt.tight_layout()
    path = os.path.join(C.OUTPUT_DIR, "segmentation_maps.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GradCAM visualisation
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Hooks into model.fusion (CBAM output, [B, 1280, 16, 16]) to produce
    class-discriminative spatial activation maps.
    """
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks = [
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())),
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())),
        ]

    def __call__(self, fundus, masks, target_class=None):
        self.model.eval()
        logits = self.model(fundus, masks)
        if target_class is None:
            target_class = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self.activations).sum(1, keepdim=True))

        # Upsample to full image size for smooth overlay
        cam = F.interpolate(cam, size=(C.IMG_SIZE, C.IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class, logits.softmax(1).detach().cpu().numpy()[0]

    def remove(self):
        for h in self._hooks:
            h.remove()


def visualize_gradcam():
    """
    GradCAM overlays on exactly 6 test images:
    DR1, DR2, DR3, DR4, DR4 (two severe cases).
    Grades shown: [1, 2, 3, 4, 4] — all with actual DR.
    Output: 6-row figure at 300 DPI.
    """
    from model.classifier import DRNet
    import matplotlib.cm as cm

    ckpt_path = os.path.join(C.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print("  No checkpoint found — run train.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(ckpt_path, map_location=device)
    mode   = ckpt.get("mode", "hadamard")
    if mode == "ablation":
        model = DRNet(use_quantum=False)
    else:
        model = DRNet(interaction_mode=mode)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    # Hook into model.fusion — CBAM output is the most semantically meaningful layer
    gcam = GradCAM(model, model.fusion)

    df        = pd.read_csv(os.path.join(C.APTOS_DIR, "test.csv"))
    img_dir   = os.path.join(C.APTOS_DIR, "test_images")
    cache_dir = os.path.join(C.MASK_CACHE_DIR, "aptos", "test")

    # Explicitly select: DR1, DR2, DR3, DR4, DR4, DR4
    # Use different random seeds per grade to get variety
    target_grades = [1, 2, 3, 4, 4]
    selected = []
    seeds = [38, 45, 47, 47, 43] #49 for dr1, 38, 36
    for grade, seed in zip(target_grades, seeds):
        subset = df[df["diagnosis"] == grade]
        if len(subset) == 0:
            # Fall back to any DR if grade not in test set
            subset = df[df["diagnosis"] > 0]
        row = subset.sample(1, random_state=seed).iloc[0]
        selected.append(row)

    n   = len(selected)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.8 * n))
    fig.suptitle(
        "Grad-CAM: regions influencing DR grade prediction\n"
        "(left: fundus image | centre: activation map | right: overlay)",
        fontsize=11, y=1.005
    )

    for i, row in enumerate(selected):
        img_path  = os.path.join(img_dir, row["id_code"] + ".png")
        mask_path = os.path.join(cache_dir, row["id_code"] + ".pt")

        pil_img = Image.open(img_path).convert("RGB")
        fundus  = _test_transform(_clahe_green(pil_img)).unsqueeze(0).to(device)

        if os.path.exists(mask_path):
            masks = torch.load(mask_path, map_location=device, weights_only=True)
            if masks.shape[-1] != C.IMG_SIZE:
                masks = F.interpolate(
                    masks.unsqueeze(0), size=(C.IMG_SIZE, C.IMG_SIZE),
                    mode="bilinear", align_corners=False).squeeze(0)
            masks = masks.unsqueeze(0)
        else:
            masks = torch.zeros(1, 5, C.IMG_SIZE, C.IMG_SIZE, device=device)

        with torch.enable_grad():
            heatmap, pred_cls, probs = gcam(fundus, masks)

        true_grade = int(row["diagnosis"])
        correct    = pred_cls == true_grade

        # Resize original image for display
        img_np = np.array(pil_img.resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR))

        # Colour map heatmap
        heat_col = cm.jet(heatmap)[:, :, :3]
        overlay  = np.clip(0.55 * (img_np / 255.0) + 0.45 * heat_col, 0, 1)

        # Title colours: green if correct, red if wrong
        title_colour = "darkgreen" if correct else "crimson"
        symbol       = "✓" if correct else "✗"

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(
            f"True: DR{true_grade} ({_GRADE_NAMES[true_grade]})",
            fontsize=9
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[i, 1].set_title(
            f"Pred: DR{pred_cls} ({_GRADE_NAMES[pred_cls]}) {symbol}",
            fontsize=9, color=title_colour
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        conf = probs[pred_cls] * 100
        axes[i, 2].set_title(f"Overlay  (conf: {conf:.1f}%)", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    gcam.remove()
    path = os.path.join(C.OUTPUT_DIR, "gradcam.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}  (300 DPI)")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["preprocess", "segmentation", "gradcam"],
                   required=True)
    p.add_argument("--n", type=int, default=None,
                   help="Number of images (preprocess/segmentation only; gradcam always 6)")
    args = p.parse_args()

    if args.mode == "preprocess":
        visualize_preprocessing(args.n or 5)
    elif args.mode == "segmentation":
        visualize_segmentation(args.n or 3)
    elif args.mode == "gradcam":
        visualize_gradcam()


if __name__ == "__main__":
    main()