#!/usr/bin/env python3
# visualize_aptos_cache.py
#
# Shows what is actually fed to the model during training:
#   Col 1 — original APTOS fundus image (after CLAHE green-channel preprocessing)
#   Col 2-6 — the 5 cached UniverSeg probability maps loaded from the .pt file
#             (MA, HE, HEX, SEX, vessel), contrast-stretched exactly like
#             visualize_seg_masks.py so low-probability signals are visible.
#
# Usage:
#   python visualize_aptos_cache.py                  # 6 random train images
#   python visualize_aptos_cache.py --split val      # from val split
#   python visualize_aptos_cache.py --split test     # from test split
#   python visualize_aptos_cache.py --n 12           # show 12 images
#   python visualize_aptos_cache.py --grade 3        # only DR grade 3 images

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config as C

LESION_NAMES = ["MA", "HE", "HEX", "SEX", "Vessel"]
DR_LABELS    = {0: "DR0 — No DR", 1: "DR1 — Mild",
                2: "DR2 — Moderate", 3: "DR3 — Severe", 4: "DR4 — Proliferative"}

SPLIT_CONFIG = {
    "train": ("train.csv",  "train_images"),
    "val":   ("valid.csv",  "val_images"),
    "test":  ("test.csv",   "test_images"),
}


def clahe_green(pil_img: Image.Image) -> Image.Image:
    """Same preprocessing as data/dataset.py — CLAHE on green channel only."""
    arr   = np.array(pil_img.convert("RGB"))
    clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE)
    arr[:, :, 1] = clahe.apply(arr[:, :, 1])
    return Image.fromarray(arr)


def stretch(prob_map: np.ndarray):
    """
    Contrast-stretch a probability map to [0, 255] uint8, same as
    visualize_seg_masks.predict_stretched().  Returns (uint8, raw_max).
    """
    raw_max = float(prob_map.max())
    if raw_max > 1e-6:
        display = (prob_map / raw_max * 255).astype(np.uint8)
    else:
        display = np.zeros_like(prob_map, dtype=np.uint8)
    return display, raw_max


def load_samples(split: str, n: int, grade_filter):
    csv_file, img_folder = SPLIT_CONFIG[split]
    csv_path  = os.path.join(C.APTOS_DIR, csv_file)
    img_dir   = os.path.join(C.APTOS_DIR, img_folder)
    cache_dir = os.path.join(C.MASK_CACHE_DIR, "aptos", split)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"Mask cache not found: {cache_dir}\n"
            "Run  python precompute_masks.py  first."
        )

    df = pd.read_csv(csv_path)
    if grade_filter is not None:
        df = df[df["diagnosis"] == grade_filter]
        if df.empty:
            raise RuntimeError(f"No samples with grade={grade_filter} in {split} split.")

    # Only keep rows that have both image and .pt file
    valid = []
    for _, row in df.iterrows():
        img_id    = row["id_code"]
        img_path  = os.path.join(img_dir,   img_id + ".png")
        mask_path = os.path.join(cache_dir, img_id + ".pt")
        if os.path.exists(img_path) and os.path.exists(mask_path):
            valid.append((img_id, int(row["diagnosis"]), img_path, mask_path))

    if not valid:
        raise RuntimeError("No samples with both image and .pt cache found.")

    random.seed(C.RANDOM_SEED)
    return random.sample(valid, min(n, len(valid)))


def build_figure(samples, split: str, save_path: str):
    """
    One row per sample.
    Columns: Original | MA | HE | HEX | SEX | Vessel
    """
    n_rows = len(samples)
    n_cols = 6   # 1 fundus + 5 lesion channels
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols + 1.0, 3.0 * n_rows + 1.0),
        gridspec_kw={"wspace": 0.04, "hspace": 0.35},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = ["Fundus (CLAHE-green)"] + LESION_NAMES
    for j, h in enumerate(col_headers):
        axes[0, j].set_title(h, fontsize=9, pad=6)

    for i, (img_id, grade, img_path, mask_path) in enumerate(samples):
        # ── fundus ──────────────────────────────────────────────────────────
        pil_fundus = clahe_green(Image.open(img_path).convert("RGB"))
        fundus_arr = np.array(pil_fundus.resize((256, 256), Image.BILINEAR))
        axes[i, 0].imshow(fundus_arr)
        axes[i, 0].axis("off")
        row_label = f"{img_id}\n{DR_LABELS.get(grade, f'DR{grade}')}"
        axes[i, 0].set_ylabel(row_label, fontsize=7, rotation=90,
                               labelpad=8, va="center")

        # ── 5 cached probability maps ────────────────────────────────────────
        masks = torch.load(mask_path, map_location="cpu", weights_only=True)
        # Handle old 128px cache
        if masks.shape[-1] != C.MASK_SIZE:
            import torch.nn.functional as F
            masks = F.interpolate(
                masks.unsqueeze(0),
                size=(C.MASK_SIZE, C.MASK_SIZE),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        for ch, lname in enumerate(LESION_NAMES):
            prob_np        = masks[ch].numpy()
            display, rmax  = stretch(prob_np)
            axes[i, ch+1].imshow(display, cmap="hot", vmin=0, vmax=255)
            axes[i, ch+1].axis("off")
            axes[i, ch+1].text(
                0.5, -0.06, f"max={rmax:.3f}",
                transform=axes[i, ch+1].transAxes,
                ha="center", va="top", fontsize=6, color="gray",
            )

    title = (f"APTOS {split} split — fundus vs cached UniverSeg masks\n"
             f"(contrast-stretched per channel; raw max annotated)")
    fig.suptitle(title, fontsize=11, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualise APTOS images alongside their cached .pt segmentation masks.")
    parser.add_argument("--split",  default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--n",      type=int, default=6,
                        help="Number of images to show (default 6)")
    parser.add_argument("--grade",  type=int, default=None,
                        choices=[0, 1, 2, 3, 4],
                        help="Filter to a specific DR grade (0-4)")
    args = parser.parse_args()

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    print(f"Loading {args.n} samples from APTOS {args.split} split"
          + (f" (grade={args.grade})" if args.grade is not None else "") + "...")

    samples = load_samples(args.split, args.n, args.grade)
    print(f"  Found {len(samples)} valid samples.")

    grade_tag = f"_gr{args.grade}" if args.grade is not None else ""
    save_path = os.path.join(
        C.OUTPUT_DIR,
        f"aptos_cache_viz_{args.split}{grade_tag}_n{len(samples)}.png",
    )
    build_figure(samples, args.split, save_path)


if __name__ == "__main__":
    main()
