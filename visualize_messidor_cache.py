#!/usr/bin/env python3
# visualize_messidor_cache.py
#
# Identical layout to visualize_aptos_cache.py but for Messidor-2.
# Shows fundus image (CLAHE-green) + 5 cached UniverSeg probability maps.
#
# Run precompute_masks_messidor.py first.
#
# Usage:
#   python visualize_messidor_cache.py              # 6 random gradable images
#   python visualize_messidor_cache.py --n 12       # show 12 images
#   python visualize_messidor_cache.py --grade 3    # only DR grade 3

import os, sys, argparse, random
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

MESSIDOR_DIR   = os.path.join("data", "raw", "MESSIDOR")
MESSIDOR_CSV   = os.path.join(MESSIDOR_DIR, "messidor_data.csv")
MESSIDOR_IMG   = os.path.join(MESSIDOR_DIR, "images")
MESSIDOR_CACHE = os.path.join(C.MASK_CACHE_DIR, "messidor")

LESION_NAMES = ["MA", "HE", "HEX", "SEX", "Vessel"]
DR_LABELS    = {0: "DR0 — No DR", 1: "DR1 — Mild",
                2: "DR2 — Moderate", 3: "DR3 — Severe", 4: "DR4 — Proliferative"}


def clahe_green(pil_img):
    arr   = np.array(pil_img.convert("RGB"))
    clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE)
    arr[:, :, 1] = clahe.apply(arr[:, :, 1])
    return Image.fromarray(arr)


def stretch(prob_map):
    raw_max = float(prob_map.max())
    display = (prob_map / raw_max * 255).astype(np.uint8) if raw_max > 1e-6 \
              else np.zeros_like(prob_map, dtype=np.uint8)
    return display, raw_max


def load_samples(n, grade_filter):
    if not os.path.exists(MESSIDOR_CSV):
        raise FileNotFoundError(f"CSV not found: {MESSIDOR_CSV}")
    if not os.path.isdir(MESSIDOR_CACHE):
        raise FileNotFoundError(
            f"Mask cache not found: {MESSIDOR_CACHE}\n"
            "Run  python precompute_masks_messidor.py  first.")

    df = pd.read_csv(MESSIDOR_CSV)
    df = df[df["adjudicated_gradable"] == 1]   # gradable only

    if grade_filter is not None:
        df = df[df["diagnosis"] == grade_filter]
        if df.empty:
            raise RuntimeError(f"No gradable samples with grade={grade_filter}.")

    valid = []
    for _, row in df.iterrows():
        img_id    = row["id_code"]                        # e.g. xxx.png
        stem      = os.path.splitext(img_id)[0]           # strip .png
        img_path  = os.path.join(MESSIDOR_IMG,   img_id)
        mask_path = os.path.join(MESSIDOR_CACHE, stem + ".pt")
        if os.path.exists(img_path) and os.path.exists(mask_path):
            valid.append((img_id, int(row["diagnosis"]), img_path, mask_path))

    if not valid:
        raise RuntimeError("No samples with both image and .pt cache found.")

    random.seed(C.RANDOM_SEED)
    return random.sample(valid, min(n, len(valid)))


def build_figure(samples, save_path):
    n_rows = len(samples)
    fig, axes = plt.subplots(
        n_rows, 6,
        figsize=(3.2 * 6 + 1.0, 3.0 * n_rows + 1.0),
        gridspec_kw={"wspace": 0.04, "hspace": 0.35},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for j, h in enumerate(["Fundus (CLAHE-green)"] + LESION_NAMES):
        axes[0, j].set_title(h, fontsize=9, pad=6)

    for i, (img_id, grade, img_path, mask_path) in enumerate(samples):
        # Fundus
        fundus = np.array(
            clahe_green(Image.open(img_path).convert("RGB"))
            .resize((256, 256), Image.BILINEAR))
        axes[i, 0].imshow(fundus)
        axes[i, 0].axis("off")
        axes[i, 0].set_ylabel(
            f"{img_id}\n{DR_LABELS.get(grade, f'DR{grade}')}",
            fontsize=7, rotation=90, labelpad=8, va="center")

        # 5 mask channels
        masks = torch.load(mask_path, map_location="cpu", weights_only=True)
        if masks.shape[-1] != C.MASK_SIZE:
            import torch.nn.functional as F
            masks = F.interpolate(masks.unsqueeze(0),
                                  size=(C.MASK_SIZE, C.MASK_SIZE),
                                  mode="bilinear", align_corners=False).squeeze(0)

        for ch in range(5):
            display, rmax = stretch(masks[ch].numpy())
            axes[i, ch+1].imshow(display, cmap="hot", vmin=0, vmax=255)
            axes[i, ch+1].axis("off")
            axes[i, ch+1].text(0.5, -0.06, f"max={rmax:.3f}",
                                transform=axes[i, ch+1].transAxes,
                                ha="center", va="top", fontsize=6, color="gray")

    fig.suptitle(
        "Messidor-2 — fundus vs cached UniverSeg masks\n"
        "(contrast-stretched per channel; raw max annotated)",
        fontsize=11, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved → {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",     type=int, default=6)
    p.add_argument("--grade", type=int, default=None, choices=[0,1,2,3,4])
    args = p.parse_args()

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print(f"Loading {args.n} Messidor-2 samples"
          + (f" (grade={args.grade})" if args.grade is not None else "") + "...")

    samples = load_samples(args.n, args.grade)
    print(f"  Found {len(samples)} valid samples.")

    grade_tag = f"_gr{args.grade}" if args.grade is not None else ""
    save_path = os.path.join(C.OUTPUT_DIR,
                             f"messidor_cache_viz{grade_tag}_n{len(samples)}.png")
    build_figure(samples, save_path)


if __name__ == "__main__":
    main()