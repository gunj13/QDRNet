# segmentation/support_loader.py

# Loads (image, mask) support pairs from IDRID and DRIVE.
# UniverSeg expects: images normalised to [0,1], masks binary {0,1}.
# # Returns tensors shaped [1, 1, H, W] so they can be stacked directly
# into the [B, N, C, H, W] format UniverSeg forward() expects.

# CLAHE green channel everywhere — consistent with training pipeline.
# Resolution: 256x256 (was 128) to preserve small lesion structure.

# FIX 1: _gray_tensor() now uses PIL Image.convert("L") — standard luminosity
#         grayscale (0.299R + 0.587G + 0.114B).  Previously used green-channel
#         only, which made red lesions (MA, HE) invisible to UniverSeg.
# FIX 2a: CLAHE clip/tile now read from config.CLAHE_CLIP / CLAHE_TILE.
# FIX 2b: SEG_SUPPORT_SIZE now reads from config.MASK_SIZE (256).

import os
import random
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

# Single source of truth — matches config.MASK_SIZE
SEG_SUPPORT_SIZE = C.MASK_SIZE   # 256

_IDRID_MAP = {
    "MA":  ("1. Microaneurysms",  "_MA.tif"),
    "HE":  ("2. Haemorrhages",    "_HE.tif"),
    "HEX": ("3. Hard Exudates",   "_EX.tif"),
    "SEX": ("4. Soft Exudates",   "_SE.tif"),
}


def _gray_tensor(pil_img, size):
    """
    FIX 1: Standard PIL grayscale (L mode) → optional CLAHE → [1,1,H,W].

    PIL "L" mode uses 0.299R + 0.587G + 0.114B, preserving red channel signal
    that is critical for detecting microaneurysms (MA) and haemorrhages (HE),
    which appear as dark-red spots — invisible if only the green channel is used.

    CLAHE is applied after grayscale conversion to enhance local contrast
    without destroying the red-lesion signal.
    """
    gray = np.array(pil_img.convert("RGB").convert("L"), dtype=np.uint8)
    if C.CLAHE_CLIP > 0:
        clahe    = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP,
                                    tileGridSize=C.CLAHE_TILE)
        gray = clahe.apply(gray)
    pil_g = Image.fromarray(gray).resize((size, size), Image.BILINEAR)
    t = torch.from_numpy(np.array(pil_g, dtype=np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]


def _mask_to_tensor(pil_mask, size):
    pil_mask = pil_mask.resize((size, size), Image.NEAREST)
    t = torch.from_numpy((np.array(pil_mask) > 127).astype(np.float32))
    return t.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]


def load_idrid_support(lesion_type, size=SEG_SUPPORT_SIZE):
    """
    Load all available (image, mask) pairs for one IDRID lesion type.
    Uses standard grayscale + CLAHE (matches universeg_wrapper query preprocessing).
    """
    n_shots  = C.SUPPORT_SHOTS[lesion_type]
    subfolder, suffix = _IDRID_MAP[lesion_type]
    img_dir  = os.path.join(C.IDRID_DIR, "1. Original Images", "a. Training Set")
    mask_dir = os.path.join(C.IDRID_DIR, "2. All Segmentation Groundtruths",
                            "a. Training Set", subfolder)
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"IDRID mask dir not found: {mask_dir}")
    mask_files = sorted(f for f in os.listdir(mask_dir)
                        if f.lower().endswith(suffix.lower()))
    pairs = []
    for mf in mask_files:
        base     = mf.replace(suffix, "")
        img_path = os.path.join(img_dir, base + ".jpg")
        msk_path = os.path.join(mask_dir, mf)
        if not os.path.exists(img_path):
            continue
        img_t  = _gray_tensor(Image.open(img_path), size)
        mask_t = _mask_to_tensor(Image.open(msk_path).convert("L"), size)
        pairs.append((img_t, mask_t))
    if not pairs:
        raise RuntimeError(f"No valid pairs for IDRID {lesion_type}")
    random.seed(C.RANDOM_SEED)
    return random.sample(pairs, min(n_shots, len(pairs)))


def load_drive_support(size=SEG_SUPPORT_SIZE):
    """
    Load DRIVE vessel support pairs.
    Uses same standard grayscale preprocessing as IDRID.
    """
    n_shots  = C.SUPPORT_SHOTS["vessel"]
    img_dir  = os.path.join(C.DRIVE_DIR, "training", "images")
    mask_dir = os.path.join(C.DRIVE_DIR, "training", "1st_manual")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"DRIVE image dir not found: {img_dir}")
    img_files  = sorted(f for f in os.listdir(img_dir)  if f.lower().endswith(".tif"))
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(".gif"))
    n = min(n_shots, len(img_files), len(mask_files))
    pairs = []
    for i in range(n):
        img_t  = _gray_tensor(
            Image.open(os.path.join(img_dir, img_files[i])), size)
        mask_t = _mask_to_tensor(
            Image.open(os.path.join(mask_dir, mask_files[i])).convert("L"), size)
        pairs.append((img_t, mask_t))
    return pairs


def load_all_support_sets():
    """
    Returns dict:
        { "MA": [...], "HE": [...], "HEX": [...], "SEX": [...], "vessel": [...] }
    Each value is a list of (img [1,1,H,W], mask [1,1,H,W]) tensors.
    """
    sets = {}
    for ltype in ["MA", "HE", "HEX", "SEX"]:
        sets[ltype] = load_idrid_support(ltype)
        print(f"  {ltype:6s}: {len(sets[ltype]):2d} pairs  "
              f"(std gray L, CLAHE clip={C.CLAHE_CLIP}, {SEG_SUPPORT_SIZE}px)")
    sets["vessel"] = load_drive_support()
    print(f"  vessel: {len(sets['vessel']):2d} pairs  "
          f"(std gray L, CLAHE clip={C.CLAHE_CLIP}, {SEG_SUPPORT_SIZE}px)")
    return sets
