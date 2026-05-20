
# Key fix: predict_stretched() scales each prediction map to its own max.
# UniverSeg outputs P(lesion) ~ 0.01–0.15 for tiny MA/HE lesions.
# Raw display (prob*255) → pixel values 2–38 → visually black even if signal exists.
# Contrast stretching reveals the spatial structure.  Raw max is printed and
# annotated on the figure so the display is honest.

import os, random, glob
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as C
from segmentation.support_loader import load_idrid_support, load_drive_support, SEG_SUPPORT_SIZE
from segmentation.universeg_wrapper import load_universeg

os.makedirs(C.OUTPUT_DIR, exist_ok=True)

_IDRID_LESION_MAP = {
    "MA":  ("1. Microaneurysms", "_MA.tif",  "Microaneurysms"),
    "HE":  ("2. Haemorrhages",   "_HE.tif",  "Haemorrhages"),
    "HEX": ("3. Hard Exudates",  "_EX.tif",  "Hard Exudates"),
    "SEX": ("4. Soft Exudates",  "_SE.tif",  "Soft Exudates"),
}

IDRID_TEST_IMG  = os.path.join(C.IDRID_DIR, "1. Original Images", "b. Testing Set")
IDRID_TEST_MASK = os.path.join(C.IDRID_DIR, "2. All Segmentation Groundtruths", "b. Testing Set")
DRIVE_TEST_IMG  = os.path.join(C.DRIVE_DIR, "training", "images")
DRIVE_TEST_MASK = os.path.join(C.DRIVE_DIR, "training", "1st_manual")


def load_query_tensor(img_path):
    pil  = Image.open(img_path).convert("RGB")
    gray = np.array(pil.convert("L"), dtype=np.uint8)
    if C.CLAHE_CLIP > 0:
        clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE)
        gray  = clahe.apply(gray)
    t = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(SEG_SUPPORT_SIZE, SEG_SUPPORT_SIZE),
                         mode="bilinear", align_corners=False)


def load_gt_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 0).astype(np.uint8) * 255


def load_display_image(img_path, size=256):
    return np.array(Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR))


@torch.no_grad()
def predict_stretched(query_t, sup_imgs, sup_masks, model, device):
    """
    Contrast-stretched prediction: scales prob map to [0,255] using its own max.
    Returns (uint8 display array, raw_max float).
    raw_max is printed so you know the actual probability level.
    """
    logits  = model(query_t.to(device), sup_imgs.to(device), sup_masks.to(device))
    prob    = torch.sigmoid(logits).squeeze().cpu().numpy()
    raw_max = float(prob.max())
    if raw_max > 1e-6:
        display = (prob / raw_max * 255).astype(np.uint8)
    else:
        display = np.zeros_like(prob, dtype=np.uint8)
    return display, raw_max


@torch.no_grad()
def predict_binary(query_t, sup_imgs, sup_masks, model, device, threshold=0.4):
    logits = model(query_t.to(device), sup_imgs.to(device), sup_masks.to(device))
    prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (prob > threshold).astype(np.uint8) * 255


def stack_support(pairs, device):
    imgs  = torch.cat([p[0] for p in pairs], dim=0).unsqueeze(0)
    masks = torch.cat([p[1] for p in pairs], dim=0).unsqueeze(0)
    return imgs.to(device), masks.to(device)


def pick_idrid_test_pairs(lesion_type, n=2):
    subfolder, suffix, _ = _IDRID_LESION_MAP[lesion_type]
    mask_dir = os.path.join(IDRID_TEST_MASK, subfolder)
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Missing: {mask_dir}")
    mask_files = sorted(glob.glob(os.path.join(mask_dir, f"*{suffix}")))
    valid = []
    for mf in mask_files:
        base = os.path.basename(mf).replace(suffix, "")
        ip   = os.path.join(IDRID_TEST_IMG, base + ".jpg")
        if not os.path.exists(ip):
            continue
        if np.array(Image.open(mf).convert("L")).max() == 0:
            continue
        valid.append((ip, mf))
    if not valid:
        raise RuntimeError(f"No non-empty test masks for {lesion_type}")
    random.seed(C.RANDOM_SEED)
    return random.sample(valid, min(n, len(valid)))


def pick_drive_test_pairs(n=1):
    if not os.path.isdir(DRIVE_TEST_IMG):
        raise FileNotFoundError(f"Missing: {DRIVE_TEST_IMG}")
    imgs  = sorted(glob.glob(os.path.join(DRIVE_TEST_IMG,  "*.tif")))
    masks = sorted(glob.glob(os.path.join(DRIVE_TEST_MASK, "*.gif")))
    return list(zip(imgs, masks))[-n:]


def build_figure(row_data, title, save_path, pred_col="Predict", show_raw_max=True):
    n_rows  = len(row_data)
    n_pairs = len(row_data[0]["pairs"])
    n_cols  = n_pairs * 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5*n_cols+1.2, 3.2*n_rows+0.8),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.30})
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_headers = []
    for _ in range(n_pairs):
        col_headers += ["Image", "Ground Truth", pred_col]
    for j, h in enumerate(col_headers):
        axes[0, j].set_title(h, fontsize=10, pad=6)
    for i, row in enumerate(row_data):
        axes[i, 0].set_ylabel(row["label"], fontsize=10,
                               rotation=90, labelpad=8, va="center")
        for p_idx, entry in enumerate(row["pairs"]):
            orig, gt, pred, raw_max = entry
            cb = p_idx * 3
            gt_disp   = np.array(Image.fromarray(gt).resize((256,256), Image.NEAREST))
            pred_disp = np.array(Image.fromarray(pred).resize((256,256), Image.NEAREST))
            axes[i, cb+0].imshow(orig); axes[i, cb+0].axis("off")
            axes[i, cb+1].imshow(gt_disp, cmap="gray", vmin=0, vmax=255); axes[i, cb+1].axis("off")
            axes[i, cb+2].imshow(pred_disp, cmap="gray", vmin=0, vmax=255); axes[i, cb+2].axis("off")
            if show_raw_max:
                axes[i, cb+2].text(0.5, -0.06, f"raw max={raw_max:.3f}",
                                   transform=axes[i, cb+2].transAxes,
                                   ha="center", va="top", fontsize=7, color="gray")
    fig.suptitle(title, fontsize=12, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved → {save_path}")


def idrid_seg_test(model, device):
    print("\nIDRID test set segmentation (contrast-stretched display)")
    row_data = []
    for ltype, (_, _, label) in _IDRID_LESION_MAP.items():
        print(f"  {ltype} ...", end=" ", flush=True)
        sup_pairs           = load_idrid_support(ltype)
        sup_imgs, sup_masks = stack_support(sup_pairs, device)
        try:
            test_pairs = pick_idrid_test_pairs(ltype, n=2)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"SKIP ({e})")
            continue
        display_pairs = []
        for ip, mp in test_pairs:
            orig          = load_display_image(ip)
            gt            = load_gt_mask(mp)
            query_t       = load_query_tensor(ip)
            pred, raw_max = predict_stretched(query_t, sup_imgs, sup_masks, model, device)
            display_pairs.append((orig, gt, pred, raw_max))
        row_data.append({"label": label, "pairs": display_pairs})
        maxes = [f"{p[3]:.3f}" for p in display_pairs]
        print(f"done  raw_max: {', '.join(maxes)}")
    if not row_data:
        print("  No IDRID test data found.")
        return
    build_figure(
        row_data,
        "IDRID segmentation — UniverSeg output (contrast-normalized per image)\n"
        "raw max prob annotated below each prediction",
        os.path.join(C.OUTPUT_DIR, "idrid_seg_test_fixed.png"),
        pred_col="UniverSeg (normalized)",
    )


def drive_ves_test(model, device):
    print("\nDRIVE vessel segmentation")
    sup_pairs           = load_drive_support()
    sup_imgs, sup_masks = stack_support(sup_pairs, device)
    try:
        test_pairs = pick_drive_test_pairs(n=1)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  SKIP ({e})")
        return
    display_pairs = []
    for ip, mp in test_pairs:
        print(f"  {os.path.basename(ip)} ...", end=" ", flush=True)
        orig  = load_display_image(ip)
        gt    = load_gt_mask(mp)
        query_t = load_query_tensor(ip)
        pred  = predict_binary(query_t, sup_imgs, sup_masks, model, device, threshold=0.4)
        display_pairs.append((orig, gt, pred, 1.0))
        print("done")
    build_figure(
        [{"label": "Vessels", "pairs": display_pairs}],
        "Segmentation results on DRIVE\n(Image / Ground Truth / Binary threshold=0.4)",
        os.path.join(C.OUTPUT_DIR, "drive_ves_test_fixed.png"),
        show_raw_max=False,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("\nLoading UniverSeg...")
    model = load_universeg(device)
    idrid_seg_test(model, device)
    drive_ves_test(model, device)
    print(f"\nDone. Check {C.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
