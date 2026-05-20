#!/usr/bin/env python3
# precompute_masks_messidor.py
#
# Generates 5-channel UniverSeg lesion masks for all gradable Messidor-2 images.
# Identical pipeline to precompute_masks.py — same support sets, same UniverSeg
# model, same output format ([5, MASK_SIZE, MASK_SIZE] float32 tensor).
#
# Run ONCE before evaluate_messidor.py.
#
# Usage:
#   python precompute_masks_messidor.py
#
# Output: data/mask_cache/messidor/{id_code_without_ext}.pt
#   e.g.  data/mask_cache/messidor/20051020_43808_0100_PP.pt

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

import config as C
from segmentation.support_loader    import load_all_support_sets
from segmentation.universeg_wrapper import load_universeg, segment_image

MESSIDOR_DIR  = os.path.join("data", "raw", "MESSIDOR")
MESSIDOR_CSV  = os.path.join(MESSIDOR_DIR, "messidor_data.csv")
MESSIDOR_IMG  = os.path.join(MESSIDOR_DIR, "images")
MESSIDOR_CACHE = os.path.join(C.MASK_CACHE_DIR, "messidor")

_transform = T.Compose([
    T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    os.makedirs(MESSIDOR_CACHE, exist_ok=True)

    # Load CSV — keep only gradable images
    df_all = pd.read_csv(MESSIDOR_CSV)
    df     = df_all[df_all["adjudicated_gradable"] == 1].reset_index(drop=True)
    print(f"Messidor-2: {len(df_all)} total → {len(df)} gradable images to process\n")

    # Load support sets (same IDRiD/DRIVE support as APTOS training)
    print("Loading support sets...")
    support_sets = load_all_support_sets()

    print("\nLoading UniverSeg...")
    universeg_model = load_universeg(device)

    # Pre-move support tensors to device
    support_sets_dev = {
        ltype: [(img.to(device), mask.to(device)) for img, mask in pairs]
        for ltype, pairs in support_sets.items()
    }

    print(f"\nGenerating masks → {MESSIDOR_CACHE}\n")
    skipped = computed = errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Messidor-2"):
        img_id    = row["id_code"]              # e.g. 20051020_43808_0100_PP.png
        stem      = os.path.splitext(img_id)[0] # strip .png → 20051020_43808_0100_PP
        save_path = os.path.join(MESSIDOR_CACHE, stem + ".pt")

        if os.path.exists(save_path):
            skipped += 1
            continue

        img_path = os.path.join(MESSIDOR_IMG, img_id)
        if not os.path.exists(img_path):
            print(f"\n  [!] Missing image: {img_path}")
            errors += 1
            continue

        try:
            img_tensor = _transform(Image.open(img_path).convert("RGB"))
            masks      = segment_image(img_tensor, universeg_model,
                                       support_sets_dev, device)
            torch.save(masks.cpu(), save_path)
            computed += 1
        except Exception as e:
            print(f"\n  [!] Error on {img_id}: {e}")
            errors += 1

    print(f"\nDone.  computed={computed}  skipped={skipped}  errors={errors}")
    print(f"Masks saved to: {MESSIDOR_CACHE}")
    print("Next: python evaluate_messidor.py")


if __name__ == "__main__":
    main()