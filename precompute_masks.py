#!/usr/bin/env python3
# precompute_masks.py
# Run ONCE before training. Generates 5-channel lesion masks for every
# APTOS image using pretrained UniverSeg and saves them as .pt files.
#
# Usage:
#   python precompute_masks.py
#
# Output: data/mask_cache/aptos/{train,val,test}/{id_code}.pt
#         Each file is a float32 tensor [5, 128, 128]
#
# Subsequent training runs just load these cached files — no UniverSeg
# overhead in the training loop at all.

import os
import sys
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

import config as C
from segmentation.support_loader import load_all_support_sets
from segmentation.universeg_wrapper import load_universeg, segment_image

# Same transforms as training (we need to denorm inside segment_image)
_transform = T.Compose([
    T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def process_split(split_name: str, csv_file: str, img_folder: str,
                  universeg_model, support_sets: dict, device: str):
    csv_path  = os.path.join(C.APTOS_DIR, csv_file)
    img_dir   = os.path.join(C.APTOS_DIR, img_folder)
    cache_dir = os.path.join(C.MASK_CACHE_DIR, "aptos", split_name)
    os.makedirs(cache_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    skipped = computed = errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        img_id    = row["id_code"]
        save_path = os.path.join(cache_dir, img_id + ".pt")

        if os.path.exists(save_path):
            skipped += 1
            continue

        img_path = os.path.join(img_dir, img_id + ".png")
        if not os.path.exists(img_path):
            print(f"\n  [!] Missing image: {img_path}")
            errors += 1
            continue

        try:
            img_tensor = _transform(Image.open(img_path).convert("RGB"))
            masks = segment_image(img_tensor, universeg_model, support_sets, device)
            torch.save(masks.cpu(), save_path)
            computed += 1
        except Exception as e:
            print(f"\n  [!] Error on {img_id}: {e}")
            errors += 1

    print(f"    computed={computed}  skipped={skipped}  errors={errors}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading support sets...")
    support_sets = load_all_support_sets()

    print("\nLoading UniverSeg...")
    universeg_model = load_universeg(device)

    # Pre-move support tensors to device to avoid per-image transfers
    support_sets_dev = {
        ltype: [(img.to(device), mask.to(device)) for img, mask in pairs]
        for ltype, pairs in support_sets.items()
    }

    print("\nGenerating masks for all APTOS splits...")
    splits = [
        ("train", "train.csv", "train_images"),
        ("val",   "valid.csv", "val_images"),
        ("test",  "test.csv",  "test_images"),
    ]
    for split_name, csv_file, img_folder in splits:
        process_split(split_name, csv_file, img_folder,
                      universeg_model, support_sets_dev, device)

    print("\nDone. Set USE_MASK_CACHE=True (already default) and run: python train.py")


if __name__ == "__main__":
    main()
