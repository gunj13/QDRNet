# data/dataset.py

import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORM = T.Compose([
    T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])
_EVAL_TRANSFORM = T.Compose([
    T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])


def _clahe_green(pil_img):
    arr          = np.array(pil_img.convert("RGB"))
    clahe        = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE)
    arr[:, :, 1] = clahe.apply(arr[:, :, 1])
    return Image.fromarray(arr)


class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, cache_dir, split):
        self.df        = df.reset_index(drop=True)
        self.split     = split
        self.transform = _TRAIN_TRANSFORM if split == "train" else _EVAL_TRANSFORM
        self.labels    = self.df["diagnosis"].tolist()

        print(f"  [{split}] Preloading {len(df)} images + masks into RAM...", flush=True)
        self.images = []
        self.masks  = []

        for _, row in self.df.iterrows():
            img_id = row["id_code"]

            # CLAHE image — stored as PIL, transform applied per-call in __getitem__
            # (PIL is needed so ColorJitter and Resize can work on it)
            pil = _clahe_green(
                Image.open(os.path.join(img_dir, img_id + ".png")).convert("RGB"))
            self.images.append(pil)

            # Mask
            mask_path = os.path.join(cache_dir, img_id + ".pt")
            if os.path.exists(mask_path):
                m = torch.load(mask_path, map_location="cpu", weights_only=True)
                if m.shape[-1] != C.MASK_SIZE:
                    m = F.interpolate(m.unsqueeze(0),
                                      size=(C.MASK_SIZE, C.MASK_SIZE),
                                      mode="bilinear",
                                      align_corners=False).squeeze(0)
            else:
                m = torch.zeros(5, C.MASK_SIZE, C.MASK_SIZE)
            self.masks.append(m)

        print(f"  [{split}] Preload done.", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img   = self.transform(self.images[idx])  # PIL → tensor [3,H,W]
        masks = self.masks[idx].clone()            # [5,MASK_SIZE,MASK_SIZE]
        if getattr(C, 'ZERO_MASKS', False):
            masks = torch.zeros_like(masks)

        # Paired tensor flips (train only) — fast, spatially aligned
        if self.split == "train":
            if torch.rand(1).item() < 0.5:
                img   = torch.flip(img,   dims=[2])
                masks = torch.flip(masks, dims=[2])
            if torch.rand(1).item() < 0.5:
                img   = torch.flip(img,   dims=[1])
                masks = torch.flip(masks, dims=[1])
            if torch.rand(1).item() < 0.5:
                angle = (torch.rand(1).item() - 0.5) * 30  # -15 to +15 degrees
                img   = T.functional.rotate(img,   angle)
                masks = T.functional.rotate(masks, angle)

        return img, masks, self.labels[idx]


def _sample_weights(labels):
    counts = Counter(labels)
    total  = len(labels)
    return torch.tensor([total / counts[l] for l in labels], dtype=torch.float32)


def get_loaders():
    train_df = pd.read_csv(os.path.join(C.APTOS_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(C.APTOS_DIR, "valid.csv"))
    test_df  = pd.read_csv(os.path.join(C.APTOS_DIR, "test.csv"))
    print(f"  APTOS — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    def _ds(df, folder, split):
        return APTOSDataset(df,
                            os.path.join(C.APTOS_DIR, folder),
                            os.path.join(C.MASK_CACHE_DIR, "aptos", split),
                            split)

    train_ds  = _ds(train_df, "train_images", "train")
    val_ds    = _ds(val_df,   "val_images",   "val")
    test_ds   = _ds(test_df,  "test_images",  "test")
    nw        = C.NUM_WORKERS

    train_loader = DataLoader(
        train_ds, batch_size=C.BATCH_SIZE,
        sampler=WeightedRandomSampler(
            _sample_weights(train_df["diagnosis"].tolist()),
            num_samples=len(train_df), replacement=True),
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
    val_loader  = DataLoader(val_ds,   batch_size=C.BATCH_SIZE, shuffle=False,
                             num_workers=nw, pin_memory=True,
                             persistent_workers=(nw > 0))
    test_loader = DataLoader(test_ds,  batch_size=C.BATCH_SIZE, shuffle=False,
                             num_workers=nw, pin_memory=True,
                             persistent_workers=(nw > 0))
    return train_loader, val_loader, test_loader