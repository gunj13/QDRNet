# segmentation/universeg_wrapper.py
# Loads pretrained UniverSeg (frozen) and runs per-lesion inference.
#
# We run once per lesion type, concatenate → [B, 5, H, W] composite map.
# UniverSeg weights are downloaded automatically via torch.hub on first run.

# Ensembled inference: 3 runs × 16 random support pairs, averaged. 
#      Runs UniverSeg K times with random support subsets,
#      averages probability maps → more stable, less support-set dependent
# Query resolution: 256x256 to match support size
#      UniverSeg handles any square size at inference
#
# UniverSeg API:
#   model(target_image, support_images, support_labels)
#     target_image:   [B, 1, H, W]
#     support_images: [B, N, 1, H, W]
#     support_labels: [B, N, 1, H, W]
#     returns:        [B, 1, H, W]  soft logits, apply sigmoid for prob

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C
from segmentation.support_loader import SEG_SUPPORT_SIZE


def load_universeg(device):
    from universeg import universeg
    model = universeg(pretrained=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  UniverSeg v1 loaded (frozen, 1.18M params)")
    return model


def _fundus_to_query(fundus_rgb_tensor):
    """
    Pipeline:
      1. Denormalise from ImageNet stats back to [0, 1] RGB
      2. Convert to uint8 PIL RGB image
      3. PIL.convert("L") — standard luminosity grayscale (0.299R+0.587G+0.114B)
         This preserves the RED channel which encodes MA/HE lesion contrast.
         Previously we extracted only the green channel — MA/HE (red lesions)
         were nearly invisible in green, so UniverSeg output all zeros.
      4. CLAHE for local contrast enhancement
      5. Resize to SEG_SUPPORT_SIZE and return as float32 [1,1,H,W] in [0,1]

    Must match support_loader._gray_tensor() exactly.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb  = (fundus_rgb_tensor.cpu() * std + mean).clamp(0, 1)

    # Convert to uint8 PIL image then to standard grayscale
    rgb_np  = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_rgb = Image.fromarray(rgb_np, mode="RGB")
    gray_np = np.array(pil_rgb.convert("L"), dtype=np.uint8)

    # CLAHE for local contrast (same params as support_loader)
    if C.CLAHE_CLIP > 0:
        clahe    = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP,
                                    tileGridSize=C.CLAHE_TILE)
        gray_np = clahe.apply(gray_np)

    pil_g = Image.fromarray(gray_np).resize(
        (SEG_SUPPORT_SIZE, SEG_SUPPORT_SIZE), Image.BILINEAR)
    t = torch.from_numpy(np.array(pil_g, dtype=np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]


def _stack_support(pairs, device, max_n=16):
    import random
    if len(pairs) > max_n:
        pairs = random.sample(pairs, max_n)
    imgs  = torch.cat([p[0] for p in pairs]).unsqueeze(0).to(device)
    masks = torch.cat([p[1] for p in pairs]).unsqueeze(0).to(device)
    return imgs, masks


@torch.no_grad()
def segment_image(fundus_rgb, universeg_model, support_sets, device,
                  n_ensemble=3, support_per_run=16):
    """
    Segment one fundus image for all 5 lesion types.
    Ensembled: n_ensemble runs with random support subsets, averaged.

    Args:
        fundus_rgb:      [3,H,W] ImageNet-normalised tensor
        n_ensemble:      number of UniverSeg runs to average (default 3)
        support_per_run: max support examples per run (default 16)

    Returns:
        [5, SEG_SUPPORT_SIZE, SEG_SUPPORT_SIZE] float32 probability map
    """
    query = _fundus_to_query(fundus_rgb).to(device)

    masks = []
    for ltype in ["MA", "HE", "HEX", "SEX", "vessel"]:
        probs = []
        for _ in range(n_ensemble):
            si, sm = _stack_support(support_sets[ltype], device,
                                    max_n=support_per_run)
            logits = universeg_model(query, si, sm)
            probs.append(torch.sigmoid(logits).squeeze())
        masks.append(torch.stack(probs).mean(0))   # [H, W]

    return torch.stack(masks, dim=0)   # [5, H, W]


@torch.no_grad()
def segment_batch(fundus_batch, universeg_model, support_sets, device):
    """[B,3,H,W] → [B,5,H,W]"""
    return torch.stack([
        segment_image(fundus_batch[i], universeg_model, support_sets, device)
        for i in range(fundus_batch.shape[0])
    ])
