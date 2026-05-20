# model/backbone.py
#
# EfficientNet backbone utilities for DRNet.
# The encoder is fixed to EfficientNet-B0 for reproducibility.
#
# LesionInjectionHook:
#   Hooks into EfficientNet-B0 features[4] (32x32 at 512px input).
#   Injects lesion mask as a multiplicative spatial gate at this mid-level feature,
#   forcing the backbone to attend to lesion locations from inside the network.
#   This helps GradCAM explanations towards the retinal interior.
#
# LightweightAuxNet uses GroupNorm instead of BatchNorm:
#   Lesion mask probabilities are 0.01–0.10. BatchNorm normalises by batch std,
#   amplifying near-zero inputs by ~200x into noise. GroupNorm normalises within
#   channel groups, independent of input magnitude — sparse signals stay sparse.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


# ── Lesion injection hook ──────────────────────────────────────────────────

class LesionInjectionHook(nn.Module):
    """
    Registers a forward hook on a mid-level backbone feature map.
    Injects lesion signal as a multiplicative gate: output = feat * (1 + gate)

    Where lesions are absent: gate ≈ 0, feature unchanged.
    Where lesions are present: gate > 0, feature amplified.
    Gradients flow through gated locations → GradCAM follows lesions.

    hook_channels: number of channels at the hooked layer
    """
    def __init__(self, hook_channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(C.AUX_IN_CHANNELS, hook_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hook_channels),
        )
        self._mask = None   # set externally before each forward pass

    def set_mask(self, mask: torch.Tensor):
        self._mask = mask

    def hook_fn(self, module, input, output):
        if self._mask is None:
            return output
    
        # ── ABLATION: comment out below 4 lines to disable injection hook ──
        B, C_feat, H, W = output.shape
        mask_hw = F.interpolate(self._mask, size=(H, W),
                                mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gate_conv(mask_hw))
        return output * (1.0 + gate)
        # ── END ABLATION block ──

        # return output # comment this to run WITH injection hook


# ── EfficientNet backbone ──────────────────────────────────────────────────

class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B0 pretrained feature extractor.
    At IMG_SIZE=512: output is [B, 1280, 16, 16]
    At IMG_SIZE=224: output is [B, 1280, 7, 7]

    Injection hook placed at features[4] (80 channels for EfficientNet-B0).
    """
    def __init__(self):
        super().__init__()
        base = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        self.features     = base.features
        self.out_channels = 1280  # EfficientNet-B0 final feature channels, change to 1536 for B3

        # Hook at features[4] — mid-level, good spatial resolution
        # EfficientNet-B0 features[4] has 80 output channels
        self.injector = LesionInjectionHook(hook_channels=80)
        self.features[4].register_forward_hook(self.injector.hook_fn)

        if C.BACKBONE_FREEZE:
            for p in self.features.parameters():
                p.requires_grad = False

    def set_mask(self, mask: torch.Tensor):
        self.injector.set_mask(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)   # [B, 1280, H', W']


# ── Lightweight auxiliary network for lesion masks ─────────────────────────

class LightweightAuxNet(nn.Module):
    """
    Processes 5-channel lesion mask → [B, out_channels, H', W'].
    Uses AdaptiveAvgPool2d to match backbone spatial output.
    Uses GroupNorm instead of BatchNorm — handles sparse near-zero inputs.
    """
    def __init__(self, out_channels: int = 1280):
        super().__init__()
        self.blocks = nn.Sequential(
            self._block(C.AUX_IN_CHANNELS, 16),
            self._block(16, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.proj = nn.Conv2d(128, out_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)    # final pooling to 1x1 for fusion

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        # GroupNorm: groups=min(4,out_c) handles sparse inputs without amplifying noise
        n_groups = min(4, out_c)
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(n_groups, out_c),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

    def forward(self, seg_map: torch.Tensor) -> torch.Tensor:
        # seg_map: [B, 5, MASK_SIZE, MASK_SIZE]
        x = F.interpolate(seg_map, size=(C.IMG_SIZE, C.IMG_SIZE),
                          mode="bilinear", align_corners=False)
        x = self.blocks(x)
        x = self.proj(x)        # [B, out_channels, H'', W'']
        return x


# ── Bottleneck and quantum projections ─────────────────────────────────────

class BottleneckProjection(nn.Module):
    """[B, channels, H, W] → [B, CLASSIFIER_DIM] via GAP."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(C.DROPOUT),
            nn.Linear(256, C.CLASSIFIER_DIM),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.gap(x))


class QuantumProjection(nn.Module):
    """CLASSIFIER_DIM ↔ NUM_QUBITS around the quantum/Hadamard layer."""
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(C.CLASSIFIER_DIM, C.BOTTLENECK_DIM), nn.Tanh())
        self.up   = nn.Sequential(
            nn.Linear(C.BOTTLENECK_DIM, C.CLASSIFIER_DIM), nn.GELU())

    def project_down(self, x):
        return self.down(x)

    def project_up(self, x):
        return self.up(x)
