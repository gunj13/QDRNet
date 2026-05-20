# model/classifier.py
#
# Pipeline:
#   fundus [B,3,512,512]
#       → EfficientNetBackbone (with LesionInjectionHook at features[4])
#       → [B, 1280, 16, 16]
#
#   masks [B,5,512,512]
#       → LightweightAuxNet (GroupNorm)
#       → [B, 1280, 16, 16]  (same spatial size via interpolation matching)
#
#       → LesionGuidedAttention (CBAM multiplicative gate)
#       → [B, 1280, 16, 16]
#
#       → BottleneckProjection (GAP → 256 → CLASSIFIER_DIM=64)
#       → [B, 64]
#
#       → QuantumProjection.down → [B, 4]
#       → HadamardLayer / VQCLayer → [B, 4]
#       → QuantumProjection.up → [B, 64]
#
#       → head: Linear(64→64)→GELU→LayerNorm→Dropout→Linear(64→5)
#       → [B, 5] logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

from model.backbone      import EfficientNetBackbone, LightweightAuxNet, \
                                BottleneckProjection, QuantumProjection
from model.cbam          import LesionGuidedAttention
from model.quantum_layer import build_interaction_layer


class DRNet(nn.Module):
    def __init__(self, use_quantum: bool = None,
                 interaction_mode: Optional[str] = None):
        super().__init__()

        self.backbone   = EfficientNetBackbone()
        ch              = self.backbone.out_channels

        self.aux_net    = LightweightAuxNet(out_channels=ch)
        self.fusion     = LesionGuidedAttention(channels=ch,
                                                reduction=C.CBAM_REDUCTION)
        self.bottleneck = BottleneckProjection(in_channels=ch)
        self.q_proj     = QuantumProjection()

        if use_quantum is False:
            self.interaction = nn.Sequential(
                nn.Linear(C.BOTTLENECK_DIM, C.BOTTLENECK_DIM), nn.Tanh())
            print("  Interaction: Linear (classical ablation)")
        else:
            mode = interaction_mode or C.INTERACTION_MODE
            self.interaction = build_interaction_layer(mode)

        self.head = nn.Sequential(
            nn.Linear(C.CLASSIFIER_DIM, C.CLASSIFIER_DIM),
            nn.GELU(),
            nn.LayerNorm(C.CLASSIFIER_DIM),
            nn.Dropout(C.DROPOUT),
            nn.Linear(C.CLASSIFIER_DIM, C.NUM_CLASSES),
        )

    def forward(self, fundus: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        """
        fundus: [B, 3, 512, 512]
        masks:  [B, 5, 512, 512]
        """
        # Tell the injection hook what the current batch's masks are
        # The hook fires automatically inside backbone.forward()
        self.backbone.set_mask(masks)

        feat_a = self.backbone(fundus)   # [B, 1280, 16, 16]
        feat_b = self.aux_net(masks)     # [B, 1280, H'', W'']

        # Align spatial dims if they differ (they should match via AdaptivePool)
        if feat_b.shape[-2:] != feat_a.shape[-2:]:
            feat_b = F.interpolate(feat_b, size=feat_a.shape[-2:],
                                   mode="bilinear", align_corners=False)
        
        # FOR NO CBAM ABLATION: comment out fusion(a,b) and uncomment the fused=feat_a
        fused  = self.fusion(feat_a, feat_b)     # [B, 1280, 16, 16]    # NORMAL
        # fused = feat_a    # NO CBAM ABLATION


        z64    = self.bottleneck(fused)           # [B, 64]
        z4     = self.q_proj.project_down(z64)   # [B, 4]
        q_out  = self.interaction(z4)             # [B, 4]
        z64_up = self.q_proj.project_up(q_out)   # [B, 64]
        return self.head(z64_up)                  # [B, 5]
