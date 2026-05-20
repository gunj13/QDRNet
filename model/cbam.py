# model/cbam.py
# CBAM + LesionGuidedAttention with multiplicative gating.

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = x.view(B, C, -1).mean(-1)
        mx  = x.view(B, C, -1).max(-1).values
        return torch.sigmoid(self.fc(avg) + self.fc(mx)).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ch_att = ChannelAttention(channels, reduction)
        self.sp_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ch_att(x)
        x = x * self.sp_att(x)
        return x


class LesionGuidedAttention(nn.Module):
    """
    Fuses backbone features (Path A) and lesion-map features (Path B).

    Gate = sigmoid(conv(lesion_feat))       ∈ (0, 1)
    Fused = backbone_feat * (1 + gate)

    DR0: gate ≈ 0 → fused ≈ backbone_feat   (no lesions, no change)
    DR3/4: gate > 0 → lesion regions amplified in backbone features

    Works on 2D feature maps before GAP.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # 1x1 conv gate for spatial feature maps
        self.lesion_gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.cbam        = CBAM(channels, reduction)
        self.norm        = nn.BatchNorm2d(channels)

    def forward(self, backbone_feat: torch.Tensor,
                lesion_feat: torch.Tensor) -> torch.Tensor:
        # Both inputs: [B, channels, H, W]
        gate  = torch.sigmoid(self.lesion_gate(lesion_feat))
        fused = backbone_feat * (1.0 + gate)
        return self.norm(self.cbam(fused))
