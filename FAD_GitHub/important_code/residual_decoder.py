"""BatchNorm-free feature-conditioned residual image decoder."""

import math

import torch
import torch.nn as nn


def _group_count(channels, maximum=32):
    groups = min(maximum, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = _group_count(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual_blocks=1):
        super().__init__()
        blocks = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        ]
        blocks.extend(ResidualBlock(out_channels) for _ in range(residual_blocks))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class ResidualFeatureDecoder(nn.Module):
    """Feature decoder with a bounded residual refinement branch.

    The base branch maps a defended feature to an image.  The refinement
    branch is conditioned on both that feature and the base image and returns
    ``residual_scale * tanh(residual)``.  No BatchNorm is used anywhere.
    """

    decoder_type = "residual_gn_v1"

    def __init__(self, feature_dim=2048, img_size=64, out_channels=3, residual_scale=0.1):
        super().__init__()
        if img_size < 16 or (img_size & (img_size - 1)) != 0:
            raise ValueError("img_size must be a power of two and >= 16")
        self.feature_dim = feature_dim
        self.img_size = img_size
        self.out_channels = out_channels
        self.residual_scale = float(residual_scale)
        self.start_size = 4
        self.start_channels = 512
        self.num_upsample = int(math.log2(img_size)) - 2

        self.feature_norm = nn.LayerNorm(feature_dim)
        self.base_fc = nn.Sequential(
            nn.Linear(feature_dim, 4096),
            nn.GELU(),
            nn.Linear(4096, self.start_channels * self.start_size * self.start_size),
        )

        base_channels = [512, 256, 128, 64, 32, 16, 8]
        base_stages = []
        for index in range(self.num_upsample):
            base_stages.append(UpsampleBlock(base_channels[index], base_channels[index + 1], residual_blocks=1))
        self.base_stages = nn.Sequential(*base_stages)
        self.base_out = nn.Conv2d(base_channels[self.num_upsample], out_channels, 3, 1, 1)

        self.condition_fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 128 * self.start_size * self.start_size),
        )
        condition_channels = [128, 64, 32, 16, 8, 8, 8]
        condition_stages = []
        for index in range(self.num_upsample):
            condition_stages.append(UpsampleBlock(condition_channels[index], condition_channels[index + 1], residual_blocks=1))
        self.condition_stages = nn.Sequential(*condition_stages)
        final_condition_channels = condition_channels[self.num_upsample]

        refine_channels = 32
        self.refine_in = nn.Sequential(
            nn.Conv2d(out_channels + final_condition_channels, refine_channels, 3, 1, 1),
            nn.GroupNorm(_group_count(refine_channels), refine_channels),
            nn.SiLU(inplace=True),
        )
        self.refine_blocks = nn.Sequential(ResidualBlock(refine_channels), ResidualBlock(refine_channels))
        self.refine_out = nn.Conv2d(refine_channels, out_channels, 3, 1, 1)

    def forward_with_parts(self, feature):
        feature = self.feature_norm(feature)
        base = self.base_fc(feature).view(-1, self.start_channels, self.start_size, self.start_size)
        base = self.base_stages(base)
        base = torch.tanh(self.base_out(base))

        condition = self.condition_fc(feature).view(-1, 128, self.start_size, self.start_size)
        condition = self.condition_stages(condition)
        residual = self.refine_in(torch.cat([base, condition], dim=1))
        residual = self.refine_blocks(residual)
        residual = self.residual_scale * torch.tanh(self.refine_out(residual))
        output = torch.clamp(base + residual, -1.0, 1.0)
        return output, base, residual

    def forward(self, feature):
        output, _, _ = self.forward_with_parts(feature)
        return output
