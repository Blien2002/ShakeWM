"""Deterministic single-frame decoders for normalized row-major teacher tokens."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def normalize_tokens(z):
    # Same FP32, non-affine per-token LN as the native WM evaluation interface.
    with torch.autocast(device_type=z.device.type, enabled=False):
        return F.layer_norm(z.detach().float(), (z.shape[-1],))


def check_tokens(z, dim, grid):
    if z.ndim != 3 or z.shape[1:] != (grid * grid, dim):
        raise ValueError(f'expected [B,{grid * grid},{dim}] row-major tokens')
    return z.detach()  # Neither training nor evaluation can backpropagate to WM.


def group_norm(channels):
    return nn.GroupNorm(math.gcd(32, channels), channels)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(group_norm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), group_norm(channels),
            nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, feature_dim=768, grid=16, image_size=256, entry=768):
        super().__init__()
        if image_size != grid * 16:
            raise ValueError('resize-conv requires four x2 stages')
        self.feature_dim, self.grid = feature_dim, grid
        self.entry = nn.Sequential(nn.Conv2d(feature_dim, entry, 1), group_norm(entry))
        self.mix = nn.Sequential(nn.Conv2d(entry, entry, 7, padding=3, groups=entry),
            group_norm(entry), nn.Conv2d(entry, entry * 2, 1), nn.SiLU(),
            nn.Conv2d(entry * 2, entry, 1))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)
        blocks, previous = [], entry
        for i, channel in enumerate([192, 96, 48, 32]):
            blocks.extend([nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(previous, channel, 3, padding=1)])
            if i < 3:
                blocks.append(Residual(channel))
            previous = channel
        self.up = nn.Sequential(*blocks)
        self.head = nn.Sequential(group_norm(previous), nn.SiLU(),
            nn.Conv2d(previous, 3, 3, padding=1))

    def forward(self, z):
        z = check_tokens(z, self.feature_dim, self.grid)
        x = z.transpose(1, 2).reshape(len(z), self.feature_dim, self.grid, self.grid)
        x = self.entry(x)
        return self.head(self.up(x + self.mix(x)))



def make_decoder(config):
    """Build the supported convolutional decoder (also the default)."""
    spec = dict(config.get('architecture', {}))
    kind = spec.pop('kind', 'resize_conv')
    if kind != 'resize_conv':
        raise ValueError('only the resize_conv decoder is supported')
    return ConvDecoder(**spec)
