"""Short causal CNN and optional long causal anti-alias/TCN history encoders."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Conv1d):
    def forward(self, x):
        left = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(x, (left, 0)))


class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, 128))
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)

    def forward(self, x):
        # Only complete eligible windows reach a branch; no padded sample is pooled.
        return self.attention(self.query.expand(x.shape[0], -1, -1), x, x, need_weights=False)[0][:, 0]


class ShortEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout, dilation in [(6, 64, 1), (64, 128, 2), (128, 128, 4)]:
            layers += [CausalConv(cin, cout, 5, dilation=dilation), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.pool = AttentionPool()

    def forward(self, x):
        return self.pool(self.net(x.transpose(1, 2)).transpose(1, 2))


class ResidualTCN(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.net = nn.Sequential(CausalConv(128, 128, 3, dilation=dilation), nn.GELU(),
                                 CausalConv(128, 128, 3, dilation=dilation), nn.GELU())

    def forward(self, x):
        return x + self.net(x)


class LongEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 63-tap Hamming-windowed sinc, cutoff 20 Hz at 200 Hz. Causal delay 155 ms.
        n = torch.arange(63, dtype=torch.float32) - 31
        taps = 0.2 * torch.sinc(0.2 * n) * torch.hamming_window(63, periodic=False)
        self.register_buffer("fir", (taps / taps.sum()).repeat(6, 1, 1))
        self.input = nn.Sequential(nn.Conv1d(6, 64, 1), nn.GELU(), nn.Conv1d(64, 128, 1))
        self.blocks = nn.Sequential(*(ResidualTCN(d) for d in [1, 2, 4, 8, 16, 32]))
        self.pool = AttentionPool()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.conv1d(F.pad(x, (62, 0)), self.fir.to(x.dtype), groups=6)[..., ::4]
        return self.pool(self.blocks(self.input(x)).transpose(1, 2))


class IMUTokens(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.mode = mode
        # Keep all parameters and token slots identical across V0/V1/RGB-only.
        self.short = ShortEncoder()
        self.long = LongEncoder()
        self.project = nn.ModuleList([nn.Sequential(nn.LayerNorm(128), nn.Linear(128, width)) for _ in range(2)])
        self.kind = nn.Parameter(torch.zeros(2, width))
        self.no_imu = nn.Parameter(torch.zeros(2, width))
        nn.init.normal_(self.no_imu, std=0.02)

    def forward(self, batch, time, short=None, long=None, eligible=None, dropped=None):
        out = self.no_imu[None, None].expand(batch, time, -1, -1).clone()
        if short is None or self.mode == "none":
            return out
        if eligible is None:
            raise ValueError("IMU eligibility is required")
        active = eligible.clone()
        if dropped is not None:
            active &= ~dropped[:, None, None]
        if self.mode == "v0":
            active[..., 1] = False
        for i, (values, encoder) in enumerate([(short, self.short), (long, self.long)]):
            selected = active[..., i]
            if selected.any():
                if values is None:
                    raise ValueError("eligible IMU window missing")
                out[:, :, i][selected] = self.project[i](encoder(values[selected])) + self.kind[i]
        return out
