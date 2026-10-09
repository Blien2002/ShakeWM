"""Short causal CNN and optional long causal anti-alias/TCN history encoders."""
import torch
from torch import nn
from torch.nn import functional as F

SHORT_SAMPLES = 20
LONG_SAMPLES = 600
LONG_STRIDE = 4
LONG_FIR_TAPS = 31  # Hamming-windowed sinc, 20 Hz cutoff at 200 Hz; causal group delay 15 samples = 75 ms


class CausalConv(nn.Conv1d):
    """1D convolution padded only on the past side, so output time stays causal."""

    def forward(self, x):
        """Convolve `[batch, channels, time]` input without reading future samples."""
        left = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(x, (left, 0)))


class AttentionPool(nn.Module):
    """Single-query attention pooling that keeps time order.

    A learned position embedding is added to keys and values, so the pooled vector can encode
    *when* inside the window a feature occurred (phase, time since the last impact). The newest
    time step, whose causal receptive field covers the whole window, is merged in explicitly.
    """

    def __init__(self, length):
        """Create a learned query and positional embedding for a fixed window length."""
        super().__init__()
        self.length = length
        self.query = nn.Parameter(torch.zeros(1, 1, 128))
        self.position = nn.Parameter(torch.zeros(1, length, 128))
        nn.init.normal_(self.position, std=0.02)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.merge = nn.Linear(256, 128)

    def forward(self, x):
        """Pool `[batch, time, 128]` features and fuse attention with the newest step."""
        if x.shape[1] != self.length:
            raise ValueError(f"expected {self.length} time steps, got {x.shape[1]}")
        # Only complete eligible windows reach a branch; no padded sample is pooled.
        keys = x + self.position.to(x.dtype)
        pooled = self.attention(self.query.expand(x.shape[0], -1, -1), keys, keys, need_weights=False)[0][:, 0]
        return self.merge(torch.cat([pooled, x[:, -1]], -1))


class ShortEncoder(nn.Module):
    """Encode the most recent 20 six-channel IMU samples with causal convolutions."""

    def __init__(self):
        """Build the short-window CNN and fixed-length attention pool."""
        super().__init__()
        layers = []
        for cin, cout, dilation in [(6, 64, 1), (64, 128, 2), (128, 128, 4)]:
            layers += [CausalConv(cin, cout, 5, dilation=dilation), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.pool = AttentionPool(SHORT_SAMPLES)

    def forward(self, x):
        """Map `[batch, 20, 6]` normalized IMU samples to one 128D vector per window."""
        return self.pool(self.net(x.transpose(1, 2)).transpose(1, 2))


class ResidualTCN(nn.Module):
    """Two-layer causal temporal block with an identity residual connection."""

    def __init__(self, dilation):
        """Build the block using the requested temporal dilation."""
        super().__init__()
        self.net = nn.Sequential(CausalConv(128, 128, 3, dilation=dilation), nn.GELU(),
                                 CausalConv(128, 128, 3, dilation=dilation), nn.GELU())

    def forward(self, x):
        """Return the input plus its causal convolutional residual."""
        return x + self.net(x)


class LongEncoder(nn.Module):
    """Anti-alias and encode the 600-sample IMU history into one 128D vector."""

    def __init__(self, taps=LONG_FIR_TAPS):
        """Build the causal FIR, stride-four decimator, residual TCN, and pool."""
        super().__init__()
        # Hamming-windowed sinc, cutoff 20 Hz at 200 Hz, applied causally before stride-4 decimation.
        # 31 taps keep >0.99 gain up to 9 Hz (all ShakeBench excitation lines) and <0.01 above 30 Hz,
        # with half the delay of a 63-tap filter (75 ms instead of 155 ms).
        n = torch.arange(taps, dtype=torch.float32) - (taps - 1) / 2
        weights = 0.2 * torch.sinc(0.2 * n) * torch.hamming_window(taps, periodic=False)
        self.register_buffer("fir", (weights / weights.sum()).repeat(6, 1, 1))
        self.input = nn.Sequential(nn.Conv1d(6, 64, 1), nn.GELU(), nn.Conv1d(64, 128, 1))
        self.blocks = nn.Sequential(*(ResidualTCN(d) for d in [1, 2, 4, 8, 16, 32]))
        self.pool = AttentionPool(LONG_SAMPLES // LONG_STRIDE)

    def forward(self, x):
        """Map `[batch, 600, 6]` normalized samples to one long-history vector."""
        x = x.transpose(1, 2)
        taps = self.fir.shape[-1]
        x = F.conv1d(F.pad(x, (taps - 1, 0)), self.fir.to(x.dtype), groups=6)
        # Decimate so that the last kept output is the newest sample (indices 3, 7, ..., 599 for 600).
        x = x[..., (x.shape[-1] - 1) % LONG_STRIDE::LONG_STRIDE]
        return self.pool(self.blocks(self.input(x)).transpose(1, 2))


class IMUTokens(nn.Module):
    """Produce the two per-time IMU token slots consumed by the visual predictor."""

    def __init__(self, width, mode):
        """Build both encoders and projections while keeping token parameters mode-invariant."""
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
        """Return `[batch, time, 2, width]` short/long or learned no-IMU tokens.

        `eligible` and `dropped` decide which input windows are encoded. In v0,
        the long slot is always inactive; in none mode both slots stay no-IMU.
        """
        out = self.no_imu[None, None].expand(batch, time, -1, -1).clone()
        if short is None or self.mode == "none":
            return out
        if eligible is None:
            raise ValueError("IMU eligibility is required")
        # `active[b,t,stream]` means that sample b may encode this IMU stream at time t.
        active = eligible.clone()
        if dropped is not None:
            active &= ~dropped[:, None, None]
        if self.mode == "v0":
            active[..., 1] = False
        # Stream order matches token slots: 0=short history, 1=long history.
        for i, (values, encoder) in enumerate([(short, self.short), (long, self.long)]):
            selected = active[..., i]
            if selected.any():
                if values is None:
                    raise ValueError("eligible IMU window missing")
                out[:, :, i][selected] = self.project[i](encoder(values[selected])) + self.kind[i]
        return out
