"""Block-causal transformer with bounded per-rollout K/V and explicit 3D RoPE."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .imu import IMUTokens


def block_causal_mask(query_blocks, key_blocks, tokens, offset=0, device=None):
    """Build a token mask where each query block sees its own and earlier key blocks.

    The returned boolean matrix is `[query_blocks*tokens, key_blocks*tokens]`;
    `offset` gives the absolute block index of the first query.
    """
    q = torch.arange(offset, offset + query_blocks, device=device).repeat_interleave(tokens)
    k = torch.arange(key_blocks, device=device).repeat_interleave(tokens)
    return k[None, :] <= q[:, None]


def rope_positions(blocks, grid, offset, device):
    """Create `[time, row, column]` coordinates for IMU slots and visual patches.

    Each visual block has `grid²` row-major patches plus two IMU positions whose
    row and column coordinates are zero.
    """
    tokens = grid * grid + 2
    t = torch.arange(offset, offset + blocks, device=device).repeat_interleave(tokens)
    row = torch.cat([torch.zeros(2, device=device), torch.arange(grid, device=device).repeat_interleave(grid)])
    col = torch.cat([torch.zeros(2, device=device), torch.arange(grid, device=device).repeat(grid)])
    return torch.stack([t, row.repeat(blocks), col.repeat(blocks)], -1)


def rope(x, positions):
    """Apply 3D rotary position embedding to time/row/column channel groups.

    `x` is `[batch, heads, tokens, head_dim]`; `positions` has one three-axis
    coordinate per token. The returned tensor keeps the input shape and dtype.
    """
    # Split even channels across time/row/column; IMU spatial positions are zero.
    dim = x.shape[-1]
    spatial = 2 * (dim // 6)
    # Give row and column equal even-sized groups; the remaining even channels encode time.
    sizes = [dim - 2 * spatial, spatial, spatial]
    chunks = []
    for part, pos, size in zip(x.split(sizes, -1), positions.T, sizes):
        freq = 10000 ** (-torch.arange(0, size, 2, device=x.device, dtype=torch.float32) / size)
        angle = pos.float()[:, None] * freq
        a, b = part.float().unflatten(-1, (-1, 2)).unbind(-1)
        chunks.append(torch.stack([a * angle.cos() - b * angle.sin(),
                                   a * angle.sin() + b * angle.cos()], -1).flatten(-2).to(x.dtype))
    return torch.cat(chunks, -1)


@dataclass
class KVCache:
    """Per-layer attention keys/values plus current and maximum block counts.

    `layers[i]` stores the accumulated `(key, value)` tensors for transformer
    layer i. `blocks` is the current visual-block count; `limit` bounds rollout.
    """

    layers: list
    blocks: int
    limit: int

    def detach(self):
        """Return the same cache metadata with every K/V tensor detached from autograd."""
        return KVCache([(k.detach(), v.detach()) for k, v in self.layers], self.blocks, self.limit)


class Attention(nn.Module):
    """Multi-head self-attention evaluated one complete time block at a time."""

    def __init__(self, width, heads):
        """Create the combined Q/K/V projection and output projection."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, x, positions, old, tokens, reference=False):
        """Attend each query block to its prefix K/V and return updated layer cache.

        `x` is `[batch, tokens, width]`; `old` is the optional cached `(K,V)` pair.
        The Q/K/V projections are queries, keys, and values for scaled dot-product attention.
        When `reference` is true, use explicit float32 attention for parity checks
        instead of PyTorch's scaled-dot-product kernel.
        """
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope(q, positions), rope(k, positions)
        # Number of cached tokens before this call; used to align each query block's prefix.
        previous = 0 if old is None else old[0].shape[-2]
        if old is not None:
            k, v = torch.cat([old[0], k], -2), torch.cat([old[1], v], -2)
        # One block at a time: all 258 queries see all tokens in their block.
        # No quadratic full-sequence attention matrix or token-causal approximation.
        outputs = []
        for start in range(0, n, tokens):
            # `end` includes old tokens and the current query block, but no future block.
            end = previous + start + tokens
            qb, kb, vb = q[..., start:start + tokens, :], k[..., :end, :], v[..., :end, :]
            if reference:
                score = qb.float() @ kb.float().transpose(-1, -2) / (d // self.heads) ** 0.5
                result = (score.softmax(-1) @ vb.float()).to(q.dtype)
            else:
                # The K/V slice is already causal, so SDPA needs no token-level causal mask.
                result = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False)
            outputs.append(result)
        out = torch.cat(outputs, -2).transpose(1, 2).reshape(b, n, d)
        return self.proj(out), (k, v)


class Block(nn.Module):
    """Pre-norm transformer block with residual attention and MLP sublayers."""

    def __init__(self, config):
        """Build normalization, attention, and the configured-width MLP."""
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.attention = Attention(config.width, config.heads)
        self.mlp = nn.Sequential(nn.Linear(config.width, config.width * config.mlp_ratio), nn.GELU(),
                                 nn.Linear(config.width * config.mlp_ratio, config.width))

    def forward(self, x, positions, old, tokens, reference=False):
        """Apply one transformer block and return its updated K/V cache."""
        y, kv = self.attention(self.norm1(x), positions, old, tokens, reference)
        x = x + y
        return x + self.mlp(self.norm2(x)), kv


class ShakeWM(nn.Module):
    """Predict future frozen visual features from visual history and delivered IMU."""

    def __init__(self, config):
        """Build feature projections, IMU tokenizers, transformer blocks, and output head."""
        super().__init__()
        self.config = config
        self.visual_in = nn.Linear(config.feature_dim, config.width)
        self.imu = IMUTokens(config.width, config.imu_mode)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.width)
        self.visual_out = nn.Linear(config.width, config.feature_dim)

    def forward(self, visual, short=None, long=None, eligible=None, dropped=None,
                cache=None, cache_limit=None, reference=False, future=False):
        """Predict features for input visual blocks and return an updated KV cache.

        Args:
            visual: `[B,T,grid²,feature_dim]` history features or prior predictions.
            short/long: optional normalized IMU windows `[B,T,20,6]` / `[B,T,600,6]`.
            eligible: boolean `[B,T,2]` availability for short and long streams.
            dropped: optional boolean `[B]` joint per-example IMU dropout mask.
            cache: K/V state from a previous call; `None` starts a prefill.
            cache_limit: maximum total visual blocks, required for a new cache.
            reference: use the explicit attention implementation for parity checks.
            future: rollout steps after the forecast origin. Both IMU slots get the learned
                future placeholder; IMU windows are rejected and a history cache is required.

        Returns `(predicted_features, cache)`; the prediction shape matches `visual`.
        """
        b, t, p, d = visual.shape
        if (p, d) != (self.config.grid ** 2, self.config.feature_dim):
            raise ValueError("visual feature shape does not match model configuration")
        offset = 0 if cache is None else cache.blocks
        limit = cache_limit if cache is None else cache.limit
        if limit is None or offset + t > limit:
            raise ValueError("each call needs a bounded rollout cache (context+horizon-1)")
        if future and cache is None:
            raise ValueError("future steps must continue a history cache")
        imu = self.imu(b, t, short, long, eligible, dropped, future)
        # Flatten in block order: short slot, long slot, then that block's visual patches.
        x = torch.cat([imu, self.visual_in(visual)], 2).flatten(1, 2)
        positions = rope_positions(t, self.config.grid, offset, x.device)
        updated = []
        for i, block in enumerate(self.blocks):
            old = None if cache is None else cache.layers[i]
            if self.training and self.config.activation_checkpoint and not reference:
                x, kv = checkpoint(block, x, positions, old, p + 2, use_reentrant=False)
            else:
                x, kv = block(x, positions, old, p + 2, reference)
            updated.append(kv)
        # IMU slots condition attention but are not forecast targets; return patch outputs only.
        x = self.norm(x).reshape(b, t, p + 2, -1)[:, :, 2:]
        return self.visual_out(x), KVCache(updated, offset + t, limit)

    @torch.no_grad()
    def rollout(self, history, short, long, eligible, horizon, dropped=None):
        """Generate a fixed-origin open-loop forecast without future sensor inputs.

        The history is prefetched once; each later step feeds the previous feature
        prediction back with the accumulated cache. Returns `[B,horizon,P,D]`.
        """
        # Targets and future observations cannot be passed to this API.
        predicted, cache = self(history, short, long, eligible, dropped,
                                cache_limit=history.shape[1] + horizon - 1)
        last = predicted[:, -1:]
        predictions = [last]
        for _ in range(1, horizon):
            last, cache = self(last, cache=cache, future=True)
            predictions.append(last)
        return torch.cat(predictions, 1)
