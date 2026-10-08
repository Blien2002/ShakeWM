"""Block-causal transformer with bounded per-rollout K/V and explicit 3D RoPE."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .imu import IMUTokens


def block_causal_mask(query_blocks, key_blocks, tokens, offset=0, device=None):
    q = torch.arange(offset, offset + query_blocks, device=device).repeat_interleave(tokens)
    k = torch.arange(key_blocks, device=device).repeat_interleave(tokens)
    return k[None, :] <= q[:, None]


def rope_positions(blocks, grid, offset, device):
    tokens = grid * grid + 2
    t = torch.arange(offset, offset + blocks, device=device).repeat_interleave(tokens)
    row = torch.cat([torch.zeros(2, device=device), torch.arange(grid, device=device).repeat_interleave(grid)])
    col = torch.cat([torch.zeros(2, device=device), torch.arange(grid, device=device).repeat(grid)])
    return torch.stack([t, row.repeat(blocks), col.repeat(blocks)], -1)


def rope(x, positions):
    # Split even channels across time/row/column; IMU spatial positions are zero.
    dim = x.shape[-1]
    spatial = 2 * (dim // 6)
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
    layers: list
    blocks: int
    limit: int

    def detach(self):
        return KVCache([(k.detach(), v.detach()) for k, v in self.layers], self.blocks, self.limit)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, x, positions, old, tokens, reference=False):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope(q, positions), rope(k, positions)
        previous = 0 if old is None else old[0].shape[-2]
        if old is not None:
            k, v = torch.cat([old[0], k], -2), torch.cat([old[1], v], -2)
        # One block at a time: all 258 queries see all tokens in their block.
        # No quadratic full-sequence attention matrix or token-causal approximation.
        outputs = []
        for start in range(0, n, tokens):
            end = previous + start + tokens
            qb, kb, vb = q[..., start:start + tokens, :], k[..., :end, :], v[..., :end, :]
            if reference:
                score = qb.float() @ kb.float().transpose(-1, -2) / (d // self.heads) ** 0.5
                result = (score.softmax(-1) @ vb.float()).to(q.dtype)
            else:
                result = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False)
            outputs.append(result)
        out = torch.cat(outputs, -2).transpose(1, 2).reshape(b, n, d)
        return self.proj(out), (k, v)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.attention = Attention(config.width, config.heads)
        self.mlp = nn.Sequential(nn.Linear(config.width, config.width * config.mlp_ratio), nn.GELU(),
                                 nn.Linear(config.width * config.mlp_ratio, config.width))

    def forward(self, x, positions, old, tokens, reference=False):
        y, kv = self.attention(self.norm1(x), positions, old, tokens, reference)
        x = x + y
        return x + self.mlp(self.norm2(x)), kv


class ShakeWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual_in = nn.Linear(config.feature_dim, config.width)
        self.imu = IMUTokens(config.width, config.imu_mode)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.width)
        self.visual_out = nn.Linear(config.width, config.feature_dim)

    def forward(self, visual, short=None, long=None, eligible=None, dropped=None,
                cache=None, cache_limit=None, reference=False):
        b, t, p, d = visual.shape
        if (p, d) != (self.config.grid ** 2, self.config.feature_dim):
            raise ValueError("visual feature shape does not match model configuration")
        offset = 0 if cache is None else cache.blocks
        limit = cache_limit if cache is None else cache.limit
        if limit is None or offset + t > limit:
            raise ValueError("each call needs a bounded rollout cache (context+horizon-1)")
        imu = self.imu(b, t, short, long, eligible, dropped)
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
        x = self.norm(x).reshape(b, t, p + 2, -1)[:, :, 2:]
        return self.visual_out(x), KVCache(updated, offset + t, limit)

    @torch.no_grad()
    def rollout(self, history, short, long, eligible, horizon, dropped=None):
        # Targets and future observations cannot be passed to this API.
        predicted, cache = self(history, short, long, eligible, dropped,
                                cache_limit=history.shape[1] + horizon - 1)
        last = predicted[:, -1:]
        predictions = [last]
        for _ in range(1, horizon):
            last, cache = self(last, cache=cache)
            predictions.append(last)
        return torch.cat(predictions, 1)
