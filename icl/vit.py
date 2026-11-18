"""A plain ViT, as in big_vision's `models/vit.py` under the config `vit_s16_i1k.py` (see RECIPE.md).

What follows the source: patch embedding by a strided convolution, fixed 2D sin-cos position embedding, pre-LN blocks
(LayerNorm eps 1e-6, tanh-approximated GELU, no dropout), a final LayerNorm, global average pooling, a `pre_logits`
layer (Linear + tanh, the config's `rep_size=True`) and a zero-initialised linear head.

What is ours: the attention keeps four separate d x d matrices (`q`, `k`, `v`, `proj`) instead of one stacked 3d x d
matrix, so that Muon orthogonalises each on its own.  flax's attention also holds four kernels and initialises each as
a d x d matrix, so the initial distribution is the source's.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F
from torch import nn

# name: (width, depth, mlp_dim, heads); big_vision `decode_variant`
PRESETS = {"Ti": (192, 12, 768, 3), "S": (384, 12, 1536, 6), "B": (768, 12, 3072, 12)}

_TRUNC_STD = 0.87962566103423978        # std of a standard normal truncated to (-2, 2); jax `variance_scaling`


def posemb_sincos_2d(h: int, w: int, width: int, temperature: float = 10_000.0) -> torch.Tensor:
    """big_vision `posemb_sincos_2d`: (1, h*w, width), tokens in row-major order."""
    if width % 4:
        raise ValueError("width must be a multiple of 4 for the sin-cos position embedding")
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    omega = torch.arange(width // 4, dtype=torch.float64) / (width // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1).float()[None]


def _lecun_normal_(w: torch.Tensor, fan_in: int, gen: torch.Generator | None) -> None:
    """flax's default kernel init: variance 1/fan_in, normal truncated at two (pre-correction) standard deviations."""
    std = math.sqrt(1.0 / fan_in) / _TRUNC_STD
    nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std, generator=gen)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(x)), split(self.v(x)))
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """`ViT(num_classes, variant="S/16")`, or the four sizes given explicitly (the tests use a very small one)."""

    def __init__(self, num_classes: int, variant: str | None = "S/16", *, image_size: int = 224, patch: int = 16,
                 width: int | None = None, depth: int | None = None, mlp_dim: int | None = None,
                 heads: int | None = None, rep_size: bool = True, init_seed: int | None = None):
        super().__init__()
        if variant is not None:
            name, _, p = variant.partition("/")
            w0, d0, m0, h0 = PRESETS[name]
            width, depth, mlp_dim, heads = width or w0, depth or d0, mlp_dim or m0, heads or h0
            patch = int(p) if p else patch
        if None in (width, depth, mlp_dim, heads):
            raise ValueError("give a variant or all of width, depth, mlp_dim, heads")
        if image_size % patch:
            raise ValueError("image_size must be a multiple of the patch size")
        self.num_classes, self.width, self.patch, self.image_size = num_classes, width, patch, image_size
        self.embedding = nn.Conv2d(3, width, patch, stride=patch)
        self.blocks = nn.ModuleList(Block(width, heads, mlp_dim) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pre_logits = nn.Linear(width, width) if rep_size else None
        self.head = nn.Linear(width, num_classes)
        g = image_size // patch
        self.register_buffer("posemb", posemb_sincos_2d(g, g, width), persistent=False)
        self.reset_parameters(init_seed)

    def reset_parameters(self, seed: int | None = None) -> None:
        """The source's initialisation.  With a seed, the result depends on that seed alone."""
        gen = None if seed is None else torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            _lecun_normal_(self.embedding.weight, self.embedding.weight[0].numel(), gen)
            nn.init.zeros_(self.embedding.bias)
            for blk in self.blocks:
                for lin in (blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.proj):
                    nn.init.xavier_uniform_(lin.weight, generator=gen)
                    nn.init.zeros_(lin.bias)
                for lin in (blk.mlp.fc1, blk.mlp.fc2):
                    nn.init.xavier_uniform_(lin.weight, generator=gen)
                    nn.init.normal_(lin.bias, std=1e-6, generator=gen)
                for ln in (blk.norm1, blk.norm2):
                    nn.init.ones_(ln.weight)
                    nn.init.zeros_(ln.bias)
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)
            if self.pre_logits is not None:
                _lecun_normal_(self.pre_logits.weight, self.width, gen)
                nn.init.zeros_(self.pre_logits.bias)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).flatten(2).transpose(1, 2)            # (B, h*w, width), row-major
        if x.shape[1] != self.posemb.shape[1]:
            raise ValueError(f"expected {self.image_size} px inputs, got a grid of {x.shape[1]} tokens")
        x = x + self.posemb.to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).mean(dim=1)                             # global average pooling

    def features(self, x: torch.Tensor, which: str = "pre_logits") -> torch.Tensor:
        """The pooled representation before the head: `pre_logits` (tanh of the extra layer, what the head reads;
        big_vision's few-shot evaluator reads the same layer) or `gap` (the pooled encoder output)."""
        gap = self._encode(x)
        if which == "gap" or self.pre_logits is None:
            return gap
        if which != "pre_logits":
            raise ValueError(which)
        return torch.tanh(self.pre_logits(gap))

    def both_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(gap, pre_logits) from one pass."""
        gap = self._encode(x)
        return gap, (gap if self.pre_logits is None else torch.tanh(self.pre_logits(gap)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def weights_sha256(model: nn.Module) -> str:
    """A hash of every parameter (name, shape, float32 bytes), to show that two runs start from the same weights."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()
