import math

import numpy as np
import torch
import torch.nn.functional as F

from icl.vit import ViT, posemb_sincos_2d, weights_sha256


def tiny(**kw):
    return ViT(5, None, image_size=32, patch=8, width=32, depth=2, mlp_dim=64, heads=2, **kw)


def test_s16_matches_the_published_size():
    assert sum(p.numel() for p in ViT(1000).parameters()) == 22_122_472          # pre_logits included
    assert sum(p.numel() for p in ViT(1000, rep_size=False).parameters()) == 22_122_472 - (384 * 384 + 384)
    m = ViT(1000)
    assert (m.width, len(m.blocks), m.blocks[0].attn.heads, m.blocks[0].mlp.fc1.out_features) == (384, 12, 6, 1536)
    assert m.posemb.shape == (1, 196, 384) and "posemb" not in m.state_dict()


def test_position_embedding_is_the_source_formula():
    h, w, width = 3, 5, 16
    y, x = np.mgrid[:h, :w]
    omega = 1.0 / (10_000.0 ** (np.arange(width // 4) / (width // 4 - 1)))
    yy, xx = np.einsum("m,d->md", y.flatten(), omega), np.einsum("m,d->md", x.flatten(), omega)
    want = np.concatenate([np.sin(xx), np.cos(xx), np.sin(yy), np.cos(yy)], axis=1)
    np.testing.assert_allclose(posemb_sincos_2d(h, w, width)[0].numpy(), want, atol=1e-6)


def test_attention_keeps_four_separate_square_matrices_and_computes_attention():
    m = tiny(init_seed=0)
    att = m.blocks[0].attn
    for name in ("q", "k", "v", "proj"):
        assert getattr(att, name).weight.shape == (32, 32)
    x = torch.randn(3, 7, 32)
    q, k, v = (getattr(att, n)(x).view(3, 7, 2, 16).transpose(1, 2) for n in ("q", "k", "v"))
    want = (torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(16), -1) @ v).transpose(1, 2).reshape(3, 7, 32)
    torch.testing.assert_close(att(x), att.proj(want), atol=1e-5, rtol=1e-5)


def test_initialisation_follows_the_source():
    m = ViT(10, "S/16", init_seed=3)
    blk = m.blocks[0]
    bound = math.sqrt(6.0 / (384 + 384))                                           # xavier uniform on a d x d matrix
    for lin in (blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.proj):
        assert lin.weight.abs().max() <= bound and lin.weight.abs().max() > 0.95 * bound
        assert torch.count_nonzero(lin.bias) == 0
    assert blk.mlp.fc1.weight.abs().max() <= math.sqrt(6.0 / (384 + 1536))
    assert 0 < blk.mlp.fc1.bias.abs().max() < 1e-4                                 # normal(1e-6)
    assert abs(m.embedding.weight.std().item() - math.sqrt(1 / 768)) < 0.05 * math.sqrt(1 / 768)
    assert abs(m.pre_logits.weight.std().item() - math.sqrt(1 / 384)) < 0.05 * math.sqrt(1 / 384)
    assert torch.count_nonzero(m.head.weight) == 0 and torch.count_nonzero(m.head.bias) == 0
    assert blk.norm1.eps == 1e-6 and m.norm.eps == 1e-6


def test_init_depends_on_the_seed_alone():
    torch.manual_seed(1)
    a = weights_sha256(tiny(init_seed=7))
    torch.manual_seed(2)
    torch.randn(5)
    assert weights_sha256(tiny(init_seed=7)) == a
    assert weights_sha256(tiny(init_seed=8)) != a


def test_forward_features_and_the_zero_head():
    m = tiny(init_seed=0)
    x = torch.randn(4, 3, 32, 32)
    logits = m(x)
    assert logits.shape == (4, 5) and torch.count_nonzero(logits) == 0
    assert abs(F.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).item() - math.log(5)) < 1e-6
    gap, pre = m.both_features(x)
    assert gap.shape == pre.shape == (4, 32)
    torch.testing.assert_close(m.features(x), pre)
    torch.testing.assert_close(m.features(x, "gap"), gap)
    torch.testing.assert_close(pre, torch.tanh(m.pre_logits(gap)))
    assert pre.abs().max() <= 1.0
    linear = tiny(init_seed=0, rep_size=False)
    torch.testing.assert_close(linear.features(x), linear.features(x, "gap"))
