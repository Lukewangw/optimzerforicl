"""icl/flops.py counts what torch's own FLOP counter counts, on a small ViT and on Newton-Schulz."""
import torch
from torch.optim._muon import _zeropower_via_newtonschulz
from torch.utils.flop_counter import FlopCounterMode

from icl import flops, vit

SMALL = dict(image_size=32, patch=8, width=48, depth=2, mlp_dim=96, heads=4)


def _count(fn) -> int:
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def _small(pre_logits=True):
    return vit.ViT(10, None, rep_size=pre_logits, init_seed=0, **SMALL)


def _parts(pre_logits=True):
    return flops.forward_parts(SMALL["width"], SMALL["depth"], SMALL["mlp_dim"], SMALL["image_size"],
                               SMALL["patch"], 10, pre_logits)


def test_forward_matches_torch():
    for pre in (True, False):
        m, x = _small(pre), torch.randn(3, 3, 32, 32)
        with torch.no_grad():
            got = _count(lambda: m(x))
        assert got == 3 * sum(_parts(pre).values())


def test_training_matches_torch():
    m, x = _small(), torch.randn(2, 3, 32, 32)
    got = _count(lambda: m(x).sum().backward())
    # torch counts the attention's backward as five products (it recomputes the scores); model FLOPs count four
    assert got == 2 * (flops.train_per_image(_parts()) + 0.5 * _parts()["attention products"])


def test_newton_schulz_matches_torch():
    for shape in ((48, 48), (48, 96), (96, 48)):
        g = torch.randn(*shape)
        got = _count(lambda: _zeropower_via_newtonschulz(g, (3.4445, -4.7750, 2.0315), 5, 1e-7))
        assert got == flops.ns_flops(shape)


def test_muon_shapes_follow_the_split():
    rj = {"num_classes": 10, "args": {"optimizer": "muon", "variant": None, **SMALL}}
    assert len(flops.muon_shapes(rj)) == 2 * 6 + 1
    rj["args"].update(muon_blocks="mlp", muon_skip_pre_logits=True)
    assert sorted(flops.muon_shapes(rj)) == [(48, 96)] * 2 + [(96, 48)] * 2
    rj["args"].update(muon_blocks="attn")
    assert flops.muon_shapes(rj) == [(48, 48)] * 8
    rj["args"]["optimizer"] = "adamw"
    assert flops.muon_shapes(rj) == []
