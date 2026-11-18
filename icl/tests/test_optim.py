import inspect
import math

import pytest
import torch

from icl import optim
from icl.vit import ViT


def tiny():
    return ViT(5, None, image_size=32, patch=8, width=32, depth=2, mlp_dim=64, heads=2, init_seed=0)


def test_split_covers_every_parameter_once():
    m = tiny()
    parts = optim.split_params(m)
    names = [n for n, _ in parts["matrix"]] + [n for n, _ in parts["other"]]
    assert sorted(names) == sorted(n for n, _ in m.named_parameters()) and len(set(names)) == len(names)
    per_block = {"attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc1", "mlp.fc2"}
    want = {f"blocks.{i}.{n}.weight" for i in range(2) for n in per_block} | {"pre_logits.weight"}
    assert {n for n, _ in parts["matrix"]} == want
    assert all(p.ndim == 2 for _, p in parts["matrix"])
    other = {n for n, _ in parts["other"]}
    assert {"embedding.weight", "head.weight", "head.bias", "norm.weight", "blocks.0.norm1.bias"} <= other
    without = optim.split_params(m, muon_pre_logits=False)
    assert "pre_logits.weight" in {n for n, _ in without["other"]}
    assert len(without["matrix"]) == len(parts["matrix"]) - 1


def test_muon_takes_only_block_matrices_and_adamw_decays_only_weights():
    m = tiny()
    o = optim.build_optimizer(m, "muon", lr=1e-3, weight_decay=0.1)
    muon, adamw = o.optimizers
    assert type(muon).__name__ == "Muon" and type(adamw).__name__ == "AdamW"
    ids = {id(p) for g in muon.param_groups for p in g["params"]}
    assert ids == {id(p) for _, p in optim.split_params(m)["matrix"]}
    decay, rest = adamw.param_groups
    assert decay["weight_decay"] == 0.1 and all(p.ndim >= 2 for p in decay["params"])
    assert rest["weight_decay"] == 0.0 and all(p.ndim == 1 for p in rest["params"])
    assert {id(p) for p in decay["params"]} == {id(m.embedding.weight), id(m.head.weight)}
    every = ids | {id(p) for g in adamw.param_groups for p in g["params"]}
    assert every == {id(p) for p in m.parameters()}

    a = optim.build_optimizer(m, "adamw")
    (only,) = a.optimizers
    decay, rest = only.param_groups
    assert (only.defaults["betas"], only.defaults["eps"], decay["lr"], decay["weight_decay"]) == ((0.9, 0.999), 1e-8, 1e-3, 0.1)
    assert sum(len(g["params"]) for g in only.param_groups) == len(list(m.parameters()))
    assert all(p.ndim >= 2 for p in decay["params"]) and all(p.ndim == 1 for p in rest["params"])
    (sgd,) = optim.build_optimizer(m, "sgd", lr=0.1, weight_decay=1e-4).optimizers
    decay, rest = sgd.param_groups
    assert type(sgd).__name__ == "SGD" and sgd.defaults["momentum"] == 0.9 and sgd.defaults["nesterov"] is True
    assert (decay["lr"], decay["weight_decay"], rest["weight_decay"]) == (0.1, 1e-4, 0.0)
    assert sum(len(g["params"]) for g in sgd.param_groups) == len(list(m.parameters()))
    with pytest.raises(ValueError):
        optim.build_optimizer(m, "lion")


def test_builtin_muon_has_the_arguments_we_pass():
    params = inspect.signature(torch.optim.Muon.__init__).parameters
    assert all(a in params for a in optim.MUON_ARGS_TORCH_2_14)


def test_schedule_is_big_visions():
    total, warm = 1000, 100
    assert optim.schedule(0, total, warm) == 0.0
    assert optim.schedule(50, total, warm) == pytest.approx(0.5)
    assert optim.schedule(100, total, warm) == pytest.approx(1.0)
    assert optim.schedule(550, total, warm) == pytest.approx(0.5)
    assert optim.schedule(999, total, warm) == pytest.approx(0.5 * (1 + math.cos(math.pi * 899 / 900)))
    assert optim.schedule(1000, total, warm) == pytest.approx(0.0, abs=1e-12)
    assert optim.schedule(10, 20, 0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        optim.schedule(0, 10, 10)


@pytest.mark.parametrize("name", ["adamw", "muon", "sgd"])
def test_scale_reaches_every_group_and_zero_scale_changes_nothing(name):
    m = tiny()
    o = optim.build_optimizer(m, name, lr=2e-3, muon_lr=5e-3)
    before = [p.detach().clone() for p in m.parameters()]
    m(torch.randn(2, 3, 32, 32)).logsumexp(1).sum().backward()
    for p in m.parameters():                                         # the zero head hides most gradients
        p.grad = torch.randn_like(p)
    o.set_lr_scale(0.0)
    o.step()
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
    o.set_lr_scale(0.25)
    lrs = [g["lr"] for opt in o.optimizers for g in opt.param_groups]
    want = [5e-3 * 0.25, 2e-3 * 0.25, 2e-3 * 0.25] if name == "muon" else [2e-3 * 0.25] * 2
    assert lrs == pytest.approx(want)
    o.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
    o.zero_grad()
    assert all(p.grad is None for p in m.parameters())


def test_decay_per_step_is_lr_times_weight_decay_for_both():
    for name in ("adamw", "muon"):
        m = tiny()
        o = optim.build_optimizer(m, name, lr=1e-2, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
        w, b = m.blocks[0].mlp.fc1.weight, m.blocks[0].mlp.fc1.bias
        w0, b0 = w.detach().clone(), b.detach().clone()
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        o.set_lr_scale(0.5)
        o.step()
        torch.testing.assert_close(w, w0 * (1 - 0.5 * 1e-2 * 0.1), rtol=1e-6, atol=0)
        assert torch.equal(b, b0)


@pytest.mark.parametrize("name", ["adamw", "muon", "sgd"])
def test_state_round_trip(name):
    torch.manual_seed(0)
    m1, m2 = tiny(), tiny()
    o1, o2 = optim.build_optimizer(m1, name), optim.build_optimizer(m2, name)
    x = torch.randn(4, 3, 32, 32)
    for o, m in ((o1, m1),):
        for s in (0.5, 1.0):
            (m.features(x) ** 2).mean().backward()
            o.set_lr_scale(s)
            o.step()
            o.zero_grad()
    m2.load_state_dict(m1.state_dict())
    o2.load_state_dict(o1.state_dict())
    for o, m in ((o1, m1), (o2, m2)):
        (m.features(x) ** 2).mean().backward()
        o.set_lr_scale(0.7)
        o.step()
    assert all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))
    with pytest.raises(ValueError):
        optim.build_optimizer(tiny(), "adamw" if name == "muon" else "muon").load_state_dict(o1.state_dict())
