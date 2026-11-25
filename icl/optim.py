"""The optimizers under comparison, behind one small interface.  All are torch's own (`torch.optim`).

`adamw`: one AdamW over every parameter, the published recipe in PyTorch terms (RECIPE.md).
`muon`:  `torch.optim.Muon` on the hidden 2D weight matrices, AdamW (same settings as above) on everything else.
`sgd`:   SGD with Nesterov momentum over every parameter.  Its weight decay is torch's (added to the gradient), so
         at momentum m a weight shrinks by about `lr * weight_decay / (1 - m)` per step.

Weight decay follows big_vision's default mask (`.*/kernel$`): weights with two or more dimensions are decayed,
biases and LayerNorm parameters are not.  The schedule (linear warm-up times cosine) multiplies every group's
learning rate; in both optimizers the decay applied per step is `lr_t * weight_decay`, so it follows the schedule
too, as it does in big_vision.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

MUON_ARGS_TORCH_2_14 = ("lr", "weight_decay", "momentum", "nesterov", "ns_coefficients", "eps", "ns_steps",
                        "adjust_lr_fn")


def schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    """big_vision `create_learning_rate_schedule(decay_type="cosine", warmup_steps=...)` with base 1.
    `step` counts the updates already made, so the first update is taken at 0."""
    if total_steps <= warmup_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")
    progress = min(max((step - warmup_steps) / float(total_steps - warmup_steps), 0.0), 1.0)
    s = 0.5 * (1.0 + math.cos(math.pi * progress))
    if warmup_steps:
        s *= min(1.0, step / warmup_steps)
    return s


MUON_BLOCKS = ("all", "attn", "mlp", "none")   # the block matrices a Muon run gives Muon: all, the attention's, the
                                               # MLP's, or none (Muon then has the pre_logits matrix alone)


def split_params(model: nn.Module, muon_pre_logits: bool = True,
                 muon_blocks: str = "all") -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Every parameter of the model, each in exactly one list.

    matrix: the 2D weight matrices of the transformer blocks (q, k, v, proj, fc1, fc2; with `muon_blocks` "attn"
            only the attention's q, k, v and proj, with "mlp" only the MLP's fc1 and fc2, with "none" none) and, unless
            `muon_pre_logits` is false, the hidden `pre_logits` matrix between the pooling and the head.
    other:  the patch-embedding weight, the head, every bias, every LayerNorm parameter, and the block matrices
            that `muon_blocks` leaves out.
    """
    if muon_blocks not in MUON_BLOCKS:
        raise ValueError(f"muon_blocks must be one of {', '.join(MUON_BLOCKS)}, not {muon_blocks!r}")
    matrix, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        block = name.startswith("blocks.") and (muon_blocks == "all" or f".{muon_blocks}." in name)
        hidden = block or (muon_pre_logits and name.startswith("pre_logits."))
        (matrix if hidden and p.ndim == 2 and name.endswith(".weight") else other).append((name, p))
    return {"matrix": matrix, "other": other}


def _decay_groups(named: list[tuple[str, nn.Parameter]], lr: float, wd: float) -> list[dict]:
    decay = [p for _, p in named if p.ndim >= 2]
    rest = [p for _, p in named if p.ndim < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": wd})
    if rest:
        groups.append({"params": rest, "lr": lr, "weight_decay": 0.0})
    return groups


class Optim:
    """One or two torch optimizers stepped together under one learning-rate scale."""

    def __init__(self, name: str, optimizers: list[torch.optim.Optimizer]):
        self.name, self.optimizers = name, optimizers
        self.base_lrs = [[g["lr"] for g in o.param_groups] for o in optimizers]
        self.scale = 1.0

    def set_lr_scale(self, s: float) -> None:
        self.scale = float(s)
        for o, bases in zip(self.optimizers, self.base_lrs):
            for g, base in zip(o.param_groups, bases):
                g["lr"] = base * self.scale

    def step(self) -> None:
        for o in self.optimizers:
            o.step()

    def zero_grad(self) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=True)

    def state_dict(self) -> dict:
        return {"name": self.name, "scale": self.scale, "base_lrs": self.base_lrs,
                "optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state: dict) -> None:
        if state["name"] != self.name or len(state["optimizers"]) != len(self.optimizers):
            raise ValueError(f"optimizer state is for '{state['name']}', this run uses '{self.name}'")
        for o, s in zip(self.optimizers, state["optimizers"]):
            o.load_state_dict(copy.deepcopy(s))      # torch keeps the tensors it is given; never share them
        self.base_lrs = state["base_lrs"]
        self.set_lr_scale(state["scale"])

    def describe(self) -> list[dict]:
        out = []
        for o, bases in zip(self.optimizers, self.base_lrs):
            for g, base in zip(o.param_groups, bases):
                out.append({"optimizer": type(o).__name__, "tensors": len(g["params"]),
                            "parameters": sum(p.numel() for p in g["params"]), "lr": base,
                            "weight_decay": g["weight_decay"]})
        return out


def build_optimizer(model: nn.Module, name: str, *, lr: float = 1e-3, weight_decay: float = 0.1,
                    betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                    muon_lr: float | None = None, muon_weight_decay: float | None = None,
                    muon_momentum: float = 0.95, muon_nesterov: bool = True, muon_ns_steps: int = 5,
                    adjust_lr_fn: str | None = "match_rms_adamw", muon_pre_logits: bool = True,
                    muon_blocks: str = "all", sgd_momentum: float = 0.9, sgd_nesterov: bool = True) -> Optim:
    """`lr`, `weight_decay`, `betas`, `eps` are AdamW's (defaults: the recipe).  In a `muon` run they govern the
    parameters Muon does not take; `muon_lr` and `muon_weight_decay` default to the same two numbers;
    `muon_pre_logits` and `muon_blocks` say which matrices Muon takes (split_params).  In an `sgd` run `lr` and
    `weight_decay` are SGD's."""
    parts = split_params(model, muon_pre_logits, muon_blocks)
    if name == "sgd":
        groups = _decay_groups(parts["matrix"] + parts["other"], lr, weight_decay)
        return Optim(name, [torch.optim.SGD(groups, lr=lr, momentum=sgd_momentum, nesterov=sgd_nesterov and sgd_momentum > 0,
                                            weight_decay=weight_decay)])
    if name == "adamw":
        groups = _decay_groups(parts["matrix"] + parts["other"], lr, weight_decay)
        return Optim(name, [torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)])
    if name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(f"torch {torch.__version__} has no torch.optim.Muon")
        if not parts["matrix"]:
            raise ValueError("the model has no hidden 2D matrices for Muon")
        muon = torch.optim.Muon(
            [p for _, p in parts["matrix"]], lr=lr if muon_lr is None else muon_lr,
            weight_decay=weight_decay if muon_weight_decay is None else muon_weight_decay,
            momentum=muon_momentum, nesterov=muon_nesterov, ns_steps=muon_ns_steps, adjust_lr_fn=adjust_lr_fn)
        adamw = torch.optim.AdamW(_decay_groups(parts["other"], lr, weight_decay), lr=lr, betas=betas, eps=eps,
                                  weight_decay=weight_decay)
        return Optim(name, [muon, adamw])
    raise ValueError(f"unknown optimizer '{name}' (adamw, muon or sgd)")
