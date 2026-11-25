"""Compute: how many floating-point operations (FLOPs) the runs spent, and what each optimizer bought per FLOP and per
GPU-hour.  Nothing is trained or run on a GPU; everything is counted from the model's shape and read from the logs.

    python -m icl.flops --runs RUN [RUN ...] --out DIR

  model FLOPs     a multiply-add counts 2.  Forward: the patch embedding, q, k, v, proj, the two attention products
                  (scores and their weighted sum), fc1, fc2, `pre_logits` and the head; LayerNorm, GELU, softmax,
                  biases and the optimizer's elementwise work are left out (well under 1 % here).  A training image
                  costs the forward three times over (the backward is twice the forward), except the patch
                  embedding, whose input needs no gradient (twice).  Recomputation inside flash attention is not
                  counted, as is usual for model FLOPs.
  Muon FLOPs      Newton-Schulz on every matrix Muon trains, as torch.optim.Muon does it: per iteration X Xᵀ,
                  b A + c A A and a X + B X, so 4 m² n + 2 m³ for an m × n matrix (m the shorter side), 5 iterations.
                  Under DistributedDataParallel every GPU runs the whole optimizer, so a step costs that times the
                  number of GPUs.
  GPU-hours       steps times the measured time of a step (the median of the logged images per second), times the
                  GPUs, as summary.py and seeds.py count them; evaluations at checkpoints are left out.
  MFU             model FLOPs done per second per GPU over the A100's dense bf16 peak (312 TFLOP/s).

Writes DIR/flops.md (the tables) and DIR/flops.json (the numbers and the curves, for a page or another figure).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from icl import optim, vit, why

A100_BF16 = 312e12
NS_STEPS = 5


def forward_parts(width: int, depth: int, mlp_dim: int, image_size: int, patch: int, num_classes: int,
                  pre_logits: bool = True) -> dict[str, float]:
    """Forward FLOPs of one image, part by part; the block parts are the sum over all blocks."""
    n, d = (image_size // patch) ** 2, width
    return {"patch embedding": 2.0 * n * 3 * patch * patch * d,
            "q, k, v, proj": depth * 4 * 2.0 * n * d * d,
            "attention products": depth * 2 * 2.0 * n * n * d,
            "fc1, fc2": depth * 2 * 2.0 * n * d * mlp_dim,
            "pre_logits": 2.0 * d * d if pre_logits else 0.0,
            "head": 2.0 * d * num_classes}


def train_per_image(parts: dict[str, float]) -> float:
    """FLOPs of one training image: forward and backward."""
    return sum(3 * v for v in parts.values()) - parts["patch embedding"]


def ns_flops(shape: tuple[int, int], steps: int = NS_STEPS) -> float:
    """Newton-Schulz on one matrix, as torch.optim.Muon runs it."""
    m, n = sorted(shape)
    return steps * (4.0 * m * m * n + 2.0 * m ** 3)


def _model(run_json: dict) -> vit.ViT:
    a = run_json["args"]
    return vit.ViT(run_json["num_classes"], a.get("variant", "S/16"), image_size=a.get("image_size", 224),
                   width=a.get("width"), depth=a.get("depth"), mlp_dim=a.get("mlp_dim"), heads=a.get("heads"),
                   rep_size=not a.get("no_pre_logits"))


def muon_shapes(run_json: dict) -> list[tuple[int, int]]:
    """The shapes of the matrices Muon trains in a run (none for AdamW or SGD), from the same split as training."""
    a = run_json["args"]
    if a["optimizer"] != "muon":
        return []
    split = optim.split_params(_model(run_json), muon_pre_logits=not a.get("muon_skip_pre_logits"),
                               muon_blocks=a.get("muon_blocks") or "all")
    return [tuple(p.shape) for _, p in split["matrix"]]


def per_block(width: int, mlp_dim: int, tokens: int) -> dict[str, float]:
    """Forward FLOPs of one block of one image, by kind of matrix, and Newton-Schulz FLOPs of its matrices."""
    d = width
    return {"attention matrices": 4 * 2.0 * tokens * d * d, "attention products": 2 * 2.0 * tokens * tokens * d,
            "MLP matrices": 2 * 2.0 * tokens * d * mlp_dim,
            "Newton-Schulz, attention matrices": 4 * ns_flops((d, d)),
            "Newton-Schulz, MLP matrices": 2 * ns_flops((d, mlp_dim))}


def measure(path: str) -> dict:
    """One run: its FLOPs, speed, GPU-hours and curves against both."""
    with open(os.path.join(path, "run.json")) as f:
        rj = json.load(f)
    run = why.load(path)
    a, m = rj["args"], _model(rj)
    parts = forward_parts(m.width, len(m.blocks), m.blocks[0].mlp.fc1.out_features, m.image_size, m.patch,
                          rj["num_classes"], m.pre_logits is not None)
    world, batch = rj["world"], a["batch_size"]
    model_step = batch * train_per_image(parts)
    shapes = muon_shapes(rj)
    opt_step = world * sum(ns_flops(s) for s in shapes)
    speed = sorted(t["images_per_s"] for t in run["train"][2:] if t.get("images_per_s"))
    ips = float(np.median(speed)) if speed else None
    gpu_h_step = batch / ips * world / 3600.0 if ips else None
    probe = {c["step"]: c for c in (run["probe"] or {}).get("checkpoints", [])}
    curve = []
    for c in run["ckpt"]:
        s = c["step"]
        pk = probe.get(s)
        row = {"step": s, "eflops": s * (model_step + opt_step) / 1e18,
               "gpu_hours": s * gpu_h_step if gpu_h_step else None, "val": 100 * c["val_top1"]}
        for layer in ("gap", "pre_logits"):
            v = why.acc(run, pk, "white", layer, "unseen") if pk and run["probe"] else None
            row[f"unseen_{layer}"] = None if v is None else 100 * v
        curve.append(row)
    return {"name": os.path.basename(path.rstrip("/")), "setting": run["setting"], "optimizer": a["optimizer"],
            "lr": a["lr"], "epochs": a["epochs"], "seed": a["init_seed"], "variant": run["variant"],
            "muon_blocks": a.get("muon_blocks") or ("all" if a["optimizer"] == "muon" else None),
            "steps": rj["total_steps"], "batch": batch, "gpus": world, "forward_per_image": sum(parts.values()),
            "train_per_image": train_per_image(parts), "model_per_step": model_step, "muon_per_step": opt_step,
            "muon_matrices": len(shapes), "images_per_s": ips,
            "eflops": rj["total_steps"] * (model_step + opt_step) / 1e18,
            "muon_eflops": rj["total_steps"] * opt_step / 1e18,
            "gpu_hours": rj["total_steps"] * gpu_h_step if gpu_h_step else None,
            "mfu": ips / world * train_per_image(parts) / A100_BF16 if ips else None,
            "final_val": 100 * run["ckpt"][-1]["val_top1"] if run["ckpt"] else None, "curve": curve,
            "parts": parts, "shape": (m.width, m.blocks[0].mlp.fc1.out_features, (m.image_size // m.patch) ** 2)}


def mean_curve(runs: list[dict], key: str) -> list[float | None]:
    """The mean over seeds at each checkpoint (runs of one setting share their checkpoint steps)."""
    out = []
    for rows in zip(*(r["curve"] for r in runs)):
        v = [x[key] for x in rows if x[key] is not None]
        out.append(float(np.mean(v)) if len(v) == len(rows) else None)
    return out


def reached(runs: list[dict], key: str, level: float, x: str) -> float | None:
    """Where the setting's mean curve of `key` first reaches `level`, in units of `x` (eflops, gpu_hours)."""
    ys, xs = mean_curve(runs, key), mean_curve(runs, x)
    pts = [(a, b) for a, b in zip(ys, xs) if a is not None and b is not None]
    return why.at_level([p[0] for p in pts], [p[1] for p in pts], level) if pts else None


def label(r: dict) -> str:
    s = r["setting"]
    return s + (f", {r['epochs']} epochs" if r["epochs"] != 90 else "")


def tables(runs: list[dict], baseline: str) -> tuple[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(label(r), []).append(r)
    base = groups[baseline]
    b_val = float(np.mean([r["final_val"] for r in base]))
    b_fs = float(np.mean([r["curve"][-1]["unseen_pre_logits"] for r in base]))
    b_gpu = float(np.mean([r["gpu_hours"] for r in base]))
    r0 = runs[0]
    parts = r0["parts"]
    fwd = sum(parts.values())
    blk = per_block(*r0["shape"])
    md = ["# Compute: FLOPs and GPU-hours of the runs", "",
          "Counted from the model's shape and read from the logs (icl/flops.py); nothing was run for it.", "",
          "## One image through ViT-S/16 at 224 (196 patches)", "",
          "| part | forward GFLOPs | share |", "|---|---|---|"]
    md += [f"| {k} | {v / 1e9:.3f} | {100 * v / fwd:.1f} % |" for k, v in parts.items()]
    md += [f"| all | {fwd / 1e9:.3f} | |", "",
           f"Training one image (forward and backward): {r0['train_per_image'] / 1e9:.2f} GFLOPs; a step of "
           f"{r0['batch']} images: {r0['model_per_step'] / 1e12:.1f} TFLOPs; a 90-epoch run of "
           f"{r0['steps']} steps: {r0['steps'] * r0['model_per_step'] / 1e18:.2f} EFLOPs of model FLOPs.", "",
           "One block, one image, forward (GFLOPs): attention matrices "
           f"{blk['attention matrices'] / 1e9:.3f}, attention products {blk['attention products'] / 1e9:.3f}, "
           f"MLP matrices {blk['MLP matrices'] / 1e9:.3f}.  Newton-Schulz per step, one GPU, one block (GFLOPs): "
           f"attention matrices {blk['Newton-Schulz, attention matrices'] / 1e9:.2f}, MLP matrices "
           f"{blk['Newton-Schulz, MLP matrices'] / 1e9:.2f}.", "",
           "## Every run", "",
           "| run | steps | EFLOPs | of which Muon | images / s (4 GPUs) | GPU-hours, training | MFU | val top-1 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["optimizer"], r["lr"], r["epochs"], r["variant"], r["seed"])):
        md.append(f"| {r['name']} | {r['steps']} | {r['eflops']:.3f} | {100 * r['muon_eflops'] / r['eflops']:.2f} % | "
                  f"{r['images_per_s']:.0f} | {r['gpu_hours']:.2f} | {100 * r['mfu']:.1f} % | {r['final_val']:.1f} |")
    md += ["", f"## Each setting against {baseline}", "",
           f"Its final validation top-1 is {b_val:.1f} %, its final few-shot accuracy on the unseen pools at "
           f"`pre_logits` (whitened) {b_fs:.1f} %, a run {b_gpu:.2f} GPU-hours.  \"Reached\": where the mean "
           "curve of the setting's seeds first reaches that level, on the straight line between the two "
           "checkpoints around it, in the run's own schedule (a shorter run with its own decay can do better, "
           "as the 70-epoch run shows).", "",
           "| setting | runs | EFLOPs a run | Muon's share | GPU-hours a run | time against the baseline | "
           "final val top-1 | reached the baseline's val at (EFLOPs / GPU-hours) | final unseen, `pre_logits` | "
           "reached the baseline's unseen at (EFLOPs / GPU-hours) |", "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for name, rs in sorted(groups.items()):
        e = float(np.mean([r["eflops"] for r in rs]))
        g = float(np.mean([r["gpu_hours"] for r in rs]))
        share = float(np.mean([r["muon_eflops"] / r["eflops"] for r in rs]))
        v = float(np.mean([r["final_val"] for r in rs]))
        fsv = [r["curve"][-1]["unseen_pre_logits"] for r in rs]
        fs = float(np.mean(fsv)) if None not in fsv else None
        rv = (reached(rs, "val", b_val, "eflops"), reached(rs, "val", b_val, "gpu_hours"))
        rf = (reached(rs, "unseen_pre_logits", b_fs, "eflops"), reached(rs, "unseen_pre_logits", b_fs, "gpu_hours"))
        cell = lambda t: "–" if t[0] is None else f"{t[0]:.2f} / {t[1]:.1f}"
        md.append(f"| {name} | {len(rs)} | {e:.3f} | {100 * share:.2f} % | {g:.2f} | {100 * (g / b_gpu - 1):+.0f} % | "
                  f"{v:.1f} | {cell(rv)} | {'–' if fs is None else f'{fs:.1f}'} | {cell(rf)} |")
        summary[name] = {"runs": len(rs), "eflops": e, "gpu_hours": g, "muon_share": share, "final_val": v,
                         "final_unseen_pre_logits": fs, "val_reached": rv, "unseen_reached": rf,
                         "steps": [x["step"] for x in rs[0]["curve"]],
                         "curve": {k: mean_curve(rs, k) for k in ("eflops", "gpu_hours", "val", "unseen_gap",
                                                                  "unseen_pre_logits")}}
    return "\n".join(md) + "\n", {"baseline": baseline, "baseline_val": b_val, "baseline_unseen": b_fs,
                                  "baseline_gpu_hours": b_gpu, "parts": parts, "per_block": blk,
                                  "settings": summary}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="adamw lr 0.001")
    a = ap.parse_args(argv)
    runs = [measure(p) for p in a.runs]
    md, data = tables(runs, a.baseline)
    data["runs"] = [{k: v for k, v in r.items() if k not in ("curve", "parts", "shape")} for r in runs]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "flops.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.out, "flops.json"), "w") as f:
        json.dump(data, f, indent=1)
    print(f"flops: {len(runs)} runs, {len(data['settings'])} settings -> {a.out}")


if __name__ == "__main__":
    main()
