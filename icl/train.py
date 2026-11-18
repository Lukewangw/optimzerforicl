"""Train the ViT from scratch on the base classes of a pack, with AdamW or with Muon.

    python -m icl.train --train-pack PACK --val-pack PACK --roles roles.json --optimizer adamw --out RUN
    torchrun --nproc_per_node=4 -m icl.train ...                      # one node, four GPUs

Defaults are the published recipe (RECIPE.md).  Two runs given the same `--init-seed` and `--data-seed` start from
the same weights and see the same batches whatever `--optimizer` says; the run directory records the hash of the
initial weights and of the first batches so that this can be checked afterwards.

The run directory holds `run.json` (settings), `log.jsonl`, `ckpt-<step>.pt` at fixed fractions of training
(weights only) and `last.pt` (weights, optimizer state, step).  Starting again in the same directory resumes from
`last.pt`; to start over, use a new directory.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import aug, optim
from .data import Pack, Roles, eval_loader, train_loader
from .pack import cpus
from .vit import ViT, weights_sha256

CKPT_FRACTIONS = (0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-pack", required=True)
    ap.add_argument("--val-pack", help="base-class top-1 is measured on it at every checkpoint")
    ap.add_argument("--roles", help="roles.json; every class in it is left out of training")
    ap.add_argument("--out", required=True)
    g = ap.add_argument_group("model")
    g.add_argument("--variant", default="S/16")
    g.add_argument("--image-size", type=int, default=224)
    g.add_argument("--no-pre-logits", action="store_true", help="a plain linear head (the paper's ablation)")
    for name in ("width", "depth", "mlp-dim", "heads", "patch"):
        g.add_argument(f"--{name}", type=int, help="overrides the variant (used by the tests)")
    g = ap.add_argument_group("optimizer")
    g.add_argument("--optimizer", choices=("adamw", "muon", "sgd"), required=True)
    g.add_argument("--lr", type=float, default=1e-3, help="AdamW's (in a muon run: of what Muon does not take)")
    g.add_argument("--wd", type=float, default=0.1, help="AdamW's, in PyTorch terms (big_vision's 1e-4 / lr)")
    g.add_argument("--muon-lr", type=float, help="default: --lr")
    g.add_argument("--muon-wd", type=float, help="default: --wd")
    g.add_argument("--muon-momentum", type=float, default=0.95)
    g.add_argument("--muon-ns-steps", type=int, default=5)
    g.add_argument("--no-muon-nesterov", action="store_true")
    g.add_argument("--adjust-lr-fn", default="match_rms_adamw", choices=("match_rms_adamw", "original"))
    g.add_argument("--muon-skip-pre-logits", action="store_true", help="leave the pre_logits matrix to AdamW")
    g.add_argument("--sgd-momentum", type=float, default=0.9)
    g.add_argument("--no-sgd-nesterov", action="store_true")
    g.add_argument("--warmup-steps", type=int, default=10_000)
    g.add_argument("--grad-clip", type=float, default=1.0, help="global norm; 0 turns it off")
    g = ap.add_argument_group("data")
    g.add_argument("--epochs", type=float, default=90)
    g.add_argument("--batch-size", type=int, default=1024, help="global, over all processes")
    g.add_argument("--mixup", type=float, default=0.2, help="the Beta parameter; 0 turns mixup off")
    g.add_argument("--randaug-layers", type=int, default=2)
    g.add_argument("--randaug-magnitude", type=float, default=10)
    g.add_argument("--init-seed", type=int, default=0)
    g.add_argument("--data-seed", type=int, default=0)
    g.add_argument("--workers", type=int, help="loader workers per process (default: its CPUs minus two)")
    g = ap.add_argument_group("run")
    g.add_argument("--max-steps", type=int, default=0, help="stop after this many steps; the schedule is unchanged")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--save-every", type=int, default=2000, help="steps between writes of last.pt")
    g.add_argument("--eval-batch", type=int, default=500)
    g.add_argument("--device", help="cuda, mps or cpu (default: cuda if there is one, else cpu)")
    g.add_argument("--no-amp", action="store_true", help="no bf16 autocast on CUDA")
    return ap.parse_args(argv)


def checkpoint_steps(total_steps: int) -> list[int]:
    return sorted({int(round(f * total_steps)) for f in CKPT_FRACTIONS})


def model_kwargs(args, num_classes: int) -> dict:
    kw = {"num_classes": num_classes, "variant": args.variant, "image_size": args.image_size,
          "rep_size": not args.no_pre_logits}
    for name in ("width", "depth", "mlp_dim", "heads", "patch"):
        if getattr(args, name) is not None:
            kw[name] = getattr(args, name)
    if all(k in kw for k in ("width", "depth", "mlp_dim", "heads", "patch")):
        kw["variant"] = None
    return kw


def autocast(amp: bool):
    """bf16 autocast on CUDA, nothing elsewhere."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> float:
    """Top-1 over a loader of (uint8 images, labels)."""
    was_training = model.training
    model.eval()
    hit = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(amp):
            pred = model(aug.to_model_input(x)).argmax(1)
        hit += int((pred == y).sum())
        n += len(y)
    model.train(was_training)
    return hit / max(n, 1)


def _replica_spread(model, device) -> float:
    """Relative spread, across processes, of the sum of absolute weights: zero when the replicas agree."""
    s = torch.zeros((), dtype=torch.float64, device=device)
    for p in model.parameters():
        s += p.detach().double().abs().sum()
    lo, hi = s.clone(), s.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    return float((hi - lo) / hi.clamp(min=1e-30))


def main(argv=None) -> None:
    args = parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world > 1
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = device.type == "cuda" and not args.no_amp
    main_proc = rank == 0
    os.makedirs(args.out, exist_ok=True)

    train_pack = Pack(args.train_pack)
    roles = Roles.load(args.roles, train_pack.num_classes)
    num_classes = len(roles.base)
    n_train = int((roles.base_map[train_pack.labels] >= 0).sum())
    total_steps = int(round(args.epochs * n_train / args.batch_size))
    if total_steps < 2:
        raise SystemExit("train: fewer than two steps; check --epochs and --batch-size")
    warmup = args.warmup_steps if args.warmup_steps < total_steps else max(1, total_steps // 10)
    ckpt_steps = checkpoint_steps(total_steps)
    stop = min(total_steps, args.max_steps) if args.max_steps else total_steps

    kw = model_kwargs(args, num_classes)
    model = ViT(**kw, init_seed=args.init_seed)
    init_sha = weights_sha256(model)
    model.to(device)
    opt = optim.build_optimizer(
        model, args.optimizer, lr=args.lr, weight_decay=args.wd, muon_lr=args.muon_lr,
        muon_weight_decay=args.muon_wd, muon_momentum=args.muon_momentum, muon_nesterov=not args.no_muon_nesterov,
        muon_ns_steps=args.muon_ns_steps, adjust_lr_fn=args.adjust_lr_fn,
        muon_pre_logits=not args.muon_skip_pre_logits, sgd_momentum=args.sgd_momentum,
        sgd_nesterov=not args.no_sgd_nesterov)

    step, last = 0, os.path.join(args.out, "last.pt")
    resumed = os.path.exists(last)
    if not resumed and glob.glob(os.path.join(args.out, "ckpt-*.pt")):
        raise SystemExit(f"train: {args.out} holds checkpoints but no last.pt; use a new directory")
    if resumed:
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["init_sha"] != init_sha or state["total_steps"] != total_steps:
            raise SystemExit(f"train: {last} belongs to a different run (other seed, data or length)")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optim"])
        step = state["step"]
    net = model
    if distributed:
        ids = [torch.cuda.current_device()] if device.type == "cuda" else None
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=ids)

    log_f = open(os.path.join(args.out, "log.jsonl"), "a" if resumed else "w") if main_proc else None

    def log(**rec) -> None:
        if log_f:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    if main_proc:
        commit = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COMMIT")
        info = {"args": vars(args), "model": kw, "num_classes": num_classes, "n_train": n_train,
                "held_out": {k: len(v) for k, v in roles.groups.items()}, "total_steps": total_steps,
                "warmup_steps": warmup, "ckpt_steps": ckpt_steps, "init_sha": init_sha, "world": world,
                "device": str(device), "amp_bf16": amp, "torch": torch.__version__,
                "parameters": sum(p.numel() for p in model.parameters()), "optimizer_groups": opt.describe(),
                "code": open(commit).read().strip() if os.path.exists(commit) else None}
        if not resumed:
            with open(os.path.join(args.out, "run.json"), "w") as f:
                json.dump(info, f, indent=1)
        log(event="resume" if resumed else "start", step=step, init_sha=init_sha, total_steps=total_steps,
            n_train=n_train, num_classes=num_classes, world=world)
        print(f"train: {args.optimizer}, {num_classes} classes, {n_train} images, {total_steps} steps "
              f"(warm-up {warmup}), {info['parameters']} parameters, {world} process(es) on {device}, "
              f"init {init_sha[:12]}" + (f", resuming at step {step}" if resumed else ""), flush=True)

    workers = args.workers if args.workers is not None else max(0, cpus() // world - 2)
    val = None
    if main_proc and args.val_pack:
        val = eval_loader(Pack(args.val_pack), roles, "base", batch_size=args.eval_batch, workers=workers,
                          image_size=args.image_size, pin=device.type == "cuda")

    def save_last() -> None:
        tmp = last + ".tmp"
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "step": step, "init_sha": init_sha,
                    "total_steps": total_steps}, tmp)
        os.replace(tmp, last)

    def checkpoint() -> None:
        """Weights after `step` updates, with the base-class top-1 if there is a validation pack."""
        spread = _replica_spread(model, device) if distributed else 0.0
        if not main_proc:
            return
        name = f"ckpt-{step:07d}.pt"
        if os.path.exists(os.path.join(args.out, name)):
            return
        top1 = evaluate(model, val, device, amp) if val is not None else None
        save_last()                                  # first: a process killed here writes this checkpoint when resumed
        tmp = os.path.join(args.out, name + ".tmp")
        torch.save({"model": model.state_dict(), "step": step, "val_top1": top1, "model_kwargs": kw,
                    "init_sha": init_sha, "optimizer": args.optimizer, "total_steps": total_steps}, tmp)
        os.replace(tmp, os.path.join(args.out, name))
        log(event="ckpt", step=step, epoch=step * args.batch_size / n_train, val_top1=top1, file=name,
            replica_spread=spread)
        print(f"train: step {step}/{total_steps} checkpoint, val top-1 "
              f"{'n/a' if top1 is None else format(top1, '.4f')}", flush=True)

    loader = train_loader(train_pack, roles, batch_size=args.batch_size, data_seed=args.data_seed,
                          total_steps=stop, start_step=step, rank=rank, world=world, workers=workers,
                          image_size=args.image_size, randaug_layers=args.randaug_layers,
                          randaug_magnitude=args.randaug_magnitude, pin=device.type == "cuda")
    net.train()
    t_run = t_win = time.time()
    win_loss, win_steps, first_step = 0.0, 0, step
    for x, y in loader:
        if step in ckpt_steps:
            t0 = time.time()
            checkpoint()
            t_win += time.time() - t0                                # keep evaluation out of the speed figure
        if main_proc and not resumed and step < 3:
            log(event="batch", step=step, sha=hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest())
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        a = aug.mixup_coefficient(args.data_seed, step, args.mixup)
        x, y1, y2 = aug.mixup(aug.to_model_input(x), y, a)
        with autocast(amp):
            logits = net(x)
        logits = logits.float()
        loss = F.cross_entropy(logits, y1)
        if a < 1.0:
            loss = a * loss + (1.0 - a) * F.cross_entropy(logits, y2)
        loss_value = loss.item()
        if loss_value != loss_value or abs(loss_value) == float("inf"):
            log(event="diverged", step=step)
            raise SystemExit(f"train: the loss is not finite at step {step}")
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scale = optim.schedule(step, total_steps, warmup)
        opt.set_lr_scale(scale)
        opt.step()
        opt.zero_grad()
        step += 1
        win_loss += loss_value
        win_steps += 1
        if main_proc and (step % args.log_every == 0 or step == stop):
            now = time.time()
            log(event="train", step=step, epoch=step * args.batch_size / n_train, lr_scale=scale,
                loss=win_loss / win_steps, last_loss=loss_value,
                images_per_s=win_steps * args.batch_size / max(now - t_win, 1e-9))
            win_loss, win_steps, t_win = 0.0, 0, now
        if main_proc and step % args.save_every == 0 and step not in ckpt_steps:
            save_last()
    if step in ckpt_steps:
        checkpoint()
    elif main_proc:
        save_last()
    if main_proc:
        log(event="done" if step == total_steps else "stopped", step=step, seconds=time.time() - t_run,
            steps_this_process=step - first_step)
        print(f"train: {'finished' if step == total_steps else 'stopped'} at step {step}/{total_steps} "
              f"after {time.time() - t_run:.0f} s", flush=True)
        log_f.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
