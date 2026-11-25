"""In-context learning proper: a small transformer reads examples and answers about one more image.

`fewshot.py` applies a fixed rule (nearest class mean) to frozen features.  Here the rule is learned.  A transformer
reads a set of (image feature, label) pairs and one more image feature, and has to name that image's label.  Which
label a class gets is drawn afresh for every sequence (a random one of N slots), so nothing about a class can be
stored in the weights: the only way to answer is to look at the examples.  It is trained on classes the encoder was
trained on and tested on classes nobody saw (dog breeds).  Two questions: does the rule it learns carry over to
unseen classes, and at which point of its own training does that ability appear, under each optimizer?

    # features of one frozen checkpoint, once (a GPU, a minute or two)
    python -m icl.seq features --ckpt RUN/ckpt-N.pt --roles ROLES --train-pack P/train --val-pack P/val \\
        --tests TESTPACK --out FEATS.npz
    # one sequence model
    python -m icl.seq train --features FEATS.npz --out SEQRUN --optimizer muon --lr 1e-3 --seed 0

A token is the sum of a projection of the feature and an embedding of its label slot; the image to be answered
carries a slot of its own ("?") and is read out through a linear head over the N slots.  There is no position
embedding: the examples are a set.  The blocks are the ViT's (`vit.Block`), so `optim.build_optimizer` splits the
parameters as it does there: Muon takes the blocks' matrices; AdamW the two embeddings, the head, biases and norms.

Runs that share `--seed` start from the same weights and are asked the same training sequences, whatever the
optimizer; the test sequences are the same for every run (`--eval-seed`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .optim import build_optimizer, schedule
from .vit import Block, _lecun_normal_, weights_sha256

KINDS = {"pre_logits": "pre", "gap": "gap"}
# Test pools cut from a pack numbered like ImageNet-1k (see cluster/launch.sh): name -> inclusive class ranges.
IMAGENET_POOLS = {"dogs": [(151, 268)], "wildcanids": [(269, 280)], "cats": [(281, 285)], "felines": [(281, 293)],
                  "birds": [(7, 24), (80, 100), (127, 146)], "inet": [(0, 999)]}


# ------------------------------------------------------------------------------------------------- features

def cmd_features(args) -> None:
    from . import aug, features
    from .data import Pack, Roles, eval_loader
    from .pack import cpus
    from .train import autocast
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    workers = args.workers if args.workers is not None else max(0, cpus() - 2)
    model, ck = features.load_model(args.ckpt, device)
    train = Pack(args.train_pack)
    roles = Roles.load(args.roles, train.num_classes)

    @torch.no_grad()
    def pool(pack, class_ids, per_class):
        """Features and pack class ids of up to `per_class` images of each class (0: all of them), read as a stream:
        the decoded images of a whole training set would not fit in memory."""
        ids = sorted(int(c) for c in class_ids)
        loader = eval_loader(pack, Roles(pack.num_classes, {"picked": ids}), "picked", batch_size=500, workers=workers,
                             image_size=model.image_size, per_class=per_class)
        gap, pre, ys = [], [], []
        for x, y in loader:
            x = aug.to_model_input(x.to(device, non_blocking=True))
            with autocast(device.type == "cuda"):
                g, p = model.both_features(x)
            gap.append(g.float().cpu().numpy().astype(np.float16))
            pre.append(p.float().cpu().numpy().astype(np.float16))
            ys.append(np.asarray(ids, dtype=np.int64)[y.numpy()])
        return np.concatenate(gap), np.concatenate(pre), np.concatenate(ys)

    t0 = time.time()
    out = {"step": ck["step"], "val_top1": -1.0 if ck["val_top1"] is None else ck["val_top1"],
           "base": np.asarray(roles.base, dtype=np.int64)}
    todo = [("train", train, roles.base, args.per_class)]
    if args.val_pack:
        todo.append(("val", Pack(args.val_pack), roles.base, 0))
    for name, ids in roles.groups.items():
        if len(ids) > 1:                                    # a held-out group of several classes, e.g. the controls
            todo.append((name, train, ids, args.per_class))
    if args.tests:
        tests = Pack(args.tests)
        todo.append(("tests", tests, range(tests.num_classes), 0))
    for name, pack, ids, per_class in todo:
        out[f"{name}_gap"], out[f"{name}_pre"], out[f"{name}_y"] = pool(pack, ids, per_class)
        print(f"seq features: {name}: {len(out[f'{name}_y'])} images of {len(set(out[f'{name}_y'].tolist()))} classes "
              f"({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, args.out)
    print(f"seq features: checkpoint step {ck['step']} -> {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")


# ------------------------------------------------------------------------------------------------- sequences

def distinct(count: torch.Tensor, m: int, gen: torch.Generator) -> torch.Tensor:
    """For every entry c of `count` (any shape, each at least m), m different integers below c, every set of m
    equally likely and in random order: (*count.shape, m).  Floyd's algorithm, so the cost does not grow with c
    (a class may have twelve thousand images)."""
    dev = count.device
    out = torch.zeros(*count.shape, m, dtype=torch.long, device=dev)
    for i in range(m):
        j = count - m + i                                               # this round draws from 0..j
        t = (torch.rand(count.shape, generator=gen, device=dev) * (j + 1)).long().clamp_(max=j)
        taken = (out[..., :i] == t[..., None]).any(-1)
        out[..., i] = torch.where(taken, j, t)
    order = torch.rand(*count.shape, m, generator=gen, device=dev).argsort(-1)
    return out.gather(-1, order)


class Pool:
    """Prepared features of a group of classes on one device, and where each class's images are."""

    def __init__(self, x: torch.Tensor, y: np.ndarray, min_images: int):
        classes, counts = np.unique(y, return_counts=True)
        keep = counts >= min_images
        self.classes, counts = classes[keep], counts[keep]
        if not len(self.classes):
            raise ValueError(f"no class has {min_images} images")
        self.x = x
        idx = torch.zeros((len(self.classes), int(counts.max())), dtype=torch.long)
        for i, c in enumerate(self.classes):
            pos = np.flatnonzero(y == c)
            idx[i, :len(pos)] = torch.from_numpy(pos)
        self.idx = idx.to(x.device)
        self.count = torch.from_numpy(counts).to(x.device)

    def draw(self, batch: int, n_way: int, k_shot: int, gen: torch.Generator):
        """`batch` sequences as positions in `x`: (examples (B, N*K), their label slots (B, N*K), the image asked
        about (B,), its slot (B,)).  Classes differ within a sequence, images differ within a class."""
        dev = self.x.device
        n_cls = self.idx.shape[0]
        if n_way > n_cls:
            raise ValueError(f"{n_way}-way sequences from {n_cls} classes")
        cls = torch.rand(batch, n_cls, generator=gen, device=dev).argsort(1)[:, :n_way]
        pick = distinct(self.count[cls], k_shot + 1, gen)
        img = self.idx[cls[:, :, None].expand_as(pick), pick]
        slot = torch.rand(batch, n_way, generator=gen, device=dev).argsort(1)
        asked = torch.randint(n_way, (batch,), generator=gen, device=dev)
        rows = torch.arange(batch, device=dev)
        return (img[:, :, :k_shot].reshape(batch, n_way * k_shot),
                slot[:, :, None].expand(batch, n_way, k_shot).reshape(batch, n_way * k_shot),
                img[rows, asked, k_shot], slot[rows, asked])

    def tokens(self, ctx, ctx_slot, qry, n_way: int):
        """The model's input for sequences from `draw`: features (B, T, D) and label slots (B, T), the image asked
        about last with the slot `n_way` ("?")."""
        feats = torch.cat([self.x[ctx], self.x[qry][:, None]], dim=1)
        slots = torch.cat([ctx_slot, torch.full((len(qry), 1), n_way, dtype=torch.long, device=ctx.device)], dim=1)
        return feats, slots


def nearest_mean(feats: torch.Tensor, slots: torch.Tensor, n_way: int) -> torch.Tensor:
    """The fixed rule on the same sequences: the slot whose examples' mean is nearest to the image asked about."""
    ctx, qry = feats[:, :-1], feats[:, -1]
    onehot = F.one_hot(slots[:, :-1], n_way).to(feats.dtype)
    means = torch.einsum("btn,btd->bnd", onehot, ctx) / onehot.sum(1)[:, :, None]
    return ((qry[:, None, :] - means) ** 2).sum(-1).argmin(1)


class SeqModel(nn.Module):
    def __init__(self, feat_dim: int, n_way: int, *, width: int = 256, depth: int = 4, heads: int = 4,
                 mlp_dim: int | None = None, init_seed: int | None = None):
        super().__init__()
        self.feat_dim, self.n_way, self.width = feat_dim, n_way, width
        self.embed = nn.Linear(feat_dim, width)
        self.label = nn.Embedding(n_way + 1, width)             # N label slots and "?"
        self.blocks = nn.ModuleList(Block(width, heads, mlp_dim or 4 * width) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.head = nn.Linear(width, n_way)
        self.reset_parameters(init_seed)

    def reset_parameters(self, seed: int | None = None) -> None:
        """As the ViT's: with a seed, the weights depend on that seed alone."""
        gen = None if seed is None else torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            _lecun_normal_(self.embed.weight, self.feat_dim, gen)
            nn.init.zeros_(self.embed.bias)
            nn.init.normal_(self.label.weight, std=1.0, generator=gen)
            for blk in self.blocks:
                for lin in (blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2):
                    nn.init.xavier_uniform_(lin.weight, generator=gen)
                    nn.init.zeros_(lin.bias)
                for ln in (blk.norm1, blk.norm2):
                    nn.init.ones_(ln.weight)
                    nn.init.zeros_(ln.bias)
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, feats: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        x = self.embed(feats) + self.label(slots)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, -1]))


def prepare(x: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Centre, scale to unit length (as the few-shot test does), then to unit variance per coordinate."""
    f = x.astype(np.float32) - centre
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    return f * math.sqrt(f.shape[1])


def load_pools(path: str, kind: str, device, min_images: int) -> tuple[dict[str, Pool], dict]:
    """The pools of a features file: `train` and `seen` (the trained classes: training and validation images), the
    held-out groups of the roles file, and the ranges of IMAGENET_POOLS cut from the test pack."""
    d = np.load(path)
    k = KINDS[kind]
    centre = d[f"{'val' if 'val_y' in d else 'train'}_{k}"].astype(np.float32).mean(0)

    def pool(name, mask=None):
        x, y = d[f"{name}_{k}"], d[f"{name}_y"]
        if mask is not None:
            x, y = x[mask], y[mask]
        return Pool(torch.from_numpy(prepare(x, centre)).to(device), y, min_images)

    pools = {"train": pool("train")}
    if "val_y" in d:
        pools["seen"] = pool("val")
    for key in d.files:
        name = key[:-2]
        if key.endswith("_y") and name not in ("train", "val", "tests"):
            pools[name] = pool(name)
    if "tests_y" in d:
        y = d["tests_y"]
        for name, ranges in IMAGENET_POOLS.items():
            mask = np.zeros(len(y), dtype=bool)
            for lo, hi in ranges:
                mask |= (y >= lo) & (y <= hi)
            if mask.any():
                pools[name] = pool("tests", mask)
    return pools, {"encoder_step": int(d["step"]), "encoder_val_top1": float(d["val_top1"]),
                   "feat_dim": int(d[f"train_{k}"].shape[1])}


def eval_steps(total: int, every: int) -> list[int]:
    """0, then 1-2-5 steps per decade from 10 (to see the start closely), every `every` steps, and the last."""
    steps = {0, total}
    decade = 10
    while decade < total:
        steps |= {s for s in (decade, 2 * decade, 5 * decade) if s < total}
        decade *= 10
    steps |= set(range(every, total, every))
    return sorted(steps)


def dense_steps(total: int) -> list[int]:
    """Where the ability appears: every 10 steps up to 500, every 50 up to 2,000."""
    return [s for s in list(range(0, 500, 10)) + list(range(500, 2001, 50)) if s <= total]


@torch.no_grad()
def evaluate(model: SeqModel, pool: Pool, questions, n_way: int, chunk: int = 1000) -> float:
    ctx, ctx_slot, qry, answer = questions
    model.eval()
    hits = 0
    for i in range(0, len(qry), chunk):
        feats, slots = pool.tokens(ctx[i:i + chunk], ctx_slot[i:i + chunk], qry[i:i + chunk], n_way)
        hits += int((model(feats, slots).argmax(1) == answer[i:i + chunk]).sum())
    model.train()
    return hits / len(qry)


def cmd_train(args) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    shots = sorted({int(s) for s in args.eval_shots.split(",")} | {args.shots})
    pools, meta = load_pools(args.features, args.feature, device, max(shots) + 1)
    if not args.seeds:
        train_one(args, pools, meta, shots, device)
        return
    out, seeds = args.out, [int(s) for s in args.seeds.split(",")]       # several runs: OUT-s0, OUT-s1, ...
    if args.parallel and len(seeds) > 1:            # side by side on the one GPU, which a model this small leaves idle
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=_one_seed, args=(vars(args), seed, f"{out}-s{seed}")) for seed in seeds]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        if any(p.exitcode for p in procs):
            raise SystemExit(f"seq: seeds {[s for s, p in zip(seeds, procs) if p.exitcode]} did not finish")
        return
    for seed in seeds:
        args.seed, args.out = seed, f"{out}-s{seed}"
        train_one(args, pools, meta, shots, device)


def _one_seed(args: dict, seed: int, out: str) -> None:
    cmd_train(argparse.Namespace(**{**args, "seed": seed, "out": out, "seeds": None}))


def train_one(args, pools: dict[str, Pool], meta: dict, shots: list[int], device) -> None:
    model = SeqModel(meta["feat_dim"], args.ways, width=args.width, depth=args.depth, heads=args.heads,
                     init_seed=args.seed).to(device)
    opt = build_optimizer(model, args.optimizer, lr=args.lr, weight_decay=args.wd, muon_momentum=args.muon_momentum,
                          sgd_momentum=args.sgd_momentum)
    os.makedirs(args.out, exist_ok=True)

    # The same questions for every run: drawn once from --eval-seed, the fixed rule's answers kept as the yardstick.
    questions, yardstick = {}, {}
    for name, pool in pools.items():
        if len(pool.classes) < args.ways:
            continue
        for k in (shots if name != "train" else [args.shots]):
            gen = torch.Generator(device=device).manual_seed(args.eval_seed * 1000 + k)
            q = pool.draw(args.eval_sequences, args.ways, k, gen)
            key = name if k == args.shots else f"{name}@{k}"
            questions[key] = (pool, q)
            feats, slots = pool.tokens(q[0], q[1], q[2], args.ways)
            yardstick[key] = float((nearest_mean(feats, slots, args.ways) == q[3]).float().mean())
    run = {"args": vars(args), "optimizer": args.optimizer, **meta, "ways": args.ways, "shots": args.shots,
           "parameters": sum(p.numel() for p in model.parameters()), "init_sha": weights_sha256(model),
           "optimizer_groups": opt.describe(), "torch": torch.__version__, "device": str(device),
           "pools": {n: {"classes": int(len(p.classes)), "images": int(p.count.sum())} for n, p in pools.items()},
           "nearest_mean": yardstick, "chance": 1.0 / args.ways}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    when = set(eval_steps(args.steps, args.eval_every))
    watched = [k for k in args.watch.split(",") if k in questions]       # asked often at the start; all pools at `when`
    often = set(dense_steps(args.steps)) if watched else set()
    log = open(os.path.join(args.out, "log.jsonl"), "w")
    t0, window, hit = time.time(), [], []
    model.train()
    for step in range(args.steps + 1):
        if step in when or step in often:
            keys = list(questions) if step in when else watched
            rec = {"event": "eval", "step": step,
                   **{key: evaluate(model, questions[key][0], questions[key][1], args.ways) for key in keys}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step == args.steps:
            break
        opt.set_lr_scale(schedule(step, args.steps, args.warmup_steps))
        ctx, ctx_slot, qry, answer = pools["train"].draw(args.batch_size, args.ways, args.shots, gen)
        feats, slots = pools["train"].tokens(ctx, ctx_slot, qry, args.ways)
        logits = model(feats, slots)
        loss = F.cross_entropy(logits, answer)
        if not torch.isfinite(loss):
            log.write(json.dumps({"event": "diverged", "step": step}) + "\n")
            log.close()
            raise SystemExit(f"seq: the loss is not finite at step {step}")
        opt.zero_grad()
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        window.append(loss.detach())
        hit.append((logits.detach().argmax(1) == answer).float().mean())
        if (step + 1) % args.log_every == 0:
            log.write(json.dumps({"event": "train", "step": step + 1, "loss": float(torch.stack(window).mean()),
                                  "acc": float(torch.stack(hit).mean()), "lr_scale": opt.scale}) + "\n")
            window, hit = [], []
    log.close()
    torch.save({"model": model.state_dict(), "step": args.steps, "args": vars(args)}, os.path.join(args.out, "seq.pt"))
    last = {key: evaluate(model, pool, q, args.ways) for key, (pool, q) in questions.items()}
    shown = ", ".join(f"{k} {v:.3f} (fixed rule {yardstick[k]:.3f})" for k, v in last.items() if "@" not in k)
    print(f"seq: {args.optimizer} lr {args.lr:g} seed {args.seed}: {args.steps} steps in {time.time() - t0:.0f} s; {shown}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="features of one checkpoint for every pool")
    f.add_argument("--ckpt", required=True)
    f.add_argument("--roles", required=True)
    f.add_argument("--train-pack", required=True)
    f.add_argument("--val-pack")
    f.add_argument("--tests", help="a pack numbered like ImageNet-1k, for the unseen breeds")
    f.add_argument("--per-class", type=int, default=0, help="training images per class (0: all)")
    f.add_argument("--workers", type=int)
    f.add_argument("--device")
    f.add_argument("--out", required=True)
    t = sub.add_parser("train", help="one sequence model on a features file")
    t.add_argument("--features", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--feature", default="pre_logits", choices=sorted(KINDS))
    t.add_argument("--optimizer", default="adamw", choices=("adamw", "muon", "sgd"))
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.1)
    t.add_argument("--muon-momentum", type=float, default=0.95)
    t.add_argument("--sgd-momentum", type=float, default=0.9)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--steps", type=int, default=10000)
    t.add_argument("--warmup-steps", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--ways", type=int, default=5)
    t.add_argument("--shots", type=int, default=5)
    t.add_argument("--eval-shots", default="1,5,10")
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--seed", type=int, default=0, help="initial weights and training sequences")
    t.add_argument("--seeds", help="comma-separated: one run per seed, into OUT-s<seed>")
    t.add_argument("--parallel", action="store_true", help="with --seeds: the runs side by side, one process each")
    t.add_argument("--eval-seed", type=int, default=0)
    t.add_argument("--eval-sequences", type=int, default=5000)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--watch", default="seen,dogs,control",
                   help="pools also asked every 10 steps up to 500 and every 50 up to 2,000 ('' for none)")
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--device")
    args = ap.parse_args(argv)
    (cmd_features if args.cmd == "features" else cmd_train)(args)


if __name__ == "__main__":
    main()
