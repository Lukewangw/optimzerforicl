"""What the layers of a checkpoint hold: few-shot accuracy by depth, the shape of the features, the last layer and
the weights, at every checkpoint of a run.  Nothing is trained.

    python -m icl.probe --run RUN --roles roles.json --heldout-pack TRAINPACK --base-pack VALPACK \
        [--extra dogs=TESTS:151-268:5,20,0 ...] [--train-pack TRAINPACK] [--keep DIR]

The images, the pools and the questions are fewshot.py's (same arguments, same seed), so the rows of `gap` and
`pre_logits` under the `plain` rule repeat RUN/fewshot.json.  The output is RUN/probe.json.

Layers, in the order an image passes them:
  embed              patch embedding plus position, averaged over the tokens
  block1 .. blockN   the token stream after each block, averaged over the tokens
  gap                the final LayerNorm, then the average: the encoder's output
  pre_act            the `pre_logits` matrix and bias applied to `gap`
  pre_logits         its tanh: what the head reads

Per checkpoint:
  fewshot   nearest class mean at every layer, under two rules.  `plain` is fewshot.py's: subtract the mean feature
            of the trained classes, scale to unit length.  `white` also whitens by the covariance of the trained
            classes' validation features before scaling.  An invertible linear map of a layer's features leaves
            `white` as it was and changes `plain`, so the two tell a change of metric from a loss of information.
  shape     per layer and set of images: how many directions the features use (participation ratio and entropy
            rank of the covariance spectrum, the share of the variance in its first directions) and how far apart
            the classes are (the share of the variance that lies between class means), under both rules.
  tanh      how far the units of `pre_logits` are driven into the flat ends of the tanh.
  val       top-1, top-5 and loss on every validation image of the trained classes, with the hits of each class;
            `train` is the same on a sample of training images, unaugmented.
  weights   for every weight matrix: size and rank measures of the matrix, of its change since the checkpoint
            before and of its change since the first checkpoint read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

from . import aug, features, fewshot
from .data import Pack, Roles
from .pack import cpus
from .train import autocast

RULES = ("plain", "white")
SHARES = (1, 2, 5, 10, 20, 50, 100, 200)             # the first k directions whose share of the variance is kept
KEPT_SPECTRA = ("pre_logits.weight", "head.weight")  # matrices whose singular values are written out in full


def layer_names(model) -> list[str]:
    return ["embed", *(f"block{i}" for i in range(1, len(model.blocks) + 1)), "gap", "pre_act", "pre_logits"]


def layers(model, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """({layer: (B, D)}, logits) for model inputs `x`: the model's own forward pass, read at every stage."""
    t = model.embedding(x).flatten(2).transpose(1, 2)
    t = t + model.posemb.to(t.dtype)
    out = {"embed": t.mean(dim=1)}
    for i, blk in enumerate(model.blocks, 1):
        t = blk(t)
        out[f"block{i}"] = t.mean(dim=1)
    out["gap"] = model.norm(t).mean(dim=1)
    out["pre_act"] = out["gap"] if model.pre_logits is None else model.pre_logits(out["gap"])
    out["pre_logits"] = out["gap"] if model.pre_logits is None else torch.tanh(out["pre_act"])
    return out, model.head(out["pre_logits"])


@torch.no_grad()
def extract(model, images_u8: torch.Tensor, device, *, batch_size: int = 500, amp: bool = False
            ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """float32 features of every layer, (M, D) each, and the logits (M, C), left on `device`."""
    model.eval()
    names = layer_names(model)
    feats, logits = {n: [] for n in names}, []
    for i in range(0, len(images_u8), batch_size):
        x = aug.to_model_input(images_u8[i:i + batch_size].to(device, non_blocking=True))
        with autocast(amp):
            f, z = layers(model, x)
        for n in names:
            feats[n].append(f[n].float())
        logits.append(z.float())
    return {n: torch.cat(v) for n, v in feats.items()}, torch.cat(logits)


# ---- the two rules ----------------------------------------------------------------------------------------------

def plain(f: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """fewshot.py's preparation, by its own arithmetic: minus the mean of `ref`, unit length."""
    return torch.from_numpy(fewshot.cl2n(f.cpu().numpy(), ref.cpu().numpy().mean(0))).to(f.device)


def whitened(f: torch.Tensor, ref: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Rows of `f` minus the mean of `ref`, whitened by the covariance of `ref`, scaled to unit length.  `ridge`, a
    share of the mean eigenvalue, is added to every eigenvalue, so that directions `ref` hardly uses are not blown
    up; up to that ridge the result is the same for `f` and for any invertible linear map of `f`."""
    r = ref.double()
    mu = r.mean(0)
    ev, vec = torch.linalg.eigh(torch.cov((r - mu).T))
    ev = ev.clamp_min(0.0)
    scale = (ev + ridge * ev.mean()).clamp_min(1e-300).rsqrt()
    x = (f.double() - mu) @ ((vec * scale) @ vec.T)
    return (x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)).float()


# ---- the shape of a layer's features ------------------------------------------------------------------------------

def _entropy_rank(shares: torch.Tensor) -> float:
    nz = shares[shares > 0]
    return float(torch.exp(-(nz * nz.log()).sum()))


def spectrum(f: torch.Tensor) -> dict:
    """How many directions the rows of `f` use, from the eigenvalues of their covariance: `pr` = (sum)^2 / (sum of
    squares), the participation ratio; `erank` = exp of the entropy of the eigenvalues taken as shares; `top`, the
    share of the variance in the first k directions for k in SHARES.  Both ranks are 1 for features on a line and
    D for features that fill D directions evenly."""
    x = f.double()
    x = x - x.mean(0)
    ev = torch.linalg.eigvalsh(x.T @ x / max(len(x) - 1, 1)).clamp_min(0.0).flip(0)
    total = ev.sum()
    if not total > 0:
        return {"trace": 0.0, "pr": 0.0, "erank": 0.0, "top": [0.0] * len(SHARES)}
    p = ev / total
    return {"trace": float(total), "pr": float(total ** 2 / ev.pow(2).sum()), "erank": _entropy_rank(p),
            "top": [float(p[:k].sum()) for k in SHARES]}


def between_share(f: torch.Tensor, labels: np.ndarray) -> float | None:
    """The share of the variance of the rows of `f` that lies between the class means: about (C - 1) / (M - 1) when
    the classes are not told apart at all, 1 when every class is a single point."""
    classes, inv = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        return None
    x = f.double()
    x = x - x.mean(0)
    total = x.pow(2).sum()
    if not total > 0:
        return None
    idx = torch.from_numpy(inv).to(x.device)
    sums = torch.zeros(len(classes), x.shape[1], dtype=x.dtype, device=x.device).index_add_(0, idx, x)
    counts = torch.bincount(idx, minlength=len(classes)).double()
    return float((sums.pow(2).sum(1) / counts).sum() / total)


def saturation(pre_act: torch.Tensor) -> dict:
    """How flat the tanh is where the units sit: the share of unit values beyond 0.9 and 0.99 in size, their mean
    size, and the root mean square of what goes into the tanh."""
    z = pre_act.double()
    a = torch.tanh(z).abs()
    return {"over90": float((a > 0.9).double().mean()), "over99": float((a > 0.99).double().mean()),
            "mean_abs": float(a.mean()), "rms_in": float(z.pow(2).mean().sqrt())}


def class_hits(logits: torch.Tensor, labels: np.ndarray, num_classes: int) -> dict:
    """Top-1, top-5 and cross-entropy of `logits` against `labels`, and the top-1 hits of each class."""
    y = torch.from_numpy(labels).to(logits.device)
    z = logits.double()
    top = z.topk(min(5, z.shape[1]), dim=1).indices
    hit = top[:, 0] == y
    return {"images": int(len(y)), "top1": float(hit.double().mean()),
            "top5": float((top == y[:, None]).any(dim=1).double().mean()),
            "loss": float(torch.nn.functional.cross_entropy(z, y)),
            "hits": torch.bincount(y[hit], minlength=num_classes).tolist()}


# ---- the weights ----------------------------------------------------------------------------------------------------

def matrices(state: dict) -> dict[str, torch.Tensor]:
    """The weight matrices of a checkpoint by name; the patch embedding as (width, 3 * patch * patch)."""
    return {k: v.reshape(v.shape[0], -1) for k, v in state.items() if k.endswith(".weight") and v.ndim >= 2}


def singular_values(w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(w.double())


def matrix_stats(w: torch.Tensor) -> dict:
    """Size and rank of a matrix from its singular values s: Frobenius and spectral norm, `srank` = sum s^2 / max
    s^2 (the stable rank), `erank` = exp of the entropy of s^2 taken as shares."""
    s2 = singular_values(w).pow(2)
    total = s2.sum()
    if not total > 0:
        return {"fro": 0.0, "spec": 0.0, "srank": 0.0, "erank": 0.0}
    return {"fro": float(total.sqrt()), "spec": float(s2.max().sqrt()), "srank": float(total / s2.max()),
            "erank": _entropy_rank(s2 / total)}


def weight_stats(now: dict, before: dict | None, first: dict | None, device) -> dict:
    """{matrix: its stats, with `since_last` and `since_first` (the stats of its change) where there is one}."""
    out = {}
    for name, w in now.items():
        w = w.to(device).double()
        row = matrix_stats(w)
        for key, ref in (("since_last", before), ("since_first", first)):
            if ref is not None:
                row[key] = matrix_stats(w - ref[name].to(device).double())
        out[name] = row
    return out


# ---- one run ----------------------------------------------------------------------------------------------------------

def distinct_sets(pools: dict) -> dict[str, tuple[slice, np.ndarray]]:
    """One entry per set of images the pools draw on (the pools of one --extra share theirs, and so do the seen
    pools): {name: (slice, labels)}, named after the first such pool without its `_<N>way` ending."""
    out, seen = {}, set()
    for name, (sl, labels, _) in pools.items():
        if (sl.start, sl.stop) not in seen:
            seen.add((sl.start, sl.stop))
            out[re.sub(r"_\d+way$", "", name)] = (sl, labels)
    return out


def kept_rows(sets: dict, names: list[str], per_set: int) -> np.ndarray:
    """Evenly spaced rows of the named sets, at most `per_set` of each."""
    rows = [sl.start + np.unique(np.linspace(0, sl.stop - sl.start - 1, min(per_set, sl.stop - sl.start)).round()
                                 .astype(np.int64))
            for sl, _ in (sets[n] for n in names if n in sets) if sl.stop > sl.start]
    return np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)


def _brief(res: dict, shots) -> dict:
    return {str(k): None if res[str(k)] is None else {"acc": res[str(k)]["acc"], "ci95": res[str(k)]["ci95"]}
            for k in shots}


def probe_checkpoint(model, state: dict, images: torch.Tensor, device, *, amp: bool, pools: dict, sets: dict,
                     ref: slice, asked: dict, shots, n_query: int, n_episodes: int, seed: int,
                     scored: dict[str, tuple[slice, np.ndarray]], ridge: float = 1e-3
                     ) -> tuple[dict, dict[str, torch.Tensor]]:
    """Everything above for one loaded checkpoint, and the features it was read from."""
    feats, logits = extract(model, images, device, amp=amp)
    res = {"fewshot": {r: {} for r in RULES}, "shape": {}}
    for layer, f in feats.items():
        prepared = {"plain": plain(f, f[ref]), "white": whitened(f, f[ref], ridge)}
        for rule, p in prepared.items():
            res["fewshot"][rule][layer] = {
                name: _brief(fewshot.evaluate_pool(p[sl], labels, name, n_way=n, shots=shots, n_query=n_query,
                                                   n_episodes=n_episodes, seed=seed, episodes=asked[name]), shots)
                for name, (sl, labels, n) in pools.items()}
        res["shape"][layer] = {s: {**spectrum(f[sl]), "between": between_share(prepared["plain"][sl], labels),
                                   "between_white": between_share(prepared["white"][sl], labels)}
                               for s, (sl, labels) in sets.items()}
    res["tanh"] = {s: saturation(feats["pre_act"][sl]) for s, (sl, _) in sets.items()}
    for name, (sl, labels) in scored.items():
        res[name] = class_hits(logits[sl], labels, model.num_classes)
    res["sv"] = {k: singular_values(state[k].to(device)).tolist() for k in KEPT_SPECTRA if k in state}
    return res, feats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run directory of train.py")
    ap.add_argument("--roles")
    ap.add_argument("--heldout-pack", nargs="*", default=[], help="pack(s) holding the held-out classes' images")
    ap.add_argument("--base-pack", help="validation images of the trained classes")
    ap.add_argument("--train-pack", help="training images; a sample of them is scored as `train`")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=PACK[:IDS[:WAYS]]")
    ap.add_argument("--shots", default="5")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seen-per-class", type=int, default=50)
    ap.add_argument("--train-per-class", type=int, default=40)
    ap.add_argument("--ridge", type=float, default=1e-3, help="of the `white` rule: the share of the mean eigenvalue "
                                                              "added to every eigenvalue before whitening")
    ap.add_argument("--steps", help="comma-separated checkpoint steps (default: all in the run)")
    ap.add_argument("--keep", metavar="DIR", help="write there the features of some images at the last checkpoint "
                                                  "read, every layer, for cka.py")
    ap.add_argument("--keep-sets", default="seen,inet")
    ap.add_argument("--keep-per-set", type=int, default=2000)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out", help="default: RUN/probe.json")
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    workers = args.workers if args.workers is not None else max(0, cpus() - 2)
    shots = [int(s) for s in args.shots.split(",")]
    ckpts = sorted(glob.glob(os.path.join(args.run, "ckpt-*.pt")))
    if args.steps:
        want = {int(s) for s in args.steps.split(",")}
        ckpts = [c for c in ckpts if int(os.path.basename(c)[5:-3]) in want]
    if not ckpts:
        raise SystemExit(f"probe: no checkpoints in {args.run}")
    with open(os.path.join(args.run, "run.json")) as f:
        run = json.load(f)
    image_size = run["model"]["image_size"]
    held = [Pack(p) for p in args.heldout_pack]
    base = Pack(args.base_pack) if args.base_pack else None
    extra = {name: (Pack(path), ids, ways) for name, path, ids, ways in map(fewshot.parse_extra, args.extra)}
    source = held[0] if held else base
    roles = Roles.load(args.roles, source.num_classes) if source is not None else Roles(run["num_classes"])
    if source is not None and len(roles.base) != run["num_classes"]:
        raise SystemExit(f"probe: the roles file leaves {len(roles.base)} base classes, the run was trained on "
                         f"{run['num_classes']}")

    t0 = time.time()
    pooled, pools, centre = fewshot.build_pools(roles, held, base, extra, image_size=image_size,
                                                per_class=args.per_class, seen_per_class=args.seen_per_class,
                                                workers=workers)
    ref = centre if centre is not None else slice(0, len(pooled))    # fewshot.py: the mean of all pooled images
    train_pack = Pack(args.train_pack) if args.train_pack else None
    chunks, scored, at = [pooled], {}, len(pooled)
    for name, pack, per_class in (("val", base, 0), ("train", train_pack, args.train_per_class)):
        if pack is not None:
            x, y = features.decode_pool(pack, roles, "base", image_size=image_size, per_class=per_class,
                                        workers=workers)
            chunks.append(x)
            scored[name] = (slice(at, at + len(x)), y)
            at += len(x)
    images = torch.cat(chunks)
    del chunks, pooled
    sets = distinct_sets(pools)
    print(f"probe: {len(images)} images, {len(pools)} pools on {len(sets)} sets, decoded in {time.time() - t0:.0f} s",
          flush=True)
    asked = {name: fewshot.pool_episodes(labels, name, n_way=n, shots=shots, n_query=args.query,
                                         n_episodes=args.episodes, seed=args.seed)
             for name, (sl, labels, n) in pools.items()}
    out = {"run": os.path.abspath(args.run), "optimizer": run["args"]["optimizer"], "shots": shots,
           "episodes": args.episodes, "query": args.query, "seed": args.seed, "rules": list(RULES),
           "ridge": args.ridge,
           "shares": list(SHARES), "amp_bf16": amp,
           "pools": {k: {"images": int(len(v[1])), "n_way": v[2]} for k, v in pools.items()},
           "sets": {s: {"images": int(len(y)), "classes": int(len(np.unique(y)))} for s, (_, y) in sets.items()},
           "scored": {s: {"images": int(len(y)), "per_class": np.bincount(y, minlength=run["num_classes"]).tolist()}
                      for s, (_, y) in scored.items()},
           "checkpoints": []}
    if train_pack is not None:                        # how many training images each trained class has
        counts = np.bincount(train_pack.labels, minlength=roles.num_classes)
        out["train_images_per_class"] = counts[roles.base].tolist()
    first = before = None
    for path in ckpts:
        t1 = time.time()
        model, ck = features.load_model(path, device)
        out.setdefault("layers", layer_names(model))
        res, feats = probe_checkpoint(model, ck["model"], images, device, amp=amp, pools=pools, sets=sets, ref=ref,
                                      asked=asked, shots=shots, n_query=args.query, n_episodes=args.episodes,
                                      seed=args.seed, scored=scored, ridge=args.ridge)
        now = matrices(ck["model"])
        res["weights"] = weight_stats(now, before, first, device)
        first, before = (now if first is None else first), now
        out["checkpoints"].append({"step": ck["step"], "file": os.path.basename(path), "val_top1": ck["val_top1"],
                                   **res})
        if args.keep and path == ckpts[-1]:
            rows = kept_rows(sets, args.keep_sets.split(","), args.keep_per_set)
            os.makedirs(args.keep, exist_ok=True)
            dst = os.path.join(args.keep, f"{os.path.basename(os.path.normpath(args.run))}-{ck['step']}.npz")
            pick = torch.from_numpy(rows).to(device)
            np.savez(dst, rows=rows, step=ck["step"], layers=np.array(out["layers"]),
                     **{n: f[pick].cpu().numpy() for n, f in feats.items()})
            print(f"probe: kept {len(rows)} rows of {len(feats)} layers -> {dst}", flush=True)
        k, pool = str(shots[-1]), next(iter(pools))
        r = res["fewshot"]["plain"]["pre_logits"][pool][k]
        print(f"probe: step {ck['step']}: {pool} {k}-shot {'n/a' if r is None else format(r['acc'], '.3f')}"
              f"{'' if 'val' not in res else ', val top-1 ' + format(res['val']['top1'], '.4f')}"
              f" ({time.time() - t1:.0f} s)", flush=True)
        del feats, model
    dst = args.out or os.path.join(args.run, "probe.json")
    with open(dst + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(dst + ".tmp", dst)
    print(f"probe: {len(ckpts)} checkpoints -> {dst} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
