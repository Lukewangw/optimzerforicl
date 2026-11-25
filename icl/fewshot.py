"""Few-shot classification of held-out classes on frozen features, at every checkpoint of a run.

    python -m icl.fewshot --run RUN --roles roles.json --heldout-pack TRAINPACK --base-pack VALPACK \
        [--extra pets=PETPACK] --shots 1,5,10 --episodes 1000

An episode takes N classes, k labelled examples of each (the support) and some more images of each to classify
(the queries).  A query goes to the class whose mean support feature is nearest.  Before that every feature has
the mean base-class feature subtracted and is scaled to unit length (the "CL2N" of SimpleShot, arXiv 1911.04623).

Pools of images:
  one per group of the roles file, taken from `--heldout-pack` (N = the group's size);
  `seen_<N>way`: classes the model was trained on, images from `--base-pack`, N drawn per episode, for every N
      that another pool uses.  It says what the same test gives on classes the model knows;
  one per `--extra NAME=PACK[:IDS[:WAYS]]`: classes of another pack, which has its own class numbering (say, dog
      breeds from another dataset).  IDS picks classes ("151-268"; empty for all), WAYS is a list of N ("5,20,0",
      where 0 means all the picked classes at once); with several N the pools are called NAME_<N>way.

The episodes of a pool depend on (seed, pool name, k) and on the pool's labels only, so every checkpoint of every
run is asked the same questions.  The output, `RUN/fewshot.json`, has mean accuracy and a 95 % interval
(1.96 standard errors over episodes) for each checkpoint, feature kind, pool and k.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import zlib

import numpy as np
import torch

from . import features
from .data import Pack, Roles, _parse_ids
from .pack import cpus


def make_episodes(labels: np.ndarray, n_way: int, k: int, n_query: int, n_episodes: int, seed: int, name: str):
    """(support, query, q): index arrays of shape (E, N, k) and (E, N, q) into the pool, or None when no class
    set of N classes has k + 1 images each.  q is `n_query`, or fewer when the smallest class is short."""
    classes, counts = np.unique(labels, return_counts=True)
    ok = classes[counts >= k + n_query]                 # classes that can give every query, when there are enough
    if len(ok) < n_way:
        ok = classes[counts >= k + 1]
    if len(ok) < n_way:
        return None
    q = int(min(n_query, counts[np.isin(classes, ok)].min() - k))
    by_class = {int(c): np.flatnonzero(labels == c) for c in ok}
    rng = np.random.default_rng([int(seed), zlib.crc32(name.encode()), int(k), int(n_way)])
    sup = np.empty((n_episodes, n_way, k), dtype=np.int64)
    qry = np.empty((n_episodes, n_way, q), dtype=np.int64)
    for e in range(n_episodes):
        chosen = ok if len(ok) == n_way else rng.choice(ok, size=n_way, replace=False)
        for j, c in enumerate(chosen):
            pick = rng.permutation(by_class[int(c)])[:k + q]
            sup[e, j], qry[e, j] = pick[:k], pick[k:]
    return sup, qry, q


def cl2n(feats: np.ndarray, centre: np.ndarray) -> np.ndarray:
    f = feats.astype(np.float32) - centre.astype(np.float32)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


def episode_accuracy(f, sup: np.ndarray, qry: np.ndarray, chunk: int = 100) -> np.ndarray:
    """Accuracy of each episode for prepared features `f` (M, D): nearest class mean, Euclidean.  `f` is a numpy
    array or a torch tensor; the arithmetic runs where the tensor lives (the GPU, in a real evaluation)."""
    ft = f if isinstance(f, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(f))
    sup_t, qry_t = torch.from_numpy(sup).to(ft.device), torch.from_numpy(qry).to(ft.device)
    n_ep, n_way, q = qry.shape
    target = torch.arange(n_way, device=ft.device).repeat_interleave(q)[None, :]
    acc = np.empty(n_ep, dtype=np.float64)
    for s in range(0, n_ep, chunk):
        cent = ft[sup_t[s:s + chunk]].mean(dim=2)                               # (e, N, D)
        qf = ft[qry_t[s:s + chunk]].reshape(-1, n_way * q, ft.shape[1])         # (e, N*q, D)
        d = cent.pow(2).sum(-1)[:, None, :] - 2.0 * torch.bmm(qf, cent.transpose(1, 2))
        acc[s:s + chunk] = (d.argmin(-1) == target).double().mean(dim=1).cpu().numpy()
    return acc


def pool_episodes(labels: np.ndarray, name: str, *, n_way: int, shots, n_query: int, n_episodes: int, seed: int) -> dict:
    """The questions of one pool, {k: episodes or None}: made once and asked of every checkpoint."""
    return {k: make_episodes(labels, n_way, k, n_query, n_episodes, seed, name) for k in shots}


def evaluate_pool(f, labels: np.ndarray, name: str, *, n_way: int, shots, n_query: int, n_episodes: int, seed: int,
                  episodes: dict | None = None) -> dict:
    out = {"n_way": n_way, "images": int(len(labels)), "classes": int(len(np.unique(labels)))}
    if episodes is None:
        episodes = pool_episodes(labels, name, n_way=n_way, shots=shots, n_query=n_query, n_episodes=n_episodes, seed=seed)
    for k in shots:
        ep = episodes[k]
        if ep is None:
            out[str(k)] = None
            continue
        acc = episode_accuracy(f, ep[0], ep[1])
        out[str(k)] = {"acc": float(acc.mean()), "ci95": float(1.96 * acc.std(ddof=1) / np.sqrt(len(acc))),
                       "episodes": int(len(acc)), "n_query": ep[2]}
    return out


def build_pools(roles: Roles, held: list[Pack], base: Pack | None, extra: dict[str, Pack], *, image_size: int,
                per_class: int, seen_per_class: int, workers: int):
    """All pool images decoded once: (uint8 tensor, {pool: (slice, labels, n_way)}, slice of the centring set)."""
    chunks, pools, at = [], {}, 0

    def add(name, images, labels, n_way):
        nonlocal at
        chunks.append(images)
        pools[name] = (slice(at, at + len(images)), labels, n_way)
        at += len(images)

    for g, ids in roles.groups.items():
        parts = [features.decode_pool(p, roles, g, image_size=image_size, per_class=per_class, workers=workers)
                 for p in held]
        if parts:
            add(g, torch.cat([x for x, _ in parts]), np.concatenate([y for _, y in parts]), len(ids))
    for name, (pack, ids, ways) in extra.items():
        x, y = features.decode_pool(pack, Roles(pack.num_classes, {"picked": ids} if ids else None),
                                    "picked" if ids else "all", image_size=image_size, per_class=per_class,
                                    workers=workers)
        n_all = int(len(np.unique(y)))
        chunks.append(x)
        for w in ways:                                                # the pools of one --extra share its images
            pools[name if len(ways) == 1 else f"{name}_{w or n_all}way"] = (slice(at, at + len(x)), y, w or n_all)
        at += len(x)
    centre = None
    if base is not None:                                             # last: one seen pool for every N in use
        x, y = features.decode_pool(base, roles, "base", image_size=image_size, per_class=seen_per_class,
                                    workers=workers)
        centre = slice(at, at + len(x))
        for n in sorted({n for _, _, n in pools.values() if n <= len(roles.base)} or {5}):
            pools[f"seen_{n}way"] = (centre, y, n)
        chunks.append(x)
        at += len(x)
    if not chunks:
        raise SystemExit("fewshot: no pools; give --heldout-pack with a roles file, --base-pack or --extra")
    return torch.cat(chunks), pools, centre


def parse_extra(spec: str) -> tuple[str, str, list[int] | None, list[int]]:
    """NAME=PACK[:IDS[:WAYS]] -> (name, pack path, class ids or None for all, ways with 0 for "all picked")."""
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    if len(parts) > 3:
        raise SystemExit(f"fewshot: cannot read --extra {spec}")
    ids = _parse_ids(parts[1]) if len(parts) > 1 and parts[1] else None
    ways = [int(w) for w in parts[2].split(",")] if len(parts) > 2 and parts[2] else [0]
    return name, parts[0], ids, ways


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run directory of train.py")
    ap.add_argument("--roles")
    ap.add_argument("--heldout-pack", nargs="*", default=[], help="pack(s) holding the held-out classes' images")
    ap.add_argument("--base-pack", help="pack with base-class images the model was not trained on (validation)")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=PACK[:IDS[:WAYS]]")
    ap.add_argument("--shots", default="1,5,10")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--query", type=int, default=15, help="queries per class and episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=300, help="images per class kept from a held-out or extra pack")
    ap.add_argument("--seen-per-class", type=int, default=50)
    ap.add_argument("--steps", help="comma-separated checkpoint steps (default: all in the run)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out", help="default: RUN/fewshot.json")
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    workers = args.workers if args.workers is not None else max(0, cpus() - 2)
    shots = [int(s) for s in args.shots.split(",")]
    ckpts = sorted(glob.glob(os.path.join(args.run, "ckpt-*.pt")))
    if args.steps:
        want = {int(s) for s in args.steps.split(",")}
        ckpts = [c for c in ckpts if int(os.path.basename(c)[5:-3]) in want]
    if not ckpts:
        raise SystemExit(f"fewshot: no checkpoints in {args.run}")
    with open(os.path.join(args.run, "run.json")) as f:
        run = json.load(f)
    image_size = run["model"]["image_size"]
    held = [Pack(p) for p in args.heldout_pack]
    base = Pack(args.base_pack) if args.base_pack else None
    extra = {name: (Pack(path), ids, ways) for name, path, ids, ways in map(parse_extra, args.extra)}
    source = held[0] if held else base                 # the pack numbering the roles file refers to
    roles = Roles.load(args.roles, source.num_classes) if source is not None else Roles(run["num_classes"])
    if source is not None and len(roles.base) != run["num_classes"]:
        raise SystemExit(f"fewshot: the roles file leaves {len(roles.base)} base classes, the run was trained on "
                         f"{run['num_classes']}")

    t0 = time.time()
    images, pools, centre = build_pools(roles, held, base, extra, image_size=image_size, per_class=args.per_class,
                                        seen_per_class=args.seen_per_class, workers=workers)
    print(f"fewshot: {len(images)} images in {len(pools)} pools decoded in {time.time() - t0:.0f} s", flush=True)
    out = {"run": os.path.abspath(args.run), "optimizer": run["args"]["optimizer"], "shots": shots,
           "episodes": args.episodes, "query": args.query, "seed": args.seed,
           "centre": "mean base-class feature (--base-pack)" if centre is not None else "mean of all pooled images",
           "pools": {k: {"images": int(len(v[1])), "n_way": v[2]} for k, v in pools.items()}, "checkpoints": []}
    asked = {name: pool_episodes(labels, name, n_way=n, shots=shots, n_query=args.query, n_episodes=args.episodes,
                                 seed=args.seed) for name, (sl, labels, n) in pools.items()}
    print(f"fewshot: episodes drawn ({time.time() - t0:.0f} s)", flush=True)
    for path in ckpts:
        model, ck = features.load_model(path, device)
        feats = features.extract(model, images, device, amp=device.type == "cuda")
        res = {}
        for kind, f in feats.items():
            mean = f[centre].mean(0) if centre is not None else f.mean(0)
            prepared = torch.from_numpy(cl2n(f, mean)).to(device)
            res[kind] = {name: evaluate_pool(prepared[sl], labels, name, n_way=n, shots=shots, n_query=args.query,
                                             n_episodes=args.episodes, seed=args.seed, episodes=asked[name])
                         for name, (sl, labels, n) in pools.items()}
        out["checkpoints"].append({"step": ck["step"], "file": os.path.basename(path), "val_top1": ck["val_top1"],
                                   "results": res})
        first = next(iter(pools))
        k = str(shots[-1])
        r = res["pre_logits"][first][k]
        print(f"fewshot: step {ck['step']}: {first} {k}-shot "
              f"{'n/a' if r is None else format(r['acc'], '.3f')}", flush=True)
    dst = args.out or os.path.join(args.run, "fewshot.json")
    with open(dst + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(dst + ".tmp", dst)
    print(f"fewshot: {len(ckpts)} checkpoints -> {dst} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
