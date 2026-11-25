"""A third read-out of the frozen features: a linear classifier trained on the support set of each episode.

    python -m icl.linear --run RUN --roles roles.json --heldout-pack TRAINPACK --base-pack VALPACK \
        [--extra dogs=TESTS:151-268:5,20,0 ...] [--shots 1,5,10,25] [--layers gap,pre_act,pre_logits]

The images, pools, questions and seed are fewshot.py's and probe.py's (same arguments, same functions), so for a k
that both scripts have, the questions are the same.  A k that a pool cannot supply (too few images per class for
k + query) is left out of that pool.  Only one checkpoint is read, the last of the run (or `--step N`).  The output
is RUN/linear.json.

Rules, per layer, pool and k:
  plain    probe.py's: minus the mean of the base-class reference rows, unit length, nearest class mean.
  white    probe.py's whitened version of it, nearest class mean.
  linear   every feature is standardised with the mean and standard deviation of the reference rows (the std plus
           a small epsilon, 1e-3 of the mean std).  Per episode, ridge regression of the one-hot labels on the
           support features, both centred on their support means: W = (X'X + lam * tr(X'X)/d * I)^-1 X'Y, a query
           goes to the argmax of (x - support mean) W + target mean.  One eigendecomposition per episode serves
           every lam.  Float64; the smaller of the d x d (primal) and n x n (dual) forms.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from . import aug, features, fewshot, probe
from .data import Pack, Roles
from .pack import cpus
from .train import autocast

RULES = ("plain", "white", "linear")
STD_EPS = 1e-3                      # of the mean std, added to every std


def ridge_weights(xc: torch.Tensor, yc: torch.Tensor, lams, form: str | None = None) -> torch.Tensor:
    """Ridge weights W (L, E, d, C) for centred support features `xc` (E, n, d) and centred targets `yc` (E, n, C)
    or (n, C), for every lam of `lams`, with the penalty lam * trace(X'X) / d.  `form` is "primal", "dual" or None
    for the smaller system.  One eigendecomposition per episode."""
    E, n, d = xc.shape
    if yc.ndim == 2:
        yc = yc.expand(E, -1, -1)
    form = form or ("primal" if d <= n else "dual")
    lam = torch.as_tensor(list(lams), dtype=xc.dtype, device=xc.device)
    if form == "primal":
        s, v = torch.linalg.eigh(xc.transpose(1, 2) @ xc)                      # (E, d), (E, d, d)
        tr = s.sum(-1)
        rhs = v.transpose(1, 2) @ (xc.transpose(1, 2) @ yc)                    # (E, d, C)
    elif form == "dual":
        s, v = torch.linalg.eigh(xc @ xc.transpose(1, 2))                      # (E, n), (E, n, n)
        tr = s.sum(-1)
        rhs = v.transpose(1, 2) @ yc                                           # (E, n, C)
    else:
        raise ValueError(form)
    s = s.clamp_min(0.0)
    pen = (lam[:, None] * tr[None, :] / d).clamp_min(1e-12)                    # (L, E)
    inv = 1.0 / (s[None] + pen[:, :, None])                                    # (L, E, d or n)
    sol = v[None] @ (inv[..., None] * rhs[None])                               # (L, E, d or n, C)
    return sol if form == "primal" else xc.transpose(1, 2)[None] @ sol


def linear_episode_accuracy(z: torch.Tensor, sup: np.ndarray, qry: np.ndarray, lams, chunk: int | None = None,
                            form: str | None = None) -> np.ndarray:
    """Accuracy (L, E) of the ridge classifier of each episode, on standardised features `z` (M, d) in float64."""
    n_ep, n_way, k = sup.shape
    q = qry.shape[2]
    d = z.shape[1]
    sup_t, qry_t = torch.from_numpy(sup).to(z.device), torch.from_numpy(qry).to(z.device)
    onehot = torch.eye(n_way, dtype=z.dtype, device=z.device).repeat_interleave(k, dim=0)        # (n, C)
    ym = onehot.mean(0)
    target = torch.arange(n_way, device=z.device).repeat_interleave(q)[None, None, :]
    if chunk is None:
        chunk = max(1, int(2e8 // (8 * max(d, n_way * k) ** 2)))
    out = np.empty((len(list(lams)), n_ep))
    for s in range(0, n_ep, chunk):
        xs = z[sup_t[s:s + chunk]].reshape(-1, n_way * k, d)
        xm = xs.mean(1, keepdim=True)
        w = ridge_weights(xs - xm, onehot - ym, lams, form)                                       # (L, e, d, C)
        qf = z[qry_t[s:s + chunk]].reshape(-1, n_way * q, d) - xm
        score = qf[None] @ w + ym                                                                 # (L, e, Nq, C)
        out[:, s:s + chunk] = (score.argmax(-1) == target).double().mean(-1).cpu().numpy()
    return out


def _cell(acc: np.ndarray) -> dict:
    return {"acc": float(acc.mean()), "ci95": float(1.96 * acc.std(ddof=1) / np.sqrt(len(acc)))}


def standardise(f: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    r = ref.double()
    sd = r.std(0, unbiased=False)
    return (f.double() - r.mean(0)) / (sd + STD_EPS * sd.mean())


def evaluate_layer(f: torch.Tensor, ref: slice, pools: dict, asked: dict, shots, lams, *, n_query: int,
                   n_episodes: int, seed: int, ridge: float = 1e-3) -> dict:
    """{pool: {str(k): {"plain", "white", "linear"}}} for the features `f` (M, d) of one layer; a k whose episodes
    could not be drawn is absent."""
    prepared = {"plain": probe.plain(f, f[ref]), "white": probe.whitened(f, f[ref], ridge)}
    z = standardise(f, f[ref])
    out = {}
    for name, (sl, labels, n) in pools.items():
        out[name] = {}
        for rule, p in prepared.items():
            res = fewshot.evaluate_pool(p[sl], labels, name, n_way=n, shots=shots, n_query=n_query,
                                        n_episodes=n_episodes, seed=seed, episodes=asked[name])
            for k in shots:
                if res[str(k)] is not None:
                    out[name].setdefault(str(k), {})[rule] = {"acc": res[str(k)]["acc"], "ci95": res[str(k)]["ci95"]}
        for k in shots:
            ep = asked[name][k]
            if ep is None:
                continue
            acc = linear_episode_accuracy(z[sl], ep[0], ep[1], lams)
            out[name][str(k)]["linear"] = {str(float(lam)): _cell(a) for lam, a in zip(lams, acc)}
    return out


def available_layers(model, wanted: list[str]) -> list[str]:
    have = probe.layer_names(model)
    if model.pre_logits is None:                       # no last hidden layer: pre_act and pre_logits are `gap`
        have = [n for n in have if n not in ("pre_act", "pre_logits")]
    return [n for n in wanted if n in have]


@torch.no_grad()
def extract_layers(model, images_u8: torch.Tensor, device, names: list[str], *, batch_size: int = 500,
                   amp: bool = False) -> dict[str, torch.Tensor]:
    model.eval()
    feats = {n: [] for n in names}
    for i in range(0, len(images_u8), batch_size):
        x = aug.to_model_input(images_u8[i:i + batch_size].to(device, non_blocking=True))
        with autocast(amp):
            f, _ = probe.layers(model, x)
        for n in names:
            feats[n].append(f[n].float())
    return {n: torch.cat(v) for n, v in feats.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run directory of train.py")
    ap.add_argument("--roles")
    ap.add_argument("--heldout-pack", nargs="*", default=[], help="pack(s) holding the held-out classes' images")
    ap.add_argument("--base-pack", help="validation images of the trained classes")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=PACK[:IDS[:WAYS]]")
    ap.add_argument("--shots", default="1,5,10,25")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--query", type=int, default=15)
    ap.add_argument("--layers", default="gap,pre_act,pre_logits")
    ap.add_argument("--lams", default="0.01,0.1,1,10", help="ridge penalties, as shares of the mean eigenvalue of X'X")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seen-per-class", type=int, default=50)
    ap.add_argument("--ridge", type=float, default=1e-3, help="of the `white` rule, as in probe.py")
    ap.add_argument("--step", type=int, help="checkpoint step (default: the last)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out", help="default: RUN/linear.json")
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    workers = args.workers if args.workers is not None else max(0, cpus() - 2)
    shots = [int(s) for s in args.shots.split(",")]
    lams = [float(s) for s in args.lams.split(",")]
    wanted = [s for s in args.layers.split(",") if s]
    ckpts = sorted(glob.glob(os.path.join(args.run, "ckpt-*.pt")))
    if args.step is not None:
        ckpts = [c for c in ckpts if int(os.path.basename(c)[5:-3]) == args.step]
    if not ckpts:
        raise SystemExit(f"linear: no checkpoint in {args.run}")
    path = ckpts[-1]
    with open(os.path.join(args.run, "run.json")) as f:
        run = json.load(f)
    image_size = run["model"]["image_size"]
    held = [Pack(p) for p in args.heldout_pack]
    base = Pack(args.base_pack) if args.base_pack else None
    extra = {name: (Pack(p), ids, ways) for name, p, ids, ways in map(fewshot.parse_extra, args.extra)}
    source = held[0] if held else base
    roles = Roles.load(args.roles, source.num_classes) if source is not None else Roles(run["num_classes"])
    if source is not None and len(roles.base) != run["num_classes"]:
        raise SystemExit(f"linear: the roles file leaves {len(roles.base)} base classes, the run was trained on "
                         f"{run['num_classes']}")

    t0 = time.time()
    images, pools, centre = fewshot.build_pools(roles, held, base, extra, image_size=image_size,
                                                per_class=args.per_class, seen_per_class=args.seen_per_class,
                                                workers=workers)
    ref = centre if centre is not None else slice(0, len(images))
    print(f"linear: {len(images)} images in {len(pools)} pools decoded in {time.time() - t0:.0f} s", flush=True)
    asked = {name: fewshot.pool_episodes(labels, name, n_way=n, shots=shots, n_query=args.query,
                                         n_episodes=args.episodes, seed=args.seed)
             for name, (sl, labels, n) in pools.items()}
    model, ck = features.load_model(path, device)
    names = available_layers(model, wanted)
    if not names:
        raise SystemExit(f"linear: none of {wanted} is a layer of this model")
    feats = extract_layers(model, images, device, names, amp=amp)
    out = {"run": os.path.basename(os.path.normpath(args.run)), "optimizer": run["args"]["optimizer"],
           "step": ck["step"], "val_top1": ck["val_top1"], "shots": shots, "episodes": args.episodes,
           "query": args.query, "seed": args.seed, "layers": names, "rules": list(RULES), "lams": lams,
           "pools": {k: {"n_way": v[2], "images": int(len(v[1])), "classes": int(len(np.unique(v[1])))}
                     for k, v in pools.items()},
           "results": {}}
    for layer in names:
        t1 = time.time()
        out["results"][layer] = evaluate_layer(feats[layer], ref, pools, asked, shots, lams, n_query=args.query,
                                               n_episodes=args.episodes, seed=args.seed, ridge=args.ridge)
        pool = next(iter(pools))
        cell = out["results"][layer][pool].get(str(shots[-1]))
        txt = "n/a" if cell is None else (f"plain {cell['plain']['acc']:.3f}, linear "
                                           f"{max(c['acc'] for c in cell['linear'].values()):.3f} (best lam)")
        print(f"linear: step {ck['step']} {layer}: {pool} {shots[-1]}-shot {txt} ({time.time() - t1:.0f} s)",
              flush=True)
    dst = args.out or os.path.join(args.run, "linear.json")
    with open(dst + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(dst + ".tmp", dst)
    print(f"linear: -> {dst} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
