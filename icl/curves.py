"""Curves from finished runs (needs matplotlib; meant for the Mac, not for the cluster environment).

    python -m icl.curves --runs RUN_A RUN_B ... --out DIR [--feature pre_logits]

For every pool and k of `fewshot.json`, two PNGs: few-shot accuracy against the training step, and against the
base-class validation top-1 at the same checkpoint (so that two optimizers can be compared at equal accuracy on
what they were trained on, not only at equal step).  Runs are coloured by optimizer; every run is drawn as its own
thin line, and runs that differ only in their seeds share a thick mean line.  Also `val_top1.png` and
`train_loss.png`.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

COLOURS = {"adamw": "#2a78d6", "muon": "#eb6834", "sgd": "#1baf7a"}   # as in summary.py


VARIANTS = ("no_pre_logits", "muon_skip_pre_logits", "muon_blocks", "start_from")   # train.py's options that make a run a control
SHORT = {"attn": "the attention", "mlp": "the MLP"}


def variant_of(a: dict) -> str:
    """What a run's arguments change in the usual model, or in the usual split of the parameters between the
    optimizers ('' for a usual run): "no pre_logits" (a plain linear head), "pre_logits by AdamW" (a Muon run
    that leaves that one matrix to AdamW), "on the MLP only" or "on the attention only" (a Muon run that gives
    Muon one kind of block matrix and AdamW the rest; "... and pre_logits" if Muon keeps that matrix too),
    "on pre_logits only" (a Muon run whose blocks are all AdamW's: Muon has that one matrix; "... at LR" if at a
    learning rate of its own),
    "from RUN at step N" (a run that went on from another run's checkpoint with its own optimizer)."""
    parts = ["no pre_logits"] if a.get("no_pre_logits") else []
    blocks = a.get("muon_blocks") or "all"
    if a.get("optimizer") == "muon" and blocks == "none":
        own = a.get("muon_lr") not in (None, a.get("lr"))      # that matrix at a rate of its own: another setting
        parts.append(f"on pre_logits only at {a['muon_lr']:g}" if own else "on pre_logits only")
    elif a.get("optimizer") == "muon" and blocks != "all":
        on = SHORT.get(blocks, blocks)
        parts.append(f"on {on} only" if a.get("muon_skip_pre_logits") or a.get("no_pre_logits")
                     else f"on {on} and pre_logits only")
    elif a.get("muon_skip_pre_logits") and not a.get("no_pre_logits"):
        parts.append("pre_logits by AdamW")
    if a.get("start_from"):
        ckpt = a["start_from"].rstrip("/")
        step = os.path.basename(ckpt).removeprefix("ckpt-").removesuffix(".pt").lstrip("0") or "0"
        parts.append(f"from {os.path.basename(os.path.dirname(ckpt))} at step {step}")
    return ", ".join(parts)


def load_run(path: str) -> dict:
    """A run's arguments and log.  "variant" names what it changes in the usual model or in the usual split of the
    parameters between optimizers (VARIANTS; empty for a usual run), so that such a run is not taken for one more
    seed of the setting it varies."""
    with open(os.path.join(path, "run.json")) as f:
        run = json.load(f)
    a = run["args"]
    variant = variant_of(a)
    setting = f"{a['optimizer']} lr {a['lr']:g}"
    if a["optimizer"] == "muon" and a.get("muon_lr") not in (None, a["lr"]):
        setting += f" / muon {a['muon_lr']:g}"
    if variant:
        setting += f", {variant}"
    out = {"path": path, "optimizer": a["optimizer"], "setting": setting, "variant": variant, "lr": a["lr"],
           "total_steps": run.get("total_steps"), "batch_size": a.get("batch_size"),
           "seeds": (a["init_seed"], a["data_seed"]), "epochs": a.get("epochs"), "train": [], "ckpt": [],
           "fewshot": None}
    log = os.path.join(path, "log.jsonl")
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("event") == "resume":               # what a stopped process logged past its last save
                    for key in ("train", "ckpt"):              # is repeated by the process that resumed
                        out[key] = [r for r in out[key] if r["step"] <= rec["step"]]
                elif rec.get("event") in ("train", "ckpt"):
                    out[rec["event"]].append(rec)
    fs = os.path.join(path, "fewshot.json")
    if os.path.exists(fs):
        with open(fs) as f:
            out["fewshot"] = json.load(f)
    return out


def fewshot_series(run: dict, feature: str, pool: str, k: int):
    """(steps, val top-1 or nan, accuracy, 95 % half-width) over the run's checkpoints."""
    rows = []
    for ck in run["fewshot"]["checkpoints"]:
        r = ck["results"].get(feature, {}).get(pool, {}).get(str(k))
        if r:
            rows.append((ck["step"], np.nan if ck["val_top1"] is None else ck["val_top1"], r["acc"], r["ci95"]))
    return tuple(np.array(c) for c in zip(*sorted(rows))) if rows else None


def _mean_lines(ax, groups: dict, xkey: int) -> None:
    """One thick line per setting: the mean over its runs where they share the x values (the steps)."""
    for setting, items in groups.items():
        if len(items) < 2 or xkey != 0:
            continue
        steps = set.intersection(*(set(s[0].tolist()) for _, s in items))
        if not steps:
            continue
        xs = sorted(steps)
        ys = np.mean([[s[2][list(s[0]).index(x)] for x in xs] for _, s in items], axis=0)
        ax.plot(xs, ys, color=COLOURS.get(items[0][0]["optimizer"], "black"), lw=2.6, alpha=0.9,
                label=f"{setting} (mean of {len(items)})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature", default="pre_logits", choices=("pre_logits", "gap"))
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [load_run(p) for p in args.runs]
    os.makedirs(args.out, exist_ok=True)
    written = []

    def finish(fig, ax, name, xlabel, ylabel, title):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, name), dpi=140)
        plt.close(fig)
        written.append(name)

    styles = ["-", "--", ":", "-."]
    settings = sorted({r["setting"] for r in runs})
    style = {s: styles[[x for x in settings if x.split()[0] == s.split()[0]].index(s) % len(styles)]
             for s in settings}

    for key, ylabel, name in (("train", "training loss", "train_loss.png"), ("ckpt", "base-class val top-1",
                                                                              "val_top1.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for r in runs:
            recs = [x for x in r[key] if (key == "train" or x.get("val_top1") is not None)]
            if recs:
                ax.plot([x["step"] for x in recs], [x["loss"] if key == "train" else x["val_top1"] for x in recs],
                        color=COLOURS.get(r["optimizer"], "black"), ls=style[r["setting"]], lw=1.1,
                        marker=None if key == "train" else "o", ms=3, label=f"{r['setting']} seed {r['seeds'][0]}")
        if ax.lines:
            finish(fig, ax, name, "training step", ylabel, ylabel)
        else:
            plt.close(fig)

    with_fs = [r for r in runs if r["fewshot"]]
    pools = sorted({p for r in with_fs for p in r["fewshot"]["pools"]})
    shots = sorted({k for r in with_fs for k in r["fewshot"]["shots"]})
    for pool in pools:
        for k in shots:
            for xkey, xname, xlabel in ((0, "step", "training step"), (1, "val", "base-class val top-1")):
                fig, ax = plt.subplots(figsize=(6.4, 4.2))
                groups, n_way = {}, None
                for r in with_fs:
                    s = fewshot_series(r, args.feature, pool, k)
                    if s is None or (xkey == 1 and np.isnan(s[1]).all()):
                        continue
                    n_way = r["fewshot"]["pools"][pool]["n_way"]
                    c = COLOURS.get(r["optimizer"], "black")
                    ax.errorbar(s[xkey], s[2], yerr=s[3], color=c, ls=style[r["setting"]], lw=1.0, marker="o",
                                ms=2.5, capsize=1.5, alpha=0.8, label=f"{r['setting']} seed {r['seeds'][0]}")
                    ax.plot(s[xkey][-1:], s[2][-1:], color=c, marker="o", ms=6, ls="none")   # the last checkpoint
                    groups.setdefault(r["setting"], []).append((r, s))
                if not groups:
                    plt.close(fig)
                    continue
                _mean_lines(ax, groups, xkey)
                ax.axhline(1.0 / n_way, color="gray", lw=0.8, ls=":")
                finish(fig, ax, f"fewshot_{pool}_k{k}_vs_{xname}.png", xlabel, f"{k}-shot accuracy",
                       f"{pool}: {n_way}-way {k}-shot, nearest class mean on {args.feature} (dotted: chance)")
    print(f"curves: {len(written)} files -> {args.out}: {', '.join(written)}")


if __name__ == "__main__":
    main()
