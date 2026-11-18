"""Sequence models on the features of several encoders: does the encoder's optimizer change what a learned rule
reaches, and does the sequence model's optimizer matter more or less than the encoder's?

Needs matplotlib.

    python -m icl.seqcross --encoders "AdamW ViT=DIR_A" "Muon ViT=DIR_B" ... --out OUT [--pool dogs]

Each DIR holds one encoder's sequence-model runs, the same settings and seeds in each;
a backslash and an n typed in a NAME break the line of its label in the figure.
The questions are drawn from the same seed over the same images, so every encoder is asked the same sequences.

  cross.png   per encoder, the final accuracy of each sequence-model optimizer at its best learning rate (chosen on
              the seen classes, as in seqsummary.py), with the fixed rule (nearest class mean) on that encoder's
              features: on the unseen pool, on the left-out classes, and on the classes the encoder was trained on
  cross.md    the numbers, for the learning rate chosen on the seen classes and for the one chosen on `control`
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .seqsummary import appears, best, groups, load
from .summary import HUE, INK2, LAYER, NAME, ORDER, SURFACE, _head, _style


def collect(path: str, pool: str) -> dict:
    runs = [r for r in (load(p) for p in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isdir(p)) if r]
    runs = [r for r in runs if r["done"] and pool in r["eval"]]
    if not runs:
        raise SystemExit(f"seqcross: no finished run with the pool {pool!r} under {path}")
    grouped = groups(runs)
    picks = {"seen": best(grouped, "seen")}
    if "control" in runs[0]["eval"]:
        picks["control"] = best(grouped, "control")
    return {"fixed": runs[0]["fixed"], "chance": runs[0]["chance"], "picks": picks,
            "encoder_step": runs[0]["encoder_step"], "feature": runs[0]["feature"]}


def cell(rs: list[dict], key: str) -> tuple[float, float]:
    v = np.array([r["eval"][key][-1] for r in rs if key in r["eval"]])
    return (100 * float(v.mean()), 100 * float(v.std())) if len(v) else (float("nan"), float("nan"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", required=True, metavar="NAME=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="dogs")
    ap.add_argument("--what", default="dog breeds")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    encoders = []
    for item in args.encoders:
        name, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"seqcross: --encoders takes NAME=DIR, got {item!r}")
        encoders.append((name, collect(path, args.pool)))
    opts = [o for o in ORDER if all(o in e["picks"]["seen"] for _, e in encoders)]
    if not opts:
        raise SystemExit("seqcross: the encoders share no sequence-model optimizer")
    panels = [(args.pool, f"Unseen classes ({args.what})")]
    if all("control" in e["picks"] for _, e in encoders) and args.pool != "control":
        panels.append(("control", "Left-out classes of the training source"))
    panels.append(("seen", "Classes the encoder was trained on"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.7), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    step = 0.17
    for ax, (key, title) in zip(axes[0], panels):
        for j, opt in enumerate(opts):
            xs = np.arange(len(encoders)) + (j - (len(opts) - 1) / 2) * step - step / 2
            m, sd = zip(*(cell(e["picks"]["seen"][opt][1], key) for _, e in encoders))
            colour = HUE.get(opt, INK2)
            ax.errorbar(xs, m, yerr=sd, fmt="none", ecolor=colour, elinewidth=0.9, alpha=0.7, zorder=3)
            ax.plot(xs, m, ls="none", marker="o", ms=7, color=colour, mec=SURFACE, mew=1.4, zorder=4,
                    label=f"{NAME.get(opt, opt)} sequence model")
        fx = np.arange(len(encoders)) + ((len(opts) - 1) / 2 + 1) * step - step / 2
        fixed = [100 * e["fixed"].get(key, np.nan) for _, e in encoders]
        ax.plot(fx, fixed, ls="none", marker="D", ms=6.5, color=INK2, mec=SURFACE, mew=1.4, zorder=4,
                label="nearest class mean, no training")
        for x, y in zip(fx, fixed):
            if np.isfinite(y):
                ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(7, 0), textcoords="offset points", fontsize=7.5,
                            color=INK2, va="center")
        _style(ax, f"{title}: final accuracy (%)", "")
        ax.set_xticks(np.arange(len(encoders)))
        ax.set_xticklabels([n.replace("\\n", "\n") for n, _ in encoders], fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(encoders) - 0.35)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.10 * (hi - lo))
    seeds = max(len(rs) for _, e in encoders for _, rs in e["picks"]["seen"].values())
    layers = " / ".join(sorted({LAYER.get(e["feature"], e["feature"]) for _, e in encoders}))
    top = _head(fig, "The same sequence-model sweep on the features of differently trained encoders",
                f"Each encoder is asked the same sequences.  A point: a sequence-model optimizer at its best learning "
                f"rate (chosen on the encoder's own classes), mean of {seeds} seed{'s' if seeds > 1 else ''}; bars: "
                f"their spread.\nThe label under each group is the encoder; the colour is the optimizer of the small "
                f"transformer trained on its frozen features ({layers}).",
                legend_from=axes[0][0], ncol=len(opts) + 1)
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=2.0)
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "cross.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    keys = [k for k, _ in panels]
    lines = ["# Sequence models across encoders", ""]
    for pick, text in (("seen", "chosen on the encoder's own classes (`seen`)"),
                       ("control", "chosen on the left-out classes (`control`)")):
        if not all(pick in e["picks"] for _, e in encoders):
            continue
        lines += [f"## Learning rate {text}", "",
                  "| encoder | sequence model | learning rate | " + " | ".join(keys) + " | appears at step (seen) |",
                  "|---|---|---|" + "---|" * (len(keys) + 1)]
        for name, e in encoders:
            name = name.replace("\\n", " ")
            for opt in opts:
                lr, rs = e["picks"][pick][opt]
                when = [x for x in (appears(r, "seen") for r in rs) if x is not None]
                lines.append(f"| {name} | {NAME.get(opt, opt)} | {lr:g} | "
                             + " | ".join("{:.1f} ± {:.1f} %".format(*cell(rs, k)) for k in keys)
                             + (f" | {np.mean(when):,.0f} |" if when else " | never |"))
            lines.append(f"| {name} | nearest class mean (no training) | | "
                         + " | ".join(f"{100 * e['fixed'][k]:.1f} %" if k in e["fixed"] else "n/a" for k in keys)
                         + " | |")
        lines.append("")
    chance = encoders[0][1]["chance"]
    lines.append(f"Chance is {100 * chance:.0f} %.  Accuracy is the mean over seeds ± their standard deviation at the "
                 "end of the sequence model's training.  \"Appears\": the first step at which accuracy on the seen "
                 "classes is half-way from chance to the run's final value.")
    with open(os.path.join(args.out, "cross.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"seqcross: cross.png, cross.md -> {args.out}: {len(encoders)} encoders, "
          + ", ".join(NAME.get(o, o) for o in opts))


if __name__ == "__main__":
    main()
