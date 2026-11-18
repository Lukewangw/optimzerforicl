"""The sequence model along one encoder's training: what a learned rule reaches on each checkpoint's features.

Needs matplotlib.

    python -m icl.seqseries --root RUNS --encoder nodog-adamw-lr1e-3-s0 --out DIR [--seeds 0,1,2]

Reads the folders ROOT/ENCODER-STEP (one per checkpoint whose features were extracted) and
keeps the settings (optimizer and learning rate) and seeds that every one of them has.

  series.png   against the encoder's training progress: the sequence models' final accuracy on unseen classes and on
               the classes the encoder was trained on, by the sequence model's optimizer, with the fixed rule
               (nearest class mean) on the same sequences; and the step of the sequence model's own training at
               which the ability appears
  series.md    the numbers
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .seqsummary import appears, load
from .summary import HUE, INK2, LAYER, NAME, ORDER, SURFACE, _end_labels, _head, _level, _style


def collect(root: str, encoder: str, seeds: list[int] | None) -> tuple[list[int], dict]:
    """(checkpoint steps, {(optimizer, lr): {step: [runs]}}) for the settings and seeds present at every step."""
    by_step = {}
    for d in glob.glob(os.path.join(root, encoder + "-*")):
        tail = os.path.basename(d)[len(encoder) + 1:]
        if not tail.isdigit():
            continue
        runs = [r for r in (load(p) for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)) if r and r["done"]]
        if runs:
            by_step[int(tail)] = runs
    if not by_step:
        raise SystemExit(f"seqseries: no finished run under {root}/{encoder}-*")
    steps = sorted(by_step)
    keys = set.intersection(*({(r["optimizer"], r["lr"], r["seed"]) for r in by_step[s]} for s in steps))
    if seeds is not None:
        keys = {k for k in keys if k[2] in seeds}
    out = {}
    for s in steps:
        for r in by_step[s]:
            if (r["optimizer"], r["lr"], r["seed"]) in keys:
                out.setdefault((r["optimizer"], r["lr"]), {}).setdefault(s, []).append(r)
    order = sorted(out, key=lambda k: ((ORDER + [k[0]]).index(k[0]), k[1]))
    return steps, {k: out[k] for k in order}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="dogs")
    ap.add_argument("--what", default="dog breeds")
    ap.add_argument("--seeds", help="comma-separated; default: every seed all checkpoints have")
    ap.add_argument("--total", type=int, help="the encoder's last step (default: the largest step found)")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    steps, settings = collect(args.root, args.encoder, [int(s) for s in args.seeds.split(",")] if args.seeds else None)
    if not settings:
        raise SystemExit("seqseries: the checkpoints share no setting and seed")
    total = args.total or steps[-1]
    pct = 100.0 * np.array(steps) / total
    any_setting = next(iter(settings.values()))
    fixed = {key: np.array([any_setting[s][0]["fixed"][key] for s in steps]) for key in (args.pool, "seen")}
    chance = any_setting[steps[0]][0]["chance"]
    n_seeds = len(any_setting[steps[0]])
    feature = any_setting[steps[0]][0]["feature"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9))
    fig.patch.set_facecolor(SURFACE)
    rows = []
    several = len({k[0] for k in settings}) < len(settings)          # more than one learning rate per optimizer
    for (opt, lr), per_step in settings.items():
        colour = HUE.get(opt, INK2)
        name = f"{NAME.get(opt, opt)}, lr {lr:g}"
        row = {"setting": name}
        for ax, key in ((axes[0], args.pool), (axes[1], "seen")):
            m = np.array([np.mean([r["eval"][key][-1] for r in per_step[s]]) for s in steps])
            sd = np.array([np.std([r["eval"][key][-1] for r in per_step[s]]) for s in steps])
            ax.plot(pct, 100 * m, color=colour, lw=1.7, solid_capstyle="round", zorder=3, label=name,
                    alpha=0.6 if several else 1.0)
            ax.fill_between(pct, 100 * (m - sd), 100 * (m + sd), color=colour, alpha=0.10, lw=0, zorder=2)
            ax.plot(pct, 100 * m, ls="none", marker="o", ms=3.6, color=colour, mec=SURFACE, mew=0.7, zorder=4)
            ax.plot(pct[-1:], 100 * m[-1:], ls="none", marker="o", ms=6.5, color=colour, mec=SURFACE, mew=1.4, zorder=5)
            ax._ends = getattr(ax, "_ends", []) + [(pct[-1], 100 * m[-1], f"{100 * m[-1]:.1f}")]
            row[key] = (m, sd)
        when = []
        for s in steps:
            v = [appears(r, "seen") for r in per_step[s]]
            v = [x for x in v if x is not None]
            when.append(np.mean(v) if v else np.nan)
        axes[2].plot(pct, when, color=colour, lw=1.7, solid_capstyle="round", zorder=3, alpha=0.6 if several else 1.0)
        axes[2].plot(pct, when, ls="none", marker="o", ms=3.6, color=colour, mec=SURFACE, mew=0.7, zorder=4)
        row["when"] = np.array(when)
        rows.append(row)
    for ax, key in ((axes[0], args.pool), (axes[1], "seen")):
        ax.plot(pct, 100 * fixed[key], color=INK2, lw=1.3, zorder=2.5, label="nearest class mean, no training")
        ax.plot(pct, 100 * fixed[key], ls="none", marker="D", ms=3.4, color=INK2, mec=SURFACE, mew=0.7, zorder=4)
        ax._ends.append((pct[-1], 100 * fixed[key][-1], f"{100 * fixed[key][-1]:.1f}"))
    _style(axes[0], f"Unseen classes ({args.what}): final accuracy (%)", "% of the encoder's training")
    _style(axes[1], "Classes the encoder was trained on: final accuracy (%)", "% of the encoder's training")
    _style(axes[2], "Step at which the ability appears (log scale)", "% of the encoder's training")
    axes[2].set_yscale("log")
    axes[2].minorticks_off()
    lo, hi = axes[2].get_ylim()
    ticks = [t for t in (10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 2000, 5000, 10000) if lo <= t <= hi]
    axes[2].set_yticks(ticks)
    axes[2].set_yticklabels([f"{t:,}" for t in ticks])
    for ax in axes:
        ax.set_xlim(-3, 114 if ax is not axes[2] else 104)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    for ax in axes[:2]:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 100 * chance - 0.05 * (hi - lo)), hi)
        _level(ax, 100 * chance, f"chance {100 * chance:.0f} %")
    top = _head(fig, "A learned rule for using examples, on the encoder's features as the encoder trains",
                f"Each point: sequence models trained on the features of one encoder checkpoint (mean of {n_seeds} "
                f"seed{'s' if n_seeds > 1 else ''}; shaded: their spread), tested on sequences the same for all.  "
                f"Features of {LAYER.get(feature, feature)}.\n"
                f"Colour is the sequence model's optimizer.  \"Appears\": the step of its own training at which "
                f"accuracy on the trained classes is half-way from chance to its final value.",
                legend_from=axes[0], ncol=len(settings) + 1)
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=2.0)
    for ax in axes[:2]:
        _end_labels(ax, ax._ends)
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "series.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    lines = [f"# Sequence models along the training of {args.encoder} ({n_seeds} seeds each)", "",
             "| encoder step | % of training | " + " | ".join(
                 f"{r['setting']}: {args.pool} | {r['setting']}: seen | {r['setting']}: appears at step" for r in rows)
             + f" | nearest class mean: {args.pool} | nearest class mean: seen |",
             "|---|---|" + "---|---|---|" * len(rows) + "---|---|"]
    for i, s in enumerate(steps):
        cells = []
        for r in rows:
            cells += [f"{100 * r[args.pool][0][i]:.1f} ± {100 * r[args.pool][1][i]:.1f} %",
                      f"{100 * r['seen'][0][i]:.1f} ± {100 * r['seen'][1][i]:.1f} %",
                      "never" if np.isnan(r["when"][i]) else f"{r['when'][i]:,.0f}"]
        lines.append(f"| {s:,} | {pct[i]:.1f} | " + " | ".join(cells)
                     + f" | {100 * fixed[args.pool][i]:.1f} % | {100 * fixed['seen'][i]:.1f} % |")
    with open(os.path.join(args.out, "series.md"), "w") as f:
        f.write("\n".join(lines) + f"\n\nChance is {100 * chance:.0f} %. Accuracy is the mean over seeds ± their "
                "standard deviation, at the end of the sequence model's training.\n")
    print(f"seqseries: series.png, series.md -> {args.out}: {len(steps)} checkpoints, {len(settings)} settings, "
          f"{n_seeds} seeds")


if __name__ == "__main__":
    main()
