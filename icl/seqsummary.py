"""Figures and a table for the sequence-model runs on one encoder's features (see seq.py).  Needs matplotlib.

    python -m icl.seqsummary --runs DIR/* --out OUT [--pool dogs] [--what "dog breeds"]

  emergence.png   accuracy on sequences of unseen classes against the sequence model's own training step, each
                  optimizer at its best learning rate (thin: single seeds, thick: their mean); the same for the
                  classes the encoder was trained on; and the training loss
  sweep.png       every learning rate, one column per optimizer (means over seeds)
  tradeoff.png    final accuracy on the unseen classes against final accuracy on the seen classes, one point per
                  optimizer and learning rate: do the optimizers lie on one line or on different ones?
  summary.md      final accuracy, the fixed rule (nearest class mean) on the same sequences, and the step at which
                  the ability appears

The best learning rate of an optimizer is chosen on the `seen` pool (classes the encoder was trained on, images it
was not), never on the unseen classes that are reported.  The table also marks the learning rate that is best on
`control` (classes left out of training, same source as the training images): the choice a reader makes who wants
the rule to carry over to new classes.  "Appears" is the first step at which accuracy is half-way
from chance to that run's final accuracy, on the straight line between the two evaluations around it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .summary import (HUE, INK2, LAYER, NAME, ORDER, RAMP, SURFACE, _end_labels, _head, _legend_below, _level,
                      _log_steps, _marks, _place, _style, crossing)


def load(path: str) -> dict | None:
    try:
        with open(os.path.join(path, "run.json")) as f:
            run = json.load(f)
        with open(os.path.join(path, "log.jsonl")) as f:
            rows = [json.loads(line) for line in f]
    except FileNotFoundError:
        return None
    evals = [r for r in rows if r["event"] == "eval"]
    trains = [r for r in rows if r["event"] == "train"]
    keys = sorted({k for r in evals for k in r} - {"event", "step"})
    a = run["args"]
    return {"path": path, "optimizer": run["optimizer"], "lr": a["lr"], "seed": a["seed"], "steps": a["steps"],
            "chance": run["chance"], "fixed": run["nearest_mean"], "encoder_step": run.get("encoder_step"),
            "ways": run["ways"], "shots": run["shots"], "feature": a.get("feature", "pre_logits"),
            # a pool is not asked at every evaluation (seq.py --watch), so each has its own steps
            "at": {k: np.array([r["step"] for r in evals if k in r]) for k in keys},
            "eval": {k: np.array([r[k] for r in evals if k in r]) for k in keys},
            "train_step": np.array([r["step"] for r in trains]), "loss": np.array([r["loss"] for r in trains]),
            "diverged": any(r["event"] == "diverged" for r in rows),
            "done": bool(evals) and evals[-1]["step"] == a["steps"]}


def appears(run: dict, pool: str) -> float | None:
    y = run["eval"][pool]
    if not run["done"] or y[-1] < run["chance"] + 0.05:
        return None
    return crossing(run["at"][pool], y, run["chance"] + 0.5 * (y[-1] - run["chance"]))


def onset(run: dict, pool: str = "seen") -> dict | None:
    """How the ability arrives in one run: the last training step whose loss is still within 2 % of ln N (the
    examples unused), and the steps at which accuracy on `pool` has made a tenth, half and nine tenths of its way
    from chance to its final value."""
    y = run["eval"][pool]
    if not run["done"] or y[-1] < run["chance"] + 0.05:
        return None
    flat = run["train_step"][run["loss"] > 0.98 * np.log(run["ways"])]
    out = {"flat_until": float(flat.max()) if len(flat) else 0.0}
    for name, part in (("tenth", 0.1), ("half", 0.5), ("most", 0.9)):
        out[name] = crossing(run["at"][pool], y, run["chance"] + part * (y[-1] - run["chance"]))
    return out


def onset_table(picks: list[tuple[str, dict]]) -> str:
    """The onset of the settings in `picks` ((what the choice was made on, {optimizer: (lr, runs)})), means over seeds."""
    rows, seen = [], set()
    for _, chosen in picks:
        for opt, (lr, rs) in chosen.items():
            got = [o for o in (onset(r) for r in rs if "seen" in r["eval"]) if o]
            if (opt, lr) in seen or not got:
                continue
            seen.add((opt, lr))
            rows.append(f"| {NAME.get(opt, opt)} | {lr:g} | " + " | ".join(
                f"{np.mean([o[k] for o in got]):,.0f}" for k in ("flat_until", "tenth", "half", "most")) + " |")
    if not rows:
        return ""
    return "\n".join(["## How the ability arrives, at the learning rates marked above (seen classes, means over seeds)",
                      "", "| optimizer | learning rate | training loss within 2 % of ln N until step | a tenth of the "
                      "final gain at step | half | nine tenths |", "|---|---|---|---|---|---|", *rows, ""])


def groups(runs: list[dict]) -> dict:
    """{(optimizer, lr): [runs, by seed]} for finished runs, in the order optimizer, learning rate."""
    out = {}
    for r in sorted(runs, key=lambda r: ((ORDER + [r["optimizer"]]).index(r["optimizer"]), r["lr"], r["seed"])):
        out.setdefault((r["optimizer"], r["lr"]), []).append(r)
    return out


def best(grouped: dict, pool: str = "seen") -> dict:
    """{optimizer: (lr, runs)} by the mean final accuracy on `pool` over the seeds that finished."""
    out = {}
    for (opt, lr), rs in grouped.items():
        done = [r for r in rs if r["done"] and pool in r["eval"]]
        if not done:
            continue
        score = float(np.mean([r["eval"][pool][-1] for r in done]))
        if opt not in out or score > out[opt][2]:
            out[opt] = (lr, done, score)
    return {opt: (lr, rs) for opt, (lr, rs, _) in out.items()}


def _ramp(opt: str, n: int) -> list[str]:
    if n == 1:
        return [HUE[opt]]
    if n == 2:
        return [RAMP[opt][3][0], RAMP[opt][3][2]]
    if n in RAMP[opt]:
        return RAMP[opt][n]
    raise SystemExit(f"seqsummary: {n} learning rates for {opt}; the ramps have at most five steps")


def _mean(rs: list[dict], key: str):
    return np.mean([r["eval"][key] for r in rs], axis=0)


def draw_emergence(plt, chosen: dict, pool: str, what: str, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9))
    fig.patch.set_facecolor(SURFACE)
    a, b, c = axes
    any_run = next(iter(chosen.values()))[1][0]
    ends = {id(a): [], id(b): []}
    for opt, (lr, rs) in chosen.items():
        colour = HUE.get(opt, INK2)
        for ax, key in ((a, pool), (b, "seen")):
            if key not in rs[0]["eval"]:
                continue
            at = rs[0]["at"][key]
            pos = at > 0
            for r in rs:
                ax.plot(at[pos], 100 * r["eval"][key][pos], color=colour, lw=0.7, alpha=0.45, zorder=2)
            m = 100 * _mean(rs, key)
            ax.plot(at[pos], m[pos], color=colour, lw=1.7, solid_capstyle="round", zorder=3,
                    label=f"{NAME.get(opt, opt)}, lr {lr:g}")
            ax.plot(at[-1:], m[-1:], ls="none", marker="o", ms=6.5, color=colour, mec=SURFACE, mew=1.4, zorder=5)
            ends[id(ax)].append((at[-1], m[-1], f"{m[-1]:.1f}"))
        for r in rs:
            c.plot(r["train_step"], r["loss"], color=colour, lw=0.7, alpha=0.45, zorder=2)
        c.plot(rs[0]["train_step"], np.mean([r["loss"] for r in rs], axis=0), color=colour, lw=1.7, zorder=3)
    n, k = any_run["ways"], any_run["shots"]
    _style(a, f"Unseen classes ({what}): accuracy (%)", "training step of the sequence model (log scale)")
    _style(b, "Classes the encoder was trained on: accuracy (%)", "training step of the sequence model (log scale)")
    _style(c, "Training loss", "training step of the sequence model (log scale)")
    for ax in axes:
        ax.set_xscale("log")
    for ax, key in ((a, pool), (b, "seen")):
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 3.2)
        lo, hi = ax.get_ylim()
        fixed = 100 * any_run["fixed"].get(key, 0.0)
        ax.set_ylim(min(lo, 100 * any_run["chance"] - 3), max(hi, fixed + 0.08 * (fixed - lo)))
        _level(ax, 100 * any_run["chance"], f"chance {100 * any_run['chance']:.0f} %")
        if key in any_run["fixed"]:
            _level(ax, fixed, f"nearest class mean on the same sequences {fixed:.1f} %", at="left")
    _level(c, float(np.log(n)), f"no use of the examples: ln {n}")
    for ax in axes:
        _log_steps(ax)
    seeds = max(len(rs) for _, rs in chosen.values())
    top = _head(fig, f"A transformer learning to use the examples it is shown, by optimizer ({n}-way, {k} examples each)",
                f"Trained on sequences of classes the encoder knows, with labels drawn afresh for every sequence; "
                f"tested on {what}, which neither model saw.\nEach optimizer at its best learning rate; thick: mean "
                f"of {seeds} seed{'s' if seeds > 1 else ''}, thin: single seeds.  Encoder: checkpoint at step "
                f"{any_run['encoder_step']:,}, features of {LAYER.get(any_run['feature'], any_run['feature'])}.",
                legend_from=a, ncol=len(chosen))
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=2.0)
    for ax in (a, b):
        _end_labels(ax, ends[id(ax)])
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def draw_sweep(plt, grouped: dict, pool: str, what: str, path: str) -> None:
    opts = [o for o in ORDER if any(k[0] == o for k in grouped)]
    fig, axes = plt.subplots(1, len(opts), figsize=(4.3 * len(opts), 4.6), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    any_run = next(iter(grouped.values()))[0]
    for ax, opt in zip(axes[0], opts):
        mine = [(lr, [r for r in rs if r["done"]]) for (o, lr), rs in grouped.items() if o == opt]
        mine = [(lr, rs) for lr, rs in mine if rs]
        ends = []
        for (lr, rs), colour in zip(mine, _ramp(opt, len(mine))):
            at = rs[0]["at"][pool]
            pos = at > 0
            m = 100 * _mean(rs, pool)
            ax.plot(at[pos], m[pos], color=colour, lw=1.7, solid_capstyle="round", zorder=3, label=f"{lr:g}")
            ax.plot(at[-1:], m[-1:], ls="none", marker="o", ms=6.5, color=colour, mec=SURFACE, mew=1.4, zorder=5)
            ends.append((at[-1], m[-1], f"{lr:g}"))
        _style(ax, f"{NAME.get(opt, opt)}: {what}, accuracy (%)", "training step of the sequence model (log scale)")
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 3.2)
        _log_steps(ax)
        _legend_below(ax)
        ax._ends = ends
    drawn = [y for ax in axes[0] for ln in ax.lines for y in np.atleast_1d(ln.get_ydata()) if np.isfinite(y)]
    chance, fixed = 100 * any_run["chance"], 100 * any_run["fixed"].get(pool, np.nan)
    hi = max(drawn + ([fixed] if np.isfinite(fixed) else []))
    axes[0][0].set_ylim(chance - 0.08 * (hi - chance) - 1, hi + 0.22 * (hi - chance))
    for ax in axes[0]:
        _level(ax, chance, f"chance {chance:.0f} %")
        if np.isfinite(fixed):
            _level(ax, fixed, f"nearest class mean {fixed:.1f} %", at="left")
    top = _head(fig, "Every learning rate of the sequence model, by optimizer",
                "Means over seeds.  Light to dark is low to high learning rate; the number at a line's end is its "
                "learning rate.")
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=1.6)
    for ax in axes[0]:
        _end_labels(ax, ax._ends, below=fixed if np.isfinite(fixed) else None)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def draw_tradeoff(plt, grouped: dict, pool: str, what: str, path: str) -> list[str]:
    """One point per optimizer and learning rate (mean over seeds, bars: their standard deviation), learning rates
    joined in order.  Settings whose seeds disagree by more than five points on the seen classes are left out and
    returned."""
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    fig.patch.set_facecolor(SURFACE)
    any_run = next(iter(grouped.values()))[0]
    unstable, labels, dots, strokes = [], [], [], []
    for opt in [o for o in ORDER if any(k[0] == o for k in grouped)]:
        pts = []
        for (o, lr), rs in grouped.items():
            done = [r for r in rs if r["done"] and "seen" in r["eval"]]
            if o != opt or not done:
                continue
            x = np.array([r["eval"]["seen"][-1] for r in done])
            y = np.array([r["eval"][pool][-1] for r in done])
            if x.std() > 0.05:
                unstable.append(f"{NAME.get(opt, opt)} lr {lr:g}")
                continue
            pts.append((lr, 100 * x.mean(), 100 * y.mean(), 100 * x.std(), 100 * y.std()))
        if not pts:
            continue
        lrs, xs, ys, xe, ye = (np.array(c) for c in zip(*pts))
        colour = HUE.get(opt, INK2)
        ax.plot(xs, ys, color=colour, lw=1.3, alpha=0.55, zorder=2)
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="none", ecolor=colour, elinewidth=0.8, alpha=0.6, zorder=3)
        ax.plot(xs, ys, ls="none", marker="o", ms=7, color=colour, mec=SURFACE, mew=1.4, zorder=4,
                label=NAME.get(opt, opt))
        labels += [(xs[i], ys[i], f"lr {lrs[i]:g}") for i in range(len(lrs))]
        p, s = _marks(xs, ys, xe, ye)
        dots, strokes = dots + p, strokes + s
    if "seen" in any_run["fixed"] and pool in any_run["fixed"]:
        fx, fy = 100 * any_run["fixed"]["seen"], 100 * any_run["fixed"][pool]
        ax.plot([fx], [fy], ls="none", marker="D", ms=7, color=INK2, mec=SURFACE, mew=1.4, zorder=4)
        labels.insert(0, (fx, fy, "nearest class mean (no training)"))
        dots.append((fx, fy))
    _style(ax, "", "accuracy on the classes the encoder was trained on (%)", f"accuracy on {what} (%)")
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.margins(0.12)
    top = _head(fig, "Fitting the trained classes against carrying over to unseen ones",
                f"Each point is a sequence model's final accuracy for one optimizer and learning rate (mean of the "
                f"seeds; bars: their spread).\nA line joins an optimizer's learning rates in order.  Up and to the "
                f"right is better on both."
                + (f"  Not drawn (seeds far apart): {', '.join(unstable)}." if unstable else ""),
                legend_from=ax, ncol=3)
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top))
    _place(ax, labels, dots, strokes)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return unstable


def table(grouped: dict, chosen: dict, pool: str, on_control: dict | None = None) -> str:
    any_run = next(iter(grouped.values()))[0]
    extra = [k for k in ("control",) if k != pool and k in any_run["eval"]]      # other unseen classes, same source
    lines = [f"| optimizer | learning rate | seeds finished | {pool}: final accuracy | "
             + f"{pool}: highest during training (at step) | "
             + "".join(f"{k}: final accuracy | " for k in extra) + "seen classes: final accuracy | "
             f"{pool}: ability appears at step | seen: appears at step |", "|---|---|---|---|---|---|---|---|" + "---|" * len(extra)]

    def peak(rs):
        if not rs:
            return "n/a"
        m = _mean(rs, pool)
        i = int(np.argmax(m))
        return f"{100 * m[i]:.1f} % ({int(rs[0]['at'][pool][i]):,})"

    def pm(v):
        return "n/a" if not len(v) else f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f} %"

    def steps(v):
        v = [x for x in v if x is not None]
        return "never" if not v else f"{np.mean(v):,.0f} ({min(v):,.0f} to {max(v):,.0f})"

    for (opt, lr), rs in grouped.items():
        done = [r for r in rs if r["done"]]
        mark = " **(best on seen)**" if opt in chosen and chosen[opt][0] == lr else ""
        if on_control and opt in on_control and on_control[opt][0] == lr:
            mark += " **(best on control)**"
        note = f"{len(done)} of {len(rs)}" + (", diverged: " + str(sum(r["diverged"] for r in rs)) if any(r["diverged"] for r in rs) else "")
        lines.append(f"| {NAME.get(opt, opt)}{mark} | {lr:g} | {note} | {pm([r['eval'][pool][-1] for r in done])} | "
                     + f"{peak(done)} | " + "".join(f"{pm([r['eval'][k][-1] for r in done if k in r['eval']])} | " for k in extra) +
                     f"{pm([r['eval']['seen'][-1] for r in done if 'seen' in r['eval']])} | "
                     f"{steps([appears(r, pool) for r in done])} | "
                     f"{steps([appears(r, 'seen') for r in done if 'seen' in r['eval']])} |")
    fixed = ", ".join(f"{k} {100 * v:.1f} %" for k, v in any_run["fixed"].items() if "@" not in k)
    return "\n".join([*lines, "", f"Chance is {100 * any_run['chance']:.0f} %.  The fixed rule (nearest class mean) on "
                      f"the same sequences: {fixed}.", "",
                      "Accuracy is the mean over seeds, ± their standard deviation; steps are the mean and the range "
                      "over seeds.  \"Best on seen\": the learning rate with the highest final accuracy on the seen "
                      "classes (the figures use it); \"best on control\": on the classes left out of training.", ""])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="dogs")
    ap.add_argument("--what", default="dog breeds")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    runs = [r for r in (load(p) for p in args.runs if os.path.isdir(p)) if r is not None]
    runs = [r for r in runs if args.pool in r["eval"]]
    if not runs:
        raise SystemExit(f"seqsummary: no run with results for the pool {args.pool!r}")
    os.makedirs(args.out, exist_ok=True)
    grouped = groups(runs)
    chosen = best(grouped)
    if not chosen:
        raise SystemExit("seqsummary: no run has finished")
    draw_emergence(plt, chosen, args.pool, args.what, os.path.join(args.out, "emergence.png"))
    draw_sweep(plt, grouped, args.pool, args.what, os.path.join(args.out, "sweep.png"))
    unstable = draw_tradeoff(plt, grouped, args.pool, args.what, os.path.join(args.out, "tradeoff.png"))
    on_control = best(grouped, "control") if "control" in runs[0]["eval"] and args.pool != "control" else None
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"# Sequence models on the features of the encoder at step {runs[0]['encoder_step']:,}\n\n"
                + table(grouped, chosen, args.pool, on_control) + "\n"
                + onset_table([("seen", chosen)] + ([("control", on_control)] if on_control else []))
                + (f"\nLeft out of tradeoff.png (seeds more than five points apart on the seen classes): "
                   f"{', '.join(unstable)}.\n" if unstable else ""))
    print(f"seqsummary: emergence.png, sweep.png, tradeoff.png, summary.md -> {args.out}; best on seen: "
          + ", ".join(f"{NAME.get(o, o)} lr {lr:g}" for o, (lr, _) in chosen.items())
          + ("" if not on_control else "; best on control: "
             + ", ".join(f"{NAME.get(o, o)} lr {lr:g}" for o, (lr, _) in on_control.items())))


if __name__ == "__main__":
    main()
