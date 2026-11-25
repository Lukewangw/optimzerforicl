"""The study's summary: three figures and a table, from runs that are finished or still under way.

Needs matplotlib (the Mac, not the cluster environment).

    python -m icl.summary --runs RUN [RUN ...] --out DIR [--pool dogs_5way] [--shots 5] [--feature pre_logits]

  best.png     each optimizer's best learning rate: validation accuracy on the trained classes and few-shot accuracy
               on the held-out pool against training progress, and the one against the other
  sweep.png    every learning rate, one column per optimizer
  pools.png    the best runs on other pools (more ways, other families, classes seen in training)
  final.png    where every finished run ended: validation accuracy against few-shot accuracy, a point per run
  layers.png   the same end points read from both feature layers side by side (the same file whatever --feature)
  summary.md   the numbers behind the figures, when the few-shot ability appears, and what each run cost

"Best" is the highest validation top-1 at the last checkpoint that the optimizer's runs have in common, so a figure
drawn while runs are under way compares like with like.  A run that is less than half as far as the furthest run of
its optimizer is drawn but not considered.

Colours: one hue per optimizer (the first three slots of the chart palette, which pass its colour-blindness checks
for every pair); within an optimizer the learning rates take a one-hue ramp, light for low and dark for high.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .curves import fewshot_series, load_run

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ORDER = ["adamw", "muon", "sgd"]
NAME = {"adamw": "AdamW", "muon": "Muon", "sgd": "SGD"}
HUE = {"adamw": "#2a78d6", "muon": "#eb6834", "sgd": "#1baf7a"}
RAMP = {  # light -> dark; each passes `validate_palette.js --ordinal` on the light surface
    "adamw": {3: ["#86b6ef", "#3987e5", "#184f95"], 4: ["#86b6ef", "#5598e7", "#256abf", "#104281"],
              5: ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]},
    "muon": {3: ["#ef9a70", "#eb6834", "#9c3d16"], 4: ["#ef9a70", "#eb6834", "#b84a1c", "#7d3010"],
             5: ["#ef9a70", "#eb6834", "#c4501f", "#963b15", "#662609"]},
    "sgd": {3: ["#5cc39e", "#1baf7a", "#0d6b4a"], 4: ["#5cc39e", "#1baf7a", "#128058", "#0a5239"],
            5: ["#5cc39e", "#1baf7a", "#149062", "#0d6b4a", "#064a32"]},
}
LAYER = {"pre_logits": "the layer the head reads (pre_logits)", "gap": "the pooled encoder output (gap)"}
POOLS = [  # the pools of pools.png, in reading order, with what each one is
    ("dogs_5way", "Dog breeds, 5 at a time"), ("dogs_20way", "Dog breeds, 20 at a time"),
    ("dogs_118way", "Dog breeds, all 118 at once"), ("wildcanids_5way", "Wolves, foxes and kin, 5 at a time"),
    ("cats", "Domestic cat classes, all 5"), ("birds_5way", "Bird species, 5 at a time"),
    ("inet_5way", "Any ImageNet classes, 5 at a time"), ("control", "Left-out Objects365 classes, all 20"),
    ("seen_5way", "Trained classes, 5 at a time"), ("seen_20way", "Trained classes, 20 at a time"),
]


def val_series(run: dict):
    """(steps, validation top-1) at the run's checkpoints."""
    recs = [r for r in run["ckpt"] if r.get("val_top1") is not None]
    return np.array([r["step"] for r in recs]), np.array([r["val_top1"] for r in recs])


def label_of(run: dict) -> str:
    return f"{NAME.get(run['optimizer'], run['optimizer'])}, lr {run['lr']:g}"


def best_runs(runs: list[dict]) -> dict:
    """{optimizer: (run, step it was picked at)}."""
    out = {}
    for opt in sorted({r["optimizer"] for r in runs}, key=lambda o: (ORDER + [o]).index(o)):
        mine = [r for r in runs if r["optimizer"] == opt and len(val_series(r)[0])]
        if not mine:
            continue
        furthest = max(val_series(r)[0][-1] for r in mine)
        mine = [r for r in mine if val_series(r)[0][-1] >= 0.5 * furthest]
        at = max(set.intersection(*(set(val_series(r)[0].tolist()) for r in mine)))
        out[opt] = (max(mine, key=lambda r: val_series(r)[1][val_series(r)[0].tolist().index(at)]), at)
    return out


def crossing(x, y, level: float):
    """The x at which y first reaches `level`, on the straight line between the two points around it; None if never."""
    for i in range(len(x)):
        if y[i] >= level:
            if i == 0 or y[i] == y[i - 1]:
                return float(x[i])
            return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return None


def value_at(x, y, where: float):
    """y on the path (x, y) where x first reaches `where` (x need not be monotone); None if it never does."""
    for i in range(len(x)):
        if x[i] >= where:
            if i == 0 or x[i] == x[i - 1]:
                return float(y[i])
            return float(y[i - 1] + (where - x[i - 1]) * (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
    return None


def emergence(steps, acc, ci) -> dict:
    """When the few-shot ability appears: the first checkpoint clearly above the untrained network (the two 95 %
    intervals apart), and the steps at which half and nine tenths of the gain so far were reached."""
    gain = acc[-1] - acc[0]
    above = next((int(steps[i]) for i in range(1, len(steps)) if acc[i] - ci[i] > acc[0] + ci[0]), None)
    if gain <= 0:
        return {"untrained": float(acc[0]), "last": float(acc[-1]), "first_above": above, "half": None, "most": None}
    return {"untrained": float(acc[0]), "last": float(acc[-1]), "first_above": above,
            "half": crossing(steps, acc, acc[0] + 0.5 * gain), "most": crossing(steps, acc, acc[0] + 0.9 * gain)}


# ---------------------------------------------------------------------------------------------------- drawing

def _style(ax, title: str, xlabel: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8, labelcolor=INK2)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=7)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)


def _line(ax, x, y, colour: str, *, label: str | None = None, band=None, lw: float = 1.6) -> None:
    ax.plot(x, y, color=colour, lw=lw, solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)
    if band is not None:
        ax.fill_between(x, y - band, y + band, color=colour, alpha=0.10, lw=0, zorder=2)
    ax.plot(x, y, ls="none", marker="o", ms=3.6, color=colour, mec=SURFACE, mew=0.7, zorder=4)
    ax.plot(x[-1:], y[-1:], ls="none", marker="o", ms=6.5, color=colour, mec=SURFACE, mew=1.4, zorder=5)


def _level(ax, y: float, text: str, at: str = "right") -> None:
    """A labelled hairline for a reference level (chance, the untrained network); the label sits at the end of the
    axes where no line is expected near that level."""
    ax.axhline(y, color=MUTED, lw=0.7, zorder=1)
    ax.annotate(text, xy=(0.988 if at == "right" else 0.012, y), xycoords=("axes fraction", "data"), xytext=(0, 2.5),
                textcoords="offset points", fontsize=7.5, color=MUTED, va="bottom", ha=at, zorder=6)


def _log_steps(ax) -> None:
    """Plain numbers at the decades of a log axis of training steps (10, 100, 1k, 10k)."""
    lo, hi = ax.get_xlim()
    ticks = [10 ** e for e in range(0, 8) if lo <= 10 ** e <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
    ax.minorticks_off()


def _end_labels(ax, items: list[tuple[float, float, str]], below: float | None = None) -> None:
    """Values at the line ends, right of the last point; labels that would overlap are moved apart and tied to their
    line with a hairline.  `below`: a reference level (data) whose hairline the labels stay under, when every
    labelled point lies under it."""
    fig = ax.figure
    fig.canvas.draw()
    px = fig.dpi / 72.0
    pts = sorted(((ax.transData.transform((x, y)), text) for x, y, text in items), key=lambda p: p[0][1])
    ys = [p[0][1] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 10.5 * px)
    shift = (sum(ys) - sum(p[0][1] for p in pts)) / max(1, len(ys))
    ys = [y - shift for y in ys]
    box = ax.get_window_extent()
    if ys:                                               # keep the column of labels inside the height of the axes
        top = box.y1 - 5 * px
        if below is not None:
            level = ax.transData.transform((0.0, below))[1]
            if all(p[0][1] < level for p in pts):
                top = min(top, level - 7 * px)
        over = max(0.0, ys[-1] - top)
        under = max(0.0, (box.y0 + 5 * px) - (ys[0] - over))
        ys = [y - over + under for y in ys]
    inv = ax.transData.inverted()
    for (xy, text), y in zip(pts, ys):
        moved = abs(y - xy[1]) > 2 * px
        ax.annotate(text, xy=inv.transform(xy), xytext=inv.transform((xy[0] + 9 * px, y)), fontsize=8, color=INK2,
                    va="center", ha="left", annotation_clip=False, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=1, shrinkB=4) if moved else None)


def _legend_below(ax, title: str = "learning rate") -> None:
    """The panel's legend in one row under its x-axis label, where no line and no end label can run into it."""
    from matplotlib.transforms import ScaledTranslation
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    under = ax.transAxes + ScaledTranslation(0, -34 / 72, ax.figure.dpi_scale_trans)
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, 0), bbox_transform=under, ncol=len(labels),
                    frameon=False, fontsize=8, handlelength=1.2, handletextpad=0.5, columnspacing=1.0,
                    borderaxespad=0.0, borderpad=0.0, title=title, title_fontsize=8)
    leg.get_title().set_color(INK2)
    leg._legend_box.align = "left"
    for t in leg.get_texts():
        t.set_color(INK2)


def _place(ax, labels: list[tuple[float, float, str]], points, segments=()) -> None:
    """A label beside each point (x, y, text), on the side where its box covers no marker of `points`, as little as
    possible of the lines and bars in `segments` (both in data coordinates, see `_marks`), and no label placed
    before it.  Call after the layout is final."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    dots = ax.transData.transform(np.asarray(points, dtype=float).reshape(-1, 2))
    runs = [np.zeros((0, 2))]
    for a, b in segments:                                    # a sample every three points of length along each
        a, b = ax.transData.transform(a), ax.transData.transform(b)
        runs.append(a + np.linspace(0, 1, max(2, int(np.hypot(*(b - a)) / (3 * px)) + 1))[:, None] * (b - a))
    strokes = np.concatenate(runs)

    def covered(p, bb, pad: float) -> int:
        return int(((p[:, 0] > bb.x0 - pad) & (p[:, 0] < bb.x1 + pad)
                    & (p[:, 1] > bb.y0 - pad) & (p[:, 1] < bb.y1 + pad)).sum())

    box = ax.get_window_extent()
    sides = [((7, 5), "left", "bottom"), ((7, -5), "left", "top"), ((-7, 5), "right", "bottom"),
             ((-7, -5), "right", "top"), ((0, 9), "center", "bottom"), ((0, -9), "center", "top"),
             ((10, 0), "left", "center"), ((-10, 0), "right", "center")]
    placed = []
    for x, y, text in labels:
        note = ax.annotate(text, xy=(x, y), xytext=(0, 0), textcoords="offset points", fontsize=7.5, color=INK2,
                           zorder=6)
        found = None
        for i, (off, ha, va) in enumerate(sides):
            note.xyann = off
            note.set_ha(ha)
            note.set_va(va)
            bb = note.get_window_extent(rend)
            cost = (40 * covered(dots, bb, 5 * px) + covered(strokes, bb, 1.5 * px)
                    + 100 * sum(bb.overlaps(o) for o in placed)
                    + 1000 * (bb.x0 < box.x0 or bb.x1 > box.x1 or bb.y0 < box.y0 or bb.y1 > box.y1))
            if found is None or cost < found[0]:
                found = (cost, i)
        off, ha, va = sides[found[1]]
        note.xyann = off
        note.set_ha(ha)
        note.set_va(va)
        placed.append(note.get_window_extent(rend))


def _marks(xs, ys, xe=None, ye=None, joined: bool = True) -> tuple[list, list]:
    """(points, segments) that a label should keep clear of: the markers, the line joining them, and their bars."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    segments = [((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(len(xs) - 1)] if joined else []
    for i in range(len(xs)):
        if xe is not None:
            segments.append(((xs[i] - xe[i], ys[i]), (xs[i] + xe[i], ys[i])))
        if ye is not None:
            segments.append(((xs[i], ys[i] - ye[i]), (xs[i], ys[i] + ye[i])))
    return list(zip(xs, ys)), segments


def _head(fig, title: str, sub: str, legend_from=None, ncol: int = 3) -> float:
    """Title, subtitle and (optionally) one legend for the whole figure, top left.  Returns the fraction of the
    figure's height left for the panels."""
    h = fig.get_figheight()
    fig.text(0.05, 1 - 0.16 / h, title, fontsize=12, color=INK, va="top", ha="left", weight="bold")
    fig.text(0.05, 1 - 0.50 / h, sub, fontsize=8.5, color=INK2, va="top", ha="left", linespacing=1.4)
    used = 0.50 + 0.185 * (sub.count("\n") + 1)
    if legend_from is not None:
        handles, labels = legend_from.get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.043, 1 - (used + 0.02) / h), ncol=ncol,
                         frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.8, borderaxespad=0.0)
        for t in leg.get_texts():
            t.set_color(INK2)
        used += 0.30
    return 1 - (used + 0.22) / h


def _pct(total: int):
    return lambda steps: 100.0 * np.asarray(steps, dtype=float) / total


def draw_best(plt, best: dict, pool: str, k: int, feature: str, n_way: int, what: str, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))
    fig.patch.set_facecolor(SURFACE)
    (a, b), (c, d) = axes
    total = next(iter(best.values()))[0]["total_steps"]
    pct = _pct(total)
    ends = {id(ax): [] for ax in (a, b, c)}
    untrained = None
    for opt, (run, _) in best.items():
        colour = HUE.get(opt, INK2)
        vs, vv = val_series(run)
        _line(a, pct(vs), 100 * vv, colour, label=label_of(run))
        ends[id(a)].append((pct(vs)[-1], 100 * vv[-1], f"{100 * vv[-1]:.1f}"))
        s = fewshot_series(run, feature, pool, k) if run["fewshot"] else None
        if s is None:
            continue
        steps, val, acc, ci = s
        untrained = acc[0] if steps[0] == 0 else untrained
        _line(b, pct(steps), 100 * acc, colour, band=100 * ci)
        ends[id(b)].append((pct(steps)[-1], 100 * acc[-1], f"{100 * acc[-1]:.1f}"))
        pos = steps > 0
        _line(c, steps[pos], 100 * acc[pos], colour, band=100 * ci[pos])
        ends[id(c)].append((steps[pos][-1], 100 * acc[pos][-1], f"{100 * acc[-1]:.1f}"))
        ok = ~np.isnan(val)
        _line(d, 100 * val[ok], 100 * acc[ok], colour)
    _style(a, "Validation accuracy on the classes it was trained on", "% of training", "top-1 (%)")
    _style(b, f"{what}: {n_way}-way, {k} examples each", "% of training", "accuracy (%)")
    _style(c, "The same on a log axis: the start of training", "training step (log scale)", "accuracy (%)")
    _style(d, "Few-shot accuracy at equal validation accuracy", "validation top-1 on trained classes (%)",
           "accuracy (%)")
    c.set_xscale("log")
    ticks = [t for t in (500, 1000, 2000, 5000, 10000, 20000, 50000, 100000) if t <= total]
    c.set_xticks(ticks)
    c.set_xticklabels([f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
    c.minorticks_off()
    for ax in (a, b):
        ax.set_xlim(-2, 113)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    c.set_xlim(c.get_xlim()[0], c.get_xlim()[1] * 2.6)
    for ax in (b, c, d):
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 100.0 / n_way - 2.5), hi)
        _level(ax, 100.0 / n_way, f"chance {100.0 / n_way:.0f} %")
    if untrained is not None:
        _level(c, 100 * untrained, f"untrained network {100 * untrained:.1f} %")
    done = max(val_series(r)[0][-1] for r, _ in best.values()) / total
    top = _head(fig, f"Few-shot accuracy on {what.lower()} the network never saw, by optimizer",
                f"ViT-S/16 trained from scratch without dogs; each optimizer at its best learning rate; a point is a "
                f"saved checkpoint.\nFrozen features of {LAYER.get(feature, feature)}, nearest class mean, 1,000 "
                f"episodes (shaded: 95 % interval)."
                + ("" if done >= 0.999 else f"  Runs under way: {100 * done:.0f} % of training so far."),
                legend_from=a, ncol=len(best))
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), h_pad=2.2, w_pad=2.0)
    for ax in (a, b, c):
        _end_labels(ax, ends[id(ax)])
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def draw_sweep(plt, runs: list[dict], pool: str, k: int, feature: str, n_way: int, what: str, path: str) -> None:
    opts = [o for o in ORDER if any(r["optimizer"] == o for r in runs)]
    fig, axes = plt.subplots(2, len(opts), figsize=(3.75 * len(opts), 6.9), sharey="row", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    total = runs[0]["total_steps"]
    pct = _pct(total)
    for j, opt in enumerate(opts):
        mine = sorted((r for r in runs if r["optimizer"] == opt), key=lambda r: r["lr"])
        ramp = RAMP[opt].get(len(mine)) or RAMP[opt][5]
        top, bottom = axes[0][j], axes[1][j]
        ends_top, ends_bottom = [], []
        for run, colour in zip(mine, ramp):
            vs, vv = val_series(run)
            s = fewshot_series(run, feature, pool, k) if run["fewshot"] else None
            if len(vs):
                _line(top, pct(vs), 100 * vv, colour)
                ends_top.append((pct(vs)[-1], 100 * vv[-1], f"{run['lr']:g}"))
            if s is not None:
                _line(bottom, pct(s[0]), 100 * s[2], colour, band=100 * s[3])
                ends_bottom.append((pct(s[0])[-1], 100 * s[2][-1], f"{run['lr']:g}"))
            bottom.plot([], [], color=colour, lw=1.6, label=f"{run['lr']:g}")      # one legend entry per run
        _style(top, f"{NAME[opt]}: validation top-1 (%)", "% of training")
        _style(bottom, f"{NAME[opt]}: {what.lower()}, {n_way}-way {k}-shot (%)", "% of training")
        for ax in (top, bottom):
            ax.set_xlim(-2, 118)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
        _legend_below(bottom)
        _level(bottom, 100.0 / n_way, f"chance {100.0 / n_way:.0f} %")
        top._ends, bottom._ends = ends_top, ends_bottom
    drawn = [y for ax in axes[1] for ln in ax.lines for y in np.atleast_1d(ln.get_ydata()) if np.isfinite(y)]
    if drawn:                                    # the row shares one y axis: limits from every column's lines
        span = max(drawn) - min(min(drawn), 100.0 / n_way)
        axes[1][0].set_ylim(min(min(drawn), 100.0 / n_way) - 0.08 * span, max(drawn) + 0.06 * span)
    head = _head(fig, "Every learning rate, by optimizer",
                 "Light to dark is low to high learning rate; the number at a line's end is its learning rate.  "
                 f"One seed each.  Few-shot on {LAYER.get(feature, feature)}.")
    fig.tight_layout(rect=(0.02, 0.0, 0.99, head), h_pad=2.2, w_pad=1.6)
    for row in axes:
        for ax in row:
            _end_labels(ax, ax._ends)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def draw_pools(plt, best: dict, k: int, feature: str, path: str) -> list[str]:
    have = [(p, t) for p, t in POOLS if any(r["fewshot"] and p in r["fewshot"]["pools"] for r, _ in best.values())]
    if not have:
        return []
    cols = 5 if len(have) > 8 else 4 if len(have) > 4 else len(have)
    rows = -(-len(have) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.25 * rows + 1.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    total = next(iter(best.values()))[0]["total_steps"]
    pct = _pct(total)
    for ax, (pool, title) in zip(axes.ravel(), have):
        ends, n_way = [], None
        for opt, (run, _) in best.items():
            s = fewshot_series(run, feature, pool, k) if run["fewshot"] else None
            if s is None:
                continue
            n_way = run["fewshot"]["pools"][pool]["n_way"]
            _line(ax, pct(s[0]), 100 * s[2], HUE.get(opt, INK2), label=label_of(run), band=100 * s[3])
            ends.append((pct(s[0])[-1], 100 * s[2][-1], f"{100 * s[2][-1]:.1f}"))
        _style(ax, title, "% of training")
        ax.set_xlim(-2, 124)
        ax.set_xticks([0, 25, 50, 75, 100])
        if n_way:
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, 100.0 / n_way - 0.04 * (hi - lo)), hi)
            _level(ax, 100.0 / n_way, f"chance {100.0 / n_way:.1f} %" if n_way > 20 else f"chance {100.0 / n_way:.0f} %")
        ax._ends = ends
    for ax in axes.ravel()[len(have):]:
        ax.set_visible(False)
    head = _head(fig, f"The same test on other groups of classes: accuracy (%) with {k} examples per class",
                 "Dogs, wolves and foxes were never in training, nor were the 20 left-out Objects365 classes.  Cats "
                 "and birds were, as the single classes Cat, Parrot, Pigeon and the like, not by breed or species.\n"
                 f"Features: {LAYER.get(feature, feature)}.",
                 legend_from=axes[0][0], ncol=len(best))
    fig.tight_layout(rect=(0.02, 0.0, 0.99, head), h_pad=2.0, w_pad=1.4)
    for ax in axes.ravel()[:len(have)]:
        _end_labels(ax, ax._ends)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return [p for p, _ in have]


def _ends(runs: list[dict], pool: str, k: int, feature: str) -> tuple[dict, float | None]:
    """({optimizer: [(lr, validation %, few-shot %, its half-width %)]} for the runs that finished and were tested at
    their last step, the untrained network's few-shot accuracy)."""
    total = runs[0]["total_steps"]
    pts, untrained = {}, None
    for run in runs:
        vs, vv = val_series(run)
        s = fewshot_series(run, feature, pool, k) if run["fewshot"] else None
        if s is None or not len(vs) or vs[-1] != total or s[0][-1] != total:
            continue
        untrained = s[2][0] if s[0][0] == 0 else untrained
        pts.setdefault(run["optimizer"], []).append((run["lr"], 100 * vv[-1], 100 * s[2][-1], 100 * s[3][-1]))
    return pts, untrained


def _ends_panel(ax, pts: dict, floor: float, title: str, ylabel: str | None) -> None:
    """The runs with at least `floor` validation top-1 as points, an optimizer's learning rates joined in order;
    their labels are placed by `_place` once the layout is final."""
    labels, dots, strokes = [], [], []
    for opt in sorted(pts, key=lambda o: (ORDER + [o]).index(o)):
        ps = sorted(p for p in pts[opt] if p[1] >= floor)
        if not ps:
            continue
        lrs, xs, ys, ye = (np.array(c) for c in zip(*ps))
        colour = HUE.get(opt, INK2)
        ax.plot(xs, ys, color=colour, lw=1.3, alpha=0.55, zorder=2)
        ax.errorbar(xs, ys, yerr=ye, fmt="none", ecolor=colour, elinewidth=0.8, alpha=0.6, zorder=3)
        ax.plot(xs, ys, ls="none", marker="o", ms=7, color=colour, mec=SURFACE, mew=1.4, zorder=4,
                label=NAME.get(opt, opt))
        labels += [(x, y, f"lr {lr:g}") for lr, x, y in zip(lrs, xs, ys)]
        p, s = _marks(xs, ys, ye=ye)
        dots, strokes = dots + p, strokes + s
    _style(ax, title, "validation top-1 on the trained classes (%)", ylabel)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.margins(0.16)
    ax._labels, ax._marks = labels, (dots, strokes)


def draw_final(plt, runs: list[dict], pool: str, k: int, feature: str, n_way: int, what: str, path: str,
               near: float = 8.0) -> bool:
    """Where each finished run ended: validation top-1 against few-shot accuracy, one labelled point per run.  The
    second panel is the same for the runs within `near` points of the best validation accuracy, where the runs
    that one would actually choose between lie."""
    pts, untrained = _ends(runs, pool, k, feature)
    if not pts:
        return False
    best_val = max(p[1] for ps in pts.values() for p in ps)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    fig.patch.set_facecolor(SURFACE)
    ylabel = f"{what.lower()}: {n_way}-way, {k} examples each (%)"
    _ends_panel(axes[0], pts, -np.inf, "Every finished run", ylabel)
    _ends_panel(axes[1], pts, best_val - near, f"The runs within {near:g} points of the best validation accuracy",
                ylabel)
    top = _head(fig, f"Where each run ended: accuracy on its own classes against few-shot accuracy on {what.lower()}",
                "One point per run (one seed each) with its learning rate; a line joins an optimizer's learning "
                "rates in order.\nBars: 95 % interval of the few-shot test (its episodes), not the spread between "
                f"training seeds.  Chance is {100.0 / n_way:.0f} %"
                + ("." if untrained is None else f", the untrained network {100 * untrained:.1f} %.")
                + f"  Features: {LAYER.get(feature, feature)}.",
                legend_from=axes[0], ncol=len(pts))
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=2.4)
    for ax in axes:
        _place(ax, ax._labels, *ax._marks)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def draw_layers(plt, runs: list[dict], pool: str, k: int, n_way: int, what: str, path: str, near: float = 8.0) -> bool:
    """The end points of the runs near the best validation accuracy, read from each of the two feature layers side
    by side on one accuracy axis: does the order of the runs depend on the layer?"""
    both = {f: _ends(runs, pool, k, f)[0] for f in LAYER}
    if not all(both.values()):
        return False
    best_val = max(p[1] for ps in both["pre_logits"].values() for p in ps)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, feature in zip(axes, LAYER):
        _ends_panel(ax, both[feature], best_val - near, f"Read from {LAYER[feature]}",
                    f"{what.lower()}: {n_way}-way, {k} examples each (%)" if ax is axes[0] else None)
    top = _head(fig, f"Few-shot accuracy on {what.lower()} at the end of training, read from two layers of the same "
                     f"networks",
                f"The runs within {near:g} points of the best validation accuracy, one seed each, with their "
                "learning rates; a line joins an optimizer's learning rates in order.\nBars: 95 % interval of the "
                "few-shot test (its episodes), not the spread between training seeds.",
                legend_from=axes[0], ncol=len(both["pre_logits"]))
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=2.0)
    for ax in axes:
        _place(ax, ax._labels, *ax._marks)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------------------------------- the table

def table(runs: list[dict], best: dict, pool: str, k: int, feature: str) -> str:
    total = runs[0]["total_steps"]
    chosen = {id(r) for r, _ in best.values()}
    asked = set().union(*(r["fewshot"]["pools"] for r in runs if r["fewshot"]))
    cols = [p for p in ("dogs_5way", "dogs_20way", "dogs_118way", "birds_5way", "inet_5way", "control", "seen_5way")
            if p != pool and p in asked]
    lines = [f"| run | steps done | val top-1 | {pool} {k}-shot | untrained | clearly above untrained from | "
             f"half of the gain by | 90 % of the gain by | " + " | ".join(f"{p} {k}-shot" for p in cols) + " |",
             "|---|---|---|---|---|---|---|---|" + "---|" * len(cols)]

    def pc(v):
        return "n/a" if v is None else f"{100 * v:.1f} %"

    def at(v):
        return "n/a" if v is None else f"step {v:,.0f} ({100 * v / total:.1f} %)"

    for run in sorted(runs, key=lambda r: ((ORDER + [r["optimizer"]]).index(r["optimizer"]), r["lr"])):
        vs, vv = val_series(run)
        s = fewshot_series(run, feature, pool, k) if run["fewshot"] else None
        name = label_of(run) + (" **(best)**" if id(run) in chosen else "")
        if s is None or not len(vs):
            lines.append(f"| {name} | {vs[-1] if len(vs) else 0:,} | {pc(vv[-1]) if len(vs) else 'n/a'} |" + " n/a |" * (5 + len(cols)))
            continue
        e = emergence(s[0], s[2], s[3])
        others = []
        for p in cols:
            o = fewshot_series(run, feature, p, k)
            others.append("n/a" if o is None else f"{100 * o[2][-1]:.1f} %")
        lines.append(f"| {name} | {int(s[0][-1]):,} ({100 * s[0][-1] / total:.0f} %) | {pc(vv[-1])} | "
                     f"{100 * s[2][-1]:.1f} ± {100 * s[3][-1]:.1f} % | {pc(e['untrained'])} | {at(e['first_above'])} | "
                     f"{at(e['half'])} | {at(e['most'])} | " + " | ".join(others) + " |")
    out = ["## Runs", "", *lines, "",
           "\"Gain\" is the rise from the untrained network to the run's last checkpoint; the steps for half and 90 % of "
           "it lie on the straight line between the two checkpoints around them.", ""]
    levels = [0.2, 0.3, 0.4, 0.5, 0.6]
    rows = []
    for opt, (run, _) in best.items():
        s = fewshot_series(run, feature, pool, k) if run["fewshot"] else None
        if s is None:
            continue
        ok = ~np.isnan(s[1])
        cells = [value_at(s[1][ok], s[2][ok], lv) for lv in levels]
        rows.append(f"| {label_of(run)} | " + " | ".join("not reached" if c is None else f"{100 * c:.1f} %" for c in cells) + " |")
    if rows:
        out += [f"## {pool} {k}-shot at equal validation top-1 (best learning rate of each optimizer)", "",
                "| run | " + " | ".join(f"at {100 * lv:.0f} %" for lv in levels) + " |", "|---|" + "---|" * len(levels),
                *rows, ""]
    return "\n".join(out)


def cost_table(runs: list[dict], pool: str, k: int, feature: str, baseline: tuple[str, float], gpus: int) -> str:
    """What each run cost and how early it reached what the baseline run ended with."""
    total = runs[0]["total_steps"]
    base = next((r for r in runs if r["optimizer"] == baseline[0] and abs(r["lr"] - baseline[1]) < 1e-12 * max(1, baseline[1])
                 and len(val_series(r)[0]) and val_series(r)[0][-1] == total), None)
    if base is None:
        return ""
    target_val = val_series(base)[1][-1]
    bs = fewshot_series(base, feature, pool, k) if base["fewshot"] else None
    target_fs = None if bs is None else bs[2][-1]
    lines = [f"## Cost, and how early each run reached what the baseline ({label_of(base)}) ended with", "",
             f"| run | images per second ({gpus} GPUs) | GPU-hours for the whole run | reached the baseline's final val top-1 "
             f"({100 * target_val:.1f} %) at | GPU-hours to that point | reached its final {pool} {k}-shot "
             + ("(n/a)" if target_fs is None else f"({100 * target_fs:.1f} %)") + " at | GPU-hours to that point |",
             "|---|---|---|---|---|---|---|"]
    for run in sorted(runs, key=lambda r: ((ORDER + [r["optimizer"]]).index(r["optimizer"]), r["lr"])):
        speed = [t["images_per_s"] for t in run["train"] if t.get("images_per_s")]
        vs, vv = val_series(run)
        if not speed or not len(vs) or vs[-1] != total:
            continue
        ips = float(np.median(speed))
        per_step = run["batch_size"] / ips * gpus / 3600.0                 # GPU-hours per step
        v_at = crossing(vs, vv, target_val)
        f_at = None
        if target_fs is not None and run["fewshot"]:
            fs = fewshot_series(run, feature, pool, k)
            f_at = None if fs is None else crossing(fs[0], fs[2], target_fs)
        cell = lambda x: "not reached" if x is None else f"step {x:,.0f} ({100 * x / total:.0f} % of training)"
        lines.append(f"| {label_of(run)} | {ips:,.0f} | {total * per_step:.1f} | {cell(v_at)} | "
                     + ("n/a" if v_at is None else f"{v_at * per_step:.1f}") + f" | {cell(f_at)} | "
                     + ("n/a" if f_at is None else f"{f_at * per_step:.1f}") + " |")
    return "\n".join(lines + ["", "The steps lie on the straight line between the two checkpoints around them. A run "
                              "stopped at that point is not a run planned to be that short: its learning "
                              "rate would still be high there. Whether a shorter run with its own schedule gets "
                              "as far is a separate run.", ""])


def transfer_table(runs: list[dict], k: int, baseline: tuple[str, float]) -> str:
    """Every finished run against the baseline run on the pools other than the trained classes themselves, read from
    each feature layer: on how many pools it is clearly ahead and clearly behind (by more than the two 95 %
    intervals combined), and the mean difference.  One pool or one layer alone can flatter a run; this counts."""
    total = runs[0]["total_steps"]

    def last(run: dict, feature: str) -> dict:
        ck = run["fewshot"]["checkpoints"][-1]
        if ck["step"] != total:
            return {}
        return {p: r[str(k)] for p, r in ck["results"].get(feature, {}).items()
                if not p.startswith("seen_") and run["fewshot"]["pools"][p]["n_way"] > 1 and str(k) in r}

    done = [r for r in runs if r["fewshot"]]
    base = next((r for r in done if r["optimizer"] == baseline[0] and abs(r["lr"] - baseline[1]) < 1e-12
                 and last(r, "pre_logits")), None)
    if base is None:
        return ""
    pools = sorted(last(base, "pre_logits"))
    lines = [f"## Against the baseline ({label_of(base)}) on the {len(pools)} pools other than the trained classes, "
             f"{k} examples per class", "",
             "| run | " + " | ".join(f"{f}: clearly ahead on | {f}: clearly behind on | {f}: mean difference"
                                     for f in LAYER) + " |", "|---|" + "---|---|---|" * len(LAYER)]
    for run in sorted(done, key=lambda r: ((ORDER + [r["optimizer"]]).index(r["optimizer"]), r["lr"])):
        if run is base:
            continue
        cells = []
        for f in LAYER:
            a, b = last(base, f), last(run, f)
            shared = [p for p in pools if p in a and p in b]
            if not shared:
                cells += ["n/a"] * 3
                continue
            d = [(b[p]["acc"] - a[p]["acc"], (a[p]["ci95"] ** 2 + b[p]["ci95"] ** 2) ** 0.5) for p in shared]
            cells += [str(sum(x > ci for x, ci in d)), str(sum(x < -ci for x, ci in d)),
                      f"{100 * float(np.mean([x for x, _ in d])):+.1f} points"]
        if cells.count("n/a") < len(cells):
            lines.append(f"| {label_of(run)} | " + " | ".join(cells) + " |")
    return "\n".join(lines + ["", "The pools: " + ", ".join(pools) + ". \"Clearly\": by more than the two runs' 95 % "
                                 "intervals combined (the intervals cover the episodes drawn, not the training seed).", ""])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="adamw:0.001", help="optimizer:learning rate of the recipe's own run")
    ap.add_argument("--gpus", type=int, default=4, help="GPUs per run, for the GPU-hours")
    ap.add_argument("--pool", default="dogs_5way")
    ap.add_argument("--what", default="Dog breeds", help="what the pool's classes are, for the titles")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--feature", default="pre_logits", choices=("pre_logits", "gap"))
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    runs = [load_run(p) for p in args.runs]
    runs = [r for r in runs if len(val_series(r)[0])]
    if not runs:
        raise SystemExit("summary: none of the runs has a checkpoint yet")
    os.makedirs(args.out, exist_ok=True)
    best = best_runs(runs)
    with_pool = [r for r in runs if r["fewshot"] and args.pool in r["fewshot"]["pools"]]
    if not with_pool:
        raise SystemExit(f"summary: no run has few-shot results for the pool {args.pool!r}")
    n_way = with_pool[0]["fewshot"]["pools"][args.pool]["n_way"]
    draw_best(plt, best, args.pool, args.shots, args.feature, n_way, args.what, os.path.join(args.out, "best.png"))
    draw_sweep(plt, runs, args.pool, args.shots, args.feature, n_way, args.what, os.path.join(args.out, "sweep.png"))
    pools = draw_pools(plt, best, args.shots, args.feature, os.path.join(args.out, "pools.png"))
    final = draw_final(plt, runs, args.pool, args.shots, args.feature, n_way, args.what,
                       os.path.join(args.out, "final.png"))
    layers = draw_layers(plt, runs, args.pool, args.shots, n_way, args.what, os.path.join(args.out, "layers.png"))
    text = table(runs, best, args.pool, args.shots, args.feature)
    b_opt, _, b_lr = args.baseline.partition(":")
    text += "\n" + transfer_table(runs, args.shots, (b_opt, float(b_lr)))
    text += "\n" + cost_table(runs, args.pool, args.shots, args.feature, (b_opt, float(b_lr)), args.gpus)
    picked = ", ".join(f"{label_of(r)} (picked at step {at:,})" for r, at in best.values())
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"# Summary ({args.feature} features)\n\nBest learning rates: {picked}.\n\n{text}")
    print(f"summary: best.png, sweep.png{', pools.png' if pools else ''}{', final.png' if final else ''}"
          f"{', layers.png' if layers else ''}, summary.md "
          f"-> {args.out}\n  best: {picked}")


if __name__ == "__main__":
    main()
