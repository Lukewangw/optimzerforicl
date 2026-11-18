"""One setting under several seeds, and a run planned shorter than the others of its setting.

Needs matplotlib.

    python -m icl.seeds --runs RUN [RUN ...] --out DIR [--baseline adamw:0.001] [--pool dogs_5way] [--shots 5]

A setting is an optimizer, a learning rate and a length of training; runs that differ only in their seed are one
setting.  Seed n starts every setting from the same weights and shows it the images in the same order, so the
settings are compared seed by seed.  Only finished runs are read: trained to their last step and tested there.

  seeds.png    where every run ended, setting by setting and seed by seed: validation top-1, and few-shot accuracy
               read from both feature layers
  curves.png   the same against training progress: the mean of a setting's seeds, shaded from its lowest to its
               highest seed
  short.png    when a run was planned with fewer steps than the others of its optimizer and learning rate: accuracy
               against GPU-hours for it, for the full-length setting and for the baseline
  seeds.md     the numbers: per setting; seed by seed against the baseline; the settings run once against the
               baseline's seeds; over all pools and both layers; how far the seeds of one setting lie apart; the
               short run

The baseline (--baseline optimizer:learning rate, at the usual length of training) is what the others are set
against.  A setting that shares several seeds with it is compared seed by seed, with Student's t on the differences;
a setting run once is compared with the interval in which one more seed of the baseline would fall.  With three
seeds both intervals are wide, and that is the point.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .curves import fewshot_series, load_run
from .summary import (HUE, INK2, LAYER, MUTED, NAME, ORDER, RAMP, SURFACE, _end_labels, _head, _level, _marks,
                      _place, _style, emergence, val_series)

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}  # by count of values
MEASURES = ("val", *LAYER)


def finished(run: dict) -> bool:
    """Trained to its last step and tested there."""
    vs, _ = val_series(run)
    fs = run["fewshot"]
    return bool(len(vs) and vs[-1] == run["total_steps"] and fs and fs["checkpoints"]
                and fs["checkpoints"][-1]["step"] == run["total_steps"])


def seed_name(seed: tuple) -> str:
    return str(seed[0]) if seed[0] == seed[1] else f"{seed[0]}/{seed[1]}"


def settings(runs: list[dict]) -> dict:
    """{(optimizer, learning rate, total steps): {(init seed, data seed): run}}: optimizers in the usual order, then
    by learning rate, the longer training first."""
    out = {}
    for r in runs:
        mine = out.setdefault((r["optimizer"], r["lr"], r["total_steps"]), {})
        if tuple(r["seeds"]) in mine:
            raise SystemExit(f"seeds: {r['path']} and {mine[tuple(r['seeds'])]['path']} are the same setting and seed")
        mine[tuple(r["seeds"])] = r
    order = sorted(out, key=lambda k: ((ORDER + [k[0]]).index(k[0]), k[1], -k[2]))
    return {k: dict(sorted(out[k].items())) for k in order}


def label(key: tuple, by_seed: dict, standard: int) -> str:
    text = f"{NAME.get(key[0], key[0])}, lr {key[1]:g}"
    if key[2] != standard:
        epochs = next(iter(by_seed.values())).get("epochs")
        text += f", {epochs:g} epochs" if epochs else f", {key[2]:,} steps"
    return text


def ends(run: dict, measure: str, pool: str, k: int):
    """(value, 95 % half-width of the test) at the run's last step; `measure` is "val" or a feature layer."""
    if measure == "val":
        return float(val_series(run)[1][-1]), 0.0
    r = run["fewshot"]["checkpoints"][-1]["results"].get(measure, {}).get(pool, {}).get(str(k))
    return None if r is None else (r["acc"], r["ci95"])


def values(by_seed: dict, measure: str, pool: str, k: int) -> list[float]:
    got = [ends(r, measure, pool, k) for r in by_seed.values()]
    return [g[0] for g in got if g is not None]


def differences(a: dict, b: dict, measure: str, pool: str, k: int) -> dict:
    """{seed: b - a} at the last step, for the seeds both settings have."""
    out = {}
    for seed in a:
        if seed in b:
            x, y = ends(a[seed], measure, pool, k), ends(b[seed], measure, pool, k)
            if x is not None and y is not None:
                out[seed] = y[0] - x[0]
    return out


def interval(vals) -> float | None:
    """Half-width of the 95 % interval of the mean of `vals` (Student's t); None for one value or more than ten."""
    n = len(vals)
    return T95[n] * float(np.std(vals, ddof=1)) / n ** 0.5 if n in T95 else None


def prediction(vals) -> float | None:
    """Half-width of the interval around the mean of `vals` in which one more value of their kind would fall 95 times
    in 100 (Student's t); None for one value or more than ten."""
    n = len(vals)
    return T95[n] * float(np.std(vals, ddof=1)) * (1 + 1 / n) ** 0.5 if n in T95 else None


def compare(base: dict, by_seed: dict, measure: str, pool: str, k: int):
    """(difference from the baseline, half-width of its 95 % interval or None) at the last step; None without data.
    A setting that shares several seeds with the baseline: the mean of the seed-by-seed differences, and Student's t
    on them.  A setting run once: its distance from the mean of the baseline's seeds, and the interval in which that
    distance would fall if the run were one more seed of the baseline."""
    d = list(differences(base, by_seed, measure, pool, k).values())
    if len(d) > 1:
        return float(np.mean(d)), interval(d)
    mine, theirs = values(by_seed, measure, pool, k), values(base, measure, pool, k)
    if len(mine) != 1 or not theirs:
        return None
    return mine[0] - float(np.mean(theirs)), prediction(theirs)


def clear(c) -> bool:
    """The 95 % interval of a comparison leaves out zero."""
    return c is not None and c[1] is not None and abs(c[0]) > c[1]


def gpu_hours_per_step(run: dict, gpus: int) -> float | None:
    speed = [t["images_per_s"] for t in run["train"] if t.get("images_per_s")]
    return run["batch_size"] / float(np.median(speed)) * gpus / 3600.0 if speed else None


def per_step(by_seed: dict, gpus: int) -> float | None:
    got = [h for h in (gpu_hours_per_step(r, gpus) for r in by_seed.values()) if h]
    return float(np.mean(got)) if got else None


def series(by_seed: dict, measure: str, pool: str, k: int):
    """(steps, mean, lowest, highest) over the setting's seeds at the checkpoints they share; None without data."""
    per = []
    for r in by_seed.values():
        if measure == "val":
            steps, vals = val_series(r)
        else:
            fs = fewshot_series(r, measure, pool, k)
            if fs is None:
                return None
            steps, vals = fs[0], fs[2]
        per.append({int(s): float(v) for s, v in zip(steps, vals)})
    steps = sorted(set.intersection(*(set(p) for p in per)))
    if not steps:
        return None
    m = np.array([[p[s] for s in steps] for p in per])
    return np.array(steps), m.mean(0), m.min(0), m.max(0)


def transfer_pools(run: dict) -> list[str]:
    """The pools other than the trained classes themselves."""
    return sorted(p for p, info in run["fewshot"]["pools"].items() if not p.startswith("seen_") and info["n_way"] > 1)


def short_pairs(groups: dict) -> list[tuple]:
    """(short setting, the longest setting of the same optimizer and learning rate), for every setting planned
    shorter than another."""
    out = []
    for key in groups:
        longer = [other for other in groups if other[:2] == key[:2] and other[2] > key[2]]
        if longer:
            out.append((key, max(longer, key=lambda o: o[2])))
    return out


# ---------------------------------------------------------------------------------------------------- drawing

def _panels(pool_title: str) -> list[tuple[str, str]]:
    return ([("val", "Validation top-1 on the trained classes (%)")]
            + [(f, f"{pool_title}, read from {f} (%)") for f in LAYER])


def _band(ax, x, mean, lo, hi, colour: str, label: str | None = None) -> None:
    ax.plot(x, mean, color=colour, lw=1.7, solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0, zorder=2)
    ax.plot(x, mean, ls="none", marker="o", ms=3.4, color=colour, mec=SURFACE, mew=0.7, zorder=4)
    ax.plot(x[-1:], mean[-1:], ls="none", marker="o", ms=6.5, color=colour, mec=SURFACE, mew=1.4, zorder=5)


def _floor(ax, n_way: int) -> None:
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 100.0 / n_way - 2.5), hi)
    _level(ax, 100.0 / n_way, f"chance {100.0 / n_way:.0f} %")


def draw_ends(plt, groups: dict, names: dict, pool: str, k: int, n_way: int, what: str, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(max(11.8, 4.6 + 1.76 * len(groups)), 5.0))
    fig.patch.set_facecolor(SURFACE)
    axes[2].sharey(axes[1])
    for ax, (measure, title) in zip(axes, _panels(f"{what}, {n_way}-way {k}-shot")):
        named = set()
        for j, (key, by_seed) in enumerate(groups.items()):
            got = [ends(r, measure, pool, k) for r in by_seed.values()]
            if any(g is None for g in got):
                continue
            ys, ye = 100 * np.array([g[0] for g in got]), 100 * np.array([g[1] for g in got])
            xs = j + (np.arange(len(ys)) - (len(ys) - 1) / 2) * 0.17
            colour, half = HUE.get(key[0], INK2), 0.3 if len(ys) > 1 else 0.13
            ax.plot([j - half, j + half], [ys.mean()] * 2, color=colour, lw=1.4, alpha=0.6, solid_capstyle="round",
                    zorder=2)
            if measure != "val":
                ax.errorbar(xs, ys, yerr=ye, fmt="none", ecolor=colour, elinewidth=0.8, alpha=0.6, zorder=3)
            ax.plot(xs, ys, ls="none", marker="o", ms=6.5, color=colour, mec=SURFACE, mew=1.3, zorder=4,
                    label=None if key[0] in named else NAME.get(key[0], key[0]))
            named.add(key[0])
            ax.annotate(f"{ys.mean():.1f}", xy=(j + half, ys.mean()), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK2, va="center", ha="left", zorder=6)
        _style(ax, title, "")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([names[key].replace(", ", "\n") for key in groups], fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(groups) - 0.2)
        ax.margins(y=0.2)
    top = _head(fig, "Where every run ended, setting by setting and seed by seed",
                "A dot is one training run; a setting's dots are its seeds, left to right, and the line is their mean "
                "(the number beside it).\nSeed n starts every setting from the same weights and the same order of "
                "images.  Bars on the few-shot panels: 95 % interval of the test's episodes.  "
                f"Chance is {100.0 / n_way:.0f} %.",
                legend_from=axes[0], ncol=len({key[0] for key in groups}))
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=2.2)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def draw_curves(plt, seeded: dict, names: dict, pool: str, k: int, n_way: int, what: str, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    axes[2].sharey(axes[1])
    for ax, (measure, title) in zip(axes, _panels(f"{what}, {n_way}-way {k}-shot")):
        ax._ends = []
        for key, by_seed in seeded.items():
            s = series(by_seed, measure, pool, k)
            if s is None:
                continue
            x = 100.0 * s[0] / key[2]
            _band(ax, x, 100 * s[1], 100 * s[2], 100 * s[3], HUE.get(key[0], INK2), names[key])
            ax._ends.append((x[-1], 100 * s[1][-1], f"{100 * s[1][-1]:.1f}"))
        _style(ax, title, "% of training")
        ax.set_xlim(-2, 114)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    for ax in axes[1:]:
        _floor(ax, n_way)
    seeds = max(len(v) for v in seeded.values())
    top = _head(fig, "The same settings under different seeds: accuracy as training goes on",
                f"Line: mean of a setting's seeds ({seeds} at most); shaded: from its lowest to its highest seed; a "
                "point is a saved checkpoint.\nFew-shot: frozen features, nearest class mean, classes the network "
                "never saw.",
                legend_from=axes[0], ncol=len(seeded))
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=2.0)
    for ax in axes:
        _end_labels(ax, ax._ends)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def draw_short(plt, groups: dict, names: dict, base_key: tuple, short: tuple, full: tuple, pool: str, k: int,
               n_way: int, what: str, gpus: int, path: str, depth: float = 12.0) -> bool:
    """Accuracy against GPU-hours over the last part of training: the short run, the full-length setting it was cut
    from, and the baseline.  The three panels show one stretch of training, from the checkpoint before the last
    line comes within `depth` points of where the lowest line ends on validation accuracy."""
    dark = RAMP.get(short[0], {}).get(3, [INK2] * 3)[2]
    rows = [(base_key, HUE.get(base_key[0], INK2))] if base_key != full else []
    rows += [(full, HUE.get(full[0], INK2)), (short, dark)]
    if any(per_step(groups[key], gpus) is None for key, _ in rows):
        return False
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    axes[2].sharey(axes[1])
    for ax, (measure, title) in zip(axes, _panels(f"{what}, {n_way}-way {k}-shot")):
        ax._labels, ax._dots, ax._strokes, ax._shown = [], [], [], []
        for key, colour in rows:
            s = series(groups[key], measure, pool, k)
            if s is None:
                continue
            x, y, lo, hi = s[0] * per_step(groups[key], gpus), 100 * s[1], 100 * s[2], 100 * s[3]
            _band(ax, x, y, lo, hi, colour, names[key] + ("" if len(groups[key]) == 1 else
                                                          f" (mean of {len(groups[key])} seeds)"))
            ax._labels.append((x[-1], y[-1], f"{y[-1]:.1f}"))
            dots, strokes = _marks(x, y)
            ax._dots, ax._strokes = ax._dots + dots, ax._strokes + strokes
            ax._shown.append((x, y, lo, hi))
        _style(ax, title, f"GPU-hours of training ({gpus} GPUs a run)")
    floor = min(y[-1] for _, y, _, _ in axes[0]._shown) - depth
    enter = max(x[max(0, int(np.argmax(y >= floor)) - 1)] for x, y, _, _ in axes[0]._shown)
    right = max(x[-1] for ax in axes for x, *_ in ax._shown)
    for group in (axes[:1], axes[1:]):                                    # the two few-shot panels share their y axis
        seen = [(lo[x >= enter].min(), hi[x >= enter].max()) for ax in group for x, _, lo, hi in ax._shown
                if (x >= enter).any()]
        if seen:
            lo, hi = min(a for a, _ in seen), max(b for _, b in seen)
            for ax in group:
                ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.18 * (hi - lo))
    for ax in axes:
        ax.set_xlim(enter, right + 0.10 * (right - enter))
    target = 100 * float(np.mean(values(groups[base_key], "val", pool, k)))
    axes[0].axhline(target, color=MUTED, lw=0.7, zorder=1)        # no text on it: the lines cross it where it matters
    top = _head(fig, f"A run planned shorter: {names[short]} against the full-length runs, by what it costs",
                "The last part of training.  Each line ends where its training ends; a point is a saved checkpoint; "
                f"shaded: from the lowest to the highest seed.  Hairline, first panel: where {names[base_key]} ends "
                f"({target:.1f} %).\nGPU-hours are steps times the measured time a step takes.  The short run has "
                "its own learning-rate schedule, which reaches zero at its last step.  "
                f"Chance on the few-shot panels is {100.0 / n_way:.0f} %.",
                legend_from=axes[0], ncol=len(rows))
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=2.0)
    for ax in axes:
        _place(ax, ax._labels, ax._dots, ax._strokes)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------------------------------- the tables

def _pts(v: float) -> str:
    """A difference in points with its sign; one that rounds to zero is +0.0."""
    return f"{round(100 * v, 1) + 0.0:+.1f}"


def _measure(measure: str, pool: str, k: int) -> str:
    return "val top-1" if measure == "val" else f"{pool} {k}-shot from {measure}"


def _span(vals: list[float]) -> str:
    return f"{100 * min(vals):.1f} to {100 * max(vals):.1f}" if len(vals) > 1 else "one seed"


def table_settings(groups: dict, names: dict, pool: str, k: int, gpus: int) -> list[str]:
    lines = ["## The settings", "",
             "| setting | seeds | val top-1, mean | lowest to highest | "
             + " | ".join(f"{pool} {k}-shot from {f}, mean | lowest to highest" for f in LAYER)
             + " | GPU-hours a run |", "|---|---|" + "---|---|" * len(MEASURES) + "---|"]
    for key, by_seed in groups.items():
        cells = []
        for measure in MEASURES:
            v = values(by_seed, measure, pool, k)
            cells += [f"{100 * np.mean(v):.1f} %", _span(v)] if v else ["n/a", "n/a"]
        hours = per_step(by_seed, gpus)
        lines.append(f"| {names[key]} | {', '.join(seed_name(s) for s in by_seed)} | " + " | ".join(cells)
                     + (" | n/a |" if hours is None else f" | {hours * key[2]:.1f} |"))
    return lines + ["", "The few-shot test itself has a 95 % interval of about half a point (its 1,000 episodes), the "
                        "same episodes for every run.", ""]


def table_pairs(groups: dict, names: dict, base_key: tuple, pool: str, k: int) -> list[str]:
    base = groups[base_key]
    lines = [f"## Seed by seed against {names[base_key]} (points)", "",
             "| setting | measure | " + " | ".join(f"seed {seed_name(s)}" for s in base)
             + " | mean | 95 % interval of the mean |", "|---|---|" + "---|" * len(base) + "---|---|"]
    rows = []
    for key, by_seed in groups.items():
        if key == base_key:
            continue
        for measure in MEASURES:
            d = differences(base, by_seed, measure, pool, k)
            if len(d) < 2:
                continue
            vals = list(d.values())
            mean, half = float(np.mean(vals)), interval(vals)
            rows.append(f"| {names[key]} | {_measure(measure, pool, k)} | "
                        + " | ".join(_pts(d[s]) if s in d else "" for s in base) + f" | {_pts(mean)} | "
                        + ("n/a" if half is None else f"{_pts(mean - half)} to {_pts(mean + half)}") + " |")
    if not rows:
        return []
    return lines + rows + ["", "A difference is the setting minus the baseline in runs of the same seed. The interval "
                               "is Student's t on those differences; it leaves out the few-shot test's own half "
                               "point.", ""]


def _once(ref: dict, ref_name: str, by_seed: dict, name: str, pool: str, k: int) -> list[str]:
    """Rows of a table that sets a setting run once against the seeds of `ref`."""
    rows = []
    for measure in MEASURES:
        c, mine, theirs = (compare(ref, by_seed, measure, pool, k), values(by_seed, measure, pool, k),
                           values(ref, measure, pool, k))
        if c is None or len(mine) != 1:
            continue
        mark = "**" if clear(c) else ""
        rows.append(f"| {name} | {_measure(measure, pool, k)} | {100 * mine[0]:.1f} % | {100 * np.mean(theirs):.1f} % "
                    f"| {mark}{_pts(c[0])}{mark} | " + ("n/a" if c[1] is None else f"±{100 * c[1]:.1f}") + " |")
    return rows


ONCE = ("The last column is how far from the mean of those seeds one more seed of the same setting would fall 95 "
        "times in 100 (Student's t on the seeds). A difference in bold lies outside it: more than another seed "
        "would explain.")


def table_single(groups: dict, names: dict, base_key: tuple, pool: str, k: int) -> list[str]:
    once = [key for key in groups if key != base_key and len(groups[key]) == 1]
    rows = [row for key in once for row in _once(groups[base_key], names[base_key], groups[key], names[key], pool, k)]
    if not rows:
        return []
    return [f"## The settings run once, against the seeds of {names[base_key]}", "",
            f"| setting | measure | this run | {names[base_key]}, mean of its seeds | difference (points) | "
            "one more seed would fall within (points) |", "|---|---|---|---|---|---|"] + rows + ["", ONCE, ""]


def table_pools(groups: dict, names: dict, base_key: tuple, k: int) -> list[str]:
    base = groups[base_key]
    pools = transfer_pools(next(iter(base.values())))
    got = {(key, f, p): compare(base, groups[key], f, p, k)
           for key in groups if key != base_key for f in LAYER for p in pools}
    others = [key for key in groups if key != base_key and any(got[key, f, p] for f in LAYER for p in pools)]
    if not others or not pools:
        return []
    lines = [f"## Over the {len(pools)} pools other than the trained classes, {k} examples per class, against "
             f"{names[base_key]}", "",
             "| setting | layer | runs | clearly ahead on | clearly behind on | neither on | mean difference |",
             "|---|---|---|---|---|---|---|"]
    for key in others:
        runs = f"{len(groups[key])} seeds" if len(groups[key]) > 1 else "1 run"
        for f in LAYER:
            cs = [c for c in (got[key, f, p] for p in pools) if c is not None]
            if not cs:
                continue
            ahead, behind = sum(clear(c) and c[0] > 0 for c in cs), sum(clear(c) and c[0] < 0 for c in cs)
            lines.append(f"| {names[key]} | {f} | {runs} | {ahead} | {behind} | {len(cs) - ahead - behind} | "
                         f"{_pts(float(np.mean([c[0] for c in cs])))} points |")
    lines += ["", "\"Clearly\" means that the 95 % interval of the difference leaves out zero. For a setting with "
                  "several seeds the difference is the mean of its seed-by-seed differences from the baseline, with "
                  "Student's t on them; for a setting run once it is the distance from the mean of the baseline's "
                  "seeds, with the interval in which one more seed of the baseline would fall. Of 20 pools on which "
                  "nothing differs, one would still count; and the pools overlap (three of them are dog breeds), so "
                  "the counts are not independent. The few-shot test's own half point is left out.", ""]
    for f in LAYER:
        lines += [f"### Read from {f}: difference in points ± half the width of its 95 % interval; bold where the "
                  "interval leaves out zero", "",
                  "| pool | " + " | ".join(names[key] for key in others) + " |", "|---|" + "---|" * len(others)]
        for p in pools:
            cells = []
            for key in others:
                c = got[key, f, p]
                mark = "**" if clear(c) else ""
                cells.append("n/a" if c is None else f"{mark}{_pts(c[0])}{mark}"
                             + ("" if c[1] is None else f" ± {100 * c[1]:.1f}"))
            lines.append(f"| {p} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def table_spread(seeded: dict, names: dict, pool: str, k: int) -> list[str]:
    if not seeded:
        return []
    lines = ["## How far the seeds of one setting lie apart (highest minus lowest, points)", "",
             "| setting | val top-1 | " + " | ".join(f"{pool} {k}-shot from {f}" for f in LAYER) + " | "
             + " | ".join(f"median over the pools, {f}" for f in LAYER) + " |", "|---|---|" + "---|" * (2 * len(LAYER))]
    for key, by_seed in seeded.items():
        pools = transfer_pools(next(iter(by_seed.values())))
        cells = []
        for measure in MEASURES:
            v = values(by_seed, measure, pool, k)
            cells.append(f"{100 * (max(v) - min(v)):.1f}" if len(v) > 1 else "n/a")
        for f in LAYER:
            spans = [max(v) - min(v) for v in (values(by_seed, f, p, k) for p in pools) if len(v) > 1]
            cells.append(f"{100 * float(np.median(spans)):.1f}" if spans else "n/a")
        lines.append(f"| {names[key]} | " + " | ".join(cells) + " |")
    return lines + ["", "This is the yardstick for two runs that differ in something else as well: a difference "
                        "smaller than these is not one.", ""]


def table_emergence(seeded: dict, names: dict, pool: str, k: int) -> list[str]:
    """When the few-shot ability appears in training, for every seed: `summary.emergence` on each run."""
    if not seeded:
        return []
    lines = [f"## When the few-shot ability appears ({pool} {k}-shot), in % of training, seed by seed", "",
             "| setting | layer | clearly above the untrained network from | half of the gain by | nine tenths by |",
             "|---|---|---|---|---|"]
    for key, by_seed in seeded.items():
        for f in LAYER:
            got = []
            for r in by_seed.values():
                s = fewshot_series(r, f, pool, k)
                if s is not None and len(s[0]) > 1:
                    got.append(emergence(s[0], s[2], s[3]))
            if got:
                cols = [", ".join("never" if e[name] is None else f"{100 * e[name] / key[2]:.1f}" for e in got)
                        for name in ("first_above", "half", "most")]
                lines.append(f"| {names[key]} | {f} | " + " | ".join(cols) + " |")
    return lines + ["", "\"Gain\" is the rise from the untrained network to the run's last checkpoint; the steps lie "
                        "on the straight line between the two checkpoints around them.", ""]


def table_short(groups: dict, names: dict, base_key: tuple, pairs: list[tuple], pool: str, k: int,
                gpus: int) -> list[str]:
    lines = []
    for short, full in pairs:
        rows = ([base_key] if base_key not in (short, full) else []) + [full, short]
        lines += [f"## The run planned shorter: {names[short]}", "",
                  "| setting | steps | GPU-hours a run | val top-1 | lowest to highest | "
                  + " | ".join(f"{pool} {k}-shot from {f} | lowest to highest" for f in LAYER) + " |",
                  "|---|---|---|" + "---|---|" * len(MEASURES)]
        for key in rows:
            cells = []
            for measure in MEASURES:
                v = values(groups[key], measure, pool, k)
                cells += [f"{100 * np.mean(v):.1f} %", _span(v)] if v else ["n/a", "n/a"]
            hours = per_step(groups[key], gpus)
            lines.append(f"| {names[key]} | {key[2]:,} | " + ("n/a" if hours is None else f"{hours * key[2]:.1f}")
                         + " | " + " | ".join(cells) + " |")
        base_v, short_v = values(groups[base_key], "val", pool, k), values(groups[short], "val", pool, k)
        hb, hs = per_step(groups[base_key], gpus), per_step(groups[short], gpus)
        if base_v and short_v and hb and hs:
            gap = 100 * (float(np.mean(short_v)) - float(np.mean(base_v)))
            cost = 100 * (hs * short[2] / (hb * base_key[2]) - 1)
            lines += ["", f"{names[short]} ends {abs(gap):.1f} points {'above' if gap >= 0 else 'below'} the mean of "
                          f"{names[base_key]} on validation top-1 (whose seeds run from {_span(base_v)}), for "
                          f"{abs(cost):.0f} % {'more' if cost >= 0 else 'fewer'} GPU-hours."]
        hf = per_step(groups[full], gpus)
        rows = _once(groups[full], names[full], groups[short], names[short], pool, k) if len(groups[full]) > 1 else []
        if rows and hf and hs:
            cost = 100 * (hs * short[2] / (hf * full[2]) - 1)
            lines += ["", f"Against its own setting at full length, for {abs(cost):.0f} % "
                          f"{'more' if cost >= 0 else 'fewer'} GPU-hours:", "",
                      f"| setting | measure | this run | {names[full]}, mean of its seeds | difference (points) | "
                      "one more seed would fall within (points) |", "|---|---|---|---|---|---|"] + rows + ["", ONCE]
        lines.append("")
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="adamw:0.001", help="optimizer:learning rate of the recipe's own setting")
    ap.add_argument("--gpus", type=int, default=4, help="GPUs per run, for the GPU-hours")
    ap.add_argument("--pool", default="dogs_5way")
    ap.add_argument("--what", default="Dog breeds", help="what the pool's classes are, for the titles")
    ap.add_argument("--shots", type=int, default=5)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    loaded = [load_run(p) for p in args.runs]
    runs = [r for r in loaded if finished(r)]
    waiting = [os.path.basename(os.path.normpath(r["path"])) for r in loaded if not finished(r)]
    if not runs:
        raise SystemExit("seeds: none of the runs is finished and tested at its last step")
    groups = settings(runs)
    lengths = [key[2] for key in groups for _ in groups[key]]
    standard = max(set(lengths), key=lengths.count)
    names = {key: label(key, by_seed, standard) for key, by_seed in groups.items()}
    b_opt, _, b_lr = args.baseline.partition(":")
    base_key = next((key for key in groups
                     if key[0] == b_opt and abs(key[1] - float(b_lr)) < 1e-12 and key[2] == standard), None)
    if base_key is None:
        raise SystemExit(f"seeds: no finished run of the baseline setting {args.baseline} at {standard:,} steps")
    with_pool = next((r for r in runs if args.pool in r["fewshot"]["pools"]), None)
    if with_pool is None:
        raise SystemExit(f"seeds: no run has few-shot results for the pool {args.pool!r}")
    n_way = with_pool["fewshot"]["pools"][args.pool]["n_way"]
    seeded = {key: by_seed for key, by_seed in groups.items() if len(by_seed) > 1}
    pairs = short_pairs(groups)

    os.makedirs(args.out, exist_ok=True)
    made = []
    if seeded:
        draw_ends(plt, groups, names, args.pool, args.shots, n_way, args.what, os.path.join(args.out, "seeds.png"))
        draw_curves(plt, seeded, names, args.pool, args.shots, n_way, args.what, os.path.join(args.out, "curves.png"))
        made += ["seeds.png", "curves.png"]
    for i, (short, full) in enumerate(pairs):
        name = "short.png" if i == 0 else f"short-{i + 1}.png"
        if draw_short(plt, groups, names, base_key, short, full, args.pool, args.shots, n_way, args.what, args.gpus,
                      os.path.join(args.out, name)):
            made.append(name)
    text = ["# Seeds and the shorter run", ""]
    if waiting:
        text += ["Not read, because not finished and tested at the last step: " + ", ".join(waiting) + ".", ""]
    text += table_settings(groups, names, args.pool, args.shots, args.gpus)
    text += table_pairs(groups, names, base_key, args.pool, args.shots)
    text += table_single(groups, names, base_key, args.pool, args.shots)
    text += table_pools(groups, names, base_key, args.shots)
    text += table_spread(seeded, names, args.pool, args.shots)
    text += table_emergence(seeded, names, args.pool, args.shots)
    text += table_short(groups, names, base_key, pairs, args.pool, args.shots, args.gpus)
    with open(os.path.join(args.out, "seeds.md"), "w") as f:
        f.write("\n".join(text))
    print(f"seeds: {', '.join(made + ['seeds.md'])} -> {args.out}: {len(groups)} settings, {len(runs)} finished runs"
          + (f", {len(waiting)} not finished" if waiting else ""))


if __name__ == "__main__":
    main()
