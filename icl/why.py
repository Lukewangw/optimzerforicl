"""Why the optimizers differ on classes the network never saw: where in the network the difference sits, and what
it goes with.

Needs matplotlib.

    python -m icl.why --runs RUN [RUN ...] --out DIR [--cka cka.json] [--baseline adamw:0.001]

Reads what probe.py wrote into each run (probe.json) and, when given, what cka.py wrote.  Only runs probed at their
last step are read.  A setting is an optimizer, a learning rate and a length of training; the figures show the
settings that have several seeds, as the mean of the seeds with a band from the lowest to the highest seed, and the
tables have every setting.  A run probed again with another ridge of the whitened rule (probe.py --ridge, written as
probe-*.json beside probe.json) adds a row to the table of ridges.

  depth.png     few-shot accuracy at every layer at the end of training, under both rules of probe.py
  last.png      the last three stages along training: what the step from `gap` to `pre_logits` costs on unseen
                classes under each rule, how saturated the tanh is, the stable rank of the `pre_logits` matrix
  shape.png     per layer at the end: how many directions the features use and how far apart the classes are
  classes.png   validation top-1 by how many training images a class has
  rare.png      the classes with the fewest and with the most training images, checkpoint by checkpoint
  weights.png   per kind of weight matrix: the stable rank of the matrix and of its change since the start
  blocks.png    block by block: the stable rank of the attention's and of the MLP's matrices and of their change
                since the start, how far each block moved from its start, and few-shot accuracy on unseen classes
                after each block against the baseline
  cka.png       how alike two runs are at every layer: two seeds of one setting, two settings on one seed
  rates.png     the learning rates of each optimizer side by side: validation accuracy, the shape of the encoder's
                output, and few-shot accuracy on trained and on unseen classes
  control-*.png a run that changes one thing in a setting (curves.VARIANTS), along training against that setting
                and the baseline: the layer the head reads, what it does to unseen classes, the rarest classes
  why.md        the numbers

The baseline (--baseline optimizer:learning rate) is what the others are set against, as in seeds.py: seed by seed
with Student's t where a setting shares several seeds with it, and against the interval for one more seed of the
baseline where a setting was run once.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import textwrap

import numpy as np

from .curves import load_run
from .seeds import _band, clear, interval, label, prediction, settings
from .summary import HUE, INK2, MUTED, NAME, SURFACE, _head, _style

RULES = (("plain", "the test's rule"), ("white", "whitened"))
GROUPS = (("dogs_5way", "Dog breeds, 5 at a time"), ("unseen", "Mean of the unseen pools"),
          ("seen_118way", "Trained classes, 118 at a time"))
STAGES = ("gap", "pre_act", "pre_logits")
FAMILIES = (("attn.q", "q"), ("attn.k", "k"), ("attn.v", "v"), ("attn.proj", "proj"), ("mlp.fc1", "fc1"),
            ("mlp.fc2", "fc2"), ("pre_logits", "pre_logits"), ("head", "head"), ("embedding", "patch\nembedding"))
UNSEEN_SET, SEEN_SET, DOG_SET = "inet", "seen", "dogs"       # the image sets of probe.py that the shape tables use
PARTS = (("attention", ("attn.q", "attn.k", "attn.v", "attn.proj")), ("MLP", ("mlp.fc1", "mlp.fc2")))
ONE_HUE = {"on the MLP only": "#9b3fa8", "on the attention only": "#8a5a2b"}   # runs that split a block's matrices
TRAINED = 0.6                                                # validation top-1 from which a run counts as trained
CONTROL = {"pre_logits by AdamW": "the `pre_logits` matrix is trained by AdamW instead of Muon, at the learning rate "
                                  "and decay of the AdamW part of a Muon run",
           "no pre_logits": "no `pre_logits` layer: the head reads `gap` directly",
           "on the MLP only": "Muon trains only the MLP's matrices (fc1, fc2) and AdamW everything else, the "
                              "attention's matrices and the `pre_logits` matrix included, at the same learning rate "
                              "and decay",
           "on the attention only": "Muon trains only the attention's matrices (q, k, v, proj) and AdamW everything "
                                    "else, the MLP's matrices and the `pre_logits` matrix included, at the same "
                                    "learning rate and decay"}
RIDGE = 1e-3                                                 # the whitened rule's ridge before probe.py recorded it
SAME_TEST = ("layers", "pools", "shots", "episodes", "query", "seed")   # what two probes must share to be compared


def load(path: str) -> dict:
    """A run with what probe.py wrote into it: "probe" (probe.json, or None) and "ridges", {ridge of the whitened
    rule: the run's last checkpoint as probed with it}, from probe.json and from every probe-*.json beside it that
    puts the same test to the last step."""
    run = load_run(path)
    run["probe"], run["ridges"] = None, {}
    p = os.path.join(path, "probe.json")
    if not os.path.exists(p):
        return run
    with open(p) as f:
        run["probe"] = main = json.load(f)
    found = []
    for other in sorted(glob.glob(os.path.join(path, "probe-*.json"))):
        with open(other) as f:
            q = json.load(f)
        if "ridge" in q and all(q.get(k) == main.get(k) for k in SAME_TEST):
            found.append(q)
    for q in found + [main]:                                  # probe.json last: it wins where two have its ridge
        if q["checkpoints"] and q["checkpoints"][-1]["step"] == run["total_steps"]:
            run["ridges"][float(q.get("ridge", RIDGE))] = q["checkpoints"][-1]
    return run


def probed(run: dict) -> bool:
    p = run["probe"]
    return bool(p and p["checkpoints"] and p["checkpoints"][-1]["step"] == run["total_steps"])


def last(run: dict) -> dict:
    return run["probe"]["checkpoints"][-1]


def unseen(run: dict) -> list[str]:
    """The pools of classes the network was not trained on."""
    return sorted(p for p, info in run["probe"]["pools"].items() if not p.startswith("seen_") and info["n_way"] > 1)


def acc(run: dict, ck: dict, rule: str, layer: str, group: str) -> float | None:
    """Few-shot accuracy at one checkpoint, layer and rule: of one pool, or the mean over the unseen pools."""
    k = str(run["probe"]["shots"][-1])
    rows = [ck["fewshot"][rule][layer].get(p, {}).get(k) for p in (unseen(run) if group == "unseen" else [group])]
    return float(np.mean([r["acc"] for r in rows])) if rows and all(rows) else None


def over_seeds(by_seed: dict, fn) -> list[float]:
    return [v for v in (fn(r) for r in by_seed.values()) if v is not None]


def compare(base: dict, by_seed: dict, fn):
    """seeds.compare for any number `fn` reads off a run: (difference from the baseline, half-width of its 95 %
    interval or None), seed by seed where the two settings share several seeds, else one run against the scatter
    of the baseline's seeds; None without data."""
    d = [fn(by_seed[s]) - fn(base[s]) for s in base if s in by_seed
         if fn(by_seed[s]) is not None and fn(base[s]) is not None]
    if len(d) > 1:
        return float(np.mean(d)), interval(d)
    mine, theirs = over_seeds(by_seed, fn), over_seeds(base, fn)
    if len(mine) != 1 or not theirs:
        return None
    return mine[0] - float(np.mean(theirs)), prediction(theirs)


def within(by_seed: dict, fn):
    """(mean, half-width of its 95 % interval or None) of a number each run of a setting gives, such as the
    difference between two of its layers."""
    vals = over_seeds(by_seed, fn)
    return (float(np.mean(vals)), interval(vals)) if vals else None


def ahead_behind(base: dict, by_seed: dict, of_pool, pools) -> tuple[int, int]:
    """On how many of `pools` a setting is clearly above the baseline and on how many clearly below it:
    of_pool(pool) is the number compare() reads off a run, and clearly means its 95 % interval leaves out zero."""
    found = [c for c in (compare(base, by_seed, of_pool(p)) for p in pools) if clear(c)]
    return sum(c[0] > 0 for c in found), sum(c[0] < 0 for c in found)


def ridges(by_seed: dict) -> list[float]:
    """The ridges of the whitened rule that every run of a setting was probed with at its last step."""
    return sorted(set.intersection(*(set(r["ridges"]) for r in by_seed.values())))


def at_ridge(ridge: float, layer: str, group: str):
    """fn(run): whitened few-shot accuracy at the last step as probed with `ridge`; None where the run lacks it."""
    return lambda r: acc(r, r["ridges"][ridge], "white", layer, group) if ridge in r["ridges"] else None


def controls(groups: dict) -> list[tuple[tuple, tuple]]:
    """[(setting with a variant, the setting it varies)], for every variant whose usual setting is there too."""
    return [(key, key[:3]) for key in groups if len(key) > 3 and key[:3] in groups]


def along(by_seed: dict, fn):
    """(% of training, mean, lowest, highest) of fn(run, checkpoint) over a setting's seeds, at the checkpoints
    all of them have; None when a value is missing."""
    per = [{c["step"]: fn(r, c) for c in r["probe"]["checkpoints"]} for r in by_seed.values()]
    steps = sorted(set.intersection(*(set(p) for p in per)))
    if not steps or any(p[s] is None for p in per for s in steps):
        return None
    m = np.array([[p[s] for s in steps] for p in per], dtype=float)
    total = next(iter(by_seed.values()))["total_steps"]
    return 100.0 * np.array(steps) / total, m.mean(0), m.min(0), m.max(0)


def by_frequency(run: dict, parts: int = 5, ck: dict | None = None) -> list[dict]:
    """Validation top-1 over `parts` groups of equally many trained classes, from the classes with the fewest
    training images to those with the most, at checkpoint `ck` (the last one if not given):
    [{"fewest", "most", "images", "top1"}]."""
    p = run["probe"]
    counts, per = np.array(p["train_images_per_class"]), np.array(p["scored"]["val"]["per_class"])
    hits = np.array((ck or last(run))["val"]["hits"])
    out = []
    for g in np.array_split(np.argsort(counts, kind="stable"), parts):
        n = int(per[g].sum())
        out.append({"fewest": int(counts[g].min()), "most": int(counts[g].max()), "images": n,
                    "top1": float(hits[g].sum() / n) if n else None})
    return out


def rare_common(by_seed: dict, parts: int = 5):
    """Validation top-1 (%) along a setting's training, mean of its seeds: (% of training, all images, the group
    of classes with the fewest training images, the group with the most); None where a checkpoint lacks them."""
    def group(i):
        return lambda r, c: None if by_frequency(r, parts, c)[i]["top1"] is None else 100 * by_frequency(r, parts, c)[i]["top1"]
    got = [along(by_seed, fn) for fn in (lambda r, c: 100 * c["val"]["top1"], group(0), group(parts - 1))]
    return None if any(g is None for g in got) else (got[0][0], got[0][1], got[1][1], got[2][1])


def at_level(x, y, level: float) -> float | None:
    """y where the curve x first reaches `level`, on the straight line between the two checkpoints around that
    point; None where x starts at or above the level or never gets there."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    reached = np.nonzero(x >= level)[0]
    if not len(reached) or reached[0] == 0:
        return None
    i = int(reached[0])
    t = (level - x[i - 1]) / (x[i] - x[i - 1])
    return float(y[i - 1] + t * (y[i] - y[i - 1]))


def macro(run: dict, which: str = "val") -> float | None:
    """Top-1 at the end with every class counting the same, over the classes that have images in `which`."""
    per, hits = np.array(run["probe"]["scored"][which]["per_class"]), np.array(last(run)[which]["hits"])
    return float(np.mean(hits[per > 0] / per[per > 0])) if (per > 0).any() else None


def family(ck: dict, fam: str, key: str, sub: str | None = None) -> float | None:
    """A weight statistic of probe.py averaged over the matrices of one kind (`attn.q` of every block, `head`)."""
    rows = [v for name, v in ck["weights"].items() if name == f"{fam}.weight" or name.endswith(f".{fam}.weight")]
    vals = [(v[sub][key] if sub else v[key]) for v in rows if sub is None or sub in v]
    return float(np.mean(vals)) if vals else None


def block_numbers(run: dict) -> list[int]:
    """The blocks probe.py read the output of, counting from 1."""
    return [int(n[5:]) for n in run["probe"]["layers"] if n.startswith("block")]


def in_block(ck: dict, block: int, fams, key: str, sub: str | None = None) -> float | None:
    """A weight statistic of probe.py averaged over some matrices of one block (`fams`: "attn.q", ...; blocks
    count from 1); None where one is missing."""
    rows = [ck["weights"].get(f"blocks.{block - 1}.{fam}.weight") for fam in fams]
    if any(v is None or (sub and sub not in v) for v in rows):
        return None
    return float(np.mean([v[sub][key] if sub else v[key] for v in rows]))


def moved(run: dict, block: int) -> float | None:
    """How far a block's six matrices moved over training: the norm of their change since the start over their
    norm at the start (both over the six together); None without the start."""
    first, end = run["probe"]["checkpoints"][0], last(run)
    if first["step"] != 0:
        return None
    fams = [f for _, part in PARTS for f in part]
    start = [first["weights"].get(f"blocks.{block - 1}.{f}.weight") for f in fams]
    now = [end["weights"].get(f"blocks.{block - 1}.{f}.weight") for f in fams]
    if any(v is None for v in start) or any(v is None or "since_first" not in v for v in now):
        return None
    return float(np.sqrt(sum(v["since_first"]["fro"] ** 2 for v in now) / sum(v["fro"] ** 2 for v in start)))


def lead(base: dict, rule: str, group: str = "unseen"):
    """fn(run, layer): few-shot accuracy (points) at the end of training at that layer, less the baseline's run of
    the same seeds; None where the baseline has no such run."""
    def fn(r: dict, layer: str) -> float | None:
        b = base.get(tuple(r["seeds"]))
        x = acc(r, last(r), rule, layer, group)
        y = None if b is None else acc(b, last(b), rule, layer, group)
        return None if x is None or y is None else 100.0 * (x - y)
    return fn


def splits(groups: dict) -> dict:
    """The runs that give Muon one kind of block matrix only (their variant says "only"), and the settings they
    vary: {key: runs by seed}, in the order of `groups`."""
    mine = {key for key in groups if len(key) > 3 and key[3].endswith(" only")}
    refs = {key[:3] for key in mine if key[:3] in groups}
    return {key: by_seed for key, by_seed in groups.items() if key in mine | refs}


def block_singles(groups: dict, drawn: dict, base_key: tuple) -> dict:
    """The settings blocks.png draws dashed (those not drawn with seeds): the other optimizers at the baseline's own
    learning rate and length, which sets the optimizer apart from the learning rate, and the runs that give Muon
    one kind of block matrix, with the settings they vary."""
    same = {key for key in groups if len(key) == 3 and key[0] != base_key[0] and abs(key[1] - base_key[1]) < 1e-12
            and key[2] == base_key[2]}
    keys = same | set(splits(groups))
    return {key: by_seed for key, by_seed in groups.items() if key in keys and key not in drawn}


def by_rate(groups: dict, standard: int, floor: float = TRAINED) -> dict:
    """{optimizer: [(learning rate, the setting's runs by seed)]} in rising order of the rate, for the settings of
    the usual length whose runs reached a validation top-1 of `floor` on average: a run that collapsed or was
    far from trained says nothing about which trained network transfers better."""
    out = {}
    for key, by_seed in groups.items():
        if len(key) == 3 and key[2] == standard and np.mean([last(r)["val"]["top1"] for r in by_seed.values()]) >= floor:
            out.setdefault(key[0], []).append((key[1], by_seed))
    return {opt: sorted(rows, key=lambda row: row[0]) for opt, rows in out.items()}


def _cka_name(run: dict) -> str:
    """The name cka.py gives a run's kept features: the run's folder and its last step."""
    return f"{os.path.basename(os.path.normpath(run['path']))}-{run['total_steps']}"


def cka_between(cka: dict, a: dict, b: dict) -> dict | None:
    """{layer: linear CKA between runs a and b}, or None where cka.py did not have both."""
    at = {name: i for i, name in enumerate(cka["files"])}
    i, j = at.get(_cka_name(a)), at.get(_cka_name(b))
    if i is None or j is None:
        return None
    return {layer: float(cka["between"][layer][i][j]) for layer in cka["layers"]}


def cka_lines(cka: dict, groups: dict) -> tuple[dict, dict]:
    """({setting: per-layer CKA between its seeds, mean over the pairs},
        {(setting a, setting b): per-layer CKA between the two on the same seed, mean over the seeds they share})."""
    at = {name: i for i, name in enumerate(cka["files"])}
    m = np.array([cka["between"][layer] for layer in cka["layers"]])                 # (layers, files, files)

    def row(run):
        return at.get(_cka_name(run))

    same, across, keys = {}, {}, list(groups)
    for key, by_seed in groups.items():
        rows = [row(r) for r in by_seed.values() if row(r) is not None]
        pairs = [(a, b) for i, a in enumerate(rows) for b in rows[i + 1:]]
        if pairs:
            same[key] = np.mean([m[:, a, b] for a, b in pairs], axis=0)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs = [(row(groups[a][s]), row(groups[b][s])) for s in groups[a] if s in groups[b]]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if pairs:
                across[a, b] = np.mean([m[:, x, y] for x, y in pairs], axis=0)
    return same, across


# ---------------------------------------------------------------------------------------------------- drawing

def _ticks(ax, layers: list[str]) -> None:
    names = ["emb" if n == "embed" else n[5:] if n.startswith("block") else n.replace("_", "-") for n in layers]
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(names, fontsize=7.5, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(layers) - 0.4)


def _by_layer(ax, drawn: dict, names: dict, layers: list[str], fn) -> None:
    """One banded line per setting over the layers; fn(run, layer) -> value or None."""
    for key, by_seed in drawn.items():
        m = np.array([[np.nan if fn(r, n) is None else fn(r, n) for n in layers] for r in by_seed.values()], dtype=float)
        _band(ax, np.arange(len(layers)), m.mean(0), m.min(0), m.max(0), HUE.get(key[0], INK2), names[key])
    _ticks(ax, layers)


def _along(ax, drawn: dict, names: dict, fn) -> None:
    for key, by_seed in drawn.items():
        s = along(by_seed, fn)
        if s is not None:
            _band(ax, s[0], s[1], s[2], s[3], HUE.get(key[0], INK2), names[key])
    ax.set_xlim(-2, 104)
    ax.set_xticks([0, 20, 40, 60, 80, 100])


def _zero(ax) -> None:
    ax.axhline(0.0, color=MUTED, lw=0.7, zorder=1)


def _finish(plt, fig, axes, title: str, sub: str, path: str, n: int) -> None:
    sub = "\n".join(textwrap.fill(line, width=int(13.6 * fig.get_figwidth())) for line in sub.split("\n"))
    top = _head(fig, title, sub, legend_from=np.ravel(axes)[0], ncol=max(1, n))
    fig.tight_layout(rect=(0.02, 0.0, 0.99, top), w_pad=2.0, h_pad=1.6)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _seeds_note(drawn: dict, dots: bool = False) -> str:
    n = max(len(v) for v in drawn.values())
    if n < 2:
        return "One run per setting."
    return (f"Dot: mean of a setting's seeds ({n} at most); bar: from its lowest to its highest seed." if dots
            else f"Line: mean of a setting's seeds ({n} at most); shaded: from its lowest to its highest seed.")


def draw_depth(plt, drawn: dict, names: dict, path: str) -> None:
    one = next(iter(next(iter(drawn.values())).values()))
    layers = one["probe"]["layers"]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.0))
    fig.patch.set_facecolor(SURFACE)
    for i, (rule, rule_name) in enumerate(RULES):
        for j, (group, title) in enumerate(GROUPS):
            _by_layer(axes[i, j], drawn, names, layers,
                      lambda r, n: None if acc(r, last(r), rule, n, group) is None else 100 * acc(r, last(r), rule, n, group))
            _style(axes[i, j], f"{title}: {rule_name} (%)", "layer" if i else "")
    k = one["probe"]["shots"][-1]
    _finish(plt, fig, axes, "Few-shot accuracy at every layer, at the end of training",
            f"Nearest class mean with {k} examples per class, on the features after each stage: the patch embedding, "
            "the 12 blocks (token stream, averaged), `gap` (final LayerNorm, averaged), `pre-act` (the pre_logits "
            "matrix) and `pre-logits` (its tanh, what the head reads).\nTop: the test's rule.  Bottom: features "
            "whitened first, which undoes any linear re-weighting of directions.  The dip at layer 12 in the top "
            "row: before the final LayerNorm a few directions carry almost all the variance, and the test's rule "
            "cannot see past them.\n" + _seeds_note(drawn),
            path, len(drawn))


def draw_last(plt, drawn: dict, names: dict, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8))
    fig.patch.set_facecolor(SURFACE)
    for ax, (rule, rule_name) in zip(axes[0], RULES):
        def cost(r, c, rule=rule):
            a, b = acc(r, c, rule, "pre_logits", "unseen"), acc(r, c, rule, "gap", "unseen")
            return None if a is None or b is None else 100 * (a - b)
        _along(ax, drawn, names, cost)
        _zero(ax)
        _style(ax, f"Unseen pools: `pre_logits` minus `gap`, {rule_name} (points)", "")
    _along(axes[1, 0], drawn, names,
           lambda r, c: 100 * c["tanh"][UNSEEN_SET]["over90"] if UNSEEN_SET in c["tanh"] else None)
    _style(axes[1, 0], "tanh units beyond ±0.9, on unseen images (%)", "% of training")
    _along(axes[1, 1], drawn, names, lambda r, c: c["weights"].get("pre_logits.weight", {}).get("srank"))
    _style(axes[1, 1], "Stable rank of the pre_logits matrix (of 384)", "% of training")
    _finish(plt, fig, axes, "The layer the head reads, as training goes on",
            "Top: what the step from the encoder's output (`gap`) to the layer the head reads (`pre_logits`) does to "
            "few-shot accuracy on unseen classes; below zero is a loss.\nBottom: how far the units of that layer sit "
            "in the flat ends of the tanh, and how evenly its matrix stretches its directions (higher: more "
            "evenly).  " + _seeds_note(drawn),
            path, len(drawn))


def draw_shape(plt, drawn: dict, names: dict, path: str) -> None:
    one = next(iter(next(iter(drawn.values())).values()))
    layers = one["probe"]["layers"]
    panels = [(UNSEEN_SET, "erank", 1.0, "Directions in use on unseen images (entropy rank)"),
              (SEEN_SET, "between", 100.0, "Variance between class means, trained classes (%)"),
              (DOG_SET, "between", 100.0, "Variance between class means, dog breeds (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.7))
    fig.patch.set_facecolor(SURFACE)
    for ax, (which, key, scale, title) in zip(axes, panels):
        def value(r, n, which=which, key=key, scale=scale):
            v = last(r)["shape"][n].get(which, {}).get(key)
            return None if v is None else scale * v
        _by_layer(ax, drawn, names, layers, value)
        _style(ax, title, "layer")
    _finish(plt, fig, axes, "The shape of the features at every layer, at the end of training",
            "Left: how many directions the features fill (1: a line; 384: all, evenly).  Middle and right: the share "
            "of the variance that lies between class means, on the features as the test's rule sees them (higher: "
            "classes are tighter and further apart).\nThe token stream before the final LayerNorm is dominated by "
            "a few directions, hence the dip at layer 12.  " + _seeds_note(drawn),
            path, len(drawn))


def _dots(ax, drawn: dict, names: dict, x_count: int, fn, bars=None) -> None:
    """Per position 0..x_count-1 one dot per setting (mean of fn(run, i) over its seeds) with a line from the
    lowest to the highest seed, or with `bars(key, i)` as a half-width."""
    width = 0.62 / max(1, len(drawn))
    for j, (key, by_seed) in enumerate(drawn.items()):
        colour = HUE.get(key[0], INK2)
        xs, ys, lo, hi = [], [], [], []
        for i in range(x_count):
            vals = [v for v in (fn(r, i) for r in by_seed.values()) if v is not None]
            if not vals:
                continue
            half = bars(key, i) if bars else None
            xs.append(i + (j - (len(drawn) - 1) / 2) * width)
            ys.append(float(np.mean(vals)))
            lo.append(ys[-1] - half if half is not None else min(vals))
            hi.append(ys[-1] + half if half is not None else max(vals))
        ax.vlines(xs, lo, hi, color=colour, lw=1.2, alpha=0.6, zorder=3)
        ax.plot(xs, ys, ls="none", marker="o", ms=6.0, color=colour, mec=SURFACE, mew=1.2, zorder=4, label=names[key])
    ax.set_xlim(-0.6, x_count - 0.4)
    ax.tick_params(axis="x", length=0)


def draw_classes(plt, drawn: dict, names: dict, base_key: tuple, path: str) -> None:
    one = next(iter(next(iter(drawn.values())).values()))
    parts = by_frequency(one)
    ticks = [f"{p['fewest']:,}–{p['most']:,}" for p in parts]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fig.patch.set_facecolor(SURFACE)
    _dots(axes[0], drawn, names, len(parts), lambda r, i: None if by_frequency(r)[i]["top1"] is None
          else 100 * by_frequency(r)[i]["top1"])
    _style(axes[0], "Validation top-1 (%)", "training images per class (five groups of equally many classes)")
    others = {k: v for k, v in drawn.items() if k != base_key}
    if base_key in drawn and others:
        cmp = {(k, i): compare(drawn[base_key], v, lambda r, i=i: by_frequency(r)[i]["top1"])
               for k, v in others.items() for i in range(len(parts))}
        width = 0.62 / len(drawn)
        for j, key in enumerate(drawn):
            if key == base_key:
                continue
            colour = HUE.get(key[0], INK2)
            xs = [i + (j - (len(drawn) - 1) / 2) * width for i in range(len(parts)) if cmp[key, i]]
            ys = [100 * cmp[key, i][0] for i in range(len(parts)) if cmp[key, i]]
            es = [100 * (cmp[key, i][1] or 0.0) for i in range(len(parts)) if cmp[key, i]]
            axes[1].errorbar(xs, ys, yerr=es, fmt="none", ecolor=colour, elinewidth=1.2, alpha=0.6, zorder=3)
            axes[1].plot(xs, ys, ls="none", marker="o", ms=6.0, color=colour, mec=SURFACE, mew=1.2, zorder=4)
        _zero(axes[1])
        axes[1].set_xlim(-0.6, len(parts) - 0.4)
        axes[1].tick_params(axis="x", length=0)
    _style(axes[1], f"Difference from {names.get(base_key, 'the baseline')} (points)",
           "training images per class (five groups of equally many classes)")
    for ax in axes:
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels(ticks, fontsize=8)
    _finish(plt, fig, axes, "Where the validation accuracy comes from: classes by how often they were trained on",
            "The trained classes in five groups of equally many classes, from those with the fewest training images "
            "to those with the most.\nLeft: a dot is the mean of a setting's seeds, the line its lowest and highest "
            "seed.  Right: seed-by-seed difference from the baseline with its 95 % interval.",
            path, len(drawn))


def draw_rare(plt, drawn: dict, names: dict, path: str) -> bool:
    one = next(iter(next(iter(drawn.values())).values()))
    parts = by_frequency(one)
    curves = {key: rare_common(by_seed) for key, by_seed in drawn.items()}
    if any(c is None for c in curves.values()):
        return False
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8))
    fig.patch.set_facecolor(SURFACE)
    for ax, i, title in ((axes[0], 0, "Rarest fifth of the classes: validation top-1 (%)"),
                         (axes[1], len(parts) - 1, "Commonest fifth: validation top-1 (%)")):
        _along(ax, drawn, names, lambda r, c, i=i: None if by_frequency(r, ck=c)[i]["top1"] is None
               else 100 * by_frequency(r, ck=c)[i]["top1"])
        _style(ax, title, "% of training")
    for key, (_, everything, rare, _) in curves.items():
        colour = HUE.get(key[0], INK2)
        axes[2].plot(everything, rare, color=colour, lw=1.7, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        axes[2].plot(everything, rare, ls="none", marker="o", ms=3.4, color=colour, mec=SURFACE, mew=0.7, zorder=4)
        axes[2].plot(everything[-1:], rare[-1:], ls="none", marker="o", ms=6.5, color=colour, mec=SURFACE, mew=1.4, zorder=5)
    _style(axes[2], "Rarest fifth against all images, checkpoint by checkpoint (%)", "validation top-1, all images (%)")
    _finish(plt, fig, axes, "Rare and common classes, checkpoint by checkpoint",
            "The trained classes in five groups of equally many classes by their number of training images: the "
            f"rarest fifth has {parts[0]['fewest']:,} to {parts[0]['most']:,} images per class, the commonest "
            f"{parts[-1]['fewest']:,} to {parts[-1]['most']:,}.  Validation top-1 within the group.\nRight: the "
            "rarest fifth set against the accuracy on all validation images at the same checkpoint; settings on "
            "one line share out a given accuracy between rare and common classes alike.  " + _seeds_note(drawn),
            path, len(drawn))
    return True


def _slug(text: str) -> str:
    return "-".join("".join(c if c.isalnum() or c == "_" else " " for c in text.lower()).split())


def _one(ax, x, y, colour: str, label: str | None = None) -> None:
    """A setting run once: a dashed line in the hue of the setting it varies."""
    ax.plot(x, y, color=colour, lw=1.7, dashes=(5, 2.2), label=label, zorder=3)
    ax.plot(x, y, ls="none", marker="o", ms=3.4, color=colour, mec=SURFACE, mew=0.7, zorder=4)
    ax.plot(x[-1:], y[-1:], ls="none", marker="o", ms=6.5, color=colour, mec=SURFACE, mew=1.4, zorder=5)


def _sv_ratio(ck: dict) -> float | None:
    sv = ck.get("sv", {}).get("pre_logits.weight")
    return float(sv[0] / np.median(sv)) if sv else None


def draw_control(plt, groups: dict, names: dict, key: tuple, ref: tuple, base_key: tuple, path: str) -> bool:
    """The control `key` along training against the setting it varies (`ref`) and the baseline."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    shown = [k for k in (base_key, ref) if k in groups and k != key]

    def cost(rule):
        def fn(r, c):
            a, b = acc(r, c, rule, "pre_logits", "unseen"), acc(r, c, rule, "gap", "unseen")
            return None if a is None or b is None else 100 * (a - b)
        return fn

    def rarest(r, c):
        v = by_frequency(r, ck=c)[0]["top1"]
        return None if v is None else 100 * v

    panels = [("tanh units beyond ±0.9, on unseen images (%)",
               lambda r, c: 100 * c["tanh"][UNSEEN_SET]["over90"] if UNSEEN_SET in c["tanh"] else None, False, False),
              ("Stable rank of the pre_logits matrix (of 384)",
               lambda r, c: c["weights"].get("pre_logits.weight", {}).get("srank"), False, False),
              ("Its largest singular value over the median one (log scale)", lambda r, c: _sv_ratio(c), True, False),
              ("Unseen pools: `pre_logits` minus `gap`, the test's rule (points)", cost("plain"), False, True),
              ("Unseen pools: `pre_logits` minus `gap`, whitened (points)", cost("white"), False, True),
              ("Validation top-1, the rarest fifth of the classes (%)", rarest, False, False)]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8))
    fig.patch.set_facecolor(SURFACE)
    drew = False
    for i, (ax, (title, fn, log, zero)) in enumerate(zip(np.ravel(axes), panels)):
        for k in shown:
            s = along(groups[k], fn)
            if s is not None:
                _band(ax, s[0], s[1], s[2], s[3], HUE.get(k[0], INK2), names[k])
        s = along(groups[key], fn)
        if s is not None:
            _one(ax, s[0], s[1], HUE.get(key[0], INK2), names[key])
            drew = True
        ax.set_xlim(-2, 104)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.minorticks_off()
        if zero:
            _zero(ax)
        _style(ax, title, "% of training" if i >= 3 else "")
    if not drew:
        plt.close(fig)
        return False
    seeds = max(len(groups[k]) for k in shown) if shown else 1
    _finish(plt, fig, axes, f"The control: {names[key]}",
            f"One run of {names[ref]} in which {CONTROL.get(key[3], key[3])}; everything else, its initial weights "
            "and order of images included, is that of the setting's run of the same seed.\nTop: the layer the head "
            "reads.  Bottom: what it does to few-shot accuracy on unseen classes (below zero is a loss), and the "
            f"classes with the fewest training images.  Lines: mean of a setting's seeds ({seeds} at most), shaded "
            "from its lowest to its highest seed; dashed: the one run.", path, len(shown) + 1)
    return True


def draw_weights(plt, drawn: dict, names: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    fig.patch.set_facecolor(SURFACE)
    for ax, (sub, title) in zip(axes, ((None, "Stable rank of the weight matrix"),
                                       ("since_first", "Stable rank of its change since the start"))):
        _dots(ax, drawn, names, len(FAMILIES), lambda r, i, sub=sub: family(last(r), FAMILIES[i][0], "srank", sub))
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels([f[1] for f in FAMILIES], fontsize=8)
        ax.set_yscale("log")
        ax.set_yticks([5, 10, 20, 50, 100, 200])
        ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
        ax.minorticks_off()
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 8.0), max(hi, 150.0))
        _style(ax, title, "kind of matrix (q, k, v, proj, fc1, fc2: mean over the 12 blocks)")
    _finish(plt, fig, axes, "How many directions the weights use, at the end of training",
            "Stable rank: the sum of the squared singular values over the largest one.  1: the matrix (or its "
            "change) is one direction; a 384 x 384 matrix at its random start has about 98, a 1536 x 384 one 172.\n"
            "In a Muon run the head and the patch embedding are trained by AdamW.  Log scale.  "
            + _seeds_note(drawn, dots=True), path, len(drawn))


def draw_blocks(plt, drawn: dict, names: dict, base_key: tuple, base: dict, single: dict, path: str) -> None:
    """Block by block at the end of training; `single`: settings run once, drawn dashed."""
    blocks = block_numbers(next(iter(next(iter(drawn.values())).values())))
    x = np.arange(len(blocks))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2))
    fig.patch.set_facecolor(SURFACE)
    panels = [(0, j, (lambda r, b, f=fams: in_block(last(r), b, f, "srank")),
               f"{part} ({', '.join(f.split('.')[1] for f in fams)}): stable rank") for j, (part, fams) in enumerate(PARTS)]
    panels += [(1, j, (lambda r, b, f=fams: in_block(last(r), b, f, "srank", "since_first")),
                f"{part}: stable rank of the change since the start") for j, (part, fams) in enumerate(PARTS)]
    panels += [(0, 2, lambda r, b: moved(r, b), "How far the block moved: |change| / |start|")]
    panels += [(1, 2, lambda r, b, fn=lead(base, "white"): fn(r, f"block{b}"),
                f"Unseen classes after the block, against {names[base_key]} (points)")]
    for i, j, fn, title in panels:
        ax = axes[i, j]
        for key, by_seed in drawn.items():
            if key == base_key and "against" in title:                 # the baseline against itself: the zero line
                continue
            m = np.array([[np.nan if fn(r, b) is None else fn(r, b) for b in blocks] for r in by_seed.values()],
                         dtype=float)
            if not np.isnan(m).all():
                _band(ax, x, np.nanmean(m, 0), np.nanmin(m, 0), np.nanmax(m, 0), HUE.get(key[0], INK2), names[key])
        for key, by_seed in single.items():
            r = next(iter(by_seed.values()))
            y = np.array([np.nan if fn(r, b) is None else fn(r, b) for b in blocks], dtype=float)
            if not np.isnan(y).all():
                _one(ax, x, y, ONE_HUE.get(key[3] if len(key) > 3 else "", HUE.get(key[0], INK2)), names[key])
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in blocks], fontsize=8)
        ax.set_xlim(-0.6, len(blocks) - 0.4)
        if "stable rank" in title:
            ax.set_ylim(bottom=0)
        else:
            _zero(ax) if "against" in title else ax.set_ylim(bottom=0)
        _style(ax, title, "block" if i == 1 else "")
    k = next(iter(next(iter(drawn.values())).values()))["probe"]["shots"][-1]
    _finish(plt, fig, axes, "Block by block, at the end of training",
            "Stable rank: the sum of the squared singular values over the largest one (a 384 x 384 matrix at its "
            "random start: about 98; 1536 x 384: about 172); for a kind of matrix, the mean over its matrices in the "
            "block.  How far the block moved: the norm of the change of its six matrices since the start, over their "
            f"norm at the start.\nBottom right: {k}-shot accuracy on the unseen pools (mean) on the block's output, "
            "whitened first, less that of the baseline's run on the same seeds, so the line climbs where a block "
            "adds to the lead and falls where it gives some back.\n" + _seeds_note(drawn)
            + ("  Dashed: a setting run once." if single else ""), path, len(drawn) + len(single))


def draw_cka(plt, cka: dict, groups: dict, names: dict, path: str) -> bool:
    same, across = cka_lines(cka, groups)
    if not same and not across:
        return False
    layers = cka["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(layers))
    for key, y in same.items():
        axes[0].plot(x, y, color=HUE.get(key[0], INK2), lw=1.7, marker="o", ms=3.6, mec=SURFACE, mew=0.7,
                     label=names[key], zorder=3)
    if same:
        level = np.array(list(same.values()))
        axes[1].fill_between(x, level.min(0), level.max(0), color=MUTED, alpha=0.22, lw=0, zorder=2,
                             label="two seeds of one setting (the left panel's range)")
    dashes = [(None, None), (5, 2), (1.5, 1.5), (6, 2, 1.5, 2)]
    for i, ((a, b), y) in enumerate(across.items()):
        axes[1].plot(x, y, color=INK2, lw=1.5, marker="o", ms=3.4, mec=SURFACE, mew=0.7, zorder=3,
                     dashes=dashes[i % len(dashes)], label=f"{names[a]}  vs  {names[b]}")
    for ax, title in zip(axes, ("Two seeds of the same setting", "Two settings on the same seed")):
        _ticks(ax, layers)
        _style(ax, f"{title}: linear CKA", "layer")
        ax.set_ylim(0.4, 1.02)
    leg = axes[1].legend(loc="lower left", frameon=False, fontsize=8, handlelength=2.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    _finish(plt, fig, axes, "How alike two networks are at every layer, at the end of training",
            f"Linear CKA between the features of the same {cka['images']:,} images (1: the same up to a rotation and a "
            "scale; unrelated features give about 0.09).  Left: runs that differ only in their seed.\nRight: runs "
            "that share seed, initial weights and order of images and differ in the optimizer; the grey band is the "
            "left panel's range.  The axis starts at 0.4.",
            path, len(same))
    return True


def draw_rates(plt, groups: dict, standard: int, path: str) -> bool:
    rates = by_rate(groups, standard)
    if not any(len(rows) > 1 for rows in rates.values()):
        return False
    gap = lambda rule, group: lambda r: 100 * acc(r, last(r), rule, "gap", group)
    panels = [("Validation top-1 (%)", lambda r: 100 * last(r)["val"]["top1"]),
              ("Directions in use at `gap`, unseen images", lambda r: last(r)["shape"]["gap"][UNSEEN_SET]["erank"]),
              ("Variance between class means at `gap`, trained classes (%)",
               lambda r: 100 * last(r)["shape"]["gap"][SEEN_SET]["between"]),
              (f"{GROUPS[2][1]}, {RULES[0][1]} (%)", gap("plain", GROUPS[2][0])),
              (f"{GROUPS[1][1]}, {RULES[0][1]} (%)", gap("plain", "unseen")),
              (f"{GROUPS[1][1]}, {RULES[1][1]} (%)", gap("white", "unseen"))]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6))
    fig.patch.set_facecolor(SURFACE)
    shown = sorted({lr for rows in rates.values() for lr, _ in rows})
    for ax, (title, fn) in zip(np.ravel(axes), panels):
        for opt, rows in rates.items():
            vals = [over_seeds(by_seed, fn) for _, by_seed in rows]
            xs, ys = [lr for lr, _ in rows], [float(np.mean(v)) for v in vals]
            colour = HUE.get(opt, INK2)
            ax.plot(xs, ys, color=colour, lw=1.7, solid_capstyle="round", zorder=3,
                    label=NAME.get(opt, opt))
            ax.vlines(xs, [min(v) for v in vals], [max(v) for v in vals], color=colour, lw=1.2, alpha=0.6, zorder=3)
            ax.plot(xs, ys, ls="none", marker="o", ms=6.0, color=colour, mec=SURFACE, mew=1.2, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(shown)
        ax.set_xticklabels([f"{lr:g}" for lr in shown], fontsize=7.5, rotation=55, ha="right", rotation_mode="anchor")
        ax.minorticks_off()
        ax.margins(x=0.08, y=0.16)
        _style(ax, title, "learning rate (log scale)")
    _finish(plt, fig, axes, "The learning rate, optimizer by optimizer, at the end of training",
            "Every setting of the usual length whose runs reached a validation top-1 of "
            f"{100 * TRAINED:.0f} % or more; a dot is a setting (the mean of its seeds, with a bar from the lowest to "
            "the highest where there are several).\nAll read at `gap`, the encoder's output: how many directions "
            "its features fill, how tightly the trained classes sit around their means, and few-shot accuracy on "
            "trained and on unseen classes.", path, len(rates))
    return True


# ---------------------------------------------------------------------------------------------------- tables

def _pct(v: float | None, digits: int = 1) -> str:
    return "n/a" if v is None else f"{100 * v:.{digits}f}"


def _num(v: float | None, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _mean(by_seed: dict, fn) -> float | None:
    vals = over_seeds(by_seed, fn)
    return float(np.mean(vals)) if vals else None


def _diff(c, scale: float = 100.0) -> str:
    """A difference (in points by default: `scale` times the value) with its half-width, bold where the interval
    leaves out zero."""
    if c is None:
        return "n/a"
    text = f"{round(scale * c[0], 1) + 0.0:+.1f}"             # + 0.0: no "-0.0"
    return (f"**{text}**" if clear(c) else text) + (f" ± {scale * c[1]:.1f}" if c[1] is not None else "")


def _runs(by_seed: dict) -> str:
    return f"{len(by_seed)} seeds" if len(by_seed) > 1 else "1 run"


def table_control(groups: dict, names: dict, base_key: tuple, cka: dict | None = None) -> list[str]:
    """Every control (a setting with a variant) at the end of training, against the setting it varies and against
    the baseline; nothing where there is none."""
    def fs(rule, layer, group="unseen"):
        return lambda r: acc(r, last(r), rule, layer, group)

    def step(rule, a, b):
        return lambda r: None if fs(rule, a)(r) is None or fs(rule, b)(r) is None else fs(rule, b)(r) - fs(rule, a)(r)

    def matrix(k):
        return lambda r: last(r)["weights"].get("pre_logits.weight", {}).get(k)

    def blocks(r):
        vals = [v for v in (family(last(r), fam, "srank") for fam, _ in FAMILIES[:6]) if v is not None]
        return float(np.mean(vals)) if vals else None

    rows = [("Validation top-1 (%)", lambda r: last(r)["val"]["top1"], 100),
            ("Validation top-1, every class the same (%)", macro, 100),
            ("Validation top-1, the rarest fifth of the classes (%)", lambda r: by_frequency(r)[0]["top1"], 100),
            ("Validation top-1, the commonest fifth (%)", lambda r: by_frequency(r)[-1]["top1"], 100),
            ("Unseen pools at `gap`, the test's rule (%)", fs("plain", "gap"), 100),
            ("Unseen pools at `gap`, whitened (%)", fs("white", "gap"), 100),
            ("Unseen pools at `pre_logits`, the test's rule (%)", fs("plain", "pre_logits"), 100),
            ("Unseen pools at `pre_logits`, whitened (%)", fs("white", "pre_logits"), 100),
            ("Dog breeds, 5 at a time, at `gap`, whitened (%)", fs("white", "gap", "dogs_5way"), 100),
            ("Dog breeds, 5 at a time, at `pre_logits`, whitened (%)", fs("white", "pre_logits", "dogs_5way"), 100),
            ("What `gap` → `pre_logits` does to the unseen pools, the test's rule (points)",
             step("plain", "gap", "pre_logits"), 100),
            ("The same, whitened (points)", step("white", "gap", "pre_logits"), 100),
            ("Of which the tanh, whitened (points)", step("white", "pre_act", "pre_logits"), 100),
            ("tanh units beyond ±0.9, on unseen images (%)",
             lambda r: last(r)["tanh"].get(UNSEEN_SET, {}).get("over90"), 100),
            ("`pre_logits` matrix: stable rank", matrix("srank"), 1),
            ("`pre_logits` matrix: largest over median singular value", lambda r: _sv_ratio(last(r)), 1),
            ("`pre_logits` matrix: Frobenius norm", matrix("fro"), 1),
            ("Block matrices: stable rank, mean over q, k, v, proj, fc1, fc2", blocks, 1),
            ("Directions in use on unseen images at `gap` (entropy rank)",
             lambda r: last(r)["shape"]["gap"].get(UNSEEN_SET, {}).get("erank"), 1),
            ("The same at `pre_logits`", lambda r: last(r)["shape"]["pre_logits"].get(UNSEEN_SET, {}).get("erank"), 1)]
    out = []
    for key, ref in controls(groups):
        mine, theirs, base = groups[key], groups[ref], groups.get(base_key, {})
        shared = next((s for s in mine if s in theirs), None)
        head = [names[ref] + f" ({_runs(theirs)})", names[key] + f" ({_runs(mine)})", f"against {names[ref]}"]
        if shared is not None:
            head.append(f"against its seed {shared[0]} alone")
        if base and base_key != ref:
            head += [names[base_key] + f" ({_runs(base)})", f"against {names[base_key]}"]
        out += [f"## The control: {names[key]}", "",
                f"{_runs(mine).capitalize()} of {names[ref]} in which {CONTROL.get(key[3], key[3])}; everything else, "
                "its initial weights and order of images included, is that of the setting's run of the same seed.  "
                "End of training.  \"Against\": the distance from the mean of a setting's seeds, ± the half-width of "
                "the interval in which one more seed of that setting would fall 95 times in 100, bold where the "
                "distance lies outside it" + ("" if shared is None else
                f"; \"its seed {shared[0]} alone\": the difference from the run that started from the same weights "
                "and saw the same images in the same order") + ".", "",
                "| | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
        for title, fn, scale in rows:
            cells = [_num(None if _mean(theirs, fn) is None else scale * _mean(theirs, fn)),
                     _num(None if _mean(mine, fn) is None else scale * _mean(mine, fn)),
                     _diff(compare(theirs, mine, fn), scale)]
            if shared is not None:
                a, b = fn(mine[shared]), fn(theirs[shared])
                cells.append("n/a" if a is None or b is None else _diff((a - b, None), scale))
            if base and base_key != ref:
                cells += [_num(None if _mean(base, fn) is None else scale * _mean(base, fn)),
                          _diff(compare(base, mine, fn), scale)]
            out.append(f"| {title} | " + " | ".join(cells) + " |")
        out.append("")
        if cka and shared is not None:
            def pair(a, b):
                c = cka_between(cka, a, b)
                return None if c is None else f"{c['gap']:.2f} / {c['pre_logits']:.2f}"
            seeds = list(theirs)
            seeds_apart = [cka_between(cka, theirs[a], theirs[b]) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
            seeds_apart = [c for c in seeds_apart if c is not None]
            lines = [f"the control and {names[ref]}'s seed {shared[0]}: {pair(mine[shared], theirs[shared])}"]
            if shared in base:
                lines.append(f"the control and {names[base_key]}'s seed {shared[0]}: {pair(mine[shared], base[shared])}")
            if seeds_apart:
                lines.append(f"for comparison, two seeds of {names[ref]}: "
                             f"{np.mean([c['gap'] for c in seeds_apart]):.2f} / {np.mean([c['pre_logits'] for c in seeds_apart]):.2f}")
            if shared in base:
                lines.append(f"{names[ref]} and {names[base_key]} on seed {shared[0]}: {pair(theirs[shared], base[shared])}")
            out += ["How alike, linear CKA on the same images at `gap` / `pre_logits` (1: the same up to a rotation and "
                    "a scale): " + "; ".join(l for l in lines if not l.endswith("None")) + ".", ""]
    return out


def table_stages(groups: dict, names: dict) -> list[str]:
    out = ["## The last three stages at the end of training", "",
           "Few-shot accuracy (%) read at `gap` → `pre_act` → `pre_logits`, mean of the setting's seeds.  "
           "\"Test's rule\" is the rule of the few-shot test; \"whitened\" whitens the features first (probe.py).", "",
           "| setting | runs | " + " | ".join(f"{title}, {rule_name}" for _, title in GROUPS for _, rule_name in RULES) + " |",
           "|---|---|" + "---|" * (len(GROUPS) * len(RULES))]
    for key, by_seed in groups.items():
        cells = [" → ".join(_pct(_mean(by_seed, lambda r: acc(r, last(r), rule, stage, group))) for stage in STAGES)
                 for group, _ in GROUPS for rule, _ in RULES]
        out.append(f"| {names[key]} | {_runs(by_seed)} | " + " | ".join(cells) + " |")
    return out + [""]


def table_steps(groups: dict, names: dict) -> list[str]:
    """What each of the two steps of the last layer does inside a run, on the unseen pools."""
    def step(rule, a, b):
        return lambda r: (None if acc(r, last(r), rule, b, "unseen") is None or acc(r, last(r), rule, a, "unseen") is None
                          else acc(r, last(r), rule, b, "unseen") - acc(r, last(r), rule, a, "unseen"))
    cols = [(rule, a, b) for rule, _ in RULES for a, b in (("gap", "pre_act"), ("pre_act", "pre_logits"), ("gap", "pre_logits"))]
    out = ["## What the last layer does to unseen classes, inside each run", "",
           "Change of the mean few-shot accuracy over the unseen pools (points) from one stage to the next, mean of "
           "the seeds ± the half-width of its 95 % interval (Student's t over the seeds); bold where the interval "
           "leaves out zero.  The matrix is `gap` → `pre_act`, the tanh is `pre_act` → `pre_logits`.", "",
           "| setting | runs | " + " | ".join(f"{dict(RULES)[rule]}: {'the matrix' if b == 'pre_act' else 'the tanh' if a == 'pre_act' else 'both'}"
                                             for rule, a, b in cols) + " |",
           "|---|---|" + "---|" * len(cols)]
    for key, by_seed in groups.items():
        out.append(f"| {names[key]} | {_runs(by_seed)} | "
                   + " | ".join(_diff(within(by_seed, step(*c))) for c in cols) + " |")
    return out + [""]


def table_against(groups: dict, names: dict, base_key: tuple) -> list[str]:
    out = [f"## Against {names[base_key]}, stage by stage", "",
           "Difference in few-shot accuracy (points) ± the half-width of its 95 % interval: seed by seed for a "
           "setting with several seeds, against the scatter of the baseline's seeds for a setting run once.  Bold "
           "where the interval leaves out zero.", "",
           "| setting | stage | " + " | ".join(f"{title}, {rule_name}" for _, title in GROUPS for _, rule_name in RULES) + " |",
           "|---|---|" + "---|" * (len(GROUPS) * len(RULES))]
    for key, by_seed in groups.items():
        if key == base_key:
            continue
        for stage in STAGES:
            cells = [_diff(compare(groups[base_key], by_seed, lambda r: acc(r, last(r), rule, stage, group)))
                     for group, _ in GROUPS for rule, _ in RULES]
            out.append(f"| {names[key]} | `{stage}` | " + " | ".join(cells) + " |")
    return out + [""]


def table_pools(groups: dict, names: dict, base_key: tuple) -> list[str]:
    base = groups[base_key]
    pools = unseen(next(iter(base.values())))
    cols = [(rule, stage) for rule, _ in RULES for stage in STAGES]
    out = [f"## Pool by pool against {names[base_key]}", "",
           f"Of the {len(pools)} unseen pools, on how many a setting is clearly ahead of the baseline / clearly "
           "behind it: the 95 % interval of the difference, taken as in the table above, leaves out zero.", "",
           "| setting | runs | " + " | ".join(f"{dict(RULES)[rule]}: `{stage}`" for rule, stage in cols) + " |",
           "|---|---|" + "---|" * len(cols)]
    for key, by_seed in groups.items():
        if key == base_key:
            continue
        cells = ["{} / {}".format(*ahead_behind(base, by_seed, lambda p: lambda r: acc(r, last(r), rule, stage, p), pools))
                 for rule, stage in cols]
        out.append(f"| {names[key]} | {_runs(by_seed)} | " + " | ".join(cells) + " |")
    return out + [""]


def table_ridge(groups: dict, names: dict, base_key: tuple) -> list[str]:
    """The whitened rule at every ridge the runs were probed with; nothing where no setting has more than one."""
    rows = {key: ridges(by_seed) for key, by_seed in groups.items()}
    if not any(len(r) > 1 for r in rows.values()):
        return []
    base = groups[base_key]
    pools = unseen(next(iter(base.values())))
    step = lambda ridge, a, b: lambda r: (None if at_ridge(ridge, a, "unseen")(r) is None
                                          else at_ridge(ridge, b, "unseen")(r) - at_ridge(ridge, a, "unseen")(r))
    out = ["## The whitened rule at other strengths of its ridge", "",
           "Before it divides every direction by its spread, the whitened rule adds a small constant to each "
           f"direction's variance: the ridge, a share of the mean variance, {RIDGE:g} in every other table here.  The "
           "last checkpoint tested again with other ridges, for the settings where that was done.  Mean of the "
           "unseen pools and of the seeds; differences in points ± the half-width of their 95 % interval, bold "
           f"where it leaves out zero; pools: of the {len(pools)} unseen ones, clearly ahead of the baseline / "
           "clearly behind it.", "",
           "| setting | runs | ridge | `gap` → `pre_act` → `pre_logits` (%) | against the baseline: `gap` | against "
           "the baseline: `pre_logits` | pools: `gap` | pools: `pre_logits` | inside a run: the matrix | inside a "
           "run: the tanh | inside a run: both |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, by_seed in groups.items():
        if len(rows[key]) < 2:
            continue
        for ridge in rows[key]:
            against = pooled = ["", ""]
            if key != base_key:
                against = [_diff(compare(base, by_seed, at_ridge(ridge, s, "unseen"))) for s in ("gap", "pre_logits")]
                pooled = ["{} / {}".format(*ahead_behind(base, by_seed, lambda p: at_ridge(ridge, s, p), pools))
                          for s in ("gap", "pre_logits")]
            out.append(f"| {names[key]} | {_runs(by_seed)} | {ridge:g} | "
                       + " → ".join(_pct(_mean(by_seed, at_ridge(ridge, s, "unseen"))) for s in STAGES) + " | "
                       + " | ".join(against + pooled) + " | "
                       + " | ".join(_diff(within(by_seed, step(ridge, a, b)))
                                    for a, b in (("gap", "pre_act"), ("pre_act", "pre_logits"), ("gap", "pre_logits"))) + " |")
    return out + [""]


def table_layer(groups: dict, names: dict) -> list[str]:
    def sv_ratio(r):
        sv = last(r).get("sv", {}).get("pre_logits.weight")
        return float(sv[0] / np.median(sv)) if sv else None
    out = ["## The `pre_logits` layer at the end of training", "",
           "tanh: the share of unit values beyond 0.9 and beyond 0.99 in size, on validation images of the trained "
           "classes and on unseen images, and the root mean square of what goes into the tanh.  Matrix: Frobenius "
           "norm, largest singular value, stable rank (sum of squared singular values over the largest), entropy "
           "rank, and the largest singular value over the median one.", "",
           "| setting | runs | beyond 0.9, trained (%) | beyond 0.9, unseen (%) | beyond 0.99, unseen (%) | rms into "
           "the tanh | Frobenius | largest | stable rank | entropy rank | largest / median |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, by_seed in groups.items():
        t = lambda which, k: _mean(by_seed, lambda r: last(r)["tanh"].get(which, {}).get(k))
        w = lambda k: _mean(by_seed, lambda r: last(r)["weights"].get("pre_logits.weight", {}).get(k))
        out.append(f"| {names[key]} | {_runs(by_seed)} | {_pct(t(SEEN_SET, 'over90'))} | {_pct(t(UNSEEN_SET, 'over90'))} | "
                   f"{_pct(t(UNSEEN_SET, 'over99'))} | {_num(t(UNSEEN_SET, 'rms_in'), 2)} | {_num(w('fro'))} | "
                   f"{_num(w('spec'), 2)} | {_num(w('srank'))} | {_num(w('erank'))} | {_num(_mean(by_seed, sv_ratio))} |")
    return out + [""]


def table_last(drawn: dict, names: dict) -> list[str]:
    """The numbers of last.png: the layer the head reads, checkpoint by checkpoint."""
    def cost(rule):
        def fn(r, c):
            a, b = acc(r, c, rule, "pre_logits", "unseen"), acc(r, c, rule, "gap", "unseen")
            return None if a is None or b is None else 100 * (a - b)
        return fn
    fns = [lambda r, c: 100 * c["tanh"][UNSEEN_SET]["over90"] if UNSEEN_SET in c["tanh"] else None,
           lambda r, c: c["weights"].get("pre_logits.weight", {}).get("srank"), cost("plain"), cost("white")]
    curves = {key: [along(by_seed, fn) for fn in fns] for key, by_seed in drawn.items()}
    curves = {key: c for key, c in curves.items() if all(x is not None for x in c)}
    if not curves:
        return []
    out = ["## The `pre_logits` layer along training", "",
           "At every checkpoint, mean of the seeds: tanh units beyond 0.9 in size on unseen images (%) / stable rank "
           "of the matrix / what the layer does to the mean few-shot accuracy over the unseen pools, `pre_logits` "
           "minus `gap`, under the test's rule / whitened (points).", "",
           "| % of training | " + " | ".join(names[key] for key in curves) + " |", "|---|" + "---|" * len(curves)]
    at = {key: {round(float(p), 1): [float(x[1][i]) for x in c] for i, p in enumerate(c[0][0])} for key, c in curves.items()}
    for pct in sorted(set().union(*at.values())):
        out.append(f"| {pct:g} | " + " | ".join(
            "{:.1f} / {:.1f} / {:+.1f} / {:+.1f}".format(*[round(v, 1) + 0.0 for v in at[key][pct]]) if pct in at[key] else "n/a"
            for key in curves) + " |")
    return out + [""]


def table_shape(groups: dict, names: dict) -> list[str]:
    def shape(which, key, stage):
        return lambda r: last(r)["shape"][stage].get(which, {}).get(key)
    out = ["## The shape of the features at the last three stages", "",
           "`gap` → `pre_act` → `pre_logits`, mean of the seeds.  Entropy rank: how many directions the features "
           "fill (of 384).  Between classes: the share of the variance that lies between class means (%), on the "
           "features as the test's rule sees them.", "",
           "| setting | entropy rank, unseen images | entropy rank, trained classes | between classes, trained | "
           "between classes, dog breeds | between classes, all unseen classes |",
           "|---|---|---|---|---|---|"]
    for key, by_seed in groups.items():
        cells = [" → ".join((_num if k == "erank" else _pct)(_mean(by_seed, shape(which, k, s))) for s in STAGES)
                 for which, k in ((UNSEEN_SET, "erank"), (SEEN_SET, "erank"), (SEEN_SET, "between"),
                                  (DOG_SET, "between"), (UNSEEN_SET, "between"))]
        out.append(f"| {names[key]} | " + " | ".join(cells) + " |")
    return out + [""]


def table_depth(drawn: dict, names: dict, base_key: tuple) -> list[str]:
    one = next(iter(next(iter(drawn.values())).values()))
    others = [key for key in drawn if key != base_key] if base_key in drawn else []
    out = ["## Few-shot accuracy by depth, at the end of training", "",
           "Mean over the unseen pools (%), mean of the seeds, at every layer: the test's rule / whitened.  "
           "\"Against\": the difference from the baseline in points ± the half-width of its 95 % interval, the test's "
           "rule / whitened, bold where the interval leaves out zero.", "",
           "| layer | " + " | ".join([names[key] for key in drawn] + [f"{names[key]} against {names[base_key]}" for key in others]) + " |",
           "|---|" + "---|" * (len(drawn) + len(others))]
    for layer in one["probe"]["layers"]:
        cells = [" / ".join(_pct(_mean(by_seed, lambda r: acc(r, last(r), rule, layer, "unseen"))) for rule, _ in RULES)
                 for by_seed in drawn.values()]
        cells += [" / ".join(_diff(compare(drawn[base_key], drawn[key], lambda r: acc(r, last(r), rule, layer, "unseen")))
                             for rule, _ in RULES) for key in others]
        out.append(f"| `{layer}` | " + " | ".join(cells) + " |")
    return out + [""]


def table_classes(groups: dict, names: dict, base_key: tuple) -> list[str]:
    one = next(iter(next(iter(groups.values())).values()))
    parts = by_frequency(one)
    head = [f"{p['fewest']:,}–{p['most']:,} images" for p in parts]
    sizes = ", ".join(f"{p['images']:,}" for p in parts)
    out = ["## Validation top-1 by how many training images a class has", "",
           "The trained classes in five groups of equally many classes, from the fewest training images to the most "
           f"(validation images per group: {sizes}).  \"Every class the same\" "
           "averages the classes' accuracies instead of the images.  Second row of a setting: difference from the "
           "baseline in points ± the half-width of its 95 % interval.", "",
           "| setting | runs | all images | every class the same | " + " | ".join(head) + " | training images, "
           "every class the same |", "|---|---|---|---|" + "---|" * (len(parts) + 1)]
    for key, by_seed in groups.items():
        fns = ([lambda r: last(r)["val"]["top1"], macro]
               + [lambda r, i=i: by_frequency(r)[i]["top1"] for i in range(len(parts))]
               + [lambda r: macro(r, "train") if "train" in last(r) else None])
        out.append(f"| {names[key]} | {_runs(by_seed)} | " + " | ".join(_pct(_mean(by_seed, f)) for f in fns) + " |")
        if key != base_key:
            out.append("| | | " + " | ".join(_diff(compare(groups[base_key], by_seed, f)) for f in fns) + " |")
    return out + [""]


def table_rare(drawn: dict, names: dict) -> list[str]:
    curves = {key: rare_common(by_seed) for key, by_seed in drawn.items()}
    curves = {key: c for key, c in curves.items() if c is not None}
    if not curves:
        return []
    one = next(iter(next(iter(drawn.values())).values()))
    parts = by_frequency(one)
    out = ["## Rare and common classes along training", "",
           "Validation top-1 (%) at every checkpoint, mean of the seeds: all images / the fifth of the classes with "
           f"the fewest training images ({parts[0]['fewest']:,}–{parts[0]['most']:,}) / the fifth with the most "
           f"({parts[-1]['fewest']:,}–{parts[-1]['most']:,}).", "",
           "| % of training | " + " | ".join(names[key] for key in curves) + " |", "|---|" + "---|" * len(curves)]
    at = {key: {round(float(p), 1): (a, r, c) for p, a, r, c in zip(*curve)} for key, curve in curves.items()}
    for pct in sorted(set().union(*at.values())):
        out.append(f"| {pct:g} | " + " | ".join(
            " / ".join(f"{v:.1f}" for v in at[key][pct]) if pct in at[key] else "n/a" for key in curves) + " |")
    top = min(curve[1][-1] for curve in curves.values())
    levels = [v for v in (40, 50, 60, 70, 80) if v < int(top)] + [int(top)]
    out += ["", "The same at a given accuracy on all images instead of a given time: the rarest fifth / the commonest "
            "fifth (%) where a setting's curve of all images passes the level, read off the straight line between "
            "its two checkpoints there (mean of the seeds; the checkpoints before the last are taken with the "
            "learning rate still above zero).", "",
            "| all images | " + " | ".join(names[key] for key in curves) + " |", "|---|" + "---|" * len(curves)]
    for level in levels:
        cells = []
        for _, everything, rare, common in curves.values():
            r, c = at_level(everything, rare, level), at_level(everything, common, level)
            cells.append("n/a" if r is None else f"{r:.1f} / {c:.1f}")
        out.append(f"| {level} % | " + " | ".join(cells) + " |")
    return out + [""]


def table_weights(groups: dict, names: dict) -> list[str]:
    out = ["## The weight matrices at the end of training", "",
           "Per kind of matrix (for the blocks, the mean over the 12 blocks): Frobenius norm of the matrix / its "
           "stable rank / the stable rank of its change since the start / the Frobenius norm of that change.", "",
           "| setting | " + " | ".join(f[1].replace("\n", " ") for f in FAMILIES) + " |", "|---|" + "---|" * len(FAMILIES)]
    for key, by_seed in groups.items():
        cells = []
        for fam, _ in FAMILIES:
            f = lambda k, sub=None: _mean(by_seed, lambda r: family(last(r), fam, k, sub))
            cells.append(f"{_num(f('fro'))} / {_num(f('srank'))} / {_num(f('srank', 'since_first'))} / "
                         f"{_num(f('fro', 'since_first'))}")
        out.append(f"| {names[key]} | " + " | ".join(cells) + " |")
    return out + [""]


def table_blocks(groups: dict, names: dict, base_key: tuple) -> list[str]:
    """The numbers of blocks.png, for every setting."""
    blocks = block_numbers(next(iter(groups[base_key].values())))
    head = "| setting | " + " | ".join(str(b) for b in blocks) + " |"
    rule = "|---|" + "---|" * len(blocks)
    out = ["## Block by block at the end of training", ""]
    for title, fn in (("Stable rank of the matrices: attention (mean of q, k, v, proj) / MLP (mean of fc1, fc2)",
                       lambda by_seed, b: " / ".join(_num(_mean(by_seed, lambda r, f=f: in_block(last(r), b, f, "srank")), 0)
                                                     for _, f in PARTS)),
                      ("Stable rank of their change since the start: attention / MLP",
                       lambda by_seed, b: " / ".join(_num(_mean(by_seed, lambda r, f=f: in_block(last(r), b, f, "srank",
                                                                                                  "since_first")), 0)
                                                     for _, f in PARTS)),
                      ("How far the block moved: the norm of the change of its six matrices since the start over "
                       "their norm at the start", lambda by_seed, b: _num(_mean(by_seed, lambda r: moved(r, b)), 2))):
        out += [f"{title}.", "", head, rule]
        out += [f"| {names[key]} | " + " | ".join(fn(by_seed, b) for b in blocks) + " |" for key, by_seed in groups.items()]
        out.append("")
    base = groups[base_key]
    for rule_key, rule_name in RULES:
        out += [f"Few-shot accuracy on the unseen pools (mean) on the block's output, {rule_name}: the difference from "
                f"{names[base_key]} in points, with its 95 % interval (bold where it excludes 0).", "", head, rule]
        for key, by_seed in groups.items():
            if key == base_key:
                continue
            cells = []
            for b in blocks:
                fn = lambda r, b=b: acc(r, last(r), rule_key, f"block{b}", "unseen")
                c = compare(base, by_seed, fn)
                cells.append("" if c is None else _diff(c))
            out.append(f"| {names[key]} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def table_cka(cka: dict, groups: dict, names: dict) -> list[str]:
    same, across = cka_lines(cka, groups)
    if not same and not across:
        return []
    cols = [(f"{names[k]}: two seeds", v) for k, v in same.items()]
    cols += [(f"{names[a]} vs {names[b]}, same seed", v) for (a, b), v in across.items() if a in same and b in same]
    out = ["## How alike the runs are, layer by layer", "",
           f"Linear CKA between the features of the same {cka['images']:,} images at the end of training: 1 when two "
           "runs hold the same features up to a rotation and a scale.  Mean over the pairs of seeds, or over the "
           "seeds two settings share.", "",
           "| layer | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for i, layer in enumerate(cka["layers"]):
        out.append(f"| `{layer}` | " + " | ".join(f"{c[1][i]:.2f}" for c in cols) + " |")
    return out + [""]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cka", help="the output of cka.py over the same runs")
    ap.add_argument("--baseline", default="adamw:0.001", help="optimizer:learning rate of the recipe's own setting")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    loaded = [load(p) for p in args.runs]
    runs = [r for r in loaded if probed(r)]
    waiting = [os.path.basename(os.path.normpath(r["path"])) for r in loaded if not probed(r)]
    if not runs:
        raise SystemExit("why: none of the runs has a probe.json that reaches its last step")
    groups = settings(runs)
    lengths = [key[2] for key in groups for _ in groups[key]]
    standard = max(set(lengths), key=lengths.count)
    names = {key: label(key, by_seed, standard) for key, by_seed in groups.items()}
    b_opt, _, b_lr = args.baseline.partition(":")
    base_key = next((key for key in groups if len(key) == 3
                     and key[0] == b_opt and abs(key[1] - float(b_lr)) < 1e-12 and key[2] == standard), None)
    if base_key is None:
        raise SystemExit(f"why: no probed run of the baseline setting {args.baseline} at {standard:,} steps")
    drawn = {key: by_seed for key, by_seed in groups.items() if len(by_seed) > 1} or groups
    cka = None
    if args.cka:
        with open(args.cka) as f:
            cka = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    draw_depth(plt, drawn, names, os.path.join(args.out, "depth.png"))
    draw_last(plt, drawn, names, os.path.join(args.out, "last.png"))
    draw_shape(plt, drawn, names, os.path.join(args.out, "shape.png"))
    draw_classes(plt, drawn, names, base_key, os.path.join(args.out, "classes.png"))
    draw_weights(plt, drawn, names, os.path.join(args.out, "weights.png"))
    single = block_singles(groups, drawn, base_key)
    draw_blocks(plt, drawn, names, base_key, groups[base_key], single, os.path.join(args.out, "blocks.png"))
    made = ["depth.png", "last.png", "shape.png", "classes.png", "weights.png", "blocks.png"]
    if draw_rare(plt, drawn, names, os.path.join(args.out, "rare.png")):
        made.append("rare.png")
    if cka and draw_cka(plt, cka, drawn, names, os.path.join(args.out, "cka.png")):
        made.append("cka.png")
    if draw_rates(plt, groups, standard, os.path.join(args.out, "rates.png")):
        made.append("rates.png")
    for key, ref in controls(groups):
        name = f"control-{_slug(key[3])}.png"
        if draw_control(plt, groups, names, key, ref, base_key, os.path.join(args.out, name)):
            made.append(name)
    text = ["# Why: what the probes of the checkpoints show", ""]
    if waiting:
        text += ["Not read, because not probed at the last step: " + ", ".join(waiting) + ".", ""]
    text += table_control(groups, names, base_key, cka)
    text += table_stages(groups, names) + table_steps(groups, names) + table_against(groups, names, base_key)
    text += table_pools(groups, names, base_key) + table_ridge(groups, names, base_key)
    text += table_layer(groups, names) + table_last(drawn, names) + table_shape(groups, names)
    text += table_depth(drawn, names, base_key)
    text += table_classes(groups, names, base_key) + table_rare(drawn, names) + table_weights(groups, names)
    text += table_blocks(groups, names, base_key)
    if cka:
        text += table_cka(cka, groups, names)
    with open(os.path.join(args.out, "why.md"), "w") as f:
        f.write("\n".join(text))
    print(f"why: {', '.join(made + ['why.md'])} -> {args.out}: {len(groups)} settings, {len(runs)} probed runs"
          + (f", {len(waiting)} not probed" if waiting else ""))


if __name__ == "__main__":
    main()
