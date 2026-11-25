"""Figures and the numbers table of the ViT study (Section 6 and Appendix G of the report).

usage: python make_figs.py [OUT_DIR]          (default OUT_DIR: figs/, next to this file)

Reads data/vit_page_data.json (next to this file) and writes every figure as .png and .pdf, plus numbers.md
(the table of end-of-training numbers quoted in the report). Nothing is typed in by hand: every plotted or printed value
comes from the data file.

The data file holds the per-seed end-of-training values, the means and seed ranges at the 17 checkpoints, and
the per-test, per-layer and control-run differences with their 95% intervals. extract_data.py says where it
comes from. The raw per-run read-outs and the model checkpoints are not in this repository.

Conventions, the same as in the study:
  - "never-seen classes" = mean five-shot accuracy (%) over 14 tests on classes the model was never trained to
    tell apart, nearest class mean on frozen features, 1,000 rounds per test;
  - read points: `gap` = encoder output (mean over patches after block 12), `pre_logits` = after the last hidden
    layer (a linear map and a tanh), which is what the classifier reads;
  - plain = features as they are, equalized = features whitened first;
  - differences are taken seed by seed (same initial weights and data order under each optimizer); intervals are
    95% Student-t intervals over seeds; bands in the training curves are the range over seeds.
Colors and plot style follow the figures of the report's first part (in-context regression and ciphers,
github.com/EshMis/optimizer-icl), so the two parts of the report look alike.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "vit_page_data.json")

COL = {"adamw": "#2a78d6", "muon": "#eb6834", "sgd": "#52514e"}      # as in the report's regression figures
INK, MUTED = "#0b0b0b", "#52514e"
NAME = {"adamw": "AdamW (1e-3)", "muon": "Muon (3e-3)", "sgd": "SGD (1.0)"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
                     "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white"})

# the 14 tests, grouped as the report lists them
TESTS = [("dogs_5way", "Dog breeds, 5-way"), ("dogs_20way", "Dog breeds, 20-way"),
         ("dogs_118way", "Dog breeds, 118-way"), ("wildcanids_5way", "Wolves and foxes, 5-way"),
         ("wildcanids_12way", "Wolves and foxes, 12-way"), ("cats", "House-cat breeds, 5-way"),
         ("felines_5way", "Cat family, 5-way"), ("felines_13way", "Cat family, 13-way"),
         ("birds_5way", "Bird species, 5-way"), ("birds_20way", "Bird species, 20-way"),
         ("birds_59way", "Bird species, 59-way"), ("inet_5way", "ImageNet classes, 5-way"),
         ("inet_20way", "ImageNet classes, 20-way"), ("control", "Held-out Objects365, 20-way")]
TICKS = [0, 0.5, 1, 2, 4, 7, 10, 20, 40, 100]          # % of training shown on the stretched time axis


def tx(pct):
    """Position of a checkpoint on the time axis: log2(% of training + 0.5), which spreads out the early
    checkpoints (0, 0.5, 1, 2, 4 %) where the optimizers differ most."""
    return np.log2(np.asarray(pct, dtype=float) + 0.5)


def time_axis(ax):
    ax.set_xticks(tx(TICKS))
    ax.set_xticklabels([f"{t:g}" for t in TICKS], fontsize=7.5)
    ax.set_xlabel("training done (%), early checkpoints stretched")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def fig_read_points(D, out):
    """Never-seen classes at the end of training, seed by seed, at the two read points and in the model built
    without the last hidden layer. Report: Figure 5."""
    sc = D["score"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5), sharey=True)
    panels = (("p", "Plain reading: features compared as they are"),
              ("w", "Equalized reading: features whitened first"))
    for ax, (r, title) in zip(axes, panels):
        gap, pl, npl = sc[f"un_gap_{r}"], sc[f"un_pl_{r}"], sc[f"np_{r}"]
        for o, k in (("sgd", "s"), ("adamw", "a"), ("muon", "m")):
            g, p = np.array(gap[k]), np.array(pl[k])
            for a, b in zip(g, p):
                ax.plot([0, 1], [a, b], color=COL[o], lw=0.7, alpha=0.3)
            ax.plot([0, 1], [g.mean(), p.mean()], color=COL[o], lw=2.2, marker="o", ms=6, label=NAME[o], zorder=3)
        for o, k, dx in (("adamw", "a", -0.09), ("muon", "m", 0.09)):
            v = np.array(npl[k])
            ax.scatter([2 + dx] * len(v), v, s=12, color=COL[o], alpha=0.35, zorder=2)
            ax.scatter([2 + dx], [v.mean()], s=64, marker="D", color=COL[o], edgecolor="white", linewidth=1.2,
                       zorder=3)
        ax.axvline(1.5, color=MUTED, lw=0.8, ls=":")
        # Muon minus AdamW at each place, 95% interval over the five paired seeds; grey where it includes zero
        for x, e in enumerate((gap, pl, npl)):
            clear = abs(e["d"]) > e["hw"]
            ax.text(x, 1.015, f"{e['d']:+.1f} ± {e['hw']:.1f}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color=INK if clear else MUTED, fontweight="bold" if clear else "normal")
        ax.text(0.0, 1.085, "Muon − AdamW in points (95% interval, 5 paired seeds; grey: includes zero)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=MUTED)
        ax.set_xlim(-0.45, 2.45)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Encoder output\n(after block 12)", "Last hidden layer\n(the classifier's input)",
                            "Model built\nwithout that layer"], fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, loc="left", pad=34)
    axes[0].set_ylabel("accuracy on never-seen classes (%)\nmean of 14 five-shot tests")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("End of training. Lines join one run's two read points; thin lines and small dots are the five "
                 "seeds, thick lines and diamonds their means.", fontsize=8.5, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "vit_read_points")


def fig_training(D, out):
    """The same test at the 17 checkpoints, with the state of the last hidden layer. Report: Figure 6."""
    al, x = D["along"], tx(D["pct"])
    keys = (("adamw", "adamw"), ("muon", "muon3"), ("sgd", "sgd"))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    ax = axes[0]
    for o, k in keys:
        s = al[k]["un_pl_w"]
        ax.fill_between(x, s["lo"], s["hi"], color=COL[o], alpha=0.18, lw=0)
        ax.plot(x, s["m"], color=COL[o], lw=2, marker="o", ms=3, label=NAME[o])
    u = al["adamw"]["un_pl_w"]["m"][0]
    ax.axhline(u, color=MUTED, lw=0.8, ls=":")
    ax.text(x[-1], u - 1.0, f"untrained model: {u:.1f}%", ha="right", va="top", fontsize=8, color=MUTED)
    ax.set_ylabel("accuracy on never-seen classes (%)\nlast hidden layer, equalized reading")
    ax.set_title("Few-shot accuracy: AdamW first falls\nbelow the untrained model", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    for ax, key, ylab, title in (
            (axes[1], "sat", "units with |tanh output| > 0.9 (%)",
             "Under AdamW the layer saturates\nin the first 0.5% of training"),
            (axes[2], "plrank", "stable rank of the layer's matrix",
             "AdamW's matrix narrows to a stable\nrank of 10 and recovers only partly")):
        for o, k in keys:
            ax.plot(x, al[k][key]["m"], color=COL[o], lw=2, marker="o", ms=3, label=NAME[o])
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=10, loc="left")
    for ax in axes:
        time_axis(ax)
    fig.suptitle("During training, read at the last hidden layer (means of 5 seeds; band: range over seeds)",
                 fontsize=8.5, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "vit_training")


def fig_tests(D, out):
    """Muon minus AdamW test by test, at both read points and in both readings. Report: Figure G1."""
    P = D["pools"]
    at = {n: i for i, n in enumerate(P["names"])}
    y = np.arange(len(TESTS))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), sharey=True)
    for ax, (layer, title) in zip(axes, (("pre_logits", "At the last hidden layer"), ("gap", "At the encoder output"))):
        for rule, dy, filled, lab in (("white", 0.17, True, "equalized reading"), ("plain", -0.17, False, "plain reading")):
            rows = [P[f"muon3_{rule}_{layer}"][at[n]] for n, _ in TESTS]
            ax.errorbar([r[0] for r in rows], y + dy, xerr=[r[1] for r in rows], fmt="o", ms=5, color=COL["muon"],
                        mfc=COL["muon"] if filled else "white", mew=1.3, elinewidth=1.2, label=lab)
        ax.axvline(0, color=MUTED, lw=0.9)
        ax.set_xlabel("Muon − AdamW (points of accuracy)")
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab in TESTS], fontsize=8, color=INK)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Each of the 14 tests on never-seen classes at the end of training (5 paired seeds; bars: 95% interval)",
                 fontsize=8.5, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "vit_tests")


def fig_depth(D, out):
    """Difference from AdamW after each layer of the model. Report: Figure G2."""
    dp = D["depth"]
    x = np.arange(len(dp["layers"]))
    names = ["patch\nemb."] + [str(i) for i in range(1, 13)] + ["enc.\nout", "pre-\ntanh", "last\nhid."]
    assert len(names) == len(x) and dp["layers"][-3:] == ["gap", "pre_act", "pre_logits"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)
    for ax, (rule, title) in zip(axes, (("plain", "Plain reading"), ("white", "Equalized reading"))):
        ax.axvspan(12.5, 15.5, color=MUTED, alpha=0.08, lw=0)
        ax.axhline(0, color=MUTED, lw=0.9)
        for o, k, dx in (("sgd", "sgd", 0.1), ("muon", "muon3", -0.1)):
            rows = dp[f"d_{k}_{rule}"]
            ax.errorbar(x + dx, [r[0] for r in rows], yerr=[r[1] for r in rows], color=COL[o], lw=1.6, marker="o",
                        ms=4, elinewidth=1, label=f"{NAME[o]} − AdamW")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5)
        ax.set_xlabel("read after block 1 to 12, then in the head (shaded): encoder output,\n"
                      "before the tanh, after the last hidden layer")
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].set_ylabel("difference from AdamW on\nnever-seen classes (points)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("End of training, read after every layer (5 paired seeds; bars: 95% interval)",
                 fontsize=8.5, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "vit_depth")


def fig_controls(D, out):
    """The 2x2 at one learning rate: what Muon on the blocks adds, and what Muon on the last hidden layer's
    matrix adds. Report: Figure G3."""
    s3 = D["robust"]["s3"]
    rows = (("val", "Trained classes\n(validation top-1)"),
            ("ICL pre_logits plain", "Never-seen classes,\nlast hidden layer, plain"),
            ("ICL pre_logits white", "Never-seen classes,\nlast hidden layer, equalized"))
    kinds = (("blocks_effect_average", 0.18, "o", True, "Muon instead of AdamW on the 12 blocks"),
             ("matrix_effect_average", -0.18, "s", False, "Muon instead of AdamW on the last hidden layer's matrix"))
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.axvline(0, color=MUTED, lw=0.9)
    for i, (k, _) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for name, dy, mk, filled, lab in kinds:
            e = s3[k][name]
            ax.errorbar([e["m"]], [y0 + dy], xerr=[e["hw"]], fmt=mk, ms=7, color=COL["muon"],
                        mfc=COL["muon"] if filled else "white", mew=1.6, elinewidth=1.4, zorder=3,
                        label=lab if i == 0 else None)
            ax.scatter(list(e["s"].values()), [y0 + dy] * len(e["s"]), s=9, color=INK, alpha=0.55, zorder=4)
            clear = abs(e["m"]) > e["hw"]
            ax.text(e["m"] + e["hw"] + 0.07, y0 + dy, f"{e['m']:+.2f} ± {e['hw']:.2f}", va="center", fontsize=8,
                    color=INK if clear else MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows][::-1], fontsize=8, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.75, 2.75)
    ax.set_xlabel("gain in points of accuracy (95% interval over 3 paired seeds; grey label: includes zero)")
    ax.set_title("On trained classes the blocks carry Muon's gain; at the last\nhidden layer the layer's own matrix has the clear effect",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=1)
    fig.text(0.01, 0.985, "All four combinations at learning rate 1e-3 (black dots: seeds)", fontsize=8.5,
             color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "vit_controls")


def fig_updates(D, out):
    """How concentrated the weight updates are: the share of a matrix's change between two checkpoints that
    lies in its strongest singular direction. Report: Figure G4."""
    b, x = D["blocks"], tx(D["pct"][1:])
    series = (("adamw", "adamw", "-", "AdamW 1e-3"), ("sgd", "sgd", "-", "SGD 1.0"),
              ("muon", "muon1", "--", "Muon 1e-3 (AdamW's rate)"), ("muon", "muon3", "-", "Muon 3e-3"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, (fam, title) in zip(axes, (("attn", "Attention matrices (4 per block)"), ("mlp", "MLP matrices (2 per block)"))):
        for o, k, ls, lab in series:
            ax.plot(x, b[k][f"u_top_{fam}"], color=COL[o], ls=ls, lw=2, marker="o", ms=3, label=lab)
        ax.axhline(100 / 384, color=MUTED, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_yticks([0.3, 1, 3, 10, 30, 100])
        ax.set_yticklabels(["0.3", "1", "3", "10", "30", "100"])
        ax.set_title(title, fontsize=10, loc="left")
        time_axis(ax)
        ax.set_xticks(tx(TICKS[1:]))
        ax.set_xticklabels([f"{t:g}" for t in TICKS[1:]], fontsize=7.5)
        ax.set_xlabel("stretch of training ending at (%), early checkpoints stretched")
    axes[0].text(x[-1], 100 / 384 * 1.12, "an even spread over 384 directions", ha="right", va="bottom", fontsize=7.5,
                 color=MUTED)
    axes[0].set_ylabel("share of an update in its\nstrongest direction (%)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("What the weights changed between neighbouring checkpoints, mean over the 12 blocks and over seeds "
                 "(5 seeds; Muon 1e-3: 3)", fontsize=8.5, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "vit_updates")


def fig_fifths(D, out):
    """Trained classes by how many training images they have. Report: Figure G5."""
    q, qd = D["quint"], D["quint_diff"]
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.axhline(0, color=MUTED, lw=0.9)
    for o, k, dx in (("muon", "muon3", -0.05), ("sgd", "sgd", 0.05)):
        ax.errorbar(x + dx, [r[0] for r in qd[k]], yerr=[r[1] for r in qd[k]], color=COL[o], lw=2, marker="o",
                    ms=5, capsize=2, label=f"{NAME[o]} − AdamW")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:,} to {c:,}\nAdamW: {v:.1f}%" for (a, c), v in zip(q["range"], q["adamw"])], fontsize=7.5)
    ax.set_xlabel("trained classes in fifths by training images per class (rarest to commonest)")
    ax.set_ylabel("difference from AdamW in\nvalidation accuracy (points)")
    ax.set_title("Muon's gain on trained classes is largest on rare classes", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    fig.text(0.01, 0.985, "End of training (5 paired seeds; bars: 95% interval)", fontsize=8.5, color=MUTED,
             ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "vit_rare_classes")


def numbers(D):
    """The end-of-training table quoted in the report, as markdown."""
    sc, ac = D["score"], D["across"]
    mean = lambda v: float(np.mean(v))                                               # noqa: E731
    pm = lambda d, hw: f"{d:+.2f} ± {hw:.2f}"                                         # noqa: E731
    rows = [("Trained classes, validation top-1", "val"), ("Rarest fifth of trained classes", "rare"),
            ("Commonest fifth of trained classes", "common"),
            ("Never-seen, encoder output, plain", "un_gap_p"), ("Never-seen, encoder output, equalized", "un_gap_w"),
            ("Never-seen, last hidden layer, plain", "un_pl_p"), ("Never-seen, last hidden layer, equalized", "un_pl_w")]
    out = ["# ViT study: end-of-training numbers", "",
           "Written by `make_figs.py` from `data/vit_page_data.json`. Accuracy in %, means over five seeds;",
           "differences are seed by seed with 95% Student-t intervals.", "",
           "| Measure | AdamW 1e-3 | Muon 3e-3 | SGD 1.0 | Muon − AdamW | SGD − AdamW |", "|---|---|---|---|---|---|"]
    for lab, k in rows:
        s = sc[k]
        out.append(f"| {lab} | {mean(s['a']):.2f} | {mean(s['m']):.2f} | {mean(s['s']):.2f} | {pm(s['d'], s['hw'])} | "
                   f"{pm(s['ds'], s['hws'])} |")
    out += ["", "Cost of the last hidden layer (last hidden layer minus encoder output, within each run):", "",
            "| Reading | AdamW | Muon | SGD |", "|---|---|---|---|"]
    for r, lab in (("p", "plain"), ("w", "equalized")):
        out.append(f"| {lab} | " + " | ".join(pm(*ac[k][r]) for k in ("adamw", "muon3", "sgd")) + " |")
    out += ["", "Model built without the last hidden layer (five seeds each; read at the encoder output):", "",
            "| Measure | AdamW 1e-3 | Muon 3e-3 | Muon − AdamW |", "|---|---|---|---|"]
    for lab, k in (("Trained classes, validation top-1", "np_val"), ("Rarest fifth of trained classes", "np_rare"),
                   ("Never-seen, plain", "np_p"), ("Never-seen, equalized", "np_w")):
        s = sc[k]
        out.append(f"| {lab} | {mean(s['a']):.2f} | {mean(s['m']):.2f} | {pm(s['d'], s['hw'])} |")
    s3 = D["robust"]["s3"]
    out += ["", "The 2x2 at learning rate 1e-3 (three seeds per cell): gain from Muon instead of AdamW on", "",
            "| Measure | the 12 blocks | the last hidden layer's matrix |", "|---|---|---|"]
    for lab, k in (("Trained classes, validation top-1", "val"), ("Never-seen, last hidden layer, plain", "ICL pre_logits plain"),
                   ("Never-seen, last hidden layer, equalized", "ICL pre_logits white")):
        b, m = s3[k]["blocks_effect_average"], s3[k]["matrix_effect_average"]
        out.append(f"| {lab} | {pm(b['m'], b['hw'])} | {pm(m['m'], m['hw'])} |")
    al = D["along"]
    lead = np.array(al["muon3"]["un_pl_w"]["m"]) - np.array(al["adamw"]["un_pl_w"]["m"])
    i = int(np.argmax(lead))
    out += ["", "During training (last hidden layer, equalized, means of five seeds):", "",
            f"- untrained model: {al['adamw']['un_pl_w']['m'][0]:.1f}%; AdamW after 0.5% of training: "
            f"{al['adamw']['un_pl_w']['m'][1]:.1f}%, after 1%: {al['adamw']['un_pl_w']['m'][2]:.1f}%",
            f"- Muon's lead over AdamW is largest after {D['pct'][i]:g}% of training ({lead[i]:.1f} points) and "
            f"{lead[-1]:.1f} at the end",
            f"- saturated units after 0.5%: AdamW {al['adamw']['sat']['m'][1]:.1f}%, Muon {al['muon3']['sat']['m'][1]:.1f}%, "
            f"SGD {al['sgd']['sat']['m'][1]:.1f}%",
            f"- stable rank of the layer's matrix: start {al['adamw']['plrank']['m'][0]:.0f}; AdamW lowest "
            f"{min(al['adamw']['plrank']['m']):.1f}, end {al['adamw']['plrank']['m'][-1]:.0f}; Muon end "
            f"{al['muon3']['plrank']['m'][-1]:.0f}; SGD end {al['sgd']['plrank']['m'][-1]:.0f}", ""]
    return "\n".join(out)


def main(out):
    with open(DATA) as f:
        D = json.load(f)
    os.makedirs(out, exist_ok=True)
    for fn in (fig_read_points, fig_training, fig_tests, fig_depth, fig_controls, fig_updates, fig_fifths):
        fn(D, out)
    with open(os.path.join(HERE, "numbers.md"), "w") as f:
        f.write(numbers(D))
    print("wrote numbers.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figs"))
