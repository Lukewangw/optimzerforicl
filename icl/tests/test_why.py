"""The read-out of the probes: the arithmetic, and that it runs on made-up probe files."""
import json
import os

import numpy as np
import pytest

from icl import curves, why
from icl.seeds import label, settings

LAYERS = ["embed", "block1", "gap", "pre_act", "pre_logits"]
POOLS = {"dog": {"n_way": 1}, "dogs_5way": {"n_way": 5}, "control": {"n_way": 20}, "seen_5way": {"n_way": 5},
         "seen_118way": {"n_way": 118}}
COUNTS = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]              # training images of the ten trained classes
VAL = [2, 4, 10, 10, 10, 10, 10, 10, 20, 20]                         # their validation images


def _run(root, opt, lr, seed, gap, matrix=0.0, tanh=0.0, rare=0.5, total=400, tag="", variant=None):
    """A probed run whose unseen pools give `gap` at the encoder output under both rules; the matrix then costs
    `matrix` under the plain rule and the tanh costs `tanh` under the whitened one.  The two classes with the
    fewest training images are right `rare` of the time, the others 80 %.  `variant`: flags of train.py that make
    the run a control (curves.VARIANTS); `tag`: a word for its folder."""
    d = root / f"nodog-{opt}-lr{lr:g}{tag}-s{seed}"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"total_steps": total, "args": {
        "optimizer": opt, "lr": lr, "batch_size": 8, "init_seed": seed, "data_seed": seed, "epochs": 4,
        **(variant or {})}}))
    (d / "log.jsonl").write_text("".join(json.dumps({"event": "ckpt", "step": s, "val_top1": 0.7 * s / total}) + "\n"
                                         for s in (0, total // 2, total)))
    hits = [round(v * (rare if i < 2 else 0.8)) for i, v in enumerate(VAL)]
    cks = []
    for step in (0, total // 2, total):
        f = step / total
        at = {"plain": {"gap": gap, "pre_act": gap - matrix, "pre_logits": gap - matrix},
              "white": {"gap": gap, "pre_act": gap, "pre_logits": gap - tanh}}
        few = {rule: {layer: {p: {"5": {"acc": 0.2 + f * (at[rule].get(layer, 0.3) - 0.2) + (0.1 if p == "dogs_5way" else 0.0)
                                        + (0.3 if p.startswith("seen") else 0.0), "ci95": 0.005}} for p in POOLS}
                      for layer in LAYERS} for rule in ("plain", "white")}
        shape = {layer: {s: {"trace": 1.0, "pr": 10.0 + i, "erank": 20.0 + 10 * i, "top": [0.1] * 8,
                             "between": 0.1 * (i + 1), "between_white": 0.05 * (i + 1)} for s in ("seen", "inet", "dogs")}
                 for i, layer in enumerate(LAYERS)}
        stat = lambda srank: {"fro": 2.0, "spec": 1.0, "srank": srank, "erank": 2 * srank}
        weights = {name: {**stat(srank), "since_first": stat(srank / 2), "since_last": stat(srank / 4)}
                   for name, srank in (("blocks.0.attn.q.weight", 10.0), ("blocks.1.attn.q.weight", 30.0),
                                       ("blocks.0.mlp.fc1.weight", 8.0), ("pre_logits.weight", 60.0 * (1 - tanh)),
                                       ("head.weight", 12.0), ("embedding.weight", 24.0))}
        cks.append({"step": step, "val_top1": 0.7 * f, "fewshot": few, "shape": shape,
                    "tanh": {s: {"over90": 0.1 + tanh, "over99": 0.02, "mean_abs": 0.5, "rms_in": 1.0 + tanh}
                             for s in ("seen", "inet", "dogs")},
                    "val": {"images": sum(VAL), "top1": sum(hits) / sum(VAL), "top5": 0.9, "loss": 1.0, "hits": hits},
                    "train": {"images": 50, "top1": 0.8, "top5": 0.9, "loss": 1.0, "hits": [4] * 10},
                    "weights": weights, "sv": {"pre_logits.weight": [4.0, 2.0, 1.0], "head.weight": [3.0, 1.0]}})
    (d / "probe.json").write_text(json.dumps({
        "layers": LAYERS, "shots": [5], "pools": POOLS, "rules": ["plain", "white"],
        "sets": {s: {"images": 100, "classes": 10} for s in ("seen", "inet", "dogs")},
        "scored": {"val": {"images": sum(VAL), "per_class": VAL}, "train": {"images": 50, "per_class": [5] * 10}},
        "train_images_per_class": COUNTS, "checkpoints": cks}))
    return str(d)


@pytest.fixture()
def made(tmp_path):
    paths = [_run(tmp_path, "adamw", 2e-3, 0, 0.47, matrix=0.03, tanh=0.02)]
    for seed, e in ((0, 0.0), (1, 0.01), (2, -0.01)):
        paths += [_run(tmp_path, "adamw", 1e-3, seed, 0.46 + e, matrix=0.03 + e, tanh=0.03, rare=0.5),
                  _run(tmp_path, "muon", 3e-3, seed, 0.47 + e, matrix=0.0, tanh=0.01, rare=1.0)]
    return paths, settings([why.load(p) for p in paths])


def test_accuracy_is_read_per_pool_or_as_the_mean_of_the_unseen_pools(made):
    _, groups = made
    run = groups["adamw", 1e-3, 400][0, 0]
    assert why.probed(run) and why.unseen(run) == ["control", "dogs_5way"]        # not the 1-way pool, not the seen ones
    ck = why.last(run)
    assert why.acc(run, ck, "plain", "gap", "control") == pytest.approx(0.46)
    assert why.acc(run, ck, "plain", "gap", "dogs_5way") == pytest.approx(0.56)
    assert why.acc(run, ck, "plain", "gap", "unseen") == pytest.approx(0.51)
    assert why.acc(run, ck, "plain", "pre_logits", "unseen") == pytest.approx(0.48)
    assert why.acc(run, ck, "white", "pre_logits", "unseen") == pytest.approx(0.48)
    assert why.acc(run, ck, "plain", "gap", "absent") is None
    run["probe"]["checkpoints"] = run["probe"]["checkpoints"][:-1]
    assert not why.probed(run)                                                    # not probed at its last step


def test_steps_inside_a_run_and_differences_from_the_baseline_carry_their_intervals(made):
    _, groups = made
    base, muon, once = groups["adamw", 1e-3, 400], groups["muon", 3e-3, 400], groups["adamw", 2e-3, 400]
    cost = lambda r: why.acc(r, why.last(r), "plain", "pre_act", "unseen") - why.acc(r, why.last(r), "plain", "gap", "unseen")
    mean, half = why.within(base, cost)                       # the matrix costs 3, 4 and 2 points in the three seeds
    assert (mean, half) == (pytest.approx(-0.03), pytest.approx(4.303 * 0.01 / 3 ** 0.5))
    assert why.within(muon, cost) == (pytest.approx(0.0), pytest.approx(0.0, abs=1e-12))
    assert why.within(once, cost) == (pytest.approx(-0.03), None)
    gap = lambda r: why.acc(r, why.last(r), "plain", "gap", "unseen")
    diff, half = why.compare(base, muon, gap)                 # one point higher in every seed
    assert diff == pytest.approx(0.01) and half == pytest.approx(0.0, abs=1e-12)
    pre = lambda r: why.acc(r, why.last(r), "plain", "pre_logits", "unseen")
    diff, half = why.compare(base, muon, pre)                 # 4, 5 and 3 points
    assert (diff, half) == (pytest.approx(0.04), pytest.approx(4.303 * 0.01 / 3 ** 0.5))
    diff, half = why.compare(base, once, gap)                 # one run against the scatter of the baseline's seeds
    assert (diff, half) == (pytest.approx(0.01), pytest.approx(4.303 * 0.01 * (4 / 3) ** 0.5))
    assert why.compare(once, base, gap) is None and why.compare(base, {}, gap) is None


def _again(path, name, ridge, shift=0.0, **changed):
    """The run at `path` probed once more at its last step, as probe-<name>.json: the whitened rule with another
    ridge, which moves its `pre_logits` accuracy by `shift`."""
    with open(os.path.join(path, "probe.json")) as f:
        probe = json.load(f)
    ck = probe["checkpoints"][-1]
    for row in ck["fewshot"]["white"]["pre_logits"].values():
        row["5"]["acc"] += shift
    with open(os.path.join(path, f"probe-{name}.json"), "w") as f:
        json.dump({**probe, "ridge": ridge, "checkpoints": [ck], **changed}, f)


def test_pools_are_counted_as_clearly_ahead_or_behind_the_baseline(made):
    _, groups = made
    base, muon, once = groups["adamw", 1e-3, 400], groups["muon", 3e-3, 400], groups["adamw", 2e-3, 400]
    at = lambda rule, layer: lambda p: lambda r: why.acc(r, why.last(r), rule, layer, p)
    pools = why.unseen(base[0, 0])
    assert why.ahead_behind(base, muon, at("plain", "gap"), pools) == (2, 0)     # one point up on both, in every seed
    for r in muon.values():                                   # three points down on one pool: two behind the baseline
        why.last(r)["fewshot"]["plain"]["gap"]["control"]["5"]["acc"] -= 0.03
    assert why.ahead_behind(base, muon, at("plain", "gap"), pools) == (1, 1)
    assert why.ahead_behind(base, once, at("plain", "gap"), pools) == (0, 0)     # one run, within what a seed does
    assert why.ahead_behind(base, muon, at("plain", "gap"), ["absent"]) == (0, 0)   # a pool no run has: nothing to compare
    assert why.ahead_behind(base, muon, at("plain", "gap"), []) == (0, 0)


def test_other_ridges_of_the_whitened_rule_are_read_from_the_files_beside_the_probe(made, tmp_path):
    paths, _ = made
    for p in paths[1:]:                                       # the six seeded runs, not the single one
        _again(p, "ridge0.01", 0.01, shift=-0.01)
    _again(paths[1], "pools", 0.1, pools={"dogs_5way": {"n_way": 5}})           # another test: not comparable
    _again(paths[1], "same", 0.001, shift=-0.2)                                 # probe.json's own ridge: probe.json wins
    with open(os.path.join(paths[1], "probe-early.json"), "w") as f:            # not at the last step
        json.dump({**json.load(open(os.path.join(paths[1], "probe.json"))), "ridge": 0.5,
                   "checkpoints": [{"step": 200}]}, f)
    with open(os.path.join(paths[1], "probe-noridge.json"), "w") as f:          # does not say which ridge
        json.dump(json.load(open(os.path.join(paths[1], "probe.json"))), f)
    runs = [why.load(p) for p in paths]
    assert sorted(runs[0]["ridges"]) == [0.001] and sorted(runs[1]["ridges"]) == [0.001, 0.01]
    assert runs[1]["ridges"][0.001] is why.last(runs[1])
    groups = settings(runs)
    base, muon = groups["adamw", 1e-3, 400], groups["muon", 3e-3, 400]
    assert why.ridges(base) == [0.001, 0.01] and why.ridges(groups["adamw", 2e-3, 400]) == [0.001]
    run = base[0, 0]
    assert why.at_ridge(0.001, "pre_logits", "unseen")(run) == pytest.approx(0.48)
    assert why.at_ridge(0.01, "pre_logits", "unseen")(run) == pytest.approx(0.47)
    assert why.at_ridge(0.01, "gap", "unseen")(run) == pytest.approx(0.51)
    assert why.at_ridge(0.1, "gap", "unseen")(run) is None
    names = {key: f"{key[0]} {key[1]:g}" for key in groups}
    text = why.table_ridge(groups, names, ("adamw", 1e-3, 400))
    assert [line.split(" | ")[:3] for line in text[6:-1]] == [
        ["| adamw 0.001", "3 seeds", "0.001"], ["| adamw 0.001", "3 seeds", "0.01"],
        ["| muon 0.003", "3 seeds", "0.001"], ["| muon 0.003", "3 seeds", "0.01"]]      # the single run has one ridge
    # the tanh costs the baseline 3 points with the usual ridge and 4 with the larger one; Muon stays 3 points ahead
    assert "| adamw 0.001 | 3 seeds | 0.001 | 51.0 → 51.0 → 48.0 |  |  |  |  | +0.0 ± 0.0 | **-3.0** ± 0.0 | **-3.0** ± 0.0 |" in text
    assert "| adamw 0.001 | 3 seeds | 0.01 | 51.0 → 51.0 → 47.0 |  |  |  |  | +0.0 ± 0.0 | **-4.0** ± 0.0 | **-4.0** ± 0.0 |" in text
    assert ("| muon 0.003 | 3 seeds | 0.01 | 52.0 → 52.0 → 50.0 | **+1.0** ± 0.0 | **+3.0** ± 0.0 | 2 / 0 | 2 / 0 | "
            "+0.0 ± 0.0 | **-2.0** ± 0.0 | **-2.0** ± 0.0 |") in text
    assert why.table_ridge(settings([r for r in runs if 0.01 not in r["ridges"]]), names,
                           ("adamw", 2e-3, 400)) == []                                  # one ridge everywhere: no table


def test_a_control_run_is_a_setting_of_its_own_set_against_the_setting_it_varies(made, tmp_path):
    paths, _ = made
    ctrl = _run(tmp_path, "muon", 3e-3, 0, 0.465, matrix=0.03, tanh=0.03, rare=0.5, tag="-pladamw",
                variant={"muon_skip_pre_logits": True})
    runs = [why.load(p) for p in paths + [ctrl]]
    assert runs[-1]["variant"] == "pre_logits by AdamW" and runs[-1]["setting"] == "muon lr 0.003, pre_logits by AdamW"
    assert runs[0]["variant"] == ""
    assert curves.variant_of({"optimizer": "muon", "muon_blocks": "mlp", "muon_skip_pre_logits": True}) == \
        "on the MLP only"
    assert curves.variant_of({"optimizer": "muon", "muon_blocks": "attn"}) == "on the attention and pre_logits only"
    assert curves.variant_of({"optimizer": "muon", "muon_blocks": "all", "muon_skip_pre_logits": True}) == \
        "pre_logits by AdamW"
    assert curves.variant_of({"optimizer": "muon", "muon_blocks": "none"}) == "on pre_logits only"
    assert curves.variant_of({"optimizer": "adamw", "muon_blocks": "all"}) == ""
    assert curves.variant_of({"optimizer": "adamw", "no_pre_logits": True}) == "no pre_logits"
    groups = settings(runs)                                   # not one more seed 0 of Muon at 3e-3
    key, ref, base = ("muon", 3e-3, 400, "pre_logits by AdamW"), ("muon", 3e-3, 400), ("adamw", 1e-3, 400)
    assert list(groups)[-1] == key and list(groups[key]) == [(0, 0)] and len(groups[ref]) == 3
    assert why.controls(groups) == [(key, ref)] and why.controls(settings(runs[:-1])) == []
    assert why.by_rate(groups, 400)["muon"] == [(3e-3, groups[ref])]           # nor one more learning rate
    names = {k: label(k, v, 400) for k, v in groups.items()}
    assert names[key] == "Muon, lr 0.003, pre_logits by AdamW"

    files = [why._cka_name(r) for r in runs]
    at = {f: i for i, f in enumerate(files)}
    m = np.full((len(files), len(files)), 0.5)
    np.fill_diagonal(m, 1.0)
    for a, b, v in (("muon-lr0.003-pladamw-s0", "muon-lr0.003-s0", 0.9), ("muon-lr0.003-pladamw-s0", "adamw-lr0.001-s0", 0.6),
                    ("muon-lr0.003-s0", "muon-lr0.003-s1", 0.8), ("muon-lr0.003-s0", "muon-lr0.003-s2", 0.76),
                    ("muon-lr0.003-s1", "muon-lr0.003-s2", 0.78), ("muon-lr0.003-s0", "adamw-lr0.001-s0", 0.66)):
        i, j = at[f"nodog-{a}-400"], at[f"nodog-{b}-400"]
        m[i, j] = m[j, i] = v
    cka = {"files": files, "layers": ["gap", "pre_logits"], "images": 60,
           "between": {"gap": m.tolist(), "pre_logits": (m / 2).tolist()}}
    text = why.table_control(groups, names, base, cka)
    assert text[0] == "## The control: Muon, lr 0.003, pre_logits by AdamW"
    assert text[4] == ("| | Muon, lr 0.003 (3 seeds) | Muon, lr 0.003, pre_logits by AdamW (1 run) | against Muon, lr 0.003 | "
                       "against its seed 0 alone | AdamW, lr 0.001 (3 seeds) | against AdamW, lr 0.001 |")

    def row(title):
        return next(line for line in text if line.startswith(f"| {title} |"))
    # the layer costs the control 3 points (whitened) where it costs Muon 1 and the baseline 3; Muon's seeds agree,
    # so one more of them would fall on their mean, and the control's 2 points from it are clear
    assert row("The same, whitened (points)").startswith("| The same, whitened (points) | -1.0 | -3.0 | **-2.0** ± 0.0 | -2.0 | -3.0 |")
    assert row("Validation top-1, the rarest fifth of the classes (%)") == (
        "| Validation top-1, the rarest fifth of the classes (%) | 100.0 | 50.0 | **-50.0** ± 0.0 | -50.0 | 50.0 | +0.0 ± 0.0 |")
    assert row("`pre_logits` matrix: stable rank") == "| `pre_logits` matrix: stable rank | 59.4 | 58.2 | **-1.2** ± 0.0 | -1.2 | 58.2 | +0.0 ± 0.0 |"
    # Muon's seeds differ (52, 53, 51): one more would fall within 4.303 x 1 x (4/3)^0.5 = 5.0 points of their mean
    assert row("Unseen pools at `gap`, the test's rule (%)") == (
        "| Unseen pools at `gap`, the test's rule (%) | 52.0 | 51.5 | -0.5 ± 5.0 | -0.5 | 51.0 | +0.5 ± 5.0 |")
    assert text[-2] == ("How alike, linear CKA on the same images at `gap` / `pre_logits` (1: the same up to a rotation "
                        "and a scale): the control and Muon, lr 0.003's seed 0: 0.90 / 0.45; the control and AdamW, lr "
                        "0.001's seed 0: 0.60 / 0.30; for comparison, two seeds of Muon, lr 0.003: 0.78 / 0.39; Muon, lr "
                        "0.003 and AdamW, lr 0.001 on seed 0: 0.66 / 0.33.")
    assert why.table_control(settings(runs[:-1]), names, base, cka) == []

    pytest.importorskip("matplotlib")
    out = tmp_path / "with-control"
    why.main(["--runs", *paths, ctrl, "--out", str(out)])
    assert "control-pre_logits-by-adamw.png" in os.listdir(out)
    md = (out / "why.md").read_text()
    assert md.index("## The control: Muon, lr 0.003, pre_logits by AdamW") < md.index("## The last three stages")
    assert "| Muon, lr 0.003, pre_logits by AdamW | 1 run | " in md                  # a row in the other tables too


def test_values_along_training_use_the_checkpoints_every_seed_has(made):
    _, groups = made
    base = groups["adamw", 1e-3, 400]
    base[1, 1]["probe"]["checkpoints"].pop(1)                 # one seed lacks the middle checkpoint
    pct, mean, lo, hi = why.along(base, lambda r, c: why.acc(r, c, "plain", "gap", "control"))
    assert pct.tolist() == [0.0, 100.0] and mean.tolist() == pytest.approx([0.2, 0.46])
    assert lo.tolist() == pytest.approx([0.2, 0.45]) and hi.tolist() == pytest.approx([0.2, 0.47])
    assert why.along(base, lambda r, c: None) is None


def test_validation_accuracy_by_class_frequency_and_with_every_class_the_same(made):
    _, groups = made
    run = groups["adamw", 1e-3, 400][0, 0]
    parts = why.by_frequency(run)                             # two classes per group; the rarest pair is right half the time
    assert [(p["fewest"], p["most"], p["images"]) for p in parts] == [(5, 10, 6), (20, 40, 20), (80, 160, 20),
                                                                      (320, 640, 20), (1280, 2560, 40)]
    assert [p["top1"] for p in parts] == pytest.approx([0.5, 0.8, 0.8, 0.8, 0.8])
    assert why.macro(run) == pytest.approx((0.5 * 2 + 0.8 * 8) / 10) and why.macro(run, "train") == pytest.approx(0.8)
    assert why.last(run)["val"]["top1"] == pytest.approx((3 + 0.8 * 100) / 106)   # images, not classes
    assert why.by_frequency(groups["muon", 3e-3, 400][0, 0])[0]["top1"] == pytest.approx(1.0)
    assert len(why.by_frequency(run, parts=2)) == 2


def test_rare_and_common_classes_are_followed_from_checkpoint_to_checkpoint(made):
    _, groups = made
    base = groups["adamw", 1e-3, 400]
    for r in base.values():                                   # nothing right at the first checkpoint
        first = r["probe"]["checkpoints"][0]
        first["val"]["hits"], first["val"]["top1"] = [0] * 10, 0.0
    run = base[0, 0]
    assert [p["top1"] for p in why.by_frequency(run, ck=run["probe"]["checkpoints"][0])] == [0.0] * 5
    assert why.by_frequency(run)[0]["top1"] == pytest.approx(0.5)                 # the last checkpoint by default
    pct, everything, rare, common = why.rare_common(base)
    assert pct.tolist() == [0.0, 50.0, 100.0]
    assert everything.tolist() == pytest.approx([0.0, 100 * 83 / 106, 100 * 83 / 106])
    assert rare.tolist() == pytest.approx([0.0, 50.0, 50.0]) and common.tolist() == pytest.approx([0.0, 80.0, 80.0])
    assert why.at_level(everything, rare, 100 * 83 / 212) == pytest.approx(25.0)  # half-way up the first step
    run["probe"]["checkpoints"][1]["val"]["hits"] = [0, 0, 8, 8, 8, 8, 8, 8, 16, 16]
    assert why.rare_common({(0, 0): run})[2].tolist() == pytest.approx([0.0, 0.0, 50.0])
    for r in base.values():                                   # a class group without validation images: no curve
        r["probe"]["scored"]["val"]["per_class"] = [0, 0] + VAL[2:]
    assert why.rare_common(base) is None


def test_a_curve_is_read_where_another_first_reaches_a_level():
    x, y = [0.0, 10.0, 30.0, 50.0], [0.0, 1.0, 5.0, 9.0]
    assert why.at_level(x, y, 20.0) == pytest.approx(3.0) and why.at_level(x, y, 30.0) == pytest.approx(5.0)
    assert why.at_level(x, y, 50.0) == pytest.approx(9.0)
    assert why.at_level(x, y, 0.0) is None and why.at_level(x, y, 60.0) is None   # starts there; never gets there
    assert why.at_level([2.0, 1.0, 5.0, 4.0, 9.0], [0.0, 10.0, 20.0, 30.0, 40.0], 3.0) == pytest.approx(15.0)   # first time
    assert why.at_level([], [], 1.0) is None


def test_weight_statistics_are_averaged_over_the_matrices_of_a_kind(made):
    _, groups = made
    ck = why.last(groups["adamw", 1e-3, 400][0, 0])
    assert why.family(ck, "attn.q", "srank") == pytest.approx(20.0)               # blocks 0 and 1
    assert why.family(ck, "attn.q", "srank", "since_first") == pytest.approx(10.0)
    assert why.family(ck, "mlp.fc1", "srank") == pytest.approx(8.0) and why.family(ck, "head", "srank") == 12.0
    assert why.family(ck, "pre_logits", "srank") == pytest.approx(60.0 * 0.97)
    assert why.family(ck, "attn.k", "srank") is None


def test_block_by_block_weights_how_far_a_block_moved_and_the_lead_after_it(made, tmp_path):
    paths, groups = made
    run, base = groups["muon", 3e-3, 400][1, 1], groups["adamw", 1e-3, 400]
    assert why.block_numbers(run) == [1]
    first, end = run["probe"]["checkpoints"][0], why.last(run)
    fams = [f for _, part in why.PARTS for f in part]
    for ck, fro in ((first, 4.0), (end, 5.0)):           # every matrix of block 1: norm 4 at the start, 5 at the end
        ck["weights"] = {f"blocks.0.{f}.weight": {"fro": fro, "srank": 10.0 + i, "since_first": {"fro": 3.0, "srank": 2.0 + i}}
                         for i, f in enumerate(fams)}
    assert why.in_block(end, 1, ("attn.q", "attn.k", "attn.v", "attn.proj"), "srank") == pytest.approx(11.5)
    assert why.in_block(end, 1, ("mlp.fc1", "mlp.fc2"), "srank", "since_first") == pytest.approx(6.5)
    assert why.in_block(end, 2, ("mlp.fc1",), "srank") is None                       # no such block
    assert why.moved(run, 1) == pytest.approx(0.75) and why.moved(run, 2) is None    # 3 / 4: change over start
    # few-shot after block 1 (the made-up runs give 0.3 there under both rules): no lead; seed 1 against seed 1
    assert why.lead(base, "white")(run, "block1") == pytest.approx(0.0)
    assert why.lead(base, "white")(groups["adamw", 2e-3, 400][0, 0], "gap") == pytest.approx(1.0)
    stray = dict(run, seeds=(7, 7))
    assert why.lead(base, "plain")(stray, "gap") is None                             # the baseline has no seed 7
    # the runs that give Muon one kind of block matrix, with the setting they vary; none here yet
    assert why.splits(groups) == {}
    mlp = _run(tmp_path, "muon", 3e-3, 0, 0.47, tag="-mlp", variant={"muon_blocks": "mlp", "muon_skip_pre_logits": True})
    other = _run(tmp_path, "muon", 3e-3, 0, 0.47, tag="-pl", variant={"muon_skip_pre_logits": True})
    more = settings([why.load(p) for p in paths + [mlp, other]])
    assert list(why.splits(more)) == [("muon", 3e-3, 400), ("muon", 3e-3, 400, "on the MLP only")]
    # drawn dashed in blocks.png: the split run (its setting has seeds, so it is drawn as a band anyway), and Muon
    # at the baseline's own learning rate; not AdamW at another rate, not the run that only moves pre_logits
    at_base = _run(tmp_path, "muon", 1e-3, 0, 0.47)
    more = settings([why.load(p) for p in paths + [mlp, other, at_base]])
    drawn = {key: by_seed for key, by_seed in more.items() if len(by_seed) > 1}
    assert list(why.block_singles(more, drawn, ("adamw", 1e-3, 400))) == [("muon", 1e-3, 400),
                                                                          ("muon", 3e-3, 400, "on the MLP only")]


def test_learning_rates_are_lined_up_per_optimizer_and_untrained_runs_left_out(made, tmp_path):
    paths, groups = made
    rates = why.by_rate(groups, 400)
    assert [lr for lr, _ in rates["adamw"]] == [1e-3, 2e-3] and [len(b) for _, b in rates["adamw"]] == [3, 1]
    assert [lr for lr, _ in rates["muon"]] == [3e-3]
    assert why.by_rate(groups, 400, floor=0.8) == {"muon": rates["muon"]}      # AdamW's runs end at 78 %
    assert why.by_rate(groups, 300) == {}                                      # no setting of that length


def test_cka_is_averaged_over_seed_pairs_and_over_the_seeds_two_settings_share(made):
    _, groups = made
    seeded = {k: v for k, v in groups.items() if len(v) > 1}
    files = [f"nodog-{o}-lr{lr}-s{s}-400" for o, lr in (("adamw", "0.001"), ("muon", "0.003")) for s in range(3)]
    m = np.full((6, 6), 0.5)
    m[:3, :3], m[3:, 3:] = 0.9, 0.8                           # within AdamW, within Muon
    for s in range(3):
        m[s, 3 + s] = m[3 + s, s] = 0.6 + 0.1 * s             # AdamW against Muon on the same seed
    np.fill_diagonal(m, 1.0)
    cka = {"files": files, "layers": ["gap", "pre_logits"], "images": 60,
           "between": {"gap": m.tolist(), "pre_logits": (m / 2).tolist()}}
    same, across = why.cka_lines(cka, seeded)
    assert same["adamw", 1e-3, 400].tolist() == pytest.approx([0.9, 0.45])
    assert same["muon", 3e-3, 400].tolist() == pytest.approx([0.8, 0.4])
    assert list(across) == [(("adamw", 1e-3, 400), ("muon", 3e-3, 400))]
    assert across[("adamw", 1e-3, 400), ("muon", 3e-3, 400)].tolist() == pytest.approx([0.7, 0.35])
    assert why.cka_lines({**cka, "files": ["other-1", "other-2"], "between": {"gap": [[1, 0], [0, 1]],
                                                                               "pre_logits": [[1, 0], [0, 1]]}}, seeded) == ({}, {})


def test_the_figures_and_tables_of_made_up_probes(made, tmp_path):
    pytest.importorskip("matplotlib")
    paths, _ = made
    files = [f"{os.path.basename(p)}-400" for p in paths]
    eye = (0.5 + 0.5 * np.eye(len(files))).tolist()
    (tmp_path / "cka.json").write_text(json.dumps({"files": files, "layers": LAYERS, "images": 60,
                                                    "between": {layer: eye for layer in LAYERS}}))
    unprobed = _run(tmp_path, "sgd", 1.0, 0, 0.45)
    os.remove(os.path.join(unprobed, "probe.json"))
    out = tmp_path / "out"
    why.main(["--runs", *paths, unprobed, "--out", str(out), "--cka", str(tmp_path / "cka.json")])
    assert {"depth.png", "last.png", "shape.png", "classes.png", "weights.png", "blocks.png", "rare.png", "cka.png",
            "rates.png", "why.md"} == set(os.listdir(out))
    text = (out / "why.md").read_text()
    assert "Not read, because not probed at the last step: nodog-sgd-lr1-s0." in text
    assert "| AdamW, lr 0.001 | 3 seeds | 56.0 → 53.0 → 53.0 | 56.0 → 56.0 → 53.0 | 51.0 → 48.0 → 48.0 |" in text
    # inside a run: the matrix costs 3 points give or take 2.5 (plain), the tanh 3 points in every seed (whitened)
    assert "| AdamW, lr 0.001 | 3 seeds | **-3.0** ± 2.5 | +0.0 ± 0.0 | **-3.0** ± 2.5 | +0.0 ± 0.0 | **-3.0** ± 0.0 | **-3.0** ± 0.0 |" in text
    assert "| AdamW, lr 0.002 | 1 run | -3.0 | +0.0 | -3.0 | +0.0 | -2.0 | -2.0 |" in text
    # against the baseline: Muon is one point up at the encoder output and four at the layer the head reads
    assert "| Muon, lr 0.003 | `gap` | **+1.0** ± 0.0 | **+1.0** ± 0.0 | **+1.0** ± 0.0 | **+1.0** ± 0.0 |" in text
    assert "| Muon, lr 0.003 | `pre_logits` | **+4.0** ± 2.5 | **+3.0** ± 0.0 | **+4.0** ± 2.5 | **+3.0** ± 0.0 |" in text
    assert "| AdamW, lr 0.002 | `gap` | +1.0 ± 5.0 |" in text                 # one run: within what a seed does
    # pool by pool (two unseen pools): Muon clearly ahead on both at every stage under both rules; the single run
    # only where the baseline's seeds agree to the last digit
    assert "| Muon, lr 0.003 | 3 seeds | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 |" in text
    assert "| AdamW, lr 0.002 | 1 run | 0 / 0 | 2 / 0 | 2 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |" in text
    assert "strengths of its ridge" not in text                                 # nothing probed with another ridge
    assert "| Muon, lr 0.003 | 3 seeds | 11.0 | 11.0 | 2.0 | 1.01 | 2.0 | 1.00 | 59.4 | 118.8 | 2.0 |" in text
    assert "| AdamW, lr 0.001 | 40.0 → 50.0 → 60.0 | 40.0 → 50.0 → 60.0 | 30.0 → 40.0 → 50.0 |" in text
    # by depth: the settings with seeds, and the second against the baseline under each rule
    assert "| `pre_act` | 48.0 / 51.0 | 52.0 / 52.0 | **+4.0** ± 2.5 / **+1.0** ± 0.0 |" in text
    assert "| AdamW, lr 0.001 | 3 seeds | 78.3 | 74.0 | 50.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 |" in text
    assert "| | | **+2.8** ± 0.0 | **+10.0** ± 0.0 | **+50.0** ± 0.0 | +0.0 ± 0.0 |" in text
    # the layer the head reads at every checkpoint: saturation / stable rank / its cost under each rule, which the
    # made-up runs build up in step with training (half of 3 points at the middle checkpoint)
    assert "| 50 | 13.0 / 58.2 / -1.5 / -1.5 | 11.0 / 59.4 / +0.0 / -0.5 |" in text
    assert "| 100 | 13.0 / 58.2 / -3.0 / -3.0 | 11.0 / 59.4 / +0.0 / -1.0 |" in text
    # rare and common classes at every checkpoint (the made-up runs are right equally often at each of them)
    assert "| 50 | 78.3 / 50.0 / 80.0 | 81.1 / 100.0 / 80.0 |" in text
    assert "| 78 % | n/a | n/a |" in text                                       # no curve rises to a level here
    assert "-0.0" not in text
    assert "| AdamW, lr 0.001 | 2.0 / 20.0 / 10.0 / 2.0 | n/a / n/a / n/a / n/a |" in text
    assert "| `gap` | 0.50 | 0.50 | 0.50 |" in text
    # block by block: the made-up runs have one block, with one of its six matrices; the same 30 % after it
    assert "## Block by block at the end of training" in text and "| setting | 1 |" in text
    assert "| Muon, lr 0.003 | n/a / n/a |" in text and "| Muon, lr 0.003 | +0.0 ± 0.0 |" in text
    why.main(["--runs", paths[0], paths[1], "--out", str(tmp_path / "single")])     # no seeds, no CKA: still drawn
    assert "cka.png" not in os.listdir(tmp_path / "single") and "depth.png" in os.listdir(tmp_path / "single")
    why.main(["--runs", paths[1], "--out", str(tmp_path / "one")])                  # one learning rate: no rates.png
    assert "rates.png" not in os.listdir(tmp_path / "one") and "rates.png" in os.listdir(tmp_path / "single")
    with pytest.raises(SystemExit, match="baseline"):
        why.main(["--runs", paths[0], "--out", str(tmp_path / "none")])
    with pytest.raises(SystemExit, match="none of the runs"):
        why.main(["--runs", unprobed, "--out", str(tmp_path / "none")])
