"""Settings under several seeds and the run planned shorter: the arithmetic, and that it runs on made-up runs."""
import json

import pytest

from icl import seeds
from icl.curves import load_run


def _run(root, opt, lr, seed, val, dogs, gap_shift=0.0, total=400, epochs=4, speed=100.0, tested=True):
    """A run of `total` steps that ends at validation top-1 `val` and few-shot `dogs` (pre_logits; gap is shifted)."""
    d = root / f"nodog-{opt}-lr{lr:g}-e{epochs}-s{seed}"
    d.mkdir()
    steps = [0, total // 8, total // 4, total // 2, total]
    (d / "run.json").write_text(json.dumps({"total_steps": total, "args": {
        "optimizer": opt, "lr": lr, "batch_size": 8, "init_seed": seed, "data_seed": seed, "epochs": epochs}}))
    rows = [{"event": "train", "step": s, "loss": 1.0, "images_per_s": speed} for s in range(10, total + 1, 10)]
    rows += [{"event": "ckpt", "step": s, "val_top1": val * i / 4} for i, s in enumerate(steps)]
    (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pools = {"dogs_5way": {"n_way": 5}, "control": {"n_way": 20}, "seen_5way": {"n_way": 5}}
    cks = [{"step": s, "val_top1": val * i / 4, "results": {layer: {
        p: {"5": {"acc": 0.26 + (dogs + shift - 0.26) * i / 4, "ci95": 0.005}} for p in pools}
        for layer, shift in (("pre_logits", 0.0), ("gap", gap_shift))}}
        for i, s in enumerate(steps if tested else steps[:-1])]
    (d / "fewshot.json").write_text(json.dumps({"pools": pools, "checkpoints": cks}))
    return str(d)


def test_the_interval_of_a_mean_is_students_t_on_the_values():
    assert seeds.interval([0.01, 0.02, 0.03]) == pytest.approx(4.303 * 0.01 / 3 ** 0.5)
    assert seeds.interval([0.5]) is None


def test_one_more_seed_would_fall_within_students_t_on_the_seeds_widened_for_their_mean():
    assert seeds.prediction([0.01, 0.02, 0.03]) == pytest.approx(4.303 * 0.01 * (4 / 3) ** 0.5)
    assert seeds.prediction([0.5]) is None


def test_seeds_are_compared_seed_by_seed_and_a_single_run_with_the_scatter_of_the_baselines_seeds(tmp_path):
    paths = [_run(tmp_path, "adamw", 2e-3, 0, 0.705, 0.56)]
    for seed, e in ((0, 0.0), (1, 0.01), (2, -0.01)):
        paths += [_run(tmp_path, "adamw", 1e-3, seed, 0.70 + e, 0.50 + e),
                  _run(tmp_path, "muon", 3e-3, seed, 0.72 + e, 0.55 + 2 * e)]
    groups = seeds.settings([load_run(p) for p in paths])
    base, once = groups["adamw", 1e-3, 400], groups["adamw", 2e-3, 400]
    diff, half = seeds.compare(base, groups["muon", 3e-3, 400], "pre_logits", "dogs_5way", 5)
    assert (diff, half) == (pytest.approx(0.05), pytest.approx(4.303 * 0.01 / 3 ** 0.5))
    assert seeds.clear((diff, half))                                           # 5 points, give or take 2.5
    diff, half = seeds.compare(base, once, "pre_logits", "dogs_5way", 5)
    assert (diff, half) == (pytest.approx(0.06), pytest.approx(4.303 * 0.01 * (4 / 3) ** 0.5))
    assert seeds.clear((diff, half))                                           # 6 points where a seed stays within 5
    assert not seeds.clear(seeds.compare(base, once, "val", "dogs_5way", 5))   # half a point: a seed could do that
    assert seeds.compare(once, base, "val", "dogs_5way", 5) is None            # several runs against a single one
    assert not seeds.clear(None) and not seeds.clear((0.1, None))


def test_runs_that_differ_only_in_seed_are_one_setting_and_a_shorter_run_is_its_own(tmp_path):
    paths = [_run(tmp_path, "muon", 3e-3, 0, 0.73, 0.56), _run(tmp_path, "muon", 3e-3, 1, 0.74, 0.55),
             _run(tmp_path, "muon", 3e-3, 0, 0.72, 0.54, total=300, epochs=3),
             _run(tmp_path, "adamw", 1e-3, 0, 0.71, 0.52)]
    groups = seeds.settings([load_run(p) for p in paths])
    assert list(groups) == [("adamw", 1e-3, 400), ("muon", 3e-3, 400), ("muon", 3e-3, 300)]
    assert list(groups["muon", 3e-3, 400]) == [(0, 0), (1, 1)]
    assert seeds.short_pairs(groups) == [(("muon", 3e-3, 300), ("muon", 3e-3, 400))]
    assert seeds.label(("muon", 3e-3, 300), groups["muon", 3e-3, 300], 400) == "Muon, lr 0.003, 3 epochs"
    d = seeds.differences(groups["adamw", 1e-3, 400], groups["muon", 3e-3, 400], "val", "dogs_5way", 5)
    assert d == {(0, 0): pytest.approx(0.02)}                                  # only the seed both settings have


def test_a_run_not_yet_tested_at_its_last_step_is_not_finished(tmp_path):
    assert seeds.finished(load_run(_run(tmp_path, "adamw", 1e-3, 0, 0.71, 0.52)))
    assert not seeds.finished(load_run(_run(tmp_path, "adamw", 1e-3, 1, 0.71, 0.52, tested=False)))


def test_the_figures_and_tables_of_made_up_seeds(tmp_path):
    pytest.importorskip("matplotlib")
    runs = []
    for seed, e in ((0, 0.0), (1, 0.01), (2, -0.01)):
        runs += [_run(tmp_path, "adamw", 1e-3, seed, 0.70 + e, 0.50 + e),
                 _run(tmp_path, "muon", 3e-3, seed, 0.72 + e, 0.55 + 2 * e, gap_shift=-0.08, speed=80.0),
                 _run(tmp_path, "sgd", 1.0, seed, 0.69 - e, 0.52 + e)]
    runs += [_run(tmp_path, "muon", 3e-3, 0, 0.71, 0.56, total=300, epochs=3, speed=80.0),
             _run(tmp_path, "adamw", 2e-3, 0, 0.705, 0.51),
             _run(tmp_path, "sgd", 0.3, 0, 0.6, 0.5, tested=False)]
    out = tmp_path / "out"
    seeds.main(["--runs", *runs, "--out", str(out)])
    for name in ("seeds.png", "curves.png", "short.png", "seeds.md"):
        assert (out / name).stat().st_size > 0
    text = (out / "seeds.md").read_text()
    assert "Not read, because not finished and tested at the last step: nodog-sgd-lr0.3-e4-s0." in text
    # per setting: mean and range over the seeds, and what a run costs (400 steps of 8 images at 100 or 80 a second)
    assert ("| AdamW, lr 0.001 | 0, 1, 2 | 70.0 % | 69.0 to 71.0 | 50.0 % | 49.0 to 51.0 | 50.0 % | 49.0 to 51.0 "
            "| 0.0 |") in text
    assert "| Muon, lr 0.003, 3 epochs | 0 | 71.0 % | one seed |" in text
    # seed by seed against the baseline: Muon is 2 points ahead on validation in every seed
    assert "| Muon, lr 0.003 | val top-1 | +2.0 | +2.0 | +2.0 | +2.0 | +2.0 to +2.0 |" in text
    assert "| Muon, lr 0.003 | dogs_5way 5-shot from pre_logits | +5.0 | +6.0 | +4.0 | +5.0 | +2.5 to +7.5 |" in text
    assert "| Muon, lr 0.003 | dogs_5way 5-shot from gap | -3.0 | -2.0 | -4.0 | -3.0 | -5.5 to -0.5 |" in text
    assert "| SGD, lr 1 | val top-1 | -1.0 | -3.0 | +1.0 | -1.0 | -6.0 to +4.0 |" in text
    assert "| AdamW, lr 0.002 | val top-1 |" not in text.split("## The settings run once")[0]
    # the settings run once: half a point is what another seed of the baseline could do (5 points here), 6 are not
    assert "| AdamW, lr 0.002 | val top-1 | 70.5 % | 70.0 % | +0.5 | ±5.0 |" in text
    assert ("| Muon, lr 0.003, 3 epochs | dogs_5way 5-shot from pre_logits | 56.0 % | 50.0 % | **+6.0** | ±5.0 |"
            in text)
    # over the two pools that are not the trained classes: clearly ahead on one layer, clearly behind on the other
    assert "| AdamW, lr 0.002 | pre_logits | 1 run | 0 | 0 | 2 | +1.0 points |" in text
    assert "| Muon, lr 0.003 | pre_logits | 3 seeds | 2 | 0 | 0 | +5.0 points |" in text
    assert "| Muon, lr 0.003 | gap | 3 seeds | 0 | 2 | 0 | -3.0 points |" in text
    assert "| Muon, lr 0.003, 3 epochs | pre_logits | 1 run | 2 | 0 | 0 | +6.0 points |" in text
    assert "| dogs_5way | +1.0 ± 5.0 | **+5.0** ± 2.5 | **+6.0** ± 5.0 | **+2.0** ± 0.0 |" in text      # pre_logits
    assert "| dogs_5way | +1.0 ± 5.0 | **-3.0** ± 2.5 | **+6.0** ± 5.0 | **+2.0** ± 0.0 |" in text      # gap
    # the seeds of one setting: two points apart on validation, four on the few-shot test for Muon
    assert "| Muon, lr 0.003 | 2.0 | 4.0 | 4.0 | 4.0 | 4.0 |" in text
    # the short run: 300 steps at 80 images a second against the baseline's 400 at 100
    assert ("Muon, lr 0.003, 3 epochs ends 1.0 points above the mean of AdamW, lr 0.001 on validation top-1 (whose "
            "seeds run from 69.0 to 71.0), for 6 % fewer GPU-hours.") in text
    # and against its own setting at full length: a point lower, which another seed could be, for a quarter less
    assert "Against its own setting at full length, for 25 % fewer GPU-hours:" in text
    assert "| Muon, lr 0.003, 3 epochs | val top-1 | 71.0 % | 72.0 % | -1.0 | ±5.0 |" in text
