"""The figure and table code: its arithmetic, and that it runs end to end on small made-up runs."""
import json

import numpy as np
import pytest

from icl import summary


def test_crossing_lies_on_the_line_between_the_two_points_around_it():
    assert summary.crossing([0, 10, 20], [0.0, 0.2, 0.6], 0.4) == pytest.approx(15.0)
    assert summary.crossing([0, 10], [0.5, 0.6], 0.4) == 0.0                   # already there at the first point
    assert summary.crossing([0, 10], [0.1, 0.2], 0.4) is None


def test_value_at_follows_a_path_that_need_not_rise_all_the_way():
    got = summary.value_at([0.0, 0.2, 0.1, 0.4], [1.0, 2.0, 3.0, 4.0], 0.3)
    assert got == pytest.approx(3.0 + (0.3 - 0.1) / (0.4 - 0.1))
    assert summary.value_at([0.0, 0.2], [1.0, 2.0], 0.5) is None


def test_emergence_names_the_first_checkpoint_clear_of_the_untrained_network():
    steps, acc, ci = np.array([0, 100, 200, 400]), np.array([0.26, 0.27, 0.36, 0.46]), np.full(4, 0.01)
    e = summary.emergence(steps, acc, ci)
    assert e["first_above"] == 200                                             # 0.27 - 0.01 is not above 0.26 + 0.01
    assert e["half"] == pytest.approx(200.0)                                   # half of the gain of 0.20
    assert e["most"] == pytest.approx(200 + (0.44 - 0.36) / (0.46 - 0.36) * 200)


def _run(opt, lr, steps, vals):
    return {"optimizer": opt, "lr": lr, "ckpt": [{"step": s, "val_top1": v} for s, v in zip(steps, vals)],
            "fewshot": None, "train": [], "total_steps": 400, "batch_size": 8}


def test_the_best_run_is_chosen_at_the_last_checkpoint_the_runs_share():
    a = _run("adamw", 1e-3, [0, 100, 200, 400], [0.0, 0.3, 0.5, 0.7])
    b = _run("adamw", 3e-3, [0, 100, 200], [0.0, 0.4, 0.6])                    # under way, ahead at step 200
    c = _run("adamw", 1e-2, [0, 100], [0.0, 0.9])                              # less than half as far: not considered
    best = summary.best_runs([a, b, c])
    assert best["adamw"][0] is b and best["adamw"][1] == 200


def _vit_run(root, opt, lr, final_val, final_dogs, total=400, gap_shift=0.0):
    d = root / f"nodog-{opt}-lr{lr:g}-s0"
    d.mkdir()
    steps = [0, 50, 100, 200, total]
    (d / "run.json").write_text(json.dumps({"total_steps": total, "args": {
        "optimizer": opt, "lr": lr, "batch_size": 8, "init_seed": 0, "data_seed": 0}}))
    rows = [{"event": "train", "step": s, "loss": 1.0, "images_per_s": 100.0} for s in range(10, total + 1, 10)]
    rows += [{"event": "ckpt", "step": s, "val_top1": final_val * i / 4} for i, s in enumerate(steps)]
    (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pools = {"dogs_5way": {"n_way": 5}, "seen_5way": {"n_way": 5}}
    cks = [{"step": s, "val_top1": final_val * i / 4, "results": {layer: {
        p: {"5": {"acc": 0.26 + (final_dogs + shift - 0.26) * i / 4, "ci95": 0.005}} for p in pools}
        for layer, shift in (("pre_logits", 0.0), ("gap", gap_shift))}}
        for i, s in enumerate(steps)]
    (d / "fewshot.json").write_text(json.dumps({"pools": pools, "checkpoints": cks}))
    return str(d)


def test_the_summary_draws_and_tabulates_made_up_runs(tmp_path):
    pytest.importorskip("matplotlib")
    runs = [_vit_run(tmp_path, "adamw", 1e-3, 0.70, 0.50), _vit_run(tmp_path, "adamw", 3e-4, 0.68, 0.45),
            _vit_run(tmp_path, "muon", 1e-3, 0.72, 0.55, gap_shift=-0.08), _vit_run(tmp_path, "sgd", 0.3, 0.66, 0.52)]
    out = tmp_path / "out"
    summary.main(["--runs", *runs, "--out", str(out)])
    for name in ("best.png", "sweep.png", "pools.png", "final.png", "layers.png", "summary.md"):
        assert (out / name).stat().st_size > 0
    text = (out / "summary.md").read_text()
    # against the baseline on the one pool that is not the trained classes: Muon is ahead on one layer (0.55 against
    # 0.50) and behind on the other (0.47 against 0.50), and the table says so
    assert "| Muon, lr 0.001 | 1 | 0 | +5.0 points | 0 | 1 | -3.0 points |" in text
    assert "| AdamW, lr 0.0003 | 0 | 1 | -5.0 points | 0 | 1 | -5.0 points |" in text
    assert "AdamW, lr 0.001 **(best)**" in text and "Muon, lr 0.001 **(best)**" in text
    # Muon passes the baseline's final validation accuracy (0.70) and few-shot accuracy (0.50) before its own end
    row = next(line for line in text.splitlines() if line.startswith("| Muon, lr 0.001 | 100 |"))
    assert "step 378 (94 % of training)" in row and "step 262 (66 % of training)" in row
    slow = next(line for line in text.splitlines() if line.startswith("| AdamW, lr 0.0003 | 100 |"))
    assert slow.count("not reached") == 2


def _seq_run(root, opt, lr, seed, seen, dogs, control, steps=100, encoder_step=400):
    d = root / f"{opt}-lr{lr:g}-s{seed}"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "optimizer": opt, "args": {"lr": lr, "seed": seed, "steps": steps}, "chance": 0.2, "ways": 5, "shots": 5,
        "nearest_mean": {"seen": 0.9, "dogs": 0.5, "control": 0.8}, "encoder_step": encoder_step}))
    rows = []
    for i, s in enumerate([0, 10, 20, 50, steps]):
        ramp = min(1.0, i / 2)                                                 # half-way at step 10, there at 20
        rows.append({"event": "eval", "step": s, **{k: 0.2 + (v - 0.2) * ramp for k, v in
                                                    (("seen", seen), ("dogs", dogs), ("control", control))}})
    rows += [{"event": "train", "step": s, "loss": 1.6 / (1 + s / 10)} for s in range(10, steps + 1, 10)]
    (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(d)


def _seq_sweep(root, shift=0.0):
    runs = []
    for seed in (0, 1):
        e = 0.002 * seed
        runs += [_seq_run(root, "adamw", 1e-4, seed, 0.90 + e, 0.48 + shift, 0.82),
                 _seq_run(root, "adamw", 1e-3, seed, 0.92 + e, 0.45 + shift, 0.80),
                 _seq_run(root, "muon", 1e-3, seed, 0.91 + e, 0.46 + shift, 0.81),
                 _seq_run(root, "sgd", 0.1, seed, 0.89 + e, 0.47 + shift, 0.79)]
    return runs


def test_the_sequence_model_figures_and_tables_run_on_made_up_runs(tmp_path):
    pytest.importorskip("matplotlib")
    from icl import seqcross, seqseries, seqsummary
    root = tmp_path / "seq"
    late, early = _seq_sweep(root / "enc-400"), _seq_sweep(root / "enc-200", shift=-0.05)
    out = tmp_path / "out"
    seqsummary.main(["--runs", *late, "--out", str(out)])
    for name in ("emergence.png", "sweep.png", "tradeoff.png", "summary.md"):
        assert (out / name).stat().st_size > 0
    text = (out / "summary.md").read_text()
    assert "| AdamW **(best on seen)** | 0.001 |" in text and "| AdamW **(best on control)** | 0.0001 |" in text
    row = next(line for line in text.splitlines() if line.startswith("| AdamW **(best on seen)**"))
    assert row.rstrip(" |").endswith("10 (10 to 10)")                           # half-way to its final value at step 10
    assert "| AdamW | 0.001 | 0 | 2 | 10 | 18 |" in text                        # a tenth, half, nine tenths of the rise

    seqseries.main(["--root", str(root), "--encoder", "enc", "--out", str(out)])
    series = (out / "series.md").read_text()
    assert (out / "series.png").stat().st_size > 0 and "| 200 | 50.0 |" in series and "| 400 | 100.0 |" in series

    seqcross.main(["--encoders", f"early={root / 'enc-200'}", f"late={root / 'enc-400'}", "--out", str(out)])
    cross = (out / "cross.md").read_text()
    assert (out / "cross.png").stat().st_size > 0
    assert "| early | AdamW | 0.001 | 40.0 ± 0.0 % |" in cross                  # chosen on seen: the higher rate
    assert "| late | AdamW | 0.0001 | 48.0 ± 0.0 % |" in cross                  # chosen on control: the lower one
    assert cross.count("nearest class mean (no training)") == 4
