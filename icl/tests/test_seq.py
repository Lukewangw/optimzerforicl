"""The sequence model: how sequences are drawn, the fixed rule, and that a small model learns to use the examples."""
import json
import os

import numpy as np
import pytest
import torch

from icl import seq


def clusters(rng, n_classes, per_class, first_label, d=16, noise=0.35):
    centres = rng.normal(size=(n_classes, d))
    x = (centres[:, None, :] + noise * rng.normal(size=(n_classes, per_class, d))).reshape(-1, d)
    return x.astype(np.float16), np.repeat(np.arange(first_label, first_label + n_classes), per_class)


@pytest.fixture(scope="module")
def feats_file(tmp_path_factory):
    rng = np.random.default_rng(0)
    tx, ty = clusters(rng, 80, 30, 0)
    cx, cy = clusters(rng, 12, 20, 200)
    sx, sy = clusters(rng, 24, 12, 151)                # "dog breeds": test classes 151-174
    path = str(tmp_path_factory.mktemp("seq") / "f.npz")
    np.savez(path, step=7, val_top1=0.5, base=np.arange(80), train_gap=tx, train_pre=tx, train_y=ty,
             val_gap=tx[::3], val_pre=tx[::3], val_y=ty[::3], control_gap=cx, control_pre=cx, control_y=cy,
             tests_gap=sx, tests_pre=sx, tests_y=sy)
    return path


def test_sequences_have_distinct_classes_and_images():
    y = np.repeat(np.arange(7), [9, 4, 3, 9, 6, 2, 8])         # class 5 has too few images for 3 shots + 1
    x = torch.arange(len(y), dtype=torch.float32)[:, None].repeat(1, 4)
    pool = seq.Pool(x, y, min_images=4)
    assert pool.classes.tolist() == [0, 1, 3, 4, 6]            # 2 and 5 are left out
    ctx, slot, qry, answer = pool.draw(200, 3, 3, torch.Generator().manual_seed(1))
    assert ctx.shape == slot.shape == (200, 9) and qry.shape == answer.shape == (200,)
    for b in range(200):
        classes = y[ctx[b].numpy()].reshape(3, 3)
        assert (classes == classes[:, :1]).all() and len(set(classes[:, 0])) == 3      # three classes, three of each
        assert len(set(ctx[b].tolist()) | {int(qry[b])}) == 10                         # ten different images
        assert sorted(set(slot[b].tolist())) == [0, 1, 2]
        of_class = {int(c): int(s) for c, s in zip(y[ctx[b].numpy()], slot[b])}
        assert of_class[int(y[int(qry[b])])] == int(answer[b])                         # the asked image's class slot
    assert len({tuple(slot[b, ::3].tolist()) for b in range(200)}) == 6                # every order of slots occurs
    feats, slots = pool.tokens(ctx, slot, qry, 3)
    assert feats.shape == (200, 10, 4) and (slots[:, -1] == 3).all() and (slots[:, :-1] == slot).all()
    assert (feats[:, -1, 0] == qry).all()


def test_distinct_draws_every_set_equally_often():
    gen = torch.Generator().manual_seed(0)
    count = torch.tensor([3, 4, 12000]).repeat(20000, 1)
    got = seq.distinct(count, 3, gen)
    assert got.shape == (20000, 3, 3) and (got >= 0).all() and (got < count[..., None]).all()
    assert (got.sort(-1).values.diff(dim=-1) > 0).all()                       # three different numbers every time
    assert (got[:, 0].sort(-1).values == torch.tensor([0, 1, 2])).all()       # 3 of 3: all of them
    first = torch.bincount(got[:, 1, 0], minlength=4) / 20000.0               # 3 of 4: each number first a quarter of the time
    assert (first - 0.25).abs().max() < 0.02
    sets = {tuple(r) for r in got[:, 1].sort(-1).values.tolist()}
    assert len(sets) == 4                                                     # and every one of the four sets occurs
    left_out = torch.tensor([6 - sum(r) for r in got[:, 1].tolist()])         # the number not drawn (0+1+2+3 = 6)
    assert (torch.bincount(left_out, minlength=4) / 20000.0 - 0.25).abs().max() < 0.02
    assert abs(float(got[:, 2].float().mean()) - 5999.5) < 80                 # a large class: uniform over its images


def test_the_fixed_rule_answers_separable_clusters():
    rng = np.random.default_rng(3)
    x, y = clusters(rng, 10, 8, 0, noise=0.05)
    pool = seq.Pool(torch.from_numpy(x.astype(np.float32)), y, min_images=4)
    ctx, slot, qry, answer = pool.draw(300, 5, 3, torch.Generator().manual_seed(0))
    feats, slots = pool.tokens(ctx, slot, qry, 5)
    assert (seq.nearest_mean(feats, slots, 5) == answer).all()


def test_eval_steps_are_dense_at_the_start():
    assert seq.eval_steps(1200, 500) == [0, 10, 20, 50, 100, 200, 500, 1000, 1200]


def test_a_small_model_learns_to_use_the_examples(feats_file, tmp_path):
    common = ["train", "--features", feats_file, "--steps", "300", "--warmup-steps", "30", "--batch-size", "128",
              "--ways", "4", "--shots", "2", "--eval-shots", "2", "--width", "64", "--depth", "2", "--heads", "2",
              "--eval-sequences", "400", "--eval-every", "100", "--device", "cpu"]
    seq.main(common + ["--out", str(tmp_path / "adamw"), "--optimizer", "adamw", "--seeds", "0,1", "--parallel"])
    seq.main(common + ["--out", str(tmp_path / "muon-s0"), "--optimizer", "muon", "--seed", "0"])

    def read(name):
        with open(tmp_path / name / "run.json") as f:
            run = json.load(f)
        with open(tmp_path / name / "log.jsonl") as f:
            rows = [json.loads(line) for line in f]
        return run, [r for r in rows if r["event"] == "eval"], [r for r in rows if r["event"] == "train"]

    run, evals, trains = read("adamw-s0")
    assert set(run["pools"]) == {"train", "seen", "control", "dogs", "inet"} and run["pools"]["dogs"]["classes"] == 24
    assert run["chance"] == 0.25 and run["encoder_step"] == 7 and all(v > 0.95 for v in run["nearest_mean"].values())
    assert [e["step"] for e in evals] == list(range(0, 301, 10)) and len(trains) == 30
    assert set(evals[3]) == {"event", "step", "seen", "dogs", "control"}      # step 30: only the watched pools
    assert {"train", "inet"} < set(evals[5])                                  # step 50: every pool
    assert abs(evals[0]["dogs"] - 0.25) < 0.1                       # nothing learned yet: chance
    assert evals[-1]["dogs"] > 0.9 and evals[-1]["control"] > 0.9   # classes it never saw, answered from examples
    assert os.path.exists(tmp_path / "adamw-s1" / "seq.pt")
    other, other_evals, _ = read("muon-s0")
    assert other["init_sha"] == run["init_sha"] != read("adamw-s1")[0]["init_sha"]     # same seed, same start
    assert other_evals[0] == evals[0] and other["nearest_mean"] == run["nearest_mean"]  # and the same questions
    assert other_evals[-1]["dogs"] > 0.8
    assert [g["optimizer"] for g in other["optimizer_groups"]][0] == "Muon"
