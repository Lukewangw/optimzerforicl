"""linear.py: the ridge solver against numpy, features whose answer is known, and end to end on a tiny run."""
import glob
import json
import os

import numpy as np
import pytest
import torch

from icl import fewshot, linear, probe, train

MODEL = ["--width", "32", "--depth", "2", "--mlp-dim", "64", "--heads", "2", "--patch", "8", "--image-size", "32"]
LAMS = [0.01, 0.1, 1.0, 10.0]


def test_ridge_solver_agrees_with_numpy_in_both_forms():
    rng = np.random.default_rng(0)
    for n, d in ((12, 5), (5, 12)):                       # more rows than dimensions, and fewer
        x = rng.normal(size=(3, n, d)) * rng.uniform(0.5, 3.0, size=d)
        y = np.eye(4)[rng.integers(0, 4, size=(3, n))]
        xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
        for form in ("primal", "dual", None):
            w = linear.ridge_weights(torch.from_numpy(xc), torch.from_numpy(yc), LAMS, form).numpy()
            assert w.shape == (len(LAMS), 3, d, 4)
            for li, lam in enumerate(LAMS):
                for e in range(3):
                    g = xc[e].T @ xc[e]
                    want = np.linalg.solve(g + lam * np.trace(g) / d * np.eye(d), xc[e].T @ yc[e])
                    np.testing.assert_allclose(w[li, e], want, rtol=1e-6, atol=1e-8)


def _episodes(labels, n_way, k, n_ep=100, q=15, name="pool"):
    return fewshot.make_episodes(labels, n_way, k, q, n_ep, 0, name)


def test_linear_separates_what_class_means_cannot_and_is_at_chance_on_random_labels():
    rng = np.random.default_rng(0)
    n_way, per, d = 4, 60, 24
    labels = np.repeat(np.arange(n_way), per)
    signal = np.zeros((len(labels), d))
    signal[np.arange(len(labels)), labels] = 3.0                                    # a class is a direction
    signal[:, n_way:] = rng.normal(size=(len(labels), d - n_way)) * 0.3
    big = rng.normal(size=d)                                                        # a large shared direction
    f = (signal + rng.normal(size=(len(labels), 1)) * 40.0 * big / np.linalg.norm(big)).astype(np.float32)
    ref = slice(0, len(f))
    pools = {"p": (slice(0, len(f)), labels, n_way)}
    asked = {"p": {5: _episodes(labels, n_way, 5)}}
    res = linear.evaluate_layer(torch.from_numpy(f), ref, pools, asked, [5], [0.01], n_query=15, n_episodes=100,
                                seed=0)["p"]["5"]
    assert res["linear"]["0.01"]["acc"] > 0.97
    assert res["plain"]["acc"] < 0.9                                                # the nearest mean is led astray
    rand = rng.integers(0, n_way, size=len(labels))
    asked = {"p": {5: _episodes(rand, n_way, 5)}}
    pools = {"p": (slice(0, len(f)), rand, n_way)}
    res = linear.evaluate_layer(torch.from_numpy(f), ref, pools, asked, [5], LAMS, n_query=15, n_episodes=100,
                                seed=0)["p"]["5"]
    for cell in res["linear"].values():
        assert abs(cell["acc"] - 0.25) < 0.04 and cell["ci95"] > 0


def test_plain_cell_is_probe_py_own_number():
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(6), 40)
    f = torch.from_numpy((rng.normal(size=(6, 12))[labels] + rng.normal(size=(240, 12)) + 5.0).astype(np.float32))
    ref = slice(0, 240)
    pools = {"p": (slice(0, 240), labels, 5)}
    asked = {"p": fewshot.pool_episodes(labels, "p", n_way=5, shots=[1, 5], n_query=10, n_episodes=60, seed=3)}
    got = linear.evaluate_layer(f, ref, pools, asked, [1, 5], [0.1], n_query=10, n_episodes=60, seed=3)["p"]
    for k in (1, 5):
        sup, qry, _ = asked["p"][k]
        for rule, prep in (("plain", probe.plain(f, f[ref])), ("white", probe.whitened(f, f[ref]))):
            acc = fewshot.episode_accuracy(prep, sup, qry)
            assert got[str(k)][rule]["acc"] == acc.mean()
            assert got[str(k)][rule]["ci95"] == pytest.approx(1.96 * acc.std(ddof=1) / np.sqrt(60))
        assert list(got[str(k)]["linear"]) == ["0.1"]


def test_a_k_a_pool_cannot_supply_is_left_out():
    labels = np.repeat(np.arange(5), 8)                                             # 8 images per class
    f = torch.randn(40, 6)
    pools = {"p": (slice(0, 40), labels, 3)}
    asked = {"p": fewshot.pool_episodes(labels, "p", n_way=3, shots=[1, 8], n_query=4, n_episodes=20, seed=0)}
    res = linear.evaluate_layer(f, slice(0, 40), pools, asked, [1, 8], [1.0], n_query=4, n_episodes=20, seed=0)
    assert list(res["p"]) == ["1"] and set(res["p"]["1"]) == {"plain", "white", "linear"}


@pytest.fixture(scope="module")
def runs(pack_dir, roles_file, tmp_path_factory):
    out = {}
    for name, flags in (("with", []), ("without", ["--no-pre-logits"])):
        dst = tmp_path_factory.mktemp("linear") / name
        train.main(["--train-pack", pack_dir, "--val-pack", pack_dir, "--roles", roles_file, "--out", str(dst),
                    "--optimizer", "adamw", *MODEL, "--batch-size", "16", "--epochs", "3", "--warmup-steps", "2",
                    "--randaug-layers", "0", "--log-every", "1", "--device", "cpu", "--workers", "0", *flags])
        out[name] = str(dst)
    return out


def test_script_runs_on_a_tiny_run_with_and_without_the_last_hidden_layer(runs, pack_dir, roles_file, tmp_path):
    for name, layers in (("with", ["gap", "pre_act", "pre_logits"]), ("without", ["gap"])):
        run = runs[name]
        common = ["--run", run, "--roles", roles_file, "--heldout-pack", pack_dir, "--base-pack", pack_dir,
                  "--extra", f"some={pack_dir}:0-3:2,0", "--episodes", "20", "--query", "5", "--per-class", "20",
                  "--seen-per-class", "20", "--workers", "0", "--device", "cpu"]
        linear.main([*common, "--shots", "1,5,20", "--lams", "0.1,1"])
        with open(os.path.join(run, "linear.json")) as f:
            out = json.load(f)
        assert out["run"] == name and out["optimizer"] == "adamw" and out["layers"] == layers
        assert out["step"] > 0 and out["step"] == sorted(int(c[-10:-3]) for c in glob.glob(run + "/ckpt-*.pt"))[-1]
        assert out["rules"] == ["plain", "white", "linear"] and out["lams"] == [0.1, 1.0]
        assert out["shots"] == [1, 5, 20] and out["episodes"] == 20 and out["query"] == 5 and out["seed"] == 0
        assert set(out["pools"]["seen_4way"]) == {"n_way", "images", "classes"}
        assert set(out["results"]) == set(layers)
        for layer in layers:
            for pool, by_k in out["results"][layer].items():
                assert pool in out["pools"]
                assert "20" not in by_k and "1" in by_k                             # 20 shots + queries do not fit
                for cell in by_k.values():
                    assert set(cell) == {"plain", "white", "linear"} and list(cell["linear"]) == ["0.1", "1.0"]
                    for c in (cell["plain"], cell["white"], *cell["linear"].values()):
                        assert set(c) == {"acc", "ci95"} and 0.0 <= c["acc"] <= 1.0
        # the plain cell is fewshot.py's own number for the last checkpoint
        fewshot.main([*common, "--shots", "1,5"])
        with open(os.path.join(run, "fewshot.json")) as f:
            fs = json.load(f)["checkpoints"][-1]
        assert fs["step"] == out["step"]
        want = fs["results"]["gap"]["seen_4way"]["5"]["acc"]
        assert out["results"]["gap"]["seen_4way"]["5"]["plain"]["acc"] == pytest.approx(want, abs=1e-6)
