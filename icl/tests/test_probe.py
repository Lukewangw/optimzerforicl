"""probe.py and cka.py on features whose answer is known, and end to end on a tiny run."""
import glob
import json
import os

import numpy as np
import pytest
import torch

from icl import cka, fewshot, probe, train
from icl.vit import ViT

MODEL = ["--width", "32", "--depth", "2", "--mlp-dim", "64", "--heads", "2", "--patch", "8", "--image-size", "32"]


def clusters(n_classes=6, per_class=40, dim=12, spread=0.8, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(n_classes, dim))
    labels = np.repeat(np.arange(n_classes), per_class)
    return (means[labels] + spread * rng.normal(size=(len(labels), dim)) + 5.0).astype(np.float32), labels


def test_layers_are_the_models_own_forward_pass_read_at_every_stage():
    kw = dict(image_size=32, patch=8, width=32, depth=3, mlp_dim=64, heads=2, init_seed=0)
    model = ViT(5, None, **kw).eval()
    torch.nn.init.normal_(model.head.weight, std=0.1, generator=torch.Generator().manual_seed(1))  # zero at init
    x = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        got, logits = probe.layers(model, x)
        gap, pre = model.both_features(x)
        assert torch.equal(logits, model(x)) and logits.abs().max() > 0
    assert list(got) == probe.layer_names(model) == ["embed", "block1", "block2", "block3", "gap", "pre_act",
                                                     "pre_logits"]
    assert torch.equal(got["gap"], gap) and torch.equal(got["pre_logits"], pre)
    assert torch.allclose(torch.tanh(got["pre_act"]), pre) and not torch.allclose(got["pre_act"], pre)
    assert all(v.shape == (4, 32) for v in got.values())
    assert not torch.allclose(got["block3"], got["gap"])                    # the final LayerNorm lies between them
    headless = ViT(5, None, rep_size=False, **kw).eval()                    # no extra layer: all three are `gap`
    with torch.no_grad():
        got, _ = probe.layers(headless, x)
    assert torch.equal(got["pre_act"], got["gap"]) and torch.equal(got["pre_logits"], got["gap"])
    feats, logits = probe.extract(model, torch.randint(0, 256, (7, 32, 32, 3), dtype=torch.uint8), "cpu",
                                  batch_size=3)
    assert all(v.shape == (7, 32) and v.dtype == torch.float32 for v in feats.values()) and logits.shape == (7, 5)


def test_whitening_undoes_an_invertible_linear_map_and_the_plain_rule_does_not():
    feats, labels = clusters()
    rng = np.random.default_rng(1)
    a = np.linalg.qr(rng.normal(size=(12, 12)))[0] * np.linspace(1.0, 8.0, 12)        # stretches some directions
    f, g = torch.from_numpy(feats), torch.from_numpy((feats @ a - 2.0).astype(np.float32))
    wf, wg = probe.whitened(f, f, ridge=0.0), probe.whitened(g, g, ridge=0.0)
    np.testing.assert_allclose(wf.norm(dim=1).numpy(), 1.0, atol=1e-5)
    assert torch.allclose(wf @ wf.T, wg @ wg.T, atol=1e-3)                  # the same rows up to a rotation
    pf, pg = probe.plain(f, f), probe.plain(g, g)
    assert not torch.allclose(pf @ pf.T, pg @ pg.T, atol=0.05)
    np.testing.assert_array_equal(pf.numpy(), fewshot.cl2n(feats, feats.mean(0)))     # fewshot.py's own numbers

    def acc(p):
        return fewshot.evaluate_pool(p, labels, "pool", n_way=6, shots=[5], n_query=15, n_episodes=200, seed=0)["5"]["acc"]
    assert acc(wf) == acc(wg) and acc(pf) != acc(pg)
    # before the scaling to unit length, the reference set has the identity as its covariance
    ref = torch.from_numpy(rng.normal(size=(4000, 5)) * np.array([1.0, 2.0, 3.0, 4.0, 50.0]))
    w = probe.whitened(ref, ref, ridge=0.0)
    assert torch.allclose(torch.cov(w.double().T) * 5, torch.eye(5, dtype=torch.float64), atol=0.1)
    # the ridge keeps a direction the reference set does not use from being blown up
    flat = torch.cat([ref[:, :4], torch.zeros(4000, 1, dtype=ref.dtype)], dim=1)
    probe_rows = torch.cat([ref[:10, :4], torch.ones(10, 1, dtype=ref.dtype) * 1e-3], dim=1)
    assert probe.whitened(probe_rows, flat)[:, 4].abs().max() < 0.2
    assert probe.whitened(probe_rows, flat, ridge=0.0)[:, 4].abs().min() > 0.99


def test_spectrum_counts_the_directions_in_use():
    rng = np.random.default_rng(0)
    iso = torch.from_numpy(rng.normal(size=(20000, 16)))
    s = probe.spectrum(iso)
    assert 15.5 < s["pr"] <= 16 and 15.5 < s["erank"] <= 16 and abs(s["top"][0] - 1 / 16) < 0.01
    assert len(s["top"]) == len(probe.SHARES) and s["top"][-1] == pytest.approx(1.0)
    assert probe.spectrum(iso * 3.0)["trace"] == pytest.approx(9 * s["trace"])
    line = torch.from_numpy(np.outer(rng.normal(size=500), rng.normal(size=16)) + 7.0)   # one direction and an offset
    s = probe.spectrum(line)
    assert s["pr"] == pytest.approx(1.0) and s["erank"] == pytest.approx(1.0, abs=1e-3) and s["top"][0] > 0.999999
    two = iso.clone()
    two[:, 2:] = 0.0
    assert probe.spectrum(two)["pr"] == pytest.approx(2.0, abs=0.01)
    uneven = iso * torch.tensor([10.0] + [1.0] * 15, dtype=torch.float64)            # one direction holds most
    s = probe.spectrum(uneven)
    assert 1.0 < s["pr"] < s["erank"] < 8 and s["top"][0] == pytest.approx(100 / 115, abs=0.02)
    assert probe.spectrum(torch.ones(10, 4)) == {"trace": 0.0, "pr": 0.0, "erank": 0.0, "top": [0.0] * 8}


def test_between_share_is_one_for_points_and_chance_for_noise():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(8), 50)
    means = rng.normal(size=(8, 10))
    assert probe.between_share(torch.from_numpy(means[labels]), labels) == pytest.approx(1.0)
    noise = torch.from_numpy(rng.normal(size=(400, 10)))
    assert abs(probe.between_share(noise, labels) - 7 / 399) < 0.01
    mixed = torch.from_numpy(means[labels] + rng.normal(size=(400, 10)))
    share = probe.between_share(mixed, labels)
    assert 0.3 < share < 0.7
    assert probe.between_share(mixed, labels[::-1] * 7 + 3) == pytest.approx(probe.between_share(mixed, labels[::-1]))
    assert probe.between_share(mixed + 100.0, labels) == pytest.approx(share)          # the common mean is removed
    assert probe.between_share(noise, np.zeros(400, dtype=np.int64)) is None           # one class: nothing to say
    assert probe.between_share(torch.zeros(400, 10), labels) is None


def test_saturation_reads_the_flat_ends_of_the_tanh():
    z = torch.tensor([[0.0, 1.0], [2.0, -3.0]])                  # tanh: 0, 0.7616, 0.9640, -0.9951
    s = probe.saturation(z)
    assert s["over90"] == 0.5 and s["over99"] == 0.25
    assert s["mean_abs"] == pytest.approx((0.761594 + 0.964028 + 0.995055) / 4, abs=1e-6)
    assert s["rms_in"] == pytest.approx((14 / 4) ** 0.5)
    assert probe.saturation(-z) == s


def test_class_hits_count_top1_top5_and_each_class():
    logits = torch.full((6, 8), -5.0)
    labels = np.array([0, 0, 1, 2, 2, 2])
    for i, top in enumerate([0, 3, 1, 2, 2, 7]):                 # the first guesses: four are right
        logits[i, top] = 5.0
    logits[1, 0] = 4.0                                           # image 1: its class is the second guess
    logits[5, [0, 1, 3, 4]] = torch.tensor([4.0, 3.0, 2.0, 1.0])  # image 5: its class is not among the first five
    r = probe.class_hits(logits, labels, 8)
    assert r["images"] == 6 and r["top1"] == pytest.approx(4 / 6) and r["top5"] == pytest.approx(5 / 6)
    assert r["hits"] == [1, 1, 2, 0, 0, 0, 0, 0]
    assert r["loss"] == pytest.approx(float(torch.nn.functional.cross_entropy(logits.double(), torch.from_numpy(labels))))
    assert probe.class_hits(logits[:, :3], labels, 3)["top5"] == 1.0          # fewer than five classes


def test_matrix_stats_are_those_of_the_singular_values():
    assert probe.matrix_stats(torch.eye(6) * 2.0) == pytest.approx({"fro": 24 ** 0.5, "spec": 2.0, "srank": 6.0,
                                                                    "erank": 6.0})
    one = probe.matrix_stats(torch.outer(torch.arange(1.0, 5.0), torch.arange(1.0, 4.0)))
    assert one["srank"] == pytest.approx(1.0) and one["erank"] == pytest.approx(1.0, abs=1e-6)
    assert one["fro"] == pytest.approx(one["spec"]) == pytest.approx((30 * 14) ** 0.5)
    assert probe.matrix_stats(torch.diag(torch.tensor([3.0, 1.0, 1.0])))["srank"] == pytest.approx(11 / 9)
    w = torch.randn(8, 5, generator=torch.Generator().manual_seed(0))
    assert probe.matrix_stats(w)["fro"] == pytest.approx(float(w.norm())) and 1 < probe.matrix_stats(w)["srank"] < 5
    assert probe.matrix_stats(torch.zeros(3, 4)) == {"fro": 0.0, "spec": 0.0, "srank": 0.0, "erank": 0.0}


def test_weight_stats_follow_each_matrix_and_its_changes():
    state = {"embedding.weight": torch.randn(8, 3, 4, 4), "embedding.bias": torch.zeros(8),
             "blocks.0.attn.q.weight": torch.eye(8), "blocks.0.norm1.weight": torch.ones(8),
             "pre_logits.weight": torch.eye(8), "head.weight": torch.zeros(3, 8), "head.bias": torch.zeros(3)}
    m = probe.matrices(state)
    assert set(m) == {"embedding.weight", "blocks.0.attn.q.weight", "pre_logits.weight", "head.weight"}
    assert m["embedding.weight"].shape == (8, 48)
    w0, w1 = {"w": torch.eye(4)}, {"w": torch.eye(4) * 2.0}
    w2 = {"w": w1["w"] + 3.0 * torch.outer(torch.eye(4)[0], torch.eye(4)[1])}
    assert set(probe.weight_stats(w0, None, None, "cpu")["w"]) == {"fro", "spec", "srank", "erank"}
    got = probe.weight_stats(w2, w1, w0, "cpu")["w"]
    assert got["since_last"] == pytest.approx({"fro": 3.0, "spec": 3.0, "srank": 1.0, "erank": 1.0})
    assert got["since_first"]["fro"] == pytest.approx((4 + 9) ** 0.5) and got["fro"] == pytest.approx((16 + 9) ** 0.5)
    assert probe.weight_stats(w1, w0, w0, "cpu")["w"]["since_last"]["srank"] == pytest.approx(4.0)


def test_sets_are_the_distinct_image_slices_and_kept_rows_are_spread_over_them():
    y = np.zeros(1, dtype=np.int64)
    pools = {"dog": (slice(0, 10), y, 1), "dogs_5way": (slice(10, 110), y, 5), "dogs_118way": (slice(10, 110), y, 118),
             "inet_5way": (slice(110, 1110), y, 5), "seen_1way": (slice(1110, 1200), y, 1),
             "seen_20way": (slice(1110, 1200), y, 20)}
    sets = probe.distinct_sets(pools)
    assert list(sets) == ["dog", "dogs", "inet", "seen"] and sets["dogs"][0] == slice(10, 110)
    rows = probe.kept_rows(sets, ["seen", "inet", "absent"], 50)
    assert len(rows) == len(set(rows.tolist())) == 100
    assert (rows[0], rows[49], rows[50], rows[99]) == (1110, 1199, 110, 1109)
    assert probe.kept_rows(sets, ["dog"], 50).tolist() == list(range(10))
    assert len(probe.kept_rows(sets, ["absent"], 50)) == 0


def test_linear_cka_is_one_under_rotation_and_scale_and_low_for_unrelated_features():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 12)) * np.linspace(1.0, 3.0, 12)
    q = np.linalg.qr(rng.normal(size=(12, 12)))[0]
    assert cka.linear_cka(x, 5.0 * x @ q + 2.0) == pytest.approx(1.0)
    assert cka.linear_cka(x, rng.normal(size=(2000, 12))) < 0.05
    half = x[:, :6] + 0.5 * rng.normal(size=(2000, 6))
    assert 0.1 < cka.linear_cka(x, half) < 0.95 and cka.linear_cka(x, half) == pytest.approx(cka.linear_cka(half, x))
    assert np.isnan(cka.linear_cka(x, np.ones((2000, 3))))
    m = cka.cka_matrix([x, half, x @ q])
    assert m.shape == (3, 3) and np.allclose(m, m.T) and np.allclose(np.diag(m), 1.0)
    assert m[0, 2] == pytest.approx(1.0) and m[0, 1] == pytest.approx(cka.linear_cka(x, half))


@pytest.fixture(scope="module")
def tiny_run(pack_dir, roles_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("probe") / "run"
    train.main(["--train-pack", pack_dir, "--val-pack", pack_dir, "--roles", roles_file, "--out", str(out),
                "--optimizer", "adamw", *MODEL, "--batch-size", "16", "--epochs", "10", "--warmup-steps", "5",
                "--randaug-layers", "0", "--log-every", "1", "--device", "cpu", "--workers", "0", "--lr", "3e-3"])
    return str(out)


def test_probe_of_a_run_repeats_its_fewshot_and_validation_numbers(tiny_run, pack_dir, roles_file, tmp_path):
    common = ["--run", tiny_run, "--roles", roles_file, "--heldout-pack", pack_dir, "--base-pack", pack_dir,
              "--extra", f"again={pack_dir}", "--extra", f"some={pack_dir}:0-3:2,0", "--episodes", "40",
              "--query", "5", "--per-class", "20", "--seen-per-class", "20", "--workers", "0", "--device", "cpu"]
    fewshot.main([*common, "--shots", "1,5"])
    probe.main([*common, "--train-pack", pack_dir, "--train-per-class", "10", "--keep", str(tmp_path / "kept"),
                "--keep-sets", "seen,again", "--keep-per-set", "30"])
    with open(os.path.join(tiny_run, "fewshot.json")) as f:
        fs = json.load(f)
    with open(os.path.join(tiny_run, "probe.json")) as f:
        pr = json.load(f)
    names = ["embed", "block1", "block2", "gap", "pre_act", "pre_logits"]
    assert pr["layers"] == names and pr["rules"] == ["plain", "white"] and pr["shots"] == [5] and pr["ridge"] == 1e-3
    assert pr["pools"] == fs["pools"] and list(pr["sets"]) == ["held", "one", "again", "some", "seen"]
    assert pr["sets"]["seen"] == {"images": 80, "classes": 4} and pr["sets"]["one"] == {"images": 20, "classes": 1}
    assert pr["scored"]["val"]["per_class"] == [27, 24, 24, 24] and pr["scored"]["train"]["per_class"] == [10] * 4
    assert pr["train_images_per_class"] == [27, 24, 24, 24]
    assert [c["step"] for c in pr["checkpoints"]] == [c["step"] for c in fs["checkpoints"]] == train.checkpoint_steps(62)
    moved = []                                                  # validation images scored otherwise than in the log
    for got, want in zip(pr["checkpoints"], fs["checkpoints"]):
        for layer in ("gap", "pre_logits"):                      # the plain rule is fewshot.py's test
            for pool, row in want["results"][layer].items():
                mine = got["fewshot"]["plain"][layer][pool]["5"]
                assert (mine is None) == (row["5"] is None)
                if mine is not None:
                    assert mine["acc"] == pytest.approx(row["5"]["acc"], abs=1e-9)
                    assert mine["ci95"] == pytest.approx(row["5"]["ci95"], abs=1e-9)
        assert got["val_top1"] == pytest.approx(want["val_top1"])              # the logged value, carried over
        moved.append(round(abs(got["val"]["top1"] - got["val_top1"]) * got["val"]["images"]))
        assert sum(got["val"]["hits"]) == round(got["val"]["top1"] * 99) and got["val"]["images"] == 99
        assert got["train"]["images"] == 40 and 0 <= got["train"]["top1"] <= got["train"]["top5"] <= 1
        assert set(got["fewshot"]) == {"plain", "white"} and list(got["fewshot"]["white"]) == names
        assert list(got["shape"]) == names and set(got["tanh"]) == set(pr["sets"])
        for layer in names:
            shape = got["shape"][layer]
            assert shape["one"]["between"] is None and shape["one"]["between_white"] is None
            assert 0 < shape["seen"]["between"] <= 1 and 0 < shape["seen"]["between_white"] <= 1
            assert 1 <= shape["seen"]["pr"] <= shape["seen"]["erank"] + 1e-9 <= 32 + 1e-6 and len(shape["seen"]["top"]) == 8
        assert len(got["sv"]["pre_logits.weight"]) == 32 and len(got["sv"]["head.weight"]) == 4
        assert {"embedding.weight", "blocks.0.attn.q.weight", "blocks.1.mlp.fc2.weight", "pre_logits.weight",
                "head.weight"} <= set(got["weights"])
    # Training scores with argmax, which gives a tie to the first class; the probe with topk, whose order among equal
    # scores is not specified (on the cluster's processors it differs from the Mac's).  The head starts at zero and
    # the first step's learning rate is zero, so at the first two checkpoints every score is tied (the log's 27 of 99
    # images are class 0's); from the third checkpoint on the two agree image for image.
    assert [c["val_top1"] for c in pr["checkpoints"][:2]] == pytest.approx([27 / 99] * 2)
    assert moved[2:] == [0] * (len(moved) - 2)
    first, last = pr["checkpoints"][0], pr["checkpoints"][-1]
    assert "since_last" not in first["weights"]["head.weight"] and first["weights"]["head.weight"]["fro"] == 0.0
    assert last["weights"]["head.weight"]["since_first"]["fro"] == pytest.approx(last["weights"]["head.weight"]["fro"])
    assert last["weights"]["pre_logits.weight"]["since_last"]["fro"] > 0
    assert last["val"]["top1"] > 0.9 and last["val"]["loss"] < first["val"]["loss"]
    assert last["shape"]["gap"]["seen"]["between"] > first["shape"]["gap"]["seen"]["between"] - 0.2
    assert last["fewshot"]["plain"]["gap"]["seen_4way"]["5"]["acc"] > 0.9
    # the features kept for cka.py: the last checkpoint read, evenly spaced rows of the sets asked for
    kept = glob.glob(str(tmp_path / "kept" / "*.npz"))
    assert [os.path.basename(k) for k in kept] == [f"run-{last['step']}.npz"]
    with np.load(kept[0]) as d:
        assert [str(n) for n in d["layers"]] == names and len(d["rows"]) == 60 and d["gap"].shape == (60, 32)
        assert int(d["step"]) == last["step"]
    probe.main([*common, "--steps", f"0,{pr['checkpoints'][3]['step']}", "--keep", str(tmp_path / "kept"),
                "--keep-sets", "seen,again", "--keep-per-set", "30", "--out", str(tmp_path / "two.json"),
                "--ridge", "0.5"])
    with open(tmp_path / "two.json") as f:
        two = json.load(f)
    assert [c["step"] for c in two["checkpoints"]] == [0, pr["checkpoints"][3]["step"]] and "val" in two["checkpoints"][0]
    assert "train" not in two["checkpoints"][0] and "train_images_per_class" not in two and two["ridge"] == 0.5
    mine, theirs = two["checkpoints"][1], pr["checkpoints"][3]           # the ridge is the whitened rule's alone
    assert mine["fewshot"]["plain"] == theirs["fewshot"]["plain"] and mine["fewshot"]["white"] != theirs["fewshot"]["white"]
    assert all(mine["shape"][n]["seen"]["between"] == pytest.approx(theirs["shape"][n]["seen"]["between"], rel=1e-4)
               for n in names)
    # the change since the first checkpoint read is the same whichever checkpoints lie between
    assert (two["checkpoints"][1]["weights"]["pre_logits.weight"]["since_first"]
            == pytest.approx(pr["checkpoints"][3]["weights"]["pre_logits.weight"]["since_first"]))
    cka.main(["--features", str(tmp_path / "kept"), "--out", str(tmp_path / "cka.json")])       # a folder of them
    with open(tmp_path / "cka.json") as f:
        c = json.load(f)
    assert len(c["files"]) == 2 and c["layers"] == names and c["images"] == 60
    for layer in names:
        m = np.array(c["between"][layer])
        assert m.shape == (2, 2) and m[0, 0] == m[1, 1] == 1.0 and m[0, 1] == m[1, 0] and 0 < m[0, 1] <= 1 + 1e-9
    assert c["between"]["embed"][0][1] > 0.5                    # the patch embedding has hardly moved
    for name in c["files"]:
        m = np.array(c["within"][name])
        assert m.shape == (6, 6) and np.allclose(m, m.T) and np.allclose(np.diag(m), 1.0)
