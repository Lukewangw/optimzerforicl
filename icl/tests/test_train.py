"""Tiny runs on CPU: pairing of the two optimizers, learning, resuming, and the steps after training."""
import glob
import json
import math
import os
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch

from icl import curves, features, fewshot, train
from icl.data import Pack, Roles

MODEL = ["--width", "32", "--depth", "2", "--mlp-dim", "64", "--heads", "2", "--patch", "8", "--image-size", "32"]


def run(pack_dir, roles_file, out, optimizer, *extra, epochs="10"):
    train.main(["--train-pack", pack_dir, "--val-pack", pack_dir, "--roles", roles_file, "--out", str(out),
                "--optimizer", optimizer, *MODEL, "--batch-size", "16", "--epochs", epochs, "--warmup-steps", "5",
                "--randaug-layers", "0", "--log-every", "1", "--device", "cpu", "--workers", "0", "--lr", "3e-3",
                *extra])
    with open(os.path.join(out, "log.jsonl")) as f:
        return [json.loads(line) for line in f]


def losses(recs):
    return {r["step"]: r["last_loss"] for r in recs if r["event"] == "train"}


@pytest.fixture(scope="module")
def pair(pack_dir, roles_file, tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    return {name: (str(root / name), run(pack_dir, roles_file, root / name, name)) for name in ("adamw", "muon")}


def test_a_pair_shares_init_and_batches_and_both_learn(pair):
    (dir_a, a), (dir_m, m) = pair["adamw"], pair["muon"]
    info = {}
    for name, d in (("adamw", dir_a), ("muon", dir_m)):
        with open(os.path.join(d, "run.json")) as f:
            info[name] = json.load(f)
    assert info["adamw"]["init_sha"] == info["muon"]["init_sha"] == a[0]["init_sha"] == m[0]["init_sha"]
    assert (info["adamw"]["num_classes"], info["adamw"]["n_train"], info["adamw"]["total_steps"]) == (4, 99, 62)
    sha = lambda recs: [r["sha"] for r in recs if r["event"] == "batch"]
    assert len(sha(a)) == 3 and sha(a) == sha(m) and len(set(sha(a))) == 3
    assert [g["optimizer"] for g in info["muon"]["optimizer_groups"]] == ["Muon", "AdamW", "AdamW"]
    assert [g["optimizer"] for g in info["adamw"]["optimizer_groups"]] == ["AdamW", "AdamW"]
    for recs in (a, m):
        ls = losses(recs)
        assert sorted(ls) == list(range(1, 63))
        assert abs(ls[1] - math.log(4)) < 1e-4                                   # zero head, and no update at lr 0
        assert np.mean([ls[s] for s in range(53, 63)]) < 0.6 * np.mean([ls[s] for s in range(1, 6)])
        assert recs[-1]["event"] == "done" and recs[-1]["step"] == 62
    assert losses(a) != losses(m)                                                 # same data, different updates


def test_checkpoints_are_written_at_the_fixed_fractions(pair):
    d, recs = pair["muon"]
    want = train.checkpoint_steps(62)
    assert want[0] == 0 and want[-1] == 62 and 12 <= len(want) <= 17
    ck = [r for r in recs if r["event"] == "ckpt"]
    assert [r["step"] for r in ck] == want
    # with the zero head every logit is 0 and the first class wins: 27 of the 99 base-class images are class 0
    assert ck[0]["val_top1"] == pytest.approx(27 / 99) and ck[-1]["val_top1"] > 0.9
    for r in ck:
        state = torch.load(os.path.join(d, r["file"]), map_location="cpu", weights_only=True)
        assert state["step"] == r["step"] and state["val_top1"] == r["val_top1"] and state["optimizer"] == "muon"
    assert train.checkpoint_steps(112_000)[:4] == [0, 560, 1120, 2240] and len(train.checkpoint_steps(112_000)) == 17
    model, state = features.load_model(os.path.join(d, ck[-1]["file"]), "cpu")
    assert state["model_kwargs"]["num_classes"] == 4 and model.width == 32


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_resume_reproduces_the_uninterrupted_run(pair, pack_dir, roles_file, tmp_path, optimizer):
    whole = losses(pair[optimizer][1])
    out = tmp_path / "resumed"
    first = run(pack_dir, roles_file, out, optimizer, "--max-steps", "23")
    assert first[-1]["event"] == "stopped" and first[-1]["step"] == 23
    again = run(pack_dir, roles_file, out, optimizer)
    assert [r["event"] for r in again].count("resume") == 1
    resumed = losses(again)
    assert sorted(resumed) == list(range(1, 63))
    assert all(resumed[s] == whole[s] for s in range(1, 63)), "a resumed run must repeat the same losses exactly"
    a = torch.load(os.path.join(pair[optimizer][0], "ckpt-0000062.pt"), weights_only=True)["model"]
    b = torch.load(os.path.join(out, "ckpt-0000062.pt"), weights_only=True)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    # a run directory refuses another run's settings
    with pytest.raises(SystemExit):
        run(pack_dir, roles_file, out, optimizer, "--init-seed", "1")


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_two_processes_agree_with_each_other_and_with_one(pack_dir, roles_file, tmp_path, optimizer):
    """`torchrun` with two CPU processes (gloo).  Skipped where the launcher cannot open its local port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = ["--train-pack", pack_dir, "--val-pack", pack_dir, "--roles", roles_file, "--optimizer", optimizer,
            *MODEL, "--batch-size", "16", "--epochs", "3", "--warmup-steps", "5", "--randaug-layers", "0",
            "--mixup", "0", "--log-every", "1", "--device", "cpu", "--workers", "0", "--lr", "3e-3"]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")])}
    try:
        r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master-addr",
                            "127.0.0.1", "--master-port", str(port), "-m", "icl.train", *args, "--out",
                            str(tmp_path / "two")], capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        pytest.skip("the two-process launcher did not come up on this machine")
    if r.returncode != 0 and any(w in r.stderr for w in ("Address", "address", "Permission", "connect")):
        pytest.skip("the two-process launcher could not use a local port: " + r.stderr[-200:])
    assert r.returncode == 0, r.stderr[-2000:]
    train.main([*args, "--out", str(tmp_path / "one")])
    with open(tmp_path / "two" / "log.jsonl") as f:
        two = [json.loads(line) for line in f]
    assert two[0]["world"] == 2 and two[-1]["event"] == "done"
    assert all(r["replica_spread"] == 0.0 for r in two if r["event"] == "ckpt")     # the two replicas agree
    assert sum(r["event"] == "train" for r in two) == two[-1]["step"]              # only one process logs
    a = torch.load(tmp_path / "two" / "last.pt", weights_only=True)["model"]
    b = torch.load(tmp_path / "one" / "last.pt", weights_only=True)["model"]
    # same global batches, gradients averaged over two halves: the same training up to rounding.  Muon's
    # orthogonalisation runs in bf16, which turns rounding differences into ones of about 1e-3.
    tol = 2e-4 if optimizer == "adamw" else 2e-2
    assert max(float((a[k] - b[k]).abs().max()) for k in a) < tol


def test_other_seeds_change_what_they_should(pair, pack_dir, roles_file, tmp_path):
    base = pair["adamw"][1]
    other_init = run(pack_dir, roles_file, tmp_path / "i", "adamw", "--init-seed", "1", "--max-steps", "4")
    other_data = run(pack_dir, roles_file, tmp_path / "d", "adamw", "--data-seed", "1", "--max-steps", "4")
    sha = lambda recs: [r["sha"] for r in recs if r["event"] == "batch"]
    assert other_init[0]["init_sha"] != base[0]["init_sha"] and sha(other_init) == sha(base)
    assert other_data[0]["init_sha"] == base[0]["init_sha"] and sha(other_data) != sha(base)


def test_muon_blocks_is_for_a_muon_run_and_names_the_matrices_muon_takes(pack_dir, roles_file, tmp_path):
    with pytest.raises(SystemExit):                            # an AdamW run has no Muon to give matrices to
        run(pack_dir, roles_file, tmp_path / "a", "adamw", "--muon-blocks", "mlp", "--max-steps", "1")
    assert not (tmp_path / "a").exists()
    run(pack_dir, roles_file, tmp_path / "m", "muon", "--muon-blocks", "attn", "--muon-skip-pre-logits",
        "--max-steps", "1")
    with open(tmp_path / "m" / "run.json") as f:
        a = json.load(f)["args"]
    assert (a["muon_blocks"], a["muon_skip_pre_logits"]) == ("attn", True)
    assert curves.variant_of(a) == "on the attention only"
    for bad in ("--muon-skip-pre-logits", "--no-pre-logits"):   # "none" leaves Muon one matrix: it has to exist
        with pytest.raises(SystemExit):
            run(pack_dir, roles_file, tmp_path / "n", "muon", "--muon-blocks", "none", bad, "--max-steps", "1")
        assert not (tmp_path / "n").exists()
    for name, more, variant, lr in (("p", (), "on pre_logits only", 0.003),            # run() sets --lr 3e-3
                                    ("q", ("--muon-lr", "0.01"), "on pre_logits only at 0.01", 0.01)):
        run(pack_dir, roles_file, tmp_path / name, "muon", "--muon-blocks", "none", *more, "--max-steps", "1")
        with open(tmp_path / name / "run.json") as f:
            info = json.load(f)
        assert curves.variant_of(info["args"]) == variant
        muon = [g for g in info["optimizer_groups"] if g["optimizer"] == "Muon"]
        assert [(g["tensors"], g["lr"]) for g in muon] == [(1, lr)]   # the pre_logits matrix alone, at Muon's rate


def test_start_from_goes_on_from_another_runs_checkpoint_with_its_own_optimizer(pair, pack_dir, roles_file, tmp_path):
    src = os.path.join(pair["adamw"][0], "ckpt-0000006.pt")
    out = tmp_path / "switched"
    recs = run(pack_dir, roles_file, out, "muon", "--start-from", src, "--start-ramp", "4")
    assert recs[0]["event"] == "start" and recs[0]["step"] == 6 and recs[-1]["event"] == "done"
    assert recs[-1]["steps_this_process"] == 62 - 6
    # its first checkpoint is the one it started from; the later ones are at the usual steps and are its own
    steps = [r["step"] for r in recs if r["event"] == "ckpt"]
    assert steps == [s for s in train.checkpoint_steps(62) if s >= 6]
    a = torch.load(src, weights_only=True)
    b = torch.load(out / "ckpt-0000006.pt", weights_only=True)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]) and b["val_top1"] == a["val_top1"]
    end = torch.load(out / "ckpt-0000062.pt", weights_only=True)["model"]
    for other in ("adamw", "muon"):
        ref = torch.load(os.path.join(pair[other][0], "ckpt-0000062.pt"), weights_only=True)["model"]
        assert any(not torch.equal(end[k], ref[k]) for k in end)
    # the batches go on where the first run's were: the first loss is that run's loss at the same step under the
    # same weights (the update that follows differs), and the learning rate climbs over the ramp
    theirs, mine = losses(pair["adamw"][1]), losses(recs)
    assert mine[7] == theirs[7] and mine[8] != theirs[8]
    scale = {r["step"]: r["lr_scale"] for r in recs if r["event"] == "train"}
    plain = {r["step"]: r["lr_scale"] for r in pair["muon"][1] if r["event"] == "train"}
    assert [round(scale[s] / plain[s], 6) for s in (7, 8, 9, 10, 11)] == [0.25, 0.5, 0.75, 1.0, 1.0]
    with open(out / "run.json") as f:
        info = json.load(f)
    assert info["started_at"] == 6 and curves.variant_of(info["args"]) == f"from adamw at step 6"
    assert curves.load_run(str(out))["setting"] == "muon lr 0.003, from adamw at step 6"
    # stopped and submitted again, it resumes from its own state and ends with the same weights
    again = tmp_path / "switched-twice"
    run(pack_dir, roles_file, again, "muon", "--start-from", src, "--start-ramp", "4", "--max-steps", "23")
    run(pack_dir, roles_file, again, "muon", "--start-from", src, "--start-ramp", "4")
    twice = torch.load(again / "ckpt-0000062.pt", weights_only=True)["model"]
    assert all(torch.equal(end[k], twice[k]) for k in end)
    # a checkpoint of a run with another seed or length is refused, and so is the start or the end of a run
    for bad in (["--init-seed", "1"], ["--epochs", "5"]):
        with pytest.raises(SystemExit):
            train.main(["--train-pack", pack_dir, "--roles", roles_file, "--out", str(tmp_path / "bad"),
                        "--optimizer", "muon", *MODEL, "--batch-size", "16", "--epochs", "10", "--device", "cpu",
                        "--workers", "0", "--start-from", src, *bad])
    for edge in ("ckpt-0000000.pt", "ckpt-0000062.pt"):
        with pytest.raises(SystemExit):
            run(pack_dir, roles_file, tmp_path / "edge", "muon", "--start-from", os.path.join(pair["adamw"][0], edge))


def test_a_run_directory_is_one_run_and_a_lost_checkpoint_is_written_again(pack_dir, roles_file, tmp_path):
    out = tmp_path / "r"
    run(pack_dir, roles_file, out, "adamw", "--max-steps", "4")
    assert sorted(n for n in os.listdir(out) if n.startswith("ckpt-")) == [f"ckpt-{s:07d}.pt" for s in (0, 1, 2, 4)]
    # a process killed after saving its state at step 4 and before writing that checkpoint: the next one writes it
    os.remove(out / "ckpt-0000004.pt")
    recs = run(pack_dir, roles_file, out, "adamw", "--max-steps", "4")
    assert [r["step"] for r in recs if r["event"] == "ckpt"] == [0, 1, 2, 4, 4] and recs[-1]["steps_this_process"] == 0
    a = torch.load(out / "ckpt-0000004.pt", weights_only=True)["model"]
    b = torch.load(out / "last.pt", weights_only=True)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    # checkpoints without the state they belong to are somebody else's: no silent mixing of two runs
    os.remove(out / "last.pt")
    with pytest.raises(SystemExit):
        run(pack_dir, roles_file, out, "adamw", "--max-steps", "4")


def test_curves_drop_what_a_stopped_process_logged_after_its_last_save(tmp_path):
    with open(tmp_path / "run.json", "w") as f:
        json.dump({"args": {"optimizer": "adamw", "lr": 1e-3, "init_seed": 0, "data_seed": 0}}, f)
    recs = [{"event": "start", "step": 0}, {"event": "ckpt", "step": 0, "val_top1": 0.1}]
    recs += [{"event": "train", "step": s, "loss": 9.0} for s in (1, 2, 3, 4, 5)]       # saved at 3, killed after 5
    recs += [{"event": "ckpt", "step": 3, "val_top1": 0.2}, {"event": "resume", "step": 3}]
    recs += [{"event": "train", "step": s, "loss": 1.0} for s in (4, 5, 6)] + [{"event": "ckpt", "step": 6, "val_top1": 0.3}]
    with open(tmp_path / "log.jsonl", "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))
    got = curves.load_run(str(tmp_path))
    assert [(r["step"], r["loss"]) for r in got["train"]] == [(1, 9.0), (2, 9.0), (3, 9.0), (4, 1.0), (5, 1.0), (6, 1.0)]
    assert [r["step"] for r in got["ckpt"]] == [0, 3, 6]


def test_fewshot_and_curves_on_the_runs(pair, pack_dir, roles_file, tmp_path):
    for name in ("adamw", "muon"):
        fewshot.main(["--run", pair[name][0], "--roles", roles_file, "--heldout-pack", pack_dir, "--base-pack",
                      pack_dir, "--extra", f"again={pack_dir}", "--extra", f"some={pack_dir}:0-3:2,0",
                      "--shots", "1,5", "--episodes", "40", "--query", "5",
                      "--per-class", "20", "--seen-per-class", "20", "--workers", "0", "--device", "cpu"])
    with open(os.path.join(pair["muon"][0], "fewshot.json")) as f:
        out = json.load(f)
    assert out["pools"] == {"held": {"images": 40, "n_way": 2}, "one": {"images": 20, "n_way": 1},
                            "seen_1way": {"images": 80, "n_way": 1}, "seen_2way": {"images": 80, "n_way": 2},
                            "seen_4way": {"images": 80, "n_way": 4},        # 6-way: there are only four seen classes
                            "again": {"images": 120, "n_way": 6},
                            "some_2way": {"images": 80, "n_way": 2}, "some_4way": {"images": 80, "n_way": 4}}
    assert [c["step"] for c in out["checkpoints"]] == train.checkpoint_steps(62)
    last = out["checkpoints"][-1]
    assert last["val_top1"] > 0.9 and set(last["results"]) == {"gap", "pre_logits"}
    for kind in ("gap", "pre_logits"):
        r = last["results"][kind]
        assert r["held"]["5"]["episodes"] == 40 and r["held"]["5"]["n_query"] == 5
        assert r["held"]["5"]["acc"] > 0.9            # two unseen colours, told apart from five examples each
        assert r["seen_2way"]["5"]["acc"] > 0.9 and r["again"]["5"]["acc"] > 0.8
        assert r["some_4way"]["5"]["acc"] > 0.8 and r["some_2way"]["5"]["acc"] >= r["some_4way"]["5"]["acc"] - 0.05
        assert r["one"]["1"]["acc"] == 1.0            # a one-way task has one answer
    # the two runs were asked the same questions: same pools, and at step 0 (same weights) the same answers
    with open(os.path.join(pair["adamw"][0], "fewshot.json")) as f:
        other = json.load(f)
    assert other["pools"] == out["pools"]
    assert other["checkpoints"][0]["results"] == out["checkpoints"][0]["results"]

    dst = tmp_path / "curves"
    curves.main(["--runs", pair["adamw"][0], pair["muon"][0], "--out", str(dst)])
    made = sorted(os.listdir(dst))
    assert {"train_loss.png", "val_top1.png", "fewshot_held_k5_vs_step.png", "fewshot_held_k5_vs_val.png",
            "fewshot_again_k1_vs_step.png"} <= set(made)
    assert all(os.path.getsize(dst / n) > 5000 for n in made)


def test_features_cli_writes_both_kinds(pair, pack_dir, roles_file, tmp_path):
    out = tmp_path / "f.npz"
    features.main(["--ckpt", os.path.join(pair["adamw"][0], "ckpt-0000062.pt"), "--pack", pack_dir, "--roles",
                   roles_file, "--select", "held", "--per-class", "6", "--device", "cpu", "--out", str(out)])
    d = np.load(out)
    assert d["gap"].shape == d["pre_logits"].shape == (12, 32) and d["labels"].tolist() == [1] * 6 + [0] * 6
    assert int(d["step"]) == 62 and np.abs(d["pre_logits"]).max() <= 1.0
    pack = Pack(pack_dir)
    images, labels = features.decode_pool(pack, Roles.load(roles_file, 6), "held", image_size=32, per_class=6)
    model, _ = features.load_model(os.path.join(pair["adamw"][0], "ckpt-0000062.pt"), "cpu")
    again = features.extract(model, images, "cpu", batch_size=5)
    np.testing.assert_allclose(again["gap"], d["gap"], atol=1e-5)


def test_anchors_tell_the_right_numbering_from_a_wrong_one(pair, pack_dir, roles_file):
    """The trained classifier names the test images of a class it knows; under a shuffled numbering it does not."""
    from icl import anchors
    from icl.data import Pack, Roles
    pack = Pack(pack_dir)
    roles = Roles.load(roles_file, pack.num_classes)
    model, _ = features.load_model(sorted(glob.glob(os.path.join(pair["adamw"][0], "ckpt-*.pt")))[-1], "cpu")
    right = {pack.classes[c]: [c] for c in roles.base}
    rows = anchors.check(model, roles, pack.classes, pack, right, "cpu", amp=False)
    assert len(rows) == len(roles.base) and all(r["hit"] and r["top1"] > 0.9 for r in rows)
    wrong = {pack.classes[c]: [roles.base[(i + 1) % len(roles.base)]] for i, c in enumerate(roles.base)}
    rows = anchors.check(model, roles, pack.classes, pack, wrong, "cpu", amp=False)
    assert not any(r["hit"] for r in rows) and max(r["top1"] for r in rows) < 0.1
