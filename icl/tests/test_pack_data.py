import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import PER_CLASS, jpeg_bytes, make_list_pack, synth_image
from icl import aug, data, pack as packer
from icl.data import Pack, Roles, StreamSampler, eval_loader, make_roles, select_indices, train_loader


def test_list_pack_round_trips_images_and_labels(image_dir, pack_dir):
    root, rows = image_dir
    pack = Pack(pack_dir)
    assert len(pack) == len(rows) == 6 * PER_CLASS + 3 and pack.num_classes == 6
    assert pack.labels.tolist() == [c for _, c in rows]
    assert pack.keys() == [n for n, _ in rows]
    assert len(pack.meta["shards"]) == 3 and pack.meta["skipped"] == 0             # 147 images, 64 a shard
    for i in (0, 63, 64, 100, len(rows) - 1):
        im = Image.open(io.BytesIO(pack.jpeg(i)))
        assert im.format == "JPEG" and im.mode == "RGB" and min(im.size) <= 40
    # a small RGB JPEG keeps its bytes; nothing is scaled up
    with open(root / rows[5][0], "rb") as f:
        assert pack.jpeg(5) == f.read()
    sizes = {n: Image.open(io.BytesIO(pack.jpeg(i))).size for i, (n, _) in enumerate(rows)}
    assert sizes["class_0/big.jpg"] == (53, 40)                                    # 400 x 300, short side to 40
    assert sizes["class_0/odd.png"] == (48, 40) and sizes["class_0/gray.jpg"] == (48, 40)
    assert all(v == (48, 40) for n, v in sizes.items() if n != "class_0/big.jpg")
    # what was stored is the picture that went in (colour survives the PNG and the downscale)
    for name in ("class_0/big.jpg", "class_0/odd.png"):
        a = np.asarray(Image.open(io.BytesIO(pack.jpeg([n for n, _ in rows].index(name)))))
        assert a[..., 0].mean() > 120 > a[..., 1].mean()
    with pytest.raises(SystemExit):
        packer.main(["list", "--out", pack_dir, "--root", str(root), "--list", str(root / "list.txt")])


def test_a_pack_of_many_shards_is_read_within_the_open_file_limit(image_dir, tmp_path, monkeypatch):
    root, rows = image_dir
    path = make_list_pack(root, tmp_path / "many", shard_size=5)
    pack = Pack(path)
    assert len(pack.meta["shards"]) == 30                                          # 147 images, 5 a shard
    want = []
    for sh in pack.meta["shards"]:                                                 # read independently of Pack
        off = np.load(os.path.join(path, sh["name"] + ".idx.npy"))
        with open(os.path.join(path, sh["name"] + ".bin"), "rb") as f:
            blob = f.read()
        want += [blob[int(a):int(b)] for a, b in zip(off[:-1], off[1:])]
    order = np.random.default_rng(0).permutation(len(pack)).tolist() * 2           # jumps between shards
    monkeypatch.setattr(data, "_BUDGET", 3)
    assert all(pack.jpeg(i) == want[i] for i in order)
    mine = [p for p in data._FDS if p.startswith(path)]
    assert len(mine) <= 3 and len(data._FDS) <= 3
    assert 8 <= data._open_file_budget() <= 3840
    # a process whose table of open files is full where the budget thought it was not: it gives files back and
    # goes on.  In a child process, because a lowered hard limit cannot be raised again.
    code = (
        "import resource, sys, numpy as np\n"
        "from icl import data\n"
        "pack = data.Pack(sys.argv[1])\n"
        "ref = [pack.jpeg(i) for i in range(len(pack))]\n"
        "for fd in list(data._FDS.values()): __import__('os').close(fd)\n"
        "data._FDS.clear()\n"
        "used = len(__import__('os').listdir('/dev/fd'))\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (used + 6, used + 6))\n"
        "data._BUDGET = 1000\n"
        "order = np.random.default_rng(1).permutation(len(pack)).tolist()\n"
        "assert all(pack.jpeg(i) == ref[i] for i in order)\n"
        "print('budget', data._BUDGET, 'open', len(data._FDS))\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")])}
    r = subprocess.run([sys.executable, "-c", code, path], capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    budget, opened = int(r.stdout.split()[1]), int(r.stdout.split()[3])
    assert 1 <= budget <= 6 and opened <= budget


def test_tree_pack_from_a_tar_gz_with_a_worker_pool(image_dir, tmp_path):
    root, rows = image_dir
    tar_path = tmp_path / "tree.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for c in (3, 1, 2):                                                     # not in sorted order on purpose
            tar.add(root / f"class_{c}", arcname=f"top/class_{c}")
    out = tmp_path / "tree"
    packer.main(["tree", "--out", str(out), "--src", str(tar_path), "--short-side", "40", "--workers", "2",
                 "--shard-size", "50", "--chunk", "7"])
    pack = Pack(str(out))
    assert pack.classes == ["class_1", "class_2", "class_3"] and len(pack) == 3 * PER_CLASS
    assert len(pack.meta["shards"]) == 2
    for key, label in zip(pack.keys(), pack.labels):
        assert key.split("/")[1] == pack.classes[label]
    assert pack.keys()[0].startswith("top/class_3/")                               # the tar's order is kept
    a = np.asarray(Image.open(io.BytesIO(pack.jpeg(0)))).reshape(-1, 3).mean(0)
    assert a[0] > 150 and a[1] > 150 and a[2] < 120                                # class 3 is the yellow one
    # a second tar with fewer folders takes its numbering from the first pack
    tar2 = tmp_path / "val.tar"
    with tarfile.open(tar2, "w") as tar:
        tar.add(root / "class_3", arcname="val/class_3")
    out2 = tmp_path / "tree_val"
    packer.main(["tree", "--out", str(out2), "--src", str(tar2), "--workers", "1", "--classes-from", str(out)])
    val = Pack(str(out2))
    assert val.classes == pack.classes and set(val.labels.tolist()) == {2}
    tar3 = tmp_path / "other.tar"
    with tarfile.open(tar3, "w") as tar:
        tar.add(root / "class_0", arcname="x/class_0")
    with pytest.raises(SystemExit):
        packer.main(["tree", "--out", str(tmp_path / "bad"), "--src", str(tar3), "--workers", "1",
                     "--classes-from", str(out)])


def test_wds_pack_reads_jpg_and_cls_members(tmp_path):
    def add(tar, name, data):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    want = []
    for t in range(2):
        with tarfile.open(tmp_path / f"set-train-{t:04d}.tar", "w") as tar:
            for i in range(5):
                label, key = (t * 5 + i) % 4, f"n{t}_{i}"
                first, second = ((".cls", str(label).encode()), (".jpg", jpeg_bytes(synth_image(label, i))))[::1 - 2 * (i % 2)]
                add(tar, key + first[0], first[1])
                add(tar, key + ".json", b"{}")
                add(tar, key + second[0], second[1])
                want.append((key, label))
            if t == 1:
                add(tar, "broken.jpg", b"not a jpeg")
                add(tar, "broken.cls", b"1")
    out = tmp_path / "wds"
    packer.main(["wds", "--out", str(out), "--src", str(tmp_path / "set-train-*.tar"), "--workers", "1",
                 "--num-classes", "6"])
    pack = Pack(str(out))
    assert list(zip(pack.keys(), pack.labels.tolist())) == want
    assert pack.num_classes == 6 and len(pack.meta["shards"]) == 2 and pack.meta["skipped"] == 1
    a = np.asarray(Image.open(io.BytesIO(pack.jpeg(3)))).reshape(-1, 3).mean(0)     # label 3: yellow
    assert a[0] > 150 and a[1] > 150 and a[2] < 120


def test_wds_pack_reads_the_label_from_json_and_refuses_what_it_cannot_read(tmp_path):
    def add(tar, name, data):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with tarfile.open(tmp_path / "a-0000.tar", "w") as tar:                  # no .cls member: the label is in the json
        for i in range(4):
            add(tar, f"s{i}.json", json.dumps({"label": i % 3, "filename": f"n0000000{i % 3}_{i}.JPEG"}).encode())
            add(tar, f"s{i}.jpg", jpeg_bytes(synth_image(i % 3, i)))
        add(tar, "nolabel.jpg", jpeg_bytes(synth_image(0, 9)))               # an image without a label is left out
    src = ["--src", str(tmp_path / "a-*.tar"), "--workers", "1"]
    packer.main(["wds", "--out", str(tmp_path / "ok"), *src, "--expect-images", "4"])
    pack = Pack(str(tmp_path / "ok"))
    assert pack.keys() == [f"n0000000{i % 3}_{i}.JPEG" for i in range(4)]    # the json's file name is the stored name
    assert pack.labels.tolist() == [0, 1, 2, 0] and pack.meta["skipped"] == 1

    with pytest.raises(SystemExit, match="expected 5 images, got 4"):        # a wrong count sets the pack aside
        packer.main(["wds", "--out", str(tmp_path / "short"), *src, "--expect-images", "5"])
    assert (tmp_path / "short" / "meta.rejected.json").exists()
    with pytest.raises(FileNotFoundError):
        Pack(str(tmp_path / "short"))

    with tarfile.open(tmp_path / "b-0000.tar", "w") as tar:                  # a layout the reader does not know
        for i in range(3):
            add(tar, f"x{i}.jpg", jpeg_bytes(synth_image(0, i)))
            add(tar, f"x{i}.txt", b"0")
    with pytest.raises(SystemExit, match="gave no image with a label"):
        packer.main(["wds", "--out", str(tmp_path / "none"), "--src", str(tmp_path / "b-*.tar"), "--workers", "1"])
    assert not (tmp_path / "none" / "meta.json").exists()


def test_roles_drop_and_renumber(pack_dir, roles_file):
    pack = Pack(pack_dir)
    roles = Roles.load(roles_file, pack.num_classes)
    assert roles.base == [0, 1, 2, 3] and roles.held_out == [4, 5]
    assert roles.label_map("base").tolist() == [0, 1, 2, 3, -1, -1]
    assert roles.label_map("held").tolist() == [-1, -1, -1, -1, 1, 0]               # in the order the group lists
    assert roles.label_map("one").tolist() == [-1, -1, -1, -1, -1, 0]
    assert roles.label_map("all").tolist() == list(range(6))
    with pytest.raises(KeyError):
        roles.label_map("nope")
    with pytest.raises(ValueError):
        Roles(6, {"bad": [1, 1]})
    with pytest.raises(ValueError):
        Roles.load(roles_file, 7)
    base = select_indices(pack, roles.label_map("base"))
    assert len(base) == 4 * PER_CLASS + 3 and set(pack.labels[base].tolist()) == {0, 1, 2, 3}
    few = select_indices(pack, roles.label_map("held"), per_class=5)
    assert pack.labels[few].tolist() == [4] * 5 + [5] * 5 and np.all(np.diff(few) > 0)
    assert few.tolist() == np.flatnonzero(pack.labels == 4)[:5].tolist() + np.flatnonzero(pack.labels == 5)[:5].tolist()
    # without a roles file nothing is held out
    assert Roles.load(None, 6).base == list(range(6))

    loader = eval_loader(pack, roles, "held", batch_size=7, image_size=32, per_class=5)
    xs, ys = zip(*loader)
    assert sum(len(y) for y in ys) == 10 and xs[0].shape == (7, 32, 32, 3) and xs[0].dtype == torch.uint8
    assert np.concatenate([y.numpy() for y in ys]).tolist() == [1] * 5 + [0] * 5


def test_make_roles_draws_a_fixed_control_outside_the_groups(tmp_path):
    groups = {"cats_domestic": list(range(281, 286)), "cats_all": list(range(281, 294))}
    a = make_roles(1000, groups, control=13, seed=0)
    b = make_roles(1000, groups, control=13, seed=0)
    c = make_roles(1000, groups, control=13, seed=1)
    assert a == b and a["groups"]["control"] != c["groups"]["control"]
    assert len(a["groups"]["control"]) == 13 and not set(a["groups"]["control"]) & set(range(281, 294))
    assert len(Roles(1000, a["groups"]).base) == 1000 - 26
    d = make_roles(1000, groups, control=5, seed=0, control_from=list(range(151, 269)))
    assert all(151 <= x <= 268 for x in d["groups"]["control"])
    from icl import data
    out = tmp_path / "roles.json"
    data.main(["make-roles", "--num-classes", "1000", "--group", "cats_domestic=281-285", "--group",
               "cats_all=281-293", "--control", "13", "--seed", "0", "--out", str(out)])
    with open(out) as f:
        assert json.load(f) == a


def test_stream_covers_each_epoch_once_and_ranks_share_a_batch():
    n, batch, total = 37, 8, 14
    s = StreamSampler(n, batch, data_seed=3, total_steps=total)
    flat = [k for b in s for k in b]
    assert len(flat) == total * batch and len(s) == total
    for e in range((total * batch) // n):
        part = flat[e * n:(e + 1) * n]
        assert {k[0] for k in part} == {e} and sorted(k[1] for k in part) == list(range(n))
    assert [k[1] for k in flat[:n]] != [k[1] for k in flat[n:2 * n]]                 # a new order every epoch
    # two processes take alternate elements of the same global batches
    r0 = list(StreamSampler(n, batch, 3, total, rank=0, world=2))
    r1 = list(StreamSampler(n, batch, 3, total, rank=1, world=2))
    whole = list(s)
    for b, a0, a1 in zip(whole, r0, r1):
        assert len(a0) == len(a1) == 4 and a0 == b[0::2] and a1 == b[1::2]
    # resuming starts at the right place; another seed is another stream
    assert list(StreamSampler(n, batch, 3, total, start_step=9)) == whole[9:]
    assert list(StreamSampler(n, batch, 4, total)) != whole
    with pytest.raises(ValueError):
        StreamSampler(n, 9, 3, total, world=2)


def test_train_batches_depend_on_the_data_seed_only(pack_dir, roles_file):
    pack = Pack(pack_dir)
    roles = Roles.load(roles_file, pack.num_classes)
    kw = dict(batch_size=8, total_steps=3, image_size=32, randaug_layers=2)
    a = [(x.numpy().copy(), y.numpy().copy()) for x, y in train_loader(pack, roles, data_seed=5, **kw)]
    b = [(x.numpy().copy(), y.numpy().copy()) for x, y in train_loader(pack, roles, data_seed=5, **kw)]
    c = [(x.numpy().copy(), y.numpy().copy()) for x, y in train_loader(pack, roles, data_seed=6, **kw)]
    assert len(a) == 3 and a[0][0].shape == (8, 32, 32, 3) and a[0][0].dtype == np.uint8
    assert all(np.array_equal(p[0], q[0]) and np.array_equal(p[1], q[1]) for p, q in zip(a, b))
    assert not all(np.array_equal(p[0], q[0]) for p, q in zip(a, c))
    assert set(np.concatenate([y for _, y in a]).tolist()) <= {0, 1, 2, 3}           # held-out classes never appear
    # a worker process gives the same batches as the main process does
    w = [(x.numpy().copy(), y.numpy().copy()) for x, y in train_loader(pack, roles, data_seed=5, workers=1, **kw)]
    assert all(np.array_equal(p[0], q[0]) and np.array_equal(p[1], q[1]) for p, q in zip(a, w))
    # and starting at step 2 gives step 2
    late = list(train_loader(pack, roles, data_seed=5, start_step=2, **kw))
    assert len(late) == 1 and np.array_equal(late[0][0].numpy(), a[2][0])
