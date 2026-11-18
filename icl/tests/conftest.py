"""Tiny synthetic image sets for the tests: six classes told apart by colour, a few odd files among them."""
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repo root

from icl import pack as packer  # noqa: E402

CLASS_COLOURS = [(220, 40, 40), (40, 200, 60), (50, 70, 230), (230, 220, 50), (200, 60, 210), (60, 210, 220)]
PER_CLASS = 24


def synth_image(label: int, i: int, size=(48, 40)) -> Image.Image:
    rng = np.random.default_rng([label, i])
    a = np.clip(np.array(CLASS_COLOURS[label], dtype=np.float32) + rng.normal(0, 30, size=(size[1], size[0], 3)), 0, 255)
    a[rng.integers(0, size[1] // 2):, :rng.integers(4, size[0] // 2)] *= 0.6          # some structure
    return Image.fromarray(a.astype(np.uint8))


def jpeg_bytes(im: Image.Image, quality=90) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


@pytest.fixture(scope="session")
def image_dir(tmp_path_factory):
    """`root/<class>/<i>.jpg` plus `list.txt`.  Class 0 also holds a large image, a PNG and a grayscale JPEG."""
    root = tmp_path_factory.mktemp("images")
    rows = []
    for c in range(len(CLASS_COLOURS)):
        os.makedirs(root / f"class_{c}")
        for i in range(PER_CLASS):
            name = f"class_{c}/{i:03d}.jpg"
            synth_image(c, i).save(root / name, "JPEG", quality=90)
            rows.append((name, c))
    synth_image(0, 900, size=(400, 300)).save(root / "class_0/big.jpg", "JPEG", quality=90)
    synth_image(0, 901).save(root / "class_0/odd.png", "PNG")
    synth_image(0, 902).convert("L").save(root / "class_0/gray.jpg", "JPEG")
    rows += [("class_0/big.jpg", 0), ("class_0/odd.png", 0), ("class_0/gray.jpg", 0)]
    with open(root / "list.txt", "w") as f:
        f.write("".join(f"{n} {c}\n" for n, c in rows))
    return root, rows


def make_list_pack(root, out, short_side=40, workers=1, shard_size=64):
    packer.main(["list", "--out", str(out), "--root", str(root), "--list", str(root / "list.txt"),
                 "--short-side", str(short_side), "--workers", str(workers), "--shard-size", str(shard_size)])
    return str(out)


@pytest.fixture(scope="session")
def pack_dir(image_dir, tmp_path_factory):
    root, _ = image_dir
    return make_list_pack(root, tmp_path_factory.mktemp("packs") / "train")


@pytest.fixture(scope="session")
def roles_file(tmp_path_factory):
    import json
    path = tmp_path_factory.mktemp("roles") / "roles.json"
    with open(path, "w") as f:
        json.dump({"num_classes": 6, "groups": {"held": [5, 4], "one": [5]}}, f)
    return str(path)
