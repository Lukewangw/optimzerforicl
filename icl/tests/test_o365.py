"""The Objects365 reader on small parquet files written here: labels read back from the answers, the rules for
choosing boxes, and the cut-outs themselves."""
import io
import os

import numpy as np
import pytest
from PIL import Image

from icl import o365
from icl.data import Pack

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

COLOURS = {"Dog": (200, 30, 30), "Cat": (30, 200, 30), "Bus": (30, 30, 200), "Goose": (220, 220, 40), "Key": (220, 40, 220)}
HEAD = "at the following normalized coordinates (x, y, w, h):"
LIST = "In this image, I can identify the following objects at their respective normalized coordinates (x, y, w, h): "
SIZE = (400, 300)
BUS, DOG, CAT = (0.3, 0.5, 0.4, 0.5), (0.8, 0.3, 0.3, 0.4), (0.8, 0.75, 0.3, 0.4)
GEESE = [(0.75, 0.25, 0.3, 0.3), (0.75, 0.7, 0.3, 0.3)]


def fmt(box) -> str:
    return "(" + ", ".join(f"{v:.5f}" for v in box) + ")"


def many(name: str) -> str:
    return name if name.endswith("s") else name + "s"


def photo(objects) -> bytes:
    a = np.full((SIZE[1], SIZE[0], 3), 128, dtype=np.uint8)
    for name, (cx, cy, w, h) in objects:
        a[round((cy - h / 2) * SIZE[1]):round((cy + h / 2) * SIZE[1]),
          round((cx - w / 2) * SIZE[0]):round((cx + w / 2) * SIZE[0])] = COLOURS[name]
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def turns(objects, complete=True) -> list[dict]:
    by: dict[str, list] = {}
    for name, box in objects:
        by.setdefault(name, []).append(box)
    name, boxes = next(iter(by.items()))
    answer = (f"Yes, there is one {name} {HEAD} {fmt(boxes[0])}." if len(boxes) == 1 else
              f"Yes, there are {len(boxes)} {many(name)} {HEAD} " + ", ".join(map(fmt, boxes)) + ".")
    out = [{"user": f"Can you find all instances of {many(name)}?", "assistant": answer}]
    if complete:
        out.append({"user": "Where are the objects in this image?", "assistant": LIST + ", ".join(
            f"{len(bs)} {n if len(bs) == 1 else many(n)} at " + ", ".join(map(fmt, bs)) for n, bs in by.items()) + "."})
    return out


def rows() -> list[list]:
    """80 photos: a bus in each, geese in some, a dog or a cat in a few, and three odd ones at the end."""
    out = []
    for i in range(77):
        objects = [("Bus", BUS)]
        if i % 3 == 0:
            objects += [("Goose", g) for g in GEESE[:1 + i % 2]]
        elif i % 10 == 1:
            objects.append(("Dog", DOG))
        elif i % 10 == 2:
            objects.append(("Cat", CAT))
        out.append(objects)
    out.append([("Key", (0.5, 0.5, 0.1, 0.1)), ("Bus", BUS)])                # a box of 40 x 30 pixels: too small
    out.append([("Key", (0.5, 0.5, 0.9, 0.22))])                             # 360 x 66: sides' ratio near 5.5, kept
    out.append([("Bus", BUS)])                                               # written below without a complete list
    return out


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    root = tmp_path_factory.mktemp("o365")
    data = rows()
    for part in range(2):
        chunk = data[part * 40:(part + 1) * 40]
        table = pa.table({"images": [[{"bytes": photo(o), "path": None}] for o in chunk],
                          "texts": [turns(o, complete=part * 40 + i != 79) for i, o in enumerate(chunk)]})
        pq.write_table(table, root / f"train-{part:05d}-of-00002.parquet", row_group_size=7)
    index = root / "index"
    assert o365.main(["scan", "--src", str(root / "*.parquet"), "--out", str(index), "--workers", "2"]) == 0
    return root, str(index), data


def test_answers_give_names_counts_and_boxes():
    items, unread = o365.parse_answer(f"Yes, there are 2 Stuffed Toys {HEAD} (0.77938, 0.45396, 0.15306, 0.17237), "
                                      "(0.64864, 0.58706, 0.22832, 0.19726).")
    assert unread == 0 and items == [("Stuffed Toys", 2, [(0.77938, 0.45396, 0.15306, 0.17237),
                                                           (0.64864, 0.58706, 0.22832, 0.19726)])]
    items, unread = o365.parse_answer(LIST + "1 Formula 1 at (0.1, 0.2, 0.3, 0.4), 2 Cell Phones at (0.5, 0.5, 0.1, 0.1), "
                                      "(0.6, 0.6, 0.1, 0.1), 1 Cabinet/Shelf at (0.2, 0.2, 0.2, 0.2).")
    assert unread == 0 and [(n, c, len(b)) for n, c, b in items] == [("Formula 1", 1, 1), ("Cell Phones", 2, 2),
                                                                     ("Cabinet/Shelf", 1, 1)]
    assert o365.parse_answer("There are boxes at (0.1, 0.2, 0.3, 0.4) but no name.") == ([], 1)
    res = o365.parse_turns(turns([("Bus", BUS), ("Goose", GEESE[0]), ("Goose", GEESE[1])]))
    assert {k: len(v) for k, v in res["boxes"].items()} == {"Bus": 1, "Gooses": 2}       # the list repeats the bus: once
    assert res["single"] == {"Bus"} and res["unread"] == 0 and res["miscount"] == 0


def test_many_object_forms_are_mapped_onto_classes():
    singles = {"Goose": 3, "Bus": 5, "Other Shoes": 2, "Person": 9, "Formula 1": 1}
    classes, mapping, lost = o365.class_table(singles, {"Gooses": 4, "Bus": 2, "Other Shoes": 7, "Persons": 30,
                                                        "Formula 1s": 1, "Zebras": 2})
    assert classes == sorted(singles) and lost == ["Zebras"]
    assert mapping["Gooses"] == "Goose" and mapping["Bus"] == "Bus" and mapping["Other Shoes"] == "Other Shoes"
    assert mapping["Persons"] == "Person" and mapping["Formula 1s"] == "Formula 1"


def test_crop_window_widens_the_box_and_stays_inside():
    assert o365.crop_window(400, 300, BUS, 0.1) == (24, 60, 216, 240)
    assert o365.crop_window(400, 300, BUS, 0.0) == (40, 75, 200, 225)
    assert o365.crop_window(400, 300, (0.05, 0.5, 0.1, 1.0), 0.5) == (0, 0, 60, 300)
    x0, y0, x1, y1 = o365.crop_window(400, 300, (0.999, 0.999, 0.0001, 0.0001), 0.0)
    assert 0 <= x0 < x1 <= 400 and 0 <= y0 < y1 <= 300


def test_scan_reads_every_photo_and_box(scanned):
    root, index, data = scanned
    ix = o365.Index(index)
    assert ix.classes == ["Bus", "Cat", "Dog", "Goose", "Key"] and ix.lost == []
    c = ix.counters
    assert c["rows"] == 80 and c["unread_boxes"] == 0 and c["miscounts"] == 0 and c["format_JPEG"] == 80
    assert sum(len(f["row"]) for f in ix.files) == sum(len(o) for o in data)
    assert int(sum(f["full"].sum() for f in ix.files)) == 79
    f = ix.files[0]
    bus = f["cls"] == ix.class_id("Bus")
    assert np.allclose(f["short"][bus], 150, atol=0.01) and np.allclose(f["long"][bus], 160, atol=0.01)
    assert o365.main(["scan", "--src", str(root / "*.parquet"), "--out", index, "--workers", "1"]) == 0     # nothing to redo
    assert o365.main(["stats", index, "--show", "Dog"]) == 0


def test_leak_counts_kept_boxes_in_photos_of_the_held_class(scanned, capsys):
    root, index, data = scanned
    assert o365.main(["leak", "--index", index, "--held", "Dog", "--kept", "Cat", "--min-side", "60"]) == 0
    words = capsys.readouterr().out.split("o365 leak: ")[1].split()
    kept, same_photo, overlap = int(words[0]), int(words[words.index("rules;") + 1]), int(words[words.index("of", 12) + 1])
    assert kept >= same_photo >= overlap >= 0
    both = sum(any(n == "Dog" for n, _ in o) and any(n == "Cat" for n, _ in o) for o in data[:79])
    assert (same_photo > 0) == (both > 0)


def test_extract_follows_the_rules(scanned, tmp_path):
    root, index, data = scanned
    out = tmp_path / "packs"
    args = ["extract", "--index", index, "--out", str(out), "--min-side", "60", "--cap", "20", "--val-cap", "3",
            "--val-mod", "4", "--exclude-with", "Dog", "Cat", "--workers", "1"]
    assert o365.main(args) == 0
    with pytest.raises(SystemExit, match="already holds a pack"):
        o365.main(args)
    train, val = Pack(str(out / "train")), Pack(str(out / "val"))
    assert train.classes == ["Bus", "Cat", "Dog", "Goose", "Key"] == val.classes
    names = {c: i for i, c in enumerate(train.classes)}

    def photos(pack):                                                        # "file/row" of each cut-out
        return [k.rsplit("/", 1)[0] for k in pack.keys()]

    assert not set(photos(train)) & set(photos(val))                         # no photo feeds both sets
    per = np.bincount(train.labels, minlength=5)
    assert per[names["Bus"]] == 20 and np.bincount(val.labels, minlength=5).max() <= 3      # the caps
    assert train.meta["rules"]["exclude_with"] == ["Dog", "Cat"] and train.meta["per_class"] == per.tolist()

    row_of = {f"train-{i // 40:05d}-of-00002/{i % 40}": i for i in range(80)}
    for pack in (train, val):
        for key, label in zip(photos(pack), pack.labels.tolist()):
            objects = data[row_of[key]]
            shown = {n for n, _ in objects}
            assert row_of[key] != 79                                         # the photo without a complete list
            if shown & {"Dog", "Cat"}:
                assert train.classes[label] in ("Dog", "Cat")                # nothing else is cut from such a photo
            assert train.classes[label] in shown
    assert per[names["Dog"]] + np.bincount(val.labels, minlength=5)[names["Dog"]] == sum(("Dog", DOG) in o for o in data)
    keys = np.flatnonzero(np.concatenate([train.labels, val.labels]) == names["Key"])
    assert len(keys) == 1                                                    # the 40 x 30 one is too small

    i = int(np.flatnonzero(train.labels == names["Bus"])[0])                 # a cut-out: the box plus a tenth each side
    im = Image.open(io.BytesIO(train.jpeg(i)))
    assert im.size == (192, 180)
    r, g, b = np.asarray(im)[90, 96]
    assert b > 150 and r < 90 and g < 90


def test_a_cut_out_larger_than_the_pack_size_is_scaled_down():
    im = Image.fromarray(np.zeros((600, 900, 3), dtype=np.uint8))
    assert Image.open(io.BytesIO(o365.encode_crop(im, 288, 92))).size == (432, 288)
    assert Image.open(io.BytesIO(o365.encode_crop(im.resize((100, 80)), 288, 92))).size == (100, 80)
