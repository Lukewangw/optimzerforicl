"""Objects365, as stored inside FineVision (`finevision/objects365_qa`), turned into a one-label-per-image set.

FineVision keeps each Objects365 photo with a few question/answer turns written from its detection labels:

    "Yes, there is one Person at the following normalized coordinates (x, y, w, h): (0.32, 0.50, 0.65, 1.00)."
    "Yes, there are 2 Stuffed Toys at the following normalized coordinates (x, y, w, h): (...), (...)."
    "In this image, I can identify the following objects at their respective normalized coordinates (x, y, w, h):
     1 Person at (...), 2 Stuffed Toys at (...), (...)."

(x, y) is the centre of a box, all four numbers are fractions of the photo's width and height.  This module reads
the class names and boxes back out of the answers, cuts the boxes out of the photos and writes the cut-outs as packs
(pack.py's format), one class per cut-out.  Nothing here is specific to cats or dogs: which classes are held out of
training is the roles file's business, except that `extract --exclude-with` leaves out every cut-out of another
class that comes from a photo showing one of the named classes (so those animals are in no training picture at all).

    python -m icl.o365 scan    --src 'DIR/*.parquet' --out INDEX      # one file each
    python -m icl.o365 stats   INDEX                                  # classes, sizes, what did not parse
    python -m icl.o365 extract --index INDEX --out DIR                # the packs DIR/train and DIR/val
    python -m icl.o365 colours DIR/train                              # do bananas come out yellow? (labels vs pictures)

`scan` and `extract` read parquet with pyarrow (imported only there); the rest needs numpy and Pillow.
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zlib

import numpy as np
from PIL import Image

from . import pack as packer

# ---- reading the labels back out of the answers --------------------------------------------------------------------

_NUM = r"[0-9]*\.?[0-9]+"
_BOX_NC = rf"\(\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*\)"
_BOX = re.compile(rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\)")
_ITEM = re.compile(rf"\b(one|\d+)\s+([^():,]+?)\s+at(?: the following normalized coordinates \(x, y, w, h\):)?\s*"
                   rf"((?:{_BOX_NC})(?:\s*,\s*{_BOX_NC})*)")
_HEADER = "(x, y, w, h):"                               # boxes are only counted after this; it holds no numbers itself


def parse_answer(text: str) -> tuple[list[tuple[str, int, list[tuple[float, float, float, float]]]], int]:
    """([(name as written, the count the text states, boxes)], boxes in the text that no name was read for)."""
    items, used = [], 0
    for m in _ITEM.finditer(text):
        boxes = [tuple(float(v) for v in b) for b in _BOX.findall(m.group(3))]
        items.append((m.group(2).strip(), 1 if m.group(1) == "one" else int(m.group(1)), boxes))
        used += len(boxes)
    return items, len(_BOX.findall(text)) - used


def parse_turns(texts) -> dict:
    """The labels of one photo from all its turns: {"boxes": {name as written: set of boxes}, "single": names
    written for exactly one object, "unread": boxes without a name, "miscount": items whose stated count is not
    the number of boxes that follow}.  A box repeated in several turns is kept once."""
    boxes: dict[str, set] = collections.defaultdict(set)
    single, unread, miscount = set(), 0, 0
    for t in texts:
        items, lost = parse_answer(t["assistant"])
        unread += lost
        for name, count, bs in items:
            boxes[name].update(bs)
            miscount += count != len(bs)
            if count == 1:
                single.add(name)
    return {"boxes": dict(boxes), "single": single, "unread": unread, "miscount": miscount}


_IRREGULAR = {"Geese": "Goose", "Mice": "Mouse", "Knives": "Knife", "People": "Person", "Men": "Man",
              "Women": "Woman", "Children": "Child", "Feet": "Foot", "Teeth": "Tooth"}


def singular_of(name: str, singles) -> str | None:
    """The class a many-objects form stands for, given the names that were written for a single object.  The
    answers were written by a plain rule: a name that ends in "s" is left as it is ("Other Shoes", "Bus"), any
    other gets an "s" ("Persons", "Gooses", "Potatos").  The more careful forms are tried after that, in case."""
    cands = [name, name[:-1] if name.endswith("s") else None, _IRREGULAR.get(name)]
    if name.endswith("ies"):
        cands.append(name[:-3] + "y")
    if name.endswith("ves"):
        cands += [name[:-3] + "f", name[:-3] + "fe"]
    if name.endswith("es"):
        cands.append(name[:-2])
    return next((c for c in cands if c and c in singles), None)


def class_table(single_counts: dict, plural_counts: dict) -> tuple[list[str], dict[str, str], list[str]]:
    """(class names sorted, {name as written: class}, plural forms that fit no class).  A class is a name that was
    written for a single object at least once; each plural form is mapped onto one of them."""
    classes = sorted(single_counts)
    mapping = {c: c for c in classes}
    lost = []
    for p in plural_counts:
        if p in mapping:
            continue
        c = singular_of(p, single_counts)
        if c is None:
            lost.append(p)
        else:
            mapping[p] = c
    return classes, mapping, sorted(lost)


# ---- parquet files ----------------------------------------------------------------------------------------------------

def list_sources(src: list[str]) -> list[str]:
    """The parquet files of `src` (globs)."""
    out = []
    for s in src:
        out += glob.glob(s)
    return sorted(out)


class Local:
    """`with Local(src, tmp) as path`: the file on disk."""

    def __init__(self, src: str, tmp: str):
        self.src, self.tmp = src, tmp

    def __enter__(self) -> str:
        return self.src

    def __exit__(self, *exc) -> None:
        pass


def batches(path: str, columns: list[str], batch_size: int = 32):
    """(row number of the batch's first row, batch) over a parquet file."""
    import pyarrow.parquet as pq
    start = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        yield start, batch
        start += batch.num_rows


def _stem(src: str) -> str:
    return os.path.basename(src)[:-len(".parquet")]


def _tmp_dir(given: str | None) -> str:
    return given or os.path.join(os.environ.get("TMPDIR") or tempfile.gettempdir(),
                                 f"icl-o365-{os.getuid()}")


# ---- scan: every photo's labels and size -> an index ----------------------------------------------------------------

def _scan_file(job) -> dict:
    src, out, tmp = job
    stem = _stem(src)
    dst = os.path.join(out, stem)
    if os.path.exists(dst + ".json"):
        return {"file": stem, "already": True}
    t0 = time.time()
    names: dict[str, int] = {}
    rows, name_ids, boxes, sizes, full = [], [], [], [], []
    single, plural, c = collections.Counter(), collections.Counter(), collections.Counter()
    try:
        with Local(src, tmp) as path:
            for start, batch in batches(path, ["images", "texts"]):
                imgs, texts = batch.column("images"), batch.column("texts").to_pylist()
                for j, tx in enumerate(texts):
                    tx = tx or []
                    c["rows"] += 1
                    size = (0, 0)
                    held = imgs[j].as_py() or []
                    if len(held) != 1 or not held[0].get("bytes"):
                        c["rows_without_exactly_one_image"] += 1
                    else:
                        try:
                            with Image.open(io.BytesIO(held[0]["bytes"])) as im:
                                size = im.size
                                c["exif_rotated"] += im.getexif().get(0x0112, 1) not in (0, 1)
                                c[f"format_{im.format}"] += 1
                                c[f"mode_{im.mode}"] += 1
                        except Exception:
                            c["unreadable_images"] += 1
                    sizes.append(size)
                    res = parse_turns(tx)
                    lists = [t for t in tx if t["assistant"].startswith("In this image")]
                    full.append(bool(lists))
                    c["unread_boxes"] += res["unread"]
                    c["miscounts"] += res["miscount"]
                    if lists:                               # how much of the list do the one-class questions cover?
                        listed = set(parse_turns(lists[:1])["boxes"])
                        asked = set(parse_turns([t for t in tx if t not in lists])["boxes"])
                        c["listed_classes"] += len(listed)
                        c["listed_classes_also_asked"] += len(listed & asked)
                    for name, bs in res["boxes"].items():
                        (single if name in res["single"] else plural)[name] += len(bs)
                        k = names.setdefault(name, len(names))
                        for b in sorted(bs):
                            rows.append(start + j)
                            name_ids.append(k)
                            boxes.append(b)
    except Exception as e:
        return {"file": stem, "error": f"{type(e).__name__}: {e}"}
    np.savez(dst + ".tmp.npz", row=np.asarray(rows, dtype=np.int32), name=np.asarray(name_ids, dtype=np.int32),
             box=np.asarray(boxes, dtype=np.float32).reshape(-1, 4), size=np.asarray(sizes, dtype=np.int32).reshape(-1, 2),
             full=np.asarray(full, dtype=bool))
    os.replace(dst + ".tmp.npz", dst + ".npz")
    meta = {"file": stem, "src": src, "names": list(names), "single": dict(single), "plural": dict(plural),
            "counters": dict(c), "boxes": len(rows), "seconds": round(time.time() - t0, 1)}
    with open(dst + ".json.tmp", "w") as f:
        json.dump(meta, f)
    os.replace(dst + ".json.tmp", dst + ".json")            # the json is written last: it marks the file as done
    return {"file": stem, "rows": c["rows"], "boxes": len(rows), "seconds": meta["seconds"]}


def scan(args) -> int:
    srcs = list_sources(args.src)
    if args.limit:
        srcs = srcs[:args.limit]
    if not srcs:
        raise SystemExit(f"o365: no parquet file in {args.src}")
    os.makedirs(args.out, exist_ok=True)
    tmp = _tmp_dir(args.tmp)
    jobs = [(s, args.out, tmp) for s in srcs]
    t0, failed, rows, boxes = time.time(), [], 0, 0
    with packer._pool(min(args.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(_scan_file, jobs) if pool else map(_scan_file, jobs), 1):
            if "error" in r:
                failed.append(r)
                print(f"o365: FAILED {r['file']}: {r['error']}", flush=True)
            rows += r.get("rows", 0)
            boxes += r.get("boxes", 0)
            if i % 20 == 0 or i == len(jobs):
                print(f"o365: scanned {i}/{len(jobs)} files, {rows} photos, {boxes} boxes, {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"o365: scan done, {len(failed)} files failed -> {args.out}")
    return 1 if failed else 0


# ---- the index as a whole -----------------------------------------------------------------------------------------------

class Index:
    """All scanned files: the class table, and per file the boxes with their class, photo row and photo size."""

    def __init__(self, path: str):
        metas = []
        for p in sorted(glob.glob(os.path.join(path, "*.json"))):
            if os.path.basename(p) not in ("stats.json",):
                with open(p) as f:
                    metas.append(json.load(f))
        if not metas:
            raise SystemExit(f"o365: no scanned file in {path}")
        self.path, self.metas = path, metas
        single, plural, self.counters = collections.Counter(), collections.Counter(), collections.Counter()
        for m in metas:
            single.update(m["single"])
            plural.update(m["plural"])
            self.counters.update(m["counters"])
        self.classes, mapping, self.lost = class_table(single, plural)
        cid = {c: i for i, c in enumerate(self.classes)}
        self.files = []
        for m in metas:
            z = np.load(os.path.join(path, m["file"] + ".npz"))
            to_class = np.array([cid.get(mapping.get(n), -1) for n in m["names"]] or [-1], dtype=np.int32)
            size = z["size"].astype(np.float32)
            row, box = z["row"], z["box"]
            wh = box[:, 2:4] * size[row] if len(row) else np.zeros((0, 2), dtype=np.float32)
            self.files.append({"stem": m["file"], "src": m["src"], "row": row, "cls": to_class[z["name"]] if len(row) else
                               np.zeros(0, dtype=np.int32), "box": box, "size": z["size"], "full": z["full"],
                               "short": wh.min(1) if len(row) else np.zeros(0, dtype=np.float32),
                               "long": wh.max(1) if len(row) else np.zeros(0, dtype=np.float32)})

    def class_id(self, name: str) -> int:
        if name not in self.classes:
            raise SystemExit(f"o365: no class '{name}' (classes are named as in the answers, e.g. 'Dog', 'Cat')")
        return self.classes.index(name)


def stats(args) -> int:
    ix = Index(args.index)
    c = ix.counters
    cls = np.concatenate([f["cls"] for f in ix.files])
    short = np.concatenate([f["short"] for f in ix.files])
    long_ = np.concatenate([f["long"] for f in ix.files])
    sizes = np.concatenate([f["size"] for f in ix.files])
    full_rows = int(sum(f["full"].sum() for f in ix.files))
    n = len(ix.classes)
    print(f"{len(ix.files)} files, {c['rows']} photos ({full_rows} with a complete list), {len(cls)} boxes, {n} classes")
    print("counters:", {k: v for k, v in sorted(c.items())})
    if c["listed_classes"]:
        print(f"one-class questions cover {c['listed_classes_also_asked'] / c['listed_classes']:.1%} of the classes a "
              f"complete list names")
    print(f"many-object forms that fit no class: {ix.lost}; boxes of such forms: {int((cls < 0).sum())}")
    ok = sizes[(sizes > 0).all(1)]
    if len(ok):
        print("photo short side, percentiles 1/10/50/90/99:", np.percentile(ok.min(1), [1, 10, 50, 90, 99]).round().tolist(),
              "| long side:", np.percentile(ok.max(1), [1, 10, 50, 90, 99]).round().tolist())
    table = {}
    print("boxes by the short side of the box in pixels (ratio of sides at most 6), and classes having at least k such boxes:")
    for t in (32, 48, 64, 96, 128, 160, 224):
        keep = (short >= t) & (long_ <= 6 * np.maximum(short, 1e-6)) & (cls >= 0)
        per = np.bincount(cls[keep], minlength=n)
        table[t] = per
        print(f"  >= {t:3d} px: {int(keep.sum()):>9,d} boxes | classes with >=100: {int((per >= 100).sum())}, >=500: "
              f"{int((per >= 500).sum())}, >=1000: {int((per >= 1000).sum())}, >=3000: {int((per >= 3000).sum())} | "
              f"total if capped at 3000/4000/5000/6000: " + "/".join(f"{int(np.minimum(per, k).sum()):,d}" for k in (3000, 4000, 5000, 6000)))
    for name in args.show:
        if name in ix.classes:
            i = ix.classes.index(name)
            photos = sum(len(np.unique(f["row"][f["cls"] == i])) for f in ix.files)
            print(f"  {name}: {int((cls == i).sum())} boxes in {photos} photos; by size " +
                  ", ".join(f">={t}: {int(table[t][i])}" for t in table))
    t = args.at
    order = np.argsort(table[t])
    print(f"smallest classes at >= {t} px:", [(ix.classes[i], int(table[t][i])) for i in order[:25]])
    print(f"largest classes at >= {t} px:", [(ix.classes[i], int(table[t][i])) for i in order[::-1][:12]])
    with open(os.path.join(args.index, "stats.json"), "w") as f:
        json.dump({"files": len(ix.files), "photos": int(c["rows"]), "full_photos": full_rows, "boxes": int(len(cls)),
                   "classes": ix.classes, "lost_forms": ix.lost, "counters": dict(c),
                   "per_class": {str(t): table[t].tolist() for t in table}}, f)
    return 0


# ---- extract: choose boxes, cut them out, write the packs ---------------------------------------------------------------

SPLITS = ("train", "val")


def crop_window(width: int, height: int, box, margin: float) -> tuple[int, int, int, int]:
    """The pixel window of a box, widened on every side by `margin` times the box's size, its edges rounded to
    whole pixels and kept inside the photo."""
    cx, cy, w, h = (float(v) for v in box)
    bw, bh = w * width, h * height
    x0 = max(0, round(cx * width - bw / 2 - margin * bw))
    y0 = max(0, round(cy * height - bh / 2 - margin * bh))
    x1 = min(width, round(cx * width + bw / 2 + margin * bw))
    y1 = min(height, round(cy * height + bh / 2 + margin * bh))
    x0, y0 = min(x0, width - 1), min(y0, height - 1)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def encode_crop(im: Image.Image, short_side: int, quality: int) -> bytes:
    w, h = im.size
    s = min(w, h)
    if s > short_side:
        f = short_side / s
        im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def choose(ix: Index, *, min_side: float, max_ratio: float, cap: int, val_cap: int, val_mod: int, seed: int,
           exclude_with: list[str], only_full: bool) -> tuple[list[dict], dict]:
    """Which boxes become cut-outs.  Returns per file {row: [(split, class id, box)]} and the counts.

    A box qualifies when its short side is at least `min_side` pixels and its sides' ratio at most `max_ratio`.
    With `only_full`, photos without a complete object list are left out.  Photos showing a class of
    `exclude_with` give cut-outs of those classes only.  A photo goes to "val" when a hash of its name says so
    (one in `val_mod`), so no photo feeds both sets.  Each class then keeps at most `cap` training cut-outs and
    `val_cap` validation cut-outs, drawn at random by `seed`."""
    excl = np.array([ix.class_id(nm) for nm in exclude_with], dtype=np.int32)
    parts = []
    for fi, f in enumerate(ix.files):
        ok = (f["cls"] >= 0) & (f["short"] >= min_side) & (f["long"] <= max_ratio * np.maximum(f["short"], 1e-6))
        if only_full:
            ok &= f["full"][f["row"]]
        if len(excl):
            is_excl = np.isin(f["cls"], excl)
            rows_with = np.unique(f["row"][is_excl])
            ok &= is_excl | ~np.isin(f["row"], rows_with)
        val_row = np.array([zlib.crc32(f"{f['stem']}:{r}".encode()) % val_mod == 0 for r in range(len(f["size"]))], dtype=bool)
        idx = np.flatnonzero(ok)
        parts.append((np.full(len(idx), fi, dtype=np.int32), idx.astype(np.int32), f["cls"][idx],
                      val_row[f["row"][idx]].astype(np.int32)))
    file_of = np.concatenate([p[0] for p in parts])
    box_of = np.concatenate([p[1] for p in parts])
    cls = np.concatenate([p[2] for p in parts])
    split = np.concatenate([p[3] for p in parts])
    n = len(ix.classes)
    counts = {s: np.zeros(n, dtype=np.int64) for s in SPLITS}
    eligible = {s: np.bincount(cls[split == i], minlength=n) for i, s in enumerate(SPLITS)}
    order = np.argsort(cls * 2 + split, kind="stable")
    bounds = np.searchsorted((cls * 2 + split)[order], np.arange(2 * n + 1))
    keep = []
    for g in range(2 * n):
        members = order[bounds[g]:bounds[g + 1]]
        limit = (cap, val_cap)[g % 2]
        if len(members) > limit:
            members = np.sort(np.random.default_rng([seed, g, 0x0365]).choice(members, size=limit, replace=False))
        counts[SPLITS[g % 2]][g // 2] = len(members)
        keep.append(members)
    keep = np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=np.int64)
    per_file: list[dict] = [collections.defaultdict(list) for _ in ix.files]
    for k in keep:
        f = ix.files[file_of[k]]
        b = box_of[k]
        per_file[file_of[k]][int(f["row"][b])].append((int(split[k]), int(cls[k]), tuple(float(v) for v in f["box"][b])))
    return [dict(d) for d in per_file], {"eligible": {s: eligible[s].tolist() for s in SPLITS},
                                         "chosen": {s: counts[s].tolist() for s in SPLITS}}


def _extract_file(job) -> dict:
    index, src, out, tmp, todo, short_side, quality, margin = job
    stem = _stem(src)
    writers: dict[int, packer.ShardWriter] = {}
    unreadable = 0
    try:
        with Local(src, tmp) as path:
            for start, batch in batches(path, ["images"]):
                imgs = batch.column("images")
                for j in range(batch.num_rows):
                    want = todo.get(start + j)
                    if not want:
                        continue
                    try:
                        with Image.open(io.BytesIO(imgs[j].as_py()[0]["bytes"])) as im:
                            im = im.convert("RGB")
                    except Exception:
                        unreadable += len(want)
                        continue
                    for split, cid, box in want:
                        crop = im.crop(crop_window(im.width, im.height, box, margin))
                        if split not in writers:
                            writers[split] = packer.ShardWriter(os.path.join(out, SPLITS[split]), index)
                        writers[split].add(encode_crop(crop, short_side, quality), cid,
                                           f"{stem}/{start + j}/" + ",".join(f"{v:.5f}" for v in box))
    except Exception as e:
        return {"file": stem, "error": f"{type(e).__name__}: {e}"}
    return {"file": stem, "index": index, "unreadable": unreadable,
            "shards": {SPLITS[s]: w.close() for s, w in writers.items()}}


def extract(args) -> int:
    for s in SPLITS:
        if os.path.exists(os.path.join(args.out, s, "meta.json")):
            raise SystemExit(f"o365: {args.out}/{s} already holds a pack")
        os.makedirs(os.path.join(args.out, s), exist_ok=True)
    ix = Index(args.index)
    rules = {"min_side": args.min_side, "max_ratio": args.max_ratio, "margin": args.margin, "cap": args.cap,
             "val_cap": args.val_cap, "val_mod": args.val_mod, "seed": args.seed, "exclude_with": args.exclude_with,
             "only_full": not args.all_photos}
    todo, counts = choose(ix, **{k: v for k, v in rules.items() if k != "margin"})
    total = {s: int(sum(counts["chosen"][s])) for s in SPLITS}
    print(f"o365: {len(ix.classes)} classes; chosen {total['train']} training and {total['val']} validation cut-outs "
          f"from {sum(len(t) for t in todo)} photos in {sum(1 for t in todo if t)} files; rules {rules}", flush=True)
    tmp = _tmp_dir(args.tmp)
    jobs = [(i, f["src"], args.out, tmp, t, args.short_side, args.quality, args.margin)
            for i, (f, t) in enumerate(zip(ix.files, todo)) if t]
    t0, failed, done, unreadable = time.time(), [], {s: [] for s in SPLITS}, 0
    with packer._pool(min(args.workers, max(1, len(jobs)))) as pool:
        for i, r in enumerate(pool.imap_unordered(_extract_file, jobs) if pool else map(_extract_file, jobs), 1):
            if "error" in r:
                failed.append(r)
                print(f"o365: FAILED {r['file']}: {r['error']}", flush=True)
                continue
            unreadable += r["unreadable"]
            for s, shard in r["shards"].items():
                done[s].append((r["index"], shard))
            if i % 20 == 0 or i == len(jobs):
                print(f"o365: {i}/{len(jobs)} files, {sum(x[1]['count'] for x in done['train'])} training cut-outs, "
                      f"{time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        print(f"o365: {len(failed)} files failed; no pack written")
        return 1
    like = argparse.Namespace(short_side=args.short_side, quality=args.quality, mode="o365")
    for s in SPLITS:
        shards = [x[1] for x in sorted(done[s], key=lambda x: x[0])]
        got = np.zeros(len(ix.classes), dtype=np.int64)
        for sh in shards:
            got += np.bincount(np.load(os.path.join(args.out, s, sh["name"] + ".lbl.npy")), minlength=len(ix.classes))
        meta = packer._write_meta(os.path.join(args.out, s), shards, ix.classes, like,
                                  {"source": os.path.dirname(ix.files[0]["src"]), "rules": rules, "skipped": unreadable,
                                   "per_class": got.tolist(), "eligible_per_class": counts["eligible"][s]})
        print(f"o365: {s}: {meta['num_images']} cut-outs, {int((got > 0).sum())} classes present, "
              f"{meta['bytes'] / 1e9:.2f} GB, {len(shards)} shards")
    print(f"o365: {unreadable} cut-outs lost to unreadable photos; {time.time() - t0:.0f} s -> {args.out}")
    return 0


COLOURED = ["Banana", "Lemon", "Broccoli", "Green Vegetables", "Cucumber", "Strawberry", "Tomato", "Orange/Tangerine",
            "Carrot", "Pumpkin", "Fire Truck", "Eggplant", "Zebra", "Person", "Dog", "Cat"]


def colours(args) -> int:
    """A check of labels against pictures that needs no eyes: the mean colour of the middle half of the cut-outs of
    classes whose colour is known (bananas yellow, broccoli green, tomatoes red, zebras grey), beside the mean over
    all classes.  If the boxes or the labels were off, these would all look like the overall mean."""
    from .data import Pack
    pack = Pack(args.pack)
    rng = np.random.default_rng(0)

    def mean_rgb(idx) -> np.ndarray:
        acc = np.zeros(3)
        for i in idx:
            a = np.asarray(Image.open(io.BytesIO(pack.jpeg(int(i)))).convert("RGB"), dtype=np.float32)
            h, w = a.shape[:2]
            acc += a[h // 4:h - h // 4, w // 4:w - w // 4].reshape(-1, 3).mean(0)
        return acc / max(len(idx), 1)

    def line(name, idx) -> str:
        r, g, b = mean_rgb(idx)
        return f"  {name:18s} n={len(idx):4d}  R {r:5.1f}  G {g:5.1f}  B {b:5.1f}   R-G {r - g:+6.1f}  G-B {g - b:+6.1f}"

    print(f"{args.pack}: mean colour of the middle of the cut-outs")
    print(line("(all classes)", rng.choice(len(pack), size=min(args.sample * 10, len(pack)), replace=False)))
    for name in args.classes:
        if name in pack.classes:
            idx = np.flatnonzero(pack.labels == pack.classes.index(name))
            print(line(name, rng.choice(idx, size=min(args.sample, len(idx)), replace=False)) if len(idx) else f"  {name}: none")
    sides = [min(Image.open(io.BytesIO(pack.jpeg(int(i)))).size) for i in rng.choice(len(pack), size=min(3000, len(pack)), replace=False)]
    print("short side of the stored cut-outs, percentiles 5/25/50/75/95:", np.percentile(sides, [5, 25, 50, 75, 95]).round().tolist())
    return 0


def leak(args) -> int:
    """Photos that hold a held-out class still give cut-outs of the classes named in `--exclude-with`.  How many of
    those cut-outs show part of the held-out class: their window (the box plus its margin) overlaps one of its boxes
    in the same photo."""
    ix = Index(args.index)
    held = ix.class_id(args.held)
    kept = np.array([ix.class_id(n) for n in args.kept], dtype=np.int32)
    n_kept = n_same_photo = n_overlap = 0
    for f in ix.files:
        if not len(f["row"]):
            continue
        ok = (f["short"] >= args.min_side) & (f["long"] <= args.max_ratio * np.maximum(f["short"], 1e-9))
        ok &= f["full"][f["row"]].astype(bool)
        is_kept = np.isin(f["cls"], kept) & ok
        n_kept += int(is_kept.sum())
        is_held = f["cls"] == held
        cand = np.flatnonzero(is_kept & np.isin(f["row"], np.unique(f["row"][is_held])))
        n_same_photo += len(cand)
        for i in cand:
            cx, cy, w, h = f["box"][i]
            w, h = w * (1 + 2 * args.margin), h * (1 + 2 * args.margin)
            d = f["box"][is_held & (f["row"] == f["row"][i])]
            n_overlap += bool(((np.abs(d[:, 0] - cx) < (d[:, 2] + w) / 2) & (np.abs(d[:, 1] - cy) < (d[:, 3] + h) / 2)).any())
    print(f"o365 leak: {n_kept} boxes of {', '.join(args.kept)} pass the size rules; {n_same_photo} of them are in a "
          f"photo with a {args.held} box; the window of {n_overlap} overlaps a {args.held} box")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--src", nargs="+", required=True, help="globs of parquet files")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="only the first N files (a trial)")
    p.add_argument("--workers", type=int, default=max(1, packer.cpus() - 2))
    p.add_argument("--tmp", help="a folder for temporary files (default: the system's)")
    p = sub.add_parser("stats")
    p.add_argument("index")
    p.add_argument("--show", nargs="*", default=["Cat", "Dog", "Lion", "Wild Bird", "Horse", "Person"])
    p.add_argument("--at", type=int, default=96, choices=(32, 48, 64, 96, 128, 160, 224))
    p = sub.add_parser("extract")
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True, help="gets train/ and val/")
    p.add_argument("--min-side", type=float, default=96.0, help="a box's short side in pixels, at least")
    p.add_argument("--max-ratio", type=float, default=6.0, help="a box's long side over its short side, at most")
    p.add_argument("--margin", type=float, default=0.1, help="widen the cut on every side by this share of the box")
    p.add_argument("--cap", type=int, default=5000, help="training cut-outs per class, at most")
    p.add_argument("--val-cap", type=int, default=50)
    p.add_argument("--val-mod", type=int, default=50, help="one photo in this many goes to validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-with", nargs="*", default=[], metavar="CLASS",
                   help="photos showing one of these classes give cut-outs of these classes only")
    p.add_argument("--all-photos", action="store_true", help="also use photos that have no complete object list")
    p.add_argument("--short-side", type=int, default=288)
    p.add_argument("--quality", type=int, default=92)
    p.add_argument("--workers", type=int, default=max(1, packer.cpus() - 2))
    p.add_argument("--tmp")
    p = sub.add_parser("colours")
    p.add_argument("pack")
    p.add_argument("--classes", nargs="*", default=COLOURED)
    p.add_argument("--sample", type=int, default=300)
    p = sub.add_parser("leak")
    p.add_argument("--index", required=True)
    p.add_argument("--held", default="Dog")
    p.add_argument("--kept", nargs="+", default=["Cat", "Lion"])
    p.add_argument("--min-side", type=float, default=64)
    p.add_argument("--max-ratio", type=float, default=6)
    p.add_argument("--margin", type=float, default=0.1)
    args = ap.parse_args(argv)
    return {"scan": scan, "stats": stats, "extract": extract, "colours": colours, "leak": leak}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
