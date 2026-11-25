"""Pack an image set into a few flat files that training can read with numpy and Pillow alone.

A pack is a directory:

    meta.json            format, counts, the shard list, how it was made
    classes.json         class names; the position in the list is the class id
    part-00000.bin       JPEG files, one after another
    part-00000.idx.npy   uint64 byte offsets, one more than there are images
    part-00000.lbl.npy   int32 class ids
    part-00000.key.txt   the source name of each image, one per line

Images whose short side exceeds `--short-side` are scaled down to it (never up) and written as JPEG at `--quality`.
An RGB JPEG that needs no scaling keeps its original bytes.

    python -m icl.pack wds  --out PACK --src 'dir/*-train-*.tar'       # <key>.jpg + <key>.cls (or a label in <key>.json)
    python -m icl.pack tree --out PACK --src images.tar.gz             # members <class_dir>/<image>.jpg
    python -m icl.pack list --out PACK --root DIR --list names.txt     # lines "<relative name> <class id>"
    python -m icl.pack peek --src some.tar                             # the first member names, to check a layout
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import threading
import time

import numpy as np
from PIL import Image

FORMAT = "icl-pack-1"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def encode_image(data: bytes, short_side: int, quality: int) -> bytes:
    """The bytes to store for one source image."""
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        s = min(w, h)
        if s <= short_side and im.format == "JPEG" and im.mode == "RGB":
            im.load()                                   # make sure it decodes before keeping it
            return data
        if s > short_side:
            f = short_side / s
            size = (max(1, round(w * f)), max(1, round(h * f)))
            if im.format == "JPEG":
                im.draft("RGB", size)                   # decode at a reduced scale when the source is far larger
            im = im.convert("RGB").resize(size, Image.BICUBIC)
        else:
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return buf.getvalue()


class ShardWriter:
    def __init__(self, out_dir: str, index: int):
        self.base = os.path.join(out_dir, f"part-{index:05d}")
        self.name = os.path.basename(self.base)
        self._f = open(self.base + ".bin.tmp", "wb")
        self.offsets, self.labels, self.keys = [0], [], []

    def add(self, jpeg: bytes, label, key: str) -> None:
        self._f.write(jpeg)
        self.offsets.append(self.offsets[-1] + len(jpeg))
        self.labels.append(label)
        self.keys.append(key)

    def close(self, labels=None) -> dict:
        """`labels` replaces what `add` was given (the `tree` mode only knows the class ids at the end)."""
        self._f.close()
        os.replace(self.base + ".bin.tmp", self.base + ".bin")
        np.save(self.base + ".idx.npy", np.asarray(self.offsets, dtype=np.uint64))
        np.save(self.base + ".lbl.npy", np.asarray(self.labels if labels is None else labels, dtype=np.int32))
        with open(self.base + ".key.txt", "w") as f:
            f.write("".join(k + "\n" for k in self.keys))
        return {"name": self.name, "count": len(self.keys), "bytes": int(self.offsets[-1])}


def _write_meta(out: str, shards: list[dict], classes: list[str], args, extra: dict) -> dict:
    meta = {"format": FORMAT, "num_images": sum(s["count"] for s in shards), "num_classes": len(classes),
            "short_side": args.short_side, "quality": args.quality, "shards": shards,
            "bytes": sum(s["bytes"] for s in shards), "mode": args.mode,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}
    with open(os.path.join(out, "classes.json"), "w") as f:
        json.dump(classes, f, indent=0)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


@contextlib.contextmanager
def _pool(workers: int):
    """A process pool, or None for `--workers 1` (everything in this process)."""
    if workers <= 1:
        yield None
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            yield pool


def _bounded_imap(pool, fn, items, max_pending: int):
    """`pool.imap` that never holds more than `max_pending` chunks waiting, whatever the reader's speed."""
    gate = threading.Semaphore(max_pending)

    def feed():
        for it in items:
            gate.acquire()
            yield it

    for res in pool.imap(fn, feed()):
        gate.release()
        yield res


# ---- wds: one output shard per input tar, the tars in parallel -------------------------------------------------

def _wds_samples(tar):
    """The webdataset rule: consecutive members that share a key (the base name up to its first dot) are one
    sample.  Yields (key, {extension: bytes})."""
    key, parts = None, {}
    for m in tar:
        if not m.isfile():
            continue
        k, _, ext = os.path.basename(m.name).partition(".")
        if k != key:
            if parts:
                yield key, parts
            key, parts = k, {}
        parts[ext.lower()] = tar.extractfile(m).read()
    if parts:
        yield key, parts


def _pack_wds_tar(job) -> dict:
    """One tar -> one shard.  A sample needs an image and a label: `<key>.cls` (an integer as text) or, without
    it, the integer `label` of `<key>.json`.  The stored name is the json's `filename` when it has one (for
    ImageNet that carries the synset), else the key."""
    index, path, out, short_side, quality = job
    writer = ShardWriter(out, index)
    skipped, max_label = 0, -1

    def skip(key, why):
        nonlocal skipped
        skipped += 1
        if skipped <= 3:                                # the first few say what is wrong; the count says how often
            print(f"pack: skipped {path}:{key}: {why}", file=sys.stderr)

    with tarfile.open(path, "r|*") as tar:
        for key, parts in _wds_samples(tar):
            img = next((parts[e[1:]] for e in IMAGE_EXT if e[1:] in parts), None)
            meta, label = {}, None
            try:
                if "json" in parts:
                    meta = json.loads(parts["json"])
                    meta = meta if isinstance(meta, dict) else {}
                if "cls" in parts:
                    label = int(parts["cls"].decode().strip())
                elif isinstance(meta.get("label"), int):
                    label = meta["label"]
            except ValueError as e:
                skip(key, f"unreadable label or json: {e}")
                continue
            if img is None or label is None:
                skip(key, f"no image or no label among {sorted(parts)}")
                continue
            name = meta.get("filename") or meta.get("file_name") or key
            try:
                writer.add(encode_image(img, short_side, quality), label, str(name).replace("\n", " "))
                max_label = max(max_label, label)
            except Exception as e:                      # an unreadable image is counted and left out
                skip(key, e)
    return {**writer.close(), "skipped": skipped, "max_label": max_label}


def pack_wds(args) -> dict:
    tars = sorted(p for pattern in args.src for p in glob.glob(pattern))
    if not tars:
        raise SystemExit(f"pack: no tar matches {args.src}")
    jobs = [(i, p, args.out, args.short_side, args.quality) for i, p in enumerate(tars)]
    res = []
    with _pool(min(args.workers, len(jobs))) as pool:
        for r in (pool.imap(_pack_wds_tar, jobs) if pool else map(_pack_wds_tar, jobs)):
            res.append(r)
            if len(res) % 50 == 0 or len(res) == len(jobs):
                print(f"pack: {len(res)}/{len(jobs)} tars, {sum(x['count'] for x in res)} images", flush=True)
    empty = [os.path.basename(p) for p, r in zip(tars, res) if r["count"] == 0]
    if empty:
        raise SystemExit(f"pack: {len(empty)} of {len(tars)} tars gave no image with a label (e.g. {empty[:3]}); "
                         "look at their layout with `pack peek`")
    n = args.num_classes or max(r["max_label"] for r in res) + 1
    classes = _read_names(args.class_names) if args.class_names else [str(i) for i in range(n)]
    if len(classes) < n:
        raise SystemExit(f"pack: {len(classes)} class names for labels up to {n - 1}")
    shards = [{k: r[k] for k in ("name", "count", "bytes")} for r in res]
    return _write_meta(args.out, shards, classes, args,
                       {"source": [os.path.basename(p) for p in tars], "skipped": sum(r["skipped"] for r in res)})


# ---- tree and list: one reader, a pool of encoders, shards of a fixed size -------------------------------------

def _encode_chunk(job):
    short_side, quality, items = job
    out = []
    for key, tag, data in items:
        try:
            if data is None:
                with open(key, "rb") as f:
                    data = f.read()
            out.append((key, tag, encode_image(data, short_side, quality), None))
        except Exception as e:
            out.append((key, tag, None, str(e)))
    return out


def _run_chunks(args, chunks, relabel=None, classes=None, extra=None) -> dict:
    """Encode chunks in order and write shards.  `tag` is the class id, or with `relabel` whatever it maps."""
    shards, tags, skipped, writer = [], [], 0, None
    open_tags: list = []
    jobs = ((args.short_side, args.quality, c) for c in chunks)
    t0 = time.time()
    with _pool(args.workers) as pool:
        for done in (_bounded_imap(pool, _encode_chunk, jobs, 4 * args.workers) if pool
                     else map(_encode_chunk, jobs)):
            for key, tag, jpeg, err in done:
                if jpeg is None:
                    skipped += 1
                    print(f"pack: skipped {key}: {err}", file=sys.stderr)
                    continue
                if writer is None:
                    writer, open_tags = ShardWriter(args.out, len(shards)), []
                writer.add(jpeg, 0, key if args.mode == "tree" else os.path.relpath(key, args.root))
                open_tags.append(tag)
                if len(open_tags) == args.shard_size:
                    shards.append(writer)
                    tags.append(open_tags)
                    writer = None
                    n = sum(len(t) for t in tags)
                    print(f"pack: {n} images, {n / (time.time() - t0):.0f}/s", flush=True)
    if writer is not None:
        shards.append(writer)
        tags.append(open_tags)
    if relabel is not None:
        classes, lookup = relabel(sorted({t for ts in tags for t in ts}))
        tags = [[lookup[t] for t in ts] for ts in tags]
    closed = [w.close(labels=ts) for w, ts in zip(shards, tags)]
    return _write_meta(args.out, closed, classes, args, {**(extra or {}), "skipped": skipped})


def pack_tree(args) -> dict:
    def chunks():
        chunk = []
        for path in args.src:
            with tarfile.open(path, "r|*") as tar:
                for m in tar:
                    if not m.isfile() or not m.name.lower().endswith(IMAGE_EXT):
                        continue
                    chunk.append((m.name, os.path.basename(os.path.dirname(m.name)), tar.extractfile(m).read()))
                    if len(chunk) == args.chunk:
                        yield chunk
                        chunk = []
        if chunk:
            yield chunk

    def relabel(found: list[str]):
        if args.classes_from:
            with open(os.path.join(args.classes_from, "classes.json")) as f:
                classes = json.load(f)
            missing = [c for c in found if c not in set(classes)]
            if missing:
                raise SystemExit(f"pack: {len(missing)} class folders are not in {args.classes_from}: {missing[:5]}")
        else:
            classes = found
        return classes, {c: i for i, c in enumerate(classes)}

    return _run_chunks(args, chunks(), relabel=relabel, extra={"source": [os.path.basename(p) for p in args.src]})


def pack_list(args) -> dict:
    rows = []
    with open(args.list) as f:
        for line in f:
            if line.strip():
                name, label = line.rsplit(None, 1)
                rows.append((os.path.join(args.root, name), int(label), None))
    n = args.num_classes or max(r[1] for r in rows) + 1
    classes = _read_names(args.class_names) if args.class_names else [str(i) for i in range(n)]
    if len(classes) < n:
        raise SystemExit(f"pack: {len(classes)} class names for labels up to {n - 1}")
    chunks = (rows[i:i + args.chunk] for i in range(0, len(rows), args.chunk))
    return _run_chunks(args, chunks, classes=classes, extra={"source": os.path.basename(args.list)})


def cpus() -> int:
    """The CPUs this process may use (a Slurm job sees its allocation, not the node)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                              # macOS
        return os.cpu_count() or 1


def peek(args) -> None:
    """Print the first members of each archive: what `wds` and `tree` would see."""
    for path in args.src:
        print(path)
        with tarfile.open(path, "r|*") as tar:
            for i, m in enumerate(tar):
                if i >= args.count:
                    break
                print(f"  {m.size:>10}  {m.name}" + ("" if m.isfile() else "  (not a file)"))


def _read_names(path: str) -> list[str]:
    if path.endswith(".json"):
        with open(path) as f:
            return list(json.load(f))
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("wds", "tree", "list"):
        p = sub.add_parser(mode)
        p.add_argument("--out", required=True)
        p.add_argument("--short-side", type=int, default=288)
        p.add_argument("--quality", type=int, default=92)
        p.add_argument("--workers", type=int, default=max(1, cpus() - 2))
        p.add_argument("--expect-images", type=int, default=0,
                       help="fail, and set the pack aside, unless it holds exactly this many images")
        if mode in ("wds", "tree"):
            p.add_argument("--src", nargs="+", required=True, help="tar files" + (" or globs" if mode == "wds" else ""))
        if mode in ("wds", "list"):
            p.add_argument("--num-classes", type=int, default=0)
            p.add_argument("--class-names", help="one name per line, or a JSON list")
        if mode in ("tree", "list"):
            p.add_argument("--shard-size", type=int, default=20_000)
            p.add_argument("--chunk", type=int, default=64, help="images handed to a worker at a time")
        if mode == "tree":
            p.add_argument("--classes-from", help="a pack whose class list to use (e.g. the training pack)")
        if mode == "list":
            p.add_argument("--root", required=True)
            p.add_argument("--list", required=True)
    p = sub.add_parser("peek")
    p.add_argument("--src", nargs="+", required=True)
    p.add_argument("--count", type=int, default=12)
    args = ap.parse_args(argv)
    if args.mode == "peek":
        return peek(args)
    if os.path.exists(os.path.join(args.out, "meta.json")):
        raise SystemExit(f"pack: {args.out} already holds a pack")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    meta = {"wds": pack_wds, "tree": pack_tree, "list": pack_list}[args.mode](args)
    print(f"pack: {meta['num_images']} images, {meta['num_classes']} classes, {meta['bytes'] / 1e9:.2f} GB, "
          f"{len(meta['shards'])} shards, {meta['skipped']} skipped, {time.time() - t0:.0f} s -> {args.out}")
    if args.expect_images and meta["num_images"] != args.expect_images:
        os.replace(os.path.join(args.out, "meta.json"), os.path.join(args.out, "meta.rejected.json"))
        raise SystemExit(f"pack: expected {args.expect_images} images, got {meta['num_images']}; the pack is set "
                         f"aside (meta.rejected.json) and will not load")


if __name__ == "__main__":
    main()
