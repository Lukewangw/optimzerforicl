"""Reading packs (pack.py), holding classes out, and the deterministic training stream.

Roles file (`roles.json`): named lists of held-out class ids, in the pack's own numbering:

    {"groups": {"cats_domestic": [281, 282, 283, 284, 285], "cats_all": [281, ..., 293], "control": [...]}}

Training drops every class that appears in any group and renumbers the rest 0..C-1 in ascending order of their
ids.  An evaluation set selects "base" (the classes trained on, renumbered the same way), one group (renumbered in
the order the group lists them) or "all".

    python -m icl.data make-roles --num-classes 1000 --group cats_domestic=281-285 --group cats_all=281-293 \
        --control 13 --seed 0 --out roles.json
"""
from __future__ import annotations

import argparse
import errno
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from . import aug


# Shard files are read with pread through descriptors kept open per process, least recently used first.  A pack
# made from N tars has N shards (1,024 for one ImageNet layout) and the usual soft limit is 1,024 open files, so
# the limit is raised where the hard one allows and whatever does not fit the budget is opened again when needed.
_FDS: dict[str, int] = {}
_BUDGET: int | None = None


def _open_file_budget() -> int:
    try:
        import resource
    except ImportError:
        return 128
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 4096
    if soft != resource.RLIM_INFINITY and soft < want:
        target = want if hard == resource.RLIM_INFINITY else min(want, hard)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        except (ValueError, OSError):
            pass
    return 3840 if soft == resource.RLIM_INFINITY else max(8, min(3840, soft - 256))


def _read(path: str, offset: int, size: int) -> bytes:
    global _BUDGET
    fd = _FDS.pop(path, None)
    if fd is None:
        if _BUDGET is None:
            _BUDGET = _open_file_budget()
        while len(_FDS) >= _BUDGET:
            os.close(_FDS.pop(next(iter(_FDS))))
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as e:                                         # the budget was too generous for this process
            if e.errno != errno.EMFILE or not _FDS:
                raise
            for _ in range((len(_FDS) + 1) // 2):
                os.close(_FDS.pop(next(iter(_FDS))))
            _BUDGET = max(1, len(_FDS))
            fd = os.open(path, os.O_RDONLY)
    _FDS[path] = fd                                                  # the most recently used goes last
    data = os.pread(fd, size, offset)
    while len(data) < size:
        more = os.pread(fd, size - len(data), offset + len(data))
        if not more:
            raise OSError(f"{path}: {len(data)} of {size} bytes at offset {offset}")
        data += more
    return data


class Pack:
    """Read access to a pack directory.  Safe to hand to DataLoader workers, forked or spawned: it holds paths
    and index arrays only, and the open files belong to each process (not to be read from several threads)."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != "icl-pack-1":
            raise ValueError(f"{path}: not an icl pack")
        with open(os.path.join(path, "classes.json")) as f:
            self.classes = json.load(f)
        names = [s["name"] for s in self.meta["shards"]]
        self._bins = [os.path.join(path, n + ".bin") for n in names]
        for b in self._bins:                                         # a pack written again at the same place
            if b in _FDS:
                os.close(_FDS.pop(b))
        self._offsets = [np.load(os.path.join(path, n + ".idx.npy")) for n in names]
        self.labels = np.concatenate([np.load(os.path.join(path, n + ".lbl.npy")) for n in names]).astype(np.int64)
        counts = np.array([len(o) - 1 for o in self._offsets], dtype=np.int64)
        self._starts = np.concatenate([[0], np.cumsum(counts)])
        if len(self.labels) != self._starts[-1] or len(self.labels) != self.meta["num_images"]:
            raise ValueError(f"{path}: the shard files do not add up to meta.json")

    def __len__(self) -> int:
        return int(self._starts[-1])

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def jpeg(self, i: int) -> bytes:
        s = int(np.searchsorted(self._starts, i, side="right") - 1)
        j = i - int(self._starts[s])
        off = self._offsets[s]
        return _read(self._bins[s], int(off[j]), int(off[j + 1]) - int(off[j]))

    def keys(self) -> list[str]:
        out = []
        for s in self.meta["shards"]:
            with open(os.path.join(self.path, s["name"] + ".key.txt")) as f:
                out += f.read().splitlines()
        return out


class Roles:
    """Which classes are held out, and the renumbering of the rest."""

    def __init__(self, num_classes: int, groups: dict[str, list[int]] | None = None):
        self.num_classes = int(num_classes)
        self.groups = {k: [int(c) for c in v] for k, v in (groups or {}).items()}
        for name, ids in self.groups.items():
            if len(set(ids)) != len(ids) or any(not 0 <= c < num_classes for c in ids):
                raise ValueError(f"roles: group '{name}' has repeated or out-of-range class ids")
        held = {c for ids in self.groups.values() for c in ids}
        self.held_out = sorted(held)
        self.base = [c for c in range(self.num_classes) if c not in held]
        self.base_map = np.full(self.num_classes, -1, dtype=np.int64)
        self.base_map[self.base] = np.arange(len(self.base))

    @classmethod
    def load(cls, path: str | None, num_classes: int) -> "Roles":
        if not path:
            return cls(num_classes)
        with open(path) as f:
            d = json.load(f)
        if d.get("num_classes", num_classes) != num_classes:
            raise ValueError(f"{path} was written for {d['num_classes']} classes, the pack has {num_classes}")
        return cls(num_classes, d["groups"])

    def label_map(self, select: str) -> np.ndarray:
        """Pack class id -> label under `select` ("base", "all" or a group name); -1 where the class is left out."""
        if select == "base":
            return self.base_map
        if select == "all":
            return np.arange(self.num_classes, dtype=np.int64)
        if select not in self.groups:
            raise KeyError(f"roles: no group '{select}' (have {sorted(self.groups)})")
        m = np.full(self.num_classes, -1, dtype=np.int64)
        m[self.groups[select]] = np.arange(len(self.groups[select]))
        return m


def select_indices(pack: Pack, label_map: np.ndarray, per_class: int = 0) -> np.ndarray:
    """Pack positions of the images whose class is kept, in pack order; at most `per_class` of each if given."""
    keep = np.flatnonzero(label_map[pack.labels] >= 0)
    if per_class:
        lab = pack.labels[keep]
        order = np.argsort(lab, kind="stable")
        sorted_lab = lab[order]
        first = np.searchsorted(sorted_lab, sorted_lab, side="left")
        keep = np.sort(keep[order[np.arange(len(order)) - first < per_class]])
    return keep


class PackDataset(Dataset):
    """Items are (uint8 HWC array, label).  A training item is asked for by (epoch, position) and its
    augmentation is seeded from the data seed, the epoch and the image's position in the pack."""

    def __init__(self, pack: Pack, indices: np.ndarray, label_map: np.ndarray, transform, data_seed: int = 0):
        self.pack, self.indices, self.transform, self.data_seed = pack, np.asarray(indices), transform, data_seed
        self.labels = label_map[pack.labels[self.indices]]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, key):
        epoch, i = key if isinstance(key, tuple) else (0, key)
        pos = int(self.indices[i])
        rng = aug.sample_rng(self.data_seed, epoch, pos)
        return self.transform(self.pack.jpeg(pos), rng), int(self.labels[i])


def collate(items):
    return torch.from_numpy(np.stack([a for a, _ in items])), torch.tensor([b for _, b in items], dtype=torch.long)


class StreamSampler(Sampler):
    """The training stream: one fresh permutation of the set per epoch, laid end to end and cut into global
    batches; step s takes stream positions [s*B, (s+1)*B), so an epoch boundary may fall inside a batch (as in
    big_vision's repeated, shuffled stream).  Rank r of `world` takes every `world`-th element of each global
    batch.  The stream depends on (data seed, set size, batch size) only."""

    def __init__(self, n: int, batch_size: int, data_seed: int, total_steps: int, start_step: int = 0,
                 rank: int = 0, world: int = 1):
        if batch_size % world:
            raise ValueError("the global batch size must be divisible by the number of processes")
        self.n, self.batch, self.seed = int(n), int(batch_size), int(data_seed)
        self.total, self.start, self.rank, self.world = int(total_steps), int(start_step), rank, world
        self._perm = (None, None)

    def permutation(self, epoch: int) -> np.ndarray:
        if self._perm[0] != epoch:
            self._perm = (epoch, np.random.default_rng([self.seed, int(epoch), 0x0DE2]).permutation(self.n))
        return self._perm[1]

    def global_batch(self, step: int) -> list[tuple[int, int]]:
        out = []
        for p in range(step * self.batch, (step + 1) * self.batch):
            e, o = divmod(p, self.n)
            out.append((e, int(self.permutation(e)[o])))
        return out

    def __iter__(self):
        for step in range(self.start, self.total):
            yield self.global_batch(step)[self.rank::self.world]

    def __len__(self) -> int:
        return max(0, self.total - self.start)


def train_loader(pack: Pack, roles: Roles, *, batch_size: int, data_seed: int, total_steps: int, start_step: int = 0,
                 rank: int = 0, world: int = 1, workers: int = 0, image_size: int = 224, randaug_layers: int = 2,
                 randaug_magnitude: float = 10.0, pin: bool = False) -> DataLoader:
    lm = roles.label_map("base")
    ds = PackDataset(pack, select_indices(pack, lm), lm,
                     aug.TrainTransform(image_size, randaug_layers, randaug_magnitude), data_seed)
    sampler = StreamSampler(len(ds), batch_size, data_seed, total_steps, start_step, rank, world)
    return DataLoader(ds, batch_sampler=sampler, num_workers=workers, collate_fn=collate, pin_memory=pin,
                      persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)


def eval_loader(pack: Pack, roles: Roles, select: str = "base", *, batch_size: int = 256, workers: int = 0,
                image_size: int = 224, per_class: int = 0, pin: bool = False) -> DataLoader:
    lm = roles.label_map(select)
    ds = PackDataset(pack, select_indices(pack, lm, per_class), lm, aug.EvalTransform(image_size))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate,
                      pin_memory=pin)


# ---- roles file ---------------------------------------------------------------------------------------------------

def _parse_ids(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def make_roles(num_classes: int, groups: dict[str, list[int]], control: int = 0, seed: int = 0,
               control_from: list[int] | None = None, control_name: str = "control") -> dict:
    """The roles dictionary: the given groups plus `control` classes drawn at random (by `seed`) from the classes
    in no group, or from `control_from` when given."""
    taken = {c for ids in groups.values() for c in ids}
    out = {k: list(v) for k, v in groups.items()}
    if control:
        pool = [c for c in (control_from if control_from is not None else range(num_classes)) if c not in taken]
        if control > len(pool):
            raise ValueError("more control classes asked for than there are candidates")
        out[control_name] = sorted(int(c) for c in np.random.default_rng([int(seed), 0xC0]).choice(
            pool, size=control, replace=False))
    Roles(num_classes, out)                                          # validates
    return {"num_classes": num_classes, "groups": out, "control_seed": seed if control else None}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-roles")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--group", action="append", default=[], metavar="NAME=IDS", help="e.g. cats_all=281-293")
    p.add_argument("--control", type=int, default=0, help="how many random classes to hold out as the control")
    p.add_argument("--control-from", help="draw the control from these ids only, e.g. 151-268")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("info")
    p.add_argument("pack")
    args = ap.parse_args(argv)
    if args.cmd == "make-roles":
        groups = {name: _parse_ids(ids) for name, ids in (g.split("=", 1) for g in args.group)}
        roles = make_roles(args.num_classes, groups, args.control, args.seed,
                           _parse_ids(args.control_from) if args.control_from else None)
        with open(args.out, "w") as f:
            json.dump(roles, f, indent=1)
        print(json.dumps({k: len(v) for k, v in roles["groups"].items()}), "->", args.out)
    else:
        pack = Pack(args.pack)
        counts = np.bincount(pack.labels, minlength=pack.num_classes)
        print(f"{args.pack}: {len(pack)} images, {pack.num_classes} classes ({int((counts > 0).sum())} present, "
              f"{int(counts.min())}-{int(counts.max())} per class), {pack.meta['bytes'] / 1e9:.2f} GB, "
              f"short side <= {pack.meta['short_side']}")


if __name__ == "__main__":
    main()
