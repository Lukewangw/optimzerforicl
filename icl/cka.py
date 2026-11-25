"""How alike the layers of several checkpoints are: linear CKA between the features probe.py kept (--keep).

    python -m icl.cka --features DIR/*.npz --out cka.json

For two sets of features of the same images, X (M, D1) and Y (M, D2), each with its column means removed, linear
CKA (Kornblith et al. 2019, arXiv 1905.00414) is ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F): 1 when one is the other
up to a rotation and one common scale, and small when they share nothing (about D / (M + D) for unrelated features
with D directions each, so 0.09 with 4,000 images and 384 directions, not 0).  The output has, for every layer, the
matrix of that number between every two files, and for every file the matrix between its own layers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    return float(cka_matrix([x, y])[0, 1])


def cka_matrix(feats: list[np.ndarray]) -> np.ndarray:
    """Linear CKA between every two of several feature sets of the same images."""
    xs = [x.astype(np.float64) - x.astype(np.float64).mean(0) for x in feats]
    own = [np.linalg.norm(x.T @ x) for x in xs]
    m = np.eye(len(xs))
    for i in range(len(xs)):
        if not own[i] > 0:
            m[i, :] = m[:, i] = np.nan
            continue
        for j in range(i + 1, len(xs)):
            m[i, j] = m[j, i] = np.linalg.norm(xs[i].T @ xs[j]) ** 2 / (own[i] * own[j]) if own[j] > 0 else np.nan
    return m


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", required=True, help="files written by probe.py --keep, or a folder of them")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    files = sorted(f for a in args.features for f in (glob.glob(os.path.join(a, "*.npz")) if os.path.isdir(a) else [a]))
    if len(files) < 2:
        raise SystemExit("cka: needs the kept features of two checkpoints or more")
    names = [os.path.basename(f)[:-4] for f in files]
    data = [np.load(f) for f in files]
    layers = [str(n) for n in data[0]["layers"]]
    for name, d in zip(names, data):
        if [str(n) for n in d["layers"]] != layers or not np.array_equal(d["rows"], data[0]["rows"]):
            raise SystemExit(f"cka: {name} holds other layers or other images than {names[0]}")
    out = {"files": names, "layers": layers, "images": int(len(data[0]["rows"])), "between": {}, "within": {}}
    for layer in layers:
        m = cka_matrix([d[layer] for d in data])
        out["between"][layer] = m.tolist()
        print(f"cka: {layer}: lowest {np.nanmin(m):.3f}", flush=True)
    for name, d in zip(names, data):
        out["within"][name] = cka_matrix([d[layer] for layer in layers]).tolist()
    with open(args.out + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(args.out + ".tmp", args.out)
    print(f"cka: {len(files)} files, {len(layers)} layers -> {args.out}")


if __name__ == "__main__":
    main()
