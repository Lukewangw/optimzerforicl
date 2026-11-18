"""Frozen features of a checkpoint: the pooled encoder output (`gap`) and what the head reads (`pre_logits`).

    python -m icl.features --ckpt RUN/ckpt-0001234.pt --pack PACK --roles roles.json --select cats_all --out f.npz

`fewshot.py` uses the functions here directly and does not write feature files.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import aug
from .data import Pack, Roles, eval_loader
from .train import autocast
from .vit import ViT


def load_model(ckpt_path: str, device) -> tuple[ViT, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = ViT(**ck["model_kwargs"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def decode_pool(pack: Pack, roles: Roles, select: str, *, image_size: int = 224, per_class: int = 0,
                workers: int = 0, batch_size: int = 256) -> tuple[torch.Tensor, np.ndarray]:
    """The selected images after the evaluation transform, as one uint8 tensor (M, H, W, 3), and their labels
    under `select`.  Decoding once and reusing the tensor for every checkpoint is what makes a sweep cheap."""
    loader = eval_loader(pack, roles, select, batch_size=batch_size, workers=workers, image_size=image_size,
                         per_class=per_class)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    if not xs:
        return torch.zeros((0, image_size, image_size, 3), dtype=torch.uint8), np.zeros(0, dtype=np.int64)
    return torch.cat(xs), torch.cat(ys).numpy()


@torch.no_grad()
def extract(model: ViT, images_u8: torch.Tensor, device, *, batch_size: int = 500, amp: bool = False
            ) -> dict[str, np.ndarray]:
    """{"gap": (M, D), "pre_logits": (M, D)} as float32."""
    model.eval()
    gap, pre = [], []
    for i in range(0, len(images_u8), batch_size):
        x = aug.to_model_input(images_u8[i:i + batch_size].to(device, non_blocking=True))
        with autocast(amp):
            g, p = model.both_features(x)
        gap.append(g.float().cpu())
        pre.append(p.float().cpu())
    d = model.width
    return {"gap": torch.cat(gap).numpy() if gap else np.zeros((0, d), np.float32),
            "pre_logits": torch.cat(pre).numpy() if pre else np.zeros((0, d), np.float32)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--roles")
    ap.add_argument("--select", default="all", help="all, base, or a group of the roles file")
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ck = load_model(args.ckpt, device)
    pack = Pack(args.pack)
    roles = Roles.load(args.roles, pack.num_classes)
    images, labels = decode_pool(pack, roles, args.select, image_size=model.image_size, per_class=args.per_class,
                                 workers=args.workers)
    feats = extract(model, images, device, amp=device.type == "cuda")
    np.savez(args.out, labels=labels, step=ck["step"], **feats)
    print(f"features: {len(labels)} images, step {ck['step']} -> {args.out}")


if __name__ == "__main__":
    main()
