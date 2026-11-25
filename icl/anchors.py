"""Is a test pack numbered the way we think?  A check that needs a trained checkpoint and no eyes.

The dog-breed and cat questions are asked of the ImageNet validation images by class number ("151-268 are dogs").
Their numbering is assumed to be the standard one (classes sorted by WordNet id).  To check it, take classes our
model was trained on that have a known ImageNet counterpart ("Banana" <-> 954, "Cat" <-> 281-285) and let the
trained classifier label the test images of those numbers.  With the standard numbering it calls most of 954's
images "Banana"; with any other numbering it does so for almost none.

    python -m icl.anchors --ckpt RUN/ckpt-XXXXXXX.pt --roles roles.json --trained PACK --tests PACK
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import torch

from . import aug, features
from .data import Pack, Roles
from .train import autocast

# Objects365 class -> ImageNet-1k class numbers (standard order) showing the same thing.
IMAGENET_OF = {
    "Banana": [954], "Lemon": [951], "Orange/Tangerine": [950], "Pineapple": [953], "Broccoli": [937],
    "Cucumber": [943], "Mushroom": [947], "Hot dog": [934], "Hamburger": [933], "Cat": [281, 282, 283, 284, 285],
    "Lion": [291], "Elephant": [385, 386, 101], "Bear": [294, 295, 296, 297], "Horse": [339], "Sheep": [348, 349],
    "Cow": [345, 346, 347], "Camel": [354], "Penguin": [145], "Parrot": [87, 88, 89, 90], "Goose": [99],
    "Swan": [100], "Butterfly": [321, 322, 323, 324, 325, 326], "Jellyfish": [107], "Crab": [118, 119, 120, 121],
    "Lobster": [122, 123], "Goldfish": [1], "Airplane": [404], "Ambulance": [407], "Fire Truck": [555],
    "Backpack": [414], "Balloon": [417], "Basketball": [430], "Soccer": [805], "Volleyball": [890],
    "Golf Ball": [574], "Baseball": [429], "Cell Phone": [487], "Keyboard": [508], "Laptop": [620], "Mouse": [673],
    "Microwave": [651], "Refrigerator": [760], "Toaster": [859], "Toilet": [861], "Umbrella": [879], "Vase": [883],
    "Violin": [889], "Cello": [486], "Piano": [579], "Guitar": [402, 546], "Saxophone": [776], "Trombone": [875],
    "Drum": [541], "Parking meter": [704], "Traffic Light": [920], "Stuffed Toy": [850], "Remote": [761],
    "Sandals": [774], "Tea pot": [849], "Pickup Truck": [717], "Sports Car": [817], "Train": [466, 820],
    "Bus": [779, 874], "Dining Table": [532], "Toilet Paper": [999],
}


@torch.no_grad()
def predict(model, images_u8: torch.Tensor, device, *, amp: bool, batch_size: int = 500, topk: int = 5) -> np.ndarray:
    """The model's `topk` class guesses per image, best first: (M, topk) of positions among the trained classes."""
    model.eval()
    out = []
    for i in range(0, len(images_u8), batch_size):
        x = aug.to_model_input(images_u8[i:i + batch_size].to(device, non_blocking=True))
        with autocast(amp):
            logits = model(x)
        out.append(logits.float().topk(min(topk, logits.shape[1]), dim=1).indices.cpu())
    return torch.cat(out).numpy()


def check(model, roles: Roles, trained_names: list[str], tests: Pack, mapping: dict[str, list[int]], device, *,
          amp: bool, workers: int = 0) -> list[dict]:
    """One row per anchor class that the model was trained on: how its test images are labelled."""
    base_name = [trained_names[c] for c in roles.base]                      # position in the logits -> class name
    rows = []
    usable = {n: ids for n, ids in mapping.items() if n in base_name and all(i < tests.num_classes for i in ids)}
    wanted = sorted({i for ids in usable.values() for i in ids})
    images, labels = features.decode_pool(tests, Roles(tests.num_classes, {"picked": wanted}), "picked",
                                          image_size=model.image_size, workers=workers)
    guesses = predict(model, images, device, amp=amp)
    number = np.asarray(wanted)[labels]                                      # the test pack's own class number
    for name, ids in usable.items():
        mine = np.isin(number, ids)
        g = guesses[mine]
        target = base_name.index(name)
        top = collections.Counter(base_name[j] for j in g[:, 0]).most_common(3)
        rows.append({"anchor": name, "test_classes": ids, "images": int(mine.sum()),
                     "top1": float((g[:, 0] == target).mean()), "top5": float((g == target).any(1).mean()),
                     "called": top, "hit": top[0][0] == name})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--trained", required=True, help="a pack with the training classes (for their names)")
    ap.add_argument("--tests", required=True, help="the pack whose numbering is checked")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ck = features.load_model(args.ckpt, device)
    trained = Pack(args.trained)
    rows = check(model, Roles.load(args.roles, trained.num_classes), trained.classes, Pack(args.tests), IMAGENET_OF,
                 device, amp=device.type == "cuda", workers=args.workers)
    print(f"anchors: checkpoint step {ck['step']} (val top-1 {ck['val_top1']:.3f}), {len(rows)} anchor classes")
    for r in sorted(rows, key=lambda r: -r["top1"]):
        print(f"  {r['anchor']:18s} <- test classes {r['test_classes']}: {r['images']:4d} images, called it {r['top1']:.0%} "
              f"(in the best five {r['top5']:.0%}); most often called {r['called']}")
    hits = sum(r["hit"] for r in rows)
    print(f"anchors: {hits} of {len(rows)} anchors are the most frequent label of their test classes; "
          f"median share {np.median([r['top1'] for r in rows]):.0%}")
    return 0 if hits >= 0.5 * len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
