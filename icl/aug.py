"""The recipe's preprocessing, ported from big_vision's TensorFlow ops to numpy and Pillow (RECIPE.md lists each op
and where it comes from).

Training: `decode_jpeg_and_inception_crop(224) | flip_lr | randaug(2, 10)`; evaluation: `resize_small(256) |
central_crop(224)`.  Both return uint8 HWC arrays; scaling to [-1, 1] and mixup happen on the device.

Every random choice for a sample comes from a generator seeded by (data seed, epoch, sample index), so what a run
sees does not depend on the optimizer, the number of workers or the order in which workers finish.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

RANDAUG_OPS = ("AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "Color", "Contrast",
               "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateX", "TranslateY", "Cutout", "SolarizeAdd")
_MAX_LEVEL = 10.0
_REPLACE = (128, 128, 128)
_CUTOUT_CONST, _TRANSLATE_CONST = 40, 100


def sample_rng(data_seed: int, epoch: int, index: int) -> np.random.Generator:
    return np.random.default_rng([int(data_seed), int(epoch), int(index), 0xA06])


# ---- Inception crop: tf.image.sample_distorted_bounding_box, as called by big_vision ---------------------------

def _generate_random_crop(w: int, h: int, min_rel: float, max_rel: float, ratio: float, rng):
    """TensorFlow's `GenerateRandomCrop`: (x, y, crop_w, crop_h), or None when the draw does not fit."""
    min_area, max_area = min_rel * w * h, max_rel * w * h
    height = int(round((min_area / ratio) ** 0.5))
    max_height = int(round((max_area / ratio) ** 0.5))
    if int(round(max_height * ratio)) > w:
        max_height = int((w + 0.5 - 1e-7) / ratio)
        if int(round(max_height * ratio)) > w:
            max_height -= 1
    max_height = min(max_height, h)
    height = min(height, max_height)
    if height < max_height:
        height += int(rng.integers(0, max_height - height + 1))
    width = int(round(height * ratio))
    area = width * height
    if area < min_area:
        height += 1
        width = int(round(height * ratio))
        area = width * height
    if area > max_area:
        height -= 1
        width = int(round(height * ratio))
        area = width * height
    if area < min_area or area > max_area or width > w or height > h or width <= 0 or height <= 0:
        return None
    y = int(rng.integers(0, h - height)) if height < h else 0
    x = int(rng.integers(0, w - width)) if width < w else 0
    return x, y, width, height


def inception_crop_box(w: int, h: int, rng, area=(0.05, 1.0), ratio=(0.75, 1.33), attempts: int = 100):
    """Aspect ratio uniform in `ratio`, height uniform between the bounds `area` allows; the whole image after
    `attempts` failures (big_vision's defaults: area 5-100 %, ratio 0.75-1.33, TensorFlow's 100 attempts)."""
    for _ in range(attempts):
        box = _generate_random_crop(w, h, area[0], area[1], float(rng.uniform(ratio[0], ratio[1])), rng)
        if box is not None:
            return box
    return 0, 0, w, h


# ---- RandAugment: big_vision/pp/archive/autoaugment.py `distort_image_with_randaugment` ------------------------

def _blend(degenerate: np.ndarray, image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 0.0:
        return degenerate
    if factor == 1.0:
        return image
    a = degenerate.astype(np.float32)
    t = a + factor * (image.astype(np.float32) - a)
    if 0.0 < factor < 1.0:
        return t.astype(np.uint8)
    return np.clip(t, 0.0, 255.0).astype(np.uint8)


def _gray(image: np.ndarray) -> np.ndarray:
    """tf.image.rgb_to_grayscale on uint8: weights on the image in [0, 1], back to uint8 by floor(x * 255.5)."""
    g = (image.astype(np.float32) / 255.0) @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    return np.minimum(np.floor(g * 255.5), 255.0).astype(np.uint8)


def _autocontrast(image):
    out = np.empty_like(image)
    for c in range(3):
        ch = image[..., c]
        lo, hi = float(ch.min()), float(ch.max())
        if hi > lo:
            scale = 255.0 / (hi - lo)
            out[..., c] = np.clip(ch.astype(np.float32) * scale - lo * scale, 0.0, 255.0).astype(np.uint8)
        else:
            out[..., c] = ch
    return out


def _equalize(image):
    out = np.empty_like(image)
    for c in range(3):
        ch = image[..., c]
        histo = np.bincount(ch.ravel(), minlength=256)
        nonzero = histo[histo != 0]
        step = (int(nonzero.sum()) - int(nonzero[-1])) // 255
        if step == 0:
            out[..., c] = ch
        else:
            lut = (np.cumsum(histo) + step // 2) // step
            lut = np.clip(np.concatenate([[0], lut[:-1]]), 0, 255).astype(np.uint8)
            out[..., c] = lut[ch]
    return out


def _color(image, factor):
    return _blend(np.repeat(_gray(image)[..., None], 3, axis=2), image, factor)


def _contrast(image, factor):
    # The source takes the sum of the histogram, not its mean: the "mean" is (number of pixels) / 256, which is
    # 196 for a 224 px image whatever it shows.  Kept as it is, since the published numbers were trained with it.
    mean = min(image.shape[0] * image.shape[1] / 256.0, 255.0)
    return _blend(np.full_like(image, np.uint8(mean)), image, factor)


def _brightness(image, factor):
    return _blend(np.zeros_like(image), image, factor)


def _sharpness(image, factor):
    f = image.astype(np.float32)
    h, w = image.shape[:2]
    if h < 3 or w < 3:
        return image
    s = sum(f[dy:h - 2 + dy, dx:w - 2 + dx] for dy in range(3) for dx in range(3)) + 4.0 * f[1:-1, 1:-1]
    result = image.copy()
    result[1:-1, 1:-1] = np.clip(s / 13.0, 0.0, 255.0).astype(np.uint8)
    return _blend(result, image, factor)


def _affine(image, coeffs):
    im = Image.fromarray(image)
    return np.asarray(im.transform(im.size, Image.AFFINE, coeffs, resample=Image.NEAREST, fillcolor=_REPLACE))


def _rotate(image, degrees):
    return np.asarray(Image.fromarray(image).rotate(degrees, resample=Image.NEAREST, fillcolor=_REPLACE))


def _cutout(image, pad, rng):
    h, w = image.shape[:2]
    cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
    out = image.copy()
    out[max(0, cy - pad):min(h, cy + pad), max(0, cx - pad):min(w, cx + pad)] = _REPLACE
    return out


def _solarize(image, threshold):
    return np.where(image.astype(np.int32) < threshold, image, 255 - image).astype(np.uint8)


def _solarize_add(image, addition, threshold=128):
    added = np.clip(image.astype(np.int32) + addition, 0, 255).astype(np.uint8)
    return np.where(image < threshold, added, image)


def apply_op(image: np.ndarray, name: str, level: float, rng) -> np.ndarray:
    """One op at `level` (0-10) on a uint8 HWC array, with the source's level-to-argument rules."""
    frac = level / _MAX_LEVEL
    sign = 1.0 if rng.random() < 0.5 else -1.0                      # drawn for every op, used by the signed ones
    if name == "AutoContrast":
        return _autocontrast(image)
    if name == "Equalize":
        return _equalize(image)
    if name == "Invert":
        return 255 - image
    if name == "Rotate":
        return _rotate(image, sign * frac * 30.0)
    if name == "Posterize":
        shift = 8 - int(frac * 4)
        return (image >> shift) << shift
    if name == "Solarize":
        return _solarize(image, int(frac * 256))                   # at level 10 the threshold is 256: no change
    if name == "SolarizeAdd":
        return _solarize_add(image, int(frac * 110))
    if name == "Color":
        return _color(image, frac * 1.8 + 0.1)
    if name == "Contrast":
        return _contrast(image, frac * 1.8 + 0.1)
    if name == "Brightness":
        return _brightness(image, frac * 1.8 + 0.1)
    if name == "Sharpness":
        return _sharpness(image, frac * 1.8 + 0.1)
    if name == "ShearX":
        return _affine(image, (1.0, sign * frac * 0.3, 0.0, 0.0, 1.0, 0.0))
    if name == "ShearY":
        return _affine(image, (1.0, 0.0, 0.0, sign * frac * 0.3, 1.0, 0.0))
    if name == "TranslateX":
        return _affine(image, (1.0, 0.0, sign * frac * _TRANSLATE_CONST, 0.0, 1.0, 0.0))
    if name == "TranslateY":
        return _affine(image, (1.0, 0.0, 0.0, 0.0, 1.0, sign * frac * _TRANSLATE_CONST))
    if name == "Cutout":
        return _cutout(image, int(frac * _CUTOUT_CONST), rng)
    raise ValueError(name)


def randaugment(image: np.ndarray, layers: int, magnitude: float, rng) -> np.ndarray:
    """`layers` ops in sequence, each drawn uniformly from the 16 and always applied, all at `magnitude`."""
    for _ in range(layers):
        image = apply_op(image, RANDAUG_OPS[int(rng.integers(0, len(RANDAUG_OPS)))], float(magnitude), rng)
    return image


# ---- the two transforms -----------------------------------------------------------------------------------------

def decode(jpeg) -> Image.Image:
    im = Image.open(io.BytesIO(jpeg))
    return im if im.mode == "RGB" else im.convert("RGB")


class TrainTransform:
    def __init__(self, size: int = 224, randaug_layers: int = 2, randaug_magnitude: float = 10.0,
                 flip: bool = True):
        self.size, self.layers, self.magnitude, self.flip = size, randaug_layers, randaug_magnitude, flip

    def __call__(self, jpeg, rng) -> np.ndarray:
        im = decode(jpeg)
        x, y, w, h = inception_crop_box(im.width, im.height, rng)
        im = im.resize((self.size, self.size), Image.BILINEAR, box=(x, y, x + w, y + h))
        a = np.asarray(im)
        if self.flip and rng.random() < 0.5:
            a = a[:, ::-1]
        if self.layers:
            a = randaugment(np.ascontiguousarray(a), self.layers, self.magnitude, rng)
        return np.ascontiguousarray(a)


class EvalTransform:
    def __init__(self, size: int = 224, resize: int | None = None):
        self.size = size
        self.resize = resize if resize is not None else round(size * 256 / 224)

    def __call__(self, jpeg, rng=None) -> np.ndarray:
        im = decode(jpeg)
        f = self.resize / min(im.width, im.height)
        w, h = max(self.size, round(im.width * f)), max(self.size, round(im.height * f))
        if (w, h) != im.size:
            im = im.resize((w, h), Image.BOX)
        x, y = (w - self.size) // 2, (h - self.size) // 2
        return np.ascontiguousarray(np.asarray(im.crop((x, y, x + self.size, y + self.size))))


# ---- on the device ------------------------------------------------------------------------------------------------

def to_model_input(batch_u8):
    """uint8 (B, H, W, 3) on any device -> float (B, 3, H, W) in [-1, 1] (big_vision `value_range(-1, 1)`)."""
    return batch_u8.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def mixup_coefficient(data_seed: int, step: int, p: float) -> float:
    """big_vision `get_mixup`: one a ~ Beta(p, p) per step, folded to a >= 0.5.  p = 0 turns mixup off (a = 1)."""
    if not p:
        return 1.0
    a = float(np.random.default_rng([int(data_seed), int(step), 0x313]).beta(p, p))
    return max(a, 1.0 - a)


def mixup(images, labels, a: float):
    """Each example mixed with its neighbour in the batch (`roll` by one), as the source does per device.
    Returns the mixed images and the two label sets whose losses are weighted a and 1 - a."""
    if a >= 1.0:
        return images, labels, labels
    return a * images + (1.0 - a) * images.roll(1, 0), labels, labels.roll(1, 0)
