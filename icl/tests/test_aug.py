import numpy as np
import pytest
import torch

from conftest import jpeg_bytes, synth_image
from icl import aug


def picture(seed=0, size=224):
    return np.random.default_rng(seed).integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_inception_crop_stays_inside_the_recipes_bounds():
    rng = np.random.default_rng(0)
    w, h = 360, 288
    fractions = []
    for _ in range(3000):
        x, y, cw, ch = aug.inception_crop_box(w, h, rng)
        assert 0 <= x and 0 <= y and x + cw <= w and y + ch <= h and cw > 0 and ch > 0
        frac = cw * ch / (w * h)
        assert 0.05 <= frac <= 1.0
        assert 0.75 - 0.01 <= cw / ch <= 1.33 + 0.01                      # the ratio is rounded to whole pixels
        fractions.append(frac)
    # height, not area, is drawn uniformly, so small crops are common: the median area is well under one half
    assert 0.15 < float(np.median(fractions)) < 0.45 and max(fractions) > 0.8 and min(fractions) < 0.07


def test_inception_crop_falls_back_to_the_whole_image():
    rng = np.random.default_rng(0)
    assert aug.inception_crop_box(10, 400, rng, area=(0.9, 1.0)) == (0, 0, 10, 400)


def test_every_randaugment_op_keeps_shape_and_type():
    rng = np.random.default_rng(1)
    a = picture()
    assert len(aug.RANDAUG_OPS) == 16
    for name in aug.RANDAUG_OPS:
        out = aug.apply_op(a.copy(), name, 10.0, rng)
        assert out.shape == a.shape and out.dtype == np.uint8, name


def test_randaugment_ops_do_what_the_source_does_at_level_10():
    rng = np.random.default_rng(2)
    a = picture(3)
    assert np.array_equal(aug.apply_op(a, "Solarize", 10.0, rng), a)                 # threshold 256
    assert np.array_equal(aug.apply_op(a, "Invert", 10.0, rng), 255 - a)
    assert np.array_equal(aug.apply_op(a, "Posterize", 10.0, rng), a & 0xF0)
    assert np.array_equal(aug.apply_op(a, "Brightness", 10.0, rng), np.clip(a.astype(np.float32) * 1.9, 0, 255).astype(np.uint8))
    want = np.clip(196.0 + 1.9 * (a.astype(np.float32) - 196.0), 0, 255).astype(np.uint8)
    assert np.array_equal(aug.apply_op(a, "Contrast", 10.0, rng), want)              # the source's constant "mean"
    sa = aug.apply_op(a, "SolarizeAdd", 10.0, rng)
    assert np.array_equal(sa[a >= 128], a[a >= 128]) and np.array_equal(sa[a < 128], a[a < 128] + 110)
    cut = aug.apply_op(a, "Cutout", 10.0, rng)
    changed = np.any(cut != a, axis=2)
    ys, xs = np.nonzero(changed)
    assert 0 < changed.sum() <= 80 * 80 and np.all(cut[changed] == 128)
    assert ys.max() - ys.min() < 80 and xs.max() - xs.min() < 80
    for name in ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"):
        out = aug.apply_op(a, name, 10.0, rng)
        assert np.all(out == 128, axis=2).sum() > 500, name                             # gray fill where empty
    tx = aug.apply_op(a, "TranslateX", 10.0, np.random.default_rng(0))
    assert np.array_equal(tx[:, 100:], a[:, :-100]) or np.array_equal(tx[:, :-100], a[:, 100:])
    flat = np.full((224, 224, 3), 77, dtype=np.uint8)
    assert np.array_equal(aug.apply_op(flat, "AutoContrast", 10.0, rng), flat)
    ramp = np.repeat(np.linspace(50, 150, 224).astype(np.uint8)[None, :, None], 224, 0).repeat(3, 2)
    ac = aug.apply_op(ramp, "AutoContrast", 10.0, rng)
    assert ac.min() == 0 and ac.max() == 255
    eq = aug.apply_op(ramp, "Equalize", 10.0, rng)
    assert eq.min() == 0 and eq.max() >= 250
    sharp = aug.apply_op(a, "Sharpness", 10.0, rng)
    assert np.array_equal(sharp[0], a[0]) and np.array_equal(sharp[:, -1], a[:, -1])    # the border is untouched
    gray = np.repeat(a[..., :1], 3, axis=2)
    assert np.abs(aug.apply_op(gray, "Color", 10.0, rng).astype(int) - gray).max() <= 2  # nothing to saturate


def test_randaugment_draws_ops_uniformly():
    rng = np.random.default_rng(5)
    counts = np.bincount([int(rng.integers(0, 16)) for _ in range(16000)], minlength=16)
    assert counts.min() > 850 and counts.max() < 1150
    a = picture(4, 64)
    assert aug.randaugment(a, 0, 10, rng) is a
    assert aug.randaugment(a, 2, 10, np.random.default_rng(9)).shape == a.shape


def test_train_transform_is_a_function_of_seed_epoch_and_index():
    jpeg = jpeg_bytes(synth_image(1, 0, size=(360, 288)))
    t = aug.TrainTransform(224)
    a = t(jpeg, aug.sample_rng(3, 0, 17))
    assert a.shape == (224, 224, 3) and a.dtype == np.uint8 and a.flags["C_CONTIGUOUS"]
    assert np.array_equal(a, t(jpeg, aug.sample_rng(3, 0, 17)))
    others = [t(jpeg, aug.sample_rng(*k)) for k in ((3, 1, 17), (3, 0, 18), (4, 0, 17))]
    assert all(not np.array_equal(a, o) for o in others)
    plain = aug.TrainTransform(64, randaug_layers=0)
    assert plain(jpeg, aug.sample_rng(0, 0, 0)).shape == (64, 64, 3)


def test_eval_transform_resizes_the_short_side_then_crops_the_centre():
    im = synth_image(2, 0, size=(360, 288))
    jpeg = jpeg_bytes(im)
    a = aug.EvalTransform(224)(jpeg)
    assert a.shape == (224, 224, 3) and np.array_equal(a, aug.EvalTransform(224)(jpeg))
    t = aug.EvalTransform(224)
    assert t.resize == 256
    assert aug.EvalTransform(32).resize == 37
    # the same picture twice as large gives nearly the same result: the short side is what is fixed
    big = jpeg_bytes(im.resize((720, 576)), quality=95)
    assert np.abs(a.astype(int) - t(big).astype(int)).mean() < 12
    tall = aug.EvalTransform(224)(jpeg_bytes(synth_image(2, 1, size=(100, 500))))
    assert tall.shape == (224, 224, 3)


def test_value_range_and_mixup():
    x = torch.tensor([0, 255, 128], dtype=torch.uint8).view(1, 1, 3, 1).repeat(2, 4, 1, 3)
    y = aug.to_model_input(x)
    assert y.shape == (2, 3, 4, 3) and y.min() == -1.0 and y.max() == 1.0
    cs = [aug.mixup_coefficient(0, s, 0.2) for s in range(2000)]
    assert all(0.5 <= c <= 1.0 for c in cs) and np.mean(np.array(cs) > 0.9) > 0.5
    assert cs[7] == aug.mixup_coefficient(0, 7, 0.2) != aug.mixup_coefficient(1, 7, 0.2)
    assert aug.mixup_coefficient(0, 7, 0.0) == 1.0
    img, lab = torch.arange(12.0).view(4, 3, 1, 1), torch.tensor([0, 1, 2, 3])
    mixed, y1, y2 = aug.mixup(img, lab, 0.75)
    torch.testing.assert_close(mixed, 0.75 * img + 0.25 * img[[3, 0, 1, 2]])
    assert y1.tolist() == [0, 1, 2, 3] and y2.tolist() == [3, 0, 1, 2]
    same, y1, y2 = aug.mixup(img, lab, 1.0)
    assert same is img and y1 is lab and y2 is lab
