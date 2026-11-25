# The training recipe and where each part comes from

The AdamW side of the comparison is the published plain-ViT baseline: Beyer, Zhai, Kolesnikov, "Better plain ViT
baselines for ImageNet-1k" (arXiv 2205.01580), which trains ViT-S/16 on ImageNet-1k to 76.5 % top-1 in 90 epochs
(78.5 % in 150, 80.0 % in 300). Its code is big_vision, config `big_vision/configs/vit_s16_i1k.py`.

Read on 2026-11-15, as source text, nothing run:

| what | revision |
|---|---|
| `google-research/big_vision` | `0127fb6b337e` (main, 2025-05-19): `configs/vit_s16_i1k.py`, `models/vit.py`, `optax.py`, `utils.py`, `train.py`, `pp/ops_image.py`, `pp/ops_general.py`, `pp/archive/randaug.py`, `pp/archive/autoaugment.py` (the copy `archive.randaug` imports; it differs from `pp/autoaugment.py` in the copyright line only) |
| `google/flax` | `204a819c2eb7` (main): `linen/attention.py`, `linen/linear.py`, `linen/normalization.py`, `linen/activation.py` |
| `jax-ml/jax` | `7ea7895a4468` (main): `_src/nn/initializers.py`, `_src/nn/functions.py` |
| `google-deepmind/optax` | `a0355e265ebd` (main): `transforms/_adding.py`, `_src/transform.py` |
| `tensorflow/tensorflow` | `1affc22e9a37` (master): `core/kernels/image/sample_distorted_bounding_box_op.cc`, `core/ops/image_ops.cc` |
| the paper | the ar5iv rendering of arXiv 2205.01580, read through a summarising fetch, so its quotes are second-hand |

"Confirmed" below means the value was read in the file named. The libraries were read at today's head, not at the
versions the paper's runs used in 2022; where a default could have changed since, that is the remaining doubt.

## The weight-decay conversion (the detail most easily got wrong)

`big_vision/optax.py`, `make()`:

```python
  lr_mult_txs = [optax.scale(config.lr)]                               # line 115
  ...
    wd_mults = config.get("wd_mults", [(".*/kernel$", 1.0)])           # line 133
    ...
        optax.add_decayed_weights(config.wd * mult, mask)              # line 136
  ...
  return optax.chain(                                                  # lines 143-149
      grad_clip_norm_tx,
      *opt_txs,
      *lr_mult_txs,
      *weight_decay_txs,
      *schedule_txs,
      optax.scale(-1.0)), schedule_fns
```

The learning rate multiplies the Adam direction *before* the decay term is added (`add_decayed_weights` returns
`g + weight_decay * p`, optax `transforms/_adding.py`), and the schedule, built with `base=1.0`, multiplies the
sum. One step is therefore

    w  <-  w - s_t * (lr * adam_direction + wd * w)            s_t: the schedule, between 0 and 1

so the decay per step is `s_t * wd`: it follows the schedule and is **not** multiplied by `lr`.
`torch.optim.AdamW` does `w <- w * (1 - lr_t * weight_decay) - lr_t * adam_direction` with `lr_t = lr * s_t`, so
its decay per step is `s_t * lr * weight_decay`. Equal when `weight_decay = wd / lr`:

    big_vision  lr = 0.001, wd = 0.0001      ==      torch.optim.AdamW(lr=1e-3, weight_decay=0.1)

with the same relative schedule on the learning rate. This holds only at `lr = 1e-3`: when a sweep changes
AdamW's learning rate, keeping the source's decay means setting `--wd` to `1e-4 / lr`; leaving `--wd 0.1` changes
the decay with the learning rate. That choice belongs to the sweep and has to be stated with it.

Decayed parameters: the default mask `.*/kernel$` matches every `kernel` (patch embedding, the four attention
matrices, both MLP matrices, `pre_logits`, the head) and nothing else. Biases and LayerNorm `scale`/`bias` are not
decayed. In `optim.py`: parameters with two or more dimensions.

## Table

| item | value | source | status |
|---|---|---|---|
| model | ViT-S/16: width 384, depth 12, MLP 1536, 6 heads, patch 16 | `configs/vit_s16_i1k.py` (`variant='S/16'`), `models/vit.py` `decode_variant` | confirmed |
| pooling | global average of the tokens after the final LayerNorm, no class token | config `pool_type='gap'`; `vit.py` `jnp.mean(x, axis=1)` after `Encoder` (which ends in `encoder_norm`) | confirmed |
| position embedding | fixed 2D sin-cos, temperature 10 000, added to the patch embeddings | config `posemb='sincos2d'`; `vit.py` `posemb_sincos_2d` | confirmed |
| block | pre-LN: `x + attn(LN(x))`, `x + mlp(LN(x))`; no dropout, no stochastic depth | `vit.py` `Encoder1DBlock` (`dropout=0.0` default; the config sets none) | confirmed |
| LayerNorm epsilon | 1e-6 | flax `linen/normalization.py` (`epsilon: float = 1e-6`) | confirmed |
| activation | GELU, tanh approximation | `vit.py` `nn.gelu(x)`; jax `gelu(x, approximate=True)` default | confirmed |
| head | `pre_logits` (Linear width->width, then tanh), then a linear layer | config `rep_size=True`; `vit.py` `nn.tanh(hid(x))` | confirmed |
| head, alternative | a single linear layer scores 76.7 / 78.6 / 79.8 against 76.5 / 78.5 / 80.0 (90 / 150 / 300 epochs): "no significant difference" | paper, Table 1 | second-hand (summarised fetch) |
| init: attention q, k, v, out | Xavier uniform on each d x d matrix, bias 0 | `vit.py` `kernel_init=xavier_uniform()`; flax `MultiHeadDotProductAttention` builds four `DenseGeneral`, whose `kernel_init_wrap` flattens the kernel to (in, out) before initialising; `bias_init = zeros` | confirmed |
| init: MLP | Xavier uniform, bias normal(std 1e-6) | `vit.py` `MlpBlock` | confirmed |
| init: patch embedding, `pre_logits` | LeCun normal: variance 1/fan_in, truncated at +-2 sd (std divided by 0.87962566), bias 0 | flax `default_kernel_init = lecun_normal()` for `Conv` and `Dense`; jax `variance_scaling(1.0, "fan_in", "truncated_normal")` | confirmed |
| init: head | weight 0, bias 0 | `vit.py` `head_zeroinit=True` -> `kernel_init=zeros`; flax `Dense` bias default | confirmed |
| loss | softmax cross-entropy against (mixed) one-hot labels, mean over the batch; no label smoothing | config `loss='softmax_xent'`; `utils.py` `softmax_xent`; `pp` `onehot(1000)` | confirmed |
| mixup | every step: one `a ~ Beta(0.2, 0.2)`, `a = max(a, 1 - a)`, each example mixed with its neighbour (`roll` by 1) within a device's part of the batch; images and labels | config `mixup=dict(p=0.2)`; `utils.py` `get_mixup` (lines 1149-1150); `train.py` `shard_map(get_mixup(rng, p))` | confirmed |
| mixup, the paper's wording | "level 10 and probability 0.2" for RandAugment and Mixup | paper | second-hand; in the code `p` is the Beta parameter and mixup is applied at every step, there is no probability gate |
| training crop | Inception crop: area 5-100 % of the image, aspect ratio uniform in [0.75, 1.33], up to 100 attempts then the whole image; resized to 224 x 224 bilinear | `pp/ops_image.py` `get_decode_jpeg_and_inception_crop` (`area_min=5, area_max=100, ratio_min=0.75, ratio_max=1.33`); TensorFlow `GenerateRandomCrop`, `max_attempts: int = 100` | confirmed |
| crop sampling | TensorFlow draws the *height* uniformly between the bounds the area range allows (not the area), then the offset | `sample_distorted_bounding_box_op.cc` | confirmed; ported line by line in `aug.py`, and it differs from torchvision's `RandomResizedCrop` (uniform area, log-uniform ratio, 10 attempts, centre-crop fallback) |
| resize filter | bilinear, `antialias=False` | `pp/ops_image.py` `get_resize` default | confirmed in source; **ours differs**: Pillow's bilinear resize averages when it scales down (assumed harmless) |
| flip | left-right with probability 0.5 | `pp/ops_image.py` `flip_lr` | confirmed |
| RandAugment | `randaug(2, 10)`: 2 ops in sequence, each drawn uniformly from 16 and always applied, magnitude 10 of 10 | config; `pp/archive/randaug.py`; `pp/archive/autoaugment.py` `distort_image_with_randaugment` | confirmed |
| RandAugment ops | AutoContrast, Equalize, Invert, Rotate (+-30 deg), Posterize (4 bits), Solarize (threshold 256: no change), Color / Contrast / Brightness / Sharpness (factor 1.9), ShearX / ShearY (+-0.3), TranslateX / TranslateY (+-100 px), Cutout (80 px square), SolarizeAdd (+110 below 128); fill value 128; nearest-neighbour geometry | `autoaugment.py` `level_to_arg`, `NAME_TO_FUNC`, `cutout_const=40, translate_const=100`, `replace_value=[128]*3` | confirmed; **not** torchvision's `RandAugment` (14 ops, other magnitudes) |
| RandAugment, a quirk kept | Contrast blends with a constant gray of (pixels / 256) = 196 for a 224 px image, because the source divides the histogram's sum, not its mean, by 256 | `autoaugment.py` `contrast` | confirmed, and reproduced as it is |
| order | crop and resize, flip, RandAugment (on uint8), then scale to [-1, 1] | config `pp` string; `pp/ops_general.py` `value_range(-1, 1)` | confirmed |
| evaluation | short side to 256 ("area" filter), central 224 crop, scale to [-1, 1] | config `pp_eval`; `pp/ops_image.py` `resize_small`, `central_crop` | confirmed; ours uses Pillow's box filter for "area" |
| batch size | 1024 | config `input.batch_size` | confirmed |
| length | 90 epochs; steps = epochs x images / batch size over a repeated, shuffled stream | config `total_epochs`; `train.py` `u.steps("total", ...)` | confirmed |
| training set | the first 99 % of ImageNet train (1 % kept as "minival") | config `split='train[:99%]'` | confirmed; **ours differs**: every training image of the base classes is used |
| schedule | `min(1, step / 10000)` x `0.5 (1 + cos(pi * progress))`, `progress = (step - 10000) / (total - 10000)` clipped to [0, 1]; the first update is taken at scale 0 | config `schedule=dict(warmup_steps=10_000, decay_type='cosine')`; `utils.py` `create_learning_rate_schedule` | confirmed |
| gradient clipping | global norm 1.0, before Adam | config `grad_clip_norm = 1.0`; `optax.py` | confirmed |
| optimizer | Adam, b1 0.9, b2 0.999, eps 1e-8, bias-corrected | config `optax_name='scale_by_adam'`; optax `scale_by_adam` defaults | confirmed (the config's `mu_dtype='bfloat16'` stores the first moment in bf16; ours keeps float32) |
| precision | the model computes in float32 | `models/vit.py` `dtype_mm: str = "float32"`; the config sets none | confirmed in source; **ours differs** on CUDA: bf16 autocast unless `--no-amp` |
| learning rate, decay | `lr = 0.001`, `wd = 0.0001` in big_vision's terms = AdamW `lr=1e-3, weight_decay=0.1` in PyTorch's | config; conversion above | confirmed |
| result | 76.5 % top-1 on ImageNet val at 90 epochs, 6h30 on a TPUv3-8 | paper; the config's docstring says "6-7h ... 76.5%" | docstring confirmed, paper second-hand |

## Where this code departs from the source, in one place

- The attention holds four separate d x d `Linear` layers. Same function and same initial distribution as flax's;
  done so that Muon treats q, k and v as separate matrices.
- Images are packed with the short side at most 288 px (`pack.py`), so crops are cut from a smaller picture than
  the original JPEG. Not measured.
- Pillow resizing (antialiased bilinear for the crop, box filter for evaluation) instead of TensorFlow's.
- All base-class training images are used, not 99 % of them.
- Adam's first moment is float32, not bf16.
- On CUDA the forward pass runs under bf16 autocast and TF32 products are allowed (`train.py`; `--no-amp` turns
  the autocast off). The source's model computes in float32 (`models/vit.py`, `dtype_mm: str = "float32"`, which
  the config does not change). Chosen for speed on the A100; its effect on accuracy is not measured.
- On more than one GPU the batch is split over processes and mixup pairs neighbours within each process's part, as
  big_vision does per device; the pairs therefore depend on the number of processes. Runs that are compared must
  use the same number.
- None of this has been run against big_vision itself, and no ImageNet run has been made with this code. The
  76.5 % is the source's number, not ours.

## Muon

`torch.optim.Muon` (PyTorch 2.14 on the cluster, 2.15 on the Mac; read from the installed 2.15 source):
momentum buffer `buf.lerp_(grad, 1 - momentum)`, Nesterov by default, five Newton-Schulz steps in bf16, then
`param.mul_(1 - lr * weight_decay)` and `param.add_(update, alpha=-adjusted_lr)`. It accepts 2D parameters only.
With `adjust_lr_fn="match_rms_adamw"` the step is scaled by `0.2 * sqrt(max(rows, cols))`; `"original"` scales by
`sqrt(max(1, rows / cols))`. Our reading, not the source's words: an orthogonalised m x n update has entries of
root-mean-square `1 / sqrt(max(m, n))`, so under `match_rms_adamw` the update's is about `0.2 * lr` and learning
rates on AdamW's scale (1e-3) are the sensible range, which is why it is the default here.

Nothing about Muon comes from the ViT recipe. What is set here is a choice, listed in the README: which matrices
Muon takes, that the rest keep the recipe's AdamW, that the same warm-up, cosine schedule, gradient clipping and
`weight_decay` are used, and the defaults `momentum 0.95`, `nesterov`, `ns_steps 5`, `match_rms_adamw`.

## What is and is not reproduced (2026-11-16)

The recipe's published result, 76.5 % top-1 after 90 epochs, is on ImageNet-1k. We do not train on ImageNet-1k
(its terms; see the README), so that number is **not reproduced**, and no accuracy on our data can stand in for it:
nobody has published this model on Objects365 cut-outs.

What is checked instead:

- the model has the size the published configuration implies: 22,122,472 parameters with a 1,000-way head (width
  384, depth 12, MLP 1536, patch 16, `rep_size=True`, no position or class-token parameters); 21,869,912 with our
  344 classes;
- every setting in the table above is read from big_vision's source at the commit named there, and the tests pin
  the schedule, the weight-decay rule, the crop, RandAugment and mixup;
- the baseline run uses the recipe as it stands (AdamW, 1e-3, decay 1e-4 in big_vision's terms), unchanged.

With ImageNet-1k cleared for use, the literal reproduction is one run of this code (about 12 GPU-hours).

The other optimizers get the recipe's schedule, warm-up, clipping and batch size, and three learning rates each
around a default: AdamW and Muon at 3e-4, 1e-3, 3e-3 with weight decay 1e-4 / lr (big_vision's absolute decay kept
fixed); SGD with Nesterov momentum 0.9 at 0.03, 0.1, 0.3 with weight decay 1e-5 / lr (the same shrink per step at
the peak, since momentum multiplies the step by ten).
SGD at 1.0 and AdamW at 2e-3 were added later, by the same rules.
