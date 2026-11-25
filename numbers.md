# ViT study: end-of-training numbers

Written by `make_figs.py` from `data/vit_page_data.json`. Accuracy in %, means over five seeds;
differences are seed by seed with 95% Student-t intervals.

| Measure | AdamW 1e-3 | Muon 3e-3 | SGD 1.0 | Muon − AdamW | SGD − AdamW |
|---|---|---|---|---|---|
| Trained classes, validation top-1 | 71.44 | 73.62 | 69.61 | +2.17 ± 0.38 | -1.83 ± 0.65 |
| Rarest fifth of trained classes | 34.89 | 42.43 | 30.82 | +7.54 ± 1.70 | -4.07 ± 2.23 |
| Commonest fifth of trained classes | 79.18 | 80.29 | 77.47 | +1.11 ± 0.48 | -1.71 ± 1.32 |
| Never-seen, encoder output, plain | 46.73 | 46.89 | 45.91 | +0.16 ± 0.59 | -0.82 ± 0.48 |
| Never-seen, encoder output, equalized | 47.42 | 49.03 | 46.76 | +1.61 ± 0.52 | -0.66 ± 0.75 |
| Never-seen, last hidden layer, plain | 44.90 | 46.42 | 45.52 | +1.52 ± 0.58 | +0.62 ± 0.41 |
| Never-seen, last hidden layer, equalized | 44.51 | 48.21 | 44.22 | +3.70 ± 0.48 | -0.29 ± 0.72 |

Cost of the last hidden layer (last hidden layer minus encoder output, within each run):

| Reading | AdamW | Muon | SGD |
|---|---|---|---|
| plain | -1.83 ± 0.20 | -0.47 ± 0.16 | -0.39 ± 0.20 |
| equalized | -2.91 ± 0.46 | -0.82 ± 0.23 | -2.55 ± 0.40 |

Model built without the last hidden layer (five seeds each; read at the encoder output):

| Measure | AdamW 1e-3 | Muon 3e-3 | Muon − AdamW |
|---|---|---|---|
| Trained classes, validation top-1 | 72.17 | 73.74 | +1.57 ± 0.47 |
| Rarest fifth of trained classes | 40.18 | 44.13 | +3.95 ± 3.35 |
| Never-seen, plain | 47.94 | 48.64 | +0.70 ± 0.58 |
| Never-seen, equalized | 48.29 | 49.19 | +0.90 ± 0.61 |

The 2x2 at learning rate 1e-3 (three seeds per cell): gain from Muon instead of AdamW on

| Measure | the 12 blocks | the last hidden layer's matrix |
|---|---|---|
| Trained classes, validation top-1 | +1.70 ± 0.18 | +0.29 ± 0.28 |
| Never-seen, last hidden layer, plain | +0.57 ± 1.01 | +0.82 ± 0.23 |
| Never-seen, last hidden layer, equalized | +0.88 ± 1.00 | +1.68 ± 0.52 |

During training (last hidden layer, equalized, means of five seeds):

- untrained model: 19.2%; AdamW after 0.5% of training: 16.3%, after 1%: 16.4%
- Muon's lead over AdamW is largest after 2% of training (7.9 points) and 3.7 at the end
- saturated units after 0.5%: AdamW 97.6%, Muon 40.4%, SGD 31.6%
- stable rank of the layer's matrix: start 98; AdamW lowest 10.3, end 63; Muon end 120; SGD end 76
