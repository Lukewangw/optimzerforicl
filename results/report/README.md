# Final results and the report's figures

This folder holds the final results of the study: 48 training runs, with five seeds of each main setting. It
contains the figures of the course report's ViT part (Section 6 and Appendix G), the script that draws them and
the numbers they are drawn from. The top-level README describes the setup and the first two rounds, which had
three seeds; where the numbers differ, the ones here are the ones to use. The report's first part, in-context
regression and ciphers in small sequence models, is at https://github.com/EshMis/optimizer-icl.

## What was added after the first two rounds

- Two more seeds of AdamW 1e-3, Muon 3e-3 and SGD 1.0, so each has five. Within a seed every optimizer starts
  from the same weights and sees the same images in the same order.
- A model built without the `pre_logits` layer, under AdamW and under Muon (five seeds each).
- A 2x2 at learning rate 1e-3: AdamW or Muon on the 12 blocks, AdamW or Muon on the `pre_logits` matrix (three
  seeds per cell).
- Single runs: Muon on the MLP matrices only and on the attention matrices only, and a change of optimizer after
  10% of training in each direction.
- More ways of reading the features: after every layer of the model, with whitened features, and with a ridge
  classifier trained on the examples.

About 690 A100 GPU-hours in all, including the learning-rate sweeps.

## Names used here and in the report

| In the code and the top-level README | In the report and the figures |
|---|---|
| `gap` | encoder output (mean over patches after block 12) |
| `pre_logits` | last hidden layer (a linear map and a tanh; what the classifier reads) |
| nearest class mean on the features as they are | plain reading |
| nearest class mean on whitened features | equalized reading |
| unseen pools | never-seen classes (mean of the 14 five-shot tests) |

## Results in brief

Accuracy in %, means over five seeds; differences are seed by seed with 95% Student-t intervals. The full table
is [numbers.md](numbers.md), written by the figure script.

| Measure | AdamW 1e-3 | Muon 3e-3 | SGD 1.0 | Muon − AdamW | SGD − AdamW |
|---|---|---|---|---|---|
| Trained classes, validation top-1 | 71.44 | 73.62 | 69.61 | +2.17 ± 0.38 | −1.83 ± 0.65 |
| Rarest fifth of trained classes | 34.89 | 42.43 | 30.82 | +7.54 ± 1.70 | −4.07 ± 2.23 |
| Never-seen, encoder output, plain | 46.73 | 46.89 | 45.91 | +0.16 ± 0.59 | −0.82 ± 0.48 |
| Never-seen, encoder output, equalized | 47.42 | 49.03 | 46.76 | +1.61 ± 0.52 | −0.66 ± 0.75 |
| Never-seen, last hidden layer, plain | 44.90 | 46.42 | 45.52 | +1.52 ± 0.58 | +0.62 ± 0.41 |
| Never-seen, last hidden layer, equalized | 44.51 | 48.21 | 44.22 | +3.70 ± 0.48 | −0.29 ± 0.72 |

1. **Half or more of Muon's lead on never-seen classes is made in one layer.** Crossing the last hidden layer
   costs AdamW 1.83 (plain) and 2.91 (equalized) points and Muon 0.47 and 0.82. The layer costs AdamW more by
   1.37 ± 0.15 and 2.09 ± 0.47 points, with the same sign in five of five seeds (exact permutation p = 0.008,
   the smallest five seeds allow). That is 90% (plain) and 56% (equalized) of Muon's lead where the classifier
   reads; the rest is already there at the encoder output.
2. **Under AdamW that layer saturates in the first 0.5% of training.** 97.6% of its units are pinned by the
   tanh, the stable rank of its matrix falls from 98 to 10, and few-shot accuracy read there falls below the
   untrained model's (16.3% against 19.2%) in five of five AdamW seeds, in none of the Muon or SGD seeds.
3. **Which matrices Muon has to train depends on the measure** (the 2x2 at learning rate 1e-3). On trained
   classes the blocks carry the gain (+1.70 ± 0.18; the last matrix +0.29 ± 0.28). On never-seen classes at the
   last hidden layer the layer's own matrix has the clear effect (+0.82 ± 0.23 plain, +1.68 ± 0.52 equalized);
   the blocks' effect is not established (+0.57 ± 1.01, +0.88 ± 1.00).
4. **In a model built without that layer Muon's lead is small.** +0.70 ± 0.58 (plain) and +0.90 ± 0.61
   (equalized), ahead in every seed.
5. **SGD does not settle whether few-shot accuracy follows accuracy on trained classes.** It is 1.8 points
   below AdamW on trained classes; on never-seen classes it is ahead or behind depending on the read-out. The
   three-seed statement in the top-level README, that SGD and AdamW cannot be told apart there, no longer holds.

Limits: one model, one dataset, and a few-shot test on frozen features, which is not in-context learning
inside the transformer. The control runs have three seeds, the optimizer swaps and some splits one run each.
Why the last hidden layer suffers under AdamW when the blocks do not is not explained.

## Figures

| File in `figs/` (.png and .pdf) | Report | What it shows |
|---|---|---|
| `vit_read_points` | Figure 5 | Never-seen classes at the end of training, seed by seed, at the encoder output, at the last hidden layer and in the model built without that layer; plain and equalized |
| `vit_training` | Figure 6 | The same test at the 17 checkpoints, with the share of saturated units and the stable rank of the last hidden layer's matrix |
| `vit_tests` | Figure G1 | Muon − AdamW for each of the 14 tests, at both read points, in both readings |
| `vit_depth` | Figure G2 | Muon − AdamW and SGD − AdamW read after every layer of the model |
| `vit_controls` | Figure G3 | The 2x2 at learning rate 1e-3: what Muon on the blocks adds and what Muon on the last matrix adds |
| `vit_updates` | Figure G4 | Share of each weight update in its strongest singular direction, for attention and MLP matrices |
| `vit_rare_classes` | Figure G5 | Trained-class accuracy by class frequency: Muon − AdamW and SGD − AdamW per fifth |

## Regenerating them

```bash
python results/report/make_figs.py     # writes figs/*.png, figs/*.pdf and numbers.md in this folder
```

It needs only `numpy` and `matplotlib` and reads `data/vit_page_data.json`. Every plotted value and every number
in `numbers.md` comes from that file; none is typed into the script. Colors and plot style are those of the
report's regression figures.

## What is here and what is not

- **Here:** the figures, `make_figs.py`, `numbers.md`, and `data/vit_page_data.json`: per-seed end-of-training
  values, means and seed ranges at the 17 checkpoints, and the per-test, per-layer and control-run differences
  with their intervals. `extract_data.py` records how the data file was taken from the study's results page.
- **Not here:** the per-run read-outs (156 MB), the model checkpoints (about 77 GB), and the analysis scripts
  that computed the data file (the layer-by-layer read-outs, the ridge classifier, the statistics). `icl/` holds
  the training and few-shot code of the first two rounds; the options added for the later control runs (Muon on
  one kind of matrix, a change of optimizer in mid-training) are not in it.
