# Seeds and the shorter run

## The settings

| setting | seeds | val top-1, mean | lowest to highest | dogs_5way 10-shot from pre_logits, mean | lowest to highest | dogs_5way 10-shot from gap, mean | lowest to highest | GPU-hours a run |
|---|---|---|---|---|---|---|---|---|
| AdamW, lr 0.001 | 0, 1, 2 | 71.4 % | 71.3 to 71.6 | 57.2 % | 57.0 to 57.5 | 59.2 % | 58.9 to 59.4 | 12.4 |
| AdamW, lr 0.002 | 0 | 70.3 % | one seed | 58.0 % | one seed | 58.5 % | one seed | 12.4 |
| Muon, lr 0.003 | 0, 1, 2 | 73.5 % | 73.4 to 73.6 | 61.5 % | 61.3 to 61.7 | 61.0 % | 60.7 to 61.2 | 14.1 |
| Muon, lr 0.003, 70 epochs | 0 | 72.8 % | one seed | 61.3 % | one seed | 59.9 % | one seed | 11.0 |
| SGD, lr 1 | 0, 1, 2 | 69.4 % | 69.0 to 69.8 | 59.0 % | 58.3 to 59.5 | 58.5 % | 58.1 to 59.1 | 12.4 |

The few-shot test itself has a 95 % interval of about half a point (its 1,000 episodes), the same episodes for every run.

## Seed by seed against AdamW, lr 0.001 (points)

| setting | measure | seed 0 | seed 1 | seed 2 | mean | 95 % interval of the mean |
|---|---|---|---|---|---|---|
| Muon, lr 0.003 | val top-1 | +2.2 | +1.9 | +2.1 | +2.1 | +1.7 to +2.5 |
| Muon, lr 0.003 | dogs_5way 10-shot from pre_logits | +4.6 | +4.3 | +4.1 | +4.3 | +3.7 to +4.9 |
| Muon, lr 0.003 | dogs_5way 10-shot from gap | +1.8 | +1.8 | +1.9 | +1.8 | +1.6 to +2.0 |
| SGD, lr 1 | val top-1 | -1.5 | -2.2 | -2.5 | -2.1 | -3.4 to -0.7 |
| SGD, lr 1 | dogs_5way 10-shot from pre_logits | +2.4 | +2.2 | +0.8 | +1.8 | -0.4 to +4.0 |
| SGD, lr 1 | dogs_5way 10-shot from gap | -0.3 | -0.5 | -1.1 | -0.7 | -1.7 to +0.4 |

A difference is the setting minus the baseline in runs of the same seed. The interval is Student's t on those differences; it leaves out the few-shot test's own half point.

## The settings run once, against the seeds of AdamW, lr 0.001

| setting | measure | this run | AdamW, lr 0.001, mean of its seeds | difference (points) | one more seed would fall within (points) |
|---|---|---|---|---|---|
| AdamW, lr 0.002 | val top-1 | 70.3 % | 71.4 % | **-1.2** | ±0.8 |
| AdamW, lr 0.002 | dogs_5way 10-shot from pre_logits | 58.0 % | 57.2 % | +0.8 | ±1.4 |
| AdamW, lr 0.002 | dogs_5way 10-shot from gap | 58.5 % | 59.2 % | -0.7 | ±1.2 |
| Muon, lr 0.003, 70 epochs | val top-1 | 72.8 % | 71.4 % | **+1.4** | ±0.8 |
| Muon, lr 0.003, 70 epochs | dogs_5way 10-shot from pre_logits | 61.3 % | 57.2 % | **+4.1** | ±1.4 |
| Muon, lr 0.003, 70 epochs | dogs_5way 10-shot from gap | 59.9 % | 59.2 % | +0.7 | ±1.2 |

The last column is how far from the mean of those seeds one more seed of the same setting would fall 95 times in 100 (Student's t on the seeds). A difference in bold lies outside it: more than another seed would explain.

## Over the 14 pools other than the trained classes, 10 examples per class, against AdamW, lr 0.001

| setting | layer | runs | clearly ahead on | clearly behind on | neither on | mean difference |
|---|---|---|---|---|---|---|
| AdamW, lr 0.002 | pre_logits | 1 run | 1 | 0 | 13 | +0.4 points |
| AdamW, lr 0.002 | gap | 1 run | 0 | 2 | 12 | -0.7 points |
| Muon, lr 0.003 | pre_logits | 3 seeds | 8 | 0 | 6 | +1.7 points |
| Muon, lr 0.003 | gap | 3 seeds | 4 | 0 | 10 | +0.0 points |
| Muon, lr 0.003, 70 epochs | pre_logits | 1 run | 9 | 0 | 5 | +2.0 points |
| Muon, lr 0.003, 70 epochs | gap | 1 run | 0 | 2 | 12 | -0.5 points |
| SGD, lr 1 | pre_logits | 3 seeds | 2 | 1 | 11 | +0.8 points |
| SGD, lr 1 | gap | 3 seeds | 1 | 3 | 10 | -0.7 points |

"Clearly" means that the 95 % interval of the difference leaves out zero. For a setting with several seeds the difference is the mean of its seed-by-seed differences from the baseline, with Student's t on them; for a setting run once it is the distance from the mean of the baseline's seeds, with the interval in which one more seed of the baseline would fall. Of 20 pools on which nothing differs, one would still count; and the pools overlap (three of them are dog breeds), so the counts are not independent. The few-shot test's own half point is left out.

### Read from pre_logits: difference in points ± half the width of its 95 % interval; bold where the interval leaves out zero

| pool | AdamW, lr 0.002 | Muon, lr 0.003 | Muon, lr 0.003, 70 epochs | SGD, lr 1 |
|---|---|---|---|---|
| birds_20way | -0.4 ± 1.1 | **+2.2** ± 0.3 | **+2.4** ± 1.1 | +0.4 ± 1.6 |
| birds_59way | -0.6 ± 0.7 | **+2.2** ± 0.3 | **+2.3** ± 0.7 | +0.4 ± 1.5 |
| birds_5way | -0.1 ± 1.6 | **+1.6** ± 0.4 | **+1.7** ± 1.6 | +0.2 ± 2.3 |
| cats | +0.8 ± 0.9 | -0.3 ± 3.6 | +0.6 ± 0.9 | +1.3 ± 2.0 |
| control | -0.1 ± 1.3 | +0.0 ± 0.6 | +0.2 ± 1.3 | **-1.4** ± 0.3 |
| dogs_118way | **+0.6** ± 0.5 | **+2.3** ± 0.5 | **+1.9** ± 0.5 | +0.9 ± 1.6 |
| dogs_20way | +0.7 ± 1.1 | **+4.1** ± 0.7 | **+3.5** ± 1.1 | +1.7 ± 2.2 |
| dogs_5way | +0.8 ± 1.4 | **+4.3** ± 0.6 | **+4.1** ± 1.4 | +1.8 ± 2.2 |
| felines_13way | +0.7 ± 1.6 | +0.7 ± 3.2 | +1.3 ± 1.6 | +1.6 ± 2.3 |
| felines_5way | +0.7 ± 1.2 | +0.7 ± 4.4 | +0.7 ± 1.2 | **+2.0** ± 0.9 |
| inet_20way | -0.3 ± 0.4 | **+2.9** ± 0.5 | **+2.8** ± 0.4 | +0.0 ± 0.9 |
| inet_5way | -0.2 ± 0.6 | **+1.3** ± 0.2 | **+1.3** ± 0.6 | -0.1 ± 0.7 |
| wildcanids_12way | +2.1 ± 2.9 | +1.1 ± 1.6 | +2.8 ± 2.9 | **+1.2** ± 1.1 |
| wildcanids_5way | +1.5 ± 2.4 | +1.0 ± 1.5 | **+2.7** ± 2.4 | +1.7 ± 1.7 |

### Read from gap: difference in points ± half the width of its 95 % interval; bold where the interval leaves out zero

| pool | AdamW, lr 0.002 | Muon, lr 0.003 | Muon, lr 0.003, 70 epochs | SGD, lr 1 |
|---|---|---|---|---|
| birds_20way | -1.0 ± 3.7 | -0.6 ± 1.6 | -1.5 ± 3.7 | -1.8 ± 3.0 |
| birds_59way | -1.3 ± 3.0 | -0.2 ± 1.4 | -0.9 ± 3.0 | -1.6 ± 2.5 |
| birds_5way | -0.3 ± 2.8 | -0.7 ± 1.3 | -1.5 ± 2.8 | -1.6 ± 2.6 |
| cats | **-1.0** ± 0.9 | -1.7 ± 3.2 | **-3.1** ± 0.9 | -0.7 ± 4.0 |
| control | -0.5 ± 1.9 | -0.1 ± 0.2 | -0.4 ± 1.9 | **-1.6** ± 0.6 |
| dogs_118way | -0.2 ± 1.3 | +0.8 ± 0.9 | +0.0 ± 1.3 | -0.5 ± 1.5 |
| dogs_20way | -0.5 ± 1.8 | **+1.6** ± 0.6 | +0.4 ± 1.8 | -0.7 ± 1.6 |
| dogs_5way | -0.7 ± 1.2 | **+1.8** ± 0.2 | +0.7 ± 1.2 | -0.7 ± 1.0 |
| felines_13way | -1.0 ± 1.1 | -1.2 ± 2.9 | **-1.2** ± 1.1 | +0.5 ± 2.2 |
| felines_5way | -0.7 ± 2.6 | -1.3 ± 5.4 | -1.5 ± 2.6 | **+0.9** ± 0.4 |
| inet_20way | **-0.9** ± 0.8 | **+1.0** ± 0.5 | +0.2 ± 0.8 | **-1.4** ± 1.2 |
| inet_5way | -0.5 ± 0.5 | **+0.5** ± 0.2 | +0.2 ± 0.5 | **-0.6** ± 0.5 |
| wildcanids_12way | +0.1 ± 2.5 | +0.3 ± 3.3 | +0.9 ± 2.5 | +0.0 ± 1.5 |
| wildcanids_5way | -0.7 ± 1.0 | -0.1 ± 3.3 | +0.7 ± 1.0 | +0.4 ± 1.5 |

## How far the seeds of one setting lie apart (highest minus lowest, points)

| setting | val top-1 | dogs_5way 10-shot from pre_logits | dogs_5way 10-shot from gap | median over the pools, pre_logits | median over the pools, gap |
|---|---|---|---|---|---|
| AdamW, lr 0.001 | 0.3 | 0.6 | 0.5 | 0.4 | 0.6 |
| Muon, lr 0.003 | 0.2 | 0.4 | 0.5 | 0.7 | 0.7 |
| SGD, lr 1 | 0.8 | 1.2 | 0.9 | 1.1 | 1.0 |

This is the yardstick for two runs that differ in something else as well: a difference smaller than these is not one.

## When the few-shot ability appears (dogs_5way 10-shot), in % of training, seed by seed

| setting | layer | clearly above the untrained network from | half of the gain by | nine tenths by |
|---|---|---|---|---|
| AdamW, lr 0.001 | pre_logits | 2.0, 2.0, 2.0 | 8.1, 5.5, 6.3 | 58.8, 65.1, 51.9 |
| AdamW, lr 0.001 | gap | 2.0, 2.0, 2.0 | 7.8, 6.2, 6.2 | 64.4, 49.6, 53.1 |
| Muon, lr 0.003 | pre_logits | 0.5, 0.5, 0.5 | 3.9, 3.2, 3.1 | 70.6, 63.9, 67.2 |
| Muon, lr 0.003 | gap | 0.5, 0.5, 0.5 | 3.8, 3.8, 2.9 | 72.2, 65.3, 77.7 |
| SGD, lr 1 | pre_logits | 0.5, 0.5, 0.5 | 8.7, 6.8, 6.0 | 81.1, 67.5, 76.8 |
| SGD, lr 1 | gap | 0.5, 0.5, 0.5 | 7.6, 5.2, 5.1 | 81.9, 67.2, 80.0 |

"Gain" is the rise from the untrained network to the run's last checkpoint; the steps lie on the straight line between the two checkpoints around them.

## The run planned shorter: Muon, lr 0.003, 70 epochs

| setting | steps | GPU-hours a run | val top-1 | lowest to highest | dogs_5way 10-shot from pre_logits | lowest to highest | dogs_5way 10-shot from gap | lowest to highest |
|---|---|---|---|---|---|---|---|---|
| AdamW, lr 0.001 | 107,859 | 12.4 | 71.4 % | 71.3 to 71.6 | 57.2 % | 57.0 to 57.5 | 59.2 % | 58.9 to 59.4 |
| Muon, lr 0.003 | 107,859 | 14.1 | 73.5 % | 73.4 to 73.6 | 61.5 % | 61.3 to 61.7 | 61.0 % | 60.7 to 61.2 |
| Muon, lr 0.003, 70 epochs | 83,891 | 11.0 | 72.8 % | one seed | 61.3 % | one seed | 59.9 % | one seed |

Muon, lr 0.003, 70 epochs ends 1.4 points above the mean of AdamW, lr 0.001 on validation top-1 (whose seeds run from 71.3 to 71.6), for 11 % fewer GPU-hours.

Against its own setting at full length, for 22 % fewer GPU-hours:

| setting | measure | this run | Muon, lr 0.003, mean of its seeds | difference (points) | one more seed would fall within (points) |
|---|---|---|---|---|---|
| Muon, lr 0.003, 70 epochs | val top-1 | 72.8 % | 73.5 % | **-0.7** | ±0.6 |
| Muon, lr 0.003, 70 epochs | dogs_5way 10-shot from pre_logits | 61.3 % | 61.5 % | -0.2 | ±1.1 |
| Muon, lr 0.003, 70 epochs | dogs_5way 10-shot from gap | 59.9 % | 61.0 % | -1.1 | ±1.3 |

The last column is how far from the mean of those seeds one more seed of the same setting would fall 95 times in 100 (Student's t on the seeds). A difference in bold lies outside it: more than another seed would explain.
