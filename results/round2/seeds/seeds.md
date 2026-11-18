# Seeds and the shorter run

## The settings

| setting | seeds | val top-1, mean | lowest to highest | dogs_5way 5-shot from pre_logits, mean | lowest to highest | dogs_5way 5-shot from gap, mean | lowest to highest | GPU-hours a run |
|---|---|---|---|---|---|---|---|---|
| AdamW, lr 0.001 | 0, 1, 2 | 71.4 % | 71.3 to 71.6 | 51.9 % | 51.8 to 52.0 | 54.2 % | 54.0 to 54.3 | 12.4 |
| AdamW, lr 0.002 | 0 | 70.3 % | one seed | 52.3 % | one seed | 53.2 % | one seed | 12.4 |
| Muon, lr 0.003 | 0, 1, 2 | 73.5 % | 73.4 to 73.6 | 56.0 % | 55.7 to 56.3 | 55.9 % | 55.8 to 56.0 | 14.1 |
| Muon, lr 0.003, 70 epochs | 0 | 72.8 % | one seed | 55.7 % | one seed | 54.6 % | one seed | 11.0 |
| SGD, lr 1 | 0, 1, 2 | 69.4 % | 69.0 to 69.8 | 53.4 % | 52.9 to 54.1 | 53.2 % | 52.7 to 53.8 | 12.4 |

The few-shot test itself has a 95 % interval of about half a point (its 1,000 episodes), the same episodes for every run.

## Seed by seed against AdamW, lr 0.001 (points)

| setting | measure | seed 0 | seed 1 | seed 2 | mean | 95 % interval of the mean |
|---|---|---|---|---|---|---|
| Muon, lr 0.003 | val top-1 | +2.2 | +1.9 | +2.1 | +2.1 | +1.7 to +2.5 |
| Muon, lr 0.003 | dogs_5way 5-shot from pre_logits | +4.4 | +3.8 | +4.1 | +4.1 | +3.3 to +4.9 |
| Muon, lr 0.003 | dogs_5way 5-shot from gap | +1.7 | +1.9 | +1.7 | +1.7 | +1.5 to +2.0 |
| SGD, lr 1 | val top-1 | -1.5 | -2.2 | -2.5 | -2.1 | -3.4 to -0.7 |
| SGD, lr 1 | dogs_5way 5-shot from pre_logits | +2.3 | +1.3 | +0.9 | +1.5 | -0.3 to +3.3 |
| SGD, lr 1 | dogs_5way 5-shot from gap | -0.5 | -1.0 | -1.6 | -1.0 | -2.4 to +0.3 |

A difference is the setting minus the baseline in runs of the same seed. The interval is Student's t on those differences; it leaves out the few-shot test's own half point.

## The settings run once, against the seeds of AdamW, lr 0.001

| setting | measure | this run | AdamW, lr 0.001, mean of its seeds | difference (points) | one more seed would fall within (points) |
|---|---|---|---|---|---|
| AdamW, lr 0.002 | val top-1 | 70.3 % | 71.4 % | **-1.2** | ±0.8 |
| AdamW, lr 0.002 | dogs_5way 5-shot from pre_logits | 52.3 % | 51.9 % | +0.3 | ±0.5 |
| AdamW, lr 0.002 | dogs_5way 5-shot from gap | 53.2 % | 54.2 % | **-1.0** | ±1.0 |
| Muon, lr 0.003, 70 epochs | val top-1 | 72.8 % | 71.4 % | **+1.4** | ±0.8 |
| Muon, lr 0.003, 70 epochs | dogs_5way 5-shot from pre_logits | 55.7 % | 51.9 % | **+3.8** | ±0.5 |
| Muon, lr 0.003, 70 epochs | dogs_5way 5-shot from gap | 54.6 % | 54.2 % | +0.4 | ±1.0 |

The last column is how far from the mean of those seeds one more seed of the same setting would fall 95 times in 100 (Student's t on the seeds). A difference in bold lies outside it: more than another seed would explain.

## Over the 14 pools other than the trained classes, 5 examples per class, against AdamW, lr 0.001

| setting | layer | runs | clearly ahead on | clearly behind on | neither on | mean difference |
|---|---|---|---|---|---|---|
| AdamW, lr 0.002 | pre_logits | 1 run | 1 | 3 | 10 | +0.2 points |
| AdamW, lr 0.002 | gap | 1 run | 0 | 3 | 11 | -0.7 points |
| Muon, lr 0.003 | pre_logits | 3 seeds | 8 | 0 | 6 | +1.3 points |
| Muon, lr 0.003 | gap | 3 seeds | 5 | 1 | 8 | -0.1 points |
| Muon, lr 0.003, 70 epochs | pre_logits | 1 run | 10 | 0 | 4 | +1.6 points |
| Muon, lr 0.003, 70 epochs | gap | 1 run | 0 | 1 | 13 | -0.5 points |
| SGD, lr 1 | pre_logits | 3 seeds | 2 | 2 | 10 | +0.6 points |
| SGD, lr 1 | gap | 3 seeds | 1 | 3 | 10 | -0.8 points |

"Clearly" means that the 95 % interval of the difference leaves out zero. For a setting with several seeds the difference is the mean of its seed-by-seed differences from the baseline, with Student's t on them; for a setting run once it is the distance from the mean of the baseline's seeds, with the interval in which one more seed of the baseline would fall. Of 20 pools on which nothing differs, one would still count; and the pools overlap (three of them are dog breeds), so the counts are not independent. The few-shot test's own half point is left out.

### Read from pre_logits: difference in points ± half the width of its 95 % interval; bold where the interval leaves out zero

| pool | AdamW, lr 0.002 | Muon, lr 0.003 | Muon, lr 0.003, 70 epochs | SGD, lr 1 |
|---|---|---|---|---|
| birds_20way | **-0.7** ± 0.4 | **+1.9** ± 0.7 | **+1.9** ± 0.4 | +0.1 ± 1.6 |
| birds_59way | **-0.6** ± 0.5 | **+1.9** ± 0.7 | **+1.8** ± 0.5 | +0.2 ± 1.4 |
| birds_5way | -0.2 ± 1.2 | **+1.4** ± 0.8 | **+1.8** ± 1.2 | +0.0 ± 1.6 |
| cats | +0.7 ± 1.0 | -1.3 ± 2.6 | -0.3 ± 1.0 | +1.5 ± 2.1 |
| control | -0.2 ± 1.8 | -0.6 ± 0.6 | -0.6 ± 1.8 | **-1.9** ± 0.7 |
| dogs_118way | **+0.5** ± 0.4 | **+1.8** ± 0.3 | **+1.5** ± 0.4 | +0.7 ± 1.0 |
| dogs_20way | +0.6 ± 0.7 | **+3.6** ± 0.8 | **+3.1** ± 0.7 | +1.5 ± 1.7 |
| dogs_5way | +0.3 ± 0.5 | **+4.1** ± 0.8 | **+3.8** ± 0.5 | +1.5 ± 1.8 |
| felines_13way | +0.3 ± 0.6 | +0.1 ± 2.6 | **+0.8** ± 0.6 | +1.3 ± 1.9 |
| felines_5way | +0.4 ± 1.6 | +0.2 ± 4.1 | +0.6 ± 1.6 | **+1.8** ± 1.4 |
| inet_20way | -0.5 ± 0.5 | **+2.9** ± 0.7 | **+2.6** ± 0.5 | -0.2 ± 1.0 |
| inet_5way | **-0.3** ± 0.1 | **+1.4** ± 0.1 | **+1.3** ± 0.1 | **-0.2** ± 0.2 |
| wildcanids_12way | +1.3 ± 2.9 | +0.7 ± 1.2 | +2.0 ± 2.9 | +1.2 ± 1.3 |
| wildcanids_5way | +1.4 ± 2.2 | +0.8 ± 1.2 | **+2.4** ± 2.2 | **+1.4** ± 1.3 |

### Read from gap: difference in points ± half the width of its 95 % interval; bold where the interval leaves out zero

| pool | AdamW, lr 0.002 | Muon, lr 0.003 | Muon, lr 0.003, 70 epochs | SGD, lr 1 |
|---|---|---|---|---|
| birds_20way | -1.2 ± 2.5 | -0.4 ± 0.8 | -1.1 ± 2.5 | -1.9 ± 2.3 |
| birds_59way | -1.1 ± 1.8 | +0.0 ± 0.6 | -0.7 ± 1.8 | -1.5 ± 2.1 |
| birds_5way | -0.8 ± 2.3 | **-1.1** ± 0.9 | -1.7 ± 2.3 | -2.2 ± 2.2 |
| cats | -1.2 ± 1.3 | -2.6 ± 2.7 | **-3.0** ± 1.3 | -0.6 ± 3.0 |
| control | -0.3 ± 2.6 | +0.1 ± 0.7 | -0.3 ± 2.6 | **-1.7** ± 1.0 |
| dogs_118way | -0.1 ± 0.9 | **+0.6** ± 0.5 | +0.0 ± 0.9 | -0.4 ± 1.0 |
| dogs_20way | -0.5 ± 1.1 | **+1.4** ± 0.4 | +0.2 ± 1.1 | -0.7 ± 1.3 |
| dogs_5way | **-1.0** ± 1.0 | **+1.7** ± 0.2 | +0.4 ± 1.0 | -1.0 ± 1.4 |
| felines_13way | -1.0 ± 1.5 | -1.4 ± 3.0 | -1.0 ± 1.5 | +0.1 ± 1.2 |
| felines_5way | -0.6 ± 3.0 | -1.5 ± 4.9 | -1.2 ± 3.0 | **+0.5** ± 0.2 |
| inet_20way | **-1.1** ± 0.5 | **+1.2** ± 0.2 | +0.3 ± 0.5 | **-1.7** ± 1.3 |
| inet_5way | **-0.5** ± 0.4 | **+0.8** ± 0.2 | +0.2 ± 0.4 | **-0.7** ± 0.5 |
| wildcanids_12way | +0.1 ± 3.0 | +0.3 ± 2.4 | +0.6 ± 3.0 | +0.2 ± 1.9 |
| wildcanids_5way | -0.2 ± 1.8 | +0.0 ± 2.0 | +0.8 ± 1.8 | +0.3 ± 1.9 |

## How far the seeds of one setting lie apart (highest minus lowest, points)

| setting | val top-1 | dogs_5way 5-shot from pre_logits | dogs_5way 5-shot from gap | median over the pools, pre_logits | median over the pools, gap |
|---|---|---|---|---|---|
| AdamW, lr 0.001 | 0.3 | 0.2 | 0.3 | 0.2 | 0.7 |
| Muon, lr 0.003 | 0.2 | 0.5 | 0.2 | 0.6 | 0.5 |
| SGD, lr 1 | 0.8 | 1.2 | 1.1 | 0.8 | 1.1 |

This is the yardstick for two runs that differ in something else as well: a difference smaller than these is not one.

## When the few-shot ability appears (dogs_5way 5-shot), in % of training, seed by seed

| setting | layer | clearly above the untrained network from | half of the gain by | nine tenths by |
|---|---|---|---|---|
| AdamW, lr 0.001 | pre_logits | 2.0, 2.0, 2.0 | 8.6, 6.5, 6.8 | 69.3, 66.0, 71.4 |
| AdamW, lr 0.001 | gap | 2.0, 2.0, 2.0 | 8.8, 8.2, 7.7 | 69.5, 65.9, 72.9 |
| Muon, lr 0.003 | pre_logits | 0.5, 0.5, 0.5 | 4.8, 3.7, 3.8 | 72.1, 64.2, 66.8 |
| Muon, lr 0.003 | gap | 0.5, 0.5, 0.5 | 5.5, 6.5, 4.9 | 72.9, 66.5, 69.5 |
| SGD, lr 1 | pre_logits | 0.5, 0.5, 0.5 | 9.4, 8.4, 6.7 | 81.2, 67.0, 80.2 |
| SGD, lr 1 | gap | 0.5, 0.5, 0.5 | 8.7, 6.7, 6.2 | 81.8, 67.5, 80.9 |

"Gain" is the rise from the untrained network to the run's last checkpoint; the steps lie on the straight line between the two checkpoints around them.

## The run planned shorter: Muon, lr 0.003, 70 epochs

| setting | steps | GPU-hours a run | val top-1 | lowest to highest | dogs_5way 5-shot from pre_logits | lowest to highest | dogs_5way 5-shot from gap | lowest to highest |
|---|---|---|---|---|---|---|---|---|
| AdamW, lr 0.001 | 107,859 | 12.4 | 71.4 % | 71.3 to 71.6 | 51.9 % | 51.8 to 52.0 | 54.2 % | 54.0 to 54.3 |
| Muon, lr 0.003 | 107,859 | 14.1 | 73.5 % | 73.4 to 73.6 | 56.0 % | 55.7 to 56.3 | 55.9 % | 55.8 to 56.0 |
| Muon, lr 0.003, 70 epochs | 83,891 | 11.0 | 72.8 % | one seed | 55.7 % | one seed | 54.6 % | one seed |

Muon, lr 0.003, 70 epochs ends 1.4 points above the mean of AdamW, lr 0.001 on validation top-1 (whose seeds run from 71.3 to 71.6), for 11 % fewer GPU-hours.

Against its own setting at full length, for 22 % fewer GPU-hours:

| setting | measure | this run | Muon, lr 0.003, mean of its seeds | difference (points) | one more seed would fall within (points) |
|---|---|---|---|---|---|
| Muon, lr 0.003, 70 epochs | val top-1 | 72.8 % | 73.5 % | **-0.7** | ±0.6 |
| Muon, lr 0.003, 70 epochs | dogs_5way 5-shot from pre_logits | 55.7 % | 56.0 % | -0.3 | ±1.4 |
| Muon, lr 0.003, 70 epochs | dogs_5way 5-shot from gap | 54.6 % | 55.9 % | **-1.3** | ±0.6 |

The last column is how far from the mean of those seeds one more seed of the same setting would fall 95 times in 100 (Student's t on the seeds). A difference in bold lies outside it: more than another seed would explain.
