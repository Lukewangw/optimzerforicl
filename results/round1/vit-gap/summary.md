# Summary (gap features)

Best learning rates: AdamW, lr 0.001 (picked at step 107,859), Muon, lr 0.003 (picked at step 107,859), SGD, lr 1 (picked at step 107,859).

## Runs

| run | steps done | val top-1 | dogs_5way 5-shot | untrained | clearly above untrained from | half of the gain by | 90 % of the gain by | dogs_20way 5-shot | dogs_118way 5-shot | birds_5way 5-shot | inet_5way 5-shot | control 5-shot | seen_5way 5-shot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AdamW, lr 0.0003 | 107,859 (100 %) | 70.3 % | 47.6 ± 0.5 % | 26.7 % | step 539 (0.5 %) | step 7,016 (6.5 %) | step 59,314 (55.0 %) | 22.1 % | 7.3 % | 58.2 % | 85.9 % | 71.2 % | 95.4 % |
| AdamW, lr 0.001 **(best)** | 107,859 (100 %) | 71.3 % | 54.3 ± 0.5 % | 26.7 % | step 2,157 (2.0 %) | step 9,499 (8.8 %) | step 74,954 (69.5 %) | 27.8 % | 10.4 % | 63.1 % | 87.7 % | 73.3 % | 95.1 % |
| AdamW, lr 0.003 | 107,859 (100 %) | 12.3 % | 36.0 ± 0.5 % | 26.7 % | step 1,079 (1.0 %) | step 1,708 (1.6 %) | step 3,800 (3.5 %) | 13.6 % | 3.7 % | 46.9 % | 55.5 % | 28.8 % | 54.3 % |
| Muon, lr 0.0003 | 107,859 (100 %) | 72.4 % | 50.8 ± 0.5 % | 26.7 % | step 539 (0.5 %) | step 5,165 (4.8 %) | step 61,768 (57.3 %) | 24.7 % | 8.6 % | 58.4 % | 86.8 % | 71.7 % | 95.9 % |
| Muon, lr 0.001 | 107,859 (100 %) | 73.3 % | 53.2 ± 0.5 % | 26.7 % | step 539 (0.5 %) | step 5,032 (4.7 %) | step 63,997 (59.3 %) | 26.8 % | 9.8 % | 59.7 % | 87.1 % | 70.5 % | 94.7 % |
| Muon, lr 0.003 **(best)** | 107,859 (100 %) | 73.4 % | 56.0 ± 0.5 % | 26.7 % | step 539 (0.5 %) | step 5,952 (5.5 %) | step 78,667 (72.9 %) | 29.0 % | 10.8 % | 62.5 % | 88.5 % | 73.1 % | 95.4 % |
| SGD, lr 0.03 | 107,859 (100 %) | 26.4 % | 39.4 ± 0.4 % | 26.7 % | step 539 (0.5 %) | step 3,201 (3.0 %) | step 13,053 (12.1 %) | 16.2 % | 4.8 % | 51.6 % | 73.0 % | 49.8 % | 80.3 % |
| SGD, lr 0.1 | 107,859 (100 %) | 54.3 % | 45.7 ± 0.5 % | 26.7 % | step 539 (0.5 %) | step 3,387 (3.1 %) | step 20,294 (18.8 %) | 20.8 % | 7.0 % | 58.9 % | 84.0 % | 64.7 % | 91.4 % |
| SGD, lr 0.3 | 107,859 (100 %) | 68.2 % | 53.3 ± 0.5 % | 26.7 % | step 539 (0.5 %) | step 8,411 (7.8 %) | step 72,559 (67.3 %) | 26.9 % | 9.6 % | 61.0 % | 87.1 % | 70.8 % | 94.3 % |
| SGD, lr 1 **(best)** | 107,859 (100 %) | 69.8 % | 53.8 ± 0.5 % | 26.7 % | step 539 (0.5 %) | step 9,388 (8.7 %) | step 88,183 (81.8 %) | 27.1 % | 9.8 % | 60.8 % | 87.1 % | 71.2 % | 94.7 % |

"Gain" is the rise from the untrained network to the run's last checkpoint; the steps for half and 90 % of it lie on the straight line between the two checkpoints around them.

## dogs_5way 5-shot at equal validation top-1 (best learning rate of each optimizer)

| run | at 20 % | at 30 % | at 40 % | at 50 % | at 60 % |
|---|---|---|---|---|---|
| AdamW, lr 0.001 | 38.1 % | 41.8 % | 44.6 % | 46.6 % | 48.0 % |
| Muon, lr 0.003 | 39.6 % | 40.5 % | 42.9 % | 43.5 % | 47.7 % |
| SGD, lr 1 | 38.6 % | 41.9 % | 43.9 % | 46.3 % | 49.4 % |

## Against the baseline (AdamW, lr 0.001) on the 14 pools other than the trained classes, 5 examples per class

| run | pre_logits: clearly ahead on | pre_logits: clearly behind on | pre_logits: mean difference | gap: clearly ahead on | gap: clearly behind on | gap: mean difference |
|---|---|---|---|---|---|---|
| AdamW, lr 0.0003 | 0 | 14 | -2.7 points | 0 | 14 | -4.3 points |
| AdamW, lr 0.003 | 0 | 14 | -16.2 points | 0 | 14 | -18.5 points |
| Muon, lr 0.0003 | 0 | 11 | -1.2 points | 0 | 14 | -3.1 points |
| Muon, lr 0.001 | 10 | 1 | +0.9 points | 0 | 13 | -2.3 points |
| Muon, lr 0.003 | 8 | 2 | +1.2 points | 6 | 5 | -0.4 points |
| SGD, lr 0.03 | 0 | 14 | -11.3 points | 0 | 14 | -12.1 points |
| SGD, lr 0.1 | 0 | 14 | -5.2 points | 0 | 14 | -5.5 points |
| SGD, lr 0.3 | 8 | 3 | +0.4 points | 2 | 11 | -0.9 points |
| SGD, lr 1 | 8 | 2 | +0.8 points | 3 | 8 | -0.6 points |

The pools: birds_20way, birds_59way, birds_5way, cats, control, dogs_118way, dogs_20way, dogs_5way, felines_13way, felines_5way, inet_20way, inet_5way, wildcanids_12way, wildcanids_5way. "Clearly": by more than the two runs' 95 % intervals combined (the intervals cover the episodes drawn, not the training seed).

## Cost, and how early each run reached what the baseline (AdamW, lr 0.001) ended with

| run | images per second (4 GPUs) | GPU-hours for the whole run | reached the baseline's final val top-1 (71.3 %) at | GPU-hours to that point | reached its final dogs_5way 5-shot (54.3 %) at | GPU-hours to that point |
|---|---|---|---|---|---|---|
| AdamW, lr 0.0003 | 9,906 | 12.4 | not reached | n/a | not reached | n/a |
| AdamW, lr 0.001 | 9,878 | 12.4 | step 107,859 (100 % of training) | 12.4 | step 107,859 (100 % of training) | 12.4 |
| AdamW, lr 0.003 | 9,853 | 12.5 | not reached | n/a | not reached | n/a |
| Muon, lr 0.0003 | 8,722 | 14.1 | step 90,477 (84 % of training) | 11.8 | not reached | n/a |
| Muon, lr 0.001 | 8,717 | 14.1 | step 83,952 (78 % of training) | 11.0 | not reached | n/a |
| Muon, lr 0.003 | 8,682 | 14.1 | step 84,175 (78 % of training) | 11.0 | step 87,806 (81 % of training) | 11.5 |
| SGD, lr 0.03 | 9,975 | 12.3 | not reached | n/a | not reached | n/a |
| SGD, lr 0.1 | 9,914 | 12.4 | not reached | n/a | not reached | n/a |
| SGD, lr 0.3 | 9,917 | 12.4 | not reached | n/a | not reached | n/a |
| SGD, lr 1 | 9,911 | 12.4 | not reached | n/a | not reached | n/a |

The steps lie on the straight line between the two checkpoints around them. A run stopped at that point is not a run planned to be that short: its learning rate would still be high there. Whether a shorter run with its own schedule gets as far is a separate run.
