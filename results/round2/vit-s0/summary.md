# Summary (pre_logits features)

Best learning rates: AdamW, lr 0.001 (picked at step 107,859), Muon, lr 0.003 (picked at step 107,859), SGD, lr 1 (picked at step 107,859).

## Runs

| run | steps done | val top-1 | dogs_5way 5-shot | untrained | clearly above untrained from | half of the gain by | 90 % of the gain by | dogs_20way 5-shot | dogs_118way 5-shot | birds_5way 5-shot | inet_5way 5-shot | control 5-shot | seen_5way 5-shot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AdamW, lr 0.0003 | 107,859 (100 %) | 70.3 % | 46.8 ± 0.5 % | 26.5 % | step 2,157 (2.0 %) | step 7,757 (7.2 %) | step 61,954 (57.4 %) | 21.1 % | 6.7 % | 58.1 % | 85.5 % | 70.6 % | 95.7 % |
| AdamW, lr 0.001 **(best)** | 107,859 (100 %) | 71.3 % | 51.8 ± 0.5 % | 26.5 % | step 2,157 (2.0 %) | step 9,319 (8.6 %) | step 74,729 (69.3 %) | 25.3 % | 8.9 % | 61.1 % | 86.7 % | 72.2 % | 95.5 % |
| AdamW, lr 0.002 | 107,859 (100 %) | 70.3 % | 52.3 ± 0.5 % | 26.5 % | step 1,079 (1.0 %) | step 9,038 (8.4 %) | step 72,357 (67.1 %) | 26.0 % | 9.4 % | 60.7 % | 86.5 % | 71.6 % | 95.0 % |
| AdamW, lr 0.003 | 107,859 (100 %) | 12.3 % | 36.5 ± 0.5 % | 26.5 % | step 1,079 (1.0 %) | step 1,745 (1.6 %) | step 82,782 (76.7 %) | 13.9 % | 3.8 % | 47.1 % | 57.7 % | 31.0 % | 57.9 % |
| Muon, lr 0.0003 | 107,859 (100 %) | 72.4 % | 51.3 ± 0.5 % | 26.5 % | step 539 (0.5 %) | step 8,031 (7.4 %) | step 66,396 (61.6 %) | 24.8 % | 8.5 % | 59.5 % | 86.8 % | 71.5 % | 96.1 % |
| Muon, lr 0.001 | 107,859 (100 %) | 73.3 % | 53.7 ± 0.5 % | 26.5 % | step 1,079 (1.0 %) | step 5,634 (5.2 %) | step 61,666 (57.2 %) | 27.1 % | 9.8 % | 62.1 % | 87.7 % | 71.7 % | 95.7 % |
| Muon, lr 0.003 **(best)** | 107,859 (100 %) | 73.4 % | 56.3 ± 0.5 % | 26.5 % | step 539 (0.5 %) | step 5,211 (4.8 %) | step 77,737 (72.1 %) | 29.2 % | 10.7 % | 62.2 % | 88.1 % | 71.4 % | 95.1 % |
| SGD, lr 0.03 | 107,859 (100 %) | 26.4 % | 37.6 ± 0.4 % | 26.5 % | step 539 (0.5 %) | step 2,749 (2.5 %) | step 11,828 (11.0 %) | 15.1 % | 4.4 % | 50.3 % | 72.3 % | 49.3 % | 80.1 % |
| SGD, lr 0.1 | 107,859 (100 %) | 54.3 % | 43.8 ± 0.5 % | 26.5 % | step 1,079 (1.0 %) | step 3,220 (3.0 %) | step 19,348 (17.9 %) | 19.2 % | 6.2 % | 56.7 % | 83.0 % | 64.2 % | 91.9 % |
| SGD, lr 0.3 | 107,859 (100 %) | 68.2 % | 53.0 ± 0.5 % | 26.5 % | step 539 (0.5 %) | step 9,206 (8.5 %) | step 78,572 (72.8 %) | 26.4 % | 9.1 % | 60.4 % | 86.7 % | 70.3 % | 94.8 % |
| SGD, lr 1 **(best)** | 107,859 (100 %) | 69.8 % | 54.1 ± 0.5 % | 26.5 % | step 539 (0.5 %) | step 10,108 (9.4 %) | step 87,585 (81.2 %) | 27.1 % | 9.6 % | 60.4 % | 86.6 % | 70.1 % | 94.7 % |

"Gain" is the rise from the untrained network to the run's last checkpoint; the steps for half and 90 % of it lie on the straight line between the two checkpoints around them.

## dogs_5way 5-shot at equal validation top-1 (best learning rate of each optimizer)

| run | at 20 % | at 30 % | at 40 % | at 50 % | at 60 % |
|---|---|---|---|---|---|
| AdamW, lr 0.001 | 36.9 % | 40.7 % | 43.4 % | 45.0 % | 46.6 % |
| Muon, lr 0.003 | 39.2 % | 41.1 % | 44.1 % | 44.5 % | 48.9 % |
| SGD, lr 1 | 37.5 % | 41.3 % | 43.5 % | 45.6 % | 49.3 % |

## Against the baseline (AdamW, lr 0.001) on the 14 pools other than the trained classes, 5 examples per class

| run | pre_logits: clearly ahead on | pre_logits: clearly behind on | pre_logits: mean difference | gap: clearly ahead on | gap: clearly behind on | gap: mean difference |
|---|---|---|---|---|---|---|
| AdamW, lr 0.0003 | 0 | 14 | -2.7 points | 0 | 14 | -4.3 points |
| AdamW, lr 0.002 | 5 | 3 | +0.2 points | 0 | 10 | -0.8 points |
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

| run | images per second (4 GPUs) | GPU-hours for the whole run | reached the baseline's final val top-1 (71.3 %) at | GPU-hours to that point | reached its final dogs_5way 5-shot (51.8 %) at | GPU-hours to that point |
|---|---|---|---|---|---|---|
| AdamW, lr 0.0003 | 9,906 | 12.4 | not reached | n/a | not reached | n/a |
| AdamW, lr 0.001 | 9,878 | 12.4 | step 107,859 (100 % of training) | 12.4 | step 96,089 (89 % of training) | 11.1 |
| AdamW, lr 0.002 | 9,910 | 12.4 | not reached | n/a | step 90,533 (84 % of training) | 10.4 |
| AdamW, lr 0.003 | 9,853 | 12.5 | not reached | n/a | not reached | n/a |
| Muon, lr 0.0003 | 8,722 | 14.1 | step 90,477 (84 % of training) | 11.8 | not reached | n/a |
| Muon, lr 0.001 | 8,717 | 14.1 | step 83,952 (78 % of training) | 11.0 | step 68,907 (64 % of training) | 9.0 |
| Muon, lr 0.003 | 8,682 | 14.1 | step 84,175 (78 % of training) | 11.0 | step 61,138 (57 % of training) | 8.0 |
| SGD, lr 0.03 | 9,975 | 12.3 | not reached | n/a | not reached | n/a |
| SGD, lr 0.1 | 9,914 | 12.4 | not reached | n/a | not reached | n/a |
| SGD, lr 0.3 | 9,917 | 12.4 | not reached | n/a | step 89,903 (83 % of training) | 10.3 |
| SGD, lr 1 | 9,911 | 12.4 | not reached | n/a | step 89,181 (83 % of training) | 10.2 |

The steps lie on the straight line between the two checkpoints around them. A run stopped at that point is not a run planned to be that short: its learning rate would still be high there. Whether a shorter run with its own schedule gets as far is a separate run.
