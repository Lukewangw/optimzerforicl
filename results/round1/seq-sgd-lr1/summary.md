# Sequence models on the features of the encoder at step 107,859

| optimizer | learning rate | seeds finished | dogs: final accuracy | dogs: highest during training (at step) | control: final accuracy | seen classes: final accuracy | dogs: ability appears at step | seen: appears at step |
|---|---|---|---|---|---|---|---|---|
| AdamW | 3e-05 | 5 of 5 | 46.6 ± 0.9 % | 46.6 % (10,000) | 85.2 ± 0.3 % | 95.1 ± 0.2 % | 385 (325 to 469) | 204 (163 to 256) |
| AdamW | 0.0001 | 5 of 5 | 48.8 ± 0.8 % | 48.8 % (10,000) | 86.5 ± 0.3 % | 95.7 ± 0.1 % | 194 (158 to 242) | 109 (97 to 122) |
| AdamW **(best on seen)** **(best on control)** | 0.0003 | 5 of 5 | 48.0 ± 0.5 % | 48.1 % (9,000) | 86.6 ± 0.1 % | 95.8 ± 0.1 % | 131 (116 to 148) | 73 (66 to 80) |
| AdamW | 0.001 | 5 of 5 | 46.5 ± 0.6 % | 46.6 % (9,000) | 85.7 ± 0.3 % | 95.7 ± 0.1 % | 90 (79 to 101) | 48 (44 to 50) |
| AdamW | 0.003 | 5 of 5 | 45.6 ± 0.9 % | 45.6 % (10,000) | 85.7 ± 0.3 % | 95.6 ± 0.1 % | 78 (65 to 106) | 35 (29 to 44) |
| Muon | 3e-05 | 5 of 5 | 45.0 ± 0.9 % | 45.0 % (10,000) | 84.8 ± 0.2 % | 95.0 ± 0.2 % | 975 (857 to 1,163) | 471 (406 to 523) |
| Muon | 0.0001 | 5 of 5 | 49.2 ± 0.6 % | 49.2 % (10,000) | 86.4 ± 0.2 % | 95.8 ± 0.1 % | 413 (358 to 467) | 200 (181 to 216) |
| Muon **(best on seen)** **(best on control)** | 0.0003 | 5 of 5 | 49.9 ± 0.3 % | 49.9 % (10,000) | 86.5 ± 0.2 % | 95.9 ± 0.1 % | 200 (167 to 235) | 112 (106 to 121) |
| Muon | 0.001 | 5 of 5 | 46.4 ± 0.8 % | 46.5 % (1,650) | 85.5 ± 0.2 % | 95.8 ± 0.1 % | 110 (93 to 122) | 69 (67 to 73) |
| Muon | 0.003 | 5 of 5 | 45.3 ± 0.9 % | 45.3 % (8,000) | 85.5 ± 0.3 % | 95.8 ± 0.1 % | 73 (60 to 96) | 45 (44 to 48) |
| SGD | 0.01 | 5 of 5 | 49.2 ± 0.7 % | 49.2 % (10,000) | 86.5 ± 0.2 % | 95.6 ± 0.1 % | 413 (351 to 479) | 250 (230 to 293) |
| SGD **(best on control)** | 0.03 | 5 of 5 | 49.5 ± 0.7 % | 49.5 % (9,500) | 86.8 ± 0.2 % | 95.8 ± 0.1 % | 208 (193 to 230) | 133 (123 to 153) |
| SGD **(best on seen)** | 0.1 | 5 of 5 | 48.1 ± 0.7 % | 48.1 % (10,000) | 86.7 ± 0.2 % | 95.9 ± 0.1 % | 130 (123 to 146) | 85 (79 to 95) |
| SGD | 0.3 | 5 of 5 | 48.1 ± 0.4 % | 48.1 % (10,000) | 86.4 ± 0.2 % | 95.7 ± 0.1 % | 126 (97 to 195) | 59 (56 to 64) |
| SGD | 1 | 5 of 5 | 40.4 ± 10.6 % | 40.6 % (7,500) | 72.4 ± 26.4 % | 80.5 ± 30.0 % | 271 (245 to 321) | 49 (46 to 55) |

Chance is 20 %.  The fixed rule (nearest class mean) on the same sequences: train 96.1 %, seen 95.3 %, control 86.4 %, dogs 54.8 %, wildcanids 39.0 %, cats 45.5 %, felines 62.1 %, birds 58.6 %, inet 86.5 %.

Accuracy is the mean over seeds, ± their standard deviation; steps are the mean and the range over seeds.  "Best on seen": the learning rate with the highest final accuracy on the seen classes (the figures use it); "best on control": on the classes left out of training.

## How the ability arrives, at the learning rates marked above (seen classes, means over seeds)

| optimizer | learning rate | training loss within 2 % of ln N until step | a tenth of the final gain at step | half | nine tenths |
|---|---|---|---|---|---|
| AdamW | 0.0003 | 56 | 51 | 73 | 122 |
| Muon | 0.0003 | 88 | 82 | 112 | 173 |
| SGD | 0.1 | 68 | 67 | 85 | 121 |
| SGD | 0.03 | 104 | 107 | 133 | 191 |

Left out of tradeoff.png (seeds more than five points apart on the seen classes): SGD lr 1.
