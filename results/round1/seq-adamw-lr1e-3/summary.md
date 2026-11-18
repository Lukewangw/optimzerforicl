# Sequence models on the features of the encoder at step 107,859

| optimizer | learning rate | seeds finished | dogs: final accuracy | dogs: highest during training (at step) | control: final accuracy | seen classes: final accuracy | dogs: ability appears at step | seen: appears at step |
|---|---|---|---|---|---|---|---|---|
| AdamW | 3e-05 | 5 of 5 | 45.2 ± 0.7 % | 45.2 % (10,000) | 85.6 ± 0.1 % | 95.0 ± 0.1 % | 415 (274 to 523) | 201 (166 to 255) |
| AdamW | 0.0001 | 5 of 5 | 48.1 ± 0.6 % | 48.1 % (10,000) | 87.0 ± 0.1 % | 95.7 ± 0.1 % | 205 (174 to 248) | 105 (94 to 116) |
| AdamW **(best on control)** | 0.0003 | 5 of 5 | 48.4 ± 0.8 % | 48.5 % (9,500) | 87.2 ± 0.2 % | 96.0 ± 0.1 % | 131 (119 to 165) | 70 (65 to 79) |
| AdamW **(best on seen)** | 0.001 | 5 of 5 | 46.4 ± 0.8 % | 46.4 % (9,500) | 86.8 ± 0.2 % | 96.1 ± 0.1 % | 83 (70 to 98) | 47 (45 to 48) |
| AdamW | 0.003 | 5 of 5 | 46.0 ± 1.0 % | 46.0 % (10,000) | 86.6 ± 0.1 % | 95.7 ± 0.2 % | 91 (69 to 109) | 33 (32 to 35) |
| Muon | 3e-05 | 5 of 5 | 44.3 ± 1.1 % | 44.4 % (8,500) | 85.1 ± 0.2 % | 95.0 ± 0.3 % | 1,008 (827 to 1,156) | 447 (386 to 561) |
| Muon | 0.0001 | 5 of 5 | 48.0 ± 0.3 % | 48.0 % (9,000) | 86.7 ± 0.2 % | 95.6 ± 0.2 % | 425 (350 to 495) | 189 (174 to 224) |
| Muon **(best on control)** | 0.0003 | 5 of 5 | 49.1 ± 0.2 % | 49.1 % (9,500) | 86.9 ± 0.2 % | 96.0 ± 0.1 % | 213 (185 to 245) | 108 (103 to 120) |
| Muon | 0.001 | 5 of 5 | 47.5 ± 0.2 % | 47.5 % (10,000) | 86.6 ± 0.3 % | 96.2 ± 0.1 % | 133 (96 to 167) | 67 (65 to 73) |
| Muon **(best on seen)** | 0.003 | 5 of 5 | 45.7 ± 0.5 % | 45.7 % (10,000) | 86.3 ± 0.3 % | 96.4 ± 0.2 % | 79 (67 to 92) | 44 (42 to 48) |
| SGD | 0.01 | 5 of 5 | 47.7 ± 0.3 % | 47.7 % (10,000) | 86.5 ± 0.3 % | 95.6 ± 0.2 % | 381 (361 to 407) | 233 (211 to 267) |
| SGD | 0.03 | 5 of 5 | 48.7 ± 0.6 % | 48.7 % (10,000) | 86.9 ± 0.2 % | 95.7 ± 0.1 % | 207 (183 to 249) | 125 (117 to 139) |
| SGD **(best on seen)** **(best on control)** | 0.1 | 5 of 5 | 48.0 ± 0.2 % | 48.0 % (9,500) | 87.1 ± 0.1 % | 95.9 ± 0.1 % | 123 (119 to 128) | 80 (73 to 91) |
| SGD | 0.3 | 5 of 5 | 47.3 ± 0.6 % | 47.3 % (9,500) | 87.0 ± 0.2 % | 95.9 ± 0.1 % | 111 (86 to 129) | 54 (51 to 60) |
| SGD | 1 | 5 of 5 | 41.1 ± 10.8 % | 41.1 % (7,500) | 73.1 ± 26.9 % | 80.7 ± 30.1 % | 405 (278 to 593) | 44 (39 to 51) |

Chance is 20 %.  The fixed rule (nearest class mean) on the same sequences: train 96.8 %, seen 95.6 %, control 88.0 %, dogs 51.7 %, wildcanids 36.9 %, cats 44.9 %, felines 58.2 %, birds 59.6 %, inet 86.5 %.

Accuracy is the mean over seeds, ± their standard deviation; steps are the mean and the range over seeds.  "Best on seen": the learning rate with the highest final accuracy on the seen classes (the figures use it); "best on control": on the classes left out of training.

## How the ability arrives, at the learning rates marked above (seen classes, means over seeds)

| optimizer | learning rate | training loss within 2 % of ln N until step | a tenth of the final gain at step | half | nine tenths |
|---|---|---|---|---|---|
| AdamW | 0.001 | 30 | 31 | 47 | 97 |
| Muon | 0.003 | 30 | 31 | 44 | 66 |
| SGD | 0.1 | 64 | 61 | 80 | 107 |
| AdamW | 0.0003 | 52 | 49 | 70 | 118 |
| Muon | 0.0003 | 84 | 74 | 108 | 161 |

Left out of tradeoff.png (seeds more than five points apart on the seen classes): SGD lr 1.
