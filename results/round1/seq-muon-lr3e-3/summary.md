# Sequence models on the features of the encoder at step 107,859

| optimizer | learning rate | seeds finished | dogs: final accuracy | dogs: highest during training (at step) | control: final accuracy | seen classes: final accuracy | dogs: ability appears at step | seen: appears at step |
|---|---|---|---|---|---|---|---|---|
| AdamW | 3e-05 | 5 of 5 | 48.2 ± 1.0 % | 48.2 % (10,000) | 85.8 ± 0.3 % | 95.4 ± 0.2 % | 418 (350 to 500) | 221 (201 to 261) |
| AdamW | 0.0001 | 5 of 5 | 51.7 ± 0.6 % | 51.7 % (9,500) | 87.3 ± 0.2 % | 96.2 ± 0.1 % | 203 (182 to 244) | 115 (109 to 124) |
| AdamW **(best on control)** | 0.0003 | 5 of 5 | 52.3 ± 0.5 % | 52.3 % (10,000) | 87.9 ± 0.1 % | 96.5 ± 0.1 % | 127 (119 to 133) | 78 (73 to 86) |
| AdamW **(best on seen)** | 0.001 | 5 of 5 | 50.6 ± 0.8 % | 50.6 % (10,000) | 87.5 ± 0.4 % | 96.6 ± 0.2 % | 93 (77 to 106) | 54 (49 to 62) |
| AdamW | 0.003 | 5 of 5 | 49.9 ± 0.6 % | 49.9 % (9,000) | 86.5 ± 0.4 % | 96.1 ± 0.3 % | 94 (83 to 128) | 40 (36 to 46) |
| Muon | 3e-05 | 5 of 5 | 47.2 ± 0.8 % | 47.3 % (9,000) | 85.3 ± 0.3 % | 95.3 ± 0.2 % | 1,001 (934 to 1,131) | 526 (489 to 552) |
| Muon | 0.0001 | 5 of 5 | 51.6 ± 0.6 % | 51.6 % (9,500) | 87.3 ± 0.1 % | 96.2 ± 0.0 % | 420 (393 to 443) | 219 (207 to 231) |
| Muon **(best on control)** | 0.0003 | 5 of 5 | 53.4 ± 0.5 % | 53.5 % (9,000) | 87.7 ± 0.2 % | 96.4 ± 0.2 % | 231 (208 to 248) | 120 (114 to 126) |
| Muon | 0.001 | 5 of 5 | 51.1 ± 0.7 % | 51.2 % (9,500) | 87.0 ± 0.4 % | 96.6 ± 0.1 % | 114 (105 to 123) | 74 (72 to 79) |
| Muon **(best on seen)** | 0.003 | 5 of 5 | 50.5 ± 0.5 % | 50.5 % (8,500) | 86.9 ± 0.2 % | 96.7 ± 0.2 % | 77 (69 to 88) | 49 (47 to 53) |
| SGD | 0.01 | 5 of 5 | 51.2 ± 0.8 % | 51.2 % (8,000) | 87.0 ± 0.3 % | 96.1 ± 0.0 % | 412 (366 to 517) | 296 (280 to 335) |
| SGD | 0.03 | 5 of 5 | 52.3 ± 0.4 % | 52.3 % (10,000) | 87.4 ± 0.3 % | 96.3 ± 0.1 % | 225 (203 to 263) | 150 (143 to 168) |
| SGD | 0.1 | 5 of 5 | 52.6 ± 0.4 % | 52.6 % (9,500) | 87.4 ± 0.2 % | 96.4 ± 0.1 % | 131 (109 to 148) | 91 (88 to 98) |
| SGD **(best on control)** | 0.3 | 5 of 5 | 51.6 ± 0.4 % | 51.6 % (10,000) | 87.7 ± 0.1 % | 96.5 ± 0.2 % | 151 (89 to 229) | 62 (58 to 67) |
| SGD **(best on seen)** | 1 | 5 of 5 | 49.4 ± 0.6 % | 49.4 % (9,500) | 87.0 ± 0.4 % | 96.6 ± 0.1 % | 312 (259 to 409) | 59 (50 to 67) |

Chance is 20 %.  The fixed rule (nearest class mean) on the same sequences: train 97.0 %, seen 95.2 %, control 87.6 %, dogs 56.3 %, wildcanids 38.0 %, cats 42.5 %, felines 58.5 %, birds 61.1 %, inet 87.2 %.

Accuracy is the mean over seeds, ± their standard deviation; steps are the mean and the range over seeds.  "Best on seen": the learning rate with the highest final accuracy on the seen classes (the figures use it); "best on control": on the classes left out of training.

## How the ability arrives, at the learning rates marked above (seen classes, means over seeds)

| optimizer | learning rate | training loss within 2 % of ln N until step | a tenth of the final gain at step | half | nine tenths |
|---|---|---|---|---|---|
| AdamW | 0.001 | 40 | 38 | 54 | 106 |
| Muon | 0.003 | 40 | 36 | 49 | 76 |
| SGD | 1 | 54 | 46 | 59 | 309 |
| AdamW | 0.0003 | 62 | 58 | 78 | 132 |
| Muon | 0.0003 | 98 | 91 | 120 | 176 |
| SGD | 0.3 | 54 | 52 | 62 | 151 |
