# Sequence models across encoders

## Learning rate chosen on the encoder's own classes (`seen`)

| encoder | sequence model | learning rate | dogs | control | seen | appears at step (seen) |
|---|---|---|---|---|---|---|
| ViT trained with AdamW (validation 71.3 %) | AdamW | 0.001 | 46.4 ± 0.8 % | 86.8 ± 0.2 % | 96.1 ± 0.1 % | 47 |
| ViT trained with AdamW (validation 71.3 %) | Muon | 0.003 | 45.7 ± 0.5 % | 86.3 ± 0.3 % | 96.4 ± 0.2 % | 44 |
| ViT trained with AdamW (validation 71.3 %) | SGD | 0.1 | 48.0 ± 0.2 % | 87.1 ± 0.1 % | 95.9 ± 0.1 % | 80 |
| ViT trained with AdamW (validation 71.3 %) | nearest class mean (no training) | | 51.7 % | 88.0 % | 95.6 % | |
| ViT trained with Muon (validation 73.4 %) | AdamW | 0.001 | 50.6 ± 0.8 % | 87.5 ± 0.4 % | 96.6 ± 0.2 % | 54 |
| ViT trained with Muon (validation 73.4 %) | Muon | 0.003 | 50.5 ± 0.5 % | 86.9 ± 0.2 % | 96.7 ± 0.2 % | 49 |
| ViT trained with Muon (validation 73.4 %) | SGD | 1 | 49.4 ± 0.6 % | 87.0 ± 0.4 % | 96.6 ± 0.1 % | 59 |
| ViT trained with Muon (validation 73.4 %) | nearest class mean (no training) | | 56.3 % | 87.6 % | 95.2 % | |
| ViT trained with SGD (validation 69.8 %) | AdamW | 0.0003 | 48.0 ± 0.5 % | 86.6 ± 0.1 % | 95.8 ± 0.1 % | 73 |
| ViT trained with SGD (validation 69.8 %) | Muon | 0.0003 | 49.9 ± 0.3 % | 86.5 ± 0.2 % | 95.9 ± 0.1 % | 112 |
| ViT trained with SGD (validation 69.8 %) | SGD | 0.1 | 48.1 ± 0.7 % | 86.7 ± 0.2 % | 95.9 ± 0.1 % | 85 |
| ViT trained with SGD (validation 69.8 %) | nearest class mean (no training) | | 54.8 % | 86.4 % | 95.3 % | |

## Learning rate chosen on the left-out classes (`control`)

| encoder | sequence model | learning rate | dogs | control | seen | appears at step (seen) |
|---|---|---|---|---|---|---|
| ViT trained with AdamW (validation 71.3 %) | AdamW | 0.0003 | 48.4 ± 0.8 % | 87.2 ± 0.2 % | 96.0 ± 0.1 % | 70 |
| ViT trained with AdamW (validation 71.3 %) | Muon | 0.0003 | 49.1 ± 0.2 % | 86.9 ± 0.2 % | 96.0 ± 0.1 % | 108 |
| ViT trained with AdamW (validation 71.3 %) | SGD | 0.1 | 48.0 ± 0.2 % | 87.1 ± 0.1 % | 95.9 ± 0.1 % | 80 |
| ViT trained with AdamW (validation 71.3 %) | nearest class mean (no training) | | 51.7 % | 88.0 % | 95.6 % | |
| ViT trained with Muon (validation 73.4 %) | AdamW | 0.0003 | 52.3 ± 0.5 % | 87.9 ± 0.1 % | 96.5 ± 0.1 % | 78 |
| ViT trained with Muon (validation 73.4 %) | Muon | 0.0003 | 53.4 ± 0.5 % | 87.7 ± 0.2 % | 96.4 ± 0.2 % | 120 |
| ViT trained with Muon (validation 73.4 %) | SGD | 0.3 | 51.6 ± 0.4 % | 87.7 ± 0.1 % | 96.5 ± 0.2 % | 62 |
| ViT trained with Muon (validation 73.4 %) | nearest class mean (no training) | | 56.3 % | 87.6 % | 95.2 % | |
| ViT trained with SGD (validation 69.8 %) | AdamW | 0.0003 | 48.0 ± 0.5 % | 86.6 ± 0.1 % | 95.8 ± 0.1 % | 73 |
| ViT trained with SGD (validation 69.8 %) | Muon | 0.0003 | 49.9 ± 0.3 % | 86.5 ± 0.2 % | 95.9 ± 0.1 % | 112 |
| ViT trained with SGD (validation 69.8 %) | SGD | 0.03 | 49.5 ± 0.7 % | 86.8 ± 0.2 % | 95.8 ± 0.1 % | 133 |
| ViT trained with SGD (validation 69.8 %) | nearest class mean (no training) | | 54.8 % | 86.4 % | 95.3 % | |

Chance is 20 %.  Accuracy is the mean over seeds ± their standard deviation at the end of the sequence model's training.  "Appears": the first step at which accuracy on the seen classes is half-way from chance to the run's final value.
