# Experiment Log

All numbers are validation accuracy (480 images, fixed seed-0 split, same as Assignment 1) unless marked
"test". The test set (400 images) is evaluated only at the end, once per reported model.

## Research questions → experiments

| Question (assignment Section 4/5) | Experiments compared |
|---|---|
| Required: supervised baseline | Exp 1 (+ Exp 1b, 3× longer training, as a control) |
| Required: SSL pretraining + fine-tuning | Exp 2 (SimCLR) and Exp 4 (DINO): linear probe and full fine-tune |
| Required: best training recipe | Exp 7, built from the results below |
| Does SSL improve accuracy with only 2,400 images? | Exp 1 / 1b vs Exp 2b / 4b |
| Which augmentations are useful for SSL here? | Exp 2 (strong views) vs Exp 5 (A1's mild views) |
| Pretrain + fine-tune vs joint SSL + classification? | Exp 2b vs Exp 3 |
| Does SSL reduce overfitting? | Train/val gaps of all runs; Exp 6 (10% / 25% labels) |
| Is a more complicated SSL method better? | Exp 2 (SimCLR) vs Exp 4 (DINO) |
| Does the backbone affect the benefit of SSL? | Not investigated (see DECISIONS.md, D6) |

## Results summary

| # | Experiment | Config | What changed | Val acc | Observation |
|---|---|---|---|---:|---|
| 1 | Supervised baseline | `configs/sup_resnet18.yaml` | ResNet-18, random init, cross-entropy only, 100 epochs | 77.5% | Severe overfitting: train acc 99.9% vs val 77.5% (22-pt gap). Val loss stops improving after ~epoch 50 (min 0.93) while train loss → 0.008. Below the much smaller A1 SceneCNN (84.7%, 1.2M params), so more capacity without more labels does not help. Weakest: Bedroom 54%, OpenCountry 57%, Industrial 62%, InsideCity 63%. |
| 2 | SimCLR pretraining (no labels) | `configs/simclr_strong.yaml` | Same ResNet-18 pretrained with the SimCLR loss on the 1,920 train images, 500 epochs | *running* (kNN 68.8% at epoch 100) | kNN accuracy of the label-free features rises steadily: 58.8% (ep 25) → 68.8% (ep 100). |

---

## Experiment 1 — Supervised baseline

**Goal.** Reference point for every SSL experiment: the same ResNet-18 backbone trained with labels only.

**Reproduce.**
```bash
python train_classifier.py --config configs/sup_resnet18.yaml
python evaluate.py --checkpoint runs/sup_resnet18/best.pt
```

| Item | Value |
|---|---|
| Backbone | ResNet-18 (torchvision, `weights=None`), 1-channel input, 11.2M parameters |
| Input | 128×128 grayscale, normalized to [-1, 1] |
| Augmentation | random resized crop 70–100%, h-flip, ±10° rotation, brightness/contrast ±20% (A1's best from-scratch setting) |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05, batch 64 |
| Schedule | 5-epoch linear warm-up, cosine decay to 0, 100 epochs |
| Model selection | weights from the epoch with the best validation accuracy |
| Hardware / time | Apple M5 (MPS), 373 s |

**Result.** Best val 77.5% (epoch 95); final train accuracy 99.9%.

**Interpretation.** With 1,920 labelled images the network memorizes the training set: it reaches ~94%
train accuracy by epoch 40 while validation stays ~70%. The labels alone do not contain enough signal to
constrain 11M parameters. This is the gap SSL is meant to close: if pretraining without labels learns
general visual features first, the labels only have to fit a small adaptation on top.

---

## Experiment 2 — SimCLR pretraining, then linear probe and fine-tuning

**Goal.** Learn the encoder without labels, then measure the features in two ways: a linear probe (frozen
encoder, only a new linear layer trained) shows how good the features are as they are; full fine-tuning
(identical recipe to Exp 1, only the starting weights differ) shows whether SSL helps the final classifier.

**Reproduce.**
```bash
python preview_views.py --config configs/simclr_strong.yaml
python train_ssl.py --config configs/simclr_strong.yaml
python train_classifier.py --config configs/probe_simclr_strong.yaml
python train_classifier.py --config configs/ft_simclr_strong.yaml
python evaluate.py --checkpoint runs/ft_simclr_strong/best.pt
```

**The objective.** Each image is augmented twice. Both views go through the encoder and a small MLP
projection head; the NT-Xent loss treats the two views of the same image as the positive pair and the
other 2·256 − 2 = 510 views in the batch as negatives, i.e. a 511-way classification "which of these is my
partner?" using cosine similarity / temperature 0.2. To solve it, the encoder must map different crops,
brightness levels and blur of the same scene to nearby points, and different scenes apart. The projection
head is discarded afterwards; the 512-d encoder features are used downstream.

**Adaptations to this dataset** (see DECISIONS.md D3, D4, D4b):
- Grayscale images → no color distortion, which is SimCLR's most important augmentation on ImageNet. Strong
  crops (20–100% of the area), brightness/contrast ±40% and blur have to create the view differences.
- No vertical flip or large rotation (both hurt in A1: scenes have a fixed "up").
- Blur σ scaled from SimCLR's 0.1–2.0 at 224px to 0.1–1.15 at 128px. The view preview showed σ up to 2.0
  made several views unrecognizable (`runs/simclr_strong/views_preview_sigma2.png` vs `views_preview.png`).
- Only the 1,920 training images; validation and test images are never seen during pretraining.

| Item | Value |
|---|---|
| Encoder | ResNet-18, random init (same as Exp 1) |
| Projection head | 512 → 512 → 128 MLP with BatchNorm |
| Batch / negatives | 256 images → 512 views, 510 negatives per view |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4; 10-epoch warm-up, cosine decay, 500 epochs |
| Checkpoint | last epoch (no label-based selection of the SSL encoder) |
| Hardware / time | Apple M5 (MPS), ~7 s/epoch → ~1 h |

**Representation quality during pretraining** (kNN classifier on frozen features, k = 20, cosine
similarity; labels used only for this measurement, never for training the encoder):

| Epoch | SSL loss | kNN val acc |
|---:|---:|---:|
| 25 | 3.39 | 58.8% |
| 50 | 2.98 | 63.5% |
| 75 | 2.83 | 66.0% |
| 100 | 2.66 | 68.8% |
| 500 | | *pending* |

**Early observation.** With zero labels, the features already support 68.8% nearest-neighbour accuracy
after 100 epochs, compared with 77.5% for the fully supervised baseline (which uses all labels and a
trained classifier). The SSL loss and kNN accuracy improve together, so the contrastive task is learning
scene-relevant features rather than a shortcut.

**Result.** *pending:* linear probe, fine-tune.
