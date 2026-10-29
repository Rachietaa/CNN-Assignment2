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
