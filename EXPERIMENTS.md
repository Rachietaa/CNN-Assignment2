# Experiment Log

All numbers are validation accuracy (480 images, fixed seed-0 split, same as Assignment 1) unless marked
"test". The test set (400 images) is evaluated only at the end, once per reported model.

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
