# ITCS 6169/8169 — Assignment 2: Self-Supervised Visual Representation Learning

16-class scene recognition from 2,400 labelled training images, using self-supervised learning (SSL).
**No externally pretrained weights and no external data:** every network starts from random weights and is
trained only on the provided images. SimCLR, DINO and MAE are implemented from scratch in this repository.

## Final result

| | Accuracy |
|---|---:|
| **Test (400 images, evaluated once on the final model)** | **TEST_FINAL** |
| Validation (480 images, used for all model selection) | 84.6% |
| Supervised baseline, same network (validation, 3 seeds) | 79.2 ± 1.8% |

**Final model:** ResNet-18 (random init) pretrained with **SimCLR** on the 1,920 training images without
labels, then fine-tuned with labels using a 10× lower learning rate for the pretrained encoder, label
smoothing and Mixup/CutMix; ensemble of 3 fine-tuning seeds (probability average).

The full experimental history — including what did not work — is in [EXPERIMENTS.md](EXPERIMENTS.md); every
design decision and its reason is in [DECISIONS.md](DECISIONS.md); AI tool usage is in
[AI_USAGE.md](AI_USAGE.md).

## Main results

Validation accuracy (mean ± std over 3 seeds where shown); test accuracy evaluated once at the end.

| Method | Backbone | Val acc | Test acc |
|---|---|---:|---:|
| Supervised baseline | ResNet-18 | 79.2 ± 1.8 | TEST_SUP |
| SimCLR pretraining + fine-tuning | ResNet-18 | 82.1 ± 1.3 | TEST_SIMCLR |
| Joint cross-entropy + SimCLR (one stage) | ResNet-18 | 81.0 | TEST_JOINT |
| DINO-style self-distillation | ResNet-18 | collapsed; fixed version stopped (kNN 39%) | — |
| Supervised baseline | ViT-Tiny | 65.0 | TEST_SUPVIT |
| MAE pretraining + fine-tuning | ViT-Tiny | 70.0 | TEST_MAE |
| **Best recipe: SimCLR + gentle, regularized fine-tuning (3 seeds)** | ResNet-18 | **84.0 ± 0.1** | TEST_7B |
| **Final model: ensemble of those 3 seeds** | ResNet-18 | **84.6** | **TEST_FINAL** |

Label efficiency (validation): with 10% of the labels SimCLR + fine-tuning reaches 64.8% vs 51.5% supervised
(+13.3); with 25%, 72.3% vs 65.4% (+6.9).

## Final recipe

| Stage | Choice |
|---|---|
| Input | 128×128 grayscale (only the Flower class is in colour, a one-class shortcut), normalized to [−1, 1] |
| Backbone | torchvision ResNet-18, `weights=None`, 1-channel first convolution, 11.2M parameters |
| **SSL pretraining** | SimCLR: NT-Xent loss, temperature 0.2, batch 256 (510 negatives per view), projection head 512→512→128 |
| SSL views | random resized crop 20–100%, horizontal flip, ±10° rotation, brightness/contrast ±40% (p 0.8), Gaussian blur σ 0.1–1.15 (p 0.5); no vertical flip |
| SSL training | 1,920 training images, labels ignored; AdamW lr 1e-3, wd 1e-4, 10-epoch warm-up + cosine, 500 epochs; last epoch kept |
| **Fine-tuning** | new 512→16 linear layer; AdamW, lr 1e-3 (new layer) and **1e-4 (pretrained encoder)**, wd 0.05, 5-epoch warm-up + cosine, 100 epochs, batch 64 |
| Fine-tuning regularization | label smoothing 0.1; Mixup (α 0.2) or CutMix (α 1.0) on 50% of batches; crop 70–100%, flip, ±10°, brightness/contrast ±20% |
| Model selection | best validation epoch per run; final model = ensemble of seeds 0, 1, 2 |
| Configs | [configs/simclr_strong.yaml](configs/simclr_strong.yaml) → [configs/ft_simclr_enclr_mix.yaml](configs/ft_simclr_enclr_mix.yaml) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Hardware and software used: MacBook (Apple M5, 32 GB), PyTorch 2.14.1 on the Apple GPU (MPS), Python 3.14.
The device is chosen automatically (CUDA, then Apple MPS, then CPU). Seeds are fixed in every config.

## Data

Same dataset as Assignment 1. Unpack it (or symlink it) so that:

```text
data/
  train/<class_name>/*.jpg   2,400 images, 150 per class (split 80/20 into train/val by the code, seed 0)
  test/<class_name>/*.jpg    400 images, 25 per class
  test2/*.jpg                400 unlabeled images (not used for training)
```

The train/validation split is the same as in Assignment 1 (seeded random permutation, 1,920 / 480).

## Checkpoints

The SimCLR encoder and the three fine-tuned checkpoints of the final model are attached to the
[v1.0 GitHub release](https://github.com/Rachietaa/CNN-Assignment2/releases/tag/v1.0). Download them into
the run folders:

```bash
BASE=https://github.com/Rachietaa/CNN-Assignment2/releases/download/v1.0
mkdir -p runs/simclr_strong runs/ft_simclr_enclr_mix runs/ft_simclr_enclr_mix_s1 runs/ft_simclr_enclr_mix_s2
curl -L $BASE/simclr_encoder.pt -o runs/simclr_strong/encoder.pt
curl -L $BASE/final_seed0.pt -o runs/ft_simclr_enclr_mix/best.pt
curl -L $BASE/final_seed1.pt -o runs/ft_simclr_enclr_mix_s1/best.pt
curl -L $BASE/final_seed2.pt -o runs/ft_simclr_enclr_mix_s2/best.pt
```

## Reproducing the final result

**Evaluate the final model** (with the downloaded checkpoints):

```bash
CKPTS="runs/ft_simclr_enclr_mix/best.pt runs/ft_simclr_enclr_mix_s1/best.pt runs/ft_simclr_enclr_mix_s2/best.pt"
python ensemble.py --split val  --out runs/ens_final --checkpoints $CKPTS   # 84.6%
python ensemble.py --split test --out runs/ens_final --checkpoints $CKPTS   # TEST_FINAL
```

**Inference on new images** (e.g. the unlabeled `test2` folder) → CSV of predicted classes:

```bash
python predict.py --images data/test2 --out predictions_test2.csv --checkpoints $CKPTS
```

**Train the final model from scratch** (~1 h SSL pretraining + 3 × 7 min fine-tuning on an Apple M5):

```bash
python preview_views.py --config configs/simclr_strong.yaml            # optional: look at the SSL views
python train_ssl.py --config configs/simclr_strong.yaml                 # -> runs/simclr_strong/encoder.pt
python train_classifier.py --config configs/ft_simclr_enclr_mix.yaml               # seed 0
python train_classifier.py --config configs/ft_simclr_enclr_mix.yaml --seed 1      # -> runs/..._s1/
python train_classifier.py --config configs/ft_simclr_enclr_mix.yaml --seed 2      # -> runs/..._s2/
```

## Reproducing every experiment

Each experiment is one YAML config; outputs (history, curves, summary, evaluation JSON) go to
`runs/<name>/`. The queue scripts in [scripts/](scripts/) run them in the order used.

| # | Experiment | Command(s) |
|---|---|---|
| 1 | Supervised baseline (+ seeds 1, 2) | `python train_classifier.py --config configs/sup_resnet18.yaml [--seed 1]` |
| 1b | 300-epoch supervised control | `python train_classifier.py --config configs/sup_resnet18_300ep.yaml` |
| 2 | SimCLR pretraining, linear probe, fine-tune | `train_ssl.py` with `simclr_strong.yaml`; `train_classifier.py` with `probe_simclr_strong.yaml`, `ft_simclr_strong.yaml` |
| 3 | Joint CE + SimCLR | `python train_joint.py --config configs/joint_simclr.yaml` |
| 4 | DINO (collapsed) / DINO with fixes | `train_ssl.py` with `dino.yaml` / `dino_bn.yaml` ([scripts/run_dino.sh](scripts/run_dino.sh)) |
| 5 | SimCLR with mild views (augmentation ablation) | `simclr_mild.yaml`, `probe_simclr_mild.yaml`, `ft_simclr_mild.yaml` |
| 6 | Label efficiency (10% / 25% labels) | `sup_{10,25}pct.yaml`, `ft_simclr_{10,25}pct.yaml` |
| 7 | Best recipe | `ft_simclr_enclr.yaml`, `ft_simclr_enclr_mix.yaml` ([scripts/run_final_seeds.sh](scripts/run_final_seeds.sh)) |
| 8 | ViT-Tiny: supervised vs MAE (+ 300 epochs, 60% masking) | [scripts/run_mae.sh](scripts/run_mae.sh), [scripts/run_best.sh](scripts/run_best.sh) |

Summary of all runs and report figures:

```bash
python collect_results.py   # one table of every run, seed mean ± std
python make_figures.py      # figures/knn_during_pretraining.png, figures/label_efficiency.png
```

## Repository layout

```text
src/data.py          grayscale loading, fixed train/val split, augmentation, label-fraction subsets
src/models.py        ResNet-18 / ViT encoders (random init) + linear classifier
src/ssl.py           SimCLR (views, projection head, NT-Xent), DINO (multi-crop, head, loss, EMA), kNN monitor
src/vit.py           ViT-Tiny, MAE decoder, masking and reconstruction loss
src/engine.py        training loop (best-val checkpoint), evaluation, LR schedule, Mixup/CutMix
train_ssl.py         self-supervised pretraining (simclr | dino | mae)
train_classifier.py  supervised training, linear probe and fine-tuning
train_joint.py       joint cross-entropy + SimCLR training
evaluate.py          accuracy, per-class accuracy, confusion matrix of one checkpoint
ensemble.py          probability-averaging ensemble evaluation
predict.py           inference on a folder of images -> CSV
configs/             one YAML per experiment
runs/                per-run history, curves, summaries (checkpoints via the release)
```
