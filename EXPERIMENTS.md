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
| Does the backbone affect the benefit of SSL? | Exp 8: ViT-Tiny supervised vs MAE + fine-tune, against the ResNet-18 pair (Exp 1 vs 2b) |

## Results summary

| # | Experiment | Config | What changed | Val acc | Observation |
|---|---|---|---|---:|---|
| 1 | Supervised baseline | `configs/sup_resnet18.yaml` | ResNet-18, random init, cross-entropy only, 100 epochs | 77.5% (seed 0); **79.2 ± 1.8** (3 seeds) | Severe overfitting: train acc 99.9% vs val 77.5% (22-pt gap). Val loss stops improving after ~epoch 50 (min 0.93) while train loss → 0.008. Below the much smaller A1 SceneCNN (84.7%, 1.2M params), so more capacity without more labels does not help. Weakest: Bedroom 54%, OpenCountry 57%, Industrial 62%, InsideCity 63%. |
| 1b | Supervised, 3× longer (control) | `configs/sup_resnet18_300ep.yaml` | Exp 1 trained 300 epochs instead of 100 | 79.8% (1 seed) | Equal to the 3-seed mean of the 100-epoch baseline (79.2 ± 1.8): training longer does not clearly help. Val loss is the worst of all runs (min 0.98, final 1.01): longer supervised training mainly makes the model more over-confident. |
| 2 | SimCLR pretraining (no labels) | `configs/simclr_strong.yaml` | Same ResNet-18 pretrained with the SimCLR loss on the 1,920 train images, 500 epochs | kNN 74.2% (final encoder, no labels) | kNN accuracy of the label-free features: 58.8% (ep 25) → 68.8% (ep 100) → plateau at 73–75% from epoch ~200 while the SSL loss keeps falling (2.32 → 1.92). 57 min on the Mac. |
| 2a | SimCLR linear probe | `configs/probe_simclr_strong.yaml` | Frozen SimCLR encoder, only a new linear layer (8,208 weights) trained | 75.2% | Within 2.3 pts of the fully trained supervised network while training 0.07% of the weights. Lower val loss than any other run (0.71). |
| 2b | SimCLR + fine-tune | `configs/ft_simclr_strong.yaml` | SimCLR encoder, all layers fine-tuned; recipe identical to Exp 1 | **82.1 ± 1.3** (3 seeds) | **+2.9 over supervised (79.2 ± 1.8, 3 seeds)**; seed 0 alone suggested +4.6 because the supervised seed 0 was unlucky. Every SimCLR seed ≥ 80.8; supervised seeds 77.5–81.0. Faster learning (70.3 vs 66.0 at epoch 20). Seed-0 run: no class worse, biggest gains OpenCountry +14, Bedroom +10, Kitchen +10. |
| 3 | Joint CE + SimCLR | `configs/joint_simclr.yaml` | One stage from random init: cross-entropy + 1.0 × SimCLR loss, 300 epochs | 81.0% | +1.8 over the supervised 3-seed mean (79.2), −1.1 vs pretrain + fine-tune (82.1); both within seed noise (±1.3–1.8). Lowest val loss of the fine-tuned models (0.74). Different per-class profile: OpenCountry +21, Industrial +17, Flower +10, but Office −17, Forest −7, InsideCity −7 vs supervised. Still 100% train accuracy. |
| 4 (1st) | DINO, first attempt | `configs/dino.yaml` | DINO-style self-distillation, paper-style head, teacher T 0.04 | **collapsed** | Loss stuck at 6.89 ≈ log(1,024) from epoch 3: teacher targets uniform for every image, nothing to learn. Stopped at epoch 13. See Exp 4 section. |
| 5 | SimCLR, mild augmentation (ablation) | `configs/simclr_mild.yaml` (+ probe, fine-tune) | SimCLR views = A1's mild augmentation (crop 70–100%, ±20% brightness/contrast, no blur) | kNN 61.0% · probe 64.6% · fine-tune 77.7% | Easier SSL task (final loss 1.58 vs 1.92) but much worse features: −13 kNN, −11 probe, −4.4 fine-tune vs strong views. Fine-tune gain over supervised disappears (77.7 vs 77.5). |
| 6 | Label efficiency | `configs/{sup,ft_simclr}_{10,25}pct.yaml` | 10% / 25% of labels, supervised vs SimCLR fine-tune, 200 epochs | 10%: 51.5 → **64.8** · 25%: 65.4 → **72.3** | SSL gain grows as labels shrink: +13.3 (10%), +6.9 (25%), +2.9 (100%, 3-seed means). SimCLR with 10% labels ≈ supervised with 25%. |

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
| 125 | 2.57 | 72.7% |
| 150 | 2.47 | 70.2% |
| 175 | 2.36 | 70.2% |
| 200 | 2.32 | 72.3% |
| 225 | 2.24 | 75.4% |
| 250 | 2.18 | 73.5% |
| 275 | 2.14 | 72.9% |
| 300 | 2.05 | 73.1% |
| 325 | 2.03 | 73.1% |
| 350 | 1.99 | 73.1% |
| 375 | 1.96 | 72.7% |
| 400 | 1.93 | 74.8% |
| 425 | 1.93 | 73.3% |
| 450 | 1.92 | 74.0% |
| 475 | 1.92 | 74.2% |
| **500 (final encoder)** | **1.92** | **74.2%** |

**Early observation.** With zero labels, the features already support 68.8% nearest-neighbour accuracy
after 100 epochs, compared with 77.5% for the fully supervised baseline (which uses all labels and a
trained classifier). The SSL loss and kNN accuracy improve together, so the contrastive task is learning
scene-relevant features rather than a shortcut. Between epochs 100 and 150 the kNN accuracy moves up and
down (68.8 → 72.7 → 70.2%) while the loss keeps falling: the learning rate is still high, and with 480
validation images a single image is 0.2 points, so ±2-point swings are expected noise.

**Plateau.** From epoch ~200 to 375 the kNN accuracy stays flat at ~73% while the SSL loss keeps falling
(2.32 → 1.96). A lower contrastive loss no longer means better class information: once scenes are
separated, the encoder keeps improving at telling *individual images* apart (instance details: exact
textures, layouts), which does not help separate *classes*. With 1,920 images this point is reached early;
most of the gain comes in the first ~200 epochs. In the last 100 epochs (LR decaying to 0) kNN settles at
73–75%; final encoder 74.2%, best checkpoint 75.4% (epoch 225, not used: no label-based selection).

**Results.**

| Model | Trainable weights | Best val acc | Min val loss | Final train acc |
|---|---:|---:|---:|---:|
| Exp 1: supervised, random init | 11.2M | 77.5% | 0.93 | 99.9% |
| Exp 2a: SimCLR encoder frozen + linear layer | 8,208 | 75.2% | 0.71 | 95.2% |
| Exp 2b: SimCLR encoder fine-tuned | 11.2M | **82.1%** | 0.77 | 99.8% |

Per-class change, fine-tuned SimCLR vs supervised baseline (validation): OpenCountry 57 → 71 (+14),
Bedroom 54 → 64 (+10), Kitchen 73 → 83 (+10), TallBuilding 89 → 96 (+7), Industrial 62 → 69 (+7),
Store 71 → 77 (+6), Coast 73 → 79 (+6), Flower/Highway/Mountain +3 to +4; the other six classes unchanged.
**No class got worse.**

**Interpretation.**
- **Seed check (3 seeds each, fine-tuning seed varied, same SimCLR encoder):**

  | | Seed 0 | Seed 1 | Seed 2 | Mean ± std | Last-10-epoch mean | Val acc at epoch 20 |
  |---|---:|---:|---:|---:|---:|---:|
  | Supervised (Exp 1) | 77.5 | 81.0 | 79.2 | 79.2 ± 1.8 | 78.4 ± 1.6 | 66.0 ± 0.9 |
  | SimCLR + fine-tune (Exp 2b) | 82.1 | 80.8 | 83.3 | **82.1 ± 1.3** | **81.3 ± 1.2** | **70.3 ± 2.8** |

- **With all labels, SSL gives a modest but consistent gain: +2.9 points** (82.1 vs 79.2), about 1.5–2
  standard deviations; every SimCLR seed is at least 80.8%, while supervised seeds range 77.5–81.0. The
  single-seed comparison (+4.6) overstated it, because supervised seed 0 happened to be the worst of the
  three. Only the fine-tuning seed is varied here; the SimCLR pretraining itself was run once.
- **Training longer is not the explanation** (Exp 1b): 300 supervised epochs give 79.8%, the same as the
  100-epoch mean (79.2 ± 1.8), with worse validation loss.
- **The label-free features are already nearly as good as supervised training.** A single linear layer on
  frozen SimCLR features (75.2%) comes within 4 points of the fully trained supervised network (79.2 ± 1.8)
  while training 0.07% of the weights, and its
  validation loss is the lowest of all runs: its predictions are less over-confident when wrong.
- **SSL reduces, but does not remove, overfitting.** The fine-tuned model still reaches ~100% train accuracy,
  but generalizes better (min val loss 0.81 vs 0.87, 3-seed means; +2.9 pts). Pretraining gives a better starting point that
  fine-tuning then specializes from; the labels still get memorized.
- **The gains are on the hardest, most confused classes** (OpenCountry, Bedroom, Kitchen, Industrial), the
  scene types defined by global layout rather than a single object; the easy classes were already near
  their ceiling.
- **Still below the A1 SceneCNN (84.7%)**, a smaller network (1.2M parameters) trained with labels only. The
  larger ResNet-18 overfits more; SSL closes much of that gap but not all of it.


---

## Experiment 3 — Joint SimCLR + classification (one stage)

**Goal.** Does SSL help more as a separate pretraining stage, or as an extra loss trained together with the
classifier? Same encoder, same images, from random init.

**Reproduce.**
```bash
python train_joint.py --config configs/joint_simclr.yaml
python evaluate.py --checkpoint runs/joint_simclr/best.pt
```

**Objective.** loss = cross-entropy(classifier view) + λ · NT-Xent(two strong views), λ = 1, on one shared
encoder in one forward pass. The classifier sees the baseline's mild augmentation; the contrastive loss sees
SimCLR's strong views. Batch 256 (510 negatives, as in Exp 2), AdamW lr 1e-3, wd 0.05, 10-epoch warm-up,
cosine, 300 epochs (DECISIONS.md D7). ~10.5 s/epoch, 52 min.

**Results.**

| Model | Best val acc | Mean val acc, last 10 epochs | Min val loss | Final train acc |
|---|---:|---:|---:|---:|
| Exp 1: supervised (seed 0; 3-seed mean 79.2 ± 1.8) | 77.5% | 77.0% | 0.93 | 99.9% |
| Exp 2b: SimCLR pretrain → fine-tune | **82.1%** | **81.6%** | 0.77 | 99.8% |
| Exp 3: joint CE + SimCLR | 81.0% | 79.9% | **0.74** | 100% |

Per class vs supervised (val): OpenCountry 57 → 79, Industrial 62 → 79, Flower 84 → 94, Mountain 86 → 93,
Kitchen 73 → 80, Bedroom 54 → 64; but Office 88 → 71, Forest 93 → 86, InsideCity 63 → 57.

**Interpretation.**
- **Both ways of using SSL are above the supervised baseline** (joint 81.0%, one seed; two-stage 82.1 ± 1.3;
  supervised 79.2 ± 1.8). The joint run's +1.8 is within seed noise, so on its own it is not conclusive.
- **Two-stage is slightly better (+1.1 best, +1.7 on the last-10-epoch mean)**, but with one seed each this is
  within plausible run-to-run noise. A likely reason it is ahead: in pretraining the encoder first learns
  general features from the SSL signal alone; in joint training the cross-entropy term reaches ~0 within
  ~100 epochs (100% train accuracy), after which the classifier has memorized the labels and only the SSL
  term still shapes the features.
- **The SSL term regularizes but does not prevent memorization:** train accuracy still reaches 100%, but the
  validation loss is the lowest of all fine-tuned models (0.74).
- **Joint training trades classes rather than lifting all of them:** large gains on OpenCountry/Industrial,
  losses on Office/Forest/InsideCity. Two-stage SimCLR improved 10 classes and hurt none.
- The joint model also learns more slowly at first (train acc 82% vs 84% at epoch 20), consistent with the
  SSL term competing with the classification term early on.

---

## Experiment 4 — DINO-style self-distillation

**Idea.** No negative pairs. A student network is trained to match the output of a teacher network, where
the teacher is an exponential moving average (EMA) of the student and receives no gradients. Both output a
distribution over K = 1,024 learned prototypes. The teacher sees two global 128px views; the student sees
those plus four local 64px crops (5–20% of the image), so it must predict the teacher's output for the whole
scene from small parts of it. Collapse (all images → the same output) is prevented only by **centering**
(subtract a running mean of the teacher output, so no prototype wins for every image) and **sharpening**
(low teacher temperature, so the target is not uniform). Implemented from scratch, trained from random init
on the 1,920 training images only; no DINO weights are used.

### First attempt: collapsed (`configs/dino.yaml`)

| Epoch | SSL loss | H(teacher target) | H(batch mean) |
|---:|---:|---:|---:|
| 1 | 6.46 | 6.13 | 6.29 |
| 2 | 6.81 | 6.62 | 6.76 |
| 3 | 6.94 | 6.75 | 6.88 |
| 5–13 | 6.89–6.91 | 6.81 | 6.90–6.91 |

Maximum entropy is log(1,024) = 6.93. From epoch 3 the teacher's target for *every image* is nearly uniform
and the loss sits at ≈ log K: the student is asked to predict "nothing", so it learns nothing. The run was
stopped at epoch 13 (log: `runs/dino/collapsed_epochs_1-13.txt`).

**Diagnosis** (randomly initialized network, 128 training images): the CNN maps all images to nearly the same
feature vector (mean cosine similarity between different images 0.89). Centering subtracts this shared part,
leaving teacher logits that differ by a standard deviation of only 0.02 across prototypes; even divided by
T = 0.04 this gives near-uniform targets (entropy 6.79/6.93) from the very first step. DINO's mechanism needs
the images to already be somewhat distinguishable; SimCLR does not, because its negatives push images apart
directly. With ImageNet-scale data and ViTs (DINO's setting) the initial outputs differ more; with 1,920
grayscale scenes and a CNN they do not.

### Second attempt (`configs/dino_bn.yaml`, DECISIONS.md D8b)

Two changes: BatchNorm in the head (rescales the between-image differences; initial target entropy
6.79 → 5.96) and teacher temperature 0.04 → 0.02. A guard stops the run at epoch 25 if the target entropy is
still above 95% of the maximum. *Pending.*


---

## Experiment 5 — Which augmentations matter for SSL? (strong vs mild views)

**Question.** SimCLR's augmentations define what the encoder learns to ignore. Are the strong SimCLR-style
views necessary, or would the mild augmentation that worked best for *supervised* training in A1 do?

**Setup.** Identical to Exp 2 except the views: crop 70–100% (instead of 20–100%), brightness/contrast ±20%
always (instead of ±40% with p 0.8), no blur. Config: `configs/simclr_mild.yaml`.

| SimCLR views | Final SSL loss | kNN (no labels) | Linear probe | Fine-tune |
|---|---:|---:|---:|---:|
| Strong (Exp 2) | 1.92 | **74.2%** | **75.2%** | **82.1%** |
| Mild (Exp 5) | **1.58** | 61.0% | 64.6% | 77.7% |
| *Supervised baseline (Exp 1)* | | | | *77.5%* |

**Interpretation.**
- **Strong augmentation is essential: +13 kNN, +11 linear probe, +4.4 fine-tune.** With mild views the
  fine-tuned model is no better than the supervised baseline: the whole SSL benefit disappears.
- **A lower SSL loss does not mean better features.** The mild run reaches a *lower* contrastive loss (1.58
  vs 1.92) because two mildly augmented views of the same photo look almost identical and are trivially
  matched (low-level cues such as exact pixel layout and texture suffice). The strong views — different
  parts of the scene, different contrast, blur — can only be matched by recognizing *what the scene is*,
  which is the information the classifier needs. This is the same lesson as the epoch-200 kNN plateau in
  Exp 2: the SSL loss measures the pretext task, not representation quality.
- The mild run's kNN accuracy also plateaus much earlier and lower (~60% from epoch 125 vs ~73% from epoch
  200), so the gap is not a matter of training longer.
- Augmentations that help supervised training (A1) are not the ones SSL needs: supervised training needs
  views that preserve the label; contrastive SSL needs views that are as different as possible while still
  showing the same scene.

---

## Experiment 6 — Label efficiency: does SSL help more when labels are scarce?

**Setup.** Keep 10% or 25% of the training labels (stratified, same images for both methods), fine-tune
for 200 epochs. The SimCLR encoder is the same one as in Exp 2 — it was pretrained on all 1,920 training
images *without labels*, so only the fine-tuning sees fewer labels.

| Labelled images | Supervised | SimCLR + fine-tune | Gain |
|---|---:|---:|---:|
| 10% (192, ~12 per class) | 51.5% | **64.8%** | **+13.3** |
| 25% (479, ~30 per class) | 65.4% | **72.3%** | **+6.9** |
| 100% (1,920) | 79.2 ± 1.8 (3 seeds) | **82.1 ± 1.3** (3 seeds) | +2.9 |

**Interpretation.**
- **The fewer the labels, the more SSL helps:** +13.3 points with 10% of labels, +6.9 with 25%, and +2.9 with
  all of them (3-seed means at 100%; single seeds at 10% / 25%, where the gaps are far larger than the
  ±1–2 point seed noise). With few labels the supervised network can only memorize (100% train accuracy, val loss
  2.2); the SimCLR encoder already separates scenes, so a few labels are enough to name the clusters.
- **SimCLR with 10% of the labels (64.8%) matches supervised training with 25% (65.4%)**, i.e. about 2.5×
  fewer labels for the same accuracy; with 25% of labels SimCLR (72.3%) is within ~5 points of supervised
  training on all labels.
- SimCLR also learns faster with few labels: at epoch 20, 50.0% vs 36.2% (10%) and 61.7% vs 46.7% (25%).
- This answers "does SSL reduce overfitting?" most directly: in the label-scarce regime where overfitting is
  worst, SSL pretraining gives the largest improvement and much lower validation loss (1.39 vs 2.23 at 10%).
