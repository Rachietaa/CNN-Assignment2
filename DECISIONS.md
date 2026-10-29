# Decision Log

Every design decision in this assignment, why it was made, and what evidence supported it.
This feeds the report sections "SSL strategy" and "Experimental analysis".

## Fixed by the assignment rules

| Rule | Consequence |
|---|---|
| No externally pretrained weights | Every network starts from random initialization. The Assignment 1 final model (ImageNet-pretrained ConvNeXt V2) cannot be reused. |
| No external data | SSL pretraining uses only the provided images. |

## Carried over from Assignment 1 (evidence already collected)

| Decision | Why (A1 evidence) |
|---|---|
| Same 1,920 / 480 train/val split (seed 0) | Results stay comparable with the A1 from-scratch models (best: Step 3b, 84.7% val). |
| Grayscale input | Only the Flower class is in color; RGB input learned a "color = Flower" shortcut (A1 Step 3a). |
| No vertical flip, no large rotation | Both hurt in A1 (−2.5 and −4.7 pts): scenes have a fixed "up". |
| Test set used only for the final numbers | All model selection on validation. |

## Decisions made in Assignment 2

| # | Decision | Options considered | Choice | Why | Evidence / result |
|---|---|---|---|---|---|
| D1 | Backbone | ResNet-18 (random init); A1 SceneCNN; small ViT | **ResNet-18, random init, 128×128 grayscale input**, same backbone in every experiment | Standard backbone in the SSL literature (SimCLR, MoCo, BYOL all report ResNets), so results are easy to relate to papers; A1 already gives an ImageNet-pretrained ResNet-18 result for context; ViTs need far more data than 2,400 images when trained from scratch. 128px was the best from-scratch resolution in A1 (Step 3b). | |
| D2 | SSL method | SimCLR; BYOL/SimSiam; MAE | **SimCLR** (contrastive) | Simplest objective whose mechanism can be explained and tested directly: two augmented views of one image are pulled together, the other images in the batch are pushed apart. MAE would require a ViT (rejected in D1); BYOL/SimSiam add a predictor and stop-gradient whose collapse-avoidance is harder to analyse. | |
| D2b | Second SSL method | SimCLR only; SimCLR + DINO-style self-distillation | **SimCLR first, then DINO** | Answers "is a more complicated SSL method necessarily better?" with the same backbone, data and budget: contrastive (needs negatives) vs self-distillation (no negatives, student matches a momentum teacher). | |
| D3 | Images used for SSL pretraining | 1,920 train only; + 480 val images; + 400 unlabeled test2 | **Only the 1,920 training images of the fixed split** (labels ignored) | Validation and test images stay completely unseen, so validation remains a clean measure for model selection. Whether test2 may be used is unclear in the rules (it may be a held-out set), so it is not used. | |
| D4 | SSL augmentations | Strong SimCLR-style adapted to grayscale; A1's mild augmentation | **Strong (main run):** crop 20–100%, h-flip, ±10° rotation, brightness/contrast ±40% (p 0.8), Gaussian blur (p 0.5). **Mild (ablation):** A1's crop 70–100%, ±20% brightness/contrast, no blur | The augmentations define what the encoder learns to ignore. The images are grayscale, so SimCLR's key augmentation (color distortion) is unavailable; strong crops and blur must create enough difference between views. No vertical flip / large rotation (hurt in A1). The mild run tests whether augmentation strength matters for SSL. | |
| D5 | Compute (Mac vs Colab) | All on Mac (MPS); SSL pretraining on Colab A100 | **Initially: SSL on Colab. Revised: everything on the Mac** | Measured on the Mac: supervised epoch 3.7 s, SimCLR epoch 6.7 s → 500 SimCLR epochs ≈ 1 h locally. Colab setup (data upload, notebook) would cost more time than it saves. | Supervised 100 epochs: 373 s; SimCLR: 6.7 s/epoch |
| D6 | Analysis experiments | | | | |
