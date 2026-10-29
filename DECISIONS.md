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
| D4b | Blur strength | SimCLR's σ 0.1–2.0; σ scaled to image size | **σ 0.1–1.15** (= 2.0 × 128/224) | SimCLR's σ is in pixels at 224px; at 128px the same σ blurs ~1.75× more relative to the content. The view preview with σ up to 2.0 (`runs/simclr_strong/views_preview_sigma2.png`) showed several unrecognizable views (Forest, Industrial). Scaling keeps the paper's relative blur. | Previews: `views_preview_sigma2.png` (before) vs `views_preview.png` (after) |
| D5 | Compute (Mac vs Colab) | All on Mac (MPS); SSL pretraining on Colab A100 | **Initially: SSL on Colab. Revised: everything on the Mac** | Measured on the Mac: supervised epoch 3.7 s, SimCLR epoch 6.7 s → 500 SimCLR epochs ≈ 1 h locally. Colab setup (data upload, notebook) would cost more time than it saves. | Supervised 100 epochs: 373 s; SimCLR: 6.7 s/epoch |
| D6 | Analysis experiments (which research questions to answer) | All six questions of Section 5; a subset | **Five of six:** SSL vs supervised (Exp 1 vs 2/4); augmentations (Exp 5: strong vs mild SimCLR); pretrain+fine-tune vs joint (Exp 2 vs 3); overfitting (train/val gaps of all runs + label-efficiency Exp 6); simple vs complex SSL (SimCLR vs DINO). **Backbone choice:** first skipped, later added with ViT + MAE (D11) | Each chosen question is answered by changing one thing against an existing run. The backbone question would need a second full set of SSL runs (≈2 h more) for a question less central to the assignment. | |
| D7 | Joint training design | λ ∈ {0.5, 1, 2}; epochs; batch size | **λ = 1, 300 epochs, batch 256, from random init**; classifier sees the baseline's mild views, SimCLR loss the strong views | One λ only (time). Batch 256 keeps the same 510 negatives as SimCLR pretraining. 300 epochs is between the baseline (100) and SSL + fine-tune (500 + 100). | |
| D8 | DINO adaptations to 1,920 images | Paper defaults (65,536 prototypes, EMA 0.996) | **1,024 prototypes, EMA 0.99 → 1, 2 global (128px) + 4 local (64px, 5–20% crops) views**; global views identical to SimCLR's | Far fewer images than ImageNet → far fewer "pseudo-classes"; only ~3,500 optimizer steps, so a slower EMA (0.996) would barely let the teacher move. Identical global views keep SimCLR vs DINO a comparison of objectives. | |
| D8b | DINO collapsed (first attempt) — what next? | Skip DINO and report the collapse; one more attempt with fixes | **One more attempt (`configs/dino_bn.yaml`): BatchNorm in the head + sharper teacher (T 0.04 → 0.02); automatic stop at epoch 25 if still collapsed** | Diagnosis: at random init the CNN gives nearly identical features for all images (mean cosine 0.89), so after centering the teacher logits barely differ (std 0.02) and the targets are almost uniform from step 1 (entropy 6.79 of max 6.93); the first run stayed there (loss ≈ log K). BatchNorm in the head rescales the between-image differences (init entropy 6.79 → 5.96); a lower teacher temperature sharpens them further. A learnable prototype scale was also tested and had no effect. If it collapses again, the collapse itself is reported. | First run: loss 6.89 ≈ log 1024 from epoch 3 on (`runs/dino/collapsed_epochs_1-13.txt`) |
| D8c | DINO second attempt: continue or stop? | Let it finish; stop now; decide at epoch 100 | **Decide at epoch 100: stop if kNN < 55%** | At epoch 25 it was no longer collapsed (target entropy 2.5, ~95 prototypes in use) but kNN was only 38.1% vs SimCLR's 58.8% at the same epoch. A rule fixed in advance avoids spending 2 h of compute on a run that cannot catch SimCLR (74%). | |
| D11 | Add masked image modeling (MAE)? | Skip; MAE with a ViT | **Add a ViT-Tiny (5.4M params, 16px patches → 64 tokens) trained supervised and with MAE (75% masking, 800 epochs), identical fine-tuning recipe** | Covers masked image modeling and answers the skipped question "does the backbone affect the benefit of SSL?". Expectation stated beforehand: a ViT from scratch on 1,920 images will be much weaker than the ResNet (no convolutional prior), so SSL may matter more for it; it will not be the best model. | |
| D9 | Supervised training-length control | none; 300-epoch supervised run | **Add a 300-epoch supervised run** | SSL models get 500 pretraining + 100 fine-tuning epochs; without this control an SSL gain could just be "more training". | |
| D10 | Label-efficiency experiment | none; 10/25/100% labels | **10% and 25% of labels (stratified, fixed seed), supervised vs SimCLR fine-tune, 200 epochs** | SSL is expected to help most when labels are scarcest; the encoder is pretrained without labels, so it is the same encoder for every fraction. | |
