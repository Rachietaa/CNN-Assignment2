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
| D3 | Images used for SSL pretraining | | | | |
| D4 | SSL augmentations | | | | |
| D5 | Compute (Mac vs Colab) | All on Mac (MPS); SSL pretraining on Colab A100 | **SSL pretraining on Colab A100; baseline, fine-tuning and evaluation on Mac** | SSL needs many epochs (~500) with large batches; estimated 1–2 h on an A100 vs 5+ h on the Mac. A1 showed both machines give the same results within seed noise (94.4% Mac vs 94.1% A100). | |
| D6 | Analysis experiments | | | | |
