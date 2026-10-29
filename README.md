# ITCS 6169/8169 — Assignment 2: Self-Supervised Visual Representation Learning

16-class scene recognition from 2,400 labelled training images, using self-supervised learning (SSL).
**No externally pretrained weights and no external data:** every model is randomly initialized and
trained only on the provided images.

## Data

Same dataset as Assignment 1. Unpack it (or symlink it) so that:

```text
data/
  train/<class_name>/*.jpg   2,400 images, 150 per class (split 80/20 into train/val by the code, seed 0)
  test/<class_name>/*.jpg    400 images, 25 per class
  test2/*.jpg                400 unlabeled images
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

*(Results, recipe and reproduction steps will be added as experiments are completed.)*
