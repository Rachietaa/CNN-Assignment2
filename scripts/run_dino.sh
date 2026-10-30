#!/bin/zsh
# Experiment 4, second attempt: DINO with BatchNorm head and sharper teacher, then linear probe and
# fine-tune. train_ssl.py stops by itself at epoch 25 if the run has collapsed again (collapse_guard);
# the probe and fine-tune are then skipped.
# Usage (from the repository root): zsh scripts/run_dino.sh 2>&1 | tee runs/dino_bn.log

echo "===== $(date '+%H:%M:%S') START: DINO second attempt ====="
python train_ssl.py --config configs/dino_bn.yaml \
  && python train_classifier.py --config configs/probe_dino.yaml \
  && python train_classifier.py --config configs/ft_dino.yaml \
  && python evaluate.py --checkpoint runs/ft_dino/best.pt
echo "===== $(date '+%H:%M:%S') DINO DONE ====="
