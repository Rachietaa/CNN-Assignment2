#!/bin/zsh
# Experiment 7 (best-recipe candidates for the SimCLR ResNet-18) and the ViT follow-ups 8d-8f.
# A failed run is reported and the next one still runs.
# Usage (from the repository root): zsh scripts/run_best.sh 2>&1 | tee runs/best.log

run() {
  echo "\n===== $(date '+%H:%M:%S') START: $* ====="
  "$@" || echo "===== FAILED: $* ====="
}

run python train_classifier.py --config configs/ft_simclr_enclr.yaml
run python train_classifier.py --config configs/ft_simclr_enclr_mix.yaml

run python train_classifier.py --config configs/sup_vit_300ep.yaml
run python train_classifier.py --config configs/ft_mae_vit_300ep.yaml
run python train_ssl.py --config configs/mae_vit_m60.yaml
run python train_classifier.py --config configs/ft_mae_vit_m60.yaml

echo "\n===== $(date '+%H:%M:%S') BEST QUEUE DONE ====="
