#!/bin/zsh
# Experiment 8: does the backbone change the benefit of SSL? Supervised ViT-Tiny, MAE pretraining of the
# same ViT, then MAE fine-tune. A failed run is reported and the next one still runs.
# Usage (from the repository root): zsh scripts/run_mae.sh 2>&1 | tee runs/mae.log

run() {
  echo "\n===== $(date '+%H:%M:%S') START: $* ====="
  "$@" || echo "===== FAILED: $* ====="
}

run python train_classifier.py --config configs/sup_vit.yaml
run python train_ssl.py --config configs/mae_vit.yaml
run python train_classifier.py --config configs/ft_mae_vit.yaml
run python evaluate.py --checkpoint runs/sup_vit/best.pt
run python evaluate.py --checkpoint runs/ft_mae_vit/best.pt
echo "\n===== $(date '+%H:%M:%S') MAE DONE ====="
