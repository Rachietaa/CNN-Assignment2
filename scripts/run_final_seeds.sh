#!/bin/zsh
# Final model (Exp 7b recipe): two more fine-tuning seeds of configs/ft_simclr_enclr_mix.yaml, then the
# 3-seed ensemble on the validation split.
# Usage (from the repository root): zsh scripts/run_final_seeds.sh 2>&1 | tee runs/final_seeds.log

run() {
  echo "\n===== $(date '+%H:%M:%S') START: $* ====="
  "$@" || echo "===== FAILED: $* ====="
}

run python train_classifier.py --config configs/ft_simclr_enclr_mix.yaml --seed 1
run python train_classifier.py --config configs/ft_simclr_enclr_mix.yaml --seed 2
run python ensemble.py --split val --out runs/ens_final \
  --checkpoints runs/ft_simclr_enclr_mix/best.pt runs/ft_simclr_enclr_mix_s1/best.pt runs/ft_simclr_enclr_mix_s2/best.pt
echo "\n===== $(date '+%H:%M:%S') FINAL SEEDS DONE ====="
