#!/bin/zsh
# Experiment 1c: supervised baseline + the best recipe's regularizers (label smoothing, Mixup/CutMix),
# 3 seeds, then test evaluation. Separates the effect of the regularizers from SimCLR pretraining.
# The final model is unchanged; this only adds a control row.
# Usage (from the repository root): zsh scripts/run_reg_control.sh 2>&1 | tee runs/reg_control.log

for seed in 0 1 2; do
  if [ $seed -eq 0 ]; then
    python train_classifier.py --config configs/sup_resnet18_reg.yaml
  else
    python train_classifier.py --config configs/sup_resnet18_reg.yaml --seed $seed
  fi
done
for run in sup_resnet18_reg sup_resnet18_reg_s1 sup_resnet18_reg_s2; do
  echo "\n===== $run ====="
  python evaluate.py --checkpoint runs/$run/best.pt --split val | head -1
  python evaluate.py --checkpoint runs/$run/best.pt --split test | head -1
done
