#!/bin/zsh
# One-time test-set evaluation of every model in the report table (run once, after the final model was
# fixed on validation; DECISIONS.md D12b). Writes runs/<name>/eval_test.json and runs/ens_*/eval_test.json.
# Usage (from the repository root): zsh scripts/run_test.sh 2>&1 | tee runs/test.log

for run in sup_resnet18 sup_resnet18_s1 sup_resnet18_s2 \
           ft_simclr_strong ft_simclr_strong_s1 ft_simclr_strong_s2 \
           joint_simclr sup_vit ft_mae_vit \
           ft_simclr_enclr_mix ft_simclr_enclr_mix_s1 ft_simclr_enclr_mix_s2; do
  echo "\n===== $run ====="
  python evaluate.py --checkpoint runs/$run/best.pt --split test | head -1
done

echo "\n===== final model: ensemble of the 3 best-recipe seeds ====="
python ensemble.py --split test --out runs/ens_final \
  --checkpoints runs/ft_simclr_enclr_mix/best.pt runs/ft_simclr_enclr_mix_s1/best.pt runs/ft_simclr_enclr_mix_s2/best.pt
