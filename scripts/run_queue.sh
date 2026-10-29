#!/bin/zsh
# Runs Experiments 1b and 3-6 one after another (validation only; the test set is evaluated later).
# A failed run is reported and the queue continues with the next one.
# Usage (from the repository root): zsh scripts/run_queue.sh 2>&1 | tee runs/queue.log

run() {
  echo "\n===== $(date '+%H:%M:%S') START: $* ====="
  "$@" || echo "===== FAILED: $* ====="
}

run python train_joint.py --config configs/joint_simclr.yaml

run python train_ssl.py --config configs/dino.yaml
run python train_classifier.py --config configs/probe_dino.yaml
run python train_classifier.py --config configs/ft_dino.yaml

run python train_classifier.py --config configs/sup_resnet18_300ep.yaml

for pct in 10 25; do
  run python train_classifier.py --config configs/sup_${pct}pct.yaml
  run python train_classifier.py --config configs/ft_simclr_${pct}pct.yaml
done

run python train_ssl.py --config configs/simclr_mild.yaml
run python train_classifier.py --config configs/probe_simclr_mild.yaml
run python train_classifier.py --config configs/ft_simclr_mild.yaml

echo "\n===== $(date '+%H:%M:%S') QUEUE DONE ====="
