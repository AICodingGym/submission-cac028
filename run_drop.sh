#!/bin/bash
# Final round: dropout 0.2, 250 epochs, 2 seeds x 5 folds. Seed-1 fold 1 copied from exp_drop02 (identical config).
# Order: all seed-1 folds first (usable submission after ~5.5 h), then seed 2. Resumable; skips runs with DONE.
cd "$(dirname "$0")"
run() { tag=$1; k=$2; seed=$3
  grep -q "^DONE" logs/${tag}_fold$k.log 2>/dev/null && return
  python train.py --fold $k --epochs 250 --seed $seed --dropout 0.2 --tag $tag --resume >> logs/${tag}_fold$k.log 2>&1; }
for k in 0 2 3 4; do run drop02s1 $k 1; done
for k in 0 1 2 3 4; do run drop02s2 $k 2; done
echo ALL_DONE >> logs/run_drop.log
