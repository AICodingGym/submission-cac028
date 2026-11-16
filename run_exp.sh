#!/bin/bash
# One-fold tests (fold 1, 250 epochs). Reference: lstm250s1 fold 1 (seed 1) = 0.2123.
# Each run changes exactly one thing vs the reference. Resumable; skips runs whose log has DONE.
cd "$(dirname "$0")"
run() { tag=$1; shift
  grep -q "^DONE" logs/${tag}_fold1.log 2>/dev/null && return
  python train.py --fold 1 --epochs 250 --tag $tag --resume "$@" >> logs/${tag}_fold1.log 2>&1; }
run exp_drop02  --seed 1 --dropout 0.2   # idea: dropout between LSTM layers (overfitting)
run exp_seed2   --seed 2                 # idea: 3rd seed for ensembling; also measures seed noise
run exp_aux025  --seed 1 --exp_w 0.25    # idea: auxiliary loss on expiratory steps
echo ALL_DONE >> logs/run_exp.log
