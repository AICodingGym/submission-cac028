#!/bin/bash
# Second model per fold: 250 epochs, seed 1, tag lstm250s1. Fold 1 already done; fold 0 skipped by choice.
# Resumable: skips folds whose log has DONE, resumes others from checkpoints/lstm250s1_fold{k}_last.pt.
cd "$(dirname "$0")"
for k in 2 3 4; do
  if grep -q "^DONE" logs/lstm250s1_fold$k.log 2>/dev/null; then continue; fi
  python train.py --fold $k --epochs 250 --seed 1 --tag lstm250s1 --resume >> logs/lstm250s1_fold$k.log 2>&1
done
echo ALL_DONE >> logs/run_s1.log
