#!/bin/bash
# 4-hour GPU plan, in priority order. Resumable; skips runs whose log has DONE.
#  1) finish fold 4 seed-1 dropout model (~15 min)
#  2) dropout seed 2 on fold 0 (~83 min)   3) dropout seed 2 on fold 3 (~83 min)
# Fold order (after review): add models where folds have the fewest; each fold is 1/5 of test.
# After each finished run, rebuilds submission_final2.csv from FINISHED models only (never submits).
cd "$(dirname "$0")"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU - aborting" | tee -a logs/run_gpu4h.log; exit 1; }
blend() { python blend.py lstm250s0 lstm250s1 exp_seed2 drop02s1 drop02s2 exp_aux025 lstm:0.25 --skip=lstm@0 --allow-missing --done-only=drop02s2 --write --out=submission_final2.csv > logs/blend_final2.log 2>&1
          echo "$(date +%T) blend rebuilt: $(grep OOF logs/blend_final2.log)" >> logs/run_gpu4h.log; }
run() { tag=$1; k=$2; seed=$3
  grep -q "^DONE" logs/${tag}_fold$k.log 2>/dev/null && return
  python train.py --fold $k --epochs 250 --seed $seed --dropout 0.2 --tag $tag --resume >> logs/${tag}_fold$k.log 2>&1
  grep -q "^DONE" logs/${tag}_fold$k.log && blend; }
run drop02s1 4 1
run drop02s2 0 2
run drop02s2 3 2
echo "$(date +%T) ALL_DONE" >> logs/run_gpu4h.log
