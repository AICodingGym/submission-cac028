#!/bin/bash
# Continue on GPU (Nov 16): dropout 0.2 seed 2 on folds 2 and 4, the only folds still at 2 full-length models.
# ~83 min each. Resumable; skips runs whose log has DONE. After each finished run rebuilds
#   submission_final4.csv  (A: full-length models)   submission_final4b.csv (B: A + aux-loss model, fold 1)
# from FINISHED models only. Never submits.
cd "$(dirname "$0")"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "$(date +%T) NO GPU - aborting" >> logs/run_gpu_next.log; exit 1; }
T="lstm250s0 lstm250s1 exp_seed2 drop02s1 drop02s2"
blend() { python blend.py $T --allow-missing --done-only=drop02s2 --write --out=submission_final4.csv > logs/blend_final4.log 2>&1
          python blend.py $T exp_aux025 --allow-missing --done-only=drop02s2 --write --out=submission_final4b.csv > logs/blend_final4b.log 2>&1
          echo "$(date +%T) blends rebuilt: A $(grep OOF logs/blend_final4.log) | B $(grep OOF logs/blend_final4b.log)" >> logs/run_gpu_next.log; }
run() { tag=$1; k=$2; seed=$3
  grep -q "^DONE" logs/${tag}_fold$k.log 2>/dev/null && return
  python train.py --fold $k --epochs 250 --seed $seed --dropout 0.2 --tag $tag --resume >> logs/${tag}_fold$k.log 2>&1
  grep -q "^DONE" logs/${tag}_fold$k.log && blend; }
run drop02s2 2 2
run drop02s2 4 2
echo "$(date +%T) ALL_DONE" >> logs/run_gpu_next.log
