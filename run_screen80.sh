#!/bin/bash
# 80-epoch screen on fold 1 (Nov 16). Baseline: hidden 256, dropout 0.2, wd 1e-5, original 30 features.
# Each candidate changes ONE thing vs baseline seed 1. A second baseline seed estimates seed noise at 80 epochs.
# Resumable; skips runs whose log has DONE. Writes a summary to logs/screen80_summary.log. Never submits.
cd "$(dirname "$0")"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "$(date +%T) NO GPU - aborting" >> logs/run_screen80.log; exit 1; }
run() { tag=$1; shift
  grep -q "^DONE" logs/${tag}_fold1.log 2>/dev/null && return
  python train.py --fold 1 --epochs 80 --dropout 0.2 --tag $tag --resume "$@" >> logs/${tag}_fold1.log 2>&1
  echo "$(date +%T) $tag: $(grep '^DONE' logs/${tag}_fold1.log)" >> logs/run_screen80.log; }
run scr80_base_s1 --seed 1                        # baseline
run scr80_rc      --seed 1 --cache cache_rc       # + R*C time-constant features (33 inputs)
run scr80_drop03  --seed 1 --dropout 0.3          # stronger dropout
run scr80_wd1e2   --seed 1 --wd 1e-2              # real weight decay (AdamW default 1e-2)
run scr80_base_s2 --seed 2                        # baseline, other seed -> noise estimate
run scr80_h512    --seed 1 --hidden 512           # wider (~2.3x slower, 10.8 GB peak)
{ echo "80-epoch screen, fold 1 (best val MAE):"; for t in scr80_base_s1 scr80_base_s2 scr80_rc scr80_drop03 scr80_wd1e2 scr80_h512; do
  echo "  $t $(grep '^DONE' logs/${t}_fold1.log | awk '{print $NF}')"; done; } > logs/screen80_summary.log
echo "$(date +%T) ALL_DONE" >> logs/run_screen80.log
