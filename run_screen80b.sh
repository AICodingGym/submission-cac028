#!/bin/bash
# Trimmed 80-epoch screen (Nov 16, due tonight): baseline + 3 candidates on fold 1.
# Dropped from run_screen80.sh: second baseline seed and hidden 512. Waits for the
# running baseline (PID $1) first. Resumable; skips DONE. Never submits.
cd "$(dirname "$0")"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 20; done
run() { tag=$1; shift
  grep -q "^DONE" logs/${tag}_fold1.log 2>/dev/null && return
  python train.py --fold 1 --epochs 80 --dropout 0.2 --tag $tag --resume "$@" >> logs/${tag}_fold1.log 2>&1
  echo "$(date +%T) $tag: $(grep '^DONE' logs/${tag}_fold1.log)" >> logs/run_screen80.log; }
run scr80_base_s1 --seed 1                        # baseline (resumes if the waited-on run died)
echo "$(date +%T) scr80_base_s1: $(grep '^DONE' logs/scr80_base_s1_fold1.log)" >> logs/run_screen80.log
run scr80_rc      --seed 1 --cache cache_rc       # + R*C time-constant features
run scr80_wd1e2   --seed 1 --wd 1e-2              # real weight decay
run scr80_drop03  --seed 1 --dropout 0.3          # stronger dropout
{ echo "80-epoch screen, fold 1 (best val MAE):"; for t in scr80_base_s1 scr80_rc scr80_wd1e2 scr80_drop03; do
  echo "  $t $(grep '^DONE' logs/${t}_fold1.log | awk '{print $NF}')"; done; } > logs/screen80_summary.log
echo "$(date +%T) ALL_DONE" >> logs/run_screen80.log
