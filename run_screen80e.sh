#!/bin/bash
# Final screen queue (Nov 16, dropout 0.3 cut for time): baseline (running, PID $1) -> wd 1e-2 -> raw 9 features.
# Resumable; skips DONE. Never submits.
cd "$(dirname "$0")"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 20; done
run() { tag=$1; shift
  grep -q "^DONE" logs/${tag}_fold1.log 2>/dev/null && return
  python train.py --fold 1 --epochs 80 --dropout 0.2 --seed 1 --tag $tag --resume "$@" >> logs/${tag}_fold1.log 2>&1
  echo "$(date +%T) $tag: $(grep '^DONE' logs/${tag}_fold1.log)" >> logs/run_screen80.log; }
run scr80_base_s1
grep -q "scr80_base_s1: DONE" logs/run_screen80.log || echo "$(date +%T) scr80_base_s1: $(grep '^DONE' logs/scr80_base_s1_fold1.log)" >> logs/run_screen80.log
run scr80_wd1e2 --wd 1e-2
run scr80_raw9  --cache cache_raw9
{ echo "80-epoch screen, fold 1 (best val MAE), one change each vs baseline:"; for t in scr80_base_s1 scr80_wd1e2 scr80_raw9; do
  echo "  $t $(grep '^DONE' logs/${t}_fold1.log | awk '{print $NF}')"; done; } > logs/screen80_summary.log
echo "$(date +%T) ALL_DONE" >> logs/run_screen80.log
