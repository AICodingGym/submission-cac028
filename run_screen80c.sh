#!/bin/bash
# Trimmed 80-epoch screen, reordered (Nov 16): baseline -> wd 1e-2 -> dropout 0.3. The third
# candidate slot (R*C features vs a "9-feature" test) is pending the user's answer and is NOT run here.
# Waits for the running baseline (PID $1). Resumable; skips DONE. Never submits.
cd "$(dirname "$0")"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 20; done
run() { tag=$1; shift
  grep -q "^DONE" logs/${tag}_fold1.log 2>/dev/null && return
  python train.py --fold 1 --epochs 80 --dropout 0.2 --tag $tag --resume "$@" >> logs/${tag}_fold1.log 2>&1
  echo "$(date +%T) $tag: $(grep '^DONE' logs/${tag}_fold1.log)" >> logs/run_screen80.log; }
run scr80_base_s1 --seed 1
grep -q "scr80_base_s1: DONE" logs/run_screen80.log || echo "$(date +%T) scr80_base_s1: $(grep '^DONE' logs/scr80_base_s1_fold1.log)" >> logs/run_screen80.log
run scr80_wd1e2   --seed 1 --wd 1e-2
run scr80_drop03  --seed 1 --dropout 0.3
echo "$(date +%T) SLOT3_PENDING (baseline, wd, dropout0.3 done)" >> logs/run_screen80.log
