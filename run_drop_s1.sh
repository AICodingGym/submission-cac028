#!/bin/bash
# Seed-1 dropout models only (seed 2 dropped for time). Waits for an already-running
# train.py (PID given as $1) to exit, then runs remaining seed-1 folds. Resumable; skips DONE.
cd "$(dirname "$0")"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 30; done
for k in 0 2 3 4; do
  grep -q "^DONE" logs/drop02s1_fold$k.log 2>/dev/null && continue
  python train.py --fold $k --epochs 250 --seed 1 --dropout 0.2 --tag drop02s1 --resume >> logs/drop02s1_fold$k.log 2>&1
done
echo ALL_DONE >> logs/run_drop_s1.log
