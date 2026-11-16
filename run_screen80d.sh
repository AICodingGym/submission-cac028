#!/bin/bash
# Third screen slot (Nov 16): raw 9 features (cache_raw9) instead of R*C features.
# Waits for run_screen80c.sh (PID $1) to finish baseline, wd 1e-2, dropout 0.3. Resumable. Never submits.
cd "$(dirname "$0")"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 20; done
tag=scr80_raw9
grep -q "^DONE" logs/${tag}_fold1.log 2>/dev/null || python train.py --fold 1 --epochs 80 --dropout 0.2 --seed 1 --cache cache_raw9 --tag $tag --resume >> logs/${tag}_fold1.log 2>&1
echo "$(date +%T) $tag: $(grep '^DONE' logs/${tag}_fold1.log)" >> logs/run_screen80.log
{ echo "80-epoch screen, fold 1 (best val MAE), one change each vs baseline:"; for t in scr80_base_s1 scr80_wd1e2 scr80_drop03 scr80_raw9; do
  echo "  $t $(grep '^DONE' logs/${t}_fold1.log | awk '{print $NF}')"; done; } > logs/screen80_summary.log
echo "$(date +%T) ALL_DONE" >> logs/run_screen80.log
