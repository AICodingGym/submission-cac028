#!/bin/bash
# Train all 5 folds sequentially (resumable). Logs in logs/, checkpoints in checkpoints/.
cd "$(dirname "$0")"
for k in 0 1 2 3 4; do
  if grep -q "^DONE" logs/lstm_fold$k.log 2>/dev/null; then continue; fi
  python train.py --fold $k --epochs 150 --resume >> logs/lstm_fold$k.log 2>&1
done
echo ALL_DONE >> logs/run_all.log
