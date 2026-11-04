"""Reference: per-(R,C,step) median pressure, scored on the shared 5 folds."""
import numpy as np
from folds import get_folds

y = np.load("cache/ytr.npy"); m = np.load("cache/mtr.npy"); rc = np.load("cache/rc_tr.npy")
fold = get_folds(rc)
errs = []
for k in range(5):
    tr, va = fold != k, fold == k
    pred = np.zeros_like(y[va])
    for c in np.unique(rc):
        pred[rc[va] == c] = np.median(y[tr & (rc == c)], axis=0)
    e = np.abs(pred - y[va])[m[va]].mean(); errs.append(e)
    print(f"fold {k}: MAE {e:.4f}")
print(f"CV MAE {np.mean(errs):.4f}")
