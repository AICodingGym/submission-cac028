"""Combine fold models: report OOF CV MAE and write submission.csv (mean of fold test preds)."""
import sys, glob
import numpy as np
import pandas as pd
from folds import get_folds

tag = sys.argv[1] if len(sys.argv) > 1 else "lstm"
y = np.load("cache/ytr.npy"); m = np.load("cache/mtr.npy"); fold = get_folds(np.load("cache/rc_tr.npy"))
oof = np.full_like(y, np.nan); tests = []
for k in range(5):
    f = f"checkpoints/{tag}_fold{k}_oof.npy"
    if not glob.glob(f):
        print(f"fold {k}: missing"); continue
    oof[fold == k] = np.load(f); tests.append(np.load(f"checkpoints/{tag}_fold{k}_test.npy"))
    print(f"fold {k}: MAE {np.abs(oof[fold == k] - y[fold == k])[m[fold == k]].mean():.4f}")
have = ~np.isnan(oof[:, 0])
print(f"OOF MAE over {len(tests)} folds: {np.abs(oof[have] - y[have])[m[have]].mean():.4f}")
pred = np.mean(tests, axis=0).reshape(-1)
ids = np.load("cache/te_ids.npy")
sub = pd.DataFrame({"id": ids, "pressure": pred})
ss = pd.read_csv("data/sample_submission.csv")
sub = ss[["id"]].merge(sub, on="id", how="left")
assert sub["pressure"].notna().all() and len(sub) == len(ss)
sub.to_csv("submission.csv", index=False)
print("wrote submission.csv", sub.shape)
