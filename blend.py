"""Compare and average fold models from several tags on the same 5 folds.

Usage: python blend.py lstm lstm250s1 [--write]
Reports, for each fold that has every tag, the MAE of each tag and of their mean.
With --allow-missing, a fold lacking some tags uses the mean of the tags it has.
With --write (all 5 folds covered), writes submission_blend.csv, or the file given by --out=NAME.
Never touches submission.csv or existing checkpoints.
"""
import os
import sys
import numpy as np
import pandas as pd
from folds import get_folds

args = [a for a in sys.argv[1:] if not a.startswith("--")]
write = "--write" in sys.argv
allow_missing = "--allow-missing" in sys.argv
done_only = "--done-only" in sys.argv
out = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--out=")), "submission_blend.csv")
tags = args or ["lstm", "lstm250s1"]

y = np.load("cache/ytr.npy"); m = np.load("cache/mtr.npy")
fold = get_folds(np.load("cache/rc_tr.npy"))


def mae(p, k):
    return np.abs(p - y[fold == k])[m[fold == k]].mean()


oof_blend = np.full_like(y, np.nan); tests = []
for k in range(5):
    have = [t for t in tags if os.path.exists(f"checkpoints/{t}_fold{k}_oof.npy")]
    if done_only:  # drop runs whose log exists but never reached DONE (interrupted mid-training)
        unfinished = [t for t in have if os.path.exists(f"logs/{t}_fold{k}.log")
                      and "DONE" not in open(f"logs/{t}_fold{k}.log").read()]
        if unfinished:
            print(f"fold {k}: skipping unfinished {unfinished}")
        have = [t for t in have if t not in unfinished]
    if len(have) < len(tags):
        print(f"fold {k}: missing {sorted(set(tags) - set(have))}" + (f" -> using {have} only" if allow_missing and have else ""))
        if not (allow_missing and have):
            continue
    oofs = [np.load(f"checkpoints/{t}_fold{k}_oof.npy") for t in have]
    blend = np.mean(oofs, axis=0)
    oof_blend[fold == k] = blend
    tests.append(np.mean([np.load(f"checkpoints/{t}_fold{k}_test.npy") for t in have], axis=0))
    parts = "  ".join(f"{t} {mae(o, k):.4f}" for t, o in zip(have, oofs))
    print(f"fold {k}: {parts}  | mean-blend {mae(blend, k):.4f}")

done = ~np.isnan(oof_blend[:, 0])
if done.any():
    nf = len(tests)
    print(f"blend OOF MAE over {nf} fold(s): {np.abs(oof_blend[done] - y[done])[m[done]].mean():.4f}"
          + ("  (compare: current 5-fold OOF 0.2332)" if nf == 5 else ""))

if write:
    assert len(tests) == 5, "need all 5 folds covered before writing a submission"
    ss = pd.read_csv("data/sample_submission.csv")
    sub = pd.DataFrame({"id": np.load("cache/te_ids.npy"), "pressure": np.mean(tests, axis=0).reshape(-1)})
    sub = ss[["id"]].merge(sub, on="id", how="left")
    assert sub["pressure"].notna().all() and len(sub) == len(ss)
    assert out != "submission.csv", "refusing to overwrite submission.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, sub.shape)
