"""Compare and average fold models from several tags on the same 5 folds.

Usage: python blend.py TAG[:WEIGHT] ... [--allow-missing] [--done-only[=TAG,TAG]]
                       [--skip=TAG@FOLD,...] [--write] [--out=NAME]
Reports each fold's model MAEs and the MAE of their (weighted) mean, then the 5-fold OOF MAE.
  TAG:WEIGHT        relative weight of that tag's model within each fold (default 1).
  --allow-missing   a fold lacking some tags uses the tags it has.
  --done-only       skip runs whose log exists but never reached DONE (interrupted training);
                    --done-only=A,B applies this only to tags A and B.
  --skip=T@K        leave tag T out of fold K (e.g. a duplicate file under another tag).
  --write           (all 5 folds covered) writes submission_blend.csv, or --out=NAME.
Never touches submission.csv or existing checkpoints.
"""
import os
import sys
import numpy as np
import pandas as pd
from folds import get_folds


def opt(name):
    for x in sys.argv[1:]:
        if x == f"--{name}":
            return ""
        if x.startswith(f"--{name}="):
            return x.split("=", 1)[1]
    return None


specs = [a for a in sys.argv[1:] if not a.startswith("--")] or ["lstm", "lstm250s1"]
weights = {s.split(":")[0]: float(s.split(":")[1]) if ":" in s else 1.0 for s in specs}
tags = list(weights)
write = opt("write") is not None
allow_missing = opt("allow-missing") is not None
done_opt = opt("done-only")
done_tags = None if done_opt is None else (set(tags) if done_opt == "" else set(done_opt.split(",")))
skip = {tuple(s.split("@")) for s in (opt("skip") or "").split(",") if s}
out = opt("out") or "submission_blend.csv"

y = np.load("cache/ytr.npy"); m = np.load("cache/mtr.npy")
fold = get_folds(np.load("cache/rc_tr.npy"))


def mae(p, k):
    return np.abs(p - y[fold == k])[m[fold == k]].mean()


def finished(t, k):
    log = f"logs/{t}_fold{k}.log"
    return not os.path.exists(log) or "DONE" in open(log).read()


oof_blend = np.full_like(y, np.nan); tests = []
for k in range(5):
    have = [t for t in tags if os.path.exists(f"checkpoints/{t}_fold{k}_oof.npy") and (t, str(k)) not in skip]
    if done_tags is not None:
        unfinished = [t for t in have if t in done_tags and not finished(t, k)]
        if unfinished:
            print(f"fold {k}: skipping unfinished {unfinished}")
        have = [t for t in have if t not in unfinished]
    expected = [t for t in tags if (t, str(k)) not in skip]
    if len(have) < len(expected):
        print(f"fold {k}: missing {sorted(set(expected) - set(have))}" + (f" -> using {have} only" if allow_missing and have else ""))
        if not (allow_missing and have):
            continue
    w = np.array([weights[t] for t in have]); w = w / w.sum()
    oofs = [np.load(f"checkpoints/{t}_fold{k}_oof.npy") for t in have]
    blend = np.tensordot(w, np.stack(oofs), 1)
    oof_blend[fold == k] = blend
    tests.append(np.tensordot(w, np.stack([np.load(f"checkpoints/{t}_fold{k}_test.npy") for t in have]), 1))
    parts = "  ".join(f"{t}{'' if weights[t] == 1 else f'(w{weights[t]:g})'} {mae(o, k):.4f}" for t, o in zip(have, oofs))
    print(f"fold {k}: {parts}  | blend {mae(blend, k):.4f}")

done = ~np.isnan(oof_blend[:, 0])
if done.any():
    print(f"blend OOF MAE over {len(tests)} fold(s): {np.abs(oof_blend[done] - y[done])[m[done]].mean():.4f}")

if write:
    assert len(tests) == 5, "need all 5 folds covered before writing a submission"
    ss = pd.read_csv("data/sample_submission.csv")
    sub = pd.DataFrame({"id": np.load("cache/te_ids.npy"), "pressure": np.mean(tests, axis=0).reshape(-1)})
    sub = ss[["id"]].merge(sub, on="id", how="left")
    assert sub["pressure"].notna().all() and len(sub) == len(ss)
    assert out != "submission.csv", "refusing to overwrite submission.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, sub.shape)
