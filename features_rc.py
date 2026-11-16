"""Build cache_rc/: the original 30 features (cache/) plus 3 RC time-constant features.

Physics motivation (generic, from the data description): a lung with resistance R and
compliance C responds like a first-order RC system with time constant tau = R*C.
  tau            R[cmH2O/L/s] * C[mL/cmH2O] / 1000  -> seconds (0.05 s .. 2.5 s)
  u_in_rc        u_in passed through a first-order low-pass with time constant tau
                 (y_t = y_{t-1} + (1 - exp(-dt/tau)) * (u_in_t - y_{t-1}), y_0 = u_in_0)
  u_in_minus_rc  u_in - u_in_rc  (how far the input is ahead of the lung's response)
New features are standardized with train mean/std. cache/ is not modified.
"""
import os
import numpy as np
import pandas as pd

T = 80
OUT = "cache_rc"


def rc_feats(df):
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)  # same order as features.py
    u = df["u_in"].values.reshape(-1, T)
    t = df["time_step"].values.reshape(-1, T)
    tau = (df["R"].values * df["C"].values / 1000.0).reshape(-1, T)
    dt = np.diff(t, axis=1, prepend=t[:, :1])
    alpha = 1.0 - np.exp(-dt / tau)
    y = np.empty_like(u)
    y[:, 0] = u[:, 0]
    for k in range(1, T):
        y[:, k] = y[:, k - 1] + alpha[:, k] * (u[:, k] - y[:, k - 1])
    return np.stack([tau, y, u - y], axis=-1).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    ftr = rc_feats(pd.read_csv("data/train.csv", usecols=["breath_id", "time_step", "u_in", "R", "C"]))
    fte = rc_feats(pd.read_csv("data/test.csv", usecols=["breath_id", "time_step", "u_in", "R", "C"]))
    mu = ftr.reshape(-1, 3).mean(0); sd = ftr.reshape(-1, 3).std(0) + 1e-6
    ftr = (ftr - mu) / sd; fte = (fte - mu) / sd
    for name, new in (("Xtr", ftr), ("Xte", fte)):
        base = np.load(f"cache/{name}.npy")
        assert base.shape[:2] == new.shape[:2], (base.shape, new.shape)
        np.save(f"{OUT}/{name}.npy", np.concatenate([base, new], axis=-1))
    print("cache_rc built:", np.load(f"{OUT}/Xtr.npy", mmap_mode="r").shape, "new feature mean/sd (raw):", mu, sd)


if __name__ == "__main__":
    main()
