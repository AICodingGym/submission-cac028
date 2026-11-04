"""Build per-breath feature tensors (N, 80, F) and cache them as .npy."""
import os
import numpy as np
import pandas as pd

CACHE = "cache"
T = 80


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    g = df.groupby("breath_id")
    out = pd.DataFrame(index=df.index)
    out["u_in"] = df["u_in"]
    out["u_out"] = df["u_out"]
    out["time_step"] = df["time_step"]
    out["dt"] = g["time_step"].diff().fillna(0)
    out["step"] = g.cumcount() / (T - 1)
    # volume proxy: integral of u_in over time (inspiration valve opening ~ flow)
    out["vol"] = (df["u_in"] * out["dt"]).groupby(df["breath_id"]).cumsum()
    out["u_in_cumsum"] = g["u_in"].cumsum()
    for k in (1, 2, 3, 4):
        out[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0)
    for k in (1, 2):
        out[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0)
    out["u_in_diff1"] = out["u_in"] - out["u_in_lag1"]
    out["u_in_diff2"] = out["u_in"] - out["u_in_lag2"]
    out["u_out_lag1"] = g["u_out"].shift(1).fillna(0)
    out["u_out_lead1"] = g["u_out"].shift(-1).fillna(1)
    out["u_in_roll_mean4"] = g["u_in"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    out["u_in_breath_max"] = g["u_in"].transform("max")
    out["u_in_breath_mean"] = g["u_in"].transform("mean")
    # physics-motivated: P ~ P0 + R*flow + V/C
    out["R_u_in"] = df["R"] * df["u_in"]
    out["vol_over_C"] = out["vol"] / df["C"]
    out["R_val"] = df["R"]
    out["C_val"] = df["C"]
    for r in (5, 20, 50):
        out[f"R_{r}"] = (df["R"] == r).astype(np.float32)
    for c in (10, 20, 50):
        out[f"C_{c}"] = (df["C"] == c).astype(np.float32)
    return df, out


def build():
    os.makedirs(CACHE, exist_ok=True)
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    tr, ftr = make_features(tr)
    te, fte = make_features(te)
    cols = list(ftr.columns)
    onehot = [c for c in cols if c.startswith(("R_5", "R_20", "R_50", "C_10", "C_20", "C_50")) or c in ("u_out", "u_out_lag1", "u_out_lead1")]
    num = [c for c in cols if c not in onehot]
    mu = ftr[num].mean()
    sd = ftr[num].std() + 1e-6
    ftr[num] = (ftr[num] - mu) / sd
    fte[num] = (fte[num] - mu) / sd
    Xtr = ftr.values.astype(np.float32).reshape(-1, T, len(cols))
    Xte = fte.values.astype(np.float32).reshape(-1, T, len(cols))
    ytr = tr["pressure"].values.astype(np.float32).reshape(-1, T)
    mtr = (tr["u_out"].values == 0).reshape(-1, T)
    mte = (te["u_out"].values == 0).reshape(-1, T)
    np.save(f"{CACHE}/Xtr.npy", Xtr); np.save(f"{CACHE}/Xte.npy", Xte)
    np.save(f"{CACHE}/ytr.npy", ytr); np.save(f"{CACHE}/mtr.npy", mtr); np.save(f"{CACHE}/mte.npy", mte)
    np.save(f"{CACHE}/rc_tr.npy", tr["R"].values.reshape(-1, T)[:, 0] * 100 + tr["C"].values.reshape(-1, T)[:, 0])
    np.save(f"{CACHE}/breath_tr.npy", tr["breath_id"].values.reshape(-1, T)[:, 0])
    np.save(f"{CACHE}/te_ids.npy", te["id"].values)
    print("features:", len(cols), cols)
    print("Xtr", Xtr.shape, "Xte", Xte.shape)


if __name__ == "__main__":
    build()
