import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
for df in (train, test):
    df["step"] = df.groupby("breath_id").cumcount()

keys = ["R", "C", "step"]
breaths = train["breath_id"].unique()
val_ids = np.random.default_rng(0).choice(breaths, len(breaths) // 5, replace=False)
is_val = train["breath_id"].isin(val_ids)

medians = train[~is_val].groupby(keys)["pressure"].median().rename("pred").reset_index()
val = train[is_val].merge(medians, on=keys, how="left")
val = val[val["u_out"] == 0]
print("Validation MAE:", (val["pressure"] - val["pred"]).abs().mean())

medians = train.groupby(keys)["pressure"].median().rename("pressure").reset_index()
sub = test.merge(medians, on=keys, how="left")[["id", "pressure"]]
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv:", sub.shape)
