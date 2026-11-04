"""Shared 5-fold split over breaths (stratified by R/C combo)."""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_folds(rc, n_splits=5, seed=42):
    fold = np.zeros(len(rc), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(rc, rc)):
        fold[va] = k
    return fold
