"""splitting.py — stratified k-fold split with internal validation split."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y,
    df,
    n_splits: int = 5,
    val_size: float = 0.18,
    random_state: int = 42,
):
    """
    Repository-compatible interface:
        split_data(y, df)

    Returns a list of (idx_train, idx_val, idx_test) tuples.

    Strategy:
    - outer StratifiedKFold creates test folds
    - inside each train_val fold, carve out a stratified validation split
    """
    y = np.asarray(y).astype(int)
    all_idx = np.arange(len(y))

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = []

    for train_val_idx, test_idx in skf.split(all_idx, y):
        y_train_val = y[train_val_idx]

        idx_train, idx_val = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=y_train_val,
            random_state=random_state,
        )

        splits.append((idx_train, idx_val, test_idx))

    return splits
