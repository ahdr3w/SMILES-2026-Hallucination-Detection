"""
splitting.py — Train / validation / test split (student-implemented).

Uses 5-fold stratified cross-validation: each sample lands in the test split
of exactly one fold, preserving the ~70/30 class ratio in every subset.
Within each fold a small validation slice is carved off the train portion
for threshold tuning by ``evaluate.evaluate_fold``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

N_FOLDS = 5


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return a list of ``(idx_train, idx_val, idx_test)`` tuples.

    Strategy: ``StratifiedKFold(n_splits=5)`` over the label array, with a
    small stratified slice of each fold's training data reserved for
    validation (used by the probe's threshold-tuning hook in
    ``evaluate.evaluate_fold``).
    """
    n = len(y)
    idx = np.arange(n)

    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=random_state
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    relative_val = val_size / (1.0 - 1.0 / N_FOLDS)

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(idx, y)):
        y_tv = y[train_val_idx]
        idx_train, idx_val = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=random_state + fold_idx,
            stratify=y_tv,
        )
        splits.append((idx_train, idx_val, test_idx))

    return splits
