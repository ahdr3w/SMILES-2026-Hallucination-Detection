"""
splitting.py — Train / validation / test split utilities (student-implementable).

Students: Modify or replace ``split_data`` to explore different splitting
strategies.  The function receives the label array ``y`` and, optionally, the
full DataFrame ``df`` (useful for group-aware splits).  It must return a list
of ``(idx_train, idx_val, idx_test)`` tuples of integer index arrays into
``X`` and ``y``.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.

Ideas to explore
----------------
* **Stratified k-fold** — rotate the test set across K folds and average metrics.
* **Group-aware splits** — keep all rows that share the same question in the
  same fold so the probe cannot memorise question wording.
* **Time-ordered splits** — hold out the largest index values to simulate
  deployment conditions.
* **Balanced splits** — oversample / undersample before fitting to equalise the
  class ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs a single stratified random split so that the
    class ratio is preserved in every subset.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Pass it when your strategy needs per-sample metadata
                      (e.g. a ``question`` column for group-aware splits).
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples.  Each element is
        a 1-D NumPy array of integer indices into ``y`` (and the feature matrix
        ``X`` built from the same dataset).  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """
    # ------------------------------------------------------------------
    # STUDENT: Pick ONE option below.  Comment out the others and make
    # sure exactly one code path ends with a `return` statement.
    # ------------------------------------------------------------------

    # --- Option A: single stratified random split (default) -----------
    idx = np.arange(len(y))

    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=y[idx_train_val],
    )
    return [(idx_train, idx_val, idx_test)]
    # ------------------------------------------------------------------

    # --- Option B: stratified k-fold (comment out A, uncomment B) -----
    # from sklearn.model_selection import StratifiedKFold
    #
    # K = 5
    # kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    # return [
    #     (idx_tr, None, idx_te)
    #     for idx_tr, idx_te in kfold.split(np.arange(len(y)), y)
    # ]
    # ------------------------------------------------------------------

    # --- Option C: group-aware split (comment out A, uncomment C) -----
    # Requires df with a column identifying the group, e.g. "question".
    # from sklearn.model_selection import GroupShuffleSplit
    #
    # assert df is not None, "df must be provided for group-aware splitting"
    # groups = df["question"].astype(str).values
    # gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    # idx_train_val, idx_test = next(gss.split(np.arange(len(y)), y, groups=groups))
    # idx_train, idx_val = train_test_split(
    #     idx_train_val,
    #     test_size=val_size / (1.0 - test_size),
    #     random_state=random_state,
    #     stratify=y[idx_train_val],
    # )
    # return [(idx_train, idx_val, idx_test)]
    # ------------------------------------------------------------------
