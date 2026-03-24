"""
splitting.py — Train / validation / test split utilities (student-implementable).

Students: Modify or replace ``split_data`` to explore different splitting
strategies.  The function must accept a pandas DataFrame and return a
``(train_df, val_df, test_df)`` triple of DataFrames with the same columns as
the input.

Ideas to explore
----------------
* **Stratified k-fold** — rotate the test set across k folds and average metrics.
* **Group-aware splits** — keep all rows that share the same ``question`` in the
  same fold so the probe cannot memorise question text.
* **Time-ordered splits** — treat the ``id`` column as a temporal index and hold
  out the most recent examples (simulates deployment conditions).
* **Balanced splits** — oversample / undersample to ensure the exact class ratio
  you want in each fold.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a labelled DataFrame into train, validation, and test subsets.

    The default strategy performs two stratified random splits so that the
    class ratio (``hallucination`` column) is preserved in every subset.

    Args:
        df:           Full DataFrame; must contain a ``hallucination`` column
                      used for stratification.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A ``(train_df, val_df, test_df)`` triple of DataFrames, each sharing
        the same columns as ``df``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns three non-overlapping DataFrames that together contain
        all rows of ``df``.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the splitting strategy below.
    # ------------------------------------------------------------------

    # First split: carve out the held-out test set.
    stratify_col = df["hallucination"] if df["hallucination"].nunique() > 1 else None
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Second split: separate validation from training.
    relative_val = val_size / (1.0 - test_size)
    stratify_col_tv = (
        train_val_df["hallucination"]
        if train_val_df["hallucination"].nunique() > 1
        else None
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        random_state=random_state,
        stratify=stratify_col_tv,
    )

    return train_df, val_df, test_df
    # ------------------------------------------------------------------
