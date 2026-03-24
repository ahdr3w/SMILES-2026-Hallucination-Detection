"""
evaluate.py — Evaluation utilities (fixed infrastructure, do not edit).

Provides helpers used by ``solution.ipynb`` to run the full evaluation loop,
print a formatted summary table, save results to a JSON file, and generate
predictions on an unlabelled test set.

For each ``(idx_train, idx_val, idx_test)`` split produced by ``splitting.py``
the pipeline runs three checkpoints:

  1. Majority-class baseline  — trivial classifier; sets the accuracy floor.
  2. HallucinationProbe (val) — student probe evaluated on the validation set.
  3. HallucinationProbe (test)— student probe evaluated on the held-out test set.

Metrics reported: Accuracy, F1-score, AUROC (primary ranking metric).
"""

from __future__ import annotations

import json
import math

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a ``[0, 1]`` metric value as a percentage string."""
    return f"{value * 100:.2f}%"


def _nanmean(values: list[float]) -> float:
    """Return the mean of *values*, ignoring NaN entries."""
    valid = [v for v in values if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------


def evaluate_fold(
    probe,
    X: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray | None,
    idx_test: np.ndarray,
) -> dict:
    """Train *probe* and return a metrics dict for the val and test splits.

    If a validation set is available and *probe* exposes a
    ``fit_hyperparameters`` method, the decision threshold is tuned on the
    validation split after training and before computing predictions.

    Args:
        probe:      A freshly instantiated ``HallucinationProbe``.
        X:          Full feature matrix of shape ``(N, feature_dim)``.
        y:          Full label array of shape ``(N,)``.
        idx_train:  Integer indices for the training subset.
        idx_val:    Integer indices for the validation subset, or ``None``.
        idx_test:   Integer indices for the test subset.

    Returns:
        A dict with keys ``"{split}_accuracy"``, ``"{split}_f1"``, and
        ``"{split}_auroc"`` for each available split (``"val"`` and/or
        ``"test"``).
    """
    probe.fit(X[idx_train], y[idx_train])

    # If the probe supports threshold tuning and a validation set is available,
    # tune the decision threshold on the validation split *before* evaluating.
    if idx_val is not None and hasattr(probe, "fit_hyperparameters"):
        probe.fit_hyperparameters(X[idx_val], y[idx_val])

    results: dict = {}

    for split_name, idx_split in [("val", idx_val), ("test", idx_test)]:
        if idx_split is None:
            continue
        y_true = y[idx_split]
        y_pred = probe.predict(X[idx_split])
        y_prob = probe.predict_proba(X[idx_split])[:, 1]

        results[f"{split_name}_accuracy"] = accuracy_score(y_true, y_pred)
        results[f"{split_name}_f1"] = f1_score(y_true, y_pred, zero_division=0)
        try:
            results[f"{split_name}_auroc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            results[f"{split_name}_auroc"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
    X: np.ndarray,
    y: np.ndarray,
    ProbeClass,
) -> list[dict]:
    """Run the full evaluation loop over all splits and return per-fold results.

    For each split the function:

    1. Trains a majority-class baseline and records its Accuracy and F1.
    2. Instantiates ``ProbeClass()``, trains it on the training split, and
       records Accuracy, F1, and AUROC for val and test splits.

    Progress is printed to stdout.

    Args:
        splits:     List of ``(idx_train, idx_val, idx_test)`` tuples produced
                    by ``splitting.split_data``.  ``idx_val`` may be ``None``.
        X:          Feature matrix of shape ``(N, feature_dim)``.
        y:          Label array of shape ``(N,)`` with values in ``{0, 1}``.
        ProbeClass: Class (not instance) to use for the probe classifier.
                    Must expose ``fit``, ``predict``, and ``predict_proba``.
                    Typically ``HallucinationProbe`` from ``probe.py``.

    Returns:
        List of result dicts, one per fold, each containing ``fold``,
        ``n_train``, ``n_val``, ``n_test``, ``baseline_*``, and probe
        metrics for available splits.
    """
    fold_results: list[dict] = []

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits):
        fold_label = f"Fold {fold_idx + 1}/{len(splits)}"
        print(f"\n{'─' * 50}")
        print(
            f"  {fold_label}  —  "
            f"train={len(idx_train)}  "
            f"val={len(idx_val) if idx_val is not None else 'N/A'}  "
            f"test={len(idx_test)}"
        )
        print(f"{'─' * 50}")

        # ── Checkpoint 1: Majority-class baseline ──────────────────────
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X[idx_train], y[idx_train])
        y_dummy = dummy.predict(X[idx_test])
        baseline_acc = accuracy_score(y[idx_test], y_dummy)
        baseline_f1 = f1_score(y[idx_test], y_dummy, zero_division=0)
        print(f"  Baseline  — Acc: {_fmt(baseline_acc)}  F1: {_fmt(baseline_f1)}")

        # ── Checkpoints 2 & 3: Student probe ───────────────────────────
        probe = ProbeClass()
        metrics = evaluate_fold(probe, X, y, idx_train, idx_val, idx_test)

        if "val_auroc" in metrics:
            print(
                f"  Probe val  — Acc: {_fmt(metrics['val_accuracy'])}  "
                f"F1: {_fmt(metrics['val_f1'])}  "
                f"AUROC: {_fmt(metrics['val_auroc'])}"
            )
        print(
            f"  Probe test — Acc: {_fmt(metrics['test_accuracy'])}  "
            f"F1: {_fmt(metrics['test_f1'])}  "
            f"AUROC: {_fmt(metrics['test_auroc'])}"
        )

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "n_train": len(idx_train),
                "n_val": len(idx_val) if idx_val is not None else 0,
                "n_test": len(idx_test),
                "baseline_accuracy": baseline_acc,
                "baseline_f1": baseline_f1,
                **metrics,
            }
        )

    return fold_results


# ---------------------------------------------------------------------------
# Summary and persistence
# ---------------------------------------------------------------------------


def print_summary(
    fold_results: list[dict],
    feature_dim: int,
    n_samples: int,
    extract_time: float,
) -> None:
    """Print a formatted evaluation summary table.

    Args:
        fold_results:  List returned by ``run_evaluation``.
        feature_dim:   Dimensionality of the feature vectors (``X.shape[1]``).
        n_samples:     Total number of samples in the dataset.
        extract_time:  Time in seconds taken by hidden-state extraction.
    """
    avg_baseline_acc = _nanmean([r["baseline_accuracy"] for r in fold_results])
    avg_baseline_f1 = _nanmean([r["baseline_f1"] for r in fold_results])
    avg_test_acc = _nanmean([r["test_accuracy"] for r in fold_results])
    avg_test_f1 = _nanmean([r["test_f1"] for r in fold_results])
    avg_test_auroc = _nanmean([r["test_auroc"] for r in fold_results])
    avg_val_auroc = _nanmean(
        [r.get("val_auroc", float("nan")) for r in fold_results]
    )

    W = 60
    print("\n" + "=" * W)
    print(" Hallucination Detection — Evaluation Summary")
    if len(fold_results) > 1:
        print(f" (averaged over {len(fold_results)} folds)")
    print("=" * W)
    print(f"  {'Checkpoint':<35} {'Accuracy':>9} {'F1':>7} {'AUROC':>7}")
    print("-" * W)
    print(
        f"  {'1. Majority-class baseline':<35} "
        f"{_fmt(avg_baseline_acc):>9} {_fmt(avg_baseline_f1):>7} {'N/A':>7}"
    )
    if not math.isnan(avg_val_auroc):
        avg_val_acc = _nanmean(
            [r.get("val_accuracy", float("nan")) for r in fold_results]
        )
        avg_val_f1 = _nanmean(
            [r.get("val_f1", float("nan")) for r in fold_results]
        )
        print(
            f"  {'2. Probe (val split)':<35} "
            f"{_fmt(avg_val_acc):>9} {_fmt(avg_val_f1):>7} {_fmt(avg_val_auroc):>7}"
        )
    print(
        f"  {'3. Probe (test split)':<35} "
        f"{_fmt(avg_test_acc):>9} {_fmt(avg_test_f1):>7} {_fmt(avg_test_auroc):>7}"
    )
    print("-" * W)
    print(f"  Feature dim  : {feature_dim}")
    print(f"  Total samples: {n_samples}")
    print(f"  Folds        : {len(fold_results)}")
    print(f"  Extract time : {extract_time:.1f} s")
    print("=" * W)
    print()
    print(f"★  Primary metric — Test AUROC: {_fmt(avg_test_auroc)}")


def save_predictions(
    probe,
    X_test: np.ndarray,
    ids: list,
    output_file: str = "predictions.csv",
) -> None:
    """Run *probe* on unlabelled competition test features and save predicted labels to CSV.

    The probe must already be fitted (via ``fit``) before calling this
    function.  Fit the probe on the **training and validation data only**
    (i.e. excluding the evaluation test split from ``splitting.split_data``)
    so that the test-set metrics from ``run_evaluation`` remain an honest,
    unbiased performance estimate.

    Note: *X_test* here refers to an **unlabelled competition test set**
    (e.g. ``data/test.csv``), which is entirely separate from the labelled
    evaluation test split used inside ``run_evaluation``.

    The output CSV contains two columns:

    * ``id``    — sample identifier, taken from *ids*.
    * ``label`` — predicted binary label (0 = truthful, 1 = hallucinated).

    Args:
        probe:        A fitted ``HallucinationProbe`` (or any object that
                      exposes a ``predict(X) -> np.ndarray`` method).
        X_test:       Feature matrix for the **unlabelled competition** test
                      set, shape ``(n_test, feature_dim)``.
        ids:          Sequence of sample identifiers aligned with ``X_test``.
        output_file:  Path to write the predictions CSV.
    """
    import pandas as pd

    y_pred = probe.predict(X_test)
    pd.DataFrame({"id": ids, "label": y_pred}).to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'  ({len(y_pred)} samples)")


def save_results(
    fold_results: list[dict],
    feature_dim: int,
    n_samples: int,
    extract_time: float,
    output_file: str = "results.json",
) -> None:
    """Save the evaluation results to a JSON file.

    Args:
        fold_results:  List returned by ``run_evaluation``.
        feature_dim:   Dimensionality of the feature vectors (``X.shape[1]``).
        n_samples:     Total number of samples in the dataset.
        extract_time:  Time in seconds taken by hidden-state extraction.
        output_file:   Path to write the JSON file.
    """
    summary = {
        "folds": fold_results,
        "avg_baseline_accuracy": _nanmean(
            [r["baseline_accuracy"] for r in fold_results]
        ),
        "avg_baseline_f1": _nanmean([r["baseline_f1"] for r in fold_results]),
        "avg_val_auroc": _nanmean(
            [r.get("val_auroc", float("nan")) for r in fold_results]
        ),
        "avg_test_accuracy": _nanmean([r["test_accuracy"] for r in fold_results]),
        "avg_test_f1": _nanmean([r["test_f1"] for r in fold_results]),
        "avg_test_auroc": _nanmean([r["test_auroc"] for r in fold_results]),
        "feature_dim": feature_dim,
        "n_samples": n_samples,
        "n_folds": len(fold_results),
        "extract_time_s": extract_time,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to '{output_file}'")
