"""
validate.py — Evaluation script (fixed infrastructure, do not edit).

Runs three evaluation checkpoints and saves results to a JSON file:
  1. Majority-class baseline  — trivial classifier; sets a floor for accuracy.
  2. Probe on last-layer features  — student aggregation + student probe,
                                     trained on the training split.
  3. Summary metrics (Accuracy, F1, AUROC) on the held-out test split.

Usage
-----
    python validate.py \\
        --data_dir  ./data \\
        --output    results.json \\
        --device    cpu

Compute budget note:
This task has no explicit compute budget.  Extracting hidden states from
DistilBERT takes roughly 30–60 seconds on CPU and < 10 seconds on a free
Colab GPU.  Feature extraction is performed once and the result is reused
for all probes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from aggregation import aggregate
from dataset import load_data
from model import get_model_and_tokenizer, extract_hidden_states
from probe import HallucinationProbe


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def build_feature_matrix(
    hidden_states_list: list[torch.Tensor],
    attention_masks_list: list[torch.Tensor],
) -> np.ndarray:
    """Apply ``aggregate`` to every sample and stack into a 2-D matrix.

    Args:
        hidden_states_list:  List of per-sample tensors, each of shape
                             ``(n_layers, seq_len, hidden_dim)``.
        attention_masks_list: List of per-sample attention masks, each of
                              shape ``(seq_len,)``.

    Returns:
        NumPy array of shape ``(n_samples, feature_dim)`` ready for
        scikit-learn estimators.
    """
    features = [
        aggregate(hs, mask).numpy()
        for hs, mask in zip(hidden_states_list, attention_masks_list)
    ]
    return np.vstack(features)


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary(results: dict) -> None:
    """Print a formatted summary table.

    Args:
        results: Dict produced by the main evaluation loop.
    """
    print("\n" + "=" * 60)
    print(" Hallucination Detection — Evaluation Summary")
    print("=" * 60)
    print(f"  {'Checkpoint':<35} {'Accuracy':>9} {'F1':>7} {'AUROC':>7}")
    print("-" * 60)

    rows = [
        (
            "1. Majority-class baseline",
            results["baseline_accuracy"],
            results["baseline_f1"],
            "  N/A",
        ),
        (
            "2. Probe (val split)",
            results["val_accuracy"],
            results["val_f1"],
            _fmt(results["val_auroc"]),
        ),
        (
            "3. Probe (test split)",
            results["test_accuracy"],
            results["test_f1"],
            _fmt(results["test_auroc"]),
        ),
    ]
    for label, acc, f1, auroc in rows:
        f1_str = _fmt(f1) if isinstance(f1, float) else f1
        acc_str = _fmt(acc) if isinstance(acc, float) else acc
        print(f"  {label:<35} {acc_str:>9} {f1_str:>7} {auroc:>7}")

    print("-" * 60)
    print(f"  Train samples : {results['n_train']}")
    print(f"  Val   samples : {results['n_val']}")
    print(f"  Test  samples : {results['n_test']}")
    print(f"  Feature dim   : {results['feature_dim']}")
    print(f"  Extract time  : {results['extract_time_s']:.1f} s")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate hallucination detection probes."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory for the dataset JSONL file (auto-created if absent).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path to write the results JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device for LLM inference: 'cpu', 'cuda', or 'mps'. "
            "Auto-detected when omitted."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for hidden-state extraction.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    print(f"\n[Data] Loading dataset from '{args.data_dir}' ...")
    train_data, val_data, test_data = load_data(data_dir=args.data_dir)

    n_train = len(train_data["texts"])
    n_val = len(val_data["texts"])
    n_test = len(test_data["texts"])
    print(
        f"[Data] Train: {n_train} | Val: {n_val} | Test: {n_test} samples"
    )

    all_texts = train_data["texts"] + val_data["texts"] + test_data["texts"]
    all_labels = train_data["labels"] + val_data["labels"] + test_data["labels"]

    # ------------------------------------------------------------------
    # Hidden-state extraction  (done once, results reused by all probes)
    # ------------------------------------------------------------------
    print("\n[Model] Loading LLM ...")
    model, tokenizer = get_model_and_tokenizer()

    print("[Model] Extracting hidden states for all splits ...")
    t0 = time.time()
    all_hidden, all_masks = extract_hidden_states(
        model, tokenizer, all_texts, device=device, batch_size=args.batch_size
    )
    extract_time = time.time() - t0
    print(f"[Model] Extraction done in {extract_time:.1f} s")

    # Split back into train / val / test
    train_hidden = all_hidden[:n_train]
    train_masks = all_masks[:n_train]
    val_hidden = all_hidden[n_train : n_train + n_val]
    val_masks = all_masks[n_train : n_train + n_val]
    test_hidden = all_hidden[n_train + n_val :]
    test_masks = all_masks[n_train + n_val :]

    # ------------------------------------------------------------------
    # Feature matrix construction via student aggregation
    # ------------------------------------------------------------------
    print("\n[Features] Applying aggregation ...")
    X_train = build_feature_matrix(train_hidden, train_masks)
    X_val = build_feature_matrix(val_hidden, val_masks)
    X_test = build_feature_matrix(test_hidden, test_masks)

    y_train = np.array(train_data["labels"])
    y_val = np.array(val_data["labels"])
    y_test = np.array(test_data["labels"])

    feature_dim = X_train.shape[1]
    print(f"[Features] Feature dimension: {feature_dim}")

    # ------------------------------------------------------------------
    # Checkpoint 1: Majority-class baseline
    # ------------------------------------------------------------------
    print("\n[Checkpoint 1/3] Majority-class baseline")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_dummy)
    baseline_f1 = f1_score(y_test, y_dummy, zero_division=0)
    print(f"  Test accuracy : {_fmt(baseline_acc)}")
    print(f"  Test F1       : {_fmt(baseline_f1)}")

    # ------------------------------------------------------------------
    # Checkpoint 2 & 3: Student probe on val and test splits
    # ------------------------------------------------------------------
    print("\n[Checkpoint 2/3] Training probe ...")
    probe = HallucinationProbe()
    probe.fit(X_train, y_train)

    # Validation
    y_val_pred = probe.predict(X_val)
    y_val_prob = probe.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    try:
        val_auroc = roc_auc_score(y_val, y_val_prob)
    except ValueError:
        val_auroc = float("nan")
    print(f"  Val accuracy : {_fmt(val_acc)}")
    print(f"  Val F1       : {_fmt(val_f1)}")
    print(f"  Val AUROC    : {_fmt(val_auroc)}")

    # Test
    print("\n[Checkpoint 3/3] Evaluating on test split ...")
    y_test_pred = probe.predict(X_test)
    y_test_prob = probe.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    try:
        test_auroc = roc_auc_score(y_test, y_test_prob)
    except ValueError:
        test_auroc = float("nan")
    print(f"  Test accuracy : {_fmt(test_acc)}")
    print(f"  Test F1       : {_fmt(test_f1)}")
    print(f"  Test AUROC    : {_fmt(test_auroc)}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "baseline_accuracy": baseline_acc,
        "baseline_f1": baseline_f1,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_auroc": test_auroc,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "feature_dim": feature_dim,
        "extract_time_s": extract_time,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results)
    print(f"[Output] Results saved to '{args.output}'")
