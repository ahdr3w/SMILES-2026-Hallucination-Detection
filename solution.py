"""
Hallucination Detection in Small Language Models

# Files you can edit:
    - aggregation.py — layer selection and token pooling 
    - aggregation.py | extract_geometric_features — optional hand-crafted features 
    - probe.py | HallucinationProbe — probe classifier (nn.Module subclass) 
    - splitting.py | split_data — train / validation / test split strategy 

# Fixed infrastructure (do not edit)
    - model.py | LLM loader (get_model_and_tokenizer) 
    - evaluate.py | Evaluation loop, summary table, JSON output 

# Data Format — ChatML and Special Tokens
    The `prompt` column uses ChatML (Chat Markup Language), the conversation
    template built into Qwen models.  Each message is wrapped in role markers:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    ... question and context ... <|im_end|>
    <|im_start|>assistant

    Special tokens and their roles:

    - `<|im_start|>` — opens a chat turn; the role (`system`, `user`, or `assistant`) immediately follows
    - `<|im_end|>` — closes the current chat turn
    - `<|endoftext|>` — end-of-sequence (EOS) token appended by the model at the end of its response

    The `prompt` ends right after `<|im_start|>assistant\n` — it provides the
    full context up to (but not including) the model's reply.  The `response`
    column holds the actual generated text, ending with `<|endoftext|>`.

    We feed the concatenation of `prompt + response` to the feature extractor
    so the hidden states capture both the question context and the model's
    specific answer — the hallucination signal lives in that joint representation.


"""

import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aggregation import aggregation_and_feature_extraction
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data
from sklearn.model_selection import train_test_split

import os

from experiment_logger import (
    start_mlflow_run,
    log_dataset_info,
    log_fold_results,
    log_results_json,
    log_artifacts,
    end_mlflow_run,
)

# ---------------------------------------------------------------------

DATA_FILE     = "./data/dataset.csv"   # path to the dataset CSV
OUTPUT_FILE   = "results.json"         # where to write the results summary
BATCH_SIZE    = 4
USE_GEOMETRIC = True                  # set True to enable geometric feature extraction
TEST_FILE        = "./data/test.csv"   # competition test set (labels are null)
PREDICTIONS_FILE = "predictions.csv"   # output file with predicted labels

MLFLOW_TRACKING_URI = "file:/content/drive/MyDrive/SMILES-2026/SMILES-2026-Hallucination-Detection/mlruns"

RUN_NAME = "exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc"

EXPERIMENT_CONFIG = {
    "run_name": RUN_NAME,
    "data_file": DATA_FILE,
    "test_file": TEST_FILE,
    "batch_size": BATCH_SIZE,
    "use_geometric": True,
    "max_length": MAX_LENGTH,

    "aggregation": "single_raw_layer_plus_default_trajectory_features",
    "features": "single_layer_last_token_plus_trajectory_scalars",
    "raw_feature_dim": 1017,
    "feature_dim_after_pca": 128,
    "pooling": ["last"],
    "raw_layers": [-4, -8, -12, -16],
    "trajectory_layers": [-8, -10, -12, -14, -16],
    "token_window": 32,
    "token_offset": -2,

    "raw_vector": {
        "enabled": True,
        "source": "last_token",
        "layers": [-4, -8, -12, -16],
        "dim": 896,
    },

    "scalar_features": {
        "enabled": True,
        "n_features": 121,
    },

    "dim_reduction": "PCA",
    "pca_n_components": 128,
    "pca_fit": "train_only",

    "probe": "ensemble_regularized_mlp",
    "ensemble_size": 5,
    "ensemble_seeds": [42, 43, 44, 45, 46],
    "base_architecture": "128_8_1",
    "hidden_dim": 8,
    "activation": "ReLU",
    "dropout": 0.5,
    "scaler": "StandardScaler",

    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "epochs": 200,
    "loss": "BCEWithLogitsLoss",
    "pos_weight": "neg_pos_ratio",

    "split_strategy": "stratified_5fold_with_inner_val",
    "n_splits": 5,
    "val_size": 0.15,
    "seed": 42,

    "threshold_tuning": True,
    "threshold_metric": "validation_accuracy",
    "target_metric": "accuracy",

    "hypothesis": (
        "A single intermediate layer may contain stronger hallucination signal "
        "than a fixed multi-layer concat. Reducing raw hidden dimensions from "
        "3584 to 896 may also reduce overfitting while preserving the most useful "
        "layer-level representation."
    ),
}

assert OUTPUT_FILE == "results.json"
assert PREDICTIONS_FILE == "predictions.csv"
# ---------------------------------------------------------------------
if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    EXPERIMENT_CONFIG["device"] = str(device)

    mlflow_run = start_mlflow_run(
        run_name=RUN_NAME,
        config=EXPERIMENT_CONFIG,
        tracking_uri=MLFLOW_TRACKING_URI,
    )

    print(f"Device       : {device}")
    print(f"Data         : {DATA_FILE}")
    print(f"Max length   : {MAX_LENGTH} tokens")
    print(f"Geometric feats: {USE_GEOMETRIC}")


    df = pd.read_csv(DATA_FILE)

    # Build the text fed to the LLM: concatenation of prompt and response.
    all_texts  = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    all_labels = np.array([int(float(h)) for h in df["label"]])

    n_total = len(all_labels)
    print(f"Loaded {n_total} samples  "
        f"({all_labels.sum()} hallucinated / {(all_labels == 0).sum()} truthful)")
    
    log_dataset_info(
        n_samples=n_total,
        n_hallucinated=int(all_labels.sum()),
        n_truthful=int((all_labels == 0).sum()),
    )

    # Preview the raw data
    print(f"Columns : {df.columns.tolist()}")
    print(f"Rows    : {len(df)}")
    print(f"Labels  : {dict(df['label'].value_counts().sort_index())}")
    print()

    # Show the first sample (truncated for readability)
    row0 = df.iloc[0]
    print("── prompt (first 500 chars) " + "─" * 34)
    print(row0["prompt"][:500])
    print()
    print("── response (first 300 chars) " + "─" * 31)
    print(row0["response"][:300])
    print()
    label_str = "hallucinated" if int(row0["label"]) else "truthful"
    print(f"── label : {int(row0['label'])}  ({label_str})")


    # Load the LLM
    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    all_features: list = []
    t0 = time.time()

    for start in tqdm(range(0, len(all_texts), BATCH_SIZE),
                    desc="Extracting & aggregating", unit="batch"):

        # ── 1. Tokenise the current mini-batch ───────────────────────────────
        batch_texts = all_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # ── 2. LLM forward pass ──────────────────────────────────────────────
        # outputs.hidden_states: tuple of (n_layers+1) tensors,
        # each with shape (batch, seq_len, hidden_dim).
        # Index 0 → token embeddings; index k → transformer layer k.
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # ── 3. Stack all layers into one tensor, move to CPU ─────────────────
        # Shape: (batch, n_layers, seq_len, hidden_dim)
        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask   = attention_mask.cpu()

        # ── 4. Aggregate each sample and store the compact feature vector ─────
        # The raw `hidden` tensor is released at the end of this loop iteration.
        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i],   # (n_layers, seq_len, hidden_dim)
                mask[i],     # (seq_len,)
                use_geometric=USE_GEOMETRIC,
            )
            all_features.append(feat.cpu())

    extract_time = time.time() - t0
    print(f"Done in {extract_time:.1f} s  —  {len(all_features)} feature vectors extracted")

    # Stack into the (N, feature_dim) matrix used by the probe.
    X = np.vstack([f.numpy() for f in all_features])   # shape: (N, feature_dim)
    y = all_labels                                       # shape: (N,)

    print("X checksum:", float(np.sum(X)))
    print("X mean/std:", float(X.mean()), float(X.std()))

    print(f"Feature matrix : {X.shape}  (feature_dim = {X.shape[1]})")
    print(f"Geometric feats: {USE_GEOMETRIC}")

    log_dataset_info(
        n_samples=len(X),
        n_hallucinated=int(y.sum()),
        n_truthful=int((y == 0).sum()),
        feature_dim=int(X.shape[1]),
    )

    splits = split_data(y, df)

    print(f"Splits : {len(splits)} fold(s)")
    for i, (tr, va, te) in enumerate(splits):
        print(f"  Fold {i + 1}: train={len(tr)}  "
            f"val={len(va) if va is not None else 'N/A'}  test={len(te)}")

    fold_results = run_evaluation(splits, X, y, HallucinationProbe)
    
    print_summary(fold_results, X.shape[1], len(X), extract_time)
    save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)

    log_fold_results(fold_results)
    log_results_json(OUTPUT_FILE)

    # ── Load test data ────────────────────────────────────────────────────────
    df_test    = pd.read_csv(TEST_FILE)
    test_texts = [f"{row['prompt']}{row['response']}" for _, row in df_test.iterrows()]
    test_ids   = df_test.index
    print(f"Test set loaded: {len(test_texts)} samples")

    # ── Extract features for test set (same loop as Section 4) ───────────────
    test_features: list = []

    for start in tqdm(range(0, len(test_texts), BATCH_SIZE),
                    desc="Test extraction & aggregation", unit="batch"):

        batch_texts = test_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask   = attention_mask.cpu()

        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i], mask[i], use_geometric=USE_GEOMETRIC,
            )
            test_features.append(feat.cpu())

    X_test = np.vstack([f.numpy() for f in test_features])  # (n_test, feature_dim)

    # ── Fit final probe on training + validation data only ──────────────────
    # Collect the union of all train and validation indices across every split.
    # For a single split this excludes idx_test; for k-fold every sample appears
    # in a training fold, so all samples are used (same as fitting on X, y).
    idx_non_test = np.unique(np.concatenate([
        np.concatenate([idx_tr, idx_va]) if idx_va is not None else idx_tr
        for idx_tr, idx_va, _ in splits
    ]))

    idx_train_final, idx_val_final = train_test_split(
        idx_non_test,
        test_size=0.15,
        random_state=42,
        stratify=y[idx_non_test],
    )

    print(
      f"Final split: train={len(idx_train_final)} "
      f"val={len(idx_val_final)}"
    )

    final_probe = HallucinationProbe()
    final_probe.fit(X[idx_train_final], y[idx_train_final])
    final_probe.fit_hyperparameters(X[idx_val_final], y[idx_val_final])

    print(f"Final tuned threshold: {final_probe._threshold:.4f}")

    # ── Predict and save ────────────────────────────────────────────────────
    save_predictions(final_probe, X_test, test_ids, PREDICTIONS_FILE)

    log_artifacts([
        OUTPUT_FILE,
        PREDICTIONS_FILE,
        "SOLUTION.md",
    ])

    end_mlflow_run()

