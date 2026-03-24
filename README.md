# 🔍 Hallucination Detection in Small Language Models

A competition-style repository for the **SMILES 2026** student challenge.

---

## Overview

Large language models often produce fluent but factually incorrect answers —
commonly called **hallucinations**. Detecting such outputs is a key problem in
building reliable AI systems.

In this challenge you will build a lightweight hallucination detector for
**Qwen2.5-0.5B** using **representation-based methods**: extract hidden states
from the model's transformer layers, then train a probe classifier on top to
predict whether a generated response is truthful or hallucinated.

This is a self-contained, resource-friendly project.  Qwen2.5-0.5B (0.5 B
parameters, bfloat16) fits comfortably on a free Google Colab T4 GPU (≈ 15 GB
VRAM) and on most modern CPUs.  Feature extraction over a few hundred samples
takes roughly 1–5 minutes on GPU.

---

## Quick Start

### 1. Open the notebook

Open **`solution.ipynb`** in Jupyter or Google Colab.

The first cell contains ready-to-use setup commands for both Google Colab and
local environments (git clone, venv creation, dependency install).

### 2. Run all cells

After the setup cell, run the remaining cells from top to bottom:

1. **Load dataset** — reads the CSV and builds input texts
2. **Extract features** — forward pass through Qwen2.5-0.5B (run once)
3. **Split data** — calls `splitting.split_data(y, df)` from `splitting.py`
4. **Evaluate** — trains your probe and reports metrics
5. **Save results** — writes `results.json`
6. **Predict on test set** — fits the probe on all labelled data, runs it on
   `data/test.csv`, and writes `predictions.csv`

**Recommended**: use a GPU runtime (`Runtime → Change runtime type → T4 GPU`)
and set `BATCH_SIZE = 4` in the configuration cell.

---

## Files You Will Edit

| File | What to implement |
|------|-------------------|
| `aggregation.py` | Layer selection and token-pooling strategy |
| `probe.py` | Probe classifier (skeleton: MLP via `torch.nn.Module`) |
| `splitting.py` | Train / validation / test split strategy |

**Do not edit** `model.py` or `evaluate.py` — these are fixed infrastructure
and will be replaced with the original versions during grading.

---

## What to Implement

### `aggregation.py` — Hidden-state aggregation (main task)

Convert a per-token hidden-state tensor into a single feature vector.

Qwen2.5-0.5B is a **decoder-only** model with 24 transformer layers (plus an
embedding layer) and a hidden dimension of 896.  Because it processes tokens
left-to-right, the **last real token** is the natural aggregation target — it
attends to the entire input sequence.  You control:

* **Which layer(s) to use** — early layers capture syntax; later layers
  capture semantics.
* **How to pool across tokens** — last real token (default for causal LMs),
  mean pooling, max pooling, or a custom combination.

The skeleton uses the **last real token** of the final transformer layer as the
feature vector — the natural choice for decoder-only models.  Useful alternatives:

* Mean pooling over all real tokens — smoother, sometimes more robust
* Middle-layer last token — intermediate abstraction level
* Concatenate multiple layers' last-token vectors
* Last token of a specific layer index (e.g., `hidden_states[-4]`)

### `probe.py` — Probe classifier

Implement a binary classifier (`fit`, `predict`, `predict_proba`) that takes
the aggregated feature vectors and outputs hallucination predictions.

The class extends ``torch.nn.Module``, making it easy to define custom
neural-network architectures in ``_build_network`` and experiment with
different training loops in ``fit``.  The skeleton uses a one-hidden-layer MLP
with Adam optimisation.  Suggestions for improvement:

* Deeper MLP: add more hidden layers or increase hidden dimension
* Batch normalisation or dropout for regularisation
* Different optimisers (`SGD` with momentum, `AdamW` with weight decay)
* Early stopping based on a hold-out validation loss

### `splitting.py` — Data splitting strategy

Implement the ``split_data(y, df)`` function to control how the dataset is
divided into train, validation, and test subsets.

``split_data`` returns a **list of ``(idx_train, idx_val, idx_test)`` tuples**
of integer index arrays.  A single-element list gives one train/val/test split;
a K-element list triggers K-fold cross-validation (metrics are averaged).

The skeleton performs a single stratified random split.  Suggestions:

* Stratified k-fold cross-validation (`StratifiedKFold`)
* Group-aware splits — keep rows with the same question in the same fold
  (pass `df` and use `df["question"]` as the group key)
* Time-ordered splits using the sample index as a temporal proxy

---

## Evaluation Checkpoints

`evaluate.py` runs three checkpoints for each split:

| # | Checkpoint | What it measures |
|---|-----------|-----------------|
| 1 | **Majority-class baseline** | Trivial classifier; sets the accuracy floor |
| 2 | **Probe (val split)** | Your probe evaluated on held-out validation data |
| 3 | **Probe (test split)** | Your probe on the held-out test set — **primary grading metric** |

### Metrics

* **Accuracy** — fraction of correctly classified samples
* **F1-score** — harmonic mean of precision and recall
* **AUROC** — area under the ROC curve (primary ranking metric)

---

## Dataset Format

The dataset is a CSV file with (at minimum) the following columns:

| Column | Description |
|--------|-------------|
| `prompt` | The full input prompt given to the model |
| `response` | The model's generated answer (what we classify) |
| `label` | Ground-truth binary label: 0 = truthful, 1 = hallucinated |

The text fed to Qwen2.5-0.5B for hidden-state extraction is the direct
concatenation `prompt + response` (no separator).

---

## Output JSON

Results are saved to `results.json` (path set by `OUTPUT_FILE` in the
notebook).  When using the default single split the file looks like:

```json
{
  "folds": [
    {
      "fold": 1,
      "n_train": 140,
      "n_val": 30,
      "n_test": 30,
      "baseline_accuracy": 0.5,
      "baseline_f1": 0.0,
      "val_accuracy": 0.72,
      "val_f1": 0.71,
      "val_auroc": 0.78,
      "test_accuracy": 0.69,
      "test_f1": 0.68,
      "test_auroc": 0.75
    }
  ],
  "avg_test_accuracy": 0.69,
  "avg_test_f1": 0.68,
  "avg_test_auroc": 0.75,
  "feature_dim": 896,
  "n_samples": 200,
  "n_folds": 1,
  "extract_time_s": 120.5
}
```

When using k-fold the `folds` array contains one entry per fold and the
`avg_*` fields contain the mean across all folds.

**`avg_test_auroc` is the primary metric used for grading.**

---

## Competition Test Predictions

`data/test.csv` is the held-out competition file — its `label` column is `null`.
After running the evaluation cells, Section 8 of the notebook:

1. Loads `data/test.csv` and builds input texts the same way as the training set.
2. Extracts features with the already-loaded model.
3. Re-fits the probe on the **full** labelled dataset (to not waste any data).
4. Saves predicted labels to `predictions.csv`:

```
id,label
0,1
1,0
2,1
...
```

Submit `predictions.csv` alongside your report.

---

## What You'll Learn

* How internal representations of LLMs encode factual correctness
* How to build linear probes for model interpretability
* Practical ML skills under resource constraints (free Colab)
* The trade-offs between model complexity and generalisation

---

## Extensions (Optional)

* Compare representations from different layers to locate the "truth layer"
* Study failure cases: which question categories fool your probe?
* Try ensembling probes trained on different layers
* Experiment with dimensionality reduction (PCA) before classification

---

## Deliverables

* Your modified `aggregation.py`, `probe.py`, and `splitting.py`
* `predictions.csv` — your predicted labels for the competition test set
* A short report describing your approach, what worked, and key insights

---

> **Simple models, when used correctly, can reveal deep properties of complex systems.**
> Your task is to uncover how hallucinations manifest inside LLM representations
> and exploit that signal as effectively as possible.
