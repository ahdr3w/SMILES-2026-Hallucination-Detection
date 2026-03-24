# 🔍 Hallucination Detection in Small Language Models

A competition-style repository for the **SMILES 2026** student challenge.

---

## Overview

Large language models often produce fluent but factually incorrect answers —
commonly called **hallucinations**. Detecting such outputs is a key problem in
building reliable AI systems.

In this challenge you will build a lightweight hallucination detector for a
small LLM using **representation-based methods**: extract hidden states from a
pre-trained DistilBERT model, then train a linear probe (or a small classifier)
on top to predict whether a response is truthful or hallucinated.

This is a self-contained, resource-friendly project. Feature extraction runs in
under a minute on a free Google Colab CPU instance.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run evaluation

```bash
python validate.py \
    --data_dir ./data \
    --output   results.json \
    --device   cpu
```

The `--device` argument is optional (`cpu`, `cuda`, or `mps`); it is
auto-detected when omitted.

The dataset is created automatically in `./data` on the first run (it either
downloads **TruthfulQA** from HuggingFace Datasets or falls back to a built-in
synthetic dataset if no internet access is available).

---

## Files You Will Edit

| File | What to implement |
|------|-------------------|
| `aggregation.py` | Layer selection and token-pooling strategy |
| `probe.py` | Probe classifier (skeleton: logistic regression) |

**Do not edit** `validate.py`, `model.py`, or `dataset.py` — these are fixed
infrastructure and will be replaced with the original versions during grading.

---

## What to Implement

### `aggregation.py` — Hidden-state aggregation (main task)

Convert a per-token hidden-state tensor into a single feature vector.

DistilBERT produces 7 sets of hidden states (embedding layer + 6 transformer
layers).  You control:

* **Which layer(s) to use** — early layers capture syntax; later layers
  capture semantics.
* **How to pool across tokens** — mean, max, `[CLS]` token, last real token,
  attention-weighted, or a custom combination.

The skeleton uses mean pooling over non-padding tokens in the final layer.
Useful alternatives:

* `[CLS]` token representation (index 0) — designed to capture global meaning
* Max pooling — highlights the most activated features
* Concatenate multiple layers — gives the probe information from several depths
* Attention-weighted pooling — weight tokens by their contribution

### `probe.py` — Probe classifier

Implement a binary classifier (`fit`, `predict`, `predict_proba`) that takes
the aggregated feature vectors and outputs hallucination predictions.

The skeleton wraps scikit-learn's `LogisticRegression` in a normalisation
pipeline.  Suggestions for improvement:

* `MLPClassifier(hidden_layer_sizes=(256, 128))` — learn non-linear features
* `SVC(kernel="rbf", probability=True)` — kernel trick
* Stronger regularisation (`C=0.01`) — prevents overfitting on small datasets
* `class_weight="balanced"` — handles class imbalance automatically

---

## Evaluation Checkpoints

`validate.py` runs three checkpoints in sequence:

| # | Checkpoint | What it measures |
|---|-----------|-----------------|
| 1 | **Majority-class baseline** | Trivial classifier; sets the accuracy floor |
| 2 | **Probe (val split)** | Your probe evaluated on held-out validation data |
| 3 | **Probe (test split)** | Your probe evaluated on the held-out test set — **primary grading metric** |

### Metrics

* **Accuracy** — fraction of correctly classified samples
* **F1-score** — harmonic mean of precision and recall
* **AUROC** — area under the ROC curve (primary ranking metric)

---

## Output JSON

Results are saved to the file specified by `--output`.  Example:

```json
{
  "baseline_accuracy": 0.5,
  "baseline_f1": 0.0,
  "val_accuracy": 0.72,
  "val_f1": 0.71,
  "val_auroc": 0.78,
  "test_accuracy": 0.69,
  "test_f1": 0.68,
  "test_auroc": 0.75,
  "n_train": 140,
  "n_val": 30,
  "n_test": 30,
  "feature_dim": 768,
  "extract_time_s": 45.2
}
```

**`test_auroc` is the primary metric used for grading.**

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

* A trained hallucination detection model (your modified `probe.py` and
  `aggregation.py`)
* A short report describing your approach, what worked, and key insights

---

> **Simple models, when used correctly, can reveal deep properties of complex systems.**
> Your task is to uncover how hallucinations manifest inside LLM representations
> and exploit that signal as effectively as possible.
