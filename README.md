# 🔍 Hallucination Detection in Small Language Models

A competition-style repository for the **SMILES 2026** student challenge.

---

## Overview

Large language models often produce fluent but factually incorrect answers —
commonly called **hallucinations**. Detecting such outputs is a key problem in
building reliable AI systems.

In this challenge you will build a lightweight hallucination detector for
**Gemma-3-4b-it** using **representation-based methods**: extract hidden states
from the model's transformer layers, then train a linear probe (or a small
classifier) on top to predict whether a generated response is truthful or
hallucinated.

This is a self-contained, resource-friendly project.  The model fits in the
15 GB VRAM of a free Google Colab T4 GPU (loaded in bfloat16).  Feature
extraction over a few hundred samples takes 1–5 minutes on GPU.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the evaluation notebook

Open **`validate.ipynb`** in Jupyter or Google Colab and run the cells from top
to bottom.

**Recommended**: use a GPU runtime (`Runtime → Change runtime type → T4 GPU`)
and set `BATCH_SIZE = 4` in the configuration cell.

---

## Files You Will Edit

| File | What to implement |
|------|-------------------|
| `aggregation.py` | Layer selection and token-pooling strategy |
| `probe.py` | Probe classifier (skeleton: MLP via `torch.nn.Module`) |
| `validate.ipynb` | **Splitting strategy** and evaluation loop |

> `splitting.py` contains a reusable helper (`split_data`) that the notebook
> imports by default — you may modify it or write your splitting logic directly
> in the notebook.

**Do not edit** `model.py` or `dataset.py` — these are fixed infrastructure.

---

## What to Implement

### `aggregation.py` — Hidden-state aggregation (main task)

Convert a per-token hidden-state tensor into a single feature vector.

Gemma-3-4b-it is a **decoder-only** model with approximately 34 transformer
layers (plus an embedding layer).  Because it processes tokens left-to-right,
the **last real token** is the natural aggregation target — it attends to the
entire input sequence.  You control:

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

The class now extends ``torch.nn.Module``, making it easy to define custom
neural-network architectures in ``_build_network`` and experiment with
different training loops in ``fit``.  The skeleton uses a one-hidden-layer MLP
with Adam optimisation.  Suggestions for improvement:

* Deeper MLP: add more hidden layers or increase hidden dimension
* Batch normalisation or dropout for regularisation
* Different optimisers (`SGD` with momentum, `AdamW` with weight decay)
* Early stopping based on a hold-out validation loss

### `splitting.py` — Data splitting strategy

Implement the ``split_data`` function to control how the dataset is divided
into train, validation, and test subsets.

The skeleton performs two stratified random splits (preserving the class
ratio).  Suggestions for improvement:

* Stratified k-fold cross-validation
* Group-aware splits (keep rows with the same question in the same fold)
* Time-ordered splits using the ``id`` column as a temporal index

---

## Evaluation Checkpoints

`validate.ipynb` runs three checkpoints for each split:

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

The dataset is a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `id` | Unique sample identifier |
| `instruction` | Optional system instruction |
| `question` | The question posed to the model |
| `prompt_template` | Template used to construct `prompt` |
| `gold_response` | The correct reference answer (**shadowed to `null` in the test split**) |
| `generated_response` | The model's generated answer (what we classify) |
| `response_history` | Dialogue history (if any) |
| `context` | Supporting context passage (if any) |
| `prompt` | The full input prompt given to the model |
| `hallucination` | Ground-truth label: 0 = truthful, 1 = hallucinated |

The text fed to the LLM for hidden-state extraction is
`prompt + "\n" + generated_response`.

> **Note:** `gold_response` is `null` in the **test split** to prevent students
> from building a trivial detector by comparing `generated_response` to the
> gold answer.

---

## Output JSON

Results are saved to `results.json` (path set by `OUTPUT_FILE` in the notebook).
When using the default single split the file looks like:

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
  "feature_dim": 2560,
  "n_samples": 200,
  "n_folds": 1,
  "extract_time_s": 120.5
}
```

When using k-fold the `folds` array contains one entry per fold and the
`avg_*` fields contain the mean across all folds.

**`avg_test_auroc` is the primary metric used for grading.**

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

* Your modified `probe.py`, `aggregation.py`, and `validate.ipynb` (with your
  chosen splitting strategy)
* A short report describing your approach, what worked, and key insights

---

> **Simple models, when used correctly, can reveal deep properties of complex systems.**
> Your task is to uncover how hallucinations manifest inside LLM representations
> and exploit that signal as effectively as possible.
