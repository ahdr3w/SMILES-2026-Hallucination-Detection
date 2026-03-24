# 🔍 SMILES Hallucination Detection

Starter kit for the **SMILES 2026** competition: detect whether a small language
model's answer is *hallucinated* (fabricated) or *truthful* using the model's
own internal representations (hidden states).

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Pipeline](#pipeline)
4. [Quick Start](#quick-start)
   - [Google Colab](#google-colab)
   - [Local Setup](#local-setup)
5. [Dataset](#dataset)
6. [What You Implement](#what-you-implement)
7. [Evaluation](#evaluation)
8. [Ways to Improve](#ways-to-improve)
   - [Feature Extraction Techniques](#feature-extraction-techniques)
   - [Probe Architecture](#probe-architecture)
   - [Training & Optimisation](#training--optimisation)
   - [Data Splitting](#data-splitting)
9. [Submission](#submission)
10. [Dependencies](#dependencies)

---

## Overview

Large (and small) language models sometimes *hallucinate* — they generate
plausible-sounding text that is factually incorrect.  This competition asks you
to build a **lightweight binary classifier** (called a *probe*) that reads the
model's internal hidden states and predicts whether a given response is
truthful (`label = 0`) or hallucinated (`label = 1`).

The language model used throughout is **[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)** — a
decoder-only causal transformer with 24 layers and a hidden dimension of 896.
It fits comfortably on a free Google Colab T4 GPU.

**Primary ranking metric:** Area Under the ROC Curve (AUROC) on the held-out
test split.

---

## Repository Structure

```
SMILES-HALLUCINATION-DETECTION/
├── data/
│   ├── dataset.csv        # Labelled training data (prompt, response, label)
│   └── test.csv           # Unlabelled competition test set
│
├── solution.ipynb         # Main workspace — run cells top-to-bottom
│
│   ── Files you implement ──────────────────────────────────────────────
├── aggregation.py         # Layer selection, token pooling, geometric features
├── probe.py               # HallucinationProbe — the binary classifier
├── splitting.py           # Train / validation / test split strategy
│
│   ── Fixed infrastructure (do not edit) ───────────────────────────────
├── model.py               # Loads Qwen2.5-0.5B and exposes get_model_and_tokenizer()
├── evaluate.py            # Evaluation loop, metrics, summary table, JSON output
│
├── requirements.txt       # Python dependencies
└── LICENSE
```

---

## Pipeline

```
dataset.csv
    │
    ▼
[Section 3] Load & format texts
    │  prompt + response → single string per sample
    ▼
[Section 4] Hidden-state extraction  (model.py)
    │  tokenise → LLM forward pass → outputs.hidden_states
    │  shape per sample: (n_layers+1, seq_len, hidden_dim) = (25, ≤512, 896)
    │
    ├─► aggregation.py :: aggregate()
    │       layer selection + token pooling → (feature_dim,)
    │
    └─► aggregation.py :: extract_geometric_features()   [optional]
            hand-crafted statistics → (n_geo_features,)
    │
    ▼  concatenated → feature matrix X  shape (N, feature_dim)
[Section 5] splitting.py :: split_data()
    │  → idx_train, idx_val, idx_test
    ▼
[Section 6] evaluate.py :: run_evaluation()
    │  trains probe.py :: HallucinationProbe on X[idx_train]
    │  tunes threshold on X[idx_val]
    │  reports Accuracy / F1 / AUROC on X[idx_test]
    ▼
[Section 7] print_summary() + save_results() → results.json
    ▼
[Section 8] save_predictions() → predictions.csv  (competition submission)
```

---

## Quick Start

### Google Colab

Open `solution.ipynb` in Colab and run the **Colab** setup block in
Section 0:

```python
!git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
%cd SMILES-HALLUCINATION-DETECTION
!pip install -r requirements.txt
```

Then run all cells from top to bottom.  A free T4 GPU is recommended
(Runtime → Change runtime type → T4 GPU).

### Local Setup

```bash
git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
jupyter notebook solution.ipynb
```

---

## Dataset

`data/dataset.csv` contains **~13 400 labelled samples** with three columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Full ChatML-formatted conversation context fed to Qwen |
| `response` | str | The model's generated response |
| `label` | float | `1.0` = hallucinated · `0.0` = truthful |

The `prompt` uses the **ChatML** template built into Qwen models:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Given the context, answer the question …<|im_end|>
<|im_start|>assistant
```

For feature extraction, `prompt` and `response` are concatenated into one
string and fed to the tokeniser.

`data/test.csv` is structured identically but the `label` column is null —
these are the samples you submit predictions for.

---

## What You Implement

You are expected to edit **three files**.  The rest of the codebase is fixed
infrastructure.

### `aggregation.py`

| Function | Task |
|----------|------|
| `aggregate(hidden_states, attention_mask)` | Convert the raw `(n_layers, seq_len, hidden_dim)` tensor into a single feature vector. |
| `extract_geometric_features(hidden_states, attention_mask)` | *(Optional)* Compute hand-crafted statistical/geometric features.  Enabled by setting `USE_GEOMETRIC = True` in the notebook. |

### `probe.py`

Implement `HallucinationProbe` — a `torch.nn.Module` subclass with four
required methods:

| Method | Signature |
|--------|-----------|
| `fit` | `(X_train, y_train) → self` |
| `fit_hyperparameters` | `(X_val, y_val) → self` — tunes decision threshold |
| `predict` | `(X) → np.ndarray` of `{0, 1}` labels |
| `predict_proba` | `(X) → np.ndarray` of shape `(n, 2)` — class probabilities |

### `splitting.py`

Implement `split_data(y, df, …)` which returns a list of
`(idx_train, idx_val, idx_test)` tuples.  Three commented-out options are
provided as starting points (random split, stratified k-fold, group-aware).

---

## Evaluation

For each fold `evaluate.py` reports three checkpoints:

| # | Checkpoint | Metrics |
|---|-----------|---------|
| 1 | Majority-class baseline | Accuracy, F1 |
| 2 | `HallucinationProbe` on **validation** split | Accuracy, F1, AUROC |
| 3 | `HallucinationProbe` on **test** split | Accuracy, F1, **AUROC** ★ |

★ **AUROC on the test split is the primary competition metric.**

Results are averaged across folds (if using k-fold) and saved to
`results.json`.

---

## Ways to Improve

### Feature Extraction Techniques

This is the highest-leverage area.  The default extracts only the **last
token's hidden state from the final transformer layer** — a good baseline,
but far from optimal.

#### Layer Selection

Different layers encode different types of information.  Early layers capture
syntax; later layers carry semantics.  Experiment with:

```python
# Single layer alternatives
layer = hidden_states[-1]          # final layer (default)
layer = hidden_states[-2]          # penultimate layer
# hidden_states: index 0 = embedding layer, 1-24 = transformer layers
# Middle *transformer* layer (skip the embedding layer at index 0):
layer = hidden_states[1 + (len(hidden_states) - 1) // 2]  # middle transformer layer

# Multi-layer fusion — concatenate several layers
feature = torch.cat([hidden_states[-1][last_pos],
                     hidden_states[-2][last_pos],
                     hidden_states[-4][last_pos]], dim=0)

# Weighted average of all layers
weights = torch.softmax(torch.ones(len(hidden_states)), dim=0)
feature = (hidden_states * weights[:, None, None]).sum(0)[last_pos]
```

#### Token Pooling

For a causal model the *last real token* is the default, but other pooling
strategies can capture complementary signal:

```python
real_mask = attention_mask.bool()                          # (seq_len,)

# Mean pooling over all non-padding tokens
feature = hidden_states[-1][real_mask].mean(0)             # (hidden_dim,)

# Max pooling
feature = hidden_states[-1][real_mask].max(0).values       # (hidden_dim,)

# Attention-score weighted pooling (if you expose attention weights)
# outputs.attentions: tuple of (batch, n_heads, seq, seq)
attn = outputs.attentions[-1][sample_idx].mean(0)          # (seq, seq)
weights = attn[-1, :][real_mask]                           # last-token row
weights = weights / weights.sum()
feature = (hidden_states[-1][real_mask] * weights.unsqueeze(1)).sum(0)
```

#### Geometric / Statistical Features (`extract_geometric_features`)

These hand-crafted features describe *how* representations evolve across
layers and tokens, capturing properties that fixed pooling may miss.
Set `USE_GEOMETRIC = True` in the notebook to append them to the feature
vector.

```python
real_mask = attention_mask.bool()

# 1. Layer-wise L2 norms — how does activation magnitude change with depth?
layer_norms = hidden_states[:, real_mask, :].norm(dim=-1).mean(dim=-1)
# shape: (n_layers,)

# 2. Representation drift — cosine similarity between consecutive layers
from torch.nn.functional import cosine_similarity
drifts = []
for l in range(hidden_states.size(0) - 1):
    sim = cosine_similarity(hidden_states[l][real_mask],
                            hidden_states[l + 1][real_mask], dim=-1)
    drifts.append(sim.mean())
drift_vec = torch.stack(drifts)  # shape: (n_layers - 1,)

# 3. Variance across real tokens in the last layer — how spread are token reps?
token_var = hidden_states[-1][real_mask].var(dim=0).mean().unsqueeze(0)

# 4. Sequence length — simple verbosity proxy
seq_len_feat = attention_mask.float().sum().unsqueeze(0)

# 5. Entropy of last-token logits (if you expose lm_head output)
# Hallucinated answers may show higher uncertainty at prediction time.

# Combine all geometric features
geo = torch.cat([layer_norms, drift_vec, token_var, seq_len_feat])
```

#### PCA / Dimensionality Reduction

With multi-layer concatenation the feature dimension can exceed 10 000.
Reduce it before feeding the probe:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=256, random_state=42)
X_reduced = pca.fit_transform(X_train)  # fit only on training data!
```

---

### Probe Architecture

The default is a shallow two-layer MLP.  Ideas to extend it:

```python
# Deeper MLP with regularisation
self._net = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
)
```

Alternatively, skip the MLP entirely and use a classical classifier that
works well in high-dimensional, low-sample regimes:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X_train_scaled, y_train)
```

---

### Training & Optimisation

- **Optimiser:** try `torch.optim.AdamW` with weight decay, or `SGD` with
  momentum and a cosine learning-rate schedule.
- **Epochs & early stopping:** monitor validation AUROC and stop when it
  plateaus to avoid overfitting.
- **Class imbalance:** the dataset may be imbalanced.  Options:
  - Adjust `pos_weight` in `BCEWithLogitsLoss` (already done in the skeleton).
  - Oversample the minority class with `imbalanced-learn`.
  - Use a Focal Loss to down-weight easy negatives.
- **Threshold tuning:** `fit_hyperparameters` already searches for the
  F1-optimal threshold on the validation set.  You can change the target
  metric (e.g. maximise AUROC-aligned threshold, or use an F-beta score).

---

### Data Splitting

- **Stratified k-fold** (`splitting.py` Option B): averages metrics over K
  folds, giving a more robust estimate with limited data.
- **Group-aware splits** (`splitting.py` Option C): keeps all rows that share
  the same question in the same fold, preventing the probe from memorising
  question wording.
- **Time-ordered splits:** hold out the last N% of rows to simulate a
  deployment scenario where the model encounters newer questions.

---

## Submission

1. Run all cells in `solution.ipynb` through **Section 8**.
2. Collect the generated `predictions.csv` — it contains two columns:
   `id` and `label` (0 or 1).
3. Submit `predictions.csv` together with `results.json` as instructed by
   the competition organisers.

---

## Dependencies

| Package | Version |
|---------|---------|
| `torch` | ≥ 2.0.0 |
| `transformers` | ≥ 4.40.0 |
| `datasets` | ≥ 2.14.0 |
| `scikit-learn` | ≥ 1.3.0 |
| `numpy` | ≥ 1.24.0 |
| `pandas` | ≥ 1.5.0 |
| `tqdm` | ≥ 4.65.0 |

Install all dependencies with:

```bash
pip install -r requirements.txt
```
