# SOLUTION.md — SMILES-2026 Hallucination Detection

This report documents the submitted solution: how to reproduce it, what was
implemented, and what was tried.

---

## 1. Reproducibility

### Environment

- Python 3.10+
- Dependencies declared in `requirements.txt`
- Recommended hardware: a CUDA-capable GPU (the LLM forward pass on Qwen2.5-0.5B
  for 689 + 100 samples takes ~2.5 min on a free Colab T4; CPU also works but is
  slower).
- Random seed: `42` (used in `splitting.py` and inside `probe.py` for the
  internal early-stopping split).

### Exact commands

```bash
git clone <this-repository>
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate.bat       # Windows

pip install -r requirements.txt
python solution.py
```

`solution.py` produces:
- `results.json` — k-fold CV metrics on `dataset.csv`.
- `predictions.csv` — labels for `data/test.csv` (columns `id,label`).

### Implementation details required for byte-identical reproduction

- Only `aggregation.py`, `probe.py`, and `splitting.py` are modified — every
  other file (`solution.py`, `model.py`, `evaluate.py`, `requirements.txt`) is
  the official version.
- `solution.py`'s `USE_GEOMETRIC = False` flag is **intentionally left
  unchanged**.  Geometric features are still applied because
  `aggregation_and_feature_extraction` in `aggregation.py` always concatenates
  them (the flag is accepted but ignored).
- Feature dimension is `4 × 896 + 896 + (25 + 24 + 3) = 4532`.
- The internal early-stop split inside `probe.fit` is also seeded with
  `random_state=42` so the same network state is selected on every run.

---

## 2. Final solution

### Modified files

| File              | Purpose                                                                |
| ----------------- | ---------------------------------------------------------------------- |
| `aggregation.py`  | Layer / token pooling and hand-crafted geometric features              |
| `probe.py`        | `HallucinationProbe` — regularised MLP with internal early stopping    |
| `splitting.py`    | Stratified 5-fold cross-validation                                     |

All other files are unchanged (per the README's "rest of the codebase shall
remain untouched" instruction).

### Aggregation (`aggregation.py`)

- **`aggregate`** — concatenates the last real (non-padding) token from four
  late layers `(14, 18, 22, 24)` of the 25-state stack, plus a masked mean of
  the final layer.  Late layers were chosen because factuality-related
  representations in decoder-only LMs tend to consolidate in the upper third
  of the stack, while the masked mean adds a token-aggregated view that
  complements the single-token snapshots.  Result: `5 × 896 = 4480` dims.

- **`extract_geometric_features`** — three families of hand-crafted features:
  - Per-layer L2 norm at the last real token (25 dims) — captures the
    activation-magnitude trajectory through the stack.
  - Inter-layer cosine similarity between consecutive layers at the last
    real token (24 dims) — measures representation drift.
  - Length features: `log(n_tokens)`, `sqrt(n_tokens)`, raw `n_tokens`
    (3 dims) — motivated by the dataset analysis below.

- **`aggregation_and_feature_extraction`** — always concatenates `aggregate`
  output with `extract_geometric_features` output (the `use_geometric` flag
  is ignored).  This was the cleanest way to enable geometric features
  without modifying the read-only `solution.py`.

- **Device-correctness:** both functions detect `hidden_states.device` and
  move the CPU-side `attention_mask` and any locally-created tensors onto it,
  matching the existing extraction loop in `solution.py` (which keeps
  `hidden` on the LLM's device and the mask on CPU).

### Probe (`probe.py`)

`HallucinationProbe` subclasses `nn.Module` and exposes the four required
public methods (`fit`, `fit_hyperparameters`, `predict`, `predict_proba`).

- **Architecture:** `Linear(d → 128) → ReLU → Dropout(0.3) → Linear(128 → 1)`
  with `StandardScaler` preprocessing on the 4532-dim feature vector.
- **Training:** Adam (`lr=1e-3`, `weight_decay=1e-4`),
  `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)` to compensate for the
  training set's 70/30 class imbalance.  Up to 500 epochs with **early
  stopping** (patience = 30) on a stratified 15% internal hold-out carved
  from the training data; the best-loss checkpoint is restored at the end.
- **Threshold:** after early stopping, the decision threshold is tuned on the
  same internal hold-out by maximising accuracy (the README's primary
  ranking metric).  `fit_hyperparameters` overrides this when an external
  validation set is available.

### Splitting (`splitting.py`)

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` preserves the
70/30 class ratio in every fold's test slice.  Within each fold a small
stratified slice of the train-plus-validation portion is reserved as the
validation set used by `evaluate.evaluate_fold` to call
`fit_hyperparameters`.  Five folds give a stable test-metric estimate; in
`solution.py` the union of every fold's train+val equals the entire
dataset, so the final probe used to produce `predictions.csv` is trained on
all 689 samples.

### Data analysis that informed the design

- 689 training samples, **70.1% hallucinated / 29.9% truthful** — mild class
  imbalance, addressed via `pos_weight` and stratified splits (no resampling
  needed).
- 100 test samples, all `label` values null.
- Prompts are unique; no prompt overlap between train and test (clean
  split).
- All responses end with `<|endoftext|>` and all prompts use the same
  ChatML template — the meaningful variation lives in the response tokens.
- **Hallucinated responses are noticeably longer** (mean 797 chars vs 421
  chars for truthful), which directly motivated the length features in
  `extract_geometric_features`.

### What contributed most

- Multi-layer last-token pooling (layers 14/18/22/24) was the biggest gain
  over the single-layer last-token baseline: late-layer factuality signal
  dominates, but adding one or two earlier late layers helped AUROC on the
  validation folds.
- Adding the masked mean of the final layer gave a small but consistent
  bump — it adds a token-aggregated signal that complements the
  single-token snapshots.
- The geometric block (norms + drifts + length) gave a small additional
  improvement and is essentially free to compute.

---

## 3. Experiments and failed attempts

- **MLP-only with no scaler.** Early experiments without `StandardScaler`
  preprocessing converged to a near-constant prediction because the
  geometric features and hidden-state activations live on very different
  scales.  Standardising fixed the issue.
- **Tuning threshold on F1 vs accuracy.** The original `probe.py` tuned the
  threshold to maximise F1, but the README explicitly lists accuracy as the
  primary metric.  Tuning on accuracy is more aligned with the metric, at
  the cost of being more sensitive to the test-set class prior — given the
  test set is unlabelled and we cannot verify its prior, this is a
  deliberate trade-off in favour of the primary metric.
- **Last-token-only baseline.** Pooling only the last real token of the
  final layer (the file's original aggregation) gave a smaller feature
  vector but consistently lower validation AUROC.  Multi-layer pooling won.
- **Single train/val/test split (the file's original splitting).** Switched
  to 5-fold stratified CV for a more stable metric estimate; the original
  single split produced fold-to-fold variation that made it hard to judge
  small changes to aggregation or probe hyperparameters.
- **Wider MLPs (256 / 512 hidden units).** Increased capacity made training
  AUROC climb but validation AUROC stagnated or dropped — a clear
  overfitting signal given the 4532-dim feature vector and only ~550
  training samples per fold.  Settled on 128 hidden units.
- **Higher dropout (0.5).** Pushed early stopping to fire too early and
  underfit slightly.  0.3 was a better operating point.
- **Resampling (SMOTE / class undersampling).** The 70/30 imbalance is mild
  and `pos_weight` already compensates inside the loss.  Adding resampling
  on top either hurt or made no difference, so it was dropped.
