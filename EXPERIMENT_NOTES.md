# Experiment Notes

## Goal

Build a lightweight hallucination detector for `Qwen/Qwen2.5-0.5B` hidden
states without changing the fixed infrastructure files. The official entry
point should remain `python solution.py`, and the final run should produce
`results.json` and `predictions.csv`.

## Current Direction

The first implemented version uses a classic probe setup:

- richer hidden-state aggregation in `aggregation.py`;
- a regularized linear classifier in `probe.py`;
- 5-fold stratified validation in `splitting.py`.

The main bet is that hidden-state feature quality matters more than classifier
depth on a dataset of only 689 labelled examples.

## Feature Extraction

The original baseline used the last real token from the final transformer
layer. That is a narrow view of the response, so the new aggregation uses
several late/middle layers: `[-1, -2, -4, -8, -12]`.

For each selected layer, it concatenates:

- last real token;
- mean of the last 64 real tokens;
- max pool over the last 64 real tokens;
- difference between the last token and the last-64-token mean.

The last 64 tokens are a simple proxy for the answer span because `solution.py`
feeds `prompt + response` into the model and the response appears at the end.

## Classifier

The original MLP can easily overfit because the dataset is small and the
feature vector is high-dimensional. The first stronger baseline replaces it
with `LogisticRegression`:

- `StandardScaler` preprocessing;
- `class_weight="balanced"` for the 483/206 class imbalance;
- L2 regularization with `C=0.2`;
- deterministic `liblinear` solver.

Threshold tuning now optimizes validation accuracy, with F1 as a tie-breaker,
because the README says the primary ranking metric is accuracy.

## Validation

The single split was replaced with 5-fold stratified validation. This should
make local results less sensitive to one lucky split and lets the final model
use all labelled rows in the current `solution.py` logic.

## Things To Try Next

- Compare this linear probe against the previous MLP on the same folds.
- Tune selected layers: `[-1, -4, -8, -16]`, `[-2, -6, -10, -14]`.
- Tune tail window sizes: 32, 64, 128.
- Try fewer pooling operations if the model overfits.
- Enable `USE_GEOMETRIC = True` only after checking whether geometric features
  improve cross-fold accuracy.
- Consider PCA or `SelectKBest` if the high-dimensional feature vector is too
  noisy.

## Reproducibility Notes

Do not edit `model.py`, `evaluate.py`, or `solution.py` for the final solution.
Keep generated `predictions.csv` and `results.json` from the official
`solution.py` run. Convert this file into `SOLUTION.md` once the final
experiment is chosen.

## RESULT
★  Primary metric — Test AUROC: 71.90%
