# Experiment Log

## baseline_v1
- aggregation.py: original repository version
- probe.py: original repository version
- splitting.py: custom Stratified 5-fold + internal validation split
- feature_dim: 896
- use_geometric: False
- dataset: data/dataset.csv
- model: Qwen/Qwen2.5-0.5B

### Final baseline metrics
- mean test AUROC: 71.23%
- mean test Accuracy: 69.81%
- mean test F1: 81.15%

### Notes
This is the fixed baseline used for all further ablations.
All future experiments must be compared against baseline_v1.
Multi-layer pooled aggregations tested earlier overfit and underperformed this baseline.
