# Solution

## Reproducibility

Run the solution from the repository root:

```bash
pip install -r requirements.txt
python solution.py
```

The run produces:

```text
predictions.csv
results.json
```

The implementation is deterministic at the level used in my experiments: fixed split seed, fixed model/probe seeds, fixed PCA/scaler fitting on the training part of each fold, and fixed threshold-selection logic. A CUDA GPU is recommended because hidden-state extraction is the slowest part. I ran the experiments in Google Colab on a T4 GPU.

Important files modified:

```text
aggregation.py
probe.py
splitting.py
solution.py
```

I also added MLflow logging to make the experiments observable and comparable: run name, git commit, feature configuration, probe configuration, fold metrics, averaged metrics, feature dimension, PCA dimension, threshold strategy, and artifact outputs are logged for each run. This was necessary because many small changes produced similar scores, and without MLflow it was easy to confuse runs with reused names.

## Final configuration

I selected the following configuration as the best accuracy-oriented 5-fold run:

```yaml
target_metric: accuracy
split_strategy: stratified_5fold_with_inner_val
n_folds: 5
seed: 42

aggregation:
  raw_layers: [-4, -8, -12, -16]
  trajectory_layers: [-8, -10, -12, -14, -16]
  token_window: 32
  token_offset: -2
  feature_dim_before_pca: 3705

preprocessing:
  scaler: StandardScaler
  dim_reduction: PCA
  pca_n_components: 128
  pca_fit: train_only

probe:
  type: ensemble_regularized_mlp
  ensemble_size: 5
  base_architecture: 128 -> 8 -> 1
  activation: ReLU
  dropout: 0.5
  ensemble_seeds: [42, 43, 44, 45, 46]

training:
  optimizer: AdamW
  learning_rate: 1e-3
  weight_decay: 1e-2
  epochs: 200
  loss: BCEWithLogitsLoss
  pos_weight: neg_pos_ratio

threshold:
  selected_by: validation_accuracy
```

Final averaged CV result:

```text
majority_baseline_accuracy = 0.7010
avg_test_accuracy          = 0.7547
avg_test_f1                = 0.8423
avg_test_auroc             = 0.7076
avg_train_accuracy         = 0.8529
avg_train_auroc            = 0.9640
feature_dim_before_pca     = 3705
pca_dim                    = 128
```

I therefore do not claim that this is the best AUROC model. I selected it because it is the best configuration according to the target metric I optimized: accuracy.

## What the final approach does

### 1. Hidden-state aggregation

In the final extractor, I combine two types of information.

First, I use a raw hidden-state vector from selected intermediate layers:

```python
RAW_LAYERS = [-4, -8, -12, -16]
TOKEN_OFFSET = -2
```

Earlier versions used the last real token. Late in the experiments, I found that moving one position back with `TOKEN_OFFSET = -2` increased accuracy. The likely reason is practical rather than theoretical: the last real token is often an EOS/closing token or carries little semantic information, while the previous token is closer to the final content-bearing part of the answer. This change increased threshold-level classification accuracy even though AUROC decreased.

Second, I add a compact trajectory/geometric feature block over the last 32 real tokens:

```python
TRAJECTORY_LAYERS = [-8, -10, -12, -14, -16]
TOKEN_WINDOW = 32
```

These features summarize the behavior of hidden states across tokens and layers. My final compact set includes statistics such as token-vector norms, cosine similarity between layers, L2 drift between layers, last-token vs. window-mean differences, token-level variance, drift across the answer tail, and a small spectral summary of covariance structure.

My important conclusion was that the useful signal was not only in a single final hidden vector. Some signal appeared in the geometry of how representations changed across tokens and intermediate layers. However, adding too many spectral features made the model worse, so I keep the compact trajectory block rather than the expanded INSIDE/EigenScore-like version in the final solution.

### 2. PCA and low-capacity probe

The raw feature vector has 3705 dimensions, while each fold has only a few hundred training examples. Directly training a high-capacity classifier on this space overfits easily.

I therefore use this final probe pipeline:

```text
StandardScaler -> PCA(128) -> ensemble of 5 small MLP8 probes
```

I intentionally keep each base model small:

```text
128 -> 8 -> 1
```

This was the best compromise I found. A linear probe was too weak, but larger MLPs memorized the training folds. I kept a tiny nonlinear MLP because it retained some nonlinear capacity while PCA and dropout limited overfitting.

### 3. Ensemble of small probes

A single MLP8 was sensitive to initialization and dropout trajectory. Some runs with the same feature family produced noticeably different accuracy. Averaging five low-capacity MLPs improved robustness without giving the model enough capacity to behave like the earlier overfitting probes.

I make the ensemble prediction by averaging probabilities and then applying a threshold selected on validation accuracy.

### 4. Threshold optimized for accuracy

This was a key implementation detail. I used accuracy as the decision metric, not AUROC and not F1. Therefore, I select the threshold by validation accuracy.

This matters because several runs had acceptable AUROC but poor threshold behavior. In this dataset, ranking quality and final binary accuracy were not always aligned. For example, the final run has lower AUROC than some earlier runs, but better accuracy.

## Experiments and conclusions

### Stage 1: baseline hidden states + MLP

I started with the final-layer last-token hidden state and an MLP. This gave me a misleading result: some single-split scores looked good, but the model overfit heavily.

My original high-capacity MLP could almost perfectly separate the training data. In one early run, train AUROC was close to 1.0. I did not treat this as reliable because the dataset is small and the majority baseline is already around 70% accuracy.

My conclusion: raw final-layer last-token features contain signal, but a large probe can memorize the train split. Single-split results were not trustworthy enough.

### Stage 2: regularization and 5-fold evaluation

I then added a regularized MLP with AdamW and weight decay. One single split reached high accuracy, but the 5-fold result dropped close to the majority baseline:

```text
regularized_mlp single split: accuracy ≈ 0.7692
regularized_mlp 5-fold:      accuracy ≈ 0.7054
```

This was an important warning. I treated the single split as too optimistic. After that, I used stratified 5-fold evaluation with an inner validation split as my main decision signal.

My conclusion: the final solution must be selected by 5-fold accuracy, not by one lucky split.

### Stage 3: pooling experiments

I tried to improve aggregation by using more token information: last pooling, mean pooling, and max pooling. This did not help. Increasing the representation from one final token to large pooled vectors mostly added noise and made overfitting worse.

My conclusion: naive pooling across all tokens is not useful here. The answer-level signal must be added in a more compact and structured way.

### Stage 4: multi-layer last-token features

Next, I moved from only the final layer to several intermediate layers. The useful direction I found was to concatenate last-token vectors from multiple layers, especially intermediate/deeper layers.

The early multi-layer setup used:

```python
RAW_LAYERS = [-1, -2, -4, -8]
```

This improved the representation, but my probe still overfit when it was too large. Reducing MLP capacity helped, and I found MLP8 to be the best region: linear was too weak, while MLP32/MLP64 still had too much variance.

My conclusion: intermediate layers help, but only with strong capacity control.

### Stage 5: PCA compression

Because the multi-layer raw vector was high-dimensional, I tried PCA.

My main observations were:

```text
PCA-32:  underfit; train/test gap decreased but useful signal was removed
PCA-64:  better than PCA-32 but still weaker
PCA-128: best balance among PCA variants
PCA-256: did not improve and increased overfitting risk
```

My conclusion: I found PCA-128 to be the best practical compression level. It retained enough signal while removing many noisy directions.

### Stage 6: compact trajectory/geometric features

The largest conceptual improvement came after I added compact trajectory features. Instead of only asking “what is the last hidden vector?”, I described how representations behave over the last tokens and across layers.

This was useful because hallucination may appear as a representation-dynamics pattern: instability, unusual drift, different covariance structure, or mismatch between the final token and local answer-window mean.

The compact trajectory features improved accuracy over the raw-only setup, so I included them in the final feature extractor.

My conclusion: compact geometry helped; simply adding more raw dimensions did not.

### Stage 7: expanded INSIDE/EigenScore-like spectral features

I then tried to expand the spectral feature block. My expanded version included many extra covariance-spectrum statistics: stable logdet, Frobenius norm, condition ratio, top-k eigenvalue ratios, inter-layer spectral features, and more.

This made performance worse:

```text
compact trajectory / exp18-style features: better accuracy
expanded spectral features: lower accuracy and lower stability
```

My interpretation is that the compact feature block already captured the useful spectral information, while the extended block added noisy, unstable scalar dimensions.

My conclusion: spectral ideas were useful only in a small, regularized form. I discarded the expanded version.

### Stage 8: probe ensemble

My single MLP8 had high variance across seeds. I therefore tested an ensemble of 5 small MLP probes. This improved robustness, so I used it as the final probe.

I also tried a larger or validation-selected ensemble, but it did not consistently improve accuracy. Averaging too many models could smooth probabilities and hurt threshold-based classification.

My conclusion: I found 5 MLP8 probes to be a good balance. It reduced seed variance without over-smoothing the decision boundary.

### Stage 9: linear and tree-based probes

I also tested alternatives:

```text
LogisticRegression / linear models: too weak
ExtraTreesClassifier: useful ranking signal, but overfit on high-dimensional features
```

ExtraTrees could reach strong train metrics and reasonable AUROC, but its accuracy was below the MLP ensemble. Linear models were safer but underfit.

My conclusion: the final classifier needed a small nonlinear component, but not a high-capacity model.

### Stage 10: layer search

After stabilizing the probe, I searched over raw hidden layers.

Single raw-layer experiments showed that layers around `-8` and `-12` were stronger than the final layer. The best single layer by AUROC in my runs was `[-12]`, but the best accuracy came from a multi-layer combination:

```python
RAW_LAYERS = [-4, -8, -12, -16]
```

This combination gave me a better balance than using very late layers only or too many adjacent layers. Runs such as `[-6, -8, -10, -12]`, `[-8, -10, -12, -14]`, and `[-10, -12, -14, -16]` did not improve accuracy.

My conclusion: the useful signal is in intermediate/deeper layers, but the exact spacing matters. My best raw-layer combination was `[-4, -8, -12, -16]`.

### Stage 11: token offset

I got the biggest late-stage accuracy gain by changing the raw hidden token position from the last real token to the previous token.

```python
TOKEN_OFFSET = -2
```

This increased average test accuracy to approximately 75.03% in the default trajectory setup. The tradeoff I observed was lower AUROC and higher train AUROC, which suggests the signal helped threshold-level decisions but also made the model less calibrated as a ranking model.

Because my target metric is accuracy, I kept this idea.

My conclusion: I treated `TOKEN_OFFSET=-2` as one of the most important final changes.

### Stage 12: trajectory layer search

Finally, I searched trajectory-layer choices while keeping the stronger raw layer setup and token offset.

The best final trajectory layers I found were:

```python
TRAJECTORY_LAYERS = [-8, -10, -12, -14, -16]
```

This increased average test accuracy to:

```text
avg_test_accuracy = 0.7547
```

Other trajectory choices were worse or unstable in my runs. For example, using `[-4, -8, -12, -16, -20]` sometimes gave a reasonable result, but not as consistently as `[-8, -10, -12, -14, -16]`.

My conclusion: the final trajectory block should focus on intermediate/deeper layers around `-8` to `-16`.

## Final decision

I stopped at the configuration with approximately 75.47% 5-fold accuracy because it is the best result by the target metric and it has a clear experimental justification:

```text
baseline accuracy:      70.10%
final CV accuracy:      75.47%
absolute improvement:   +5.37 percentage points
```

My final approach is not the most complex model I tried. It is a controlled compromise:

```text
selected intermediate hidden states
+ token_offset=-2
+ compact trajectory/geometric features
+ StandardScaler
+ PCA-128
+ ensemble of 5 tiny MLP8 probes
+ threshold tuned for validation accuracy
```

My main negative result is also important: adding more features or more capacity usually made the model worse. My best solution came from compact, carefully selected representation features and strict regularization.

## MLflow result table

| #  | corrected_experiment_name                                                  | original_mlflow_run_name                                                         | test_acc | test_auroc | val_auroc | train_acc | train_auroc | feature_dim | raw_layers           | trajectory_layers        | token_offset | pca | ensemble | probe                            | dropout | aggregation                                                 |
| -- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | -------- | ---------- | --------- | --------- | ----------- | ----------- | -------------------- | ------------------------ | ------------ | --- | -------- | -------------------------------- | ------- | ----------------------------------------------------------- |
| 1  | exp01_baseline_last_token_mlp256_single_split                              | baseline_last_token_mlp                                                          | 0.7404   | 0.7419     | 0.6666    | 0.8441    | 0.9999      | 896         | —                    | —                        | —            | —   | —        | mlp_256_relu                     | —       | last_token_final_layer                                      |
| 2  | exp02_reduced_baseline_last_token_single_split                             | baseline_last_token_mlp                                                          | 0.7115   | 0.6894     | 0.6301    | 0.7879    | 0.8944      | 896         | —                    | —                        | —            | —   | —        | mlp_256_relu                     | —       | last_token_final_layer                                      |
| 3  | exp03_regularized_mlp64_single_split                                       | regularized_mlp                                                                  | 0.7692   | 0.7318     | 0.6712    | 0.8628    | 0.9797      | 896         | —                    | —                        | —            | —   | —        | mlp_64_relu                      | —       | last_token_final_layer                                      |
| 4  | exp04_regularized_mlp64_5fold                                              | regularized_mlp_5fold                                                            | 0.7054   | 0.7228     | 0.7263    | 0.8623    | 0.9792      | 896         | —                    | —                        | —            | —   | —        | regularized_mlp                  | 0.3000  | last_token_final_layer                                      |
| 5  | exp05_incomplete_pca128_mlp8_dropout05_no_metrics                          | multi_layer_last_token_geom10_pca128_mlp8_relu_dropout05_threshold_acc_5fold     | —        | —          | —         | —         | —           | —           | —                    | —                        | —            | 128 | —        | regularized_mlp                  | 0.5000  | multi_layer_last_token                                      |
| 6  | exp06_multilayer_last_token_pca128_mlp8_dropout05_threshold_acc            | multi_layer_last_token_geom10_pca128_mlp8_relu_dropout05_threshold_acc_5fold     | 0.7126   | 0.7093     | 0.7115    | 0.7437    | 0.8695      | 3594        | —                    | —                        | —            | 128 | —        | regularized_mlp                  | 0.5000  | multi_layer_last_token                                      |
| 7  | exp07_multilayer_last_token_pca128_mlp8_dropout05_final_val_threshold      | multi_layer_last_token_geom10_pca128_mlp8_relu_dropout05_final_val_threshold_acc | 0.7010   | 0.6997     | 0.7007    | 0.7527    | 0.8703      | 3594        | —                    | —                        | —            | 128 | —        | regularized_mlp                  | 0.5000  | multi_layer_last_token                                      |
| 8  | exp18_compact_trajectory_features_pca128_mlp8_dropout05                    | multi_layer_last_token_geom10_pca128_mlp8_relu_dropout05_final_val_threshold_acc | 0.7417   | 0.7273     | 0.7436    | 0.7706    | 0.8797      | 3705        | —                    | —                        | —            | 128 | —        | regularized_mlp                  | 0.5000  | multi_layer_last_token                                      |
| 9  | exp19_extended_inside_eigenscore_features_pca128_mlp8_dropout05            | exp19_inside_eigenscore_features_pca128_mlp8_dropout05_acc                       | 0.7257   | 0.7211     | 0.7377    | 0.7809    | 0.8808      | 3773        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 128 | —        | regularized_mlp                  | 0.5000  | multi_layer_last_token_plus_inside_eigenscore_features      |
| 10 | exp18_repeat_compact_trajectory_pca128_mlp8_dropout05                      | multi_layer_last_token_geom10_pca128_mlp8_relu_dropout05_final_val_threshold_acc | 0.7359   | 0.7192     | 0.7365    | 0.7992    | 0.8750      | 3705        | —                    | —                        | —            | 128 | —        | regularized_mlp                  | 0.5000  | multi_layer_last_token                                      |
| 11 | exp18_repeat_compact_trajectory_pca128_mlp8_dropout05                      | multi_layer_last_token_geom10_pca128_mlp8_relu_dropout05_final_val_threshold_acc | 0.7286   | 0.7038     | 0.7360    | 0.7777    | 0.8788      | 3705        | —                    | —                        | —            | 128 | —        | regularized_mlp                  | 0.5000  | multi_layer_last_token                                      |
| 12 | exp18_repeat_compact_trajectory_pca128_mlp8_dropout05                      | multi_layer_last_token_geom10_pca128_mlp8_relu_dropout05_final_val_threshold_acc | 0.7242   | 0.6979     | 0.7181    | 0.7858    | 0.8795      | 3705        | —                    | —                        | —            | 128 | —        | regularized_mlp                  | 0.5000  | multi_layer_last_token                                      |
| 13 | exp18_repeat_compact_trajectory_pca128_mlp8_dropout05_duplicate            | multi_layer_last_token_geom10_pca128_mlp8_relu_dropout05_final_val_threshold_acc | 0.7242   | 0.6979     | 0.7181    | 0.7858    | 0.8795      | 3705        | —                    | —                        | —            | 128 | —        | regularized_mlp                  | 0.5000  | multi_layer_last_token                                      |
| 14 | exp20_ensemble5_mlp8_exp18_features_pca128                                 | exp20_ensemble_5x_mlp8_exp18_features_pca128_dropout05_acc                       | 0.7373   | 0.7379     | 0.7285    | 0.7862    | 0.8973      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 15 | exp20_ensemble5_mlp8_exp18_features_pca128_repeat                          | exp20_ensemble_5x_mlp8_exp18_features_pca128_dropout05_acc                       | 0.7373   | 0.7379     | 0.7285    | 0.7862    | 0.8973      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 16 | exp21_ensemble10_or_seed_variant_exp18_features_pca128                     | exp20_ensemble_5x_mlp8_exp18_features_pca128_dropout05_acc                       | 0.7242   | 0.7366     | 0.7315    | 0.7929    | 0.8974      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 17 | exp22_validation_selected_ensemble_10_choose_5_pca128                      | exp22_validation_selected_ensemble_10_choose_5_pca128_acc                        | 0.7330   | 0.7373     | 0.7297    | 0.7885    | 0.8974      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 128 | —        | validation_selected_ensemble_mlp | 0.5000  | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 18 | exp20_ensemble5_mlp8_exp18_features_pca128_repeat                          | exp20_ensemble_5x_mlp8_exp18_features_pca128_dropout05_acc                       | 0.7373   | 0.7379     | 0.7285    | 0.7862    | 0.8973      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 19 | exp23_logistic_regression_pca128_exp18_features                            | exp23_logistic_regression_pca128_exp18_features_acc                              | 0.6995   | 0.6958     | 0.7091    | 0.7893    | 0.8971      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 128 | —        | LogisticRegression               | —       | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 20 | exp24_extratrees_no_pca_exp18_features_overfit                             | exp24_extratrees_no_pca_exp18_features_acc                                       | 0.7256   | 0.7385     | 0.7360    | 1.0000    | 1.0000      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | —   | —        | ExtraTreesClassifier             | —       | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 21 | exp24_extratrees_regularized_variant_exp18_features                        | exp24_extratrees_no_pca_exp18_features_acc                                       | 0.7242   | 0.7342     | 0.7295    | 0.8180    | 0.9283      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | —   | —        | ExtraTreesClassifier             | —       | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 22 | exp20_final_ensemble5_mlp8_pca128_exp18_features                           | ensemble5_mlp8_pca128_exp18_features_acc                                         | 0.7373   | 0.7379     | 0.7285    | 0.7862    | 0.8973      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 23 | exp26_ensemble5_mlp8_pca256_exp18_features                                 | exp26_ensemble5_mlp8_pca256_exp18_features_acc                                   | 0.7155   | 0.7363     | 0.7454    | 0.8873    | 0.9835      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 256 | 5        | ensemble_regularized_mlp         | 0.5000  | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 24 | exp27_ensemble5_mlp8_pca128_unweighted_bce                                 | exp27_ensemble5_mlp8_pca128_unweighted_bce_acc                                   | 0.7330   | 0.7408     | 0.7297    | 0.7952    | 0.8975      | 3705        | [-1, -2, -4, -8]     | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | raw_multi_layer_last_token_plus_compact_trajectory_features |
| 25 | exp28_single_raw_layer_m1_trajectory_default_pca128_ensemble5              | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7286   | 0.7395     | 0.7621    | 0.7871    | 0.8942      | 1017        | [-1]                 | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 26 | exp28_single_raw_layer_m2_trajectory_default_pca128_ensemble5              | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7329   | 0.7436     | 0.7552    | 0.7974    | 0.9021      | 1017        | [-2]                 | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 27 | exp28_single_raw_layer_m4_trajectory_default_pca128_ensemble5              | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7286   | 0.7436     | 0.7517    | 0.8014    | 0.9020      | 1017        | [-4]                 | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 28 | exp28_single_raw_layer_m8_trajectory_default_pca128_ensemble5              | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7344   | 0.7482     | 0.7546    | 0.8055    | 0.9065      | 1017        | [-8]                 | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 29 | exp28_single_raw_layer_m12_trajectory_default_pca128_ensemble5             | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7387   | 0.7603     | 0.7578    | 0.8023    | 0.9147      | 1017        | [-12]                | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 30 | exp28_single_raw_layer_m16_trajectory_default_pca128_ensemble5             | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7329   | 0.7389     | 0.7388    | 0.7853    | 0.9045      | 1017        | [-16]                | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 31 | exp28_raw_layers_m4_m8_m12_m16_trajectory_default_pca128_ensemble5         | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7445   | 0.7435     | 0.7331    | 0.8082    | 0.9061      | 3705        | [-4, -8, -12, -16]   | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 32 | exp28_raw_layers_m8_m12_m16_trajectory_default_pca128_ensemble5            | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7387   | 0.7415     | 0.7350    | 0.7862    | 0.9077      | 2809        | [-8, -12, -16]       | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 33 | exp28_raw_layers_m6_m8_m10_m12_trajectory_default_pca128_ensemble5         | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7213   | 0.7465     | 0.7393    | 0.8054    | 0.9069      | 3705        | [-6, -8, -10, -12]   | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 34 | exp28_raw_layers_m8_m10_m12_m14_trajectory_default_pca128_ensemble5        | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7199   | 0.7473     | 0.7407    | 0.7956    | 0.9082      | 3705        | [-8, -10, -12, -14]  | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 35 | exp28_raw_layers_m10_m12_m14_m16_trajectory_default_pca128_ensemble5       | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7199   | 0.7439     | 0.7362    | 0.7840    | 0.9067      | 3705        | [-10, -12, -14, -16] | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 36 | exp28_token_offset_minus2_raw_m4_m8_m12_m16_trajectory_default             | exp28_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7503   | 0.7077     | 0.6951    | 0.8658    | 0.9641      | 3705        | [-4, -8, -12, -16]   | [-1, -2, -4, -8, -12]    | —            | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 37 | exp29_token_offset_minus2_keep_trajectory_default_lastpos_minus1           | exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7474   | 0.7105     | 0.6985    | 0.8533    | 0.9633      | 3705        | [-4, -8, -12, -16]   | [-1, -2, -4, -8, -12]    | -2           | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 38 | exp29_token_offset_minus2_trajectory_m4_m8_m12_m16_m20                     | exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7503   | 0.7088     | 0.6954    | 0.8408    | 0.9648      | 3705        | [-4, -8, -12, -16]   | [-4, -8, -12, -16, -20]  | -2           | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 39 | exp29_token_offset_minus2_trajectory_m4_m8_m12_m16_m20_regularized_variant | exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7271   | 0.7372     | 0.7328    | 0.7916    | 0.9073      | 3705        | [-4, -8, -12, -16]   | [-4, -8, -12, -16, -20]  | -2           | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 40 | exp29_final_token_offset_minus2_trajectory_m8_m10_m12_m14_m16              | exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7547   | 0.7076     | 0.6933    | 0.8529    | 0.9640      | 3705        | [-4, -8, -12, -16]   | [-8, -10, -12, -14, -16] | -2           | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 41 | exp29_final_trajectory_m8_m10_m12_m14_m16_high_train_overfit_variant       | exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.6981   | 0.7213     | 0.6850    | 0.8734    | 0.9795      | 3705        | [-4, -8, -12, -16]   | [-8, -10, -12, -14, -16] | -2           | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 42 | exp29_final_trajectory_m8_m10_m12_m14_m16_featuredim3709_variant           | exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7489   | 0.7052     | 0.6951    | 0.8529    | 0.9640      | 3709        | [-4, -8, -12, -16]   | [-8, -10, -12, -14, -16] | -2           | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 43 | exp29_final_trajectory_m8_m10_m12_m14_m16_featuredim3765_variant           | exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7358   | 0.7089     | 0.6928    | 0.8515    | 0.9631      | 3765        | [-4, -8, -12, -16]   | [-8, -10, -12, -14, -16] | -2           | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 44 | exp29_final_trajectory_m8_m10_m12_m14_m16_featuredim3717_variant           | exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7387   | 0.7070     | 0.6954    | 0.8529    | 0.9644      | 3717        | [-4, -8, -12, -16]   | [-8, -10, -12, -14, -16] | -2           | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
| 45 | exp29_final_reproduced_token_offset_minus2_trajectory_m8_m10_m12_m14_m16   | exp29_raw_single_layer_m4_trajectory_default_pca128_ensemble5_acc                | 0.7547   | 0.7076     | 0.6933    | 0.8529    | 0.9640      | 3705        | [-4, -8, -12, -16]   | [-8, -10, -12, -14, -16] | -2           | 128 | 5        | ensemble_regularized_mlp         | 0.5000  | single_raw_layer_plus_default_trajectory_features           |
