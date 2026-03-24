"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

This module is responsible for converting the raw per-token, per-layer hidden
states produced by ``model.extract_hidden_states`` into a flat feature vector
that is fed to the probe classifier (``probe.HallucinationProbe``).

The pipeline has **two stages** that you can customise independently:

  1. ``aggregate`` — select which transformer layers and which token positions
     to look at, and pool them into a single fixed-length vector.

  2. ``extract_geometric_features`` — optionally compute hand-crafted geometric
     or statistical features from the hidden states (disabled by default).
     Enable it by setting ``USE_GEOMETRIC = True`` in ``solution.ipynb``.

These two stages are wired together by ``aggregation_and_feature_extraction``,
which is the **single entry point** called from the notebook.  You should not
need to change ``aggregation_and_feature_extraction`` itself; focus on
``aggregate`` and, if you want to experiment, on ``extract_geometric_features``.

Qwen2.5-0.5B layer index reference
-----------------------------------
Qwen2.5-0.5B is a **decoder-only** (causal) language model.  It contains
24 transformer layers plus an embedding layer:

  Index 0        → token embedding layer (after positional encoding)
  Index 1 – 24   → transformer layers (low → high level representations)
  Index -1       → final transformer layer (richest semantic information)

The hidden dimension is 896.  Verify the exact values for the model you are
using with ``model.config.num_hidden_layers`` and ``model.config.hidden_size``.

Because this is a **causal** (left-to-right) model, the **last real token**
in the sequence sees all preceding tokens and thus captures the most complete
contextual representation.  This is the natural pooling strategy and is used
as the default below.

Compare this with encoder models (e.g. BERT), where the ``[CLS]`` token at
position 0 aggregates the full sequence.  In causal models the equivalent is
the last real token.
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        containing the hidden states from all layers
                        (embedding + transformer layers) for one sample.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with values 1 for
                        real tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if you concatenate multiple layers.

    Student task:
        Replace or extend the skeleton below.  Strategies to explore:

        **Layer selection:**
          - ``hidden_states[-1]``          → last transformer layer (default)
          - ``hidden_states[-2]``          → penultimate layer
          - ``hidden_states[len//2]``      → middle layer
          - Average of last k layers: ``hidden_states[-k:].mean(0)``

        **Token pooling (for causal LMs):**
          - Last real token (default) — sees all prior context
          - Mean pooling over all real tokens (non-padding)
          - Max pooling: ``(layer * mask_expanded).max(dim=0).values``
          - Weighted average: weight earlier tokens by position

        **Multi-layer fusion:**
          - Concatenate several layers' last-token vectors
          - Weighted sum of layers (learn the weights in the probe)
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    # Default: hidden state of the last real token in the final layer.
    # For a causal LM this position attends to the full input sequence.
    layer = hidden_states[-1]          # (seq_len, hidden_dim)

    # Find the index of the last real (non-padding) token.
    real_positions = attention_mask.nonzero(as_tuple=False)  # (n_real, 1)
    last_pos = int(real_positions[-1].item())                 # scalar index

    feature = layer[last_pos]          # (hidden_dim,)

    return feature
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    These features complement the aggregated hidden-state vector produced by
    ``aggregate`` and can help the probe pick up on properties that a fixed
    pooling strategy might miss (e.g. how much representations vary across
    layers, or the overall magnitude of activations).

    This function is called only when ``USE_GEOMETRIC = True`` is set in
    ``solution.ipynb`` (via the ``use_geometric`` flag passed to
    ``aggregation_and_feature_extraction``).  When ``USE_GEOMETRIC = False``
    (the default) this function is **not called** and contributes nothing to
    the feature matrix, so you can safely leave it as a stub.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample in the dataset.

    Student task:
        Replace the stub below with your own features.  Ideas to explore:

        **Layer-wise norms** — how does activation magnitude change across layers?
          real_mask = attention_mask.bool()
          layer_norms = hidden_states[:, real_mask, :].norm(dim=-1).mean(dim=-1)
          # → shape (n_layers,)

        **Representation drift** — cosine similarity between consecutive layers:
          from torch.nn.functional import cosine_similarity
          drifts = [cosine_similarity(hidden_states[l], hidden_states[l+1], dim=-1)
                    for l in range(hidden_states.size(0) - 1)]
          # each drift has shape (seq_len,) → pool over real tokens

        **Sequence length** — simple proxy for answer verbosity:
          seq_len_feature = attention_mask.float().sum().unsqueeze(0)  # scalar
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    # When you implement real features, return a 1-D tensor instead.
    return torch.zeros(0)

    # ------------------------------------------------------------------


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    This is the **main entry point** called from ``solution.ipynb`` for every
    sample in the dataset.  It combines:

      1. The aggregated hidden-state vector from ``aggregate`` — always
         included.
      2. (Optional) hand-crafted geometric features from
         ``extract_geometric_features`` — included only when
         ``use_geometric=True``.

    The two parts are concatenated into a single 1-D feature vector that is
    then stacked into the feature matrix ``X`` and passed to the probe
    classifier.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a **single sample**, as returned by
                        ``model.extract_hidden_states``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding, also from
                        ``model.extract_hidden_states``.
        use_geometric:  Whether to append geometric features produced by
                        ``extract_geometric_features``.  Controlled by the
                        ``USE_GEOMETRIC`` flag in ``solution.ipynb``.
                        Defaults to ``False``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where:

        - ``feature_dim = hidden_dim`` (or larger if ``aggregate`` returns a
          multi-layer concatenation) when ``use_geometric=False``.
        - ``feature_dim = hidden_dim + n_geometric_features`` when
          ``use_geometric=True``.
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
