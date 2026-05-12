"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


SELECTED_LAYER_OFFSETS = (-1, -2, -4, -8, -12)
TAIL_TOKENS = 64


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    valid_hidden = hidden_states[:, real_positions, :]
    tail_hidden = valid_hidden[:, -TAIL_TOKENS:, :]

    features = []
    n_layers = hidden_states.shape[0]
    for offset in SELECTED_LAYER_OFFSETS:
        layer_idx = n_layers + offset if offset < 0 else offset
        if layer_idx < 0 or layer_idx >= n_layers:
            continue

        layer_tail = tail_hidden[layer_idx]
        last_token = layer_tail[-1]
        tail_mean = layer_tail.mean(dim=0)
        tail_max = layer_tail.max(dim=0).values

        features.extend(
            [
                last_token,
                tail_mean,
                tail_max,
                last_token - tail_mean,
            ]
        )

    return torch.cat(features, dim=0).float()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    valid_hidden = hidden_states[:, real_positions, :].float()
    tail_hidden = valid_hidden[:, -TAIL_TOKENS:, :]

    last_by_layer = valid_hidden[:, -1, :]
    tail_mean_by_layer = tail_hidden.mean(dim=1)
    layer_norms = torch.linalg.vector_norm(last_by_layer, dim=1)
    tail_norms = torch.linalg.vector_norm(tail_mean_by_layer, dim=1)

    adjacent_cos = torch.nn.functional.cosine_similarity(
        last_by_layer[:-1],
        last_by_layer[1:],
        dim=1,
    )
    start_end_cos = torch.nn.functional.cosine_similarity(
        last_by_layer[0].unsqueeze(0),
        last_by_layer[-1].unsqueeze(0),
        dim=1,
    )
    tail_variance = tail_hidden.var(dim=1, unbiased=False).mean(dim=1)
    seq_len = torch.tensor(
        [float(real_positions.numel())],
        dtype=torch.float32,
        device=hidden_states.device,
    )

    return torch.cat(
        [
            seq_len,
            layer_norms,
            tail_norms,
            adjacent_cos,
            start_end_cos,
            tail_variance,
        ],
        dim=0,
    ).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
