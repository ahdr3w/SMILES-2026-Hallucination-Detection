"""
aggregation.py — Token aggregation strategy (student-implemented).

Students: Implement ``aggregate`` to convert a per-token hidden-state tensor
into a single fixed-length feature vector used by the probe classifier.

The function receives the hidden states from *all* transformer layers for a
single sample, giving you full control over which layer to inspect and how to
combine the token representations.

DistilBERT layer index reference
---------------------------------
  Index 0 → embedding layer       (subword embeddings + positional encoding)
  Index 1 → transformer layer 1   (low-level syntax)
  Index 2 → transformer layer 2
  Index 3 → transformer layer 3   (syntactic structure)
  Index 4 → transformer layer 4
  Index 5 → transformer layer 5
  Index 6 → transformer layer 6   (high-level semantics)

The default skeleton uses the final transformer layer (index 6) with mean
pooling over non-padding tokens — a strong baseline that you are expected to
improve upon.
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
                        containing the hidden states from all 7 layers
                        (embedding + 6 transformer layers) for one sample.
                        Layer index 0 is the embedding; index 6 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with values 1 for
                        real tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if you concatenate multiple layers.

    Student task:
        Replace or extend the skeleton below. Strategies to explore:

        **Layer selection:**
          - ``hidden_states[-1]``        → last transformer layer (default)
          - ``hidden_states[3]``         → middle layer
          - ``hidden_states[1:4].mean(0)`` → average of early layers

        **Token pooling:**
          - Mean over non-padding tokens (default)
          - ``hidden_states[-1, 0]``     → [CLS] token representation
          - ``hidden_states[-1, -1]``    → last real token
          - Max pooling across positions
          - Attention-weighted pooling using the mask as soft weights

        **Multi-layer fusion:**
          - Concatenate several layers' pooled vectors
          - Weighted sum of layers (learn the weights in the probe)
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    # Default: mean pooling over non-padding tokens in the last layer.
    layer = hidden_states[-1]          # (seq_len, hidden_dim)
    mask = attention_mask.float()      # (seq_len,)  — 1 = real token

    # Expand mask for broadcasting and compute masked mean.
    mask_expanded = mask.unsqueeze(-1)                         # (seq_len, 1)
    sum_hidden = (layer * mask_expanded).sum(dim=0)            # (hidden_dim,)
    count = mask_expanded.sum(dim=0).clamp(min=1e-9)           # (1,)
    feature = sum_hidden / count                               # (hidden_dim,)

    return feature
    # ------------------------------------------------------------------
