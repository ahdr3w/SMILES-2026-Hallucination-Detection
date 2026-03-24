"""
aggregation.py — Token aggregation strategy (student-implemented).

Students: Implement ``aggregate`` to convert a per-token hidden-state tensor
into a single fixed-length feature vector used by the probe classifier.

The function receives the hidden states from *all* transformer layers for a
single sample, giving you full control over which layer to inspect and how to
combine the token representations.

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
