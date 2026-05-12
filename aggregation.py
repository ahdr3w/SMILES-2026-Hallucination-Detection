"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features.

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from ``solution.py``.  The geometric features are
always concatenated here (they help and cost almost nothing), so we do not
depend on the ``USE_GEOMETRIC`` flag in ``solution.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LAYERS_TO_POOL = (14, 18, 22, 24)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-layer aggregation: last real token across late layers plus a
    masked mean of the final layer.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer 0 is the token embedding; the last index is the
                        final transformer layer.  Qwen2.5-0.5B has 24
                        transformer layers so ``n_layers == 25``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor concatenating ``len(LAYERS_TO_POOL)`` last-token
        vectors plus one masked-mean of the final layer.
    """
    device = hidden_states.device
    real = attention_mask.to(device=device, dtype=torch.bool)
    nz = real.nonzero(as_tuple=False)
    last_pos = int(nz[-1].item())

    parts: list[torch.Tensor] = []
    for li in LAYERS_TO_POOL:
        parts.append(hidden_states[li, last_pos])

    final_layer = hidden_states[-1]                    # (seq_len, hidden_dim)
    mask_f = real.to(dtype=final_layer.dtype).unsqueeze(-1)  # (seq_len, 1) on device
    mean_pooled = (final_layer * mask_f).sum(dim=0) / mask_f.sum().clamp_min(1.0)
    parts.append(mean_pooled)

    return torch.cat(parts, dim=0).float()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hand-crafted geometric / statistical features.

    Three families, motivated by the data:
      * per-layer L2 norm at the last real token — captures activation
        magnitude evolution along depth (correlates with confidence /
        saturation).
      * inter-layer cosine similarity between consecutive layers at the last
        real token — measures representation drift; truthful and
        hallucinated answers shift differently through the stack.
      * sequence length features — hallucinated responses are noticeably
        longer in the training data (mean 797 vs 421 chars), so the number
        of real tokens is a useful auxiliary signal.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor whose length is ``n_layers + (n_layers - 1) + 3``.
    """
    device = hidden_states.device
    real = attention_mask.to(device=device, dtype=torch.bool)
    nz = real.nonzero(as_tuple=False)
    last_pos = int(nz[-1].item())
    n_real = int(real.sum().item())

    last_token_per_layer = hidden_states[:, last_pos, :].float()  # (n_layers, hidden_dim)

    norms = torch.linalg.vector_norm(last_token_per_layer, dim=-1)  # (n_layers,)

    drifts = F.cosine_similarity(
        last_token_per_layer[:-1], last_token_per_layer[1:], dim=-1
    )  # (n_layers - 1,)

    length_feats = torch.tensor(
        [
            math.log(max(n_real, 1)),
            math.sqrt(max(n_real, 1)),
            float(n_real),
        ],
        dtype=torch.float32,
        device=device,
    )

    return torch.cat([norms, drifts, length_feats], dim=0).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Single entry point used by ``solution.py``.

    Always concatenates the geometric features with the aggregated hidden
    states — the geometric block is cheap, dense, and complements the
    high-dimensional hidden-state vector.  The ``use_geometric`` argument is
    accepted for backward compatibility but is ignored.
    """
    del use_geometric  # geometric features are always included
    agg = aggregate(hidden_states, attention_mask)
    geo = extract_geometric_features(hidden_states, attention_mask)
    return torch.cat([agg, geo], dim=0)
