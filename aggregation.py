"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Experiment 18:
- Raw anchor: last-token hidden states from layers [-1, -2, -4, -8]
- Compact trajectory scalar features from layers [-1, -2, -4, -8, -12]
- Token window: last 32 real tokens
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


RAW_LAYERS = [-4, -8, -12, -16]
TRAJECTORY_LAYERS = [-8, -10, -12, -14, -16]
TOKEN_OFFSET = -2
TOKEN_WINDOW = 32
EPS = 1e-8


def _get_real_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return indices of real non-padding tokens."""
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)

    if real_positions.numel() == 0:
        return torch.tensor([0], device=attention_mask.device, dtype=torch.long)

    return real_positions


def _safe_cosine(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Numerically safe cosine similarity between two vectors."""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=-1).squeeze()


def _effective_rank(eigvals: torch.Tensor) -> torch.Tensor:
    """Effective rank from eigenvalue distribution."""
    eigvals = torch.clamp(eigvals, min=0.0)
    total = eigvals.sum() + EPS

    p = eigvals / total
    entropy = -(p * torch.log(p + EPS)).sum()

    return torch.exp(entropy)


def _spectral_entropy(eigvals: torch.Tensor) -> torch.Tensor:
    """Normalized spectral entropy from eigenvalue distribution."""
    eigvals = torch.clamp(eigvals, min=0.0)
    total = eigvals.sum() + EPS

    p = eigvals / total
    entropy = -(p * torch.log(p + EPS)).sum()

    max_entropy = torch.log(
        torch.tensor(float(len(eigvals)), device=eigvals.device)
    ) + EPS

    return entropy / max_entropy


def _spectral_features(x_window: torch.Tensor) -> list[torch.Tensor]:
    """
    Compute compact covariance/eigenvalue trajectory features.

    Args:
        x_window: Tensor of shape (window_tokens, hidden_dim)

    Returns:
        List of scalar tensors.
    """
    if x_window.shape[0] < 2:
        zero = torch.tensor(0.0, device=x_window.device)
        return [zero, zero, zero, zero, zero, zero]

    x_centered = x_window - x_window.mean(dim=0, keepdim=True)

    gram = (x_centered @ x_centered.T) / max(x_centered.shape[0] - 1, 1)

    eigvals = torch.linalg.eigvalsh(gram).float()
    eigvals = torch.clamp(eigvals, min=0.0)

    trace = eigvals.sum()
    top1 = eigvals[-1] if eigvals.numel() > 0 else torch.tensor(0.0, device=x_window.device)
    top5 = eigvals[-5:].sum() if eigvals.numel() >= 5 else eigvals.sum()

    return [
        trace,
        torch.log1p(trace),
        top1 / (trace + EPS),
        top5 / (trace + EPS),
        _effective_rank(eigvals),
        _spectral_entropy(eigvals),
    ]


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single raw feature vector.

    Current strategy:
    concatenate last-token hidden states from layers [-1, -2, -4, -8].

    Args:
        hidden_states: Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: Tensor of shape (seq_len,).

    Returns:
        Tensor of shape (4 * hidden_dim,).
    """
    real_positions = _get_real_positions(attention_mask)
    last_pos = real_positions[TOKEN_OFFSET]

    features = []

    for layer_idx in RAW_LAYERS:
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)
        last_token = layer[last_pos]
        features.append(last_token.float())

    return torch.cat(features, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract compact trajectory / geometric scalar features.

    This function intentionally avoids adding thousands of raw hidden dimensions.
    Instead it describes the token/layer dynamics using scalar statistics.

    Args:
        hidden_states: Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: Tensor of shape (seq_len,).

    Returns:
        1-D tensor of scalar trajectory features.
    """
    real_positions = _get_real_positions(attention_mask)
    last_pos = real_positions[TOKEN_OFFSET]
    window_positions = real_positions[-TOKEN_WINDOW:]

    last_vectors = []
    window_mean_vectors = []

    features = []

    # ------------------------------------------------------------
    # 1. Per-layer token norm statistics over the last token window
    # ------------------------------------------------------------
    for layer_idx in TRAJECTORY_LAYERS:
        layer = hidden_states[layer_idx]                    # (seq_len, hidden_dim)
        x_window = layer[window_positions].float()          # (W, hidden_dim)
        x_last = layer[last_pos].float()                    # (hidden_dim,)

        last_vectors.append(x_last)
        window_mean_vectors.append(x_window.mean(dim=0))

        token_norms = torch.norm(x_window, p=2, dim=-1)     # (W,)

        features.extend([
            token_norms.mean(),
            token_norms.std(unbiased=False),
            token_norms.min(),
            token_norms.max(),
            torch.norm(x_last, p=2),
        ])

    # ------------------------------------------------------------
    # 2. Inter-layer geometry for last-token representations
    # ------------------------------------------------------------
    for i in range(len(last_vectors) - 1):
        v1 = last_vectors[i]
        v2 = last_vectors[i + 1]

        features.append(_safe_cosine(v1, v2))
        features.append(torch.norm(v1 - v2, p=2))

    # ------------------------------------------------------------
    # 3. Inter-layer geometry for window-mean representations
    # ------------------------------------------------------------
    for i in range(len(window_mean_vectors) - 1):
        v1 = window_mean_vectors[i]
        v2 = window_mean_vectors[i + 1]

        features.append(_safe_cosine(v1, v2))
        features.append(torch.norm(v1 - v2, p=2))

    # ------------------------------------------------------------
    # 4. Last token vs local window mean in each layer
    # ------------------------------------------------------------
    for x_last, x_mean in zip(last_vectors, window_mean_vectors):
        features.append(_safe_cosine(x_last, x_mean))
        features.append(torch.norm(x_last - x_mean, p=2))

    # ------------------------------------------------------------
    # 5. Token trajectory drift inside each layer
    # ------------------------------------------------------------
    for layer_idx in TRAJECTORY_LAYERS:
        layer = hidden_states[layer_idx]
        x_window = layer[window_positions].float()

        x_first = x_window[0]
        x_last = x_window[-1]
        x_mean = x_window.mean(dim=0)

        features.append(_safe_cosine(x_first, x_last))
        features.append(torch.norm(x_last - x_first, p=2))
        features.append(_safe_cosine(x_last, x_mean))
        features.append(torch.norm(x_last - x_mean, p=2))

    # ------------------------------------------------------------
    # 6. Hidden variance over the token window
    # ------------------------------------------------------------
    for layer_idx in TRAJECTORY_LAYERS:
        layer = hidden_states[layer_idx]
        x_window = layer[window_positions].float()

        dim_var = x_window.var(dim=0, unbiased=False)

        features.extend([
            dim_var.mean(),
            dim_var.std(unbiased=False),
            dim_var.max(),
            torch.log1p(dim_var.mean()),
        ])

    # ------------------------------------------------------------
    # 7. Spectral covariance / EigenScore-like features
    # ------------------------------------------------------------
    for layer_idx in TRAJECTORY_LAYERS:
        layer = hidden_states[layer_idx]
        x_window = layer[window_positions].float()

        features.extend(_spectral_features(x_window))

    geo_features = torch.stack([f.float() for f in features])
    geo_features = torch.nan_to_num(
        geo_features,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )

    return geo_features


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append trajectory features.

    Args:
        hidden_states: Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: Tensor of shape (seq_len,).
        use_geometric: Whether to append geometric / trajectory features.

    Returns:
        1-D feature tensor.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
