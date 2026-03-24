"""
probe.py — Hallucination probe classifier (student-implemented).

Students: Implement ``HallucinationProbe`` to classify feature vectors as
truthful (0) or hallucinated (1).  The skeleton provides a small MLP trained
with PyTorch — you are expected to extend it.

``validate.py`` calls the probe as follows::

    probe = HallucinationProbe()
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_val)
    y_prob = probe.predict_proba(X_val)   # for AUROC

All three methods must be implemented and their signatures must not change.
``X`` is a 2-D NumPy array of shape ``(n_samples, feature_dim)``; ``y`` is
a 1-D NumPy array of ints (0 or 1).

``HallucinationProbe`` extends ``torch.nn.Module``, so you can define the
network architecture in ``__init__`` / ``_build_network``, override
``forward``, and experiment with any PyTorch-based training loop in ``fit``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Inherits from ``torch.nn.Module`` so students can override ``forward`` and
    use standard PyTorch patterns (custom layers, optimisers, schedulers, etc.).

    The skeleton implements a single hidden-layer MLP with a ``StandardScaler``
    pre-processing step.  The network is built lazily in ``fit()`` once the
    feature dimension is known.

    Student task:
        Replace or extend the skeleton below.  Ideas to explore:

        **Architecture:**
          - Deeper MLP: ``nn.Linear → ReLU → nn.Linear → ReLU → nn.Linear``
          - Batch normalisation: ``nn.BatchNorm1d``
          - Dropout regularisation: ``nn.Dropout(p=0.3)``

        **Optimisation:**
          - Different optimisers: ``torch.optim.SGD``, ``torch.optim.AdamW``
          - Learning-rate schedulers: ``torch.optim.lr_scheduler.StepLR``
          - More / fewer training epochs; early stopping on a held-out set

        **Class imbalance:**
          - Adjust ``pos_weight`` in ``nn.BCEWithLogitsLoss``
          - Oversample the minority class before fitting
    """

    def __init__(self) -> None:
        super().__init__()
        # Network is built lazily in fit() once input_dim is known.
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.
        Override this method (or the constructor) to change the architecture.

        Args:
            input_dim: Dimensionality of the feature vectors produced by
                       ``aggregation.aggregate``.
        """
        self._net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales the features with ``StandardScaler``, builds the network (if not
        already built), and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        X_scaled = self._scaler.fit_transform(X)

        self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # Handle class imbalance: weight positive examples by neg/pos ratio.
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the training loop below.
        # ------------------------------------------------------------------
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.train()
        for _ in range(200):
            optimizer.zero_grad()
            logits = self(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
        # ------------------------------------------------------------------

        self.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

