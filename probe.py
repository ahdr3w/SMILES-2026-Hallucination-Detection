
"""
probe.py — Hallucination probe classifier (student-implemented).

Experiment: ensemble of small MLP probes.

The public API is unchanged:
    - fit
    - fit_hyperparameters
    - predict
    - predict_proba
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def seed_everything(seed: int = 42) -> None:
    """Make torch/numpy/random behavior reproducible."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HallucinationProbe(nn.Module):
    """Binary classifier ensemble for hallucination detection.

    Shared preprocessing:
        X -> StandardScaler -> PCA

    Probe:
        average of several small MLPs:
            Linear(input_dim, 8) -> ReLU -> Dropout(0.5) -> Linear(8, 1)
    """

    def __init__(self) -> None:
        super().__init__()

        self._scaler = StandardScaler()

        self._pca_n_components = 128
        self._pca: PCA | None = None

        self._threshold: float = 0.5

        # Ensemble settings
        self._seeds = [42, 43, 44, 45, 46]
        self._nets = nn.ModuleList()

        # Training settings
        self._epochs = 200
        self._lr = 1e-3
        self._weight_decay = 1e-2
        self._dropout = 0.5
        self._hidden_dim = 8

    # ------------------------------------------------------------------
    # Network definition
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> nn.Sequential:
        """Build one base MLP model."""
        return nn.Sequential(
            nn.Linear(input_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._hidden_dim, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning averaged raw logits.

        Args:
            x: Float tensor of shape (n_samples, feature_dim).

        Returns:
            1-D tensor of averaged logits.
        """
        if len(self._nets) == 0:
            raise RuntimeError("Networks have not been built yet. Call fit() first.")

        logits_list = []

        for net in self._nets:
            logits = net(x).squeeze(-1)
            logits_list.append(logits)

        logits_stack = torch.stack(logits_list, dim=0)  # (n_models, n_samples)

        return logits_stack.mean(dim=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train an ensemble of small MLP probes.

        Args:
            X: Feature matrix of shape (n_samples, feature_dim).
            y: Integer labels, 0 = truthful, 1 = hallucinated.

        Returns:
            self
        """
        # ------------------------------------------------------------
        # 1. Shared preprocessing
        # ------------------------------------------------------------
        X_scaled = self._scaler.fit_transform(X)

        n_components = min(
            self._pca_n_components,
            X_scaled.shape[0] - 1,
            X_scaled.shape[1],
        )

        self._pca = PCA(n_components=n_components, random_state=42)
        X_pca = self._pca.fit_transform(X_scaled)

        X_t = torch.from_numpy(X_pca).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # ------------------------------------------------------------
        # 2. Loss
        # ------------------------------------------------------------
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        #------------------------------------------------------------
        # 3. Train ensemble
        # ------------------------------------------------------------
        self._nets = nn.ModuleList()

        for seed in self._seeds:
            seed_everything(seed)

            net = self._build_network(X_pca.shape[1])
            optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=self._lr,
                weight_decay=self._weight_decay,
            )

            net.train()

            for _ in range(self._epochs):
                optimizer.zero_grad()

                logits = net(X_t).squeeze(-1)
                loss = criterion(logits, y_t)

                loss.backward()
                optimizer.step()

            net.eval()
            self._nets.append(net)

        self.eval()

        return self

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HallucinationProbe":
        """Tune decision threshold on validation set to maximize accuracy.

        Args:
            X_val: Validation feature matrix.
            y_val: Validation labels.

        Returns:
            self
        """
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(
            np.concatenate([
                probs,
                np.linspace(0.01, 0.99, 199),
            ])
        )

        best_threshold = 0.5
        best_accuracy = -1.0

        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = accuracy_score(y_val, y_pred_t)

            if score > best_accuracy:
                best_accuracy = score
                best_threshold = float(t)

        self._threshold = best_threshold

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using tuned threshold."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities averaged across ensemble members."""
        if self._pca is None:
            raise RuntimeError("PCA has not been fitted yet. Call fit() first.")

        X_scaled = self._scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)

        X_t = torch.from_numpy(X_pca).float()

        prob_list = []

        self.eval()

        with torch.no_grad():
            for net in self._nets:
                net.eval()
                logits = net(X_t).squeeze(-1)
                prob_pos = torch.sigmoid(logits)
                prob_list.append(prob_pos)

        probs_stack = torch.stack(prob_list, dim=0)  # (n_models, n_samples)
        prob_pos_mean = probs_stack.mean(dim=0).numpy()

        return np.stack([1.0 - prob_pos_mean, prob_pos_mean], axis=1)
