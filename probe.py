"""
probe.py — Hallucination probe classifier (student-implemented).

A regularised MLP over standard-scaled features.  Training uses an internal
stratified hold-out for early stopping and threshold tuning on **accuracy**
(the competition's primary metric).  ``fit_hyperparameters`` can override
the threshold when an external validation set is available.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Architecture: Linear(d → h) → ReLU → Dropout → Linear(h → 1) with
    ``StandardScaler`` preprocessing.  Built lazily in ``fit`` once the
    feature dimension is known.
    """

    HIDDEN_DIM = 128
    DROPOUT = 0.3
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 500
    PATIENCE = 30
    INNER_VAL_FRAC = 0.15
    RANDOM_STATE = 42

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Sequential(
            nn.Linear(input_dim, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    @staticmethod
    def _tune_threshold(
        probs: np.ndarray, y_true: np.ndarray, metric: str = "accuracy"
    ) -> float:
        candidates = np.unique(np.concatenate([probs, np.linspace(0.01, 0.99, 99)]))
        best_t, best_score = 0.5, -1.0
        for t in candidates:
            y_pred = (probs >= t).astype(int)
            if metric == "accuracy":
                score = accuracy_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)
        return best_t

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe with internal early stopping and threshold tuning."""
        y = y.astype(np.int64)
        X_scaled = self._scaler.fit_transform(X)

        n_pos_total = int(y.sum())
        n_neg_total = len(y) - n_pos_total
        can_stratify = n_pos_total >= 5 and n_neg_total >= 5 and len(y) >= 25

        if can_stratify:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_scaled,
                y,
                test_size=self.INNER_VAL_FRAC,
                random_state=self.RANDOM_STATE,
                stratify=y,
            )
        else:
            X_tr, X_es, y_tr, y_es = X_scaled, X_scaled, y, y

        self._build_network(X_scaled.shape[1])

        X_tr_t = torch.from_numpy(X_tr).float()
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
        X_es_t = torch.from_numpy(X_es).float()
        y_es_t = torch.from_numpy(y_es.astype(np.float32))

        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY
        )

        best_val_loss = float("inf")
        best_state: dict | None = None
        bad_epochs = 0

        for _ in range(self.MAX_EPOCHS):
            self.train()
            optimizer.zero_grad()
            logits = self(X_tr_t)
            loss = criterion(logits, y_tr_t)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_loss = criterion(self(X_es_t), y_es_t).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.PATIENCE:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(self(X_es_t)).numpy()
        self._threshold = self._tune_threshold(val_probs, y_es, metric="accuracy")
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on an external validation set for accuracy."""
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = self._tune_threshold(probs, y_val, metric="accuracy")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        self.eval()
        with torch.no_grad():
            prob_pos = torch.sigmoid(self(X_t)).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
