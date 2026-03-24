"""
probe.py — Hallucination probe classifier (student-implemented).

Students: Implement ``HallucinationProbe`` to classify feature vectors as
truthful (0) or hallucinated (1).  The skeleton provides a logistic regression
baseline — you are expected to extend it.

``validate.py`` calls the probe as follows::

    probe = HallucinationProbe()
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_val)
    y_prob = probe.predict_proba(X_val)   # for AUROC

All three methods must be implemented and their signatures must not change.
``X`` is a 2-D NumPy array of shape ``(n_samples, feature_dim)``; ``y`` is
a 1-D NumPy array of ints (0 or 1).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class HallucinationProbe:
    """Binary classifier that detects hallucinations from hidden-state features.

    The skeleton wraps scikit-learn's ``LogisticRegression`` in a pipeline with
    ``StandardScaler`` pre-processing — a standard and competitive baseline.

    Student task:
        Replace or extend the skeleton below.  Ideas to explore:

        **Classifier:**
          - ``MLPClassifier(hidden_layer_sizes=(256,), max_iter=500)``
          - ``RandomForestClassifier(n_estimators=200)``
          - ``SVC(kernel="rbf", probability=True)`` — supports ``predict_proba``
          - Combine multiple classifiers with ``VotingClassifier``

        **Pre-processing:**
          - ``PCA(n_components=64)`` — dimensionality reduction
          - ``StandardScaler()`` → ``PCA()`` → classifier
          - L2-normalise features before fitting

        **Regularisation (LogisticRegression):**
          - ``C=0.01``  — strong regularisation (prevents overfitting)
          - ``C=100``   — weak regularisation (allows complex boundaries)
          - ``penalty="l1"`` with ``solver="saga"`` — sparse coefficients

        **Class imbalance:**
          - ``class_weight="balanced"`` in most scikit-learn classifiers
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the classifier pipeline below.
        # ------------------------------------------------------------------
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        solver="lbfgs",
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        return self._pipeline.predict_proba(X)
