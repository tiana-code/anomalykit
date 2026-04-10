"""Isolation Forest anomaly detection for multivariate data."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class IsolationForestResult:
    """Result from Isolation Forest detection."""
    anomaly_mask: np.ndarray
    scores: np.ndarray
    decision_scores: np.ndarray
    feature_contributions: Optional[Dict[str, np.ndarray]] = None


class IsolationForestDetector:
    """Isolation Forest based anomaly detection.

    Works by isolating observations using random feature splits.
    Anomalies require fewer splits to isolate.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        max_samples: str = "auto",
        random_state: int = 42
    ):
        if not (0 < contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")

        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self._feature_names: List[str] = []
        self._is_fitted = False

    def fit(self, data: pd.DataFrame) -> "IsolationForestDetector":
        """Train Isolation Forest on data.

        Args:
            data: Training DataFrame with numeric columns.
        """
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found in input data")

        if not np.all(np.isfinite(numeric_data.values)):
            raise ValueError("Input data contains NaN or infinite values")

        self._feature_names = list(numeric_data.columns)

        X = self.scaler.fit_transform(numeric_data)
        self.model.fit(X)
        self._is_fitted = True

        return self

    def detect(self, data: pd.DataFrame) -> IsolationForestResult:
        """Detect anomalies using trained Isolation Forest.

        Args:
            data: DataFrame with numeric columns to analyze.

        Returns:
            IsolationForestResult with predictions and scores.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        missing = [f for f in self._feature_names if f not in data.columns]
        if missing:
            raise ValueError(f"Input data is missing columns: {missing}")

        numeric_data = data[self._feature_names]
        X = self.scaler.transform(numeric_data)

        predictions = self.model.predict(X)
        anomaly_mask = predictions == -1

        decision_scores = self.model.decision_function(X)

        scores = -decision_scores
        scores = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -1, 1)

        feature_contributions = self._estimate_feature_contributions(X, decision_scores)

        return IsolationForestResult(
            anomaly_mask=anomaly_mask,
            scores=scores,
            decision_scores=decision_scores,
            feature_contributions=feature_contributions
        )

    def _estimate_feature_contributions(
        self,
        X: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Estimate feature contributions via permutation importance."""
        if len(self._feature_names) == 0:
            return {}

        rng = np.random.RandomState(self.random_state)
        contributions = {}
        base_scores = scores.copy()

        for i, feature in enumerate(self._feature_names):
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, i])

            permuted_scores = self.model.decision_function(X_permuted)
            contributions[feature] = np.abs(base_scores - permuted_scores)

        return contributions

    def get_feature_importance(self) -> Dict[str, float]:
        """Get overall feature importance via permutation on synthetic data."""
        if not self._is_fitted or not self._feature_names:
            return {}

        rng = np.random.RandomState(self.random_state)
        n_samples = min(1000, 100 * len(self._feature_names))
        X_synthetic = rng.randn(n_samples, len(self._feature_names))

        base_scores = self.model.decision_function(X_synthetic)
        importances = {}

        for i, feature in enumerate(self._feature_names):
            X_perm = X_synthetic.copy()
            rng.shuffle(X_perm[:, i])
            perm_scores = self.model.decision_function(X_perm)
            importances[feature] = float(np.abs(base_scores - perm_scores).mean())

        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def save(self, path: Path) -> None:
        """Save model to disk using joblib."""
        import joblib
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self._feature_names,
            "params": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "max_samples": self.max_samples,
                "random_state": self.random_state
            }
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: Path) -> "IsolationForestDetector":
        """Load model from disk."""
        import joblib
        model_data = joblib.load(path)

        detector = cls(**model_data["params"])
        detector.model = model_data["model"]
        detector.scaler = model_data["scaler"]
        detector._feature_names = model_data["feature_names"]
        detector._is_fitted = True

        return detector
