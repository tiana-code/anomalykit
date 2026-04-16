"""Isolation Forest anomaly detection for multivariate data."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class IsolationForestResult:
    """
    Attributes:
        batch_normalized_score: Z-score normalized, clipped to [-1, 1].
            NOT comparable across different datasets or runs.
        permutation_feature_impact: NOT Shapley attribution — measures
            permutation sensitivity on decision_function.
    """
    anomaly_mask: np.ndarray
    batch_normalized_score: np.ndarray
    decision_scores: np.ndarray
    permutation_feature_impact: dict[str, np.ndarray] | None = None

    def __getattr__(self, name: str):
        _deprecated = {
            "scores": "batch_normalized_score",
            "feature_contributions": "permutation_feature_impact",
        }
        if name in _deprecated:
            warnings.warn(
                f"{name} is deprecated, use {_deprecated[name]}",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self, _deprecated[name])
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


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
        random_state: int = 42,
        scale_data: bool = True,
    ):
        """
        contamination: Sets the decision threshold, not a ground-truth rate.
            Actual anomaly rate in production may differ significantly.
        scale_data: IF doesn't require scaling, but helps when features have
            very different magnitudes.
        """
        if not (0 < contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")

        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.scale_data = scale_data

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._is_fitted = False

    def fit(self, data: pd.DataFrame) -> "IsolationForestDetector":
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found in input data")

        if not np.all(np.isfinite(numeric_data.values)):
            raise ValueError("Input data contains NaN or infinite values")

        self._feature_names = list(numeric_data.columns)

        X = self.scaler.fit_transform(numeric_data) if self.scale_data else numeric_data.values

        self.model.fit(X)
        self._is_fitted = True

        return self

    def detect(self, data: pd.DataFrame) -> IsolationForestResult:
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        missing = [f for f in self._feature_names if f not in data.columns]
        if missing:
            raise ValueError(f"Input data is missing columns: {missing}")

        numeric_data = data[self._feature_names]
        X = self.scaler.transform(numeric_data) if self.scale_data else numeric_data.values

        predictions = self.model.predict(X)
        anomaly_mask = predictions == -1

        decision_scores = self.model.decision_function(X)

        scores = -decision_scores
        scores = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -1, 1)

        permutation_impact = self._estimate_permutation_impact(X, decision_scores)

        return IsolationForestResult(
            anomaly_mask=anomaly_mask,
            batch_normalized_score=scores,
            decision_scores=decision_scores,
            permutation_feature_impact=permutation_impact,
        )

    def score_samples(self, data: pd.DataFrame) -> np.ndarray:
        """Raw sklearn score_samples passthrough.

        Lower = more anomalous. Unlike batch_normalized_score, these are
        comparable across calls on data from the same distribution.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        missing = [f for f in self._feature_names if f not in data.columns]
        if missing:
            raise ValueError(f"Input data is missing columns: {missing}")

        numeric_data = data[self._feature_names]
        X = self.scaler.transform(numeric_data) if self.scale_data else numeric_data.values
        return self.model.score_samples(X)

    def _estimate_permutation_impact(
        self,
        X: np.ndarray,
        scores: np.ndarray
    ) -> dict[str, np.ndarray]:
        if len(self._feature_names) == 0:
            return {}

        rng = np.random.RandomState(self.random_state)
        impacts = {}
        base_scores = scores.copy()

        for i, feature in enumerate(self._feature_names):
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, i])

            permuted_scores = self.model.decision_function(X_permuted)
            impacts[feature] = np.abs(base_scores - permuted_scores)

        return impacts

    def estimate_feature_sensitivity(self) -> dict[str, float]:
        """Estimate overall feature sensitivity via permutation on synthetic data.

        Generates standard-normal reference data and measures how much
        permuting each feature changes the decision function. Results
        are normalized to sum to 1.0.

        Note: this runs on synthetic Gaussian data, not on the training
        distribution. Use _estimate_permutation_impact for data-specific impacts.
        """
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

    def get_feature_importance(self) -> dict[str, float]:
        """Deprecated: use estimate_feature_sensitivity()."""
        warnings.warn(
            "get_feature_importance() is deprecated, use estimate_feature_sensitivity()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.estimate_feature_sensitivity()

    def save(self, path: Path) -> None:
        import joblib
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self._feature_names,
            "params": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "max_samples": self.max_samples,
                "random_state": self.random_state,
                "scale_data": self.scale_data,
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
