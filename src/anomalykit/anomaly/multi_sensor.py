"""Multi-Sensor Analytics Models.

Includes:
- Multi-sensor pattern detection
- Sensor fusion via PCA
- Cross-sensor correlation analysis
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PatternDetectionResult:
    """Result from multi-sensor pattern detection."""
    patterns: List[Dict]
    anomaly_mask: np.ndarray
    scores: np.ndarray
    pattern_count: int
    correlation_matrix: Optional[np.ndarray] = None


@dataclass
class SensorFusionResult:
    """Result from sensor fusion."""
    principal_component: np.ndarray
    anomaly_mask: np.ndarray
    scores: np.ndarray
    component_weights: Dict[str, float]
    reconstruction_error: np.ndarray
    explained_variance: float


@dataclass
class CrossCorrelationResult:
    """Result from cross-correlation analysis."""
    correlation_matrix: np.ndarray
    lag_matrix: np.ndarray
    anomaly_mask: np.ndarray
    scores: np.ndarray
    broken_correlations: List[Dict]
    correlation_changes: List[Dict]


class MultiSensorPatternDetector:
    """
    Multi-sensor pattern detection.

    Detects anomalous patterns across multiple correlated sensors
    using correlation analysis and subsequence matching.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        pattern_window: int = 10,
        anomaly_threshold: float = 3.0
    ):
        self.correlation_threshold = correlation_threshold
        self.pattern_window = pattern_window
        self.anomaly_threshold = anomaly_threshold
        self._baseline_patterns: List[np.ndarray] = []
        self._baseline_correlations: Optional[np.ndarray] = None
        self._scaler = StandardScaler()

    def fit(self, data: pd.DataFrame, sensor_columns: List[str]) -> None:
        """Fit on historical baseline data."""
        if len(data) < self.pattern_window * 2:
            raise ValueError(
                f"Insufficient data for pattern detection: {len(data)} rows, "
                f"need at least {self.pattern_window * 2}"
            )

        self._baseline_patterns = []
        self._baseline_correlations = None

        sensor_data = data[sensor_columns].values
        if not np.all(np.isfinite(sensor_data)):
            raise ValueError("Input data contains NaN or infinite values")

        self._scaler.fit(sensor_data)
        normalized = self._scaler.transform(sensor_data)

        self._baseline_correlations = np.corrcoef(normalized.T)

        for i in range(0, len(normalized) - self.pattern_window, self.pattern_window // 2):
            pattern = normalized[i:i + self.pattern_window]
            self._baseline_patterns.append(pattern)

    def detect(
        self,
        data: pd.DataFrame,
        sensor_columns: List[str]
    ) -> PatternDetectionResult:
        """Detect anomalous patterns in sensor data."""
        sensor_data = data[sensor_columns].values

        if len(sensor_data) < self.pattern_window:
            return PatternDetectionResult(
                patterns=[],
                anomaly_mask=np.zeros(len(data), dtype=bool),
                scores=np.zeros(len(data)),
                pattern_count=0
            )

        if not hasattr(self._scaler, 'mean_') or self._scaler.mean_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        normalized = self._scaler.transform(sensor_data)

        current_correlations = np.corrcoef(normalized.T)

        anomaly_mask = np.zeros(len(data), dtype=bool)
        scores = np.zeros(len(data))
        patterns = []

        if self._baseline_correlations is not None:
            correlation_diff = np.abs(current_correlations - self._baseline_correlations)
            max_diff = np.max(correlation_diff)

            if max_diff > (1.0 - self.correlation_threshold):
                scores += max_diff

        for i in range(0, len(normalized) - self.pattern_window + 1):
            window = normalized[i:i + self.pattern_window]

            min_distance = float('inf')
            if self._baseline_patterns:
                for baseline in self._baseline_patterns:
                    if baseline.shape == window.shape:
                        distance = np.linalg.norm(window - baseline) / np.sqrt(window.size)
                        min_distance = min(min_distance, distance)
            else:
                min_distance = np.std(window)

            if min_distance > self.anomaly_threshold:
                anomaly_mask[i:i + self.pattern_window] = True
                scores[i:i + self.pattern_window] = np.maximum(
                    scores[i:i + self.pattern_window],
                    min_distance / self.anomaly_threshold
                )
                patterns.append({
                    "start_idx": i,
                    "end_idx": i + self.pattern_window,
                    "distance": float(min_distance),
                    "type": "pattern_deviation"
                })

        if scores.max() > 0:
            scores = np.clip(scores / scores.max(), 0, 1)

        return PatternDetectionResult(
            patterns=patterns,
            anomaly_mask=anomaly_mask,
            scores=scores,
            pattern_count=len(patterns),
            correlation_matrix=current_correlations
        )


class SensorFusionDetector:
    """
    Sensor fusion for anomaly detection.

    Uses PCA-based fusion to combine multiple sensors into
    a unified representation and detect deviations.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        reconstruction_threshold: float = 3.0
    ):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.reconstruction_threshold = reconstruction_threshold
        self._pca: Optional[PCA] = None
        self._scaler = StandardScaler()
        self._baseline_reconstruction_error: Optional[float] = None

    def fit(self, data: pd.DataFrame, sensor_columns: List[str]) -> None:
        """Fit PCA model on baseline data."""
        sensor_data = data[sensor_columns].values
        if not np.all(np.isfinite(sensor_data)):
            raise ValueError("Input data contains NaN or infinite values")

        normalized = self._scaler.fit_transform(sensor_data)

        if self.n_components is None:
            temp_pca = PCA()
            temp_pca.fit(normalized)
            cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            n_components = max(1, min(n_components, len(sensor_columns) - 1))
        else:
            n_components = min(self.n_components, len(sensor_columns) - 1)

        self._pca = PCA(n_components=n_components)
        transformed = self._pca.fit_transform(normalized)

        reconstructed = self._pca.inverse_transform(transformed)
        self._baseline_reconstruction_error = np.mean(
            np.sum((normalized - reconstructed) ** 2, axis=1)
        )

    def fuse(
        self,
        data: pd.DataFrame,
        sensor_columns: List[str]
    ) -> SensorFusionResult:
        """Fuse sensors and detect anomalies."""
        sensor_data = data[sensor_columns].values

        if self._pca is None:
            raise ValueError("Model not fitted. Call fit() first.")

        normalized = self._scaler.transform(sensor_data)

        transformed = self._pca.transform(normalized)
        reconstructed = self._pca.inverse_transform(transformed)

        reconstruction_error = np.sum((normalized - reconstructed) ** 2, axis=1)

        if self._baseline_reconstruction_error is not None and self._baseline_reconstruction_error > 0:
            threshold = self._baseline_reconstruction_error * self.reconstruction_threshold
        else:
            threshold = np.percentile(reconstruction_error, 95)

        anomaly_mask = reconstruction_error > threshold

        scores = np.clip(reconstruction_error / max(threshold, 1e-6), 0, 1)

        component_weights = {}
        for i, col in enumerate(sensor_columns):
            weight = float(np.sum(self._pca.components_[:, i] ** 2))
            component_weights[col] = weight

        total_weight = sum(component_weights.values())
        if total_weight > 0:
            component_weights = {k: v / total_weight for k, v in component_weights.items()}

        return SensorFusionResult(
            principal_component=transformed[:, 0] if transformed.shape[1] > 0 else np.zeros(len(data)),
            anomaly_mask=anomaly_mask,
            scores=scores,
            component_weights=component_weights,
            reconstruction_error=reconstruction_error,
            explained_variance=float(sum(self._pca.explained_variance_ratio_))
        )


class CrossCorrelationAnalyzer:
    """
    Cross-sensor correlation analysis.

    Analyzes correlations between sensors to detect:
    - Broken correlations (sensors that should be correlated but aren't)
    - Lagged correlations
    - Correlation changes over time
    """

    def __init__(
        self,
        max_lag: int = 10,
        correlation_change_threshold: float = 0.3,
        min_correlation: float = 0.5
    ):
        self.max_lag = max_lag
        self.correlation_change_threshold = correlation_change_threshold
        self.min_correlation = min_correlation
        self._baseline_correlations: Optional[np.ndarray] = None
        self._baseline_lags: Optional[np.ndarray] = None
        self._sensor_pairs: List[Tuple[str, str]] = []

    def fit(self, data: pd.DataFrame, sensor_columns: List[str]) -> None:
        """Calculate baseline correlations."""
        sensor_data = data[sensor_columns].values
        n_sensors = len(sensor_columns)

        self._sensor_pairs = []
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                self._sensor_pairs.append((sensor_columns[i], sensor_columns[j]))

        self._baseline_correlations = np.zeros((n_sensors, n_sensors))
        self._baseline_lags = np.zeros((n_sensors, n_sensors), dtype=int)

        for i in range(n_sensors):
            for j in range(n_sensors):
                if i == j:
                    self._baseline_correlations[i, j] = 1.0
                else:
                    corr, lag = self._cross_correlate(
                        sensor_data[:, i],
                        sensor_data[:, j]
                    )
                    self._baseline_correlations[i, j] = corr
                    self._baseline_lags[i, j] = lag

    def analyze(
        self,
        data: pd.DataFrame,
        sensor_columns: List[str]
    ) -> CrossCorrelationResult:
        """Analyze cross-correlations and detect anomalies."""
        sensor_data = data[sensor_columns].values
        n_sensors = len(sensor_columns)
        n_samples = len(data)

        current_correlations = np.zeros((n_sensors, n_sensors))
        current_lags = np.zeros((n_sensors, n_sensors), dtype=int)

        for i in range(n_sensors):
            for j in range(n_sensors):
                if i == j:
                    current_correlations[i, j] = 1.0
                else:
                    corr, lag = self._cross_correlate(
                        sensor_data[:, i],
                        sensor_data[:, j]
                    )
                    current_correlations[i, j] = corr
                    current_lags[i, j] = lag

        broken_correlations = []
        correlation_changes = []

        if self._baseline_correlations is not None:
            for i in range(n_sensors):
                for j in range(i + 1, n_sensors):
                    baseline_corr = self._baseline_correlations[i, j]
                    current_corr = current_correlations[i, j]

                    if abs(baseline_corr) >= self.min_correlation:
                        corr_change = abs(current_corr - baseline_corr)

                        if corr_change > self.correlation_change_threshold:
                            broken_correlations.append({
                                "sensor1": sensor_columns[i],
                                "sensor2": sensor_columns[j],
                                "baseline_correlation": float(baseline_corr),
                                "current_correlation": float(current_corr),
                                "change": float(corr_change)
                            })

                        if abs(current_corr - baseline_corr) > 0.1:
                            correlation_changes.append({
                                "sensor1": sensor_columns[i],
                                "sensor2": sensor_columns[j],
                                "baseline": float(baseline_corr),
                                "current": float(current_corr),
                                "delta": float(current_corr - baseline_corr)
                            })

        anomaly_mask = np.zeros(n_samples, dtype=bool)
        scores = np.zeros(n_samples)

        if broken_correlations:
            total_change = sum(bc["change"] for bc in broken_correlations)
            normalized_score = min(1.0, total_change / (len(broken_correlations) * self.correlation_change_threshold))
            scores[:] = normalized_score

            if normalized_score > 0.5:
                anomaly_mask[:] = True

        return CrossCorrelationResult(
            correlation_matrix=current_correlations,
            lag_matrix=current_lags,
            anomaly_mask=anomaly_mask,
            scores=scores,
            broken_correlations=broken_correlations,
            correlation_changes=correlation_changes
        )

    def _cross_correlate(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, int]:
        """Calculate cross-correlation and optimal lag."""
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)

        correlation = correlate(x_norm, y_norm, mode='full')
        lags = np.arange(-len(x) + 1, len(x))

        valid_range = np.abs(lags) <= self.max_lag
        if not np.any(valid_range):
            return float(np.corrcoef(x, y)[0, 1]), 0

        valid_correlation = correlation[valid_range]
        valid_lags = lags[valid_range]

        best_idx = np.argmax(np.abs(valid_correlation))
        best_lag = valid_lags[best_idx]

        if best_lag > 0:
            corr = np.corrcoef(x[best_lag:], y[:-best_lag])[0, 1]
        elif best_lag < 0:
            corr = np.corrcoef(x[:best_lag], y[-best_lag:])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]

        return float(corr) if not np.isnan(corr) else 0.0, int(best_lag)
