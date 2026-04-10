"""Context-Aware Anomaly Detector.

Detects anomalies that account for asset load, speed, and environmental context.
Normal operating ranges shift depending on the operating mode, so a reading
that is anomalous in ECO mode may be perfectly normal at FULL_SPEED.

Uses Isolation Forest / Local Outlier Factor conditioned on operating mode
and builds context vectors: [speed, draft, RPM, wind_speed, wave_height].
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OperatingMode(str, Enum):
    MANEUVERING = "MANEUVERING"
    SLOW_STEAM = "SLOW_STEAM"
    ECO = "ECO"
    FULL_SPEED = "FULL_SPEED"
    AT_ANCHOR = "AT_ANCHOR"


_MODE_RPM_RANGES: Dict[OperatingMode, Tuple[float, float]] = {
    OperatingMode.AT_ANCHOR:    (0, 15),
    OperatingMode.MANEUVERING:  (15, 40),
    OperatingMode.SLOW_STEAM:   (40, 55),
    OperatingMode.ECO:          (55, 75),
    OperatingMode.FULL_SPEED:   (75, 130),
}

CONTEXT_FEATURES = ["nav_sog", "me_rpm", "me_draft_fore", "me_draft_aft", "wind_speed", "wave_height"]

CONTEXT_TAG_CODES = ["NAV_SOG", "ME_RPM", "ME_DRAFT_FORE", "ME_DRAFT_AFT", "WIND_SPEED", "WAVE_HEIGHT"]


@dataclass
class ContextualAnomalyPoint:
    """Single contextual anomaly."""
    index: int
    timestamp: Optional[str]
    tag_id: str
    value: float
    expected_range_low: float
    expected_range_high: float
    operating_mode: str
    anomaly_score: float
    severity: str


@dataclass
class ContextualDetectionResult:
    """Full contextual detection result."""
    asset_id: str
    operating_mode_distribution: Dict[str, int]
    total_points: int
    anomaly_count: int
    anomaly_rate: float
    anomalies: List[ContextualAnomalyPoint]
    processing_time_ms: float


class ContextualDetector:
    """Anomaly detection conditioned on operating mode and context.

    For each operating mode, fits a separate model so that the definition
    of "normal" adapts to the current asset state.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        method: str = "isolation_forest",
        min_samples_per_mode: int = 20,
    ):
        if not (0 < contamination < 0.5):
            raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")
        if min_samples_per_mode < 2:
            raise ValueError(f"min_samples_per_mode must be >= 2, got {min_samples_per_mode}")

        self.contamination = contamination
        self.method = method
        self.min_samples_per_mode = min_samples_per_mode

        self._models: Dict[str, object] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._baselines: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def fit(self, data: pd.DataFrame, sensor_columns: List[str]) -> None:
        """Fit per-mode models on historical data.

        Args:
            data: DataFrame with sensor columns + optional context columns.
            sensor_columns: Columns to treat as sensor readings.
        """
        modes = self._assign_modes(data)
        data = data.copy()
        data["_mode"] = modes

        for mode in OperatingMode:
            subset = data[data["_mode"] == mode.value]
            if len(subset) < self.min_samples_per_mode:
                continue

            available = [c for c in sensor_columns if c in subset.columns]
            if not available:
                continue

            numeric = subset[available].apply(pd.to_numeric, errors="coerce").ffill().fillna(0)

            scaler = StandardScaler()
            X = scaler.fit_transform(numeric)
            self._scalers[mode.value] = scaler

            if self.method == "lof":
                n_neighbors = min(20, len(X) - 1)
                model = LocalOutlierFactor(n_neighbors=max(2, n_neighbors), contamination=self.contamination, novelty=True)
            else:
                model = IsolationForest(contamination=self.contamination, random_state=42, n_estimators=100)

            model.fit(X)
            self._models[mode.value] = model

            baselines: Dict[str, Tuple[float, float]] = {}
            for col in available:
                vals = pd.to_numeric(subset[col], errors="coerce").dropna()
                if len(vals) > 2:
                    low_bound = float(vals.quantile(0.05))
                    high_bound = float(vals.quantile(0.95))
                    baselines[col] = (low_bound, high_bound)
            self._baselines[mode.value] = baselines

    def detect(
        self,
        data: pd.DataFrame,
        sensor_columns: List[str],
        asset_id: str = "",
    ) -> ContextualDetectionResult:
        """Detect contextual anomalies in current data.

        Args:
            data: Current sensor DataFrame.
            sensor_columns: Sensor columns to analyze.
            asset_id: Asset identifier.

        Returns:
            ContextualDetectionResult with anomalies.
        """
        import time
        t0 = time.time()

        modes = self._assign_modes(data)
        data = data.copy()
        data["_mode"] = modes

        available = [c for c in sensor_columns if c in data.columns]
        if not available:
            return self._empty_result(asset_id, t0)

        mode_counts: Dict[str, int] = {}
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1

        anomalies: List[ContextualAnomalyPoint] = []

        for mode in OperatingMode:
            mask = data["_mode"] == mode.value
            subset = data[mask]
            if len(subset) == 0:
                continue

            numeric = subset[available].apply(pd.to_numeric, errors="coerce").fillna(0)

            if mode.value in self._models and mode.value in self._scalers:
                scaler = self._scalers[mode.value]
                model = self._models[mode.value]
                X = scaler.transform(numeric)

                predictions = model.predict(X)
                if hasattr(model, "decision_function"):
                    scores_raw = model.decision_function(X)
                else:
                    scores_raw = np.zeros(len(X))

                if scores_raw.max() != scores_raw.min():
                    scores = 1.0 - (scores_raw - scores_raw.min()) / (scores_raw.max() - scores_raw.min())
                else:
                    scores = np.zeros(len(scores_raw))

                outlier_indices = np.where(predictions == -1)[0]
            else:
                outlier_indices, scores = self._fallback_zscore(numeric)

            baselines = self._baselines.get(mode.value, {})

            for local_idx in outlier_indices[:50]:
                row = subset.iloc[local_idx]
                global_idx = subset.index[local_idx]
                score = float(scores[local_idx]) if local_idx < len(scores) else 0.5

                ts = None
                for ts_col in ("timestamp", "bucket"):
                    if ts_col in row.index and row[ts_col] is not None:
                        ts = str(row[ts_col])
                        break

                for col in available:
                    val = float(row[col]) if not pd.isna(row[col]) else 0.0
                    bl = baselines.get(col, (val - 1, val + 1))
                    if bl[0] <= val <= bl[1]:
                        continue

                    severity = self._score_to_severity(score)
                    anomalies.append(ContextualAnomalyPoint(
                        index=int(global_idx),
                        timestamp=ts,
                        tag_id=col,
                        value=val,
                        expected_range_low=bl[0],
                        expected_range_high=bl[1],
                        operating_mode=mode.value,
                        anomaly_score=round(score, 4),
                        severity=severity,
                    ))

        anomalies.sort(key=lambda a: a.anomaly_score, reverse=True)
        anomalies = anomalies[:200]

        unique_anomalous_rows = len({a.index for a in anomalies})
        total_rows = max(1, len(data))
        anomaly_rate = min(1.0, unique_anomalous_rows / total_rows)

        elapsed = (time.time() - t0) * 1000
        return ContextualDetectionResult(
            asset_id=asset_id,
            operating_mode_distribution=mode_counts,
            total_points=len(data),
            anomaly_count=len(anomalies),
            anomaly_rate=anomaly_rate,
            anomalies=anomalies,
            processing_time_ms=elapsed,
        )

    def _assign_modes(self, data: pd.DataFrame) -> List[str]:
        """Assign operating mode to each row based on RPM or speed heuristic."""
        modes: List[str] = []
        rpm_col = None
        for candidate in ("me_rpm", "ME_RPM", "rpm"):
            if candidate in data.columns:
                rpm_col = candidate
                break

        sog_col = None
        for candidate in ("nav_sog", "NAV_SOG", "sog", "speed"):
            if candidate in data.columns:
                sog_col = candidate
                break

        for _, row in data.iterrows():
            rpm = None
            if rpm_col and not pd.isna(row.get(rpm_col)):
                rpm = float(row[rpm_col])

            if rpm is not None:
                mode = self._rpm_to_mode(rpm)
            elif sog_col and not pd.isna(row.get(sog_col)):
                sog = float(row[sog_col])
                mode = self._sog_to_mode(sog)
            else:
                mode = OperatingMode.ECO.value
            modes.append(mode)

        return modes

    @staticmethod
    def _rpm_to_mode(rpm: float) -> str:
        for mode, (lo, hi) in _MODE_RPM_RANGES.items():
            if lo <= rpm < hi:
                return mode.value
        return OperatingMode.FULL_SPEED.value if rpm >= 75 else OperatingMode.AT_ANCHOR.value

    @staticmethod
    def _sog_to_mode(sog: float) -> str:
        if sog < 1:
            return OperatingMode.AT_ANCHOR.value
        if sog < 5:
            return OperatingMode.MANEUVERING.value
        if sog < 10:
            return OperatingMode.SLOW_STEAM.value
        if sog < 16:
            return OperatingMode.ECO.value
        return OperatingMode.FULL_SPEED.value

    @staticmethod
    def _fallback_zscore(numeric: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Simple Z-score fallback when no trained model is available."""
        means = numeric.mean()
        stds = numeric.std().replace(0, 1)
        z = ((numeric - means) / stds).abs()
        row_max_z = z.max(axis=1)
        outlier_mask = row_max_z > 3.0
        scores = np.clip(row_max_z.values / 5.0, 0, 1)
        return np.where(outlier_mask)[0], scores

    @staticmethod
    def _score_to_severity(score: float) -> str:
        if score >= 0.85:
            return "critical"
        if score >= 0.65:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def _empty_result(self, asset_id: str, t0: float) -> ContextualDetectionResult:
        import time
        return ContextualDetectionResult(
            asset_id=asset_id,
            operating_mode_distribution={},
            total_points=0,
            anomaly_count=0,
            anomaly_rate=0.0,
            anomalies=[],
            processing_time_ms=(time.time() - t0) * 1000,
        )
