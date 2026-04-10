"""Adaptive Threshold Engine.

Per-asset baselines that evolve over time using Exponential Moving Average (EMA).
Threshold = baseline +/- k * adaptive_std (k configurable, default 3).

Baselines are recalculated periodically and can be persisted to DB or Redis.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TagBaseline:
    """Baseline statistics for a single tag on a single asset."""
    tag_code: str
    asset_id: str
    ema_mean: float
    ema_std: float
    threshold_low: float
    threshold_high: float
    k_factor: float
    sample_count: int
    last_updated: str


@dataclass
class ThresholdViolation:
    """A point where a value exceeds adaptive thresholds."""
    index: int
    timestamp: Optional[str]
    tag_code: str
    value: float
    threshold_low: float
    threshold_high: float
    deviation_sigma: float
    severity: str


@dataclass
class AdaptiveThresholdResult:
    """Result from adaptive threshold calculation."""
    asset_id: str
    baselines: List[TagBaseline]
    violations: List[ThresholdViolation]
    total_points: int
    violation_count: int
    processing_time_ms: float


class AdaptiveThresholdEngine:
    """Per-asset, per-tag adaptive thresholds using EMA.

    The engine maintains an exponential moving average (EMA) of mean and
    standard deviation for each tag.  New data shifts the baseline smoothly,
    while sudden deviations trigger violations.
    """

    def __init__(
        self,
        k_factor: float = 3.0,
        ema_alpha: float = 0.1,
        min_history: int = 24,
    ):
        """
        Args:
            k_factor: Number of adaptive standard deviations for threshold.
            ema_alpha: Smoothing factor for EMA (0 < alpha <= 1).
                       Lower = slower adaptation.
            min_history: Minimum data points before adaptive thresholds activate.
        """
        if not (0 < ema_alpha <= 1):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if k_factor <= 0:
            raise ValueError(f"k_factor must be positive, got {k_factor}")
        if min_history < 1:
            raise ValueError(f"min_history must be >= 1, got {min_history}")

        self.k_factor = k_factor
        self.ema_alpha = ema_alpha
        self.min_history = min_history

        self._baselines: Dict[Tuple[str, str], TagBaseline] = {}

    def calculate(
        self,
        data: pd.DataFrame,
        tag_columns: List[str],
        asset_id: str,
    ) -> AdaptiveThresholdResult:
        """Calculate adaptive thresholds and detect violations.

        Args:
            data: Timeseries DataFrame sorted by time.
            tag_columns: Column names to analyze.
            asset_id: Asset identifier.

        Returns:
            AdaptiveThresholdResult with baselines and violations.
        """
        t0 = time.time()

        available = [c for c in tag_columns if c in data.columns]
        baselines: List[TagBaseline] = []
        violations: List[ThresholdViolation] = []

        for tag in available:
            series = pd.to_numeric(data[tag], errors="coerce").dropna()
            if len(series) < 3:
                continue

            bl, tag_violations = self._process_sequential(data, asset_id, tag, series)
            baselines.append(bl)
            violations.extend(tag_violations)

        violations.sort(key=lambda v: abs(v.deviation_sigma), reverse=True)
        violations = violations[:500]

        return AdaptiveThresholdResult(
            asset_id=asset_id,
            baselines=baselines,
            violations=violations,
            total_points=len(data),
            violation_count=len(violations),
            processing_time_ms=(time.time() - t0) * 1000,
        )

    def get_baselines(self, asset_id: str) -> List[TagBaseline]:
        """Return all stored baselines for an asset."""
        return [
            bl for (vid, _), bl in self._baselines.items()
            if vid == asset_id
        ]

    def set_k_factor(self, k: float) -> None:
        """Update k-factor and recalculate all thresholds."""
        self.k_factor = k
        for key, bl in self._baselines.items():
            bl.k_factor = k
            bl.threshold_low = bl.ema_mean - k * bl.ema_std
            bl.threshold_high = bl.ema_mean + k * bl.ema_std

    def _process_sequential(
        self,
        data: pd.DataFrame,
        asset_id: str,
        tag_code: str,
        series: pd.Series,
    ) -> Tuple[TagBaseline, List[ThresholdViolation]]:
        """Process points sequentially: check violation first, then update baseline.

        This avoids look-ahead bias by never using future data to set the
        threshold that a current point is checked against.
        """
        key = (asset_id, tag_code)
        values = series.values.astype(float)

        if key not in self._baselines:
            first_value = float(values[0])
            initial_std = abs(first_value) * 0.01 if first_value != 0 else 1.0

            bl = TagBaseline(
                tag_code=tag_code,
                asset_id=asset_id,
                ema_mean=first_value,
                ema_std=initial_std,
                threshold_low=first_value - self.k_factor * initial_std,
                threshold_high=first_value + self.k_factor * initial_std,
                k_factor=self.k_factor,
                sample_count=0,
                last_updated="",
            )
            self._baselines[key] = bl

        bl = self._baselines[key]
        violations: List[ThresholdViolation] = []

        for idx in series.index:
            val = float(series[idx])

            if bl.sample_count >= self.min_history:
                if val < bl.threshold_low or val > bl.threshold_high:
                    sigma = (val - bl.ema_mean) / bl.ema_std if bl.ema_std > 0 else 0
                    severity = self._sigma_to_severity(abs(sigma))

                    ts = None
                    if idx in data.index:
                        row = data.loc[idx]
                        for ts_col in ("timestamp", "bucket"):
                            if ts_col in data.columns:
                                ts_val = row.get(ts_col)
                                if ts_val is not None:
                                    ts = str(ts_val)
                                    break

                    positional_idx = series.index.get_loc(idx)
                    if isinstance(positional_idx, slice):
                        positional_idx = positional_idx.start or 0

                    violations.append(ThresholdViolation(
                        index=int(positional_idx),
                        timestamp=ts,
                        tag_code=tag_code,
                        value=val,
                        threshold_low=bl.threshold_low,
                        threshold_high=bl.threshold_high,
                        deviation_sigma=round(sigma, 3),
                        severity=severity,
                    ))

            bl.ema_mean = self.ema_alpha * val + (1 - self.ema_alpha) * bl.ema_mean
            deviation = abs(val - bl.ema_mean)
            bl.ema_std = self.ema_alpha * deviation + (1 - self.ema_alpha) * bl.ema_std
            bl.ema_std = max(bl.ema_std, 1e-6)
            bl.threshold_low = bl.ema_mean - self.k_factor * bl.ema_std
            bl.threshold_high = bl.ema_mean + self.k_factor * bl.ema_std
            bl.sample_count += 1

        bl.last_updated = datetime.now(timezone.utc).isoformat()
        bl.k_factor = self.k_factor
        self._baselines[key] = bl

        return bl, violations

    @staticmethod
    def _sigma_to_severity(sigma: float) -> str:
        if sigma >= 5.0:
            return "critical"
        if sigma >= 4.0:
            return "high"
        if sigma >= 3.0:
            return "medium"
        return "low"
