"""Adaptive Threshold Engine.

Per-asset baselines that evolve over time using Exponential Moving Average (EMA).
Threshold = baseline +/- k * adaptive absolute deviation (k configurable, default 3).

Baselines are recalculated periodically and can be persisted to DB or Redis.
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TagBaseline:
    """Baseline statistics for a single tag on a single asset.

    ema_abs_dev is an exponentially weighted mean absolute deviation,
    not a standard deviation. Thresholds are mean +/- k * ema_abs_dev.
    """
    tag_code: str
    asset_id: str
    ema_mean: float
    ema_abs_dev: float
    threshold_low: float
    threshold_high: float
    k_factor: float
    sample_count: int
    last_updated: str

    def __getattr__(self, name: str):
        _deprecated = {"ema_std": "ema_abs_dev"}
        if name in _deprecated:
            warnings.warn(
                f"{name} is deprecated, use {_deprecated[name]}",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self, _deprecated[name])
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


@dataclass
class ThresholdViolation:
    """A point where a value exceeds adaptive thresholds.

    deviation_mad_multiple is the number of mean-absolute-deviation units
    the value deviates from the EMA mean. This is NOT a sigma/z-score.
    """
    index: int
    timestamp: Optional[str]
    tag_code: str
    value: float
    threshold_low: float
    threshold_high: float
    deviation_mad_multiple: float
    severity: str

    def __getattr__(self, name: str):
        _deprecated = {"deviation_sigma": "deviation_mad_multiple"}
        if name in _deprecated:
            warnings.warn(
                f"{name} is deprecated, use {_deprecated[name]}",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self, _deprecated[name])
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


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
    absolute deviation for each tag.  New data shifts the baseline smoothly,
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
            k_factor: Number of adaptive absolute deviations for threshold.
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

        violations.sort(key=lambda v: abs(v.deviation_mad_multiple), reverse=True)
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
            bl.threshold_low = bl.ema_mean - k * bl.ema_abs_dev
            bl.threshold_high = bl.ema_mean + k * bl.ema_abs_dev

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

            if len(values) >= self.min_history:
                warmup = values[:self.min_history]
                warmup_mean = float(np.mean(warmup))
                initial_abs_dev = float(np.mean(np.abs(warmup - warmup_mean)))
                initial_abs_dev = max(initial_abs_dev, 1e-6)
            else:
                warmup_mean = first_value
                initial_abs_dev = abs(first_value) * 0.01 if first_value != 0 else 1.0

            bl = TagBaseline(
                tag_code=tag_code,
                asset_id=asset_id,
                ema_mean=warmup_mean,
                ema_abs_dev=initial_abs_dev,
                threshold_low=warmup_mean - self.k_factor * initial_abs_dev,
                threshold_high=warmup_mean + self.k_factor * initial_abs_dev,
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
                    dev_multiple = (val - bl.ema_mean) / bl.ema_abs_dev if bl.ema_abs_dev > 0 else 0
                    severity = self._deviation_to_severity(abs(dev_multiple))

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
                        deviation_mad_multiple=round(dev_multiple, 3),
                        severity=severity,
                    ))

            bl.ema_mean = self.ema_alpha * val + (1 - self.ema_alpha) * bl.ema_mean
            deviation = abs(val - bl.ema_mean)
            bl.ema_abs_dev = self.ema_alpha * deviation + (1 - self.ema_alpha) * bl.ema_abs_dev
            bl.ema_abs_dev = max(bl.ema_abs_dev, 1e-6)
            bl.threshold_low = bl.ema_mean - self.k_factor * bl.ema_abs_dev
            bl.threshold_high = bl.ema_mean + self.k_factor * bl.ema_abs_dev
            bl.sample_count += 1

        bl.last_updated = datetime.now(timezone.utc).isoformat()
        bl.k_factor = self.k_factor
        self._baselines[key] = bl

        return bl, violations

    @staticmethod
    def _deviation_to_severity(deviation: float) -> str:
        if deviation >= 5.0:
            return "critical"
        if deviation >= 4.0:
            return "high"
        if deviation >= 3.0:
            return "medium"
        return "low"
