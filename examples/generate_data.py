"""Synthetic data generators for anomalykit examples.

Each generator produces realistic synthetic data on the fly - no CSV files stored.
All generators use deterministic seeds for reproducibility.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def generate_sensor_data(
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-sensor time series with injected anomalies.

    Produces four correlated sensor channels (temperature, pressure,
    vibration, flow_rate) with three types of injected anomalies:
    spikes, drift, and stuck-sensor episodes.

    Args:
        n: Number of time steps.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, temperature, pressure,
        vibration, flow_rate, anomaly_type.
    """
    rng = np.random.RandomState(seed)

    timestamps = pd.date_range(
        start="2025-01-01", periods=n, freq="h"
    )

    # Base signals with realistic cross-correlations
    base = rng.randn(n)
    temperature = 80.0 + 5.0 * base + 2.0 * rng.randn(n)
    pressure = 4.5 + 0.8 * base + 0.3 * rng.randn(n)
    vibration = 2.0 + 0.5 * np.abs(base) + 0.4 * rng.randn(n)
    flow_rate = 120.0 + 10.0 * base + 3.0 * rng.randn(n)

    anomaly_type = np.array(["normal"] * n, dtype=object)

    # --- Inject spikes (sudden jumps) ---
    n_spikes = max(1, n // 50)
    spike_indices = rng.choice(n, size=n_spikes, replace=False)
    for idx in spike_indices:
        channel = rng.choice(["temperature", "pressure", "vibration", "flow_rate"])
        magnitude = rng.uniform(4, 8)
        sign = rng.choice([-1, 1])
        if channel == "temperature":
            temperature[idx] += sign * magnitude * 5
        elif channel == "pressure":
            pressure[idx] += sign * magnitude * 0.8
        elif channel == "vibration":
            vibration[idx] += magnitude * 2
        else:
            flow_rate[idx] += sign * magnitude * 10
        anomaly_type[idx] = "spike"

    # --- Inject drift (gradual shift) ---
    drift_start = n // 3
    drift_end = drift_start + n // 10
    drift_ramp = np.linspace(0, 1, drift_end - drift_start)
    temperature[drift_start:drift_end] += drift_ramp * 15
    pressure[drift_start:drift_end] += drift_ramp * 2.0
    anomaly_type[drift_start:drift_end] = "drift"

    # --- Inject stuck sensor (constant value) ---
    stuck_start = 2 * n // 3
    stuck_end = stuck_start + n // 20
    vibration[stuck_start:stuck_end] = vibration[stuck_start]
    anomaly_type[stuck_start:stuck_end] = "stuck"

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.round(temperature, 2),
        "pressure": np.round(pressure, 3),
        "vibration": np.round(vibration, 3),
        "flow_rate": np.round(flow_rate, 2),
        "anomaly_type": anomaly_type,
    })


def generate_failure_data(
    n: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate time-to-failure data with censoring for Weibull analysis.

    Simulates component lifetimes drawn from a Weibull distribution
    (shape=2.5, scale=1500 hours).  Approximately 30 % of observations
    are right-censored (component still running at observation time).

    Args:
        n: Number of components.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: component_id, hours, failed (bool),
        failure_mode.
    """
    rng = np.random.RandomState(seed)

    # True Weibull parameters
    true_shape = 2.5
    true_scale = 1500.0

    # Generate failure times
    failure_times = true_scale * rng.weibull(true_shape, size=n)

    # Introduce censoring (~30%)
    censoring_times = rng.uniform(500, 2500, size=n)
    observed_times = np.minimum(failure_times, censoring_times)
    failed = failure_times <= censoring_times

    # Assign failure modes for failed components
    modes = ["wear", "fatigue", "corrosion", "overload"]
    failure_mode = np.where(
        failed,
        rng.choice(modes, size=n),
        "censored",
    )

    return pd.DataFrame({
        "component_id": [f"CMP-{i:04d}" for i in range(n)],
        "hours": np.round(observed_times, 1),
        "failed": failed,
        "failure_mode": failure_mode,
    })


def generate_fleet_kpis(
    n: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-criteria fleet performance KPIs for TOPSIS ranking.

    Each row represents an asset with six KPI dimensions:
    fuel_efficiency (benefit), safety_score (benefit),
    maintenance_cost (cost), uptime_pct (benefit),
    emissions_index (cost), crew_satisfaction (benefit).

    Args:
        n: Number of assets.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: asset_id, asset_name, and 6 KPI columns.
    """
    rng = np.random.RandomState(seed)

    asset_ids = [f"ASSET-{i:03d}" for i in range(n)]
    asset_names = [f"Vessel {chr(65 + i % 26)}{i // 26}" for i in range(n)]

    return pd.DataFrame({
        "asset_id": asset_ids,
        "asset_name": asset_names,
        "fuel_efficiency": np.round(rng.uniform(0.5, 0.95, n), 3),
        "safety_score": np.round(rng.uniform(0.6, 1.0, n), 3),
        "maintenance_cost": np.round(rng.uniform(10_000, 80_000, n), 0),
        "uptime_pct": np.round(rng.uniform(85, 99.9, n), 1),
        "emissions_index": np.round(rng.uniform(0.3, 1.2, n), 3),
        "crew_satisfaction": np.round(rng.uniform(0.5, 0.98, n), 3),
    })


def generate_forecast_data(
    n: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily time series with trend and seasonality.

    The signal combines a linear upward trend, yearly seasonality,
    weekly seasonality, and Gaussian noise.  Suitable for testing
    ProphetForecaster.

    Args:
        n: Number of days.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: ds (datetime), y (float).
    """
    rng = np.random.RandomState(seed)

    dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)

    # Trend
    trend = 100.0 + 0.15 * t

    # Yearly seasonality (amplitude ~20)
    yearly = 20.0 * np.sin(2 * np.pi * t / 365.25)

    # Weekly seasonality (amplitude ~5)
    weekly = 5.0 * np.sin(2 * np.pi * t / 7.0)

    # Noise
    noise = rng.normal(0, 3.0, size=n)

    y = trend + yearly + weekly + noise

    return pd.DataFrame({"ds": dates, "y": np.round(y, 2)})
