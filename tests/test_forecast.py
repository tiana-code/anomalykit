"""Tests for forecast models."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from anomalykit.forecast.prophet_model import ProphetForecaster, ProphetResult


class TestProphetForecaster:
    def _make_data(self, n=100, freq_hours=1):
        dates = [datetime(2025, 1, 1) + timedelta(hours=i * freq_hours) for i in range(n)]
        values = np.sin(np.linspace(0, 8 * np.pi, n)) * 10 + 50 + np.random.randn(n) * 2
        return pd.DataFrame({"ds": dates, "y": values})

    def test_fit_and_predict_fallback(self):
        data = self._make_data(n=50)
        forecaster = ProphetForecaster()
        forecaster.fit(data)

        result = forecaster.predict(periods=24, freq="H")
        assert isinstance(result, ProphetResult)
        assert len(result.forecast) == 24
        assert "yhat" in result.forecast.columns
        assert "yhat_lower" in result.forecast.columns
        assert "yhat_upper" in result.forecast.columns

    def test_predict_before_fit_raises(self):
        forecaster = ProphetForecaster()
        with pytest.raises(ValueError, match="not fitted"):
            forecaster.predict(periods=10)

    def test_too_few_points_raises(self):
        data = pd.DataFrame({
            "ds": [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(5)],
            "y": [1, 2, 3, 4, 5],
        })
        forecaster = ProphetForecaster()
        with pytest.raises(ValueError, match="at least 10"):
            forecaster.fit(data)

    def test_trend_detection(self):
        n = 100
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n)]
        values = np.linspace(10, 50, n) + np.random.randn(n) * 0.5
        data = pd.DataFrame({"ds": dates, "y": values})

        forecaster = ProphetForecaster()
        forecaster.fit(data)
        result = forecaster.predict(periods=10, freq="H")

        assert result.trend_slope > 0
        assert result.trend == "increasing"

    def test_column_renaming(self):
        n = 30
        data = pd.DataFrame({
            "timestamp": [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n)],
            "value": np.random.randn(n) + 10,
        })
        forecaster = ProphetForecaster()
        forecaster.fit(data)
        result = forecaster.predict(periods=5, freq="H")
        assert len(result.forecast) == 5

    def test_save_and_load(self, tmp_path):
        data = self._make_data(n=30)
        forecaster = ProphetForecaster()
        forecaster.fit(data)

        path = tmp_path / "prophet.pkl"
        forecaster.save(path)

        loaded = ProphetForecaster.load(path)
        result = loaded.predict(periods=5, freq="H")
        assert len(result.forecast) == 5

    def test_model_backend_fallback(self):
        data = self._make_data(n=30)
        forecaster = ProphetForecaster()
        forecaster.fit(data)
        assert forecaster.model_backend in ("prophet", "fallback_linear")

    def test_fallback_metrics_are_computed(self):
        n = 100
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n)]
        values = np.linspace(10, 50, n) + np.random.randn(n) * 0.5
        data = pd.DataFrame({"ds": dates, "y": values})
        forecaster = ProphetForecaster()
        forecaster.fit(data)
        result = forecaster.predict(periods=10, freq="H")
        assert result.in_sample_metrics["mae"] >= 0
        assert result.in_sample_metrics["rmse"] >= 0
        assert result.in_sample_metrics["mae"] != result.in_sample_metrics["rmse"]
