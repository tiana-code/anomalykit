"""Prophet-based time series forecasting."""

import logging
import pickle
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


@dataclass
class ProphetResult:
    forecast: pd.DataFrame
    trend: str
    trend_slope: float
    seasonality_components: list[str]
    in_sample_metrics: dict[str, float]


class ProphetForecaster:
    """Prophet-based time series forecasting.

    Falls back to a simple linear trend + daily seasonality model when
    Prophet is not installed. Check ``model_backend`` to see which engine
    is active.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        interval_width: float = 0.95
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width

        self._model: Any = None
        self._is_fitted = False
        self._train_data: pd.DataFrame | None = None

    @property
    def model_backend(self) -> str:
        """Returns "prophet" or "fallback_linear"."""
        if PROPHET_AVAILABLE and self._model is not None:
            return "prophet"
        return "fallback_linear"

    def fit(self, data: pd.DataFrame) -> "ProphetForecaster":
        """Accepts 'ds'/'y' columns or 'timestamp'/'value' (auto-renamed)."""
        df = self._prepare_data(data)

        if len(df) < 10:
            raise ValueError("Need at least 10 data points for Prophet fitting")

        self._train_data = df

        if PROPHET_AVAILABLE:
            self._model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                interval_width=self.interval_width
            )
            self._model.fit(df)

        self._is_fitted = True
        return self

    def predict(
        self,
        periods: int,
        freq: str = "H"
    ) -> ProphetResult:
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if PROPHET_AVAILABLE and self._model is not None:
            future = self._model.make_future_dataframe(periods=periods, freq=freq)
            forecast = self._model.predict(future)

            trend_slope = self._calculate_trend_slope(forecast)
            trend = self._classify_trend(trend_slope)

            seasonality = self._get_enabled_seasonalities()

            metrics = self._calculate_metrics()

            return ProphetResult(
                forecast=forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods),
                trend=trend,
                trend_slope=trend_slope,
                seasonality_components=seasonality,
                in_sample_metrics=metrics
            )
        else:
            logger.warning(
                "Prophet not available; using fallback linear model. "
                "Install prophet for full forecasting capabilities."
            )
            return self._simple_forecast(periods, freq)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "ds"})
        if "value" in df.columns:
            df = df.rename(columns={"value": "y"})

        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])

        df = df.dropna(subset=["ds", "y"])

        return df[["ds", "y"]]

    def _calculate_trend_slope(self, forecast: pd.DataFrame) -> float:
        if "trend" in forecast.columns and len(forecast) > 1:
            trend = forecast["trend"].values
            times = np.arange(len(trend))
            slope, _ = np.polyfit(times, trend, 1)
            return float(slope)
        return 0.0

    def _classify_trend(self, slope: float) -> str:
        if abs(slope) < 0.01:
            return "flat"
        return "increasing" if slope > 0 else "decreasing"

    def _get_enabled_seasonalities(self) -> list[str]:
        components = []
        if self.yearly_seasonality:
            components.append("yearly")
        if self.weekly_seasonality:
            components.append("weekly")
        if self.daily_seasonality:
            components.append("daily")
        return components

    def _calculate_metrics(self) -> dict[str, float]:
        if PROPHET_AVAILABLE and self._model is not None and self._train_data is not None:
            in_sample = self._model.predict(self._train_data)
            actual = self._train_data["y"].values
            predicted = in_sample["yhat"].values[:len(actual)]

            errors = actual - predicted
            abs_errors = np.abs(errors)

            mae = float(np.mean(abs_errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            nonzero = np.abs(actual) > 1e-8
            mape = float(np.mean(abs_errors[nonzero] / np.abs(actual[nonzero])) * 100) if nonzero.any() else 0.0

            return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    def _simple_forecast(self, periods: int, freq: str) -> ProphetResult:
        if self._train_data is None:
            raise ValueError("No training data available")

        df = self._train_data

        y = df["y"].values
        n = len(y)

        x = np.arange(n)
        slope, intercept = np.polyfit(x, y, 1)

        last_date = df["ds"].max()
        delta = timedelta(hours=1) if freq == "H" else timedelta(days=1)

        future_dates = [last_date + delta * (i + 1) for i in range(periods)]

        future_x = np.arange(n, n + periods)
        predictions = slope * future_x + intercept

        if n >= 24 and freq == "H":
            daily_pattern = np.array([y[i::24].mean() for i in range(24)])
            daily_pattern = daily_pattern - daily_pattern.mean()
            for i in range(periods):
                hour = future_dates[i].hour
                predictions[i] += daily_pattern[hour]

        linear_fitted = slope * x + intercept
        residuals = y - linear_fitted
        abs_residuals = np.abs(residuals)

        mae = float(np.mean(abs_residuals))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        nonzero = np.abs(y) > 1e-8
        mape = float(np.mean(abs_residuals[nonzero] / np.abs(y[nonzero])) * 100) if nonzero.any() else 0.0

        std = np.std(residuals)
        lower = predictions - 1.96 * std
        upper = predictions + 1.96 * std

        forecast_df = pd.DataFrame({
            "ds": future_dates,
            "yhat": predictions,
            "yhat_lower": lower,
            "yhat_upper": upper
        })

        return ProphetResult(
            forecast=forecast_df,
            trend=self._classify_trend(slope),
            trend_slope=float(slope),
            seasonality_components=["daily"] if n >= 24 else [],
            in_sample_metrics={
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4),
            },
        )

    def cross_validate(
        self,
        initial: str = "365 days",
        period: str = "30 days",
        horizon: str = "30 days"
    ) -> dict[str, float] | None:
        """Perform cross-validation. Returns None if Prophet is unavailable."""
        if not PROPHET_AVAILABLE or self._model is None:
            return None

        from prophet.diagnostics import cross_validation, performance_metrics

        df_cv = cross_validation(
            self._model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        metrics = performance_metrics(df_cv)

        return {
            "mae": float(metrics["mae"].mean()),
            "rmse": float(metrics["rmse"].mean()),
            "mape": float(metrics["mape"].mean()) if "mape" in metrics else 0.0
        }

    def get_components(self) -> pd.DataFrame | None:
        """Get forecast components (trend, seasonality)."""
        if PROPHET_AVAILABLE and self._model is not None:
            future = self._model.make_future_dataframe(periods=0)
            forecast = self._model.predict(future)
            return forecast
        return None

    def save(self, path: Path) -> None:
        """Save model to disk."""
        model_data = {
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "interval_width": self.interval_width,
            "train_data": self._train_data,
            "prophet_model": self._model if PROPHET_AVAILABLE else None
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: Path) -> "ProphetForecaster":
        """Load model from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        forecaster = cls(
            yearly_seasonality=model_data["yearly_seasonality"],
            weekly_seasonality=model_data["weekly_seasonality"],
            daily_seasonality=model_data["daily_seasonality"],
            changepoint_prior_scale=model_data["changepoint_prior_scale"],
            seasonality_prior_scale=model_data["seasonality_prior_scale"],
            interval_width=model_data["interval_width"]
        )
        forecaster._train_data = model_data["train_data"]
        forecaster._model = model_data.get("prophet_model")
        forecaster._is_fitted = True

        return forecaster
