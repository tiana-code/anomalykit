"""Weibull distribution-based RUL prediction."""

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class WeibullResult:
    """
    Attributes:
        conditional_quantile_interval: Conditional RUL quantiles under
            point-estimated Weibull parameters. NOT a confidence interval —
            does not account for parameter estimation uncertainty.
        survival_probability: P(T > current_hours).
    """
    predicted_rul: float
    conditional_quantile_interval: tuple[float, float]
    survival_probability: float
    failure_probability: float
    hazard_rate: float
    shape: float
    scale: float

    def __getattr__(self, name: str):
        _deprecated = {"confidence_interval": "conditional_quantile_interval"}
        if name in _deprecated:
            warnings.warn(
                f"{name} is deprecated, use {_deprecated[name]}",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self, _deprecated[name])
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class WeibullRULPredictor:
    """Weibull-based RUL prediction.

    Failure pattern interpretation by shape (beta):
    - beta < 1: decreasing failure rate (infant mortality)
    - beta = 1: constant failure rate (exponential)
    - beta > 1: increasing failure rate (wear-out)
    """

    def __init__(self):
        self._shape: float | None = None
        self._scale: float | None = None
        self._is_fitted = False
        self._n_samples = 0

    @property
    def shape(self) -> float:
        if self._shape is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._shape

    @shape.setter
    def shape(self, value: float) -> None:
        self._shape = value

    @property
    def scale(self) -> float:
        if self._scale is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value

    def fit(self, times_to_failure: np.ndarray, censored: np.ndarray | None = None) -> "WeibullRULPredictor":
        times = np.array(times_to_failure)
        mask = times > 0
        n_dropped = int((~mask).sum())
        if n_dropped > 0:
            logger.warning("Dropped %d non-positive values from times_to_failure", n_dropped)
        times = times[mask]

        if censored is not None:
            censored = np.array(censored)[mask]
            if len(times) != len(censored):
                raise ValueError(
                    f"times and censored length mismatch after filtering: {len(times)} vs {len(censored)}"
                )

        if len(times) < 3:
            raise ValueError("Need at least 3 failure times for fitting")

        if censored is not None:
            self._shape, self._scale = self._fit_mle_censored(times, censored)
        else:
            self._shape, _, self._scale = stats.weibull_min.fit(times, floc=0)

        self._is_fitted = True
        self._n_samples = len(times)

        return self

    def predict(
        self,
        current_hours: float,
        confidence_level: float = 0.95
    ) -> WeibullResult:
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if not (0.5 <= confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in [0.5, 1.0), got {confidence_level}")

        survival_prob = self._survival_function(current_hours)
        predicted_rul = self._conditional_mean_rul(current_hours)

        alpha = 1 - confidence_level
        ci_lower = self._conditional_percentile(current_hours, alpha / 2)
        ci_upper = self._conditional_percentile(current_hours, 1 - alpha / 2)

        hazard = self._hazard_function(current_hours)

        return WeibullResult(
            predicted_rul=max(0, predicted_rul),
            conditional_quantile_interval=(max(0, ci_lower), max(0, ci_upper)),
            survival_probability=survival_prob,
            failure_probability=1 - survival_prob,
            hazard_rate=hazard,
            shape=self.shape,
            scale=self.scale
        )

    def get_survival_curve(
        self,
        times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get survival curve S(t) for given times."""
        if not self._is_fitted:
            raise ValueError("Model not fitted.")

        survival = np.array([self._survival_function(t) for t in times])
        return times, survival

    def get_hazard_curve(
        self,
        times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get hazard curve h(t) for given times."""
        if not self._is_fitted:
            raise ValueError("Model not fitted.")

        hazard = np.array([self._hazard_function(t) for t in times])
        return times, hazard

    def _survival_function(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return np.exp(-((t / self.scale) ** self.shape))

    def _hazard_function(self, t: float) -> float:
        if t <= 0:
            return 0.0
        return (self.shape / self.scale) * ((t / self.scale) ** (self.shape - 1))

    def _pdf(self, t: float) -> float:
        if t <= 0:
            return 0.0
        return (self.shape / self.scale) * ((t / self.scale) ** (self.shape - 1)) * \
               np.exp(-((t / self.scale) ** self.shape))

    def _conditional_mean_rul(self, current_hours: float) -> float:
        from scipy.integrate import quad

        survival_current = self._survival_function(current_hours)
        if survival_current < 1e-10:
            return 0.0

        def integrand(u):
            return self._survival_function(u) / survival_current

        result, _ = quad(integrand, current_hours, np.inf, limit=100)

        return result

    def _conditional_percentile(
        self,
        current_hours: float,
        p: float
    ) -> float:
        survival_current = self._survival_function(current_hours)
        if survival_current < 1e-10:
            return 0.0

        target_survival = survival_current * (1 - p)

        if target_survival <= 0:
            return 0.0

        t_percentile = self.scale * ((-np.log(target_survival)) ** (1 / self.shape))

        return max(0, t_percentile - current_hours)

    def _fit_mle_censored(
        self,
        times: np.ndarray,
        censored: np.ndarray
    ) -> tuple[float, float]:
        """L-BFGS-B with log-reparameterization for numerical stability."""
        def neg_log_likelihood(log_params):
            shape = np.exp(log_params[0])
            scale = np.exp(log_params[1])

            ll = 0
            for t, c in zip(times, censored, strict=True):
                if c:
                    ll += np.log(self._survival_function_params(t, shape, scale))
                else:
                    ll += np.log(self._pdf_params(t, shape, scale) + 1e-10)

            return -ll

        uncensored = times[~censored] if censored is not None else times
        if len(uncensored) > 0:
            init_shape, _, init_scale = stats.weibull_min.fit(uncensored, floc=0)
        else:
            init_shape, init_scale = 2.0, np.median(times)

        log_init = [np.log(init_shape), np.log(init_scale)]
        bounds = [(np.log(0.1), np.log(10)), (np.log(1), np.log(np.max(times) * 10))]

        result = minimize(
            neg_log_likelihood,
            log_init,
            method='L-BFGS-B',
            bounds=bounds,
        )

        return np.exp(result.x[0]), np.exp(result.x[1])

    def _survival_function_params(self, t: float, shape: float, scale: float) -> float:
        if t <= 0:
            return 1.0
        return np.exp(-((t / scale) ** shape))

    def _pdf_params(self, t: float, shape: float, scale: float) -> float:
        if t <= 0:
            return 0.0
        return (shape / scale) * ((t / scale) ** (shape - 1)) * \
               np.exp(-((t / scale) ** shape))

    @property
    def median_life(self) -> float:
        """Median lifetime (50th percentile)."""
        if not self._is_fitted:
            return 0.0
        return self.scale * (np.log(2) ** (1 / self.shape))

    @property
    def mean_life(self) -> float:
        """Mean lifetime."""
        if not self._is_fitted:
            return 0.0
        from scipy.special import gamma
        return self.scale * gamma(1 + 1 / self.shape)

    def get_parameters(self) -> dict[str, float]:
        """Get model parameters."""
        if not self._is_fitted:
            return {}
        return {
            "shape": self.shape,
            "scale": self.scale,
            "median_life": self.median_life,
            "mean_life": self.mean_life
        }

    def save(self, path: Path) -> None:
        """Save model to disk."""
        model_data = {
            "shape": self.shape,
            "scale": self.scale,
            "n_samples": self._n_samples
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: Path) -> "WeibullRULPredictor":
        """Load model from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        predictor = cls()
        predictor._shape = model_data["shape"]
        predictor._scale = model_data["scale"]
        predictor._n_samples = model_data["n_samples"]
        predictor._is_fitted = True

        return predictor
