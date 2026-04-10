"""Tests for RUL prediction models."""

import numpy as np
import pytest

from anomalykit.rul.weibull import WeibullRULPredictor, WeibullResult


class TestWeibullRULPredictor:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        times_to_failure = rng.weibull(2.5, 50) * 1000

        predictor = WeibullRULPredictor()
        predictor.fit(times_to_failure)

        result = predictor.predict(current_hours=500)

        assert isinstance(result, WeibullResult)
        assert result.predicted_rul >= 0
        assert 0 <= result.survival_probability <= 1
        assert result.failure_probability == pytest.approx(1 - result.survival_probability)
        assert result.hazard_rate >= 0
        assert result.shape > 0
        assert result.scale > 0

    def test_predict_before_fit_raises(self):
        predictor = WeibullRULPredictor()
        with pytest.raises(ValueError, match="not fitted"):
            predictor.predict(current_hours=100)

    def test_confidence_interval_ordering(self):
        times = np.random.RandomState(1).weibull(3, 100) * 500
        predictor = WeibullRULPredictor().fit(times)
        result = predictor.predict(current_hours=200, confidence_level=0.95)

        ci_low, ci_high = result.confidence_interval
        assert ci_low <= ci_high

    def test_survival_curve(self):
        times = np.random.RandomState(2).weibull(2, 80) * 800
        predictor = WeibullRULPredictor().fit(times)

        t_values = np.linspace(0, 2000, 50)
        t_out, survival = predictor.get_survival_curve(t_values)

        assert len(t_out) == 50
        assert survival[0] == pytest.approx(1.0)
        assert all(0 <= s <= 1 for s in survival)
        # Survival should be non-increasing
        assert all(survival[i] >= survival[i + 1] - 1e-10 for i in range(len(survival) - 1))

    def test_hazard_curve(self):
        times = np.random.RandomState(3).weibull(3, 60) * 600
        predictor = WeibullRULPredictor().fit(times)

        t_values = np.linspace(1, 1000, 20)
        t_out, hazard = predictor.get_hazard_curve(t_values)

        assert len(hazard) == 20
        assert all(h >= 0 for h in hazard)

    def test_properties(self):
        times = np.random.RandomState(4).weibull(2, 100) * 1000
        predictor = WeibullRULPredictor().fit(times)

        assert predictor.median_life > 0
        assert predictor.mean_life > 0
        params = predictor.get_parameters()
        assert "shape" in params
        assert "scale" in params

    def test_save_and_load(self, tmp_path):
        times = np.random.RandomState(5).weibull(2, 50) * 500
        predictor = WeibullRULPredictor().fit(times)

        path = tmp_path / "weibull.pkl"
        predictor.save(path)

        loaded = WeibullRULPredictor.load(path)
        assert loaded.shape == pytest.approx(predictor.shape)
        assert loaded.scale == pytest.approx(predictor.scale)

    def test_too_few_samples_raises(self):
        predictor = WeibullRULPredictor()
        with pytest.raises(ValueError, match="at least 3"):
            predictor.fit(np.array([100.0, 200.0]))

    def test_no_double_subtraction_known_weibull(self):
        """C1 regression: conditional RUL must not subtract current_hours twice.

        For Weibull(shape=2, scale=1000), the unconditional mean life is
        scale * Gamma(1 + 1/shape) ~ 886.  At t=0 the conditional RUL equals
        the unconditional mean.  With the old double-subtraction bug,
        predict(0) would return mean - 0 = correct, but predict(500) would
        return (integral_result) - 500 a second time, giving a negative or
        drastically too-low value.
        """
        from scipy.special import gamma as gamma_fn

        predictor = WeibullRULPredictor()
        predictor.shape = 2.0
        predictor.scale = 1000.0
        predictor._is_fitted = True
        predictor._n_samples = 100

        unconditional_mean = predictor.scale * gamma_fn(1 + 1 / predictor.shape)

        rul_at_0 = predictor.predict(current_hours=0).predicted_rul
        assert rul_at_0 == pytest.approx(unconditional_mean, rel=0.02)

        rul_at_500 = predictor.predict(current_hours=500).predicted_rul
        assert rul_at_500 > 0
        assert rul_at_500 < unconditional_mean

    def test_censored_mask_alignment(self):
        """C2 regression: censored array must be filtered in sync with times."""
        times = np.array([0, 100, 200, 300, 400, 500, 600, 700])
        censored = np.array([False, False, True, False, False, True, False, False])

        predictor = WeibullRULPredictor()
        predictor.fit(times, censored=censored)

        assert predictor._is_fitted
        assert predictor.shape > 0
        assert predictor.scale > 0
