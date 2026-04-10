"""Tests for anomaly detection models."""

import numpy as np
import pandas as pd
import pytest

from anomalykit.anomaly.isolation_forest import IsolationForestDetector, IsolationForestResult
from anomalykit.anomaly.multi_sensor import (
    CrossCorrelationAnalyzer,
    MultiSensorPatternDetector,
    SensorFusionDetector,
)
from anomalykit.anomaly.contextual_detector import ContextualDetector, OperatingMode
from anomalykit.anomaly.adaptive_threshold import AdaptiveThresholdEngine


class TestIsolationForestDetector:
    def test_fit_and_detect_finds_anomalies(self):
        rng = np.random.RandomState(42)
        normal = rng.randn(200, 3) * 2 + 10
        anomalous = rng.randn(10, 3) * 2 + 30
        data = pd.DataFrame(
            np.vstack([normal, anomalous]),
            columns=["sensor_a", "sensor_b", "sensor_c"],
        )

        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(data)
        result = detector.detect(data)

        assert isinstance(result, IsolationForestResult)
        assert result.anomaly_mask.shape == (210,)
        assert result.batch_normalized_score.shape == (210,)
        assert result.anomaly_mask.sum() > 0

    def test_detect_before_fit_raises(self):
        detector = IsolationForestDetector()
        data = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="not fitted"):
            detector.detect(data)

    def test_feature_importance_returns_dict(self):
        data = pd.DataFrame(np.random.randn(100, 2), columns=["a", "b"])
        detector = IsolationForestDetector().fit(data)
        importance = detector.estimate_feature_sensitivity()
        assert "a" in importance
        assert "b" in importance
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_save_and_load(self, tmp_path):
        data = pd.DataFrame(np.random.randn(100, 2), columns=["x", "y"])
        detector = IsolationForestDetector(random_state=0).fit(data)
        path = tmp_path / "model.pkl"
        detector.save(path)

        loaded = IsolationForestDetector.load(path)
        result = loaded.detect(data)
        assert result.anomaly_mask.shape == (100,)


class TestIsolationForestDetectorExtended:
    def test_scale_data_false(self):
        data = pd.DataFrame(np.random.randn(100, 2), columns=["x", "y"])
        detector = IsolationForestDetector(scale_data=False, random_state=0).fit(data)
        result = detector.detect(data)
        assert result.anomaly_mask.shape == (100,)

    def test_score_samples_returns_raw(self):
        data = pd.DataFrame(np.random.randn(100, 2), columns=["x", "y"])
        detector = IsolationForestDetector(random_state=0).fit(data)
        raw_scores = detector.score_samples(data)
        assert raw_scores.shape == (100,)
        assert raw_scores.dtype == np.float64

    def test_deprecated_scores_field_warns(self):
        data = pd.DataFrame(np.random.randn(100, 2), columns=["x", "y"])
        detector = IsolationForestDetector(random_state=0).fit(data)
        result = detector.detect(data)
        with pytest.warns(DeprecationWarning, match="scores is deprecated"):
            _ = result.scores

    def test_deprecated_get_feature_importance_warns(self):
        data = pd.DataFrame(np.random.randn(100, 2), columns=["x", "y"])
        detector = IsolationForestDetector(random_state=0).fit(data)
        with pytest.warns(DeprecationWarning, match="get_feature_importance"):
            _ = detector.get_feature_importance()


class TestMultiSensorPatternDetector:
    def test_fit_and_detect(self):
        rng = np.random.RandomState(0)
        data = pd.DataFrame(rng.randn(100, 3), columns=["s1", "s2", "s3"])

        detector = MultiSensorPatternDetector(pattern_window=5, anomaly_threshold=2.0)
        detector.fit(data, ["s1", "s2", "s3"])
        result = detector.detect(data, ["s1", "s2", "s3"])

        assert result.scores.shape == (100,)
        assert isinstance(result.pattern_count, int)

    def test_short_data_raises(self):
        data = pd.DataFrame({"s1": [1, 2], "s2": [3, 4]})
        detector = MultiSensorPatternDetector(pattern_window=10)
        with pytest.raises(ValueError, match="Insufficient data"):
            detector.fit(data, ["s1", "s2"])

    def test_detect_before_fit_raises(self):
        data = pd.DataFrame({"s1": range(20), "s2": range(20)})
        detector = MultiSensorPatternDetector(pattern_window=5)
        with pytest.raises(ValueError, match="not fitted"):
            detector.detect(data, ["s1", "s2"])

    def test_refit_clears_baseline_patterns(self):
        rng = np.random.RandomState(10)
        data1 = pd.DataFrame(rng.randn(50, 2), columns=["s1", "s2"])
        data2 = pd.DataFrame(rng.randn(50, 2), columns=["s1", "s2"])

        detector = MultiSensorPatternDetector(pattern_window=5)
        detector.fit(data1, ["s1", "s2"])
        count_after_first_fit = len(detector._baseline_patterns)

        detector.fit(data2, ["s1", "s2"])
        count_after_second_fit = len(detector._baseline_patterns)

        assert count_after_second_fit == count_after_first_fit


class TestSensorFusionDetector:
    def test_fuse_before_fit_raises(self):
        data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        detector = SensorFusionDetector()
        with pytest.raises(ValueError, match="not fitted"):
            detector.fuse(data, ["a", "b", "c"])

    def test_fuse_detects_reconstruction_anomalies(self):
        rng = np.random.RandomState(1)
        normal = rng.randn(100, 4)
        anomalous = rng.randn(5, 4) * 10
        data = pd.DataFrame(
            np.vstack([normal, anomalous]),
            columns=["t1", "t2", "t3", "t4"],
        )

        detector = SensorFusionDetector(reconstruction_threshold=2.0)
        detector.fit(data, ["t1", "t2", "t3", "t4"])
        result = detector.fuse(data, ["t1", "t2", "t3", "t4"])

        assert result.principal_component.shape[0] == 105
        assert result.explained_variance > 0
        assert sum(result.component_weights.values()) > 0


class TestCrossCorrelationAnalyzer:
    def test_detects_broken_correlation(self):
        rng = np.random.RandomState(2)
        x = rng.randn(200)
        baseline = pd.DataFrame({"a": x, "b": x + rng.randn(200) * 0.1})
        analyzer = CrossCorrelationAnalyzer(correlation_change_threshold=0.3)
        analyzer.fit(baseline, ["a", "b"])

        # Break correlation
        test_data = pd.DataFrame({"a": rng.randn(50), "b": rng.randn(50)})
        result = analyzer.analyze(test_data, ["a", "b"])

        assert len(result.broken_correlations) > 0


class TestContextualDetector:
    def test_detect_with_speed_column(self):
        rng = np.random.RandomState(3)
        n = 200
        data = pd.DataFrame({
            "speed": rng.uniform(0, 20, n),
            "temperature": rng.uniform(50, 100, n),
            "pressure": rng.uniform(900, 1100, n),
        })

        detector = ContextualDetector(contamination=0.1)
        detector.fit(data, ["temperature", "pressure"])
        result = detector.detect(data, ["temperature", "pressure"], asset_id="asset-001")

        assert result.asset_id == "asset-001"
        assert result.total_points == n
        assert isinstance(result.anomaly_rate, float)

    def test_operating_mode_assignment(self):
        detector = ContextualDetector()
        assert detector._sog_to_mode(0.5) == OperatingMode.AT_ANCHOR.value
        assert detector._sog_to_mode(3.0) == OperatingMode.MANEUVERING.value
        assert detector._sog_to_mode(8.0) == OperatingMode.SLOW_STEAM.value
        assert detector._sog_to_mode(14.0) == OperatingMode.ECO.value
        assert detector._sog_to_mode(20.0) == OperatingMode.FULL_SPEED.value


class TestContextualDetectorExtended:
    def test_missing_value_strategy_median(self):
        rng = np.random.RandomState(10)
        n = 100
        data = pd.DataFrame({
            "speed": rng.uniform(0, 20, n),
            "s1": rng.uniform(50, 100, n),
        })
        data.loc[0, "s1"] = np.nan
        detector = ContextualDetector(contamination=0.1, missing_value_strategy="median")
        detector.fit(data, ["s1"])
        result = detector.detect(data, ["s1"], asset_id="test")
        assert result.total_points == n

    def test_custom_mode_ranges(self):
        from anomalykit.anomaly.contextual_detector import OperatingMode
        custom_sog = {
            OperatingMode.AT_ANCHOR: (0, 2),
            OperatingMode.MANEUVERING: (2, 8),
            OperatingMode.SLOW_STEAM: (8, 14),
            OperatingMode.ECO: (14, 20),
            OperatingMode.FULL_SPEED: (20, 100),
        }
        detector = ContextualDetector(mode_sog_ranges=custom_sog)
        assert detector._sog_to_mode(1.5) == OperatingMode.AT_ANCHOR.value
        assert detector._sog_to_mode(5.0) == OperatingMode.MANEUVERING.value
        assert detector._sog_to_mode(10.0) == OperatingMode.SLOW_STEAM.value


class TestAdaptiveThresholdEngine:
    def test_calculate_baselines_and_violations(self):
        rng = np.random.RandomState(4)
        values = rng.normal(100, 5, 100).tolist() + [200, 210, 220]
        data = pd.DataFrame({"temperature": values})

        engine = AdaptiveThresholdEngine(k_factor=3.0, min_history=10)
        result = engine.calculate(data, ["temperature"], asset_id="pump-01")

        assert result.asset_id == "pump-01"
        assert len(result.baselines) == 1
        assert result.violation_count > 0

    def test_set_k_factor_updates_thresholds(self):
        data = pd.DataFrame({"temp": np.random.normal(50, 2, 50)})
        engine = AdaptiveThresholdEngine(k_factor=3.0)
        engine.calculate(data, ["temp"], asset_id="a1")

        baselines_before = engine.get_baselines("a1")
        assert len(baselines_before) == 1

        engine.set_k_factor(5.0)
        engine.calculate(data, ["temp"], asset_id="a1")
        baselines_after = engine.get_baselines("a1")
        assert baselines_after[0].k_factor == 5.0
        assert baselines_after[0].threshold_high >= baselines_before[0].threshold_high

    def test_no_look_ahead_bias(self):
        """C3 regression: a spike at the end must not shift the baseline used
        to evaluate earlier points.  With look-ahead, the spike would widen
        the std for all points (including those before the spike), hiding
        violations that should have been caught."""
        normal = np.random.RandomState(99).normal(100, 2, 60).tolist()
        spike = [200.0, 210.0, 220.0]
        data = pd.DataFrame({"sensor": normal + spike})

        engine = AdaptiveThresholdEngine(k_factor=3.0, min_history=10)
        result = engine.calculate(data, ["sensor"], asset_id="bias-test")

        spike_indices = set(range(60, 63))
        violation_indices = {v.index for v in result.violations}
        assert spike_indices.issubset(violation_indices), (
            f"Spike points {spike_indices} should be violations but got {violation_indices}"
        )

    def test_violations_sorted_by_deviation(self):
        rng = np.random.RandomState(4)
        values = rng.normal(100, 5, 100).tolist() + [200, 210, 220]
        data = pd.DataFrame({"temperature": values})

        engine = AdaptiveThresholdEngine(k_factor=3.0, min_history=10)
        result = engine.calculate(data, ["temperature"], asset_id="sort-test")

        sorted_violations = sorted(
            result.violations, key=lambda v: v.deviation_mad_multiple, reverse=True
        )
        assert sorted_violations[0].deviation_mad_multiple >= sorted_violations[-1].deviation_mad_multiple


class TestAdaptiveThresholdEngineExtended:
    def test_mad_warmup_initialization(self):
        rng = np.random.RandomState(50)
        values = rng.normal(100, 5, 50).tolist()
        data = pd.DataFrame({"sensor": values})
        engine = AdaptiveThresholdEngine(k_factor=3.0, min_history=10)
        result = engine.calculate(data, ["sensor"], asset_id="warmup-test")
        bl = result.baselines[0]
        assert bl.ema_abs_dev > 0
        assert bl.ema_abs_dev != abs(values[0]) * 0.01


class TestContextualDetectorAnomalyRate:
    def test_anomaly_rate_never_exceeds_one(self):
        """H4 regression: anomaly_rate must be <= 1.0 even when multiple
        sensor columns flag the same row."""
        rng = np.random.RandomState(77)
        n = 50
        data = pd.DataFrame({
            "speed": rng.uniform(0, 20, n),
            "s1": rng.uniform(50, 100, n),
            "s2": rng.uniform(50, 100, n),
            "s3": rng.uniform(50, 100, n),
        })

        detector = ContextualDetector(contamination=0.3)
        detector.fit(data, ["s1", "s2", "s3"])
        result = detector.detect(data, ["s1", "s2", "s3"], asset_id="rate-test")

        assert 0.0 <= result.anomaly_rate <= 1.0, (
            f"anomaly_rate={result.anomaly_rate} is out of [0, 1]"
        )


class TestContextualAnomalyPointFields:
    def test_empirical_quantile_fields_present(self):
        rng = np.random.RandomState(5)
        n = 100
        data = pd.DataFrame({
            "speed": rng.uniform(0, 20, n),
            "temperature": rng.uniform(50, 100, n),
        })

        detector = ContextualDetector(contamination=0.1)
        detector.fit(data, ["temperature"])
        result = detector.detect(data, ["temperature"], asset_id="fields-test")

        anomalous = [p for p in result.anomalies if p is not None]
        if anomalous:
            point = anomalous[0]
            assert hasattr(point, "empirical_quantile_low")
            assert hasattr(point, "empirical_quantile_high")


class TestIsolationForestFeatureImpact:
    def test_permutation_feature_impact_present(self):
        data = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        detector = IsolationForestDetector(random_state=0).fit(data)
        result = detector.detect(data)
        assert hasattr(result, "permutation_feature_impact")
        assert result.permutation_feature_impact is not None
