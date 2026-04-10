"""Tests for clustering models."""

import numpy as np
import pandas as pd
import pytest

from anomalykit.cluster.behavior_classifier import AssetBehaviorClassifier, BehaviorClassification
from anomalykit.cluster.operating_modes import OperatingModeClusterer, OperatingModeResult


class TestAssetBehaviorClassifier:
    def test_fallback_classify_liner(self):
        classifier = AssetBehaviorClassifier(model_path="/nonexistent/path")
        result = classifier.classify({
            "avg_speed": 15.0,
            "speed_std": 2.0,
            "cog_std": 10.0,
            "avg_gap_hours": 0.3,
            "port_time_ratio": 0.15,
        })

        assert isinstance(result, BehaviorClassification)
        assert result.cluster_label == "liner"
        assert result.cluster_id == -1
        assert result.cluster_affinity == 0.5

    def test_fallback_classify_fishing(self):
        classifier = AssetBehaviorClassifier(model_path="/nonexistent/path")
        result = classifier.classify({
            "avg_speed": 3.0,
            "speed_std": 3.0,
            "cog_std": 40.0,
            "avg_gap_hours": 0.5,
            "port_time_ratio": 0.2,
        })
        assert result.cluster_label == "fishing"

    def test_fallback_classify_dark(self):
        classifier = AssetBehaviorClassifier(model_path="/nonexistent/path")
        result = classifier.classify({
            "avg_speed": 5.0,
            "speed_std": 2.0,
            "cog_std": 20.0,
            "avg_gap_hours": 8.0,
            "port_time_ratio": 0.1,
        })
        assert result.cluster_label == "dark"

    def test_classify_batch(self):
        classifier = AssetBehaviorClassifier(model_path="/nonexistent/path")
        assets = [
            {"avg_speed": 15.0, "speed_std": 2.0, "cog_std": 10.0, "avg_gap_hours": 0.3, "port_time_ratio": 0.15},
            {"avg_speed": 0.5, "speed_std": 0.5, "cog_std": 5.0, "avg_gap_hours": 0.2, "port_time_ratio": 0.8},
        ]
        results = classifier.classify_batch(assets)
        assert len(results) == 2
        assert results[0].cluster_label == "liner"
        assert results[1].cluster_label == "anchored"

    def test_cluster_descriptions(self):
        classifier = AssetBehaviorClassifier(model_path="/nonexistent/path")
        descriptions = classifier.get_cluster_descriptions()
        assert "liner" in descriptions
        assert "fishing" in descriptions

    def test_historical_baseline(self):
        classifier = AssetBehaviorClassifier(model_path="/nonexistent/path")
        baseline = classifier.get_historical_baseline("TANKER")
        assert "avg_speed" in baseline
        assert baseline["avg_speed"] > 0


class TestOperatingModeClusterer:
    def _make_data(self):
        rng = np.random.RandomState(42)
        n = 300
        data = pd.DataFrame({
            "speed": np.concatenate([
                rng.uniform(0, 2, 60),
                rng.uniform(3, 8, 60),
                rng.uniform(8, 12, 60),
                rng.uniform(12, 16, 60),
                rng.uniform(16, 22, 60),
            ]),
            "rpm": np.concatenate([
                rng.uniform(0, 100, 60),
                rng.uniform(200, 400, 60),
                rng.uniform(400, 600, 60),
                rng.uniform(600, 800, 60),
                rng.uniform(800, 1000, 60),
            ]),
            "fuel_consumption": np.concatenate([
                rng.uniform(0, 2, 60),
                rng.uniform(5, 10, 60),
                rng.uniform(10, 20, 60),
                rng.uniform(20, 35, 60),
                rng.uniform(35, 50, 60),
            ]),
            "engine_load": np.concatenate([
                rng.uniform(0, 15, 60),
                rng.uniform(20, 40, 60),
                rng.uniform(40, 60, 60),
                rng.uniform(60, 80, 60),
                rng.uniform(80, 100, 60),
            ]),
        })
        return data

    def test_fit_predict(self):
        data = self._make_data()
        clusterer = OperatingModeClusterer(n_modes=5, random_state=42)
        result = clusterer.fit_predict(data)

        assert isinstance(result, OperatingModeResult)
        assert len(result.modes) == 5
        assert result.labels.shape == (300,)
        assert result.silhouette > 0
        assert result.dominant_mode != ""

    def test_predict_after_fit(self):
        data = self._make_data()
        clusterer = OperatingModeClusterer(n_modes=3, random_state=0)
        clusterer.fit_predict(data)

        new_data = data.iloc[:10]
        labels = clusterer.predict(new_data)
        assert len(labels) == 10

    def test_predict_before_fit_raises(self):
        clusterer = OperatingModeClusterer()
        data = pd.DataFrame({"speed": [1], "rpm": [100]})
        with pytest.raises(ValueError, match="not fitted"):
            clusterer.predict(data)

    def test_transition_matrix(self):
        data = self._make_data()
        clusterer = OperatingModeClusterer(n_modes=3, random_state=0)
        result = clusterer.fit_predict(data)

        assert len(result.transitions) > 0
        assert isinstance(result.transition_matrix, dict)

    def test_feature_importance(self):
        data = self._make_data()
        clusterer = OperatingModeClusterer(n_modes=3, random_state=0)
        clusterer.fit_predict(data)

        importance = clusterer.get_feature_importance()
        assert len(importance) > 0
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_insufficient_features_raises(self):
        data = pd.DataFrame({"speed": [1, 2, 3]})
        clusterer = OperatingModeClusterer()
        with pytest.raises(ValueError, match="at least 2"):
            clusterer.fit_predict(data)

    def test_predict_missing_columns_raises(self):
        data = self._make_data()
        clusterer = OperatingModeClusterer(n_modes=3, random_state=0)
        clusterer.fit_predict(data)

        incomplete = data[["speed"]].iloc[:10]
        with pytest.raises(ValueError, match="missing columns"):
            clusterer.predict(incomplete)
