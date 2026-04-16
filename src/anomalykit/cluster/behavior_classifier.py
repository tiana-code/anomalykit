"""KMeans-based asset behavior classifier. Fallback: cluster_id=-1, affinity=0.5."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH_ENV = "BEHAVIOR_MODEL_PATH"
DEFAULT_MODEL_PATH = "models/cluster/behavior_centroids.joblib"

FEATURE_NAMES = [
    "avg_speed",
    "speed_std",
    "cog_std",
    "avg_gap_hours",
    "port_time_ratio",
]

_DEFAULT_CLUSTERS: dict[str, str] = {
    "liner": "High-speed assets on regular routes with low speed and course variability.",
    "tramp": "Medium-speed assets with high course variability - irregular routes.",
    "fishing": "Low average speed with high speed and course variability.",
    "anchored": "Assets spending most time stationary at anchorage or port.",
    "loitering": "Slow-moving assets with moderate variability and long gaps.",
    "dark": "Assets with unusually long reporting gaps.",
    "slow_steaming": "Reduced-speed operation for fuel economy.",
    "fast_transit": "High-speed repositioning or ballast transit.",
    "port_intensive": "Assets spending a large fraction of time in port.",
    "standard_transit": "Standard transit with moderate characteristics.",
}


@dataclass
class BehaviorClassification:
    cluster_id: int
    cluster_label: str
    cluster_affinity: float
    nearest_centroid_distance: float
    cluster_description: str = ""


class AssetBehaviorClassifier:
    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
        self._scaler = None
        self._model = None
        self._cluster_labels: dict[int, str] = {}
        self._cluster_descriptions: dict[int, str] = {}
        self._feature_names: list[str] = FEATURE_NAMES
        self._centroids_original: np.ndarray | None = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        path = Path(self._model_path)
        if not path.exists():
            logger.warning("Behavior model not found at %s - classifier will use defaults", path)
            return

        try:
            import joblib
            bundle = joblib.load(path)
            self._scaler = bundle["scaler"]
            self._model = bundle["model"]
            self._cluster_labels = {int(k): v for k, v in bundle["cluster_labels"].items()}
            self._cluster_descriptions = {int(k): v for k, v in bundle["cluster_descriptions"].items()}
            self._feature_names = bundle.get("feature_names", FEATURE_NAMES)
            centroids = bundle.get("centroids_original")
            if centroids is not None:
                self._centroids_original = np.array(centroids)
            self._loaded = True
            logger.info(
                "Loaded behavior model with %d clusters from %s",
                len(self._cluster_labels), path,
            )
        except Exception as e:
            logger.error("Failed to load behavior model: %s", e)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def classify(self, asset_features: dict[str, float]) -> BehaviorClassification:
        if not self._loaded:
            return self._fallback_classify(asset_features)

        feature_vector = np.array(
            [asset_features.get(f, 0.0) for f in self._feature_names]
        ).reshape(1, -1)

        assert self._scaler is not None and self._model is not None
        scaled = self._scaler.transform(feature_vector)
        cluster_id = int(self._model.predict(scaled)[0])

        centroid = self._model.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(scaled[0] - centroid))

        distances_to_all = np.linalg.norm(scaled[0] - self._model.cluster_centers_, axis=1)
        min_dist = distances_to_all.min()
        if min_dist == 0:
            confidence = 1.0
        else:
            inv_distances = 1.0 / (distances_to_all + 1e-8)
            confidence = float(inv_distances[cluster_id] / inv_distances.sum())

        label = self._cluster_labels.get(cluster_id, f"cluster_{cluster_id}")
        description = self._cluster_descriptions.get(cluster_id, "")

        return BehaviorClassification(
            cluster_id=cluster_id,
            cluster_label=label,
            cluster_affinity=round(confidence, 4),
            nearest_centroid_distance=round(distance, 4),
            cluster_description=description,
        )

    def classify_batch(
        self, assets: list[dict[str, Any]]
    ) -> list[BehaviorClassification]:
        return [self.classify(a) for a in assets]

    def get_cluster_descriptions(self) -> dict[str, dict[str, Any]]:
        """Returns dict keyed by cluster_label with description and centroid feature values."""
        if not self._loaded:
            return {
                label: {"description": desc, "centroid": {}}
                for label, desc in _DEFAULT_CLUSTERS.items()
            }

        result = {}
        for cid, label in self._cluster_labels.items():
            centroid = {}
            if self._centroids_original is not None and cid < len(self._centroids_original):
                centroid = {
                    fname: round(float(val), 4)
                    for fname, val in zip(self._feature_names, self._centroids_original[cid], strict=False)
                }
            result[label] = {
                "cluster_id": cid,
                "description": self._cluster_descriptions.get(cid, ""),
                "centroid": centroid,
            }
        return result

    def _fallback_classify(self, features: dict[str, float]) -> BehaviorClassification:
        avg_speed = features.get("avg_speed", 0.0)
        speed_std = features.get("speed_std", 0.0)
        cog_std = features.get("cog_std", 0.0)
        avg_gap = features.get("avg_gap_hours", 0.0)
        port_ratio = features.get("port_time_ratio", 0.0)

        if avg_gap > 6.0:
            label, desc = "dark", _DEFAULT_CLUSTERS["dark"]
        elif avg_speed < 1.0 and cog_std < 15.0:
            label, desc = "anchored", _DEFAULT_CLUSTERS["anchored"]
        elif avg_speed < 3.0 and cog_std > 20.0 and avg_gap > 3.0:
            label, desc = "loitering", _DEFAULT_CLUSTERS["loitering"]
        elif avg_speed < 5.0 and cog_std > 30.0 and speed_std > 2.0:
            label, desc = "fishing", _DEFAULT_CLUSTERS["fishing"]
        elif avg_speed > 12.0 and speed_std < 3.0 and cog_std < 20.0:
            label, desc = "liner", _DEFAULT_CLUSTERS["liner"]
        elif avg_speed > 12.0:
            label, desc = "fast_transit", _DEFAULT_CLUSTERS["fast_transit"]
        elif cog_std > 30.0:
            label, desc = "tramp", _DEFAULT_CLUSTERS["tramp"]
        elif port_ratio > 0.5:
            label, desc = "port_intensive", _DEFAULT_CLUSTERS["port_intensive"]
        elif avg_speed < 10.0 and speed_std < 2.0 and port_ratio < 0.3:
            label, desc = "slow_steaming", _DEFAULT_CLUSTERS["slow_steaming"]
        else:
            label, desc = "standard_transit", _DEFAULT_CLUSTERS["standard_transit"]

        return BehaviorClassification(
            cluster_id=-1,
            cluster_label=label,
            cluster_affinity=0.5,
            nearest_centroid_distance=0.0,
            cluster_description=desc,
        )

    def get_historical_baseline(self, asset_type: str) -> dict[str, float]:
        """Return historical baseline averages for an asset type.

        Uses centroids that most closely match the asset type's expected behavior.
        Falls back to type-based averages if model not loaded.
        """
        type_to_label = {
            "TANKER": "slow_steaming",
            "CONTAINER": "liner",
            "CARGO": "standard_transit",
            "BULK": "slow_steaming",
            "FISHING": "fishing",
            "PASSENGER": "liner",
            "TUG": "port_intensive",
            "HSC": "fast_transit",
            "SAILING": "tramp",
            "LNG_CARRIER": "liner",
            "RO_RO": "liner",
            "RESEARCH": "tramp",
        }

        target_label = type_to_label.get(asset_type.upper(), "standard_transit")

        if self._loaded and self._centroids_original is not None:
            for cid, label in self._cluster_labels.items():
                if label == target_label and cid < len(self._centroids_original):
                    return {
                        fname: round(float(val), 4)
                        for fname, val in zip(self._feature_names, self._centroids_original[cid], strict=False)
                    }

        _type_defaults = {
            "TANKER":      {"avg_speed": 11.0, "speed_std": 2.5, "cog_std": 15.0,
                            "avg_gap_hours": 0.5, "port_time_ratio": 0.25},
            "CONTAINER":   {"avg_speed": 16.0, "speed_std": 2.0, "cog_std": 10.0,
                            "avg_gap_hours": 0.3, "port_time_ratio": 0.15},
            "CARGO":       {"avg_speed": 11.0, "speed_std": 3.0, "cog_std": 20.0,
                            "avg_gap_hours": 0.5, "port_time_ratio": 0.30},
            "BULK":        {"avg_speed": 10.5, "speed_std": 2.0, "cog_std": 12.0,
                            "avg_gap_hours": 0.4, "port_time_ratio": 0.35},
            "FISHING":     {"avg_speed": 4.0,  "speed_std": 3.5, "cog_std": 45.0,
                            "avg_gap_hours": 1.0, "port_time_ratio": 0.20},
            "PASSENGER":   {"avg_speed": 18.0, "speed_std": 3.0, "cog_std": 12.0,
                            "avg_gap_hours": 0.2, "port_time_ratio": 0.20},
            "TUG":         {"avg_speed": 5.0,  "speed_std": 3.0, "cog_std": 50.0,
                            "avg_gap_hours": 0.3, "port_time_ratio": 0.60},
            "HSC":         {"avg_speed": 25.0, "speed_std": 5.0, "cog_std": 15.0,
                            "avg_gap_hours": 0.2, "port_time_ratio": 0.15},
            "SAILING":     {"avg_speed": 6.0,  "speed_std": 3.0, "cog_std": 35.0,
                            "avg_gap_hours": 1.5, "port_time_ratio": 0.30},
            "LNG_CARRIER": {"avg_speed": 15.0, "speed_std": 1.5, "cog_std": 8.0,
                            "avg_gap_hours": 0.3, "port_time_ratio": 0.20},
        }
        return _type_defaults.get(asset_type.upper(), {
            "avg_speed": 10.0, "speed_std": 3.0, "cog_std": 20.0,
            "avg_gap_hours": 0.5, "port_time_ratio": 0.25,
        })
