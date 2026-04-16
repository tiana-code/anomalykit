"""Operating mode clustering with automatic mode naming."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class OperatingModeInfo:
    mode_id: int
    name: str
    size: int
    percentage: float
    avg_speed: float
    avg_load: float
    avg_fuel: float
    avg_rpm: float
    characteristics: dict[str, float]


@dataclass
class ModeTransition:
    from_mode: str
    to_mode: str
    count: int
    probability: float


@dataclass
class OperatingModeResult:
    modes: list[OperatingModeInfo]
    labels: np.ndarray
    transitions: list[ModeTransition]
    transition_matrix: dict[str, dict[str, float]]
    silhouette: float
    dominant_mode: str
    time_distribution: dict[str, float]


class OperatingModeClusterer:
    """Automatic clustering and naming of asset operating modes."""

    FEATURE_NAMES = ["speed", "rpm", "fuel_consumption", "engine_load"]

    MODE_NAMES = {
        "idle": "Idle/Anchored",
        "maneuvering": "Maneuvering",
        "slow_steaming": "Slow Steaming",
        "cruising": "Cruising",
        "full_speed": "Full Speed",
        "loading": "Loading/Unloading",
        "unknown": "Unknown"
    }

    def __init__(self, n_modes: int = 5, random_state: int = 42):
        self.n_modes = n_modes
        self.random_state = random_state

        self.model = KMeans(
            n_clusters=n_modes,
            random_state=random_state,
            n_init=10
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._fitted_features: list[str] = []
        self._mode_names: dict[int, str] = {}

    def fit_predict(self, data: pd.DataFrame) -> OperatingModeResult:
        available_features = [f for f in self.FEATURE_NAMES if f in data.columns]
        if len(available_features) < 2:
            raise ValueError(f"Need at least 2 features from {self.FEATURE_NAMES}")

        X = data[available_features].fillna(data[available_features].median())
        X_scaled = self.scaler.fit_transform(X)

        labels = self.model.fit_predict(X_scaled)
        self._is_fitted = True
        self._fitted_features = available_features

        silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0

        self._mode_names = self._name_modes(data, labels, available_features)

        modes = self._create_mode_info(data, labels, available_features)

        transitions, transition_matrix = self._calculate_transitions(labels)

        time_dist = {
            self._mode_names[i]: float(np.sum(labels == i) / len(labels) * 100)
            for i in range(self.n_modes)
        }

        dominant_idx = max(range(self.n_modes), key=lambda i: np.sum(labels == i))
        dominant_mode = self._mode_names[dominant_idx]

        return OperatingModeResult(
            modes=modes,
            labels=labels,
            transitions=transitions,
            transition_matrix=transition_matrix,
            silhouette=silhouette,
            dominant_mode=dominant_mode,
            time_distribution=time_dist
        )

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict operating modes for new data."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit_predict() first.")

        missing = [f for f in self._fitted_features if f not in data.columns]
        if missing:
            raise ValueError(
                f"Input data is missing columns used during fit: {missing}"
            )

        X = data[self._fitted_features].fillna(data[self._fitted_features].median())
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def get_mode_name(self, mode_id: int) -> str:
        """Get name for a mode ID."""
        return self._mode_names.get(mode_id, "Unknown")

    def _name_modes(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        features: list[str]
    ) -> dict[int, str]:
        names = {}

        for i in range(self.n_modes):
            mask = labels == i
            cluster_data = data[mask]

            avg_speed = cluster_data.get("speed", pd.Series([5])).mean()
            avg_load = cluster_data.get("engine_load", pd.Series([50])).mean()
            _avg_rpm = cluster_data.get("rpm", pd.Series([500])).mean()
            _avg_fuel = cluster_data.get("fuel_consumption", pd.Series([10])).mean()

            if avg_speed < 2:
                if avg_load < 20:
                    names[i] = "Idle/Anchored"
                else:
                    names[i] = "Loading/Unloading"
            elif avg_speed < 6:
                names[i] = "Maneuvering"
            elif avg_speed < 10 and avg_load < 50:
                names[i] = "Slow Steaming"
            elif avg_load > 75:
                names[i] = "Full Speed"
            else:
                names[i] = "Cruising"

        name_counts: dict[str, int] = {}
        for i, name in names.items():
            if name in name_counts:
                name_counts[name] += 1
                names[i] = f"{name} {name_counts[name]}"
            else:
                name_counts[name] = 1

        return names

    def _create_mode_info(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        features: list[str]
    ) -> list[OperatingModeInfo]:
        modes = []
        total = len(labels)

        for i in range(self.n_modes):
            mask = labels == i
            cluster_data = data[mask]
            size = int(mask.sum())

            avg_speed = float(cluster_data.get("speed", pd.Series([0])).mean())
            avg_load = float(cluster_data.get("engine_load", pd.Series([0])).mean())
            avg_fuel = float(cluster_data.get("fuel_consumption", pd.Series([0])).mean())
            avg_rpm = float(cluster_data.get("rpm", pd.Series([0])).mean())

            characteristics = {}
            for f in features:
                if f in cluster_data.columns:
                    characteristics[f] = float(cluster_data[f].mean())

            modes.append(OperatingModeInfo(
                mode_id=i,
                name=self._mode_names[i],
                size=size,
                percentage=float(size / total * 100),
                avg_speed=avg_speed,
                avg_load=avg_load,
                avg_fuel=avg_fuel,
                avg_rpm=avg_rpm,
                characteristics=characteristics
            ))

        return modes

    def _calculate_transitions(
        self,
        labels: np.ndarray
    ) -> tuple[list[ModeTransition], dict[str, dict[str, float]]]:
        transitions: list[ModeTransition] = []
        transition_counts: dict[tuple[str, str], int] = {}

        for i in range(len(labels) - 1):
            from_mode = self._mode_names[labels[i]]
            to_mode = self._mode_names[labels[i + 1]]

            key = (from_mode, to_mode)
            transition_counts[key] = transition_counts.get(key, 0) + 1

        mode_counts = {name: sum(1 for lbl in labels if self._mode_names[lbl] == name)
                      for name in self._mode_names.values()}

        transition_matrix: dict[str, dict[str, float]] = {name: {} for name in self._mode_names.values()}

        for (from_mode, to_mode), count in transition_counts.items():
            prob = count / mode_counts[from_mode] if mode_counts[from_mode] > 0 else 0
            transition_matrix[from_mode][to_mode] = prob

            transitions.append(ModeTransition(
                from_mode=from_mode,
                to_mode=to_mode,
                count=count,
                probability=prob
            ))

        return transitions, transition_matrix

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance based on cluster center variance."""
        if not self._is_fitted:
            return {}

        centers = self.model.cluster_centers_
        variances = np.var(centers, axis=0)
        total = variances.sum()

        if total == 0:
            available = [f for f in self.FEATURE_NAMES if len(variances) > self.FEATURE_NAMES.index(f)]
            return {f: 1.0 / len(available) for f in available}

        importance = {}
        for i, f in enumerate(self.FEATURE_NAMES):
            if i < len(variances):
                importance[f] = float(variances[i] / total)

        return importance
