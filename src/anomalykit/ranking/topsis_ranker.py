"""TOPSIS-based multi-criteria ranking.

Multi-criteria TOPSIS ranking (Technique for Order of Preference by
Similarity to Ideal Solution) across configurable KPIs.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AssetCriteriaInput:
    """Input criteria for a single asset."""
    asset_id: str
    criteria: Dict[str, float]


@dataclass
class AssetRankResult:
    """Ranking result for a single asset."""
    asset_id: str
    rank: int
    overall_score: float
    criteria_values: Dict[str, float]
    distance_to_ideal: float
    distance_to_anti_ideal: float


@dataclass
class RankingResult:
    """Full group ranking result."""
    rankings: List[AssetRankResult]
    average_score: float
    quartile_boundaries: Dict[str, float]
    underperformers: List[str]


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    col_norms = np.linalg.norm(matrix, axis=0)
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    return matrix / col_norms


def _apply_weights(normalized: np.ndarray, weights: List[float]) -> np.ndarray:
    return normalized * np.array(weights)


def _ideal_solutions(
    weighted: np.ndarray,
    criteria_names: List[str],
    benefit_criteria: Dict[str, bool],
) -> tuple:
    ideal = np.zeros(weighted.shape[1])
    anti_ideal = np.zeros(weighted.shape[1])
    for j, criterion in enumerate(criteria_names):
        is_benefit = benefit_criteria.get(criterion, True)
        if is_benefit:
            ideal[j] = weighted[:, j].max()
            anti_ideal[j] = weighted[:, j].min()
        else:
            ideal[j] = weighted[:, j].min()
            anti_ideal[j] = weighted[:, j].max()
    return ideal, anti_ideal


def _euclidean_distances(weighted: np.ndarray, ideal: np.ndarray, anti_ideal: np.ndarray):
    d_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_anti = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
    return d_ideal, d_anti


def _relative_closeness(d_ideal: np.ndarray, d_anti: np.ndarray) -> np.ndarray:
    denominator = d_ideal + d_anti
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return d_anti / denominator


def _quartile_boundaries(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"q1": 0.0, "q2": 0.0, "q3": 0.0}
    arr = np.array(scores)
    return {
        "q1": float(np.percentile(arr, 25)),
        "q2": float(np.percentile(arr, 50)),
        "q3": float(np.percentile(arr, 75)),
    }


class TopsisRanker:
    """TOPSIS-based multi-criteria ranker.

    Generic ranker that can be configured with any set of criteria,
    weights, and benefit/cost classification.
    """

    def __init__(
        self,
        default_weights: Optional[Dict[str, float]] = None,
        benefit_criteria: Optional[Dict[str, bool]] = None,
    ):
        """
        Args:
            default_weights: Default weight per criterion (will be normalized).
            benefit_criteria: True if higher is better, False if lower is better.
        """
        self.default_weights = default_weights or {}
        self.benefit_criteria = benefit_criteria or {}

    def rank(
        self,
        assets: List[AssetCriteriaInput],
        weights: Optional[Dict[str, float]] = None,
    ) -> RankingResult:
        """Rank assets using TOPSIS.

        Args:
            assets: List of asset criteria inputs.
            weights: Optional weight overrides per criterion.

        Returns:
            RankingResult with rankings, scores, quartiles.
        """
        if not assets:
            return RankingResult(
                rankings=[],
                average_score=0.0,
                quartile_boundaries={"q1": 0.0, "q2": 0.0, "q3": 0.0},
                underperformers=[],
            )

        all_criteria_sets = [set(a.criteria.keys()) for a in assets]
        if not all(s == all_criteria_sets[0] for s in all_criteria_sets):
            raise ValueError(
                "All assets must have the same criteria keys. "
                f"First asset has {all_criteria_sets[0]}, "
                f"mismatched assets found."
            )
        all_criteria = sorted(all_criteria_sets[0])

        effective_weights = {}
        for c in all_criteria:
            effective_weights[c] = self.default_weights.get(c, 1.0 / len(all_criteria))
        if weights:
            for k, v in weights.items():
                if k in effective_weights:
                    effective_weights[k] = float(v)
        total = sum(effective_weights.values())
        if total > 0:
            effective_weights = {k: v / total for k, v in effective_weights.items()}

        weight_vector = [effective_weights[c] for c in all_criteria]

        rows = []
        for a in assets:
            rows.append([a.criteria.get(c, 0.0) for c in all_criteria])
        matrix = np.array(rows, dtype=float)

        if matrix.shape[0] == 1:
            scores_arr = np.array([1.0])
            d_ideal_arr = np.array([0.0])
            d_anti_arr = np.array([0.0])
        else:
            normalized = _normalize_matrix(matrix)
            weighted = _apply_weights(normalized, weight_vector)
            ideal, anti_ideal = _ideal_solutions(weighted, all_criteria, self.benefit_criteria)
            d_ideal_arr, d_anti_arr = _euclidean_distances(weighted, ideal, anti_ideal)
            scores_arr = _relative_closeness(d_ideal_arr, d_anti_arr)

        order = np.argsort(-scores_arr)

        quartiles = _quartile_boundaries(scores_arr.tolist())
        q1 = quartiles["q1"]

        rankings: List[AssetRankResult] = []
        underperformers: List[str] = []

        for rank_pos, idx in enumerate(order):
            a = assets[idx]
            score = float(scores_arr[idx])
            criteria_values = {
                c: float(matrix[idx, j])
                for j, c in enumerate(all_criteria)
            }
            result = AssetRankResult(
                asset_id=a.asset_id,
                rank=rank_pos + 1,
                overall_score=round(score, 4),
                criteria_values=criteria_values,
                distance_to_ideal=round(float(d_ideal_arr[idx]), 6),
                distance_to_anti_ideal=round(float(d_anti_arr[idx]), 6),
            )
            rankings.append(result)
            if score < q1:
                underperformers.append(a.asset_id)

        average_score = float(np.mean(scores_arr))

        return RankingResult(
            rankings=rankings,
            average_score=round(average_score, 4),
            quartile_boundaries=quartiles,
            underperformers=underperformers,
        )

    def rank_single(
        self,
        target_asset_id: str,
        assets: List[AssetCriteriaInput],
        weights: Optional[Dict[str, float]] = None,
    ) -> Optional[AssetRankResult]:
        """Rank all assets and return result for a specific one."""
        result = self.rank(assets, weights)
        for r in result.rankings:
            if r.asset_id == target_asset_id:
                return r
        return None
