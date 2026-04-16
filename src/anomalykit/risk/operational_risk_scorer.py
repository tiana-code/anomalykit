"""Weighted heuristic risk scorecard (equipment, weather, crew, compliance, route).

NOT a calibrated probabilistic model — output is a composite index (0-1) for
ranking and alerting only, not a literal probability of adverse events.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RiskDimension:
    dimension: str
    score: float
    weight: float
    weighted_score: float
    status: str
    description: str
    trend: str
    trend_delta: float


@dataclass
class RiskMatrixCell:
    likelihood: str
    impact: str
    risk_category: str
    color: str
    applicable_dimensions: list[str]


@dataclass
class RiskTrendPoint:
    date: str
    total_risk_score: float
    equipment_score: float
    weather_score: float
    crew_score: float
    compliance_score: float
    route_score: float


@dataclass
class OperationalRiskResult:
    asset_id: str
    asset_name: str
    assessment_timestamp: str
    overall_risk_score: float
    risk_level: str
    risk_dimensions: list[RiskDimension]
    risk_matrix: list[RiskMatrixCell]
    trend_7d: list[RiskTrendPoint] | None
    top_risk_factors: list[str]
    recommended_actions: list[str]
    notes: str


@dataclass
class GroupRiskSummary:
    assessment_timestamp: str
    group_average_risk: float
    highest_risk_asset_id: str
    highest_risk_asset_name: str
    highest_risk_score: float
    assets_at_high_risk: int
    assets_at_medium_risk: int
    assets_at_low_risk: int
    total_assets: int
    asset_risk_scores: list[dict]
    group_trend: str
    notes: str


DEFAULT_RISK_WEIGHTS = {
    "equipment_health": 0.30,
    "weather_exposure": 0.20,
    "crew_fatigue": 0.15,
    "compliance_status": 0.20,
    "route_risk": 0.15,
}

LIKELIHOOD_IMPACT = [
    ("rare", "negligible", "low", "#4CAF50"),
    ("rare", "minor", "low", "#4CAF50"),
    ("rare", "moderate", "low", "#4CAF50"),
    ("rare", "major", "medium", "#FF9800"),
    ("rare", "catastrophic", "medium", "#FF9800"),
    ("unlikely", "negligible", "low", "#4CAF50"),
    ("unlikely", "minor", "low", "#4CAF50"),
    ("unlikely", "moderate", "medium", "#FF9800"),
    ("unlikely", "major", "medium", "#FF9800"),
    ("unlikely", "catastrophic", "high", "#F44336"),
    ("possible", "negligible", "low", "#4CAF50"),
    ("possible", "minor", "medium", "#FF9800"),
    ("possible", "moderate", "medium", "#FF9800"),
    ("possible", "major", "high", "#F44336"),
    ("possible", "catastrophic", "critical", "#B71C1C"),
    ("likely", "negligible", "medium", "#FF9800"),
    ("likely", "minor", "medium", "#FF9800"),
    ("likely", "moderate", "high", "#F44336"),
    ("likely", "major", "critical", "#B71C1C"),
    ("likely", "catastrophic", "critical", "#B71C1C"),
    ("almost_certain", "negligible", "medium", "#FF9800"),
    ("almost_certain", "minor", "high", "#F44336"),
    ("almost_certain", "moderate", "high", "#F44336"),
    ("almost_certain", "major", "critical", "#B71C1C"),
    ("almost_certain", "catastrophic", "critical", "#B71C1C"),
]


class OperationalRiskScorer:
    """Weighted heuristic scorecard for operational risk."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        group_trend_thresholds: tuple[float, float] | None = None,
    ):
        self.weights = dict(weights) if weights else dict(DEFAULT_RISK_WEIGHTS)
        self._max_weight = max(self.weights.values()) if self.weights else 1.0
        if group_trend_thresholds:
            self._trend_increasing = group_trend_thresholds[0]
            self._trend_decreasing = group_trend_thresholds[1]
        else:
            self._trend_increasing = 0.5
            self._trend_decreasing = 0.3

    def score_asset(
        self,
        asset_id: str,
        equipment_health_score: float,
        weather_beaufort: int,
        crew_fatigue_score: float,
        compliance_score: float,
        route_risk_score: float,
        asset_name: str = "",
    ) -> OperationalRiskResult:
        display_name = asset_name or asset_id[:16]

        eq = equipment_health_score
        beaufort = weather_beaufort
        crew = crew_fatigue_score
        comp = compliance_score
        route = route_risk_score

        weather_risk = min(1.0, (beaufort / 12.0) ** 1.5)
        equipment_risk = 1.0 - eq
        crew_risk = 1.0 - crew
        compliance_risk = 1.0 - comp
        route_risk = route

        dimensions = [
            self._dimension("equipment_health", equipment_risk,
                            f"Equipment health degradation (Health score: {eq:.2f})"),
            self._dimension("weather_exposure", weather_risk,
                            f"Beaufort {beaufort} sea state (Weather severity: {1 - weather_risk:.2f})"),
            self._dimension("crew_fatigue", crew_risk,
                            f"Crew rest and fatigue level (Fatigue score: {crew:.2f})"),
            self._dimension("compliance_status", compliance_risk,
                            f"Regulatory compliance status (Compliance score: {comp:.2f})"),
            self._dimension("route_risk", route_risk,
                            f"Route risk assessment (Route safety: {1 - route_risk:.2f})"),
        ]

        overall = min(1.0, sum(d.weighted_score for d in dimensions))
        risk_level = self._risk_level(overall)

        top_factors = sorted(dimensions, key=lambda d: d.score, reverse=True)
        top_risk_factors = [
            f"{d.dimension.replace('_', ' ').title()} ({d.status}): {d.description}"
            for d in top_factors[:3] if d.score > 0.3
        ]

        actions = self._recommend_actions(dimensions, overall)
        matrix = self._build_risk_matrix(dimensions)

        notes = (
            f"Asset {display_name}: overall risk {overall:.3f} ({risk_level}). "
            f"{len([d for d in dimensions if d.score > 0.5])} dimensions above 0.5 threshold."
        )

        return OperationalRiskResult(
            asset_id=asset_id,
            asset_name=display_name,
            assessment_timestamp=datetime.now(timezone.utc).isoformat(),
            overall_risk_score=round(overall, 4),
            risk_level=risk_level,
            risk_dimensions=dimensions,
            risk_matrix=matrix,
            trend_7d=None,
            top_risk_factors=top_risk_factors,
            recommended_actions=actions,
            notes=notes,
        )

    def score_group(
        self,
        results: list[OperationalRiskResult],
    ) -> GroupRiskSummary:
        asset_scores = []
        for result in results:
            asset_scores.append({
                "asset_id": result.asset_id,
                "asset_name": result.asset_name,
                "risk_score": result.overall_risk_score,
                "risk_level": result.risk_level,
            })

        if not asset_scores:
            return GroupRiskSummary(
                assessment_timestamp=datetime.now(timezone.utc).isoformat(),
                group_average_risk=0.0,
                highest_risk_asset_id="",
                highest_risk_asset_name="",
                highest_risk_score=0.0,
                assets_at_high_risk=0,
                assets_at_medium_risk=0,
                assets_at_low_risk=0,
                total_assets=0,
                asset_risk_scores=[],
                group_trend="stable",
                notes="No assets to assess.",
            )

        asset_scores.sort(key=lambda v: float(v["risk_score"]), reverse=True)
        avg = sum(float(v["risk_score"]) for v in asset_scores) / len(asset_scores)
        high = sum(1 for v in asset_scores if v["risk_level"] in ("critical", "high"))
        medium = sum(1 for v in asset_scores if v["risk_level"] == "medium")
        low = len(asset_scores) - high - medium

        if avg > self._trend_increasing:
            trend = "increasing"
        elif avg < self._trend_decreasing:
            trend = "decreasing"
        else:
            trend = "stable"

        return GroupRiskSummary(
            assessment_timestamp=datetime.now(timezone.utc).isoformat(),
            group_average_risk=round(avg, 4),
            highest_risk_asset_id=str(asset_scores[0]["asset_id"]),
            highest_risk_asset_name=str(asset_scores[0]["asset_name"]),
            highest_risk_score=float(asset_scores[0]["risk_score"]),
            assets_at_high_risk=high,
            assets_at_medium_risk=medium,
            assets_at_low_risk=low,
            total_assets=len(asset_scores),
            asset_risk_scores=asset_scores,
            group_trend=trend,
            notes=f"Group of {len(asset_scores)} assets; avg risk {avg:.3f}.",
        )

    def _dimension(
        self,
        name: str,
        score: float,
        description: str,
    ) -> RiskDimension:
        weight = self.weights.get(name, 0.2)
        status = (
            "critical" if score >= 0.75
            else "high" if score >= 0.50
            else "medium" if score >= 0.25
            else "low"
        )
        return RiskDimension(
            dimension=name,
            score=round(score, 4),
            weight=weight,
            weighted_score=round(score * weight, 4),
            status=status,
            description=description,
            trend="stable",
            trend_delta=0.0,
        )

    def _risk_level(self, score: float) -> str:
        if score >= 0.75:
            return "critical"
        elif score >= 0.50:
            return "high"
        elif score >= 0.25:
            return "medium"
        return "low"

    def _recommend_actions(self, dimensions: list[RiskDimension], overall: float) -> list[str]:
        actions = []
        for d in sorted(dimensions, key=lambda x: x.score, reverse=True):
            if d.score >= 0.70:
                actions.append(f"URGENT: Address {d.dimension.replace('_', ' ')} - {d.status} risk ({d.score:.2f})")
            elif d.score >= 0.40:
                actions.append(f"Monitor {d.dimension.replace('_', ' ')} - escalating to {d.status} ({d.score:.2f})")
        if not actions:
            actions.append("All dimensions within acceptable range; maintain routine monitoring")
        return actions[:5]

    def _build_risk_matrix(self, dimensions: list[RiskDimension]) -> list[RiskMatrixCell]:
        matrix = []
        for likelihood, impact, category, color in LIKELIHOOD_IMPACT:
            applicable = [
                d.dimension for d in dimensions
                if self._dimension_in_cell(d, likelihood, impact)
            ]
            matrix.append(RiskMatrixCell(
                likelihood=likelihood,
                impact=impact,
                risk_category=category,
                color=color,
                applicable_dimensions=applicable,
            ))
        return matrix

    def _dimension_in_cell(self, d: RiskDimension, likelihood: str, impact: str) -> bool:
        likelihood_score = d.score
        impact_score = d.score * d.weight / self._max_weight if self._max_weight > 0 else 0.5

        likelihood_map = {
            "rare": (0.0, 0.15),
            "unlikely": (0.15, 0.30),
            "possible": (0.30, 0.55),
            "likely": (0.55, 0.75),
            "almost_certain": (0.75, 1.01),
        }
        impact_map = {
            "negligible": (0.0, 0.15),
            "minor": (0.15, 0.30),
            "moderate": (0.30, 0.55),
            "major": (0.55, 0.75),
            "catastrophic": (0.75, 1.01),
        }
        l_lo, l_hi = likelihood_map.get(likelihood, (0, 1))
        i_lo, i_hi = impact_map.get(impact, (0, 1))
        return (l_lo <= likelihood_score < l_hi) and (i_lo <= impact_score < i_hi)
