"""Tests for operational risk scorer."""

import pytest

from anomalykit.risk.operational_risk_scorer import (
    OperationalRiskScorer,
    OperationalRiskResult,
    GroupRiskSummary,
)

EXPLICIT_ARGS = dict(
    equipment_health_score=0.9,
    weather_beaufort=3,
    crew_fatigue_score=0.85,
    compliance_score=0.95,
    route_risk_score=0.1,
)


class TestOperationalRiskScorer:
    def test_score_asset_with_explicit_values(self):
        scorer = OperationalRiskScorer()
        result = scorer.score_asset(
            asset_id="pump-001",
            asset_name="Main Pump A",
            **EXPLICIT_ARGS,
        )

        assert isinstance(result, OperationalRiskResult)
        assert result.asset_id == "pump-001"
        assert result.asset_name == "Main Pump A"
        assert 0 <= result.overall_risk_score <= 1
        assert result.risk_level in ("low", "medium", "high", "critical")
        assert len(result.risk_dimensions) == 5

    def test_score_asset_deterministic(self):
        scorer = OperationalRiskScorer()
        r1 = scorer.score_asset(asset_id="x-123", **EXPLICIT_ARGS)
        r2 = scorer.score_asset(asset_id="x-123", **EXPLICIT_ARGS)
        assert r1.overall_risk_score == r2.overall_risk_score

    def test_risk_dimensions_sum(self):
        scorer = OperationalRiskScorer()
        result = scorer.score_asset(
            asset_id="test",
            equipment_health_score=0.5,
            weather_beaufort=6,
            crew_fatigue_score=0.7,
            compliance_score=0.8,
            route_risk_score=0.3,
        )
        weighted_sum = sum(d.weighted_score for d in result.risk_dimensions)
        assert abs(result.overall_risk_score - min(1.0, weighted_sum)) < 0.01

    def test_risk_matrix_has_25_cells(self):
        scorer = OperationalRiskScorer()
        result = scorer.score_asset(asset_id="m1", **EXPLICIT_ARGS)
        assert len(result.risk_matrix) == 25

    def test_trend_7d_is_none(self):
        scorer = OperationalRiskScorer()
        result = scorer.score_asset(asset_id="t1", **EXPLICIT_ARGS)
        assert result.trend_7d is None

    def test_recommended_actions_not_empty(self):
        scorer = OperationalRiskScorer()
        result = scorer.score_asset(asset_id="a1", **EXPLICIT_ARGS)
        assert len(result.recommended_actions) > 0

    def test_score_group(self):
        scorer = OperationalRiskScorer()
        results = [
            scorer.score_asset(asset_id="a1", asset_name="Asset One", **EXPLICIT_ARGS),
            scorer.score_asset(asset_id="a2", asset_name="Asset Two", **EXPLICIT_ARGS),
            scorer.score_asset(asset_id="a3", asset_name="Asset Three", **EXPLICIT_ARGS),
        ]
        summary = scorer.score_group(results)

        assert isinstance(summary, GroupRiskSummary)
        assert summary.total_assets == 3
        assert summary.group_average_risk > 0

    def test_score_group_empty(self):
        scorer = OperationalRiskScorer()
        summary = scorer.score_group([])
        assert summary.total_assets == 0
        assert summary.group_average_risk == 0.0

    def test_high_risk_all_bad(self):
        scorer = OperationalRiskScorer()
        result = scorer.score_asset(
            asset_id="bad",
            equipment_health_score=0.1,
            weather_beaufort=11,
            crew_fatigue_score=0.1,
            compliance_score=0.1,
            route_risk_score=0.9,
        )
        assert result.risk_level in ("high", "critical")
        assert result.overall_risk_score > 0.5

    def test_low_risk_all_good(self):
        scorer = OperationalRiskScorer()
        result = scorer.score_asset(
            asset_id="good",
            equipment_health_score=0.99,
            weather_beaufort=1,
            crew_fatigue_score=0.99,
            compliance_score=0.99,
            route_risk_score=0.01,
        )
        assert result.risk_level == "low"
        assert result.overall_risk_score < 0.25

    def test_requires_explicit_inputs(self):
        scorer = OperationalRiskScorer()
        with pytest.raises(TypeError):
            scorer.score_asset(asset_id="no-defaults")
