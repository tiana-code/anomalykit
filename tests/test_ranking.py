"""Tests for TOPSIS ranking."""

import pytest

from anomalykit.ranking.topsis_ranker import TopsisRanker, AssetCriteriaInput, RankingResult


class TestTopsisRanker:
    def _make_assets(self):
        return [
            AssetCriteriaInput(
                asset_id="asset-1",
                criteria={
                    "efficiency": 0.9,
                    "cost": 0.2,
                    "utilization": 0.85,
                },
            ),
            AssetCriteriaInput(
                asset_id="asset-2",
                criteria={
                    "efficiency": 0.5,
                    "cost": 0.8,
                    "utilization": 0.4,
                },
            ),
            AssetCriteriaInput(
                asset_id="asset-3",
                criteria={
                    "efficiency": 0.7,
                    "cost": 0.5,
                    "utilization": 0.6,
                },
            ),
        ]

    def test_rank_basic(self):
        ranker = TopsisRanker(
            benefit_criteria={"efficiency": True, "cost": False, "utilization": True},
        )
        assets = self._make_assets()
        result = ranker.rank(assets)

        assert isinstance(result, RankingResult)
        assert len(result.rankings) == 3
        assert result.rankings[0].rank == 1
        assert result.rankings[1].rank == 2
        assert result.rankings[2].rank == 3

    def test_best_asset_ranks_first(self):
        ranker = TopsisRanker(
            benefit_criteria={"efficiency": True, "cost": False, "utilization": True},
        )
        assets = self._make_assets()
        result = ranker.rank(assets)

        # Asset-1 has best efficiency and utilization, lowest cost
        assert result.rankings[0].asset_id == "asset-1"

    def test_empty_input(self):
        ranker = TopsisRanker()
        result = ranker.rank([])

        assert len(result.rankings) == 0
        assert result.average_score == 0.0

    def test_single_asset(self):
        ranker = TopsisRanker()
        result = ranker.rank([
            AssetCriteriaInput(asset_id="only", criteria={"x": 5.0}),
        ])
        assert len(result.rankings) == 1
        assert result.rankings[0].overall_score == 1.0

    def test_quartile_boundaries(self):
        ranker = TopsisRanker(
            benefit_criteria={"efficiency": True, "cost": False, "utilization": True},
        )
        assets = self._make_assets()
        result = ranker.rank(assets)

        assert "q1" in result.quartile_boundaries
        assert "q2" in result.quartile_boundaries
        assert "q3" in result.quartile_boundaries

    def test_custom_weights(self):
        ranker = TopsisRanker(
            benefit_criteria={"efficiency": True, "cost": False, "utilization": True},
        )
        assets = self._make_assets()

        result_default = ranker.rank(assets)
        result_custom = ranker.rank(assets, weights={"cost": 0.9, "efficiency": 0.05, "utilization": 0.05})

        # With heavy cost weight, asset-1 (lowest cost) should still win
        assert result_custom.rankings[0].asset_id == "asset-1"

    def test_rank_single(self):
        ranker = TopsisRanker(
            benefit_criteria={"efficiency": True, "cost": False, "utilization": True},
        )
        assets = self._make_assets()
        result = ranker.rank_single("asset-2", assets)

        assert result is not None
        assert result.asset_id == "asset-2"

    def test_rank_single_not_found(self):
        ranker = TopsisRanker()
        result = ranker.rank_single("nonexistent", self._make_assets())
        assert result is None

    def test_mismatched_criteria_raises(self):
        ranker = TopsisRanker()
        assets = [
            AssetCriteriaInput(asset_id="a1", criteria={"x": 1.0, "y": 2.0}),
            AssetCriteriaInput(asset_id="a2", criteria={"x": 1.0, "z": 3.0}),
        ]
        with pytest.raises(ValueError, match="same criteria"):
            ranker.rank(assets)

    def test_custom_underperformer_percentile(self):
        ranker = TopsisRanker(
            benefit_criteria={"efficiency": True, "cost": False, "utilization": True},
            underperformer_percentile=50.0,
        )
        assets = self._make_assets()
        result = ranker.rank(assets)
        assert len(result.underperformers) >= 1
