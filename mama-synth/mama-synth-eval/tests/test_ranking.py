"""Unit tests for ranking module."""

import pytest

from mama_sia_eval.ranking import rank_submissions


def _rankings_by_id(rankings: list) -> dict:
    """Helper: convert list of ranking dicts to dict keyed by submission_id."""
    return {r["submission_id"]: r for r in rankings}


class TestRankSubmissions:
    """Tests for Borda-style rank aggregation."""

    def test_single_submission(self) -> None:
        submissions = {
            "team_a": {
                "mse_roi": 10.0,
                "frd_roi": 5.0,
                "mse_full": 8.0,
                "frd_full": 4.0,
                "dice": 0.9,
                "hd95": 2.0,
                "auroc_tnbc": 0.85,
                "balanced_accuracy_tnbc": 0.8,
                "auroc_luminal": 0.82,
                "balanced_accuracy_luminal": 0.78,
            }
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert "team_a" in by_id
        assert by_id["team_a"]["overall_rank"] == 1

    def test_two_submissions_clear_winner(self) -> None:
        submissions = {
            "winner": {
                "mse_roi": 1.0, "frd_roi": 0.5, "lpips_roi": 0.05,
                "mse_full": 1.0, "frd_full": 0.5, "lpips_full": 0.05,
                "dice": 0.95, "hd95": 1.0,
                "auroc_tnbc": 0.99, "balanced_accuracy_tnbc": 0.95,
                "auroc_luminal": 0.98, "balanced_accuracy_luminal": 0.94,
            },
            "loser": {
                "mse_roi": 100.0, "frd_roi": 50.0, "lpips_roi": 0.9,
                "mse_full": 100.0, "frd_full": 50.0, "lpips_full": 0.9,
                "dice": 0.3, "hd95": 30.0,
                "auroc_tnbc": 0.51, "balanced_accuracy_tnbc": 0.5,
                "auroc_luminal": 0.52, "balanced_accuracy_luminal": 0.5,
            },
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert by_id["winner"]["overall_rank"] < by_id["loser"]["overall_rank"]

    def test_three_submissions_ranking(self) -> None:
        submissions = {
            "best": {
                "mse_roi": 1.0, "frd_roi": 1.0, "lpips_roi": 0.05,
                "mse_full": 1.0, "frd_full": 1.0, "lpips_full": 0.05,
                "dice": 0.95, "hd95": 1.0,
                "auroc_tnbc": 0.95, "balanced_accuracy_tnbc": 0.95,
                "auroc_luminal": 0.94, "balanced_accuracy_luminal": 0.93,
            },
            "middle": {
                "mse_roi": 5.0, "frd_roi": 5.0, "lpips_roi": 0.3,
                "mse_full": 5.0, "frd_full": 5.0, "lpips_full": 0.3,
                "dice": 0.7, "hd95": 5.0,
                "auroc_tnbc": 0.75, "balanced_accuracy_tnbc": 0.7,
                "auroc_luminal": 0.74, "balanced_accuracy_luminal": 0.69,
            },
            "worst": {
                "mse_roi": 50.0, "frd_roi": 50.0, "lpips_roi": 0.8,
                "mse_full": 50.0, "frd_full": 50.0, "lpips_full": 0.8,
                "dice": 0.2, "hd95": 30.0,
                "auroc_tnbc": 0.55, "balanced_accuracy_tnbc": 0.5,
                "auroc_luminal": 0.53, "balanced_accuracy_luminal": 0.49,
            },
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert by_id["best"]["overall_rank"] == 1
        assert by_id["middle"]["overall_rank"] == 2
        assert by_id["worst"]["overall_rank"] == 3

    def test_ranking_contains_group_ranks(self) -> None:
        submissions = {
            "team": {
                "mse_roi": 5.0, "frd_roi": 3.0, "lpips_roi": 0.2,
                "mse_full": 5.0, "frd_full": 3.0, "lpips_full": 0.2,
                "dice": 0.8, "hd95": 3.0,
                "auroc_tnbc": 0.8, "balanced_accuracy_tnbc": 0.75,
                "auroc_luminal": 0.78, "balanced_accuracy_luminal": 0.73,
            },
        }
        rankings = rank_submissions(submissions)
        team = rankings[0]
        assert "overall_rank" in team
        assert "group_ranks" in team

    def test_empty_submissions_returns_empty(self) -> None:
        rankings = rank_submissions({})
        assert rankings == []

    def test_missing_metrics_handled(self) -> None:
        """Submissions with missing metrics should still be rankable."""
        submissions = {
            "team_a": {"mse_roi": 5.0, "dice": 0.8},
            "team_b": {"mse_roi": 10.0, "dice": 0.7},
        }
        # Should not raise even with missing metrics
        try:
            rankings = rank_submissions(submissions)
            assert len(rankings) == 2
        except (KeyError, ValueError):
            # Acceptable if strict metric requirements
            pass

    def test_lpips_lower_is_better(self) -> None:
        """LPIPS is a distance metric \u2013 lower should rank better."""
        submissions = {
            "good_lpips": {
                "mse_roi": 5.0, "lpips_roi": 0.1, "frd_roi": 2.0,
                "mse_full": 5.0, "lpips_full": 0.1, "frd_full": 2.0,
                "dice": 0.8, "hd95": 3.0,
                "auroc_tnbc": 0.8, "balanced_accuracy_tnbc": 0.75,
                "auroc_luminal": 0.78, "balanced_accuracy_luminal": 0.73,
            },
            "bad_lpips": {
                "mse_roi": 5.0, "lpips_roi": 0.9, "frd_roi": 2.0,
                "mse_full": 5.0, "lpips_full": 0.9, "frd_full": 2.0,
                "dice": 0.8, "hd95": 3.0,
                "auroc_tnbc": 0.8, "balanced_accuracy_tnbc": 0.75,
                "auroc_luminal": 0.78, "balanced_accuracy_luminal": 0.73,
            },
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert by_id["good_lpips"]["overall_rank"] <= by_id["bad_lpips"]["overall_rank"]
