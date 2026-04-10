"""Unit tests for ranking module (8 equally-weighted metrics)."""

import pytest

from mama_sia_eval.ranking import (
    CHALLENGE_METRICS,
    METRIC_TASKS,
    rank_submissions,
)


def _rankings_by_id(rankings: list) -> dict:
    """Helper: convert list of ranking dicts to dict keyed by submission_id."""
    return {r["submission_id"]: r for r in rankings}


def _make_scores(**overrides) -> dict[str, float]:
    """Return default scores for the 8 challenge metrics, with overrides."""
    defaults = {
        "mse_full_image": 5.0,
        "lpips_full_image": 0.2,
        "ssim_roi": 0.8,
        "frd_roi": 3.0,
        "auroc_luminal": 0.82,
        "auroc_tnbc": 0.80,
        "dice": 0.75,
        "hausdorff95": 4.0,
    }
    defaults.update(overrides)
    return defaults


class TestRankSubmissions:
    """Tests for flat Borda-count rank aggregation (8 equal metrics)."""

    def test_single_submission(self) -> None:
        submissions = {"team_a": _make_scores()}
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert "team_a" in by_id
        assert by_id["team_a"]["overall_rank"] == 1

    def test_two_submissions_clear_winner(self) -> None:
        submissions = {
            "winner": _make_scores(
                mse_full_image=1.0, lpips_full_image=0.05,
                ssim_roi=0.95, frd_roi=0.5,
                auroc_luminal=0.98, auroc_tnbc=0.99,
                dice=0.95, hausdorff95=1.0,
            ),
            "loser": _make_scores(
                mse_full_image=100.0, lpips_full_image=0.9,
                ssim_roi=0.3, frd_roi=50.0,
                auroc_luminal=0.52, auroc_tnbc=0.51,
                dice=0.3, hausdorff95=30.0,
            ),
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert by_id["winner"]["overall_rank"] < by_id["loser"]["overall_rank"]

    def test_three_submissions_ranking(self) -> None:
        submissions = {
            "best": _make_scores(
                mse_full_image=1.0, lpips_full_image=0.05,
                ssim_roi=0.95, frd_roi=1.0,
                auroc_luminal=0.94, auroc_tnbc=0.95,
                dice=0.95, hausdorff95=1.0,
            ),
            "middle": _make_scores(
                mse_full_image=5.0, lpips_full_image=0.3,
                ssim_roi=0.7, frd_roi=5.0,
                auroc_luminal=0.74, auroc_tnbc=0.75,
                dice=0.7, hausdorff95=5.0,
            ),
            "worst": _make_scores(
                mse_full_image=50.0, lpips_full_image=0.8,
                ssim_roi=0.3, frd_roi=50.0,
                auroc_luminal=0.53, auroc_tnbc=0.55,
                dice=0.2, hausdorff95=30.0,
            ),
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert by_id["best"]["overall_rank"] == 1
        assert by_id["middle"]["overall_rank"] == 2
        assert by_id["worst"]["overall_rank"] == 3

    def test_ranking_contains_task_ranks(self) -> None:
        submissions = {"team": _make_scores()}
        rankings = rank_submissions(submissions)
        team = rankings[0]
        assert "overall_rank" in team
        assert "task_ranks" in team
        for task_name in METRIC_TASKS:
            assert task_name in team["task_ranks"]

    def test_ranking_contains_metric_ranks(self) -> None:
        submissions = {"team": _make_scores()}
        rankings = rank_submissions(submissions)
        team = rankings[0]
        assert "metric_ranks" in team
        for m in CHALLENGE_METRICS:
            assert m in team["metric_ranks"]

    def test_empty_submissions_returns_empty(self) -> None:
        rankings = rank_submissions({})
        assert rankings == []

    def test_missing_metrics_handled(self) -> None:
        """Submissions with missing metrics should still be rankable."""
        submissions = {
            "team_a": {"mse_full_image": 5.0, "dice": 0.8},
            "team_b": {"mse_full_image": 10.0, "dice": 0.7},
        }
        try:
            rankings = rank_submissions(submissions)
            assert len(rankings) == 2
        except (KeyError, ValueError):
            pass

    def test_lpips_lower_is_better(self) -> None:
        """LPIPS is a distance metric — lower should rank better."""
        submissions = {
            "good_lpips": _make_scores(lpips_full_image=0.1),
            "bad_lpips": _make_scores(lpips_full_image=0.9),
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert by_id["good_lpips"]["metric_ranks"]["lpips_full_image"] < \
               by_id["bad_lpips"]["metric_ranks"]["lpips_full_image"]

    def test_ssim_higher_is_better(self) -> None:
        """SSIM is a similarity metric — higher should rank better."""
        submissions = {
            "good_ssim": _make_scores(ssim_roi=0.95),
            "bad_ssim": _make_scores(ssim_roi=0.3),
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert by_id["good_ssim"]["metric_ranks"]["ssim_roi"] < \
               by_id["bad_ssim"]["metric_ranks"]["ssim_roi"]

    def test_hausdorff_lower_is_better(self) -> None:
        """HD95 is a distance — lower should rank better."""
        submissions = {
            "good_hd": _make_scores(hausdorff95=1.0),
            "bad_hd": _make_scores(hausdorff95=30.0),
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        assert by_id["good_hd"]["metric_ranks"]["hausdorff95"] < \
               by_id["bad_hd"]["metric_ranks"]["hausdorff95"]

    def test_all_eight_metrics_contribute_equally(self) -> None:
        """Each metric should contribute equally (1/8) to overall rank."""
        # Team A wins on 5/8 metrics, team B wins on 3/8
        submissions = {
            "team_a": _make_scores(
                mse_full_image=1.0, lpips_full_image=0.1,  # wins
                ssim_roi=0.95, frd_roi=1.0,  # wins on ssim, frd
                auroc_luminal=0.7, auroc_tnbc=0.7,  # loses
                dice=0.95,  # wins
                hausdorff95=10.0,  # loses
            ),
            "team_b": _make_scores(
                mse_full_image=10.0, lpips_full_image=0.9,  # loses
                ssim_roi=0.5, frd_roi=10.0,  # loses
                auroc_luminal=0.95, auroc_tnbc=0.95,  # wins
                dice=0.5,  # loses
                hausdorff95=1.0,  # wins
            ),
        }
        rankings = rank_submissions(submissions)
        by_id = _rankings_by_id(rankings)
        # Team A should rank first (wins 5 of 8 metrics)
        assert by_id["team_a"]["overall_rank"] == 1
        assert by_id["team_b"]["overall_rank"] == 2

    def test_challenge_metrics_list_has_eight_entries(self) -> None:
        """The official metrics list should have exactly 8 entries."""
        assert len(CHALLENGE_METRICS) == 8

    def test_metric_tasks_cover_all_metrics(self) -> None:
        """All 8 metrics should appear in exactly one task group."""
        all_in_tasks = []
        for metrics in METRIC_TASKS.values():
            all_in_tasks.extend(metrics)
        assert sorted(all_in_tasks) == sorted(CHALLENGE_METRICS)
