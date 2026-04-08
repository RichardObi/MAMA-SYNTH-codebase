#  Copyright 2025 mama-sia-eval contributors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Borda-style rank aggregation for the MAMA-SYNTH challenge.

The ranking procedure is hierarchical:

1. For each metric in a metric group:
   a. Compute metric value per test case.
   b. Average metric values across test cases to get a submission score.
   c. Rank all submissions by this metric score.

2. For each metric group (CLF, SEG, ROI, FULL):
   a. Average metric-level ranks within the group.

3. Final ranking:
   a. Average the four group-level ranks.
   b. Rank submissions by this average.

Tie-breaking order: ROI → CLF → SEG → FULL.

Reference: MAMA-SYNTH Challenge §Assessment Methods, Ranking method.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Metrics where lower is better (need ascending ranking)
LOWER_IS_BETTER = {
    "mse",
    "mae",
    "nmse",
    "lpips",
    "frd",
    "hd95",
}

# Metrics where higher is better (need descending ranking)
HIGHER_IS_BETTER = {
    "psnr",
    "ssim",
    "ncc",
    "dice",
    "auroc",
    "balanced_accuracy",
    "auroc_tnbc",
    "balanced_accuracy_tnbc",
    "auroc_luminal",
    "balanced_accuracy_luminal",
}

# Metric groups and their constituent metrics
METRIC_GROUPS = {
    "clf": [
        "auroc_tnbc",
        "balanced_accuracy_tnbc",
        "auroc_luminal",
        "balanced_accuracy_luminal",
    ],
    "seg": ["dice", "hd95"],
    "roi": ["mse_roi", "lpips_roi", "frd_roi"],
    "full": ["mse_full", "lpips_full", "frd_full"],
}

# Group weights (equal 25% each)
GROUP_WEIGHTS = {"clf": 0.25, "seg": 0.25, "roi": 0.25, "full": 0.25}

# Tie-breaking priority order
TIEBREAK_ORDER = ["roi", "clf", "seg", "full"]


def rank_submissions(
    submission_scores: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Rank submissions using Borda-style hierarchical rank aggregation.

    Args:
        submission_scores: Mapping from submission_id to a dictionary of
            metric_name → aggregated (mean) score across test cases.
            Metric names should match those defined in METRIC_GROUPS.
            Example::

                {
                    "team_a": {"auroc_tnbc": 0.85, "dice": 0.72, ...},
                    "team_b": {"auroc_tnbc": 0.90, "dice": 0.68, ...},
                }

    Returns:
        Sorted list of dictionaries, each containing:
            - submission_id
            - overall_rank
            - overall_avg_rank
            - group_ranks (dict of group → avg rank)
            - metric_ranks (dict of metric → rank)
    """
    submission_ids = list(submission_scores.keys())
    n = len(submission_ids)

    if n == 0:
        return []

    if n == 1:
        return [{
            "submission_id": submission_ids[0],
            "overall_rank": 1,
            "overall_avg_rank": 1.0,
            "group_ranks": {g: 1.0 for g in METRIC_GROUPS},
            "metric_ranks": {m: 1 for group_metrics in METRIC_GROUPS.values() for m in group_metrics},
        }]

    # Step 1: Rank submissions per metric
    metric_ranks: dict[str, dict[str, int]] = {}  # metric -> {sub_id: rank}

    all_metrics = set()
    for group_metrics in METRIC_GROUPS.values():
        all_metrics.update(group_metrics)

    for metric in all_metrics:
        scores = []
        for sid in submission_ids:
            score = submission_scores.get(sid, {}).get(metric, None)
            scores.append(score)

        # Determine higher/lower is better
        base_metric = metric.replace("_roi", "").replace("_full", "")
        ascending = base_metric in LOWER_IS_BETTER

        # Rank (handle missing by assigning worst rank)
        ranks = _rank_values(scores, ascending=ascending)
        metric_ranks[metric] = {sid: r for sid, r in zip(submission_ids, ranks)}

    # Step 2: Compute group-level ranks
    group_ranks: dict[str, dict[str, float]] = {}

    for group, metrics in METRIC_GROUPS.items():
        group_avg_ranks: dict[str, float] = {}
        for sid in submission_ids:
            ranks_in_group = [
                metric_ranks.get(m, {}).get(sid, n)
                for m in metrics
                if m in metric_ranks
            ]
            if ranks_in_group:
                group_avg_ranks[sid] = float(np.mean(ranks_in_group))
            else:
                group_avg_ranks[sid] = float(n)

        group_ranks[group] = group_avg_ranks

    # Step 3: Compute overall average rank
    overall_avg: dict[str, float] = {}
    for sid in submission_ids:
        avg = float(np.mean([
            group_ranks[g].get(sid, n)
            for g in METRIC_GROUPS
        ]))
        overall_avg[sid] = avg

    # Step 4: Sort and assign final ranks with tie-breaking
    sorted_sids = _sort_with_tiebreak(
        submission_ids, overall_avg, group_ranks, TIEBREAK_ORDER
    )

    results = []
    for rank_idx, sid in enumerate(sorted_sids, start=1):
        results.append({
            "submission_id": sid,
            "overall_rank": rank_idx,
            "overall_avg_rank": overall_avg[sid],
            "group_ranks": {g: group_ranks[g][sid] for g in METRIC_GROUPS},
            "metric_ranks": {
                m: metric_ranks.get(m, {}).get(sid, n)
                for group_metrics in METRIC_GROUPS.values()
                for m in group_metrics
            },
        })

    return results


def _rank_values(
    values: list[float | None],
    ascending: bool = True,
) -> list[int]:
    """Rank a list of values. Lower rank = better.

    Args:
        values: List of metric scores (None for missing submissions).
        ascending: If True, smaller values get better (lower) ranks.

    Returns:
        List of integer ranks (1-indexed).
    """
    n = len(values)

    # Replace None with worst possible value
    worst = float("inf") if ascending else float("-inf")
    clean_values = [v if v is not None else worst for v in values]

    # Sort indices
    indices = list(range(n))
    indices.sort(key=lambda i: clean_values[i], reverse=not ascending)

    # Assign ranks
    ranks = [0] * n
    for rank_idx, i in enumerate(indices, start=1):
        ranks[i] = rank_idx

    return ranks


def _sort_with_tiebreak(
    submission_ids: list[str],
    overall_avg: dict[str, float],
    group_ranks: dict[str, dict[str, float]],
    tiebreak_order: list[str],
) -> list[str]:
    """Sort submissions with tie-breaking on group ranks.

    Args:
        submission_ids: List of submission identifiers.
        overall_avg: Overall average rank per submission.
        group_ranks: Group-level rank per submission per group.
        tiebreak_order: Priority order for tie-breaking.

    Returns:
        Sorted list of submission IDs (best first).
    """
    def sort_key(sid: str) -> tuple:
        key = [overall_avg[sid]]
        for g in tiebreak_order:
            key.append(group_ranks.get(g, {}).get(sid, float("inf")))
        return tuple(key)

    return sorted(submission_ids, key=sort_key)
