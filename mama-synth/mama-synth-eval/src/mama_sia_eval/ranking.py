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

All **8 challenge metrics** are ranked **equally** (flat Borda count).

Procedure
---------
1. For each of the 8 metrics, rank all submissions by their mean score
   across test cases.
2. Average the 8 per-metric ranks for each submission.
3. Rank by the average — lowest average rank wins.

The 8 metrics are organised into 4 tasks for display purposes only:

  Task 1 — Full Image:  MSE (↓), LPIPS (↓)
  Task 2 — Tumor ROI:   SSIM (↑), FRD (↓)
  Task 3 — Classification: AUROC luminal (↑), AUROC TNBC (↑)
  Task 4 — Segmentation: Dice (↑), Hausdorff95 (↓)

Reference: MAMA-SYNTH Challenge — updated assessment method (GC-aligned).
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Metrics where lower is better (need ascending ranking)
LOWER_IS_BETTER = {
    "mse",
    "mse_full_image",
    "mae",
    "nmse",
    "lpips",
    "lpips_full_image",
    "frd",
    "frd_roi",
    "hd95",
    "hausdorff95",
}

# Metrics where higher is better (need descending ranking)
HIGHER_IS_BETTER = {
    "psnr",
    "ssim",
    "ssim_roi",
    "ncc",
    "dice",
    "auroc",
    "auroc_tnbc",
    "auroc_luminal",
    "balanced_accuracy",
}

# The 8 official challenge metrics — all equally weighted
CHALLENGE_METRICS: list[str] = [
    "mse_full_image",
    "lpips_full_image",
    "ssim_roi",
    "frd_roi",
    "auroc_luminal",
    "auroc_tnbc",
    "dice",
    "hausdorff95",
]

# Informational grouping (for display/reporting only, not for weighting)
METRIC_TASKS: dict[str, list[str]] = {
    "full_image": ["mse_full_image", "lpips_full_image"],
    "roi": ["ssim_roi", "frd_roi"],
    "classification": ["auroc_luminal", "auroc_tnbc"],
    "segmentation": ["dice", "hausdorff95"],
}


def rank_submissions(
    submission_scores: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Rank submissions using flat Borda-count over 8 equal metrics.

    Args:
        submission_scores: Mapping from submission_id to a dictionary of
            metric_name → aggregated (mean) score across test cases.
            Metric names should match those defined in ``CHALLENGE_METRICS``.
            Example::

                {
                    "team_a": {"mse_full_image": 0.02, "dice": 0.72, ...},
                    "team_b": {"mse_full_image": 0.03, "dice": 0.68, ...},
                }

    Returns:
        Sorted list of dictionaries, each containing:
            - submission_id
            - overall_rank
            - overall_avg_rank
            - task_ranks (dict of task → avg rank within that task)
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
            "task_ranks": {t: 1.0 for t in METRIC_TASKS},
            "metric_ranks": {m: 1 for m in CHALLENGE_METRICS},
        }]

    # Step 1: Rank submissions per metric (all 8 equally weighted)
    metric_ranks: dict[str, dict[str, int]] = {}

    for metric in CHALLENGE_METRICS:
        scores = []
        for sid in submission_ids:
            score = submission_scores.get(sid, {}).get(metric, None)
            scores.append(score)

        ascending = metric in LOWER_IS_BETTER
        ranks = _rank_values(scores, ascending=ascending)
        metric_ranks[metric] = {sid: r for sid, r in zip(submission_ids, ranks)}

    # Step 2: Compute overall average rank (flat — all 8 metrics equal)
    overall_avg: dict[str, float] = {}
    for sid in submission_ids:
        ranks_list = [
            metric_ranks.get(m, {}).get(sid, n)
            for m in CHALLENGE_METRICS
            if m in metric_ranks
        ]
        overall_avg[sid] = float(np.mean(ranks_list)) if ranks_list else float(n)

    # Step 3: Compute task-level ranks (informational only)
    task_ranks: dict[str, dict[str, float]] = {}
    for task, metrics in METRIC_TASKS.items():
        task_avg: dict[str, float] = {}
        for sid in submission_ids:
            ranks_in_task = [
                metric_ranks.get(m, {}).get(sid, n)
                for m in metrics
                if m in metric_ranks
            ]
            task_avg[sid] = float(np.mean(ranks_in_task)) if ranks_in_task else float(n)
        task_ranks[task] = task_avg

    # Step 4: Sort by overall average rank (ties broken by lowest avg arbitrarily)
    sorted_sids = sorted(submission_ids, key=lambda sid: overall_avg[sid])

    results = []
    for rank_idx, sid in enumerate(sorted_sids, start=1):
        results.append({
            "submission_id": sid,
            "overall_rank": rank_idx,
            "overall_avg_rank": overall_avg[sid],
            "task_ranks": {t: task_ranks[t][sid] for t in METRIC_TASKS},
            "metric_ranks": {
                m: metric_ranks.get(m, {}).get(sid, n)
                for m in CHALLENGE_METRICS
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
