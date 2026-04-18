"""Segmentation metrics: Dice coefficient and Hausdorff distance (HD95).

A callable ``segment_fn`` must be provided at init time.  It receives
a 2-D ``float64`` image (normalised to ``[0, 1]``) and returns a
binary ``bool`` mask of the same shape.

When no ``segment_fn`` is given the evaluator returns empty results.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

from .base import BaseEvaluator, Case, EvaluationResult


class SegmentationEvaluator(BaseEvaluator):
    """Dice and 95th-percentile Hausdorff between predicted and GT masks."""

    def __init__(
        self,
        segment_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.segment_fn = segment_fn

    # ------------------------------------------------------------------

    def evaluate(self, cases: list[Case]) -> EvaluationResult:
        if self.segment_fn is None:
            return EvaluationResult()

        per_case: dict[str, dict[str, float]] = {}
        for case in cases:
            if case.mask is None or not np.any(case.mask):
                continue
            try:
                pred_mask = self.segment_fn(case.prediction)
                if pred_mask is None or not np.any(pred_mask):
                    continue
                pred_mask = pred_mask.astype(bool)
                gt_mask = case.mask.astype(bool)

                dice = compute_dice(pred_mask, gt_mask)
                hd95 = compute_hausdorff_95(pred_mask, gt_mask)
                per_case[case.case_id] = {
                    "dice": dice,
                    "hausdorff_95": hd95,
                }
            except Exception:
                continue

        agg: dict[str, dict[str, float]] = {}
        dice_agg = self._aggregate_metric(per_case, "dice")
        if dice_agg:
            agg["dice"] = dice_agg
        hd_agg = self._aggregate_metric(per_case, "hausdorff_95")
        if hd_agg:
            agg["hausdorff_95"] = hd_agg

        return EvaluationResult(per_case=per_case, aggregates=agg)


# ======================================================================
# Metric implementations
# ======================================================================


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sørensen–Dice coefficient between two binary masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = int(np.logical_and(pred_b, gt_b).sum())
    total = int(pred_b.sum()) + int(gt_b.sum())
    if total == 0:
        return 1.0  # both empty → perfect agreement
    return float(2.0 * intersection / total)


def compute_hausdorff_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff distance between mask surfaces."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    pred_surface = pred_b ^ binary_erosion(pred_b, iterations=1)
    gt_surface = gt_b ^ binary_erosion(gt_b, iterations=1)

    pred_pts = np.argwhere(pred_surface)
    gt_pts = np.argwhere(gt_surface)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    # Directed distances in both directions
    d_pred_to_gt = cdist(pred_pts, gt_pts).min(axis=1)
    d_gt_to_pred = cdist(gt_pts, pred_pts).min(axis=1)
    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, 95))
