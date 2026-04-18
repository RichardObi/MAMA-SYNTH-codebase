"""Image-to-image metrics: MSE and LPIPS.

Both metrics operate on the full image (no masking).
LPIPS requires ``torch`` and ``lpips``; it is gracefully
skipped when these packages are unavailable.
"""

from __future__ import annotations

import sys

import numpy as np

from .base import BaseEvaluator, Case, EvaluationResult


class ImageMetricsEvaluator(BaseEvaluator):
    """Per-case MSE and LPIPS between prediction and ground truth."""

    def __init__(self) -> None:
        self._lpips_fn = None
        self._torch = None
        try:
            import torch
            import lpips  # type: ignore[import-untyped]

            self._torch = torch
            self._lpips_fn = lpips.LPIPS(net="alex", verbose=False)
            self._lpips_fn.eval()
        except Exception:  # pragma: no cover
            print(
                "WARNING: LPIPS unavailable (torch/lpips not installed), "
                "LPIPS metric will be skipped.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------

    def evaluate(self, cases: list[Case]) -> EvaluationResult:
        per_case: dict[str, dict[str, float]] = {}

        for case in cases:
            metrics: dict[str, float] = {}
            pred, gt = case.prediction, case.ground_truth
            metrics["mse"] = float(np.mean((pred - gt) ** 2))

            lpips_val = self._compute_lpips(pred, gt)
            if lpips_val is not None:
                metrics["lpips"] = lpips_val

            per_case[case.case_id] = metrics

        agg: dict[str, dict[str, float]] = {}
        agg["mse"] = self._aggregate_metric(per_case, "mse")
        lpips_agg = self._aggregate_metric(per_case, "lpips")
        if lpips_agg:
            agg["lpips"] = lpips_agg

        return EvaluationResult(per_case=per_case, aggregates=agg)

    # ------------------------------------------------------------------

    def _compute_lpips(
        self, pred: np.ndarray, gt: np.ndarray
    ) -> float | None:
        if self._lpips_fn is None or self._torch is None:
            return None
        try:
            torch = self._torch
            # LPIPS expects [B, C, H, W] in [-1, 1].  Repeat grey → 3-ch.
            p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
            g = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
            p = p.expand(-1, 3, -1, -1) * 2 - 1
            g = g.expand(-1, 3, -1, -1) * 2 - 1
            with torch.no_grad():
                return float(self._lpips_fn(p, g).item())
        except Exception:
            return None
