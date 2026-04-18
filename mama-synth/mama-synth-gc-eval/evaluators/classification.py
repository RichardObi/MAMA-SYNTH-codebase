"""Classification metrics: AUROC for contrast and tumour-ROI tasks.

Both metrics are **aggregate-only** (no meaningful per-case result).

* **AUROC contrast** – how well a pre-trained classifier can
  distinguish synthetic post-contrast images from real pre-contrast
  images based on radiomic features.
* **AUROC tumour-ROI** – how well a pre-trained classifier can
  distinguish the tumour region from a horizontally-mirrored
  contralateral region in the synthetic image.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from .base import BaseEvaluator, Case, EvaluationResult
from .roi_metrics import extract_radiomic_features


class ClassificationEvaluator(BaseEvaluator):
    """AUROC for contrast and tumour-ROI classification."""

    def __init__(
        self,
        contrast_model: Optional[Path] = None,
        tumor_roi_model: Optional[Path] = None,
    ) -> None:
        self.contrast_clf = _load_pkl(contrast_model)
        self.tumor_roi_clf = _load_pkl(tumor_roi_model)

    # ------------------------------------------------------------------

    def evaluate(self, cases: list[Case]) -> EvaluationResult:
        agg: dict[str, dict[str, float]] = {}

        if self.contrast_clf is not None:
            auroc = self._auroc_contrast(cases)
            if auroc is not None:
                agg["auroc_contrast"] = {"mean": auroc, "std": 0.0}

        if self.tumor_roi_clf is not None:
            auroc = self._auroc_tumor_roi(cases)
            if auroc is not None:
                agg["auroc_tumor_roi"] = {"mean": auroc, "std": 0.0}

        return EvaluationResult(per_case={}, aggregates=agg)

    # ---- contrast ----------------------------------------------------

    def _auroc_contrast(self, cases: list[Case]) -> Optional[float]:
        """AUROC: synthetic-post (label 1) vs real-precontrast (label 0)."""
        feats: list[np.ndarray] = []
        labels: list[int] = []

        for case in cases:
            if case.precontrast is None:
                continue
            try:
                whole = np.ones(case.prediction.shape, dtype=bool)
                sf = extract_radiomic_features(case.prediction, whole)
                pf = extract_radiomic_features(case.precontrast, whole)
                if sf.size == 0 or pf.size == 0:
                    continue
                if sf.shape != pf.shape:
                    continue
                feats.extend([sf, pf])
                labels.extend([1, 0])
            except Exception as exc:
                print(
                    f"WARNING: contrast feature extraction failed "
                    f"for {case.case_id}: {exc}",
                    file=sys.stderr,
                )

        if len(feats) < 4:
            return None

        X = np.nan_to_num(np.vstack(feats), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array(labels, dtype=np.int64)
        try:
            probs = self.contrast_clf.predict_proba(X)[:, 1]  # type: ignore[union-attr]
            return float(roc_auc_score(y, probs))
        except Exception as exc:
            print(f"WARNING: contrast AUROC failed: {exc}", file=sys.stderr)
            return None

    # ---- tumour ROI --------------------------------------------------

    def _auroc_tumor_roi(self, cases: list[Case]) -> Optional[float]:
        """AUROC: tumour ROI (label 1) vs mirrored ROI (label 0)."""
        feats: list[np.ndarray] = []
        labels: list[int] = []

        for case in cases:
            if case.mask is None or not np.any(case.mask):
                continue
            mirrored = np.fliplr(case.mask)
            # Skip if mirrored mask is empty or fully overlaps
            if not np.any(mirrored) or np.array_equal(case.mask, mirrored):
                continue
            try:
                tf = extract_radiomic_features(case.prediction, case.mask)
                mf = extract_radiomic_features(case.prediction, mirrored)
                if tf.size == 0 or mf.size == 0:
                    continue
                if tf.shape != mf.shape:
                    continue
                feats.extend([tf, mf])
                labels.extend([1, 0])
            except Exception as exc:
                print(
                    f"WARNING: tumour-ROI feature extraction failed "
                    f"for {case.case_id}: {exc}",
                    file=sys.stderr,
                )

        if len(feats) < 4:
            return None

        X = np.nan_to_num(np.vstack(feats), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array(labels, dtype=np.int64)
        try:
            probs = self.tumor_roi_clf.predict_proba(X)[:, 1]  # type: ignore[union-attr]
            return float(roc_auc_score(y, probs))
        except Exception as exc:
            print(
                f"WARNING: tumour-ROI AUROC failed: {exc}", file=sys.stderr
            )
            return None


# ---- helpers ---------------------------------------------------------


def _load_pkl(path: Optional[Path]) -> object | None:
    if path is None or not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301
