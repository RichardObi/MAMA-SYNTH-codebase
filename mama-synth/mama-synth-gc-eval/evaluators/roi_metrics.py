"""ROI-level metrics: SSIM within the tumour mask and FRD.

* **SSIM (tumour)** – per-case structural similarity computed on the
  full image and averaged within the tumour mask.
* **FRD** – Fréchet Radiomics Distance (aggregate-only).  Radiomic
  features are extracted from the tumour mask region of every
  prediction and every ground-truth image; the Fréchet distance
  between the two feature distributions is reported.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import SimpleITK as sitk
from scipy import linalg
from skimage.metrics import structural_similarity

from .base import BaseEvaluator, Case, EvaluationResult


class ROIMetricsEvaluator(BaseEvaluator):
    """SSIM within the tumour mask (per-case) and FRD (aggregate)."""

    def evaluate(self, cases: list[Case]) -> EvaluationResult:
        per_case: dict[str, dict[str, float]] = {}
        pred_features: list[np.ndarray] = []
        gt_features: list[np.ndarray] = []

        for case in cases:
            if case.mask is None or not np.any(case.mask):
                continue

            # ---- SSIM within mask ------------------------------------
            ssim_full, ssim_map = structural_similarity(
                case.prediction,
                case.ground_truth,
                data_range=1.0,
                full=True,
            )
            ssim_roi = float(np.mean(ssim_map[case.mask]))
            per_case[case.case_id] = {"ssim_tumor": ssim_roi}

            # ---- Radiomic features for FRD ---------------------------
            try:
                pf = extract_radiomic_features(case.prediction, case.mask)
                gf = extract_radiomic_features(case.ground_truth, case.mask)
                if (
                    pf.size > 0
                    and gf.size > 0
                    and pf.shape == gf.shape
                ):
                    pred_features.append(pf)
                    gt_features.append(gf)
            except Exception as exc:
                print(
                    f"WARNING: radiomic feature extraction failed "
                    f"for {case.case_id}: {exc}",
                    file=sys.stderr,
                )

        # ---- Aggregates ----------------------------------------------
        agg: dict[str, dict[str, float]] = {}
        ssim_agg = self._aggregate_metric(per_case, "ssim_tumor")
        if ssim_agg:
            agg["ssim_tumor"] = ssim_agg

        if len(pred_features) >= 2 and len(gt_features) >= 2:
            try:
                X_pred = np.vstack(pred_features)
                X_gt = np.vstack(gt_features)
                frd_val = frechet_distance(X_pred, X_gt)
                agg["frd"] = {"mean": frd_val, "std": 0.0}
            except Exception as exc:
                print(
                    f"WARNING: FRD computation failed: {exc}",
                    file=sys.stderr,
                )

        return EvaluationResult(per_case=per_case, aggregates=agg)


# ======================================================================
# Radiomic feature extraction  (thin wrapper around pyradiomics)
# ======================================================================

def extract_radiomic_features(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Extract radiomic features from a 2-D image within *mask*.

    Returns a 1-D ``float64`` feature vector.  Raises ``ImportError``
    if ``pyradiomics`` is not installed.
    """
    from radiomics import featureextractor  # type: ignore[import-untyped]

    # pyradiomics needs 3-D SimpleITK images
    img_3d = image[np.newaxis, ...] if image.ndim == 2 else image
    msk_3d = (
        mask.astype(np.uint8)[np.newaxis, ...]
        if mask.ndim == 2
        else mask.astype(np.uint8)
    )

    sitk_img = sitk.GetImageFromArray(img_3d.astype(np.float64))
    sitk_msk = sitk.GetImageFromArray(msk_3d)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings["force2D"] = True
    extractor.settings["force2Ddimension"] = 0
    extractor.disableAllFeatures()
    for cls_name in (
        "firstorder",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm",
    ):
        extractor.enableFeatureClassByName(cls_name)

    result = extractor.execute(sitk_img, sitk_msk)

    features: list[float] = []
    for key in sorted(result.keys()):
        if not key.startswith("diagnostics_"):
            try:
                features.append(float(result[key]))
            except (ValueError, TypeError):
                pass

    return np.array(features, dtype=np.float64)


# ======================================================================
# Fréchet distance
# ======================================================================


def frechet_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    r"""Fréchet distance between two sets of feature vectors.

    .. math::
        d^2 = \|\mu_1 - \mu_2\|^2
              + \operatorname{Tr}\!\bigl(
                  \Sigma_1 + \Sigma_2
                  - 2\,(\Sigma_1 \Sigma_2)^{1/2}
              \bigr)

    A small ridge (``1e-6 * I``) is added to the covariance
    matrices for numerical stability when ``N < D``.
    """
    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)
    d = mu1.shape[0]

    sigma1 = np.cov(X1, rowvar=False) + np.eye(d) * 1e-6
    sigma2 = np.cov(X2, rowvar=False) + np.eye(d) * 1e-6

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
