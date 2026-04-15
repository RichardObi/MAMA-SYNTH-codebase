#  Copyright 2025 mama-synth-eval contributors
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
Main evaluation module for the MAMA-SYNTH challenge.

Evaluates generative models that translate pre-contrast to post-contrast
breast DCE-MRI images across four tasks with **8 equally-weighted metrics**
(2 per task):

  Task 1 — Full-image comparison:
      1.1  MSE   (pixel-wise image comparison)
      1.2  LPIPS (perceptual image comparison)

  Task 2 — Tumor ROI comparison:
      2.1  SSIM  (pixel-based tumor-area intensity & texture comparison)
      2.2  FRD   (distributional comparison of tumor-area realism)

  Task 3 — Classification:
      3.1  AUROC contrast    (pre- vs post-contrast phase classification)
      3.2  AUROC tumor ROI   (tumor ROI vs contralateral mirrored ROI)

  Task 4 — Segmentation:
      4.1  Dice  (classic standardised segmentation metric)
      4.2  HD95  (complementary boundary-distance metric)

All 8 metrics are ranked equally in a flat Borda-count ranking.

The output ``metrics.json`` uses a Grand-Challenge-compatible structure::

    {
      "aggregates": {
          "mse_full_image": {"mean": ..., "std": ...},
          "lpips_full_image": {"mean": ..., "std": ...},
          "ssim_roi": {"mean": ..., "std": ...},
          "frd_roi": ...,
          "auroc_luminal": ...,
          "auroc_tnbc": ...,
          "auroc_tumor_roi": ...,
          "dice": {"mean": ..., "std": ...},
          "hausdorff95": {"mean": ..., "std": ...}
      },
      "results": [ ... per-case details ... ]
    }

All image-based metrics are computed on intensity-normalized images using
**dataset-level z-score normalization** as specified in the challenge protocol.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from eval.metrics import (
    compute_mae,
    compute_mse,
    compute_ncc,
    compute_nmse,
    compute_psnr,
    compute_ssim,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Challenge metric names (Grand Challenge jsonpath keys)
# ---------------------------------------------------------------------------

# The 9 official challenge metrics
METRIC_MSE_FULL = "mse_full_image"
METRIC_LPIPS_FULL = "lpips_full_image"
METRIC_SSIM_ROI = "ssim_roi"
METRIC_FRD_ROI = "frd_roi"
METRIC_AUROC_CONTRAST = "auroc_contrast"
METRIC_AUROC_LUMINAL = "auroc_luminal"
METRIC_AUROC_TNBC = "auroc_tnbc"
METRIC_AUROC_TUMOR_ROI = "auroc_tumor_roi"
METRIC_DICE = "dice"
METRIC_HD95 = "hausdorff95"

# Optional: tqdm progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        """Fallback no-op wrapper when tqdm is not installed."""
        return iterable


# ---------------------------------------------------------------------------
# Dataset-level normalisation
# ---------------------------------------------------------------------------


def normalize_intensity(
    image: NDArray[np.floating],
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> NDArray[np.floating]:
    """Z-score normalize an image.

    When *mean* and *std* are supplied, dataset-level normalisation is
    applied (the challenge protocol).  When omitted, per-image z-score
    is used as a fallback.

    Args:
        image: Input image array.
        mean: Pre-computed dataset mean. If ``None``, uses image mean.
        std: Pre-computed dataset std. If ``None``, uses image std.

    Returns:
        Normalized image array (float64).
    """
    image = image.astype(np.float64)
    mu = mean if mean is not None else float(np.mean(image))
    sigma = std if std is not None else float(np.std(image))
    if sigma == 0:
        return image - mu
    return (image - mu) / sigma


class DatasetNormalizer:
    """Compute dataset-level statistics for z-score normalisation.

    The MAMA-SYNTH protocol specifies *"z-score normalization applied at the
    dataset level"* — i.e. a single (mean, std) pair is computed over all
    ground-truth images, then applied identically to both GT and synthetic
    images so that metric comparisons are on a common intensity scale.
    """

    def __init__(self) -> None:
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self._fitted = False

    def fit(self, images: list[NDArray[np.floating]]) -> "DatasetNormalizer":
        """Compute global mean and std over a list of images."""
        all_vals = np.concatenate([img.ravel() for img in images])
        self.mean = float(np.mean(all_vals))
        self.std = float(np.std(all_vals))
        self._fitted = True
        logger.info(
            f"DatasetNormalizer fitted: mean={self.mean:.4f}, std={self.std:.4f}"
        )
        return self

    def transform(self, image: NDArray[np.floating]) -> NDArray[np.floating]:
        if not self._fitted:
            raise RuntimeError("DatasetNormalizer not fitted. Call fit() first.")
        return normalize_intensity(image, mean=self.mean, std=self.std)


class MamaSynthEval:
    """Evaluation class for the MAMA-SYNTH challenge.

    Implements the full evaluation pipeline across four tasks with
    8 equally-weighted metrics (2 per task):

    - Task 1 (Full Image): MSE + LPIPS
    - Task 2 (ROI):        SSIM + FRD
    - Task 3 (CLF):        AUROC luminal + AUROC TNBC
    - Task 4 (SEG):        Dice + Hausdorff95

    Output is a Grand-Challenge-compatible ``metrics.json`` with
    ``aggregates`` dict using flat JSON paths.

    Attributes:
        ground_truth_path: Path to directory with ground truth post-contrast images.
        predictions_path: Path to directory with synthetic post-contrast images.
        output_file: Path to output JSON file for metrics.
        masks_path: Optional path to directory with tumor segmentation masks.
        labels_path: Optional path to a JSON/CSV with molecular subtype labels.
        roi_margin_mm: Dilation margin (mm) for tumor ROI extraction.
        enable_lpips: Whether to compute LPIPS (requires torch + lpips).
        enable_frd: Whether to compute FRD (requires pyradiomics).
        enable_segmentation: Whether to run segmentation evaluation.
        enable_classification: Whether to run classification evaluation.
        seg_model_path: Optional path to a pre-trained segmentation model.
        clf_model_dir: Legacy fallback directory for all classifiers.
        clf_model_dir_contrast: Directory with contrast classifier models.
        clf_model_dir_tumor_roi: Directory with tumor-ROI classifier models.
        clf_model_dir_luminal: Directory with luminal classifier models.
        clf_model_dir_tnbc: Directory with TNBC classifier models.
        cache_dir: Optional directory for caching intermediate features.
    """

    SUPPORTED_EXTENSIONS = {".nii", ".nii.gz", ".mha", ".mhd", ".png"}

    def __init__(
        self,
        ground_truth_path: Path,
        predictions_path: Path,
        output_file: Path,
        masks_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        roi_margin_mm: float = 10.0,
        enable_lpips: bool = True,
        enable_frd: bool = True,
        enable_segmentation: bool = True,
        enable_classification: bool = True,
        seg_model_path: Optional[Path] = None,
        clf_model_dir: Optional[Path] = None,
        clf_model_dir_contrast: Optional[Path] = None,
        clf_model_dir_tumor_roi: Optional[Path] = None,
        clf_model_dir_luminal: Optional[Path] = None,
        clf_model_dir_tnbc: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        ensemble: bool = False,
        dual_phase: bool = False,
        precontrast_path: Optional[Path] = None,
    ) -> None:
        self.ground_truth_path = Path(ground_truth_path).resolve()
        self.predictions_path = Path(predictions_path).resolve()
        self.output_file = Path(output_file).resolve()
        self.masks_path = Path(masks_path).resolve() if masks_path else None
        self.labels_path = Path(labels_path).resolve() if labels_path else None
        self.roi_margin_mm = roi_margin_mm
        self.enable_lpips = enable_lpips
        self.enable_frd = enable_frd
        self.enable_segmentation = enable_segmentation
        self.enable_classification = enable_classification
        self.seg_model_path = Path(seg_model_path) if seg_model_path else None
        # Per-task classifier directories.  Explicit per-task dirs take
        # precedence; clf_model_dir is a legacy fallback for all tasks.
        _fallback = Path(clf_model_dir) if clf_model_dir else None
        self.clf_model_dir = _fallback  # kept for backward compat
        self.clf_model_dir_contrast = (
            Path(clf_model_dir_contrast) if clf_model_dir_contrast
            else _fallback
        )
        self.clf_model_dir_tumor_roi = (
            Path(clf_model_dir_tumor_roi) if clf_model_dir_tumor_roi
            else _fallback
        )
        self.clf_model_dir_luminal = (
            Path(clf_model_dir_luminal) if clf_model_dir_luminal
            else _fallback
        )
        self.clf_model_dir_tnbc = (
            Path(clf_model_dir_tnbc) if clf_model_dir_tnbc
            else _fallback
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ensemble = ensemble
        self.dual_phase = dual_phase
        self.precontrast_path = (
            Path(precontrast_path) if precontrast_path else None
        )
        # Will be fitted during evaluate()
        self._normalizer = DatasetNormalizer()
        # Per-evaluation image cache to avoid redundant NIfTI loads.
        # Enabled only during evaluate() and cleared afterwards.
        self._image_cache: Optional[dict[str, NDArray]] = None
        # Per-evaluation radiomic feature cache – avoids recomputing
        # pyradiomics features for the same image(+mask) across tasks
        # (e.g. classification + contrast + tumor_roi).
        self._feature_cache: Optional[dict[str, NDArray]] = None

    def _clf_model_dir_for_task(self, task: str) -> Optional[Path]:
        """Return the classifier model directory for *task*.

        Resolution order: explicit per-task attribute → ``clf_model_dir``
        fallback (already resolved during ``__init__``).
        """
        _dirs = {
            "contrast": self.clf_model_dir_contrast,
            "tumor_roi": self.clf_model_dir_tumor_roi,
            "luminal": self.clf_model_dir_luminal,
            "tnbc": self.clf_model_dir_tnbc,
        }
        return _dirs.get(task, self.clf_model_dir)

    def _extract_features_cached(
        self,
        image: NDArray,
        image_key: str,
        mask: Optional[NDArray] = None,
    ) -> NDArray:
        """Extract radiomic features, using an in-memory cache.

        Parameters
        ----------
        image : NDArray
            Image array to extract features from.
        image_key : str
            A stable identifier for this image (e.g. file path string).
        mask : NDArray | None
            Optional boolean mask for ROI-based extraction.

        Returns
        -------
        NDArray
            1-D feature vector.
        """
        import hashlib

        from eval.frd import extract_radiomic_features

        mask_part = ""
        if mask is not None:
            mask_part = hashlib.sha256(mask.tobytes()).hexdigest()[:12]
        cache_key = f"{image_key}|{mask_part}"

        if self._feature_cache is not None and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        feats = extract_radiomic_features(image, mask=mask)

        if self._feature_cache is not None:
            self._feature_cache[cache_key] = feats

        return feats

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        """Run the complete evaluation pipeline.

        Returns:
            Dictionary containing all computed metrics in
            Grand-Challenge-compatible format with ``aggregates`` and
            ``results`` keys.

        Raises:
            FileNotFoundError: If required directories don't exist.
            ValueError: If no matching image pairs are found.
        """
        # Validate paths
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth path not found: {self.ground_truth_path}"
            )
        if not self.predictions_path.exists():
            raise FileNotFoundError(
                f"Predictions path not found: {self.predictions_path}"
            )

        # Find image pairs
        gt_files = self._get_image_files(self.ground_truth_path)
        pred_files = self._get_image_files(self.predictions_path)
        pred_mapping = {self._get_stem(f): f for f in pred_files}

        pairs: list[tuple[Path, Path]] = []
        missing_stems: list[str] = []
        for gt_file in gt_files:
            stem = self._get_stem(gt_file)
            if stem in pred_mapping:
                pairs.append((gt_file, pred_mapping[stem]))
            else:
                missing_stems.append(stem)
                logger.warning(f"No prediction found for ground truth: {gt_file.name}")

        if not pairs and not missing_stems:
            raise ValueError(
                "No matching image pairs found between ground truth and predictions"
            )
        if not pairs:
            raise ValueError(
                "No matching image pairs found between ground truth and predictions"
            )

        logger.info(f"Found {len(pairs)} image pairs to evaluate")
        if missing_stems:
            logger.warning(
                f"{len(missing_stems)} ground truth images lack predictions — "
                "worst-score imputation will apply."
            )

        # Enable per-evaluation image cache so NIfTI files are decoded once.
        self._image_cache = {}
        # Enable per-evaluation radiomic feature cache.
        self._feature_cache = {}

        # --- Fit dataset-level normaliser on all GT images ---
        gt_images_raw = [
            self._load_image_cached(p) for p in tqdm(
                [p[0] for p in pairs], desc="Loading GT images", leave=False
            )
        ]
        self._normalizer.fit(gt_images_raw)

        # Load optional masks
        masks = self._load_masks([p[0] for p in pairs]) if self.masks_path else None

        # --- Aggregates dict (Grand-Challenge-compatible) ---
        aggregates: dict[str, Any] = {}

        # ---- Task 1: Full-Image metrics (MSE + LPIPS) ----
        full_image_results = self._evaluate_full_image(pairs, gt_images_raw)
        aggregates.update(full_image_results.get("aggregates", {}))

        # ---- Task 2: Tumor ROI metrics (SSIM + FRD) ----
        roi_results: dict[str, Any] = {}
        if masks:
            roi_results = self._evaluate_roi(pairs, masks)
            aggregates.update(roi_results.get("aggregates", {}))

        # ---- Task 3a: Contrast classification (AUROC contrast) ----
        # Contrast classification is independent of molecular-subtype labels:
        # it only requires a pre-trained contrast classifier and the
        # pre-contrast source images.
        contrast_results: dict[str, Any] = {}
        if self.enable_classification and self.clf_model_dir_contrast:
            contrast_results = self._evaluate_contrast(pairs)
            aggregates.update(contrast_results.get("aggregates", {}))

        # ---- Task 3b: Classification (AUROC luminal + AUROC TNBC) ----
        clf_results: dict[str, Any] = {}
        if self.enable_classification and self.labels_path:
            clf_results = self._evaluate_classification(pairs)
            aggregates.update(clf_results.get("aggregates", {}))

        # ---- Task 3c: Tumor ROI classification ----
        # Tumor ROI classification does NOT need molecular-subtype labels.
        # It creates its own binary labels (tumor=1, mirror=0) from masks.
        tumor_roi_results: dict[str, Any] = {}
        if self.enable_classification and self.masks_path:
            tumor_roi_results = self._evaluate_tumor_roi(pairs)
            aggregates.update(tumor_roi_results.get("aggregates", {}))

        # ---- Task 4: Segmentation (Dice + HD95) ----
        seg_results: dict[str, Any] = {}
        if self.enable_segmentation and masks:
            seg_results = self._evaluate_segmentation(pairs, masks)
            aggregates.update(seg_results.get("aggregates", {}))

        # ---- Per-case results (legacy + GC results list) ----
        case_metrics, legacy_aggregate = self._evaluate_pairwise_legacy(pairs)

        # Apply worst-score imputation for missing predictions
        if missing_stems:
            self._impute_missing(case_metrics, legacy_aggregate, missing_stems)

        # Assemble Grand-Challenge-compatible results
        results: dict[str, Any] = {
            "aggregates": aggregates,
            "results": case_metrics,
            # Legacy keys for backward compatibility
            "aggregate": legacy_aggregate,
            "cases": case_metrics,
        }

        # Task-level detail (for debugging / visualization)
        if full_image_results.get("detail"):
            results["full_image"] = full_image_results["detail"]
        if roi_results.get("detail"):
            results["roi"] = roi_results["detail"]
        if seg_results.get("detail"):
            results["segmentation"] = seg_results["detail"]
        # Merge contrast detail into classification block
        merged_clf_detail: dict[str, Any] = {}
        if contrast_results.get("detail"):
            merged_clf_detail.update(contrast_results["detail"])
        if clf_results.get("detail"):
            merged_clf_detail.update(clf_results["detail"])
        if tumor_roi_results.get("detail"):
            merged_clf_detail.update(tumor_roi_results["detail"])
        if merged_clf_detail:
            results["classification"] = merged_clf_detail
        if missing_stems:
            results["missing_predictions"] = missing_stems

        self._save_results(results)

        # Release per-evaluation caches to free memory.
        self._image_cache = None
        self._feature_cache = None

        return results

    # ------------------------------------------------------------------
    # Task 1: Full-image metrics (MSE + LPIPS)
    # ------------------------------------------------------------------

    def _evaluate_full_image(
        self,
        pairs: list[tuple[Path, Path]],
        gt_images_raw: list[NDArray],
    ) -> dict[str, Any]:
        """Compute full-image MSE and LPIPS.

        Returns dict with 'aggregates' and 'detail' sub-keys.
        """
        mse_values: list[float] = []
        lpips_values: list[float] = []
        real_images: list[NDArray] = []
        synth_images: list[NDArray] = []

        for (gt_path, pred_path), gt_raw in tqdm(
            list(zip(pairs, gt_images_raw)),
            desc="Full-image metrics",
            leave=False,
        ):
            gt = self._normalizer.transform(gt_raw)
            pred = self._normalizer.transform(self._load_image_cached(pred_path))
            pred = self._resize_pred_to_gt(pred, gt)

            mse_values.append(float(np.mean((pred - gt) ** 2)))
            real_images.append(gt)
            synth_images.append(pred)

        agg: dict[str, Any] = {}
        detail: dict[str, Any] = {}

        if mse_values:
            agg[METRIC_MSE_FULL] = self._aggregate(mse_values)
            detail["mse"] = self._aggregate(mse_values)

        if self.enable_lpips:
            try:
                from eval.metrics import compute_lpips

                for gt, pred in tqdm(
                    list(zip(real_images, synth_images)),
                    desc="LPIPS (full)",
                    leave=False,
                ):
                    lpips_values.append(compute_lpips(pred, gt))
                agg[METRIC_LPIPS_FULL] = self._aggregate(lpips_values)
                detail["lpips"] = self._aggregate(lpips_values)
            except ImportError:
                logger.warning("LPIPS unavailable (torch/lpips not installed), skipping.")

        return {"aggregates": agg, "detail": detail}

    # ------------------------------------------------------------------
    # Task 2: Tumor ROI metrics (SSIM + FRD)
    # ------------------------------------------------------------------

    def _evaluate_roi(
        self,
        pairs: list[tuple[Path, Path]],
        masks: dict[str, NDArray[np.bool_]],
    ) -> dict[str, Any]:
        """Compute tumor-ROI SSIM and FRD.

        SSIM is computed on BBox-cropped ROIs.  FRD is computed using the
        official ``frd-score`` package (https://github.com/RichardObi/frd-score)
        which passes full images + masks to pyradiomics for feature
        extraction internally, matching the published metric exactly.

        Returns dict with 'aggregates' and 'detail' sub-keys.
        """
        from eval.roi_utils import extract_roi_pair

        ssim_values: list[float] = []

        # Collect file paths for the official frd-score API
        # (masks are looked up from self.masks_path by stem).
        frd_gt_paths: list[str] = []
        frd_pred_paths: list[str] = []
        frd_mask_paths: list[str] = []

        mask_file_mapping: dict[str, Path] = {}
        if self.masks_path and self.masks_path.exists():
            for mf in self._get_image_files(self.masks_path):
                mask_file_mapping[self._get_stem(mf)] = mf

        for gt_path, pred_path in tqdm(pairs, desc="ROI metrics", leave=False):
            stem = self._get_stem(gt_path)
            if stem not in masks:
                logger.warning(f"No mask for {stem}, skipping ROI evaluation.")
                continue

            gt = self._normalizer.transform(self._load_image_cached(gt_path))
            pred = self._normalizer.transform(self._load_image_cached(pred_path))
            pred = self._resize_pred_to_gt(pred, gt)
            mask = masks[stem]

            # Ensure mask matches image resolution
            if mask.shape != gt.shape:
                mask = self._resize_array_to_target(mask, gt.shape)

            real_roi, synth_roi, roi_mask = extract_roi_pair(
                gt, pred, mask, margin_mm=self.roi_margin_mm
            )

            # Compute SSIM within the ROI (masked region only)
            data_range = float(
                np.max(real_roi[roi_mask]) - np.min(real_roi[roi_mask])
            )
            if data_range > 0:
                ssim_val = compute_ssim(
                    synth_roi, real_roi, data_range=data_range, mask=roi_mask
                )
            else:
                ssim_val = 1.0  # identical constant images
            ssim_values.append(ssim_val)

            # Collect file paths for FRD (absolute paths required —
            # frd-score uses multiprocessing + sitk.ReadImage internally).
            if stem in mask_file_mapping:
                frd_gt_paths.append(str(gt_path.resolve()))
                frd_pred_paths.append(str(pred_path.resolve()))
                frd_mask_paths.append(str(mask_file_mapping[stem].resolve()))

        if not ssim_values:
            return {}

        agg: dict[str, Any] = {}
        detail: dict[str, Any] = {}

        agg[METRIC_SSIM_ROI] = self._aggregate(ssim_values)
        detail["ssim"] = self._aggregate(ssim_values)

        if self.enable_frd and len(frd_gt_paths) >= 2:
            try:
                from frd_score import compute_frd as frd_compute

                # frd-score reads images with sitk.ReadImage() inside a
                # multiprocessing Pool.  Some SimpleITK builds lack the
                # PNG ImageIO plugin, causing RuntimeError.  Convert any
                # PNG files to .mha (ITK-native, always supported) in a
                # temporary directory before calling the library.
                import tempfile

                _has_png = any(
                    p.endswith(".png") for p in
                    frd_gt_paths + frd_pred_paths + frd_mask_paths
                )

                if _has_png:
                    _tmpdir = tempfile.mkdtemp(prefix="frd_mha_")
                    try:
                        frd_gt_paths = self._ensure_mha(
                            frd_gt_paths, _tmpdir, "gt"
                        )
                        frd_pred_paths = self._ensure_mha(
                            frd_pred_paths, _tmpdir, "pred"
                        )
                        frd_mask_paths = self._ensure_mha(
                            frd_mask_paths, _tmpdir, "mask"
                        )
                        frd_val = frd_compute(
                            [frd_gt_paths, frd_pred_paths],
                            paths_masks=[frd_mask_paths, frd_mask_paths],
                        )
                    finally:
                        import shutil
                        shutil.rmtree(_tmpdir, ignore_errors=True)
                else:
                    frd_val = frd_compute(
                        [frd_gt_paths, frd_pred_paths],
                        paths_masks=[frd_mask_paths, frd_mask_paths],
                    )

                agg[METRIC_FRD_ROI] = frd_val
                detail["frd"] = frd_val
            except ImportError:
                logger.warning(
                    "FRD unavailable (frd-score not installed). "
                    "Install with: pip install frd-score"
                )

        return {"aggregates": agg, "detail": detail}

    # ------------------------------------------------------------------
    # Task 4: Segmentation (Dice + HD95)
    # ------------------------------------------------------------------

    def _evaluate_segmentation(
        self,
        pairs: list[tuple[Path, Path]],
        masks: dict[str, NDArray[np.bool_]],
    ) -> dict[str, Any]:
        """Evaluate segmentation on synthetic images.

        Returns dict with 'aggregates' and 'detail' sub-keys.
        """
        from eval.segmentation import (
            ThresholdSegmenter,
            evaluate_segmentation_pair,
        )

        # Use organizer-provided nnUNet model if available, else threshold
        seg_model: Any
        if self.seg_model_path and self.seg_model_path.exists():
            try:
                from eval.segmentation import NNUNetSegmenter

                seg_model = NNUNetSegmenter(model_dir=self.seg_model_path)
                logger.info(f"Using nnUNet segmenter from {self.seg_model_path}")
            except ImportError:
                logger.warning(
                    "nnUNet not available, falling back to ThresholdSegmenter."
                )
                seg_model = ThresholdSegmenter()
        else:
            seg_model = ThresholdSegmenter()

        dice_values: list[float] = []
        hd95_values: list[float] = []

        for gt_path, pred_path in tqdm(
            pairs, desc="Segmentation", leave=False
        ):
            stem = self._get_stem(gt_path)
            if stem not in masks:
                continue

            pred_image = self._load_image_cached(pred_path).astype(np.float64)
            gt_mask = masks[stem]

            # Resize pred to match GT mask resolution if needed
            pred_image = self._resize_array_to_target(
                pred_image, gt_mask.shape
            )

            # Apply segmentation model to synthetic image
            pred_mask = seg_model.predict(pred_image)

            result = evaluate_segmentation_pair(pred_mask, gt_mask)
            dice_values.append(result["dice"])
            hd95_values.append(result["hd95"])

        if not dice_values:
            return {}

        agg: dict[str, Any] = {
            METRIC_DICE: self._aggregate(dice_values),
            METRIC_HD95: self._aggregate(hd95_values),
        }
        detail: dict[str, Any] = {
            "dice": self._aggregate(dice_values),
            "hd95": self._aggregate(hd95_values),
        }

        return {"aggregates": agg, "detail": detail}

    # ------------------------------------------------------------------
    # Task 3: Classification (AUROC luminal + AUROC TNBC)
    # ------------------------------------------------------------------

    def _evaluate_classification(
        self,
        pairs: list[tuple[Path, Path]],
    ) -> dict[str, Any]:
        """Evaluate molecular subtype classification on synthetic images.

        When ``self.ensemble`` is ``True``, all models found in
        ``clf_model_dir`` for each task are used and their predicted
        probabilities are averaged (ensemble inference).  Otherwise the
        single-model behaviour is preserved: CNN has priority, then
        radiomics.

        Returns dict with 'aggregates' and 'detail' sub-keys.
        """
        if self.labels_path is None or not self.labels_path.exists():
            return {}

        try:
            from eval.classification import (
                CNNClassifier,
                EnsembleClassifier,
                RadiomicsClassifier,
                evaluate_classification,
            )
        except ImportError:
            logger.warning("Classification dependencies unavailable, skipping.")
            return {}

        # Load labels
        labels = self._load_labels()
        if not labels:
            return {}

        # Extract radiomic features from synthetic images
        features_list = []
        tnbc_true = []
        luminal_true = []
        valid_stems = []

        for _, pred_path in tqdm(pairs, desc="CLF features", leave=False):
            stem = self._get_stem(pred_path)
            if stem not in labels:
                continue

            pred_image = self._load_image_cached(pred_path).astype(np.float64)
            feats = self._extract_features_cached(
                pred_image, str(pred_path),
            )

            # Dual-phase: also extract features from pre-contrast and concatenate
            if self.dual_phase and self.precontrast_path:
                precon_feats = None
                for ext in (".nii.gz", ".nii", ".mha"):
                    pc_path = self.precontrast_path / f"{stem}{ext}"
                    if pc_path.exists():
                        try:
                            pc_image = self._load_image_cached(pc_path).astype(
                                np.float64
                            )
                            precon_feats = self._extract_features_cached(
                                pc_image, str(pc_path),
                            )
                        except Exception as e:
                            logger.warning(
                                f"Dual-phase: failed to load pre-contrast "
                                f"for {stem}: {e}"
                            )
                        break
                if precon_feats is not None:
                    feats = np.concatenate([feats, precon_feats])
                else:
                    # Pad to double width so shapes stay consistent
                    feats = np.concatenate([feats, np.zeros_like(feats)])

            features_list.append(feats)
            tnbc_true.append(labels[stem].get("tnbc", 0))
            luminal_true.append(labels[stem].get("luminal", 0))
            valid_stems.append(stem)

        if len(features_list) < 2:
            return {}

        # Guard against ragged feature vectors (e.g. pyradiomics failure
        # returning a 1-element fallback while normal vectors are ~93-D).
        expected_dim = max(f.shape[0] for f in features_list)
        keep_idx = [
            i for i, f in enumerate(features_list)
            if f.shape[0] == expected_dim
        ]
        if len(keep_idx) < len(features_list):
            logger.warning(
                f"Dropped {len(features_list) - len(keep_idx)} cases with "
                f"unexpected feature dimension (expected {expected_dim})."
            )
            features_list = [features_list[i] for i in keep_idx]
            tnbc_true = [tnbc_true[i] for i in keep_idx]
            luminal_true = [luminal_true[i] for i in keep_idx]
            valid_stems = [valid_stems[i] for i in keep_idx]
        if len(features_list) < 2:
            return {}

        feature_matrix = np.stack(features_list)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        y_tnbc = np.array(tnbc_true, dtype=np.int64)
        y_luminal = np.array(luminal_true, dtype=np.int64)

        agg: dict[str, Any] = {}
        detail: dict[str, Any] = {"n_cases": len(features_list)}

        for task, y_true, metric_key in [
            ("luminal", y_luminal, METRIC_AUROC_LUMINAL),
            ("tnbc", y_tnbc, METRIC_AUROC_TNBC),
        ]:
            _task_dir = self._clf_model_dir_for_task(task)

            # ----------------------------------------------------------
            # Ensemble mode: discover all models and average probabilities
            # ----------------------------------------------------------
            if self.ensemble and _task_dir and _task_dir.exists():
                ensemble = EnsembleClassifier.discover_models(
                    task=task, model_dir=_task_dir,
                )
                if ensemble.n_models == 0:
                    detail[f"note_{task}"] = (
                        f"No pre-trained {task} models found in "
                        f"{_task_dir} for ensemble."
                    )
                    continue

                logger.info(
                    f"Ensemble inference for '{task}': "
                    f"{ensemble.description()}"
                )

                # Prepare CNN inputs only if the ensemble contains CNN models
                cnn_images: Optional[list[NDArray]] = None
                cnn_masks: Optional[list[Optional[NDArray]]] = None
                if ensemble.has_cnn:
                    cnn_images = []
                    cnn_masks = []
                    for _, pred_path in pairs:
                        stem = self._get_stem(pred_path)
                        if stem not in labels:
                            continue
                        img = self._load_image_cached(pred_path).astype(np.float64)
                        cnn_images.append(img)
                        mask_arr: Optional[NDArray] = None
                        if self.masks_path and self.masks_path.exists():
                            for ext in (".nii.gz", ".nii", ".mha", ".png"):
                                mp = self.masks_path / f"{stem}{ext}"
                                if mp.exists():
                                    if ext == ".png":
                                        mask_arr = (
                                            self._load_image(mp) > 0
                                        )
                                    else:
                                        mask_arr = sitk.GetArrayFromImage(
                                            sitk.ReadImage(str(mp), sitk.sitkUInt8)
                                        ).astype(bool)
                                    break
                        if mask_arr is not None and mask_arr.shape != img.shape:
                            mask_arr = self._resize_array_to_target(
                                mask_arr, img.shape
                            )
                        cnn_masks.append(mask_arr)

                y_score = ensemble.predict_proba(
                    features=feature_matrix if ensemble.has_radiomics else None,
                    images=cnn_images,
                    masks=cnn_masks,
                )
                clf_result = evaluate_classification(y_true, y_score)
                agg[metric_key] = clf_result["auroc"]
                detail[f"auroc_{task}"] = clf_result["auroc"]
                detail[f"balanced_accuracy_{task}"] = (
                    clf_result["balanced_accuracy"]
                )
                detail[f"classifier_type_{task}"] = ensemble.description()
                continue

            # ----------------------------------------------------------
            # Single-model mode (original behaviour)
            # ----------------------------------------------------------
            cnn_model_path = None
            pkl_model_path = None
            if _task_dir and _task_dir.exists():
                cnn_candidate = (
                    _task_dir / f"{task}_classifier_cnn.pt"
                )
                pkl_candidate = (
                    _task_dir / f"{task}_classifier.pkl"
                )
                if cnn_candidate.exists():
                    cnn_model_path = cnn_candidate
                if pkl_candidate.exists():
                    pkl_model_path = pkl_candidate

            if cnn_model_path:
                # CNN classifier — works on raw images, not features
                try:
                    cnn_clf = CNNClassifier(
                        task=task, model_path=cnn_model_path,
                    )
                    # Collect raw images and optional masks
                    cnn_imgs: list[NDArray] = []
                    cnn_msks: list[Optional[NDArray]] = []
                    cnn_y: list[int] = []

                    for _, pred_path in pairs:
                        stem = self._get_stem(pred_path)
                        if stem not in labels:
                            continue
                        img = self._load_image_cached(pred_path).astype(np.float64)
                        cnn_imgs.append(img)
                        # Try to load mask for better slice selection
                        mask_arr_single: Optional[NDArray] = None
                        if self.masks_path and self.masks_path.exists():
                            for ext in (".nii.gz", ".nii", ".mha", ".png"):
                                mp = self.masks_path / f"{stem}{ext}"
                                if mp.exists():
                                    if ext == ".png":
                                        mask_arr_single = (
                                            self._load_image(mp) > 0
                                        )
                                    else:
                                        mask_arr_single = sitk.GetArrayFromImage(
                                            sitk.ReadImage(str(mp), sitk.sitkUInt8)
                                        ).astype(bool)
                                    break
                        if mask_arr_single is not None and mask_arr_single.shape != img.shape:
                            mask_arr_single = self._resize_array_to_target(
                                mask_arr_single, img.shape
                            )
                        cnn_msks.append(mask_arr_single)
                        cnn_y.append(
                            labels[stem].get(task, 0)
                        )

                    if len(cnn_imgs) >= 2:
                        y_score = cnn_clf.predict_proba_from_images(
                            cnn_imgs, cnn_msks,
                        )
                        _y_true_cnn = np.array(cnn_y, dtype=np.int64)
                        clf_result = evaluate_classification(
                            _y_true_cnn, y_score,
                        )
                        agg[metric_key] = clf_result["auroc"]
                        detail[f"auroc_{task}"] = clf_result["auroc"]
                        detail[f"balanced_accuracy_{task}"] = (
                            clf_result["balanced_accuracy"]
                        )
                        detail[f"classifier_type_{task}"] = "cnn"
                except ImportError:
                    logger.warning(
                        f"CNN classifier found for {task} but dependencies "
                        "unavailable. Falling back to radiomics."
                    )
                    # Fall through to radiomics below
                    if pkl_model_path:
                        clf = RadiomicsClassifier(
                            task=task, model_path=pkl_model_path,
                        )
                        y_score = clf.predict_proba(feature_matrix)
                        clf_result = evaluate_classification(y_true, y_score)
                        agg[metric_key] = clf_result["auroc"]
                        detail[f"auroc_{task}"] = clf_result["auroc"]
                        detail[f"balanced_accuracy_{task}"] = (
                            clf_result["balanced_accuracy"]
                        )

            elif pkl_model_path:
                clf = RadiomicsClassifier(
                    task=task, model_path=pkl_model_path,
                )
                y_score = clf.predict_proba(feature_matrix)
                clf_result = evaluate_classification(y_true, y_score)
                agg[metric_key] = clf_result["auroc"]
                detail[f"auroc_{task}"] = clf_result["auroc"]
                detail[f"balanced_accuracy_{task}"] = (
                    clf_result["balanced_accuracy"]
                )
            else:
                detail[f"note_{task}"] = (
                    f"No pre-trained {task} classifier found. "
                    f"Provide model via --clf-model-dir-{task.replace('_', '-')} "
                    f"or --clf-model-dir."
                )

        # NOTE: tumor_roi classification is handled as a separate
        # standalone block in evaluate() because it does not depend on
        # molecular-subtype labels.

        return {"aggregates": agg, "detail": detail}

    # ------------------------------------------------------------------
    # Task 3a: Contrast classification (AUROC contrast)
    # ------------------------------------------------------------------

    def _evaluate_contrast(
        self,
        pairs: list[tuple[Path, Path]],
    ) -> dict[str, Any]:
        """Evaluate pre- vs post-contrast phase classification.

        A pre-trained contrast classifier is applied to both the
        *pre-contrast source images* (label=0) and the *synthetic
        post-contrast images* (label=1).  If synthesis is successful,
        the synthetic images should look clearly post-contrast and the
        classifier should easily distinguish them from pre-contrast
        sources, yielding a high AUROC.

        Requires:
        * ``--clf-model-dir-contrast`` pointing to a directory with
          ``contrast_classifier.pkl`` (and/or ``contrast_classifier_cnn.pt``).
        * ``--precontrast-path`` pointing to the pre-contrast source
          images (same stems as predictions).

        Returns dict with 'aggregates' and 'detail' sub-keys.
        """
        task = "contrast"
        metric_key = METRIC_AUROC_CONTRAST

        _task_dir = self._clf_model_dir_for_task(task)
        if _task_dir is None or not _task_dir.exists():
            return {
                "detail": {
                    f"note_{task}": (
                        "No pre-trained contrast classifier found. "
                        "Provide model via --clf-model-dir-contrast."
                    )
                }
            }

        if self.precontrast_path is None or not self.precontrast_path.exists():
            return {
                "detail": {
                    f"note_{task}": (
                        "Contrast classification requires pre-contrast "
                        "source images (--precontrast-path)."
                    )
                }
            }

        try:
            from eval.classification import (
                CNNClassifier,
                RadiomicsClassifier,
                evaluate_classification,
            )
        except ImportError:
            logger.warning(
                "Contrast classification dependencies unavailable, skipping."
            )
            return {}

        # Discover model files
        cnn_model_path = None
        pkl_model_path = None
        cnn_candidate = _task_dir / f"{task}_classifier_cnn.pt"
        pkl_candidate = _task_dir / f"{task}_classifier.pkl"
        if cnn_candidate.exists():
            cnn_model_path = cnn_candidate
        if pkl_candidate.exists():
            pkl_model_path = pkl_candidate

        if cnn_model_path is None and pkl_model_path is None:
            return {
                "detail": {
                    f"note_{task}": (
                        f"No model files found in {_task_dir}. "
                        f"Expected {task}_classifier.pkl or "
                        f"{task}_classifier_cnn.pt."
                    )
                }
            }

        # ----------------------------------------------------------
        # Collect features/images for pre-contrast (label=0) and
        # synthetic post-contrast (label=1).
        # ----------------------------------------------------------
        synth_feats: list[NDArray] = []
        precon_feats: list[NDArray] = []
        valid_stems: list[str] = []

        for _, pred_path in tqdm(
            pairs, desc="Contrast CLF features", leave=False
        ):
            stem = self._get_stem(pred_path)

            # Find matching pre-contrast image
            pc_image_arr: Optional[NDArray] = None
            for ext in (".png", ".nii.gz", ".nii", ".mha"):
                pc_path = self.precontrast_path / f"{stem}{ext}"
                if pc_path.exists():
                    try:
                        pc_image_arr = self._load_image_cached(
                            pc_path
                        ).astype(np.float64)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to load pre-contrast for {stem}: {exc}"
                        )
                    break

            if pc_image_arr is None:
                continue

            pred_image = self._load_image_cached(pred_path).astype(np.float64)

            try:
                s_feats = self._extract_features_cached(
                    pred_image, str(pred_path),
                )
                p_feats = self._extract_features_cached(
                    pc_image_arr, str(pc_path),
                )
                if s_feats.size <= 1 or p_feats.size <= 1:
                    continue
                if s_feats.shape[0] != p_feats.shape[0]:
                    continue
                synth_feats.append(s_feats)
                precon_feats.append(p_feats)
                valid_stems.append(stem)
            except Exception as exc:
                logger.warning(
                    f"Contrast feature extraction failed for {stem}: {exc}"
                )
                continue

        if len(synth_feats) < 2:
            return {
                "detail": {
                    f"note_{task}": (
                        f"Too few valid cases ({len(synth_feats)}) for "
                        "contrast classification.  Check --precontrast-path."
                    )
                }
            }

        agg: dict[str, Any] = {}
        detail: dict[str, Any] = {f"n_cases_{task}": len(valid_stems)}

        # Build combined feature matrix: synth=1, precontrast=0
        all_feats = np.vstack(
            [np.stack(synth_feats), np.stack(precon_feats)]
        )
        all_feats = np.nan_to_num(
            all_feats, nan=0.0, posinf=0.0, neginf=0.0
        )
        y_true = np.concatenate([
            np.ones(len(synth_feats), dtype=np.int64),
            np.zeros(len(precon_feats), dtype=np.int64),
        ])

        # ----------------------------------------------------------
        # Radiomics model (default path)
        # ----------------------------------------------------------
        if pkl_model_path:
            try:
                clf = RadiomicsClassifier(
                    task=task, model_path=pkl_model_path,
                )
                y_score = clf.predict_proba(all_feats)
                clf_result = evaluate_classification(y_true, y_score)
                agg[metric_key] = clf_result["auroc"]
                detail[f"auroc_{task}"] = clf_result["auroc"]
                detail[f"balanced_accuracy_{task}"] = (
                    clf_result["balanced_accuracy"]
                )
                detail[f"classifier_type_{task}"] = "radiomics"
                logger.info(
                    f"Contrast AUROC = {clf_result['auroc']:.4f} "
                    f"({len(valid_stems)} pairs)"
                )
            except Exception as exc:
                logger.error(f"Contrast radiomics classifier failed: {exc}")

        # ----------------------------------------------------------
        # CNN model (override if available and radiomics didn't run)
        # ----------------------------------------------------------
        if cnn_model_path and metric_key not in agg:
            try:
                cnn_clf = CNNClassifier(
                    task=task, model_path=cnn_model_path,
                )
                # Collect raw images: synth first, then pre-contrast
                cnn_images: list[NDArray] = []
                for _, pred_path in pairs:
                    stem = self._get_stem(pred_path)
                    if stem not in valid_stems:
                        continue
                    cnn_images.append(
                        self._load_image_cached(pred_path).astype(np.float64)
                    )
                for stem in valid_stems:
                    for ext in (".png", ".nii.gz", ".nii", ".mha"):
                        pc_p = self.precontrast_path / f"{stem}{ext}"
                        if pc_p.exists():
                            cnn_images.append(
                                self._load_image_cached(pc_p).astype(
                                    np.float64
                                )
                            )
                            break
                y_score = cnn_clf.predict_proba_from_images(
                    cnn_images, masks=None,
                )
                clf_result = evaluate_classification(y_true, y_score)
                agg[metric_key] = clf_result["auroc"]
                detail[f"auroc_{task}"] = clf_result["auroc"]
                detail[f"balanced_accuracy_{task}"] = (
                    clf_result["balanced_accuracy"]
                )
                detail[f"classifier_type_{task}"] = "cnn"
            except Exception as exc:
                logger.error(f"Contrast CNN classifier failed: {exc}")

        return {"aggregates": agg, "detail": detail}

    def _evaluate_tumor_roi(
        self,
        pairs: list[tuple[Path, Path]],
    ) -> dict[str, Any]:
        """Evaluate tumor-ROI vs contralateral-mirrored-ROI classification.

        For each synthetic prediction that has a corresponding mask:
          1. Extract radiomic features from the tumor ROI (label=1).
          2. Mirror the mask across the body midline and extract
             radiomic features from the mirrored region (label=0).
          3. Run the pre-trained ``tumor_roi`` classifier on both
             feature vectors and collect predicted probabilities.

        Returns dict with 'aggregates' and 'detail' sub-keys.
        """
        task = "tumor_roi"
        metric_key = METRIC_AUROC_TUMOR_ROI
        agg: dict[str, Any] = {}
        detail: dict[str, Any] = {}

        # Check for a pre-trained tumor_roi model
        _task_dir = self._clf_model_dir_for_task(task)
        pkl_model_path = None
        if _task_dir and _task_dir.exists():
            pkl_candidate = _task_dir / f"{task}_classifier.pkl"
            if pkl_candidate.exists():
                pkl_model_path = pkl_candidate

        if pkl_model_path is None:
            detail[f"note_{task}"] = (
                f"No pre-trained {task} classifier found. "
                "Provide model via --clf-model-dir-tumor-roi "
                "or --clf-model-dir."
            )
            return {"aggregates": agg, "detail": detail}

        if self.masks_path is None or not self.masks_path.exists():
            detail[f"note_{task}"] = (
                "Tumor ROI evaluation requires masks (--masks-path)."
            )
            return {"aggregates": agg, "detail": detail}

        try:
            from eval.classification import (
                RadiomicsClassifier,
                evaluate_classification,
            )
            from eval.mirror_utils import create_mirrored_mask
        except ImportError:
            logger.warning(
                "Tumor ROI evaluation dependencies unavailable, skipping."
            )
            return {"aggregates": agg, "detail": detail}

        clf = RadiomicsClassifier(task=task, model_path=pkl_model_path)

        tumor_feats: list[NDArray] = []
        mirror_feats: list[NDArray] = []
        valid_stems: list[str] = []

        for _, pred_path in tqdm(
            pairs, desc="Tumor ROI features", leave=False
        ):
            stem = self._get_stem(pred_path)

            # Load mask
            mask_arr: Optional[NDArray[np.bool_]] = None
            if self.masks_path:
                for ext in (".nii.gz", ".nii", ".mha", ".png"):
                    mask_file = self.masks_path / f"{stem}{ext}"
                    if mask_file.exists():
                        try:
                            if ext == ".png":
                                mask_arr = (
                                    self._load_image(mask_file) > 0
                                )
                            else:
                                mask_arr = sitk.GetArrayFromImage(
                                    sitk.ReadImage(
                                        str(mask_file), sitk.sitkUInt8
                                    )
                                ).astype(bool)
                        except Exception as e:
                            logger.warning(
                                f"Failed to load mask for {stem}: {e}"
                            )
                        break

            if mask_arr is None or not np.any(mask_arr):
                continue

            # Load the synthetic image
            pred_image = self._load_image_cached(pred_path).astype(np.float64)

            # Resize pred to match mask resolution if needed
            pred_image = self._resize_array_to_target(
                pred_image, mask_arr.shape
            )

            # Create mirrored mask
            mirrored = create_mirrored_mask(pred_image, mask_arr)
            if mirrored is None:
                continue

            # Extract features from tumor ROI and mirrored ROI
            try:
                t_feats = self._extract_features_cached(
                    pred_image, str(pred_path), mask=mask_arr,
                )
                m_feats = self._extract_features_cached(
                    pred_image, f"{pred_path}__mirror", mask=mirrored,
                )
                if t_feats.size == 0 or m_feats.size == 0:
                    continue
                tumor_feats.append(t_feats)
                mirror_feats.append(m_feats)
                valid_stems.append(stem)
            except Exception as e:
                logger.warning(
                    f"Tumor ROI feature extraction failed for {stem}: {e}"
                )
                continue

        if len(tumor_feats) < 2:
            detail[f"note_{task}"] = (
                f"Too few valid cases ({len(tumor_feats)}) for tumor ROI "
                "evaluation.  Check masks and image data."
            )
            return {"aggregates": agg, "detail": detail}

        # Build combined feature matrix: tumor=1, mirror=0
        all_feats = np.vstack(
            [np.stack(tumor_feats), np.stack(mirror_feats)]
        )
        all_feats = np.nan_to_num(
            all_feats, nan=0.0, posinf=0.0, neginf=0.0
        )
        y_true = np.concatenate([
            np.ones(len(tumor_feats), dtype=np.int64),
            np.zeros(len(mirror_feats), dtype=np.int64),
        ])

        y_score = clf.predict_proba(all_feats)
        clf_result = evaluate_classification(y_true, y_score)
        agg[metric_key] = clf_result["auroc"]
        detail[f"auroc_{task}"] = clf_result["auroc"]
        detail[f"balanced_accuracy_{task}"] = clf_result["balanced_accuracy"]
        detail[f"n_cases_{task}"] = len(valid_stems)
        detail[f"classifier_type_{task}"] = "radiomics"

        return {"aggregates": agg, "detail": detail}

    # ------------------------------------------------------------------
    # Per-case metrics (backward compatibility + GC results list)
    # ------------------------------------------------------------------

    def _evaluate_pairwise_legacy(
        self, pairs: list[tuple[Path, Path]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Compute per-case image-to-image metrics.

        Returns per-case list and aggregate dict (backward compatible).
        These also populate the ``results`` list in GC output.
        """
        all_metrics: dict[str, list[float]] = {
            "mae": [], "mse": [], "nmse": [], "psnr": [], "ssim": [], "ncc": [],
        }
        case_metrics: list[dict[str, Any]] = []

        for gt_path, pred_path in tqdm(
            pairs, desc="Per-case metrics", leave=False
        ):
            logger.info(f"Evaluating: {gt_path.name}")
            try:
                metrics = self._evaluate_pair(gt_path, pred_path)
                case_metrics.append({
                    "case_id": self._get_stem(gt_path),
                    **metrics,
                })
                for key, value in metrics.items():
                    all_metrics[key].append(value)
            except Exception as e:
                logger.error(f"Error evaluating {gt_path.name}: {e}")
                continue

        aggregate = self._compute_aggregate_metrics(all_metrics)
        return case_metrics, aggregate

    # ------------------------------------------------------------------
    # Missing-data imputation (PDF §3.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _impute_missing(
        case_metrics: list[dict[str, Any]],
        aggregate: dict[str, dict[str, float]],
        missing_stems: list[str],
    ) -> None:
        """Assign worst observed score for each missing prediction.

        Per the challenge protocol: *"Missing outputs for individual
        cases are assigned the worst observed score for that metric
        prior to aggregation."*
        """
        if not case_metrics or not missing_stems:
            return

        # Determine worst observed per metric
        higher_is_better = {"psnr", "ssim", "ncc"}
        all_keys = [k for k in case_metrics[0] if k != "case_id"]
        worst: dict[str, float] = {}
        for key in all_keys:
            vals = [c[key] for c in case_metrics if isinstance(c.get(key), (int, float))]
            if not vals:
                worst[key] = float("nan")
            elif key in higher_is_better:
                worst[key] = float(min(vals))
            else:
                worst[key] = float(max(vals))

        # Add imputed cases
        for stem in missing_stems:
            case_metrics.append({"case_id": stem, **worst, "_imputed": True})

        # Recompute aggregate
        for key in all_keys:
            vals = [
                c[key]
                for c in case_metrics
                if isinstance(c.get(key), (int, float)) and np.isfinite(c[key])
            ]
            if vals:
                aggregate[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n_samples": len(vals),
                }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_image_files(self, directory: Path) -> list[Path]:
        """Get all supported image files in a directory."""
        files: list[Path] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(directory.glob(f"*{ext}"))
        return sorted(files)

    @staticmethod
    def _get_stem(path: Path) -> str:
        """Get file stem without medical image extensions."""
        name = path.name
        for ext in [".nii.gz", ".nii", ".mha", ".mhd", ".png"]:
            if name.endswith(ext):
                return name[: -len(ext)]
        return path.stem

    def _evaluate_pair(
        self,
        gt_path: Path,
        pred_path: Path,
    ) -> dict[str, float]:
        """Evaluate a single ground truth - prediction pair.

        Uses dataset-level normalization when the normalizer has been
        fitted (the standard challenge workflow).  Falls back to raw
        intensities for backward compatibility.
        """
        gt_image = self._load_image_cached(gt_path).astype(np.float64)
        pred_image = self._load_image_cached(pred_path).astype(np.float64)
        pred_image = self._resize_pred_to_gt(pred_image, gt_image)

        # Apply dataset-level z-score normalization (challenge protocol)
        if self._normalizer._fitted:
            gt_image = self._normalizer.transform(gt_image)
            pred_image = self._normalizer.transform(pred_image)

        data_range = float(np.max(gt_image) - np.min(gt_image))

        return {
            "mae": compute_mae(pred_image, gt_image),
            "mse": compute_mse(pred_image, gt_image),
            "nmse": compute_nmse(pred_image, gt_image),
            "psnr": compute_psnr(pred_image, gt_image, data_range=data_range),
            "ssim": compute_ssim(pred_image, gt_image, data_range=data_range),
            "ncc": compute_ncc(pred_image, gt_image),
        }

    @staticmethod
    def _resize_array_to_target(
        arr: NDArray,
        target_shape: tuple[int, ...],
    ) -> NDArray:
        """Resize *arr* to *target_shape* (H, W).

        Uses Pillow BICUBIC for floating-point arrays and NEAREST for
        boolean / integer arrays (masks).  Returns *arr* unchanged when
        shapes already match.
        """
        if arr.shape == target_shape:
            return arr
        try:
            from PIL import Image as _PILImg

            # PIL size is (W, H)
            target_size = (target_shape[1], target_shape[0])

            is_bool = arr.dtype == np.bool_
            if is_bool:
                # Masks: convert to uint8, resize with NEAREST, convert back
                pil_img = _PILImg.fromarray(arr.astype(np.uint8) * 255)
                pil_resized = pil_img.resize(target_size, _PILImg.NEAREST)
                resized = np.array(pil_resized, dtype=np.uint8) > 127
            elif np.issubdtype(arr.dtype, np.integer):
                pil_img = _PILImg.fromarray(arr)
                pil_resized = pil_img.resize(target_size, _PILImg.NEAREST)
                resized = np.array(pil_resized, dtype=arr.dtype)
            else:
                pil_img = _PILImg.fromarray(arr.astype(np.float64))
                pil_resized = pil_img.resize(target_size, _PILImg.BICUBIC)
                resized = np.array(pil_resized, dtype=arr.dtype)

            logger.debug(
                f"Resized array {arr.shape} -> {resized.shape} "
                f"(dtype={arr.dtype})"
            )
            return resized
        except ImportError:
            raise ValueError(
                f"Shape mismatch: {arr.shape} vs target {target_shape} "
                f"and Pillow is not available for resizing."
            )

    @staticmethod
    def _resize_pred_to_gt(
        pred: NDArray[np.floating],
        gt: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Resize *pred* to match *gt* shape when they differ.

        This handles the common case where the synthesis model outputs
        images at a fixed resolution (e.g. 512×512) while ground-truth
        slices are at the native NIfTI resolution (e.g. 448×448).
        Uses Pillow BICUBIC interpolation.  Returns *pred* unchanged
        when shapes already match.
        """
        return MamaSynthEval._resize_array_to_target(pred, gt.shape)

    @staticmethod
    def _ensure_mha(
        paths: list[str],
        tmpdir: str,
        prefix: str,
    ) -> list[str]:
        """Convert PNG files to ``.mha`` in *tmpdir* for frd-score.

        Some SimpleITK builds lack the PNG ImageIO plugin, so
        ``frd-score``'s internal ``sitk.ReadImage()`` fails on PNG
        files.  This helper converts every PNG in *paths* to ITK-native
        ``.mha`` format (always supported) in a temporary directory.
        Non-PNG files are kept as-is.

        Returns a new list of paths where each PNG has been replaced
        with its ``.mha`` counterpart.
        """
        from PIL import Image as _PILImg

        out: list[str] = []
        for i, p in enumerate(paths):
            if p.endswith(".png"):
                arr = np.array(
                    _PILImg.open(p).convert("L"), dtype=np.float32,
                )
                mha_path = Path(tmpdir) / f"{prefix}_{i:04d}.mha"
                sitk_img = sitk.GetImageFromArray(arr)
                sitk.WriteImage(sitk_img, str(mha_path))
                out.append(str(mha_path))
            else:
                out.append(p)
        return out

    @staticmethod
    def _load_image(path: Path) -> NDArray[np.floating]:
        """Load a medical image using SimpleITK, or a 16-bit PNG."""
        if path.suffix == ".png":
            try:
                from PIL import Image

                img = Image.open(path)
                return np.array(img, dtype=np.float64)
            except ImportError:
                # Fall back to SimpleITK for PNG as well
                image = sitk.ReadImage(str(path))
                return sitk.GetArrayFromImage(image).astype(np.float64)
        image = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(image)

    def _load_image_cached(self, path: Path) -> NDArray[np.floating]:
        """Load an image, reusing the per-evaluation cache when active.

        During ``evaluate()``, a cache dict is active so that the same
        NIfTI file is decompressed only once even when multiple
        evaluation tasks need it.  A *copy* is returned each time so
        that in-place mutations (e.g. ``normalize_intensity``) do not
        corrupt the cached original.
        """
        key = str(path)
        if self._image_cache is not None:
            arr = self._image_cache.get(key)
            if arr is not None:
                return arr.copy()
        arr = self._load_image(path)
        if self._image_cache is not None:
            self._image_cache[key] = arr
            return arr.copy()
        return arr

    def _load_masks(
        self, gt_paths: list[Path]
    ) -> dict[str, NDArray[np.bool_]]:
        """Load tumor segmentation masks matching ground truth files."""
        if self.masks_path is None or not self.masks_path.exists():
            return {}

        mask_files = self._get_image_files(self.masks_path)
        mask_mapping = {self._get_stem(f): f for f in mask_files}

        masks: dict[str, NDArray[np.bool_]] = {}
        for gt_path in gt_paths:
            stem = self._get_stem(gt_path)
            if stem in mask_mapping:
                mask_data = self._load_image_cached(mask_mapping[stem])
                masks[stem] = mask_data > 0
            else:
                logger.warning(f"No mask found for {stem}")

        return masks

    def _load_labels(self) -> dict[str, dict[str, int]]:
        """Load molecular subtype labels from JSON or CSV.

        **JSON** format::

            {
              "case001": {"tnbc": 0, "luminal": 1},
              "case002": {"tnbc": 1, "luminal": 0},
              ...
            }

        **CSV** format (header: ``case_id,tnbc,luminal``)::

            case_id,tnbc,luminal
            case001,0,1
            case002,1,0
        """
        if self.labels_path is None or not self.labels_path.exists():
            return {}

        path = self.labels_path
        if path.suffix == ".csv":
            labels: dict[str, dict[str, int]] = {}
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("case_id", "")
                    entry: dict[str, int] = {
                        "tnbc": int(row.get("tnbc", 0)),
                        "luminal": int(row.get("luminal", 0)),
                    }
                    # Preserve additional label columns (e.g. "contrast")
                    for key in row:
                        if key not in ("case_id", "tnbc", "luminal") and row[key] is not None:
                            try:
                                entry[key] = int(row[key])
                            except (ValueError, TypeError):
                                pass
                    labels[cid] = entry
            return labels

        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _aggregate(values: list[float]) -> dict[str, float]:
        """Compute summary statistics for a list of metric values."""
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n_samples": 0,
            }
        return {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "n_samples": len(finite),
        }

    @staticmethod
    def _compute_aggregate_metrics(
        all_metrics: dict[str, list[float]],
    ) -> dict[str, dict[str, float]]:
        """Compute aggregate statistics for legacy metrics."""
        aggregate = {}
        for metric_name, values in all_metrics.items():
            if not values:
                continue
            finite_values = [v for v in values if np.isfinite(v)]
            if finite_values:
                aggregate[metric_name] = {
                    "mean": float(np.mean(finite_values)),
                    "std": float(np.std(finite_values)),
                    "min": float(np.min(finite_values)),
                    "max": float(np.max(finite_values)),
                    "n_samples": len(finite_values),
                }
            else:
                aggregate[metric_name] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "n_samples": 0,
                }
        return aggregate

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save results to JSON file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        logger.info(f"Results saved to {self.output_file}")

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
