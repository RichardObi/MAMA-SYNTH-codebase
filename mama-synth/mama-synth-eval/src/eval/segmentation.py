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
Segmentation evaluation for tumor delineation on synthetic DCE-MRI.

Evaluates whether synthesized post-contrast images retain sufficient
tumor-related information for automated segmentation. A fixed segmentation
model is applied to synthesized images and predicted masks are compared
against radiologist-verified ground-truth annotations.

Metrics:
  - Dice Similarity Coefficient (DSC)
  - 95th-percentile Hausdorff Distance (HD95)

Reference: MAMA-SYNTH Challenge §Assessment Methods, Metric (2) Segmentation.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np
from numpy.typing import NDArray

from mama_sia_eval.metrics import compute_dice, compute_hd95

logger = logging.getLogger(__name__)


class SegmentationModel(Protocol):
    """Protocol for segmentation models compatible with evaluation.

    The model should accept a single image and return a binary mask.
    """

    def predict(self, image: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Segment the image and return a binary mask."""
        ...


class ThresholdSegmenter:
    """Simple threshold-based segmenter for baseline local evaluation.

    Segments enhanced regions by thresholding intensity values.
    This is provided as a minimal baseline; the actual challenge uses
    a fixed, pre-trained nnUNet-based segmentation model maintained
    by the organizers.

    Args:
        threshold_percentile: Percentile of image intensities used as
                              the segmentation threshold (default: 90).
        min_size: Minimum connected component size in voxels (default: 10).
    """

    def __init__(
        self,
        threshold_percentile: float = 90.0,
        min_size: int = 10,
    ) -> None:
        self.threshold_percentile = threshold_percentile
        self.min_size = min_size

    def predict(self, image: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Segment enhanced regions using intensity thresholding.

        Args:
            image: Input image array (2D or 3D).

        Returns:
            Binary segmentation mask.
        """
        from scipy import ndimage as ndi

        threshold = np.percentile(image, self.threshold_percentile)
        mask = image > threshold

        # Remove small components
        if np.any(mask):
            labeled, n_features = ndi.label(mask)
            for label_id in range(1, n_features + 1):
                component = labeled == label_id
                if np.sum(component) < self.min_size:
                    mask[component] = False

        return mask.astype(bool)


class NNUNetSegmenter:
    """Wrapper around a pre-trained nnUNet model for tumor segmentation.

    The MAMA-SYNTH challenge uses an organizer-trained nnUNet model applied
    identically to all submissions.  This wrapper calls ``nnUNetv2_predict``
    under the hood, writing a temporary NIfTI file and reading back the
    predicted segmentation.

    Args:
        model_dir: Path to the nnUNet model directory (``nnUNetTrainer__…``).
        fold: Which fold to use (default: ``"all"`` for the ensemble).
        device: ``"cuda"`` or ``"cpu"``.

    Raises:
        ImportError: If ``nnunetv2`` is not installed.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        fold: str = "all",
        device: str = "cuda",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.fold = fold
        self.device = device
        self._predictor = None

    def _lazy_init(self) -> None:
        """Initialise the nnUNet predictor on first use."""
        if self._predictor is not None:
            return
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError as exc:
            raise ImportError(
                "NNUNetSegmenter requires 'nnunetv2'. "
                "Install it with: pip install nnunetv2"
            ) from exc

        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=self.device,
            verbose=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            str(self.model_dir),
            use_folds=(self.fold,) if self.fold != "all" else "all",
        )
        logger.info(f"nnUNet predictor initialised from {self.model_dir}")

    def predict(self, image: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Segment a single image.

        Args:
            image: 2D or 3D image array.

        Returns:
            Binary segmentation mask.
        """
        import SimpleITK as sitk

        self._lazy_init()
        assert self._predictor is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "input"
            out_dir = Path(tmpdir) / "output"
            in_dir.mkdir()
            out_dir.mkdir()

            # nnUNet expects a 3-D NIfTI with channel suffix ``_0000``
            arr = image.astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            sitk_img = sitk.GetImageFromArray(arr)
            sitk.WriteImage(sitk_img, str(in_dir / "tmp_0000.nii.gz"))

            self._predictor.predict_from_files(
                str(in_dir), str(out_dir),
                save_probabilities=False,
            )

            pred_path = out_dir / "tmp.nii.gz"
            if not pred_path.exists():
                logger.warning("nnUNet produced no output. Returning empty mask.")
                return np.zeros(image.shape, dtype=bool)
            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))

        mask = pred > 0
        # Squeeze back to 2-D if input was 2-D
        if image.ndim == 2 and mask.ndim == 3:
            mask = mask[0]
        return mask.astype(bool)


def evaluate_segmentation_pair(
    pred_mask: NDArray[np.bool_],
    gt_mask: NDArray[np.bool_],
    voxel_spacing: Optional[tuple[float, ...]] = None,
) -> dict[str, float]:
    """Compute segmentation metrics for a single prediction-ground truth pair.

    Args:
        pred_mask: Predicted binary segmentation mask.
        gt_mask: Ground truth binary segmentation mask.
        voxel_spacing: Physical spacing per dimension for HD95 computation.

    Returns:
        Dictionary with 'dice' and 'hd95' values.
    """
    dice = compute_dice(pred_mask, gt_mask)
    hd95 = compute_hd95(pred_mask, gt_mask, voxel_spacing=voxel_spacing)

    return {"dice": dice, "hd95": hd95}


def evaluate_segmentation(
    synthetic_images: list[NDArray[np.floating]],
    gt_masks: list[NDArray[np.bool_]],
    model: Optional[SegmentationModel] = None,
    pred_masks: Optional[list[NDArray[np.bool_]]] = None,
    voxel_spacing: Optional[tuple[float, ...]] = None,
) -> dict[str, list[float]]:
    """Evaluate segmentation performance on synthesized images.

    Either provides ``pred_masks`` directly, or applies ``model`` to
    ``synthetic_images`` to generate them.

    Args:
        synthetic_images: List of synthetic post-contrast images.
        gt_masks: List of ground-truth segmentation masks.
        model: Segmentation model to apply to synthetic images.
               Used only when pred_masks is None.
        pred_masks: Precomputed predicted masks. Takes precedence over model.
        voxel_spacing: Physical spacing per dimension.

    Returns:
        Dictionary with 'dice' and 'hd95' lists (one value per case).

    Raises:
        ValueError: If neither model nor pred_masks is provided, or
                    if list lengths are mismatched.
    """
    if pred_masks is None and model is None:
        raise ValueError("Must provide either pred_masks or a segmentation model")

    n_cases = len(gt_masks)
    if pred_masks is not None:
        if len(pred_masks) != n_cases:
            raise ValueError(
                f"Number of predicted masks ({len(pred_masks)}) does not "
                f"match number of ground truth masks ({n_cases})"
            )
    else:
        if len(synthetic_images) != n_cases:
            raise ValueError(
                f"Number of synthetic images ({len(synthetic_images)}) does not "
                f"match number of ground truth masks ({n_cases})"
            )

    dice_scores: list[float] = []
    hd95_scores: list[float] = []

    for i in range(n_cases):
        if pred_masks is not None:
            p_mask = pred_masks[i]
        else:
            assert model is not None
            p_mask = model.predict(synthetic_images[i])

        result = evaluate_segmentation_pair(
            p_mask, gt_masks[i], voxel_spacing=voxel_spacing
        )
        dice_scores.append(result["dice"])
        hd95_scores.append(result["hd95"])

        logger.debug(f"Case {i}: DSC={result['dice']:.4f}, HD95={result['hd95']:.2f}")

    logger.info(
        f"Segmentation eval complete: mean DSC={np.mean(dice_scores):.4f}, "
        f"mean HD95={np.mean([h for h in hd95_scores if np.isfinite(h)]):.2f}"
    )

    return {"dice": dice_scores, "hd95": hd95_scores}
