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
Image quality metrics for evaluating synthetic DCE-MRI images.

This module provides standard metrics for comparing generated post-contrast
MRI images against ground truth images. Includes image-to-image metrics
(MSE, PSNR, SSIM, NCC, MAE, NMSE), segmentation metrics (Dice, HD95),
and perceptual metrics (LPIPS).
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

logger = logging.getLogger(__name__)


def _validate_inputs(
    prediction: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
) -> None:
    """Validate that prediction and ground_truth have the same shape.

    Args:
        prediction: Predicted image array.
        ground_truth: Ground truth image array.

    Raises:
        ValueError: If shapes do not match or arrays are empty.
    """
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )
    if prediction.size == 0:
        raise ValueError("Input arrays cannot be empty")


# ---------------------------------------------------------------------------
# Image-to-image metrics
# ---------------------------------------------------------------------------


def compute_mae(
    prediction: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute Mean Absolute Error between prediction and ground truth.

    Args:
        prediction: Predicted image array.
        ground_truth: Ground truth image array.
        mask: Optional binary mask to restrict computation to specific region.

    Returns:
        Mean absolute error value.
    """
    _validate_inputs(prediction, ground_truth)

    if mask is not None:
        prediction = prediction[mask]
        ground_truth = ground_truth[mask]

    return float(np.mean(np.abs(prediction - ground_truth)))


def compute_mse(
    prediction: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute Mean Squared Error between prediction and ground truth.

    Args:
        prediction: Predicted image array.
        ground_truth: Ground truth image array.
        mask: Optional binary mask to restrict computation to specific region.

    Returns:
        Mean squared error value.
    """
    _validate_inputs(prediction, ground_truth)

    if mask is not None:
        prediction = prediction[mask]
        ground_truth = ground_truth[mask]

    return float(np.mean((prediction - ground_truth) ** 2))


def compute_nmse(
    prediction: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute Normalized Mean Squared Error between prediction and ground truth.

    NMSE = MSE / variance(ground_truth)

    Args:
        prediction: Predicted image array.
        ground_truth: Ground truth image array.
        mask: Optional binary mask to restrict computation to specific region.

    Returns:
        Normalized mean squared error value.
    """
    _validate_inputs(prediction, ground_truth)

    if mask is not None:
        prediction = prediction[mask]
        ground_truth = ground_truth[mask]

    mse = float(np.mean((prediction - ground_truth) ** 2))
    variance = float(np.var(ground_truth))

    if variance == 0:
        return 0.0 if mse == 0 else float("inf")

    return mse / variance


def compute_psnr(
    prediction: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
    data_range: Optional[float] = None,
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio between prediction and ground truth.

    PSNR = 10 * log10(data_range^2 / MSE)

    Args:
        prediction: Predicted image array.
        ground_truth: Ground truth image array.
        data_range: The data range of the input images. If None, computed as
                   max(ground_truth) - min(ground_truth).
        mask: Optional binary mask to restrict computation to specific region.

    Returns:
        PSNR value in decibels (dB). Returns inf if MSE is 0.
    """
    _validate_inputs(prediction, ground_truth)

    if mask is not None:
        prediction = prediction[mask]
        ground_truth = ground_truth[mask]

    if data_range is None:
        data_range = float(np.max(ground_truth) - np.min(ground_truth))

    mse = float(np.mean((prediction - ground_truth) ** 2))

    if mse == 0:
        return float("inf")

    return float(10 * np.log10((data_range ** 2) / mse))


def compute_ssim(
    prediction: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
    data_range: Optional[float] = None,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute Structural Similarity Index between prediction and ground truth.

    This is a simplified SSIM implementation using global image statistics
    rather than local window-based computation. For 3D volumes, SSIM is
    computed slice-by-slice and averaged.

    Args:
        prediction: Predicted image array (2D or 3D).
        ground_truth: Ground truth image array (2D or 3D).
        data_range: The data range of the input images. If None, computed as
                   max(ground_truth) - min(ground_truth).
        win_size: Kept for API compatibility (unused in this global implementation).
        k1: Constant for luminance comparison (default: 0.01).
        k2: Constant for contrast comparison (default: 0.03).

    Returns:
        SSIM value in range [-1, 1], where 1 means identical images.
    """
    _validate_inputs(prediction, ground_truth)

    if data_range is None:
        data_range = float(np.max(ground_truth) - np.min(ground_truth))

    # Handle edge case where data_range is 0
    if data_range == 0:
        return 1.0 if np.allclose(prediction, ground_truth) else 0.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # For 3D volumes, compute SSIM slice by slice and average
    if prediction.ndim == 3:
        ssim_values = []
        for i in range(prediction.shape[0]):
            ssim_val = _compute_ssim_2d(
                prediction[i], ground_truth[i], c1, c2, win_size
            )
            ssim_values.append(ssim_val)
        return float(np.mean(ssim_values))
    elif prediction.ndim == 2:
        return _compute_ssim_2d(prediction, ground_truth, c1, c2, win_size)
    else:
        raise ValueError(f"SSIM only supports 2D or 3D arrays, got {prediction.ndim}D")


def _compute_ssim_2d(
    prediction: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
    c1: float,
    c2: float,
    win_size: int,
) -> float:
    """Compute global SSIM for 2D images.

    This is a simplified SSIM implementation using global image statistics
    rather than local window-based computation. The win_size parameter is
    accepted for API compatibility but not used in this implementation.

    Args:
        prediction: 2D predicted image array.
        ground_truth: 2D ground truth image array.
        c1: Constant for luminance comparison.
        c2: Constant for contrast comparison.
        win_size: Size of local window (unused in this global implementation).

    Returns:
        Global SSIM value for the image.
    """
    # Using global statistics for a simplified, robust implementation
    _ = win_size  # Explicitly mark as unused
    mu_pred = np.mean(prediction)
    mu_gt = np.mean(ground_truth)
    sigma_pred = np.std(prediction)
    sigma_gt = np.std(ground_truth)
    sigma_pred_gt = np.mean((prediction - mu_pred) * (ground_truth - mu_gt))

    numerator = (2 * mu_pred * mu_gt + c1) * (2 * sigma_pred_gt + c2)
    denominator = (mu_pred ** 2 + mu_gt ** 2 + c1) * (sigma_pred ** 2 + sigma_gt ** 2 + c2)

    return float(numerator / denominator)


def compute_ncc(
    prediction: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute Normalized Cross-Correlation between prediction and ground truth.

    NCC = sum((pred - mean(pred)) * (gt - mean(gt))) /
          (std(pred) * std(gt) * N)

    Args:
        prediction: Predicted image array.
        ground_truth: Ground truth image array.
        mask: Optional binary mask to restrict computation to specific region.

    Returns:
        NCC value in range [-1, 1], where 1 means perfect positive correlation.
    """
    _validate_inputs(prediction, ground_truth)

    if mask is not None:
        prediction = prediction[mask]
        ground_truth = ground_truth[mask]

    pred_mean = np.mean(prediction)
    gt_mean = np.mean(ground_truth)

    pred_std = np.std(prediction)
    gt_std = np.std(ground_truth)

    # Handle edge case where standard deviation is 0
    if pred_std == 0 or gt_std == 0:
        return 1.0 if pred_std == gt_std and np.allclose(prediction, ground_truth) else 0.0

    numerator = np.mean((prediction - pred_mean) * (ground_truth - gt_mean))
    denominator = pred_std * gt_std

    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------


def compute_dice(
    prediction: NDArray[np.bool_],
    ground_truth: NDArray[np.bool_],
) -> float:
    """Compute Dice Similarity Coefficient (DSC) between two binary masks.

    DSC = 2 * |P ∩ G| / (|P| + |G|)

    Args:
        prediction: Predicted binary segmentation mask.
        ground_truth: Ground truth binary segmentation mask.

    Returns:
        DSC value in [0, 1], where 1 means perfect overlap.
    """
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )

    intersection = np.sum(prediction & ground_truth)
    sum_masks = np.sum(prediction) + np.sum(ground_truth)

    if sum_masks == 0:
        return 1.0  # Both masks are empty — perfect agreement

    return float(2.0 * intersection / sum_masks)


def compute_hd95(
    prediction: NDArray[np.bool_],
    ground_truth: NDArray[np.bool_],
    voxel_spacing: Optional[tuple[float, ...]] = None,
) -> float:
    """Compute 95th-percentile Hausdorff Distance (HD95) between two masks.

    HD95 is the 95th percentile of the symmetric surface distances,
    which is more robust to outliers than the maximum Hausdorff Distance.

    Args:
        prediction: Predicted binary segmentation mask.
        ground_truth: Ground truth binary segmentation mask.
        voxel_spacing: Physical spacing per dimension (e.g., (1.0, 0.875, 0.875)).
                       If None, unit spacing is assumed.

    Returns:
        HD95 in physical units. Returns 0.0 when both masks are empty.
        Returns inf when exactly one mask is empty.
    """
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )

    pred_sum = np.sum(prediction)
    gt_sum = np.sum(ground_truth)

    # Edge cases
    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return float("inf")

    if voxel_spacing is None:
        voxel_spacing = tuple(1.0 for _ in range(prediction.ndim))

    # Compute distance transforms
    # Distance from every background voxel to the nearest foreground voxel
    pred_border = prediction ^ ndimage.binary_erosion(prediction)
    gt_border = ground_truth ^ ndimage.binary_erosion(ground_truth)

    # If erosion yields empty borders (single-voxel masks), use the mask itself
    if not np.any(pred_border):
        pred_border = prediction
    if not np.any(gt_border):
        gt_border = ground_truth

    # Distance transform of inverse masks gives distance from every voxel to
    # nearest foreground voxel
    dt_pred = ndimage.distance_transform_edt(~prediction, sampling=voxel_spacing)
    dt_gt = ndimage.distance_transform_edt(~ground_truth, sampling=voxel_spacing)

    # Surface distances: distance from each surface point to the other surface
    surf_dist_pred_to_gt = dt_gt[pred_border]
    surf_dist_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate([surf_dist_pred_to_gt, surf_dist_gt_to_pred])

    return float(np.percentile(all_distances, 95))


# ---------------------------------------------------------------------------
# Perceptual metric (LPIPS)
# ---------------------------------------------------------------------------


def compute_lpips(
    prediction: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
    net: str = "alex",
) -> float:
    """Compute Learned Perceptual Image Patch Similarity (LPIPS).

    LPIPS measures perceptual distance between images using deep features.
    Lower values indicate higher perceptual similarity.

    Requires the ``lpips`` and ``torch`` packages (optional dependencies).

    Args:
        prediction: Predicted image array (2D: H×W or 3D: slices×H×W).
        ground_truth: Ground truth image array (same shape as prediction).
        net: Backbone network for LPIPS ('alex', 'vgg', or 'squeeze').

    Returns:
        Mean LPIPS value across slices. Lower is better.

    Raises:
        ImportError: If torch or lpips packages are not installed.
    """
    try:
        import torch
        import lpips as lpips_mod
    except ImportError as exc:
        raise ImportError(
            "LPIPS computation requires 'torch' and 'lpips' packages. "
            "Install them with: pip install torch lpips"
        ) from exc

    _validate_inputs(prediction, ground_truth)

    # Normalize to [-1, 1] (LPIPS expected range)
    def _normalize(img: NDArray[np.floating]) -> NDArray[np.floating]:
        mn, mx = float(np.min(img)), float(np.max(img))
        if mx - mn == 0:
            return np.zeros_like(img)
        return 2.0 * (img - mn) / (mx - mn) - 1.0

    prediction_norm = _normalize(prediction)
    ground_truth_norm = _normalize(ground_truth)

    # Create LPIPS model (cached on module level for efficiency)
    loss_fn = lpips_mod.LPIPS(net=net, verbose=False)
    loss_fn.eval()

    if prediction.ndim == 2:
        slices_pred = [prediction_norm]
        slices_gt = [ground_truth_norm]
    elif prediction.ndim == 3:
        slices_pred = [prediction_norm[i] for i in range(prediction_norm.shape[0])]
        slices_gt = [ground_truth_norm[i] for i in range(ground_truth_norm.shape[0])]
    else:
        raise ValueError(f"LPIPS supports 2D or 3D arrays, got {prediction.ndim}D")

    lpips_values = []
    with torch.no_grad():
        for s_pred, s_gt in zip(slices_pred, slices_gt):
            # LPIPS expects (N, 3, H, W) — replicate grayscale to 3 channels
            t_pred = torch.from_numpy(s_pred).float().unsqueeze(0).unsqueeze(0)
            t_pred = t_pred.expand(-1, 3, -1, -1)
            t_gt = torch.from_numpy(s_gt).float().unsqueeze(0).unsqueeze(0)
            t_gt = t_gt.expand(-1, 3, -1, -1)

            val = loss_fn(t_pred, t_gt)
            lpips_values.append(float(val.item()))

    return float(np.mean(lpips_values))
