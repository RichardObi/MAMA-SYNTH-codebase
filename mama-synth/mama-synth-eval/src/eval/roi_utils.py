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
Utilities for ROI (Region of Interest) extraction from breast DCE-MRI.

Tumor-ROI based evaluation uses the tumor mask dilated by a fixed margin
to capture peri-tumoral enhancement context. This ensures evaluation
of lesion conspicuity and contrast uptake realism.

Reference: MAMA-SYNTH Challenge §Assessment Methods, Metric (3) Tumor ROI.
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

logger = logging.getLogger(__name__)


def dilate_mask(
    mask: NDArray[np.bool_],
    margin_mm: float = 10.0,
    voxel_spacing: Optional[tuple[float, ...]] = None,
) -> NDArray[np.bool_]:
    """Dilate a binary mask by a physical margin (in mm).

    Args:
        mask: Binary mask to dilate.
        margin_mm: Dilation margin in millimeters (default: 10 mm).
        voxel_spacing: Physical spacing per dimension in mm.
                       If None, unit spacing (1 mm) is assumed.

    Returns:
        Dilated binary mask.
    """
    if voxel_spacing is None:
        voxel_spacing = tuple(1.0 for _ in range(mask.ndim))

    if margin_mm <= 0.0:
        return mask.astype(bool)

    # Compute structuring element radius in voxels for each dimension
    radii = [max(1, int(np.ceil(margin_mm / sp))) for sp in voxel_spacing]

    # Create ellipsoidal structuring element
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    dist_sq = sum(
        (g / r) ** 2 for g, r in zip(grids, radii) if r > 0
    )
    structure = dist_sq <= 1.0

    return ndimage.binary_dilation(mask, structure=structure).astype(bool)


def extract_roi(
    image: NDArray[np.floating],
    mask: NDArray[np.bool_],
    margin_mm: float = 10.0,
    voxel_spacing: Optional[tuple[float, ...]] = None,
    return_mask: bool = False,
) -> NDArray[np.floating]:
    """Extract tumor-centered ROI from an image using a dilated tumor mask.

    Crops the image to the bounding box of the dilated mask and applies
    the dilated mask to zero out background voxels.

    Args:
        image: Input image array.
        mask: Binary tumor mask.
        margin_mm: Dilation margin in mm around the tumor.
        voxel_spacing: Physical spacing per dimension.
        return_mask: If True, return the dilated mask instead of the image crop.

    Returns:
        Cropped masked image (or the dilated mask if return_mask is True).
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match mask shape {mask.shape}"
        )

    if not np.any(mask):
        logger.warning("Empty tumor mask, returning original image.")
        if return_mask:
            return np.ones_like(mask)
        return image.copy()

    # Dilate the mask
    dilated = dilate_mask(mask, margin_mm=margin_mm, voxel_spacing=voxel_spacing)

    if return_mask:
        return dilated

    # Extract bounding box of dilated mask
    coords = np.argwhere(dilated)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0) + 1  # +1 for exclusive end

    slices = tuple(slice(mn, mx) for mn, mx in zip(bbox_min, bbox_max))

    # Crop and mask
    cropped_image = image[slices].copy()
    cropped_mask = dilated[slices]
    cropped_image[~cropped_mask] = 0.0

    return cropped_image


def extract_roi_pair(
    real_image: NDArray[np.floating],
    synthetic_image: NDArray[np.floating],
    mask: NDArray[np.bool_],
    margin_mm: float = 10.0,
    voxel_spacing: Optional[tuple[float, ...]] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
    """Extract corresponding ROIs from real and synthetic images.

    Ensures both extractions use the same dilated mask and bounding box
    so the results are aligned for metric computation.

    Args:
        real_image: Real post-contrast image.
        synthetic_image: Synthetic post-contrast image.
        mask: Binary tumor mask.
        margin_mm: Dilation margin in mm.
        voxel_spacing: Physical spacing per dimension.

    Returns:
        Tuple of (real_roi, synthetic_roi, roi_mask).
    """
    if real_image.shape != synthetic_image.shape:
        raise ValueError(
            f"Real shape {real_image.shape} != synthetic shape {synthetic_image.shape}"
        )
    if real_image.shape != mask.shape:
        raise ValueError(
            f"Image shape {real_image.shape} != mask shape {mask.shape}"
        )

    if not np.any(mask):
        logger.warning("Empty tumor mask, using full image as ROI.")
        full_mask = np.ones(mask.shape, dtype=bool)
        return real_image.copy(), synthetic_image.copy(), full_mask

    # Dilate
    dilated = dilate_mask(mask, margin_mm=margin_mm, voxel_spacing=voxel_spacing)

    # Bounding box
    coords = np.argwhere(dilated)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0) + 1

    slices = tuple(slice(mn, mx) for mn, mx in zip(bbox_min, bbox_max))

    real_roi = real_image[slices].copy()
    synth_roi = synthetic_image[slices].copy()
    roi_mask = dilated[slices]

    # Zero out background
    real_roi[~roi_mask] = 0.0
    synth_roi[~roi_mask] = 0.0

    return real_roi, synth_roi, roi_mask
