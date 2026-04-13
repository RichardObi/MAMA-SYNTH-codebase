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
2D slice extraction from 3D DCE-MRI NIfTI volumes.

Provides automated strategies for selecting clinically relevant 2D slices
from 3D breast DCE-MRI volumes, primarily for use in the classification
pipeline. Slice extraction follows the normalisation and technical
requirements of the MAMA-SYNTH challenge:

  - Extracts axial slices from 3D NIfTI volumes.
  - Supports multiple selection strategies: largest tumour cross-section,
    tumour centre-of-mass, or multi-slice representative sampling.
  - Applies dataset-level z-score normalisation consistent with the
    evaluation pipeline.

Strategies
----------
``max_tumor``
    Select the axial slice with the largest tumour area (most foreground
    mask voxels). This maximises the visible tumour for downstream
    classification and is the recommended default.

``center_tumor``
    Select the axial slice passing through the tumour centre of mass.
    A good single-slice summary when the tumour spans many slices.

``multi_slice``
    Extract *k* equally-spaced slices spanning the tumour extent along
    the axial axis. Features from each slice are concatenated or
    aggregated, giving a richer representation of tumour heterogeneity.

``all_tumor``
    Extract **every** axial slice that contains at least one foreground
    mask voxel. Each slice becomes an independent training sample,
    maximising the amount of per-patient training data.

``middle``
    Select the middle axial slice of the volume (no mask required).
    Useful as a mask-free fallback.

Usage::

    from eval.slice_extraction import (
        extract_2d_slice,
        extract_multi_slices,
        SliceMode,
    )

    # Single best slice
    slice_2d, mask_2d, idx = extract_2d_slice(
        volume, mask, mode=SliceMode.MAX_TUMOR,
    )

    # Multiple slices
    slices, masks, indices = extract_multi_slices(
        volume, mask, n_slices=5,
    )
"""

import enum
import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "SliceMode",
    "extract_2d_slice",
    "extract_multi_slices",
    "extract_all_tumor_slices",
    "zscore_normalize_slice",
    "find_max_tumor_slice",
    "find_center_tumor_slice",
    "find_tumor_extent",
]


# ---------------------------------------------------------------------------
# Slice selection mode enum
# ---------------------------------------------------------------------------


class SliceMode(str, enum.Enum):
    """Strategy for selecting a 2D slice from a 3D volume.

    Values:
        MAX_TUMOR:    Axial slice with the largest tumour cross-section.
        CENTER_TUMOR: Axial slice through the tumour centre of mass.
        MULTI_SLICE:  Multiple equally-spaced slices across tumour extent.
        ALL_TUMOR:    Every slice with ≥1 foreground mask voxel.
        MIDDLE:       Middle axial slice (no mask required).
    """

    MAX_TUMOR = "max_tumor"
    CENTER_TUMOR = "center_tumor"
    MULTI_SLICE = "multi_slice"
    ALL_TUMOR = "all_tumor"
    MIDDLE = "middle"


# ---------------------------------------------------------------------------
# Slice-level z-score normalisation
# ---------------------------------------------------------------------------


def zscore_normalize_slice(
    image_slice: NDArray[np.floating],
    mask_slice: Optional[NDArray[np.bool_]] = None,
    clip_range: tuple[float, float] = (-5.0, 5.0),
) -> NDArray[np.floating]:
    """Apply z-score normalisation to a 2D image slice.

    Normalises to zero mean and unit variance within the mask region
    (or full image if no mask is provided), then clips extreme values.
    Consistent with the MAMA-SYNTH challenge normalisation protocol.

    Args:
        image_slice: 2D image array (H, W).
        mask_slice: Optional boolean mask for the ROI. If provided,
            statistics are computed from the masked region only.
        clip_range: Min/max clipping bounds after z-scoring.

    Returns:
        Normalised 2D array (float32).
    """
    image_slice = image_slice.astype(np.float32)

    if mask_slice is not None and np.any(mask_slice):
        roi_values = image_slice[mask_slice]
        mean = np.mean(roi_values)
        std = np.std(roi_values)
    else:
        mean = np.mean(image_slice)
        std = np.std(image_slice)

    if std < 1e-8:
        logger.debug("Near-zero std in slice normalisation; returning zeros.")
        return np.zeros_like(image_slice)

    normalised = (image_slice - mean) / std
    return np.clip(normalised, clip_range[0], clip_range[1])


# ---------------------------------------------------------------------------
# Core slice-finding helpers
# ---------------------------------------------------------------------------


def find_max_tumor_slice(
    mask: NDArray[np.bool_],
    axis: int = 0,
) -> int:
    """Find the slice index with the largest tumour cross-section.

    Args:
        mask: 3D boolean mask array (D, H, W).
        axis: Axis along which to search (0 = axial).

    Returns:
        Index of the slice with the most foreground voxels.

    Raises:
        ValueError: If the mask contains no foreground voxels.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D.")
    if not np.any(mask):
        raise ValueError("Mask contains no foreground voxels.")

    # Sum over the two non-axis dimensions to get per-slice area
    sum_axes = tuple(i for i in range(3) if i != axis)
    areas = np.sum(mask, axis=sum_axes)
    return int(np.argmax(areas))


def find_center_tumor_slice(
    mask: NDArray[np.bool_],
    axis: int = 0,
) -> int:
    """Find the slice index through the tumour centre of mass.

    Args:
        mask: 3D boolean mask array (D, H, W).
        axis: Axis for the centre-of-mass coordinate to use.

    Returns:
        Slice index closest to the tumour centroid along the given axis.

    Raises:
        ValueError: If the mask contains no foreground voxels.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D.")
    if not np.any(mask):
        raise ValueError("Mask contains no foreground voxels.")

    # Compute centre of mass along the specified axis
    coords = np.argwhere(mask)  # (N, 3)
    centroid_axis = float(np.mean(coords[:, axis]))
    return int(np.round(centroid_axis))


def find_tumor_extent(
    mask: NDArray[np.bool_],
    axis: int = 0,
) -> tuple[int, int]:
    """Find the first and last slice indices containing tumour.

    Args:
        mask: 3D boolean mask (D, H, W).
        axis: Axis along which to search.

    Returns:
        Tuple of (first_slice, last_slice) indices (inclusive).

    Raises:
        ValueError: If the mask is empty.
    """
    if not np.any(mask):
        raise ValueError("Mask contains no foreground voxels.")

    sum_axes = tuple(i for i in range(3) if i != axis)
    areas = np.sum(mask, axis=sum_axes)
    nonzero = np.nonzero(areas)[0]
    return int(nonzero[0]), int(nonzero[-1])


# ---------------------------------------------------------------------------
# Main extraction functions
# ---------------------------------------------------------------------------


def extract_2d_slice(
    volume: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]] = None,
    mode: SliceMode = SliceMode.MAX_TUMOR,
    axis: int = 0,
    normalize: bool = True,
    clip_range: tuple[float, float] = (-5.0, 5.0),
) -> tuple[NDArray[np.floating], Optional[NDArray[np.bool_]], int]:
    """Extract a single 2D slice from a 3D volume.

    Args:
        volume: 3D image array (D, H, W).
        mask: Optional 3D boolean segmentation mask.
        mode: Slice selection strategy.
        axis: Axis along which to slice (0 = axial).
        normalize: Whether to apply z-score normalisation.
        clip_range: Clipping bounds when normalising.

    Returns:
        Tuple of (image_slice, mask_slice, slice_index).
        mask_slice is None if no mask was provided.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim}D.")

    n_slices = volume.shape[axis]

    # Determine slice index based on mode
    if mode == SliceMode.MAX_TUMOR:
        if mask is None or not np.any(mask):
            logger.warning(
                "No mask available for MAX_TUMOR mode; falling back to MIDDLE."
            )
            slice_idx = n_slices // 2
        else:
            slice_idx = find_max_tumor_slice(mask, axis=axis)

    elif mode == SliceMode.CENTER_TUMOR:
        if mask is None or not np.any(mask):
            logger.warning(
                "No mask available for CENTER_TUMOR mode; falling back to MIDDLE."
            )
            slice_idx = n_slices // 2
        else:
            slice_idx = find_center_tumor_slice(mask, axis=axis)

    elif mode == SliceMode.MIDDLE:
        slice_idx = n_slices // 2

    elif mode == SliceMode.MULTI_SLICE:
        # For the single-slice API, return the centre of tumour extent
        if mask is not None and np.any(mask):
            slice_idx = find_center_tumor_slice(mask, axis=axis)
        else:
            slice_idx = n_slices // 2

    else:
        raise ValueError(f"Unknown slice mode: {mode}")

    # Clamp index
    slice_idx = max(0, min(slice_idx, n_slices - 1))

    # Extract the slice
    slicer = [slice(None)] * 3
    slicer[axis] = slice_idx
    image_slice = volume[tuple(slicer)].astype(np.float32)

    mask_slice: Optional[NDArray[np.bool_]] = None
    if mask is not None:
        mask_slice = mask[tuple(slicer)].astype(bool)

    # Normalise
    if normalize:
        image_slice = zscore_normalize_slice(
            image_slice, mask_slice=mask_slice, clip_range=clip_range
        )

    logger.debug(
        f"Extracted slice {slice_idx}/{n_slices} (mode={mode.value}, "
        f"shape={image_slice.shape})"
    )

    return image_slice, mask_slice, slice_idx


def extract_multi_slices(
    volume: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]] = None,
    n_slices: int = 5,
    axis: int = 0,
    normalize: bool = True,
    clip_range: tuple[float, float] = (-5.0, 5.0),
) -> tuple[list[NDArray[np.floating]], list[Optional[NDArray[np.bool_]]], list[int]]:
    """Extract multiple equally-spaced slices spanning the tumour extent.

    If a mask is provided, slices are sampled from the tumour extent
    (first to last foreground slice). Otherwise, slices are sampled
    from the full volume.

    Args:
        volume: 3D image array (D, H, W).
        mask: Optional 3D boolean segmentation mask.
        n_slices: Number of slices to extract.
        axis: Axis along which to slice (0 = axial).
        normalize: Whether to apply z-score normalisation.
        clip_range: Clipping bounds when normalising.

    Returns:
        Tuple of (image_slices, mask_slices, slice_indices).
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim}D.")

    # Determine range to sample from
    if mask is not None and np.any(mask):
        start, end = find_tumor_extent(mask, axis=axis)
    else:
        start, end = 0, volume.shape[axis] - 1

    # Ensure we have at least n_slices to choose from
    extent = end - start + 1
    actual_n = min(n_slices, extent)

    if actual_n <= 1:
        indices = [(start + end) // 2]
    else:
        indices = [
            int(np.round(start + i * (end - start) / (actual_n - 1)))
            for i in range(actual_n)
        ]

    # Deduplicate (can happen with very small extents)
    indices = sorted(set(indices))

    image_slices: list[NDArray[np.floating]] = []
    mask_slices: list[Optional[NDArray[np.bool_]]] = []

    for idx in indices:
        slicer = [slice(None)] * 3
        slicer[axis] = idx
        img = volume[tuple(slicer)].astype(np.float32)
        msk: Optional[NDArray[np.bool_]] = None
        if mask is not None:
            msk = mask[tuple(slicer)].astype(bool)

        if normalize:
            img = zscore_normalize_slice(img, mask_slice=msk, clip_range=clip_range)

        image_slices.append(img)
        mask_slices.append(msk)

    logger.debug(
        f"Extracted {len(indices)} slices from range [{start}, {end}] "
        f"(requested {n_slices}, axis={axis})"
    )

    return image_slices, mask_slices, indices


def extract_all_tumor_slices(
    volume: NDArray[np.floating],
    mask: NDArray[np.bool_],
    axis: int = 0,
    normalize: bool = True,
    clip_range: tuple[float, float] = (-5.0, 5.0),
) -> tuple[list[NDArray[np.floating]], list[NDArray[np.bool_]], list[int]]:
    """Extract every axial slice that contains at least one tumour voxel.

    Unlike :func:`extract_multi_slices` (which sub-samples *k* slices),
    this returns **all** slices with foreground mask pixels.  Each slice
    is intended to become an independent training sample.

    Args:
        volume: 3D image array (D, H, W).
        mask: 3D boolean segmentation mask (**required**, must contain
            at least one foreground voxel).
        axis: Axis along which to slice (0 = axial).
        normalize: Whether to apply z-score normalisation.
        clip_range: Clipping bounds when normalising.

    Returns:
        Tuple of (image_slices, mask_slices, slice_indices).

    Raises:
        ValueError: If the mask is empty or *volume* / *mask* are not 3D.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim}D.")
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D.")
    if not np.any(mask):
        raise ValueError("Mask contains no foreground voxels.")

    # Per-slice foreground count → select slices with ≥1 voxel
    sum_axes = tuple(i for i in range(3) if i != axis)
    areas = np.sum(mask, axis=sum_axes)           # shape (D,)
    indices = list(np.nonzero(areas)[0].astype(int))

    image_slices: list[NDArray[np.floating]] = []
    mask_slices: list[NDArray[np.bool_]] = []

    for idx in indices:
        slicer = [slice(None)] * 3
        slicer[axis] = idx
        img = volume[tuple(slicer)].astype(np.float32)
        msk = mask[tuple(slicer)].astype(bool)

        if normalize:
            img = zscore_normalize_slice(img, mask_slice=msk, clip_range=clip_range)

        image_slices.append(img)
        mask_slices.append(msk)

    logger.debug(
        f"Extracted {len(indices)} all-tumour slices from "
        f"{volume.shape[axis]} total (axis={axis})"
    )

    return image_slices, mask_slices, indices
