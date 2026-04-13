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

"""Contralateral mirroring utilities for breast DCE-MRI.

Provides midline detection and mask mirroring for the ``tumor_roi``
classification task.  The key idea: given a tumor segmentation mask on
one breast, mirror it across the body midline to create a contralateral
ROI that serves as a "non-tumor" reference region for binary
classification (tumor ROI vs contralateral non-tumor ROI).

**Midline detection** works by computing the column-wise mean intensity
of the breast tissue and finding the valley (local minimum) in the
central region of the image — this valley corresponds to the gap
between the two breasts in axial breast MRI slices.

**Validation** ensures that the mirrored mask actually overlaps with
breast tissue (i.e. is not out-of-body or in the background).
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_MIN_TISSUE_FRACTION = 0.3
"""Minimum fraction of mirrored mask voxels that must overlap with
tissue for the mirrored region to be considered valid."""

DEFAULT_TISSUE_THRESHOLD_PERCENTILE = 10
"""Percentile of non-zero image intensities used to determine the
tissue/background boundary.  Voxels above this threshold are
considered tissue."""

DEFAULT_MIDLINE_SEARCH_FRACTION = 0.4
"""Fraction of the image width (centered) within which to search
for the midline valley.  E.g. 0.4 means we search the central 40%
of columns."""


def detect_midline(
    image: NDArray[np.floating],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
) -> int:
    """Detect the body midline in an axial breast MRI slice.

    Computes the column-wise mean intensity and finds the column with
    the minimum average intensity in the central region of the image.
    In breast MRI, this intensity valley corresponds to the gap between
    the two breasts (sternum/midline region).

    Parameters
    ----------
    image : NDArray
        2D or 3D image array.  If 3D, the column profile is computed
        across all slices and rows.
    search_fraction : float
        Fraction of the image width (centered) within which to search
        for the midline.  Must be in ``(0, 1]``.

    Returns
    -------
    int
        Column index of the detected midline.

    Raises
    ------
    ValueError
        If the image has fewer than 4 columns or ``search_fraction`` is
        out of range.
    """
    if not 0 < search_fraction <= 1:
        raise ValueError(
            f"search_fraction must be in (0, 1], got {search_fraction}"
        )

    # Work on a 2D projection: average along all axes except the last (columns)
    if image.ndim == 3:
        # (slices, rows, cols) → average over slices and rows
        col_profile = np.mean(image, axis=(0, 1))
    elif image.ndim == 2:
        # (rows, cols) → average over rows
        col_profile = np.mean(image, axis=0)
    else:
        raise ValueError(
            f"image must be 2D or 3D, got {image.ndim}D"
        )

    n_cols = len(col_profile)
    if n_cols < 4:
        raise ValueError(
            f"Image too narrow for midline detection ({n_cols} columns)"
        )

    # Search window: central ``search_fraction`` of columns
    half_width = int(n_cols * search_fraction / 2)
    center = n_cols // 2
    lo = max(0, center - half_width)
    hi = min(n_cols, center + half_width)

    # Find column with minimum mean intensity in the search window
    window = col_profile[lo:hi]
    midline_col = lo + int(np.argmin(window))

    logger.debug(
        f"Midline detected at column {midline_col} "
        f"(search window [{lo}, {hi}), n_cols={n_cols})"
    )

    return midline_col


def mirror_mask(
    mask: NDArray[np.bool_],
    midline_col: int,
) -> NDArray[np.bool_]:
    """Mirror a binary mask horizontally about a given midline column.

    Each True voxel at column ``c`` is mapped to column
    ``2 * midline_col - c``.  Voxels that map outside the image
    boundaries are silently clipped.

    Parameters
    ----------
    mask : NDArray[bool]
        2D or 3D binary mask.
    midline_col : int
        Column index of the midline.

    Returns
    -------
    NDArray[bool]
        Mirrored mask with the same shape as the input.
    """
    n_cols = mask.shape[-1]
    mirrored = np.zeros_like(mask, dtype=bool)

    # Get coordinates of all True voxels
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mirrored

    # Mirror the column coordinate
    col_idx = -1  # last axis is always columns
    mirrored_cols = 2 * midline_col - coords[:, col_idx]

    # Keep only voxels that land within bounds
    valid = (mirrored_cols >= 0) & (mirrored_cols < n_cols)
    coords_valid = coords[valid].copy()
    coords_valid[:, col_idx] = mirrored_cols[valid]

    # Set mirrored voxels
    if mask.ndim == 2:
        mirrored[coords_valid[:, 0], coords_valid[:, 1]] = True
    elif mask.ndim == 3:
        mirrored[
            coords_valid[:, 0],
            coords_valid[:, 1],
            coords_valid[:, 2],
        ] = True
    else:
        raise ValueError(f"mask must be 2D or 3D, got {mask.ndim}D")

    return mirrored


def _compute_tissue_threshold(
    image: NDArray[np.floating],
    percentile: float = DEFAULT_TISSUE_THRESHOLD_PERCENTILE,
) -> float:
    """Compute an intensity threshold separating tissue from background.

    Uses a percentile of the non-zero intensity values.  This is robust
    to the large background (zero/near-zero) region typical in MRI.

    Parameters
    ----------
    image : NDArray
        Image array.
    percentile : float
        Percentile of non-zero values to use as threshold.

    Returns
    -------
    float
        Tissue threshold intensity.
    """
    nonzero = image[image > 0]
    if nonzero.size == 0:
        return 0.0
    return float(np.percentile(nonzero, percentile))


def validate_mirrored_region(
    image: NDArray[np.floating],
    mirrored_mask: NDArray[np.bool_],
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
    tissue_threshold: Optional[float] = None,
) -> bool:
    """Validate that a mirrored mask overlaps with actual breast tissue.

    Checks that at least ``min_tissue_fraction`` of the mirrored mask
    voxels have image intensities above the tissue threshold (i.e. are
    not background).

    Parameters
    ----------
    image : NDArray
        Image array (same shape as ``mirrored_mask``).
    mirrored_mask : NDArray[bool]
        The mirrored binary mask to validate.
    min_tissue_fraction : float
        Required minimum fraction of mask voxels overlapping tissue.
    tissue_threshold : float | None
        Explicit tissue threshold.  If ``None``, computed automatically
        from the image.

    Returns
    -------
    bool
        True if the mirrored region passes validation.
    """
    n_mask_voxels = int(np.sum(mirrored_mask))
    if n_mask_voxels == 0:
        logger.debug("Mirrored mask is empty (0 voxels).")
        return False

    if tissue_threshold is None:
        tissue_threshold = _compute_tissue_threshold(image)

    # Count how many mirrored voxels fall on tissue
    tissue_overlap = int(np.sum(image[mirrored_mask] > tissue_threshold))
    fraction = tissue_overlap / n_mask_voxels

    logger.debug(
        f"Mirror validation: {tissue_overlap}/{n_mask_voxels} voxels on "
        f"tissue ({fraction:.1%}), threshold={tissue_threshold:.1f}, "
        f"required≥{min_tissue_fraction:.0%}"
    )

    return fraction >= min_tissue_fraction


def create_mirrored_mask(
    image: NDArray[np.floating],
    mask: NDArray[np.bool_],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
) -> Optional[NDArray[np.bool_]]:
    """Create a contralateral mirrored mask with validation.

    High-level orchestration: detects the midline, mirrors the mask,
    then validates that the mirrored region falls on breast tissue.

    Parameters
    ----------
    image : NDArray
        2D or 3D image array.
    mask : NDArray[bool]
        Binary tumor segmentation mask (same shape as ``image``).
    search_fraction : float
        Passed to :func:`detect_midline`.
    min_tissue_fraction : float
        Passed to :func:`validate_mirrored_region`.

    Returns
    -------
    NDArray[bool] | None
        The mirrored mask if validation passes, or ``None`` if it fails.
    """
    if not np.any(mask):
        logger.warning("Input mask is empty, cannot create mirror.")
        return None

    midline = detect_midline(image, search_fraction=search_fraction)
    mirrored = mirror_mask(mask, midline)

    if not np.any(mirrored):
        logger.warning(
            f"Mirrored mask is empty (midline={midline}). "
            "Mask may be too close to the image boundary."
        )
        return None

    if validate_mirrored_region(
        image, mirrored, min_tissue_fraction=min_tissue_fraction
    ):
        return mirrored

    logger.info(
        f"Mirrored mask failed tissue validation "
        f"(midline={midline}, min_tissue_fraction={min_tissue_fraction})."
    )
    return None
