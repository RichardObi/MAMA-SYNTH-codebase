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
classification task.  Given a tumor segmentation mask on one breast,
mirror it across the body midline to create a contralateral ROI that
serves as a "non-tumor" reference region for binary classification
(tumor ROI vs contralateral non-tumor ROI).

**Midline detection (robust)** works in three steps:

1. Build a tissue-only column profile by masking out background air
   (voxels below ``BACKGROUND_Z_THRESHOLD``) and computing the mean
   over tissue pixels per column.  This prevents background air columns
   at the image edges and cardiac/thorax enhancement in the centre from
   biasing the minimum.

2. Smooth the profile with a box filter to suppress noise.

3. Detect the two largest breast-tissue peaks, one in each image half
   (bilateral breast check / **D2**).  The midline is the valley
   (argmin) *between* those two peaks rather than the global minimum
   of the entire central window.  Searching between identified breast
   peaks means the result is correct even when the heart or thorax
   region has high contrast-enhanced intensity in the centre.

**Orientation-invariant fallback (D4)**: ``create_mirrored_mask`` first
attempts to mirror along columns (the nominal left-right axis).  If the
bilateral check fails or tissue validation fails it automatically retries
along rows (the nominal cranio-caudal axis).  This keeps the pipeline
robust to 90°/270°-rotated inputs.

**3-D support**: when a 3-D volume is passed to ``create_mirrored_mask``
(as done during classifier training), the bilateral check and midline
detection run on a representative 2-D projection (max over the first
axis); the resulting midline and mirroring axis are then applied to every
slice of the 3-D mask identically.

**Validation** ensures that the accepted mirrored mask overlaps with
breast tissue.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================

DEFAULT_MIN_TISSUE_FRACTION = 0.3
"""Minimum fraction of mirrored mask voxels that must overlap with
tissue for the mirrored region to be considered valid."""

DEFAULT_TISSUE_THRESHOLD_PERCENTILE = 10
"""Percentile of non-zero image intensities used to determine the
tissue/background boundary."""

DEFAULT_MIDLINE_SEARCH_FRACTION = 0.4
"""Fraction of the image width (centered) within which to search
for the midline valley (legacy / fallback use only)."""

BACKGROUND_Z_THRESHOLD: float = -1.5
"""Intensity threshold separating background from breast tissue in
z-score normalised images.  Background (air) consistently falls below
−2σ; glandular tissue starts at approximately −1.5σ and above."""

_PROFILE_SMOOTH_BINS: int = 20
"""Number of bins used to define the box-filter kernel size relative
to the profile length (kernel = max(3, profile_length // bins))."""

_MIN_PEAK_HEIGHT_FRACTION: float = 0.1
"""A column peak is only considered a breast if its tissue-mean
intensity is at least this fraction of the global tissue-mean
intensity.  Eliminates spurious peaks at the image periphery."""

_MIN_PEAK_DISTANCE_FRACTION: float = 0.1
"""Two peaks must be separated by at least this fraction of the
profile length to be considered distinct breast peaks."""

_MIN_PEAK_PROMINENCE_FRACTION: float = 0.05
"""The valley between the two accepted peaks must be at least this
fraction *below* the lower of the two peak values (relative to the
global tissue mean).  This rejects flat profiles where all positions
have essentially the same intensity — characteristic of a uniform
single-breast or background image where the row/column profile
has no structural variation."""


# ======================================================================
# Internal helpers
# ======================================================================


def _tissue_profile(
    image: NDArray[np.floating],
    reduce_axis: int,
) -> NDArray[np.floating]:
    """Background-masked, smoothed mean profile.

    Computes the mean intensity over tissue pixels (> ``BACKGROUND_Z_THRESHOLD``)
    along *reduce_axis*, yielding one value per position along the
    remaining axis.  NaN is used where a column/row contains no tissue pixels.

    The result is smoothed with a box filter to suppress pixel-level noise.

    Parameters
    ----------
    image : NDArray
        2-D float image.
    reduce_axis : int
        The axis to average over (0 = average rows → column profile;
        1 = average columns → row profile).
    """
    tissue_mask = image > BACKGROUND_Z_THRESHOLD
    count = np.sum(tissue_mask, axis=reduce_axis).astype(np.float64)
    total = np.where(tissue_mask, image, 0.0).sum(axis=reduce_axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(count > 0, total / count, np.nan)

    # Box-filter smoothing ignoring NaNs
    n = len(profile)
    k = max(3, n // _PROFILE_SMOOTH_BINS)
    valid = (~np.isnan(profile)).astype(np.float64)
    filled = np.where(np.isnan(profile), 0.0, profile)
    kernel = np.ones(k) / k
    sm_sum = np.convolve(filled, kernel, mode="same")
    sm_cnt = np.convolve(valid, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(sm_cnt > 0, sm_sum / sm_cnt, np.nan)

    return smoothed.astype(np.float64)


def _find_local_maxima(
    profile: NDArray[np.floating],
    min_height: float,
    min_distance: int,
) -> list[int]:
    """Return indices of local maxima above *min_height* spaced ≥ *min_distance* apart.

    Uses a simple left-neighbour / right-neighbour comparison after NaN
    positions are skipped, then greedily filters candidates that are too
    close (keeping the taller one).
    """
    n = len(profile)
    candidates: list[tuple[int, float]] = []

    for i in range(1, n - 1):
        v = profile[i]
        if np.isnan(v) or v < min_height:
            continue
        left = profile[i - 1] if not np.isnan(profile[i - 1]) else -np.inf
        right = profile[i + 1] if not np.isnan(profile[i + 1]) else -np.inf
        if v >= left and v >= right:
            candidates.append((i, float(v)))

    # Greedy merge: if two peaks are closer than min_distance, keep taller
    merged: list[tuple[int, float]] = []
    for idx, val in candidates:
        if merged and idx - merged[-1][0] < min_distance:
            if val > merged[-1][1]:
                merged[-1] = (idx, val)
        else:
            merged.append((idx, val))

    return [idx for idx, _ in merged]


def detect_bilateral_breasts(
    image: NDArray[np.floating],
    mirror_axis: int = 1,
) -> tuple[Optional[tuple[int, int]], str]:
    """Detect two breast peaks in the tissue profile along *mirror_axis*.

    Returns the pair of peak positions ``(left_peak, right_peak)`` where
    "left" and "right" refer to the first and second halves of the profile
    along *mirror_axis*.

    Parameters
    ----------
    image : NDArray
        2-D float image (z-score normalised).
    mirror_axis : int
        The axis along which mirroring will be performed.
        1 → look for left/right breasts in a column profile (reduce rows).
        0 → look for top/bottom breasts in a row profile (reduce columns).

    Returns
    -------
    (peaks, reason) : tuple
        *peaks* is ``(peak_a, peak_b)`` on success or ``None`` on failure.
        *reason* is an empty string on success or a human-readable
        description of the failure for diagnostic logging.
    """
    reduce_axis = 1 - mirror_axis  # axis we average over
    profile = _tissue_profile(image, reduce_axis=reduce_axis)
    n = len(profile)

    # Global tissue mean as height baseline (ignoring NaN background)
    tissue_vals = profile[~np.isnan(profile)]
    if tissue_vals.size == 0:
        return None, (
            "tissue profile is entirely NaN — image may be all-background "
            "or not z-score normalised"
        )

    global_mean = float(np.nanmean(profile))
    min_height = global_mean * _MIN_PEAK_HEIGHT_FRACTION
    min_distance = max(3, int(n * _MIN_PEAK_DISTANCE_FRACTION))

    peaks = _find_local_maxima(profile, min_height=min_height, min_distance=min_distance)

    # Partition peaks into OUTER ZONES (lateral breast regions) and inner centre.
    outer_zone = max(1, n // 4)
    half = n // 2

    left_peaks = [p for p in peaks if p < outer_zone]
    right_peaks = [p for p in peaks if p >= n - outer_zone]

    if not left_peaks:
        left_peaks = [p for p in peaks if p < half]
    if not right_peaks:
        right_peaks = [p for p in peaks if p >= half]

    # Last-resort: argmax of the outer zone
    left_outer_profile = profile[:outer_zone]
    right_outer_profile = profile[n - outer_zone:]
    if not left_peaks and not np.all(np.isnan(left_outer_profile)):
        left_peaks = [int(np.nanargmax(left_outer_profile))]
    if not right_peaks and not np.all(np.isnan(right_outer_profile)):
        right_peaks = [(n - outer_zone) + int(np.nanargmax(right_outer_profile))]

    if not left_peaks and not right_peaks:
        return None, (
            f"no tissue peaks found in profile (axis={mirror_axis}); "
            "image may show a single breast, incorrect orientation, or "
            "have no bilateral structure"
        )

    if not left_peaks:
        return None, (
            f"no breast peak found in first half of profile (axis={mirror_axis}); "
            "image likely shows a single breast or is rotated — "
            f"only right-half peaks at positions {right_peaks}"
        )

    if not right_peaks:
        return None, (
            f"no breast peak found in second half of profile (axis={mirror_axis}); "
            "image likely shows a single breast or is rotated — "
            f"only left-half peaks at positions {left_peaks}"
        )

    # Take the tallest peak from each half
    peak_a = max(left_peaks, key=lambda i: float(profile[i]))
    peak_b = max(right_peaks, key=lambda i: float(profile[i]))

    # Prominence check: valley between peaks must be meaningfully lower
    lo_idx, hi_idx = min(peak_a, peak_b), max(peak_a, peak_b)
    valley_slice = profile[lo_idx: hi_idx + 1]
    valley_val = (
        float(np.nanmin(valley_slice))
        if not np.all(np.isnan(valley_slice))
        else float(profile[peak_a])
    )
    required_valley_max = global_mean * (1.0 - _MIN_PEAK_PROMINENCE_FRACTION)
    if valley_val >= required_valley_max:
        return None, (
            f"valley intensity {valley_val:.4f} >= required_max {required_valley_max:.4f} "
            f"(global_tissue_mean={global_mean:.3f}, axis={mirror_axis}, "
            f"peak_a={peak_a} val={float(profile[peak_a]):.3f}, "
            f"peak_b={peak_b} val={float(profile[peak_b]):.3f}); "
            "profile is flat or valley is not below tissue mean — "
            "likely single-breast or uniform-tissue image"
        )

    return (peak_a, peak_b), ""


def detect_midline(
    image: NDArray[np.floating],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
) -> int:
    """Detect the body midline column using a robust peak-valley strategy.

    This is the **primary midline detection entry point**.  It:

    1. Builds a tissue-only (background-masked), smoothed column profile.
    2. Calls :func:`detect_bilateral_breasts` to locate the two breast peaks
       (one in each image half).
    3. Returns the median column of the low-intensity valley *between* those
       two peaks (falls back to argmin if no clear valley columns exist).

    Falls back to the legacy ``_detect_midline_argmin`` implementation when
    the bilateral breast check fails (e.g. single-breast or rotated image),
    so that callers always receive a column index.

    Supports both 2-D images and 3-D volumes (the middle axial slice is
    used for bilateral detection in the 3-D case).

    Parameters
    ----------
    image : NDArray
        2-D or 3-D float image (z-score normalised).
    search_fraction : float
        Used only in the legacy fallback path.

    Returns
    -------
    int
        Column index of the detected midline.

    Raises
    ------
    ValueError
        If ``search_fraction`` is out of range or the image has wrong ndim.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got {image.ndim}D")
    img2d = image if image.ndim == 2 else image[image.shape[0] // 2]

    if not 0 < search_fraction <= 1:
        raise ValueError(
            f"search_fraction must be in (0, 1], got {search_fraction}"
        )

    n_cols = img2d.shape[1]
    if n_cols < 4:
        raise ValueError(
            f"Image too narrow for midline detection ({n_cols} columns)"
        )

    peaks, reason = detect_bilateral_breasts(img2d, mirror_axis=1)
    if peaks is not None:
        peak_a, peak_b = peaks
        lo, hi = min(peak_a, peak_b), max(peak_a, peak_b)
        profile = _tissue_profile(img2d, reduce_axis=0)
        inter = profile[lo: hi + 1]

        # Replace NaN with local minimum so background columns are part of valley
        finite_min = float(np.nanmin(inter)) if not np.all(np.isnan(inter)) else 0.0
        finite_mean = float(np.nanmean(inter)) if not np.all(np.isnan(inter)) else 0.0
        inter_filled = np.where(np.isnan(inter), finite_min, inter)

        # Use median of low-intensity valley columns (avoids argmin left-bias)
        valley_threshold = finite_min + 0.5 * (finite_mean - finite_min)
        valley_cols = np.where(inter_filled <= valley_threshold)[0]
        if valley_cols.size > 0:
            valley_midline = lo + int(np.median(valley_cols))
        else:
            valley_midline = lo + int(np.argmin(inter_filled))

        # Sanity-check: if valley is outside central 40% of inter-peak range,
        # fall back to geometric midpoint of the two peaks.
        inter_len = hi - lo
        inner_lo = lo + int(0.3 * inter_len)
        inner_hi = lo + int(0.7 * inter_len)
        if inner_lo <= valley_midline <= inner_hi:
            return valley_midline
        return (peak_a + peak_b) // 2

    # Fallback: legacy argmin in search window
    logger.debug(
        "detect_midline: bilateral check failed (%s); using legacy argmin fallback.",
        reason,
    )
    return _detect_midline_argmin(img2d, search_fraction=search_fraction)


def _detect_midline_argmin(
    image: NDArray[np.floating],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
) -> int:
    """Legacy midline detector: argmin of the raw column mean in a central window.

    Only used as a fallback inside :func:`detect_midline` when the bilateral
    breast check fails.  Exposed for unit tests.
    """
    if not 0 < search_fraction <= 1:
        raise ValueError(
            f"search_fraction must be in (0, 1], got {search_fraction}"
        )
    col_profile = np.mean(image, axis=0)
    n_cols = len(col_profile)
    if n_cols < 4:
        raise ValueError(
            f"Image too narrow for midline detection ({n_cols} columns)"
        )
    half_width = int(n_cols * search_fraction / 2)
    center = n_cols // 2
    lo = max(0, center - half_width)
    hi = min(n_cols, center + half_width)
    window = col_profile[lo:hi]
    return lo + int(np.argmin(window))


def mirror_mask(
    mask: NDArray[np.bool_],
    midline: int,
    axis: int = 1,
) -> NDArray[np.bool_]:
    """Mirror a binary mask about a midline position along *axis*.

    Each True voxel at position ``p`` along *axis* is mapped to position
    ``2 * midline - p``.  Positions that map outside the image boundaries
    are silently dropped.

    Supports both 2-D masks (used in gc-eval / slice-based extraction)
    and 3-D masks (used in training via ``create_tumor_roi_dataset``).
    For 3-D masks, mirroring is applied to the *last two* spatial axes
    using *axis* = 1 (column-left-right) or *axis* = 0 (row-cranio-caudal)
    relative to the 2-D slice dimensions (slice index is preserved).

    Parameters
    ----------
    mask : NDArray[bool]
        2-D or 3-D binary mask.
    midline : int
        The reflection point (column index for axis=1, row index for axis=0).
    axis : int
        The spatial axis along which to mirror.
        1 = columns / left-right (default).
        0 = rows / cranio-caudal.

    Returns
    -------
    NDArray[bool]
        Mirrored mask with the same shape as the input.
    """
    if mask.ndim not in (2, 3):
        raise ValueError(f"mask must be 2-D or 3-D, got {mask.ndim}D")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    mirrored = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mirrored

    if mask.ndim == 2:
        n = mask.shape[axis]
        mirrored_pos = 2 * midline - coords[:, axis]
        valid = (mirrored_pos >= 0) & (mirrored_pos < n)
        coords_valid = coords[valid].copy()
        coords_valid[:, axis] = mirrored_pos[valid]
        mirrored[coords_valid[:, 0], coords_valid[:, 1]] = True
    else:
        # 3-D: coords are (z, row, col); spatial 2D axes are at offsets 1 and 2
        spatial_axis = axis + 1  # 0→1 (rows), 1→2 (cols) within the 3D array
        n = mask.shape[spatial_axis]
        mirrored_pos = 2 * midline - coords[:, spatial_axis]
        valid = (mirrored_pos >= 0) & (mirrored_pos < n)
        coords_valid = coords[valid].copy()
        coords_valid[:, spatial_axis] = mirrored_pos[valid]
        mirrored[coords_valid[:, 0], coords_valid[:, 1], coords_valid[:, 2]] = True

    return mirrored


# ======================================================================
# Tissue threshold + validation
# ======================================================================


def _compute_tissue_threshold(
    image: NDArray[np.floating],
    percentile: float = DEFAULT_TISSUE_THRESHOLD_PERCENTILE,
) -> float:
    """Tissue/background threshold from a low percentile of tissue voxels.

    Filters to pixels above ``BACKGROUND_Z_THRESHOLD`` (valid for
    z-score normalised images where background air < −2σ) and returns
    the given percentile as a conservative lower bound on tissue signal.
    """
    tissue = image[image > BACKGROUND_Z_THRESHOLD]
    if tissue.size == 0:
        return BACKGROUND_Z_THRESHOLD
    return float(np.percentile(tissue, percentile))


def validate_mirrored_region(
    image: NDArray[np.floating],
    mirrored_mask: NDArray[np.bool_],
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
    tissue_threshold: Optional[float] = None,
) -> bool:
    """Validate that a mirrored mask overlaps with actual breast tissue.

    Checks that at least ``min_tissue_fraction`` of the mirrored mask
    voxels have image intensities above the tissue threshold.

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
        from the image using :func:`_compute_tissue_threshold`.

    Returns
    -------
    bool
        True if the mirrored region passes validation.
    """
    n_mask_voxels = int(np.sum(mirrored_mask))
    if n_mask_voxels == 0:
        return False

    if tissue_threshold is None:
        tissue_threshold = _compute_tissue_threshold(image)

    tissue_overlap = int(np.sum(image[mirrored_mask] >= tissue_threshold))
    fraction = tissue_overlap / n_mask_voxels

    logger.debug(
        "Mirror validation: %d/%d voxels on tissue (%.1f%%), "
        "threshold=%.3f, required≥%.0f%%",
        tissue_overlap, n_mask_voxels, fraction * 100,
        tissue_threshold, min_tissue_fraction * 100,
    )

    return fraction >= min_tissue_fraction


# ======================================================================
# Public entry point
# ======================================================================


def create_mirrored_mask(
    image: NDArray[np.floating],
    mask: NDArray[np.bool_],
    search_fraction: float = DEFAULT_MIDLINE_SEARCH_FRACTION,
    min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION,
    case_id: str = "",
) -> Optional[NDArray[np.bool_]]:
    """Create a contralateral mirrored mask with bilateral check and axis fallback.

    Algorithm (D2 + D4):

    1. **Bilateral breast check** (D2): verify that the image contains two
       breast-like tissue peaks, one in each half, along the candidate mirror
       axis.  If not found, skip immediately with an informative reason.

    2. **Primary attempt** (axis=1, columns / left-right): detect the midline,
       mirror the tumour mask, validate tissue overlap.

    3. **Orientation-invariant fallback** (D4, axis=0, rows / cranio-caudal):
       if step 2 fails (bilateral check or tissue validation), retry with the
       perpendicular axis.  Handles images rotated 90° or 270°.

    4. Return the first successfully validated mirror, or ``None`` with a
       detailed reason logged at WARNING level.

    **3-D inputs** (used during training): the bilateral check and midline
    detection run on a representative 2-D projection (middle axial slice);
    the resolved axis and midline are then applied to mirror the full 3-D
    mask identically across all slices.

    Parameters
    ----------
    image : NDArray
        2-D or 3-D float image (z-score normalised).
    mask : NDArray[bool]
        Binary tumour segmentation mask (same shape as ``image``).
    search_fraction : float
        Used only in the legacy midline-fallback path.
    min_tissue_fraction : float
        Minimum tissue overlap fraction for validation.
    case_id : str
        Case identifier included in all warning messages for traceability.

    Returns
    -------
    NDArray[bool] | None
        The mirrored mask, or ``None`` if both axes fail.
    """
    prefix = f"[{case_id}] " if case_id else ""

    if not np.any(mask):
        logger.warning(
            "%screate_mirrored_mask: tumour mask is empty — "
            "case cannot contribute to tumour-ROI AUROC.",
            prefix,
        )
        return None

    # For 3-D volumes: project to the middle axial slice for bilateral detection
    img2d = image if image.ndim == 2 else image[image.shape[0] // 2]

    tissue_threshold = _compute_tissue_threshold(image)

    for axis, axis_label in ((1, "columns/left-right"), (0, "rows/cranio-caudal")):
        # ---- D2: bilateral breast check --------------------------------
        peaks, bilateral_reason = detect_bilateral_breasts(img2d, mirror_axis=axis)
        if peaks is None:
            logger.warning(
                "%screate_mirrored_mask: bilateral breast check FAILED "
                "(axis=%d [%s]): %s",
                prefix, axis, axis_label, bilateral_reason,
            )
            continue  # try next axis (D4 fallback)

        peak_a, peak_b = peaks

        # ---- Midline detection (robust peak-valley) --------------------
        if axis == 1:
            midline = detect_midline(img2d, search_fraction=search_fraction)
        else:
            midline = detect_midline(img2d.T, search_fraction=search_fraction)

        # ---- Mirror and validate ---------------------------------------
        mirrored = mirror_mask(mask, midline, axis=axis)

        if not np.any(mirrored):
            # Determine bbox from the 2-D slice (last two dims for 3-D)
            if mask.ndim == 2:
                mask2d = mask
            else:
                mask2d = mask[mask.shape[0] // 2]
            if np.any(mask2d):
                col_coords = np.where(mask2d)[1]
                col_min, col_max = int(col_coords.min()), int(col_coords.max())
            else:
                col_min, col_max = -1, -1
            logger.warning(
                "%screate_mirrored_mask: mirrored mask is entirely empty "
                "(axis=%d [%s], midline=%d). "
                "Tumour mask may be too close to the image boundary "
                "(tumour bbox cols %d–%d, image width %d).",
                prefix, axis, axis_label, midline,
                col_min, col_max, img2d.shape[1],
            )
            continue  # try next axis

        n_mirrored = int(np.sum(mirrored))
        tissue_overlap = int(np.sum(image[mirrored] >= tissue_threshold))
        tissue_frac = tissue_overlap / n_mirrored

        if tissue_frac >= min_tissue_fraction:
            if axis == 1:
                logger.debug(
                    "%screate_mirrored_mask: success on primary axis "
                    "(axis=1 [%s], midline_col=%d, breast_peaks=%s, "
                    "tissue_overlap=%.1f%%).",
                    prefix, axis_label, midline, peaks, tissue_frac * 100,
                )
            else:
                logger.warning(
                    "%screate_mirrored_mask: primary axis (columns) FAILED; "
                    "fallback axis=0 [%s] SUCCEEDED "
                    "(midline_row=%d, breast_peaks=%s, tissue_overlap=%.1f%%). "
                    "Image appears to be rotated 90°/270° relative to "
                    "the expected axial orientation.",
                    prefix, axis_label, midline, peaks, tissue_frac * 100,
                )
            return mirrored

        logger.warning(
            "%screate_mirrored_mask: tissue validation FAILED "
            "(axis=%d [%s], midline=%d, breast_peaks=%s, "
            "tissue_overlap=%.1f%% < required %.1f%%). "
            "Contralateral region may be in body/background.",
            prefix, axis, axis_label, midline, peaks,
            tissue_frac * 100, min_tissue_fraction * 100,
        )
        # continue → try next axis (D4)

    logger.warning(
        "%screate_mirrored_mask: ALL axes FAILED — case dropped from "
        "tumour-ROI AUROC. "
        "Possible causes: single-breast FOV, severe image rotation, "
        "non-z-score-normalised intensities, tumour at image boundary.",
        prefix,
    )
    return None
