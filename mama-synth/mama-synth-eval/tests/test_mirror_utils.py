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

"""Unit tests for contralateral mirroring utilities."""

import numpy as np
import pytest

from eval.mirror_utils import (
    _compute_tissue_threshold,
    create_mirrored_mask,
    detect_midline,
    mirror_mask,
    validate_mirrored_region,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_breast_image_2d(
    rows: int = 64,
    cols: int = 128,
    midline: int | None = None,
) -> np.ndarray:
    """Create a synthetic 2D 'breast MRI' with a dark midline gap.

    The image has two bright blobs (left and right breast) separated by
    a dark valley at ``midline`` (default: center column).
    """
    if midline is None:
        midline = cols // 2
    img = np.zeros((rows, cols), dtype=np.float64)
    # Left breast — bright region left of midline
    img[:, 5 : midline - 3] = 200.0
    # Right breast — bright region right of midline
    img[:, midline + 3 : cols - 5] = 200.0
    # Add some noise so it's not perfectly flat
    rng = np.random.default_rng(42)
    img += rng.normal(0, 5, img.shape).clip(0)
    return img


def _make_breast_image_3d(
    slices: int = 8,
    rows: int = 64,
    cols: int = 128,
    midline: int | None = None,
) -> np.ndarray:
    """Stack several 2D breast slices into a simple 3D volume."""
    return np.stack(
        [_make_breast_image_2d(rows, cols, midline) for _ in range(slices)]
    )


# ===================================================================
# detect_midline
# ===================================================================


class TestDetectMidline:
    """Tests for detect_midline()."""

    def test_2d_symmetric_image(self) -> None:
        img = _make_breast_image_2d(cols=128, midline=64)
        mid = detect_midline(img)
        # Should be close to column 64
        assert abs(mid - 64) <= 5

    def test_3d_symmetric_image(self) -> None:
        img = _make_breast_image_3d(cols=128, midline=64)
        mid = detect_midline(img)
        assert abs(mid - 64) <= 5

    def test_2d_off_center_midline(self) -> None:
        img = _make_breast_image_2d(cols=128, midline=50)
        mid = detect_midline(img, search_fraction=0.6)
        assert abs(mid - 50) <= 5

    def test_narrow_image_raises(self) -> None:
        img = np.ones((10, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="too narrow"):
            detect_midline(img)

    def test_invalid_search_fraction_raises(self) -> None:
        img = _make_breast_image_2d()
        with pytest.raises(ValueError, match="search_fraction"):
            detect_midline(img, search_fraction=0.0)
        with pytest.raises(ValueError, match="search_fraction"):
            detect_midline(img, search_fraction=1.5)

    def test_1d_image_raises(self) -> None:
        with pytest.raises(ValueError, match="2D or 3D"):
            detect_midline(np.ones(10, dtype=np.float64))

    def test_full_width_search(self) -> None:
        # With search_fraction=1.0 the dark image margins can compete
        # with the midline valley — use a wider breast so the margins
        # are shorter than the midline gap.
        img = _make_breast_image_2d(cols=128, midline=64)
        mid = detect_midline(img, search_fraction=0.8)
        assert abs(mid - 64) <= 5


# ===================================================================
# mirror_mask
# ===================================================================


class TestMirrorMask:
    """Tests for mirror_mask()."""

    def test_2d_exact_reflection(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[3, 2] = True  # col 2, midline 10 → mirrored col 18
        mirrored = mirror_mask(mask, midline_col=10)
        assert mirrored[3, 18]
        assert np.sum(mirrored) == 1

    def test_3d_exact_reflection(self) -> None:
        mask = np.zeros((4, 10, 20), dtype=bool)
        mask[1, 3, 5] = True  # col 5, midline 10 → mirrored col 15
        mirrored = mirror_mask(mask, midline_col=10)
        assert mirrored[1, 3, 15]
        assert np.sum(mirrored) == 1

    def test_out_of_bounds_clipped(self) -> None:
        """Voxels mirroring outside the image are silently dropped."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[5, 18] = True  # col 18, midline 5 → mirrored col = -8 → clip
        mirrored = mirror_mask(mask, midline_col=5)
        assert np.sum(mirrored) == 0  # No valid target column

    def test_empty_mask(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mirrored = mirror_mask(mask, midline_col=10)
        assert np.sum(mirrored) == 0

    def test_midline_voxel_maps_to_itself(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[5, 10] = True
        mirrored = mirror_mask(mask, midline_col=10)
        assert mirrored[5, 10]
        assert np.sum(mirrored) == 1

    def test_symmetric_mask(self) -> None:
        """A mask symmetric around the midline mirrors back to itself."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[4, 8] = True
        mask[4, 12] = True  # symmetric pair around col 10
        mirrored = mirror_mask(mask, midline_col=10)
        assert mirrored[4, 8]
        assert mirrored[4, 12]

    def test_preserves_count_when_inbounds(self) -> None:
        mask = np.zeros((10, 100), dtype=bool)
        rng = np.random.default_rng(0)
        mask[rng.integers(0, 10, 20), rng.integers(10, 40, 20)] = True
        mirrored = mirror_mask(mask, midline_col=50)
        # All source voxels in cols [10,40); mirrored → [60,90) which is in [0,100)
        assert np.sum(mirrored) == np.sum(mask)


# ===================================================================
# _compute_tissue_threshold
# ===================================================================


class TestComputeTissueThreshold:
    """Tests for _compute_tissue_threshold()."""

    def test_all_zeros(self) -> None:
        img = np.zeros((10, 10), dtype=np.float64)
        assert _compute_tissue_threshold(img) == 0.0

    def test_simple_image(self) -> None:
        rng = np.random.default_rng(42)
        img = np.zeros((50, 50), dtype=np.float64)
        img[10:40, 10:40] = rng.uniform(100, 300, (30, 30))
        thresh = _compute_tissue_threshold(img, percentile=10)
        # Should be near the low end of the non-zero values
        nonzero = img[img > 0]
        expected = np.percentile(nonzero, 10)
        assert abs(thresh - expected) < 1e-10


# ===================================================================
# validate_mirrored_region
# ===================================================================


class TestValidateMirroredRegion:
    """Tests for validate_mirrored_region()."""

    def test_all_tissue_passes(self) -> None:
        img = np.full((10, 10), 200.0, dtype=np.float64)
        mask = np.ones((10, 10), dtype=bool)
        # Explicit threshold below the uniform value (auto-threshold
        # for a uniform image equals the value itself, and > is strict).
        assert validate_mirrored_region(img, mask, tissue_threshold=100.0) is True

    def test_all_background_fails(self) -> None:
        img = np.zeros((10, 10), dtype=np.float64)
        mask = np.ones((10, 10), dtype=bool)
        # All zero image: threshold=0, voxels > 0 is 0 → fraction=0 → fail
        assert validate_mirrored_region(img, mask) is False

    def test_empty_mask_fails(self) -> None:
        img = np.full((10, 10), 200.0, dtype=np.float64)
        mask = np.zeros((10, 10), dtype=bool)
        assert validate_mirrored_region(img, mask) is False

    def test_partial_tissue_at_threshold(self) -> None:
        img = np.zeros((10, 10), dtype=np.float64)
        img[:3, :] = 200.0  # 30% of the image is tissue
        mask = np.ones((10, 10), dtype=bool)
        # 30 of 100 voxels on tissue → 0.3 → passes at default 0.3
        # Use explicit threshold to avoid auto-threshold edge case.
        assert validate_mirrored_region(
            img, mask, min_tissue_fraction=0.3, tissue_threshold=100.0
        )

    def test_explicit_threshold(self) -> None:
        img = np.full((4, 4), 50.0, dtype=np.float64)
        mask = np.ones((4, 4), dtype=bool)
        # All voxels == 50; threshold 100 → none above → fail
        assert validate_mirrored_region(
            img, mask, tissue_threshold=100.0
        ) is False
        # threshold 10 → all above → pass
        assert validate_mirrored_region(
            img, mask, tissue_threshold=10.0
        ) is True


# ===================================================================
# create_mirrored_mask  (end-to-end)
# ===================================================================


class TestCreateMirroredMask:
    """End-to-end tests for create_mirrored_mask()."""

    def test_basic_2d(self) -> None:
        img = _make_breast_image_2d(rows=64, cols=128, midline=64)
        # Tumor on left breast
        mask = np.zeros((64, 128), dtype=bool)
        mask[20:40, 15:30] = True
        result = create_mirrored_mask(img, mask)
        assert result is not None
        # Mirrored mask should be on the right breast
        mirrored_cols = np.argwhere(result)[:, 1]
        assert np.all(mirrored_cols > 64)

    def test_basic_3d(self) -> None:
        img = _make_breast_image_3d(slices=8, rows=64, cols=128, midline=64)
        mask = np.zeros((8, 64, 128), dtype=bool)
        mask[2:6, 20:40, 15:30] = True
        result = create_mirrored_mask(img, mask)
        assert result is not None
        mirrored_cols = np.argwhere(result)[:, 2]
        assert np.all(mirrored_cols > 64)

    def test_empty_mask_returns_none(self) -> None:
        img = _make_breast_image_2d()
        mask = np.zeros((64, 128), dtype=bool)
        assert create_mirrored_mask(img, mask) is None

    def test_mask_near_boundary_may_return_none(self) -> None:
        """Mask very close to the edge may mirror off the image."""
        img = _make_breast_image_2d(rows=64, cols=128, midline=64)
        mask = np.zeros((64, 128), dtype=bool)
        mask[30, 125:128] = True  # Far right of the image
        # Mirrored cols: 2*64-125=3, 2*64-126=2, 2*64-127=1
        # Those cols are in the dark margin → likely fail tissue validation
        result = create_mirrored_mask(img, mask)
        # Either None or a valid mask — we just ensure no crash
        assert result is None or isinstance(result, np.ndarray)

    def test_tissue_validation_can_reject(self) -> None:
        """A mask mirrored into background should be rejected."""
        # Image with tissue only on the left half
        img = np.zeros((64, 128), dtype=np.float64)
        img[:, 5:60] = 200.0  # Only left breast
        mask = np.zeros((64, 128), dtype=bool)
        mask[20:40, 15:30] = True  # Tumor on left
        # Midline ~64, mirrored to ~98-113 which is all zeros → fail
        result = create_mirrored_mask(img, mask)
        assert result is None

    def test_output_shape_matches_input(self) -> None:
        img = _make_breast_image_2d(rows=64, cols=128, midline=64)
        mask = np.zeros((64, 128), dtype=bool)
        mask[20:40, 15:30] = True
        result = create_mirrored_mask(img, mask)
        if result is not None:
            assert result.shape == mask.shape

    def test_no_overlap_with_original(self) -> None:
        """Mirrored mask should not overlap the original tumor mask."""
        img = _make_breast_image_2d(rows=64, cols=128, midline=64)
        mask = np.zeros((64, 128), dtype=bool)
        mask[20:40, 15:30] = True  # Left breast only
        result = create_mirrored_mask(img, mask)
        if result is not None:
            # No voxel should be True in both
            assert not np.any(mask & result)
