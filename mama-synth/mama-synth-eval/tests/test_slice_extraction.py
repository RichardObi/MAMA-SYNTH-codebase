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
Tests for 2D slice extraction from 3D DCE-MRI NIfTI volumes.
"""

import numpy as np
import pytest

from eval.slice_extraction import (
    SliceMode,
    extract_2d_slice,
    extract_multi_slices,
    find_center_tumor_slice,
    find_max_tumor_slice,
    find_tumor_extent,
    zscore_normalize_slice,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def volume_3d():
    """Create a simple 3D volume (10 slices, 32x32)."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 32, 32).astype(np.float64)


@pytest.fixture
def mask_3d():
    """Create a 3D mask with tumor concentrated in slices 3–6."""
    mask = np.zeros((10, 32, 32), dtype=bool)
    # Small tumor in slice 3
    mask[3, 10:14, 10:14] = True  # 16 voxels
    # Larger tumor in slice 4
    mask[4, 8:18, 8:18] = True   # 100 voxels
    # Largest tumor in slice 5
    mask[5, 6:22, 6:22] = True   # 256 voxels
    # Medium tumor in slice 6
    mask[6, 10:18, 10:18] = True  # 64 voxels
    return mask


@pytest.fixture
def empty_mask():
    """Create an empty mask (no foreground)."""
    return np.zeros((10, 32, 32), dtype=bool)


# ---------------------------------------------------------------------------
# Tests: SliceMode enum
# ---------------------------------------------------------------------------


class TestSliceMode:
    def test_values(self):
        assert SliceMode.MAX_TUMOR.value == "max_tumor"
        assert SliceMode.CENTER_TUMOR.value == "center_tumor"
        assert SliceMode.MULTI_SLICE.value == "multi_slice"
        assert SliceMode.MIDDLE.value == "middle"

    def test_from_string(self):
        assert SliceMode("max_tumor") == SliceMode.MAX_TUMOR
        assert SliceMode("center_tumor") == SliceMode.CENTER_TUMOR

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            SliceMode("invalid_mode")


# ---------------------------------------------------------------------------
# Tests: zscore_normalize_slice
# ---------------------------------------------------------------------------


class TestZscoreNormalizeSlice:
    def test_basic_normalization(self):
        img = np.array([[100.0, 200.0], [300.0, 400.0]])
        normed = zscore_normalize_slice(img)
        # Should have roughly zero mean
        assert abs(np.mean(normed)) < 1e-10

    def test_with_mask(self):
        img = np.array([[100.0, 200.0], [300.0, 400.0]])
        mask = np.array([[True, False], [False, False]])
        normed = zscore_normalize_slice(img, mask_slice=mask)
        # The masked pixel (100.0) should be at 0 since it's the only one
        assert abs(normed[0, 0]) < 1e-10 or normed[0, 0] == 0.0

    def test_constant_image_returns_zeros(self):
        img = np.ones((16, 16)) * 42.0
        normed = zscore_normalize_slice(img)
        assert np.allclose(normed, 0.0)

    def test_clipping(self):
        img = np.array([[0.0, 1000.0], [-1000.0, 0.0]])
        normed = zscore_normalize_slice(img, clip_range=(-3.0, 3.0))
        assert normed.min() >= -3.0
        assert normed.max() <= 3.0

    def test_output_shape(self):
        img = np.random.randn(64, 64)
        normed = zscore_normalize_slice(img)
        assert normed.shape == (64, 64)

    def test_empty_mask_uses_full_image(self):
        img = np.random.randn(16, 16)
        mask = np.zeros((16, 16), dtype=bool)
        normed = zscore_normalize_slice(img, mask_slice=mask)
        assert normed.shape == (16, 16)


# ---------------------------------------------------------------------------
# Tests: find_max_tumor_slice
# ---------------------------------------------------------------------------


class TestFindMaxTumorSlice:
    def test_finds_largest_slice(self, mask_3d):
        idx = find_max_tumor_slice(mask_3d)
        assert idx == 5  # Slice 5 has the most foreground (256 voxels)

    def test_empty_mask_raises(self, empty_mask):
        with pytest.raises(ValueError, match="no foreground"):
            find_max_tumor_slice(empty_mask)

    def test_wrong_dimensions(self):
        mask_2d = np.ones((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="3D"):
            find_max_tumor_slice(mask_2d)

    def test_single_slice_tumor(self):
        mask = np.zeros((8, 16, 16), dtype=bool)
        mask[2, 5:10, 5:10] = True
        assert find_max_tumor_slice(mask) == 2

    def test_custom_axis(self):
        mask = np.zeros((8, 16, 16), dtype=bool)
        # Put tumor at column index 10 (axis=2)
        mask[2:5, 4:8, 10] = True
        idx = find_max_tumor_slice(mask, axis=2)
        assert idx == 10


# ---------------------------------------------------------------------------
# Tests: find_center_tumor_slice
# ---------------------------------------------------------------------------


class TestFindCenterTumorSlice:
    def test_center_of_symmetric_tumor(self):
        mask = np.zeros((10, 32, 32), dtype=bool)
        mask[3:8, 10:20, 10:20] = True  # Slices 3-7, center ~5.0
        idx = find_center_tumor_slice(mask)
        assert idx == 5

    def test_center_of_asymmetric_tumor(self, mask_3d):
        idx = find_center_tumor_slice(mask_3d)
        # Weighted towards slice 5 (largest area)
        assert 4 <= idx <= 6

    def test_empty_mask_raises(self, empty_mask):
        with pytest.raises(ValueError, match="no foreground"):
            find_center_tumor_slice(empty_mask)


# ---------------------------------------------------------------------------
# Tests: find_tumor_extent
# ---------------------------------------------------------------------------


class TestFindTumorExtent:
    def test_extent(self, mask_3d):
        start, end = find_tumor_extent(mask_3d)
        assert start == 3
        assert end == 6

    def test_single_slice(self):
        mask = np.zeros((10, 16, 16), dtype=bool)
        mask[7, 2:5, 2:5] = True
        start, end = find_tumor_extent(mask)
        assert start == 7
        assert end == 7

    def test_empty_mask_raises(self, empty_mask):
        with pytest.raises(ValueError, match="no foreground"):
            find_tumor_extent(empty_mask)


# ---------------------------------------------------------------------------
# Tests: extract_2d_slice
# ---------------------------------------------------------------------------


class TestExtract2dSlice:
    def test_max_tumor_mode(self, volume_3d, mask_3d):
        img, msk, idx = extract_2d_slice(
            volume_3d, mask_3d, mode=SliceMode.MAX_TUMOR
        )
        assert idx == 5
        assert img.ndim == 2
        assert img.shape == (32, 32)
        assert msk is not None
        assert msk.shape == (32, 32)

    def test_center_tumor_mode(self, volume_3d, mask_3d):
        img, msk, idx = extract_2d_slice(
            volume_3d, mask_3d, mode=SliceMode.CENTER_TUMOR
        )
        assert 3 <= idx <= 6
        assert img.ndim == 2

    def test_middle_mode(self, volume_3d):
        img, msk, idx = extract_2d_slice(
            volume_3d, mode=SliceMode.MIDDLE
        )
        assert idx == 5  # 10 // 2
        assert img.ndim == 2
        assert msk is None

    def test_middle_mode_no_mask(self, volume_3d):
        img, msk, idx = extract_2d_slice(
            volume_3d, mask=None, mode=SliceMode.MIDDLE
        )
        assert msk is None

    def test_fallback_when_no_mask(self, volume_3d):
        """MAX_TUMOR falls back to MIDDLE when mask is None."""
        img, msk, idx = extract_2d_slice(
            volume_3d, mask=None, mode=SliceMode.MAX_TUMOR
        )
        assert idx == 5  # Middle slice

    def test_fallback_when_empty_mask(self, volume_3d, empty_mask):
        """MAX_TUMOR falls back to MIDDLE when mask is empty."""
        img, msk, idx = extract_2d_slice(
            volume_3d, mask=empty_mask, mode=SliceMode.MAX_TUMOR
        )
        assert idx == 5

    def test_normalization_applied(self, volume_3d, mask_3d):
        img, _, _ = extract_2d_slice(
            volume_3d, mask_3d, mode=SliceMode.MAX_TUMOR, normalize=True
        )
        # Should be clipped within default range
        assert img.min() >= -5.0
        assert img.max() <= 5.0

    def test_normalization_disabled(self, volume_3d, mask_3d):
        img, _, _ = extract_2d_slice(
            volume_3d, mask_3d, mode=SliceMode.MAX_TUMOR, normalize=False
        )
        # Raw values, not clipped
        assert img.dtype == np.float64

    def test_wrong_dimensions(self):
        vol_2d = np.random.randn(32, 32)
        with pytest.raises(ValueError, match="3D"):
            extract_2d_slice(vol_2d, mode=SliceMode.MIDDLE)

    def test_slice_index_clamped(self, volume_3d):
        """Index should never be out of bounds."""
        # Create a mask at the very last slice
        mask = np.zeros((10, 32, 32), dtype=bool)
        mask[9, 15:17, 15:17] = True
        img, _, idx = extract_2d_slice(volume_3d, mask, mode=SliceMode.MAX_TUMOR)
        assert 0 <= idx < 10


# ---------------------------------------------------------------------------
# Tests: extract_multi_slices
# ---------------------------------------------------------------------------


class TestExtractMultiSlices:
    def test_basic_extraction(self, volume_3d, mask_3d):
        imgs, msks, indices = extract_multi_slices(
            volume_3d, mask_3d, n_slices=3
        )
        assert len(imgs) >= 1
        assert len(imgs) == len(msks) == len(indices)
        for img in imgs:
            assert img.ndim == 2
            assert img.shape == (32, 32)

    def test_indices_within_tumor_extent(self, volume_3d, mask_3d):
        _, _, indices = extract_multi_slices(
            volume_3d, mask_3d, n_slices=4
        )
        # Tumor is in slices 3-6
        for idx in indices:
            assert 3 <= idx <= 6

    def test_no_mask_uses_full_range(self, volume_3d):
        _, _, indices = extract_multi_slices(
            volume_3d, mask=None, n_slices=5
        )
        assert indices[0] == 0
        assert indices[-1] == 9

    def test_more_slices_than_extent(self, volume_3d, mask_3d):
        """Requesting more slices than available extent."""
        # Tumor extent is 4 slices (3-6)
        imgs, _, indices = extract_multi_slices(
            volume_3d, mask_3d, n_slices=20
        )
        assert len(indices) <= 5  # At most extent+1 unique indices

    def test_single_slice(self, volume_3d, mask_3d):
        imgs, _, indices = extract_multi_slices(
            volume_3d, mask_3d, n_slices=1
        )
        assert len(indices) == 1

    def test_normalization(self, volume_3d, mask_3d):
        imgs, _, _ = extract_multi_slices(
            volume_3d, mask_3d, n_slices=3, normalize=True
        )
        for img in imgs:
            assert img.min() >= -5.0
            assert img.max() <= 5.0

    def test_wrong_dimensions(self):
        vol_2d = np.random.randn(32, 32)
        with pytest.raises(ValueError, match="3D"):
            extract_multi_slices(vol_2d, n_slices=3)
