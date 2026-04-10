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

"""Unit tests for evaluation metrics (image-to-image, segmentation, LPIPS)."""

import numpy as np
import pytest

from eval.metrics import (
    compute_dice,
    compute_hd95,
    compute_mae,
    compute_mse,
    compute_ncc,
    compute_nmse,
    compute_psnr,
    compute_ssim,
)


# ===================================================================
# Image-to-image metrics
# ===================================================================


class TestMAE:
    """Tests for Mean Absolute Error metric."""

    def test_identical_images(self) -> None:
        img = np.random.rand(10, 10).astype(np.float64)
        assert compute_mae(img, img) == 0.0

    def test_known_difference(self) -> None:
        pred = np.ones((10, 10), dtype=np.float64)
        gt = np.zeros((10, 10), dtype=np.float64)
        assert compute_mae(pred, gt) == 1.0

    def test_with_mask(self) -> None:
        pred = np.ones((10, 10), dtype=np.float64)
        gt = np.zeros((10, 10), dtype=np.float64)
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        assert compute_mae(pred, gt, mask=mask) == 1.0

    def test_shape_mismatch_raises(self) -> None:
        pred = np.ones((10, 10), dtype=np.float64)
        gt = np.ones((5, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_mae(pred, gt)

    def test_realistic_dce(self, img_2d_real, img_2d_synth) -> None:
        """MAE between real and synthetic DCE should be reasonable."""
        mae = compute_mae(img_2d_synth, img_2d_real)
        assert mae > 0
        assert mae < 500  # Not wildly different


class TestMSE:
    """Tests for Mean Squared Error metric."""

    def test_identical_images(self) -> None:
        img = np.random.rand(10, 10).astype(np.float64)
        assert compute_mse(img, img) == 0.0

    def test_known_difference(self) -> None:
        pred = np.ones((10, 10), dtype=np.float64) * 2
        gt = np.zeros((10, 10), dtype=np.float64)
        assert compute_mse(pred, gt) == 4.0

    def test_realistic_dce(self, img_2d_real, img_2d_synth) -> None:
        mse = compute_mse(img_2d_synth, img_2d_real)
        assert mse > 0


class TestNMSE:
    """Tests for Normalized Mean Squared Error metric."""

    def test_identical_images(self) -> None:
        img = np.random.rand(10, 10).astype(np.float64) + 1
        assert compute_nmse(img, img) == 0.0

    def test_zero_variance_identical(self) -> None:
        img = np.ones((10, 10), dtype=np.float64)
        assert compute_nmse(img, img) == 0.0

    def test_zero_variance_different(self) -> None:
        pred = np.ones((10, 10), dtype=np.float64) * 2
        gt = np.ones((10, 10), dtype=np.float64)
        assert compute_nmse(pred, gt) == float("inf")


class TestPSNR:
    """Tests for Peak Signal-to-Noise Ratio metric."""

    def test_identical_images(self) -> None:
        img = np.random.rand(10, 10).astype(np.float64)
        assert compute_psnr(img, img) == float("inf")

    def test_known_value(self) -> None:
        pred = np.zeros((10, 10), dtype=np.float64)
        gt = np.ones((10, 10), dtype=np.float64)
        psnr = compute_psnr(pred, gt, data_range=1.0)
        assert psnr == pytest.approx(0.0)

    def test_with_data_range(self) -> None:
        pred = np.ones((10, 10), dtype=np.float64) * 0.5
        gt = np.zeros((10, 10), dtype=np.float64)
        psnr = compute_psnr(pred, gt, data_range=1.0)
        expected = 10 * np.log10(1.0 / 0.25)
        assert psnr == pytest.approx(expected)

    def test_realistic_dce(self, img_2d_real, img_2d_synth) -> None:
        psnr = compute_psnr(img_2d_synth, img_2d_real)
        assert psnr > 0
        assert np.isfinite(psnr)


class TestSSIM:
    """Tests for Structural Similarity Index metric."""

    def test_identical_images(self) -> None:
        img = np.random.rand(10, 10).astype(np.float64)
        ssim = compute_ssim(img, img)
        assert ssim == pytest.approx(1.0)

    def test_3d_images(self) -> None:
        img = np.random.rand(5, 10, 10).astype(np.float64)
        ssim = compute_ssim(img, img)
        assert ssim == pytest.approx(1.0)

    def test_zero_data_range_identical(self) -> None:
        img = np.ones((10, 10), dtype=np.float64)
        assert compute_ssim(img, img) == 1.0

    def test_different_images_lower_ssim(self) -> None:
        img1 = np.random.rand(10, 10).astype(np.float64)
        img2 = np.random.rand(10, 10).astype(np.float64)
        ssim = compute_ssim(img1, img2)
        assert ssim < 1.0

    def test_realistic_dce_ssim_reasonable(self, img_2d_real, img_2d_synth) -> None:
        ssim = compute_ssim(img_2d_synth, img_2d_real)
        # Synthetic should be somewhat similar but not perfect
        assert 0.0 < ssim < 1.0


class TestNCC:
    """Tests for Normalized Cross-Correlation metric."""

    def test_identical_images(self) -> None:
        img = np.random.rand(10, 10).astype(np.float64)
        assert compute_ncc(img, img) == pytest.approx(1.0)

    def test_negatively_correlated(self) -> None:
        img = np.arange(100).reshape(10, 10).astype(np.float64)
        neg_img = -img
        assert compute_ncc(img, neg_img) == pytest.approx(-1.0)

    def test_uncorrelated(self) -> None:
        np.random.seed(42)
        img1 = np.random.rand(100, 100).astype(np.float64)
        img2 = np.random.rand(100, 100).astype(np.float64)
        ncc = compute_ncc(img1, img2)
        assert abs(ncc) < 0.1

    def test_constant_images(self) -> None:
        img = np.ones((10, 10), dtype=np.float64)
        assert compute_ncc(img, img) == 1.0

    def test_realistic_dce_high_correlation(self, img_2d_real, img_2d_synth) -> None:
        """Real and synthetic DCE should be positively correlated."""
        ncc = compute_ncc(img_2d_synth, img_2d_real)
        assert ncc > 0.5


# ===================================================================
# Segmentation metrics
# ===================================================================


class TestDice:
    """Tests for Dice Similarity Coefficient."""

    def test_perfect_overlap(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        assert compute_dice(mask, mask) == 1.0

    def test_no_overlap(self) -> None:
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        pred[0:3, 0:3] = True
        gt[7:10, 7:10] = True
        assert compute_dice(pred, gt) == 0.0

    def test_both_empty(self) -> None:
        empty = np.zeros((10, 10), dtype=bool)
        assert compute_dice(empty, empty) == 1.0

    def test_partial_overlap(self) -> None:
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        pred[2:6, 2:6] = True   # 16 pixels
        gt[4:8, 4:8] = True     # 16 pixels
        # Overlap: rows 4-5, cols 4-5 = 4 pixels
        dice = compute_dice(pred, gt)
        expected = 2 * 4 / (16 + 16)
        assert dice == pytest.approx(expected)

    def test_3d_masks(self) -> None:
        mask = np.zeros((5, 10, 10), dtype=bool)
        mask[1:4, 3:7, 3:7] = True
        assert compute_dice(mask, mask) == 1.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_dice(np.ones((5, 5), dtype=bool), np.ones((3, 3), dtype=bool))

    def test_realistic_tumor_masks(self, tumor_mask_2d) -> None:
        """Slightly shifted mask should give partial but nonzero dice."""
        shifted = np.roll(tumor_mask_2d, 3, axis=0)
        dice = compute_dice(shifted, tumor_mask_2d)
        assert 0.0 < dice < 1.0


class TestHD95:
    """Tests for 95th percentile Hausdorff Distance."""

    def test_identical_masks(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        assert compute_hd95(mask, mask) == pytest.approx(0.0)

    def test_both_empty(self) -> None:
        empty = np.zeros((10, 10), dtype=bool)
        assert compute_hd95(empty, empty) == 0.0

    def test_one_empty_returns_inf(self) -> None:
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        gt[3:7, 3:7] = True
        assert compute_hd95(pred, gt) == float("inf")

    def test_known_distance(self) -> None:
        """Two masks separated by known distance."""
        pred = np.zeros((30, 30), dtype=bool)
        gt = np.zeros((30, 30), dtype=bool)
        pred[5:10, 10:15] = True
        gt[20:25, 10:15] = True
        hd95 = compute_hd95(pred, gt)
        # Distance should be ~10-15 pixels (separation is 10 rows)
        assert hd95 > 5
        assert hd95 < 25

    def test_with_voxel_spacing(self) -> None:
        mask1 = np.zeros((20, 20), dtype=bool)
        mask2 = np.zeros((20, 20), dtype=bool)
        mask1[5:10, 5:10] = True
        mask2[5:10, 10:15] = True
        # Shift of 5 pixels in x with 0.5mm spacing -> ~2.5mm
        hd95_unit = compute_hd95(mask1, mask2, voxel_spacing=(1.0, 1.0))
        hd95_half = compute_hd95(mask1, mask2, voxel_spacing=(1.0, 0.5))
        assert hd95_half < hd95_unit

    def test_3d_masks(self) -> None:
        mask = np.zeros((5, 20, 20), dtype=bool)
        mask[1:4, 5:15, 5:15] = True
        assert compute_hd95(mask, mask) == pytest.approx(0.0)


# ===================================================================
# Input validation
# ===================================================================


class TestInputValidation:
    """Tests for input validation across metrics."""

    def test_empty_array_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="empty"):
            compute_mae(empty, empty)

    def test_shape_mismatch_raises(self) -> None:
        pred = np.ones((10, 10), dtype=np.float64)
        gt = np.ones((5, 5), dtype=np.float64)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_mse(pred, gt)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_nmse(pred, gt)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_psnr(pred, gt)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ssim(pred, gt)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ncc(pred, gt)
