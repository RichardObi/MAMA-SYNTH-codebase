"""Unit tests for ROI utilities."""

import numpy as np
import pytest

from eval.roi_utils import dilate_mask, extract_roi, extract_roi_pair


class TestDilateMask:
    """Tests for mask dilation."""

    def test_no_dilation(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        dilated = dilate_mask(mask, margin_mm=0.0, voxel_spacing=(1.0, 1.0))
        np.testing.assert_array_equal(mask.astype(bool), dilated)

    def test_dilation_grows_mask(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[25:35, 25:35] = 1
        dilated = dilate_mask(mask, margin_mm=5.0, voxel_spacing=(1.0, 1.0))
        assert dilated.sum() > mask.sum()
        # Original region should still be included
        assert np.all(dilated[25:35, 25:35] == True)

    def test_anisotropic_spacing(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30, 30] = 1
        # 5mm margin with 1mm spacing → 5 voxels each direction
        dilated_iso = dilate_mask(mask, margin_mm=5.0, voxel_spacing=(1.0, 1.0))
        # 5mm margin with 2mm spacing → ~2.5 voxels each direction
        dilated_aniso = dilate_mask(mask, margin_mm=5.0, voxel_spacing=(2.0, 2.0))
        assert dilated_iso.sum() > dilated_aniso.sum()

    def test_3d_mask(self) -> None:
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[14:18, 14:18, 14:18] = 1
        dilated = dilate_mask(mask, margin_mm=3.0, voxel_spacing=(1.0, 1.0, 1.0))
        assert dilated.sum() > mask.sum()

    def test_empty_mask(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        dilated = dilate_mask(mask, margin_mm=5.0, voxel_spacing=(1.0, 1.0))
        assert dilated.sum() == 0


class TestExtractROI:
    """Tests for ROI extraction."""

    def test_basic_extraction(self) -> None:
        image = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 20:40] = 1
        roi = extract_roi(image, mask)
        assert roi.shape[0] <= 64
        assert roi.shape[1] <= 64
        # ROI should be at least as large as mask bounding box
        assert roi.shape[0] >= 20
        assert roi.shape[1] >= 20

    def test_extraction_with_dilation(self) -> None:
        image = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:30, 20:30] = 1
        roi_no_dilation = extract_roi(image, mask, margin_mm=0.0, voxel_spacing=(1.0, 1.0))
        roi_with_dilation = extract_roi(image, mask, margin_mm=5.0, voxel_spacing=(1.0, 1.0))
        assert roi_with_dilation.size >= roi_no_dilation.size

    def test_full_image_mask(self) -> None:
        image = np.random.rand(32, 32).astype(np.float32)
        mask = np.ones((32, 32), dtype=np.uint8)
        roi = extract_roi(image, mask)
        assert roi.shape == image.shape


class TestExtractROIPair:
    """Tests for paired ROI extraction (aligned crops)."""

    def test_aligned_shapes(self) -> None:
        real = np.random.rand(64, 64).astype(np.float32)
        synth = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        roi_real, roi_synth, roi_mask = extract_roi_pair(real, synth, mask)
        assert roi_real.shape == roi_synth.shape

    def test_values_from_correct_images(self) -> None:
        real = np.ones((32, 32), dtype=np.float32) * 100
        synth = np.ones((32, 32), dtype=np.float32) * 200
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        roi_real, roi_synth, roi_mask = extract_roi_pair(real, synth, mask)
        # Only check values within roi_mask (background is zeroed)
        assert roi_real[roi_mask].mean() == pytest.approx(100.0)
        assert roi_synth[roi_mask].mean() == pytest.approx(200.0)

    def test_3d_pair(self) -> None:
        real = np.random.rand(16, 32, 32).astype(np.float32)
        synth = np.random.rand(16, 32, 32).astype(np.float32)
        mask = np.zeros((16, 32, 32), dtype=np.uint8)
        mask[4:10, 10:20, 10:20] = 1
        roi_real, roi_synth, roi_mask = extract_roi_pair(real, synth, mask)
        assert roi_real.shape == roi_synth.shape
        assert roi_real.ndim == 3
